from pathlib import Path
from dataclasses import dataclass
import re
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple

from ocr_nav.utils.levenshtein_utils import levenshtein_distance
from ocr_nav.utils.ocr_utils import re_match, compute_text_freq, select_points_in_bbox
from ocr_nav.utils.mapping_utils import project_points, downsample_point_cloud, points_to_mesh, segment_floor

from ocr_nav.utils.io_utils import (
    load_depth,
    load_image,
    load_intrinsics,
    load_pose,
    load_lidar,
    load_masks,
    load_livox_poses_timestamps,
    search_latest_poses_within_timestamp_range,
    FolderIO,
)

# from ocr_nav.utils.visualization_utils import draw_bounding_boxes, draw_cube, draw_coordinate, draw_line
from ocr_nav.utils.pyvista_vis_utils import (
    draw_cube,
    draw_line,
    draw_sphere,
    draw_text,
    draw_point_cloud,
    draw_coordinate,
    create_plotter,
    convert_open3d_mesh_to_pyvista,
)
from code.ocr_nav.ocr_nav.scene_graph.pose_graph import PoseGraph, Pose, TextBag


def select_points_in_masks(
    mask: np.ndarray, pc_image_2d: np.ndarray, pc_image_3d: np.ndarray, transform: np.ndarray = np.eye(4)
) -> o3d.geometry.PointCloud:
    valid_indices = np.where(mask[pc_image_2d[:, 1].astype(int), pc_image_2d[:, 0].astype(int)] > 0)[0]
    ground_pc_image_3d = pc_image_3d[valid_indices, :]  # (K, 3)
    ground_pc_world = transform_point_cloud(ground_pc_image_3d, transform)
    # ground_pc_image_3d_hom = np.hstack((ground_pc_image_3d, np.ones((ground_pc_image_3d.shape[0], 1))))  # (K, 4)
    # ground_pc_world = (transform @ ground_pc_image_3d_hom.T).T[:, :3]  # (K, 3)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(ground_pc_world)
    return pc_o3d


def transform_point_cloud(pc: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pc_hom = np.hstack((pc, np.ones((pc.shape[0], 1))))  # (N, 4)
    pc_transformed_hom = (transform @ pc_hom.T).T
    return pc_transformed_hom[:, :3]


def select_points_near_heights(pc: np.ndarray, heights: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    mask = np.zeros(pc.shape[0], dtype=bool)
    for h in heights:
        mask |= (pc[:, 2] >= h - threshold) & (pc[:, 2] <= h + threshold)
    return pc[mask, :]


def filter_ground_points_with_obstacles_in_height_range(
    full_pc: o3d.geometry.PointCloud,
    ground_pc: o3d.geometry.PointCloud,
    heights: np.ndarray,
    voxel_res: float = 0.05,
    high_threshold: float = 0.5,
    low_threshold: float = 0.2,
) -> o3d.geometry.PointCloud:

    min_bound, max_bound = full_pc.get_min_bound(), full_pc.get_max_bound()
    height_ranges = zip(heights + low_threshold, np.concat([heights[1:], np.array([max_bound[2]])]) - high_threshold)
    full_voxel_grid = np.round((np.array(full_pc.points) - min_bound) / voxel_res)
    ground_voxel_grid = np.round((np.array(ground_pc.points) - min_bound) / voxel_res)

    traversable_ground_voxel_grid = []
    x_range, y_range, z_range = (
        np.max(full_voxel_grid[:, 0]).astype(int) + 1,
        np.max(full_voxel_grid[:, 1]).astype(int) + 1,
        np.max(full_voxel_grid[:, 2]).astype(int) + 1,
    )
    last_high = 0
    for floor_i, height_range in enumerate(height_ranges):
        low, high = np.floor((np.array(height_range) - min_bound[2]) / voxel_res).astype(int)
        obstacle_mask = (full_voxel_grid[:, 2] >= low) & (full_voxel_grid[:, 2] < high)
        ground_mask = (ground_voxel_grid[:, 2] >= last_high) & (ground_voxel_grid[:, 2] < low)
        floor_obstacle_voxels = full_voxel_grid[obstacle_mask, :]
        floor_ground_voxels = ground_voxel_grid[ground_mask, :]

        floor_obstacle_map = np.zeros((x_range, y_range), dtype=bool)
        floor_obstacle_map[floor_obstacle_voxels[:, 0].astype(int), floor_obstacle_voxels[:, 1].astype(int)] = True
        floor_ground_map = np.zeros((x_range, y_range), dtype=bool)
        floor_ground_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)] = True

        floor_obstacle_map_cv2 = floor_obstacle_map.astype(np.uint8) * 255
        floor_ground_map_cv2 = floor_ground_map.astype(np.uint8) * 255
        cv2.imshow("floor_obstacle_map", floor_obstacle_map_cv2)
        cv2.imshow("floor_ground_map", floor_ground_map_cv2)
        cv2.waitKey()

        traversable_map = floor_ground_map & (~floor_obstacle_map)
        traversable_ground_indices = np.where(
            traversable_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)]
        )[0]

        traversable_ground_voxels = floor_ground_voxels[traversable_ground_indices, :]
        traversable_ground_voxel_grid.append(traversable_ground_voxels)
        last_high = high

    traversable_ground_voxel_grid = np.vstack(traversable_ground_voxel_grid)
    traversable_ground_points = traversable_ground_voxel_grid * voxel_res + min_bound
    return traversable_ground_points


def main():
    img_dir = Path(
        # "/home/chuang/hcg/projects/ocr/data/Flexoffice_extracted/rosbag2_2025_12_10-10_25_57_perception_suite/left"
        "/home/chuang/hcg/projects/ocr/data/eth_extracted/rosbag2_2025_12_16-17_09_00_perception_suite/left"
    )
    root_dir = img_dir.parent
    folderio = FolderIO(root_dir, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")
    debug = False
    if debug:
        debug_image_dir = root_dir / "debug_lidar_proj"
        debug_image_dir.mkdir(exist_ok=True)

    # intr_mat = load_intrinsics(root_dir / "intrinsics.txt")
    intr_mat = folderio.get_intrinsics()

    # livox2zed_left_mat_path = root_dir / "tf_livox_mid360_to_zed_left_camera_optical_frame.txt"
    tf_livox2zed_left = folderio.get_livox2left_camera_tf()
    tf_zed_left2livox = np.linalg.inv(tf_livox2zed_left)

    # robosense2zed_left_mat_path = root_dir / "tf_robosense_e1r_to_zed_left_camera_optical_frame.txt"
    tf_rs2zed_left = folderio.get_rslidar2left_camera_tf()
    livox_poses_timestamps_path = root_dir / "glim_livox_poses_timestamps.npy"
    livox_poses, livox_timestamps = load_livox_poses_timestamps(livox_poses_timestamps_path)
    print(f"Loaded livox poses and timestamps.{livox_timestamps.shape}")

    textmap = PoseGraph()
    ground_pc = o3d.geometry.PointCloud()
    ground_pc_rs = o3d.geometry.PointCloud()
    full_pc_rs = o3d.geometry.PointCloud()
    pbar = tqdm(enumerate(sorted(img_dir.iterdir())), total=len(list(img_dir.iterdir())))
    for pi, p in pbar:
        # if pi > 20:
        #     break
        pbar.set_description(f"Processing frame {pi}")
        frame_id = int(p.stem.split("_")[-1])

        if pi == 0 or debug:
            image_path = root_dir / "left" / f"image_{frame_id}.jpg"
            img = load_image(image_path)
            h, w, _ = np.array(img).shape

        masks = folderio.get_mask(pi)
        pc_livox = folderio.get_livox(pi)  # (N, 3)
        pc_rslidar = folderio.get_rslidar(pi)  # (N, 3)

        # transform point cloud from robosense frame to zed left camera frame
        livox_pose, livox_timestamp = search_latest_poses_within_timestamp_range(
            livox_poses,
            livox_timestamps,
            frame_id,
        )
        if livox_pose is None:
            print(f"No livox pose found for frame {frame_id}, skipping...")
            continue
        pc_livox_hom = np.hstack((pc_livox, np.ones((pc_livox.shape[0], 1))))  # (N, 4)
        pc_hom_zed = tf_livox2zed_left @ pc_livox_hom.T
        pc_zed = pc_hom_zed[:3, :].T  # (N, 3)

        pc_rslidar_hom = np.hstack((pc_rslidar, np.ones((pc_rslidar.shape[0], 1))))  # (N, 4)
        pc_rs_hom_zed = tf_rs2zed_left @ pc_rslidar_hom.T
        pc_rs_zed = pc_rs_hom_zed[:3, :].T  # (N, 3)

        # project points to image plane
        pc_image_2d, pc_image_2d_depth, pc_image_3d = project_points(pc_zed, intr_mat, w, h)  # (N, 2)
        pc_rs_image_2d, pc_rs_image_2d_depth, pc_rs_image_3d = project_points(pc_rs_zed, intr_mat, w, h)  # (N, 2)

        # if debug:
        #     depth_scale = (pc_image_2d_depth - np.min(pc_image_2d_depth)) / (
        #         np.max(pc_image_2d_depth) - np.min(pc_image_2d_depth)
        #     ).reshape((-1, 1))
        #     depth_scale_gray = (depth_scale * 255).astype(np.uint8)
        #     # get depth color map jet
        #     depth_color = cv2.applyColorMap(depth_scale_gray, cv2.COLORMAP_JET)[0]  # (N, 3)
        #     for i in range(pc_image_2d.shape[0]):
        #         u, v = int(pc_image_2d[i, 0]), int(pc_image_2d[i, 1])
        #         # img.putpixel((u, v), tuple(depth_color[i]))
        #         # draw a small circle at the projected point
        #         for du in range(-2, 3):
        #             for dv in range(-2, 3):
        #                 if 0 <= u + du < w and 0 <= v + dv < h:
        #                     img.putpixel((u + du, v + dv), tuple(depth_color[i]))

        #     debug_image_path = debug_image_dir / f"lidar_proj_{frame_id}.png"
        #     img.save(debug_image_path)

        ground_mask = masks[0][0]
        pc_o3d = select_points_in_masks(ground_mask, pc_image_2d, pc_image_3d, transform=livox_pose @ tf_zed_left2livox)
        ground_pc += pc_o3d
        pc_o3d_rs = select_points_in_masks(
            ground_mask,
            pc_rs_image_2d,
            pc_rs_image_3d,
            transform=livox_pose @ tf_zed_left2livox,
        )
        pc_o3d_rs_full_np = transform_point_cloud(pc_rs_image_3d, livox_pose @ tf_zed_left2livox)
        ground_pc_rs += pc_o3d_rs
        pc_o3d_rs_full = o3d.geometry.PointCloud()
        pc_o3d_rs_full.points = o3d.utility.Vector3dVector(pc_o3d_rs_full_np)
        full_pc_rs += pc_o3d_rs_full

        # add pose to the textmap
        pc_world = (livox_pose @ tf_zed_left2livox @ pc_hom_zed)[:3, :].T  # (N, 3)
        pose_id = textmap.add_pose(pi, livox_pose, lidar=pc_world)
        if textmap.G.has_node(pi - 1):
            textmap.add_pose_edge(pi - 1, pi)

    heights = segment_floor(np.array(ground_pc.points), resolution=0.05)
    ground_pc_rs_numpy = select_points_near_heights(
        np.array((ground_pc + ground_pc_rs).points), heights, threshold=0.05
    )
    ground_pc_rs = o3d.geometry.PointCloud()
    ground_pc_rs.points = o3d.utility.Vector3dVector(ground_pc_rs_numpy)

    ground_pc_rs_numpy = filter_ground_points_with_obstacles_in_height_range(
        full_pc_rs,
        ground_pc_rs,
        np.array(heights),
        voxel_res=0.1,
        high_threshold=0.5,
        low_threshold=0.2,
    )
    ground_pc_rs = o3d.geometry.PointCloud()
    ground_pc_rs.points = o3d.utility.Vector3dVector(ground_pc_rs_numpy)

    # total pose number
    total_pose_nodes = sorted([node for node, att in textmap.G.nodes(data=True) if isinstance(node, int)])
    print(total_pose_nodes)
    pose_node_labels = {node: node for node in total_pose_nodes}
    pose_node_colors = {node: "lightblue" for node in total_pose_nodes}
    # x = np.linspace(0, 2, len(total_pose_nodes))
    # y = np.sin(2 * np.pi * x)
    # fixed_node_positions = {node: (a, b) for a, b, node in zip(x, y, total_pose_nodes)}
    fixed_node_positions = {
        node: (
            textmap.G.nodes[node]["pose"].pose[0, 3],
            textmap.G.nodes[node]["pose"].pose[1, 3],
        )
        for node in total_pose_nodes
    }

    pos = nx.spring_layout(
        textmap.G,
        pos=fixed_node_positions,
        fixed=total_pose_nodes,
        seed=42,
        method="force",
        k=2,
    )
    text_nodes = [node for node, att in textmap.G.nodes(data=True) if isinstance(node, str)]
    text_node_labels = {node: "/".join(list(textmap.G.nodes[node]["textbag"].text_dict.keys())) for node in text_nodes}
    text_node_colors = {node: "green" for node in text_nodes}
    combined_colors = {**text_node_colors, **pose_node_colors}
    colors = [combined_colors[node] for node in textmap.G.nodes]
    # nx.draw(
    #     textmap.G,
    #     pos,
    #     with_labels=True,
    #     labels={**text_node_labels, **pose_node_labels},
    #     node_color=colors,
    # )
    # plt.show()

    # create plotter
    num_pose_nodes = total_pose_nodes[-1] + 1
    graymap = (np.linspace(0, 1, num_pose_nodes) * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(graymap, cv2.COLORMAP_JET).reshape((num_pose_nodes, 3))
    plotter = create_plotter()
    # visualize the camera poses as cubes and the edges between poses and texts as lines
    vis_elements = []
    for node, att in textmap.G.nodes(data=True):
        if isinstance(node, int):
            pose_node: Pose = att["pose"]
            pose_id = pose_node.pose_id
            cam_cube = draw_cube(pose_node.pose[:3, 3], size=0.5, color="blue")
            plotter.add_mesh(cam_cube, color=colormap[pose_id] / 255.0)
            if node == 0:
                axis = draw_coordinate(pose_node.pose[:3, 3], size=2)
                plotter.add_actor(axis)  # type: ignore
                # plotter = draw_point_cloud(plotter, pose_node.lidar, point_size=2.0)
                # plotter.add_mesh(axis)
            # if node in {47, 59}:
            #     plotter = draw_point_cloud(plotter, pose_node.lidar, point_size=2.0)
        elif isinstance(node, str):
            textbag: TextBag = att["textbag"]
            if textbag.pc is not None:
                text = "/".join(list(textbag.text_dict.keys()))
                point_3d = np.mean(textbag.pc, axis=0)  # take the average locations for visualization
                sphere = draw_sphere(point_3d, radius=0.03)
                plotter.add_mesh(sphere, color="red")

                text_actor = draw_text(text, point_3d, height=0.2, normal=np.array((0, 1, 0)))
                plotter.add_mesh(text_actor, color="red")
    # label_dict = {}
    print("Start adding edges...")
    for edge in textmap.G.edges():
        node1, node2 = edge
        if isinstance(node1, int) and isinstance(node2, int):
            pose1: Pose = textmap.G.nodes[node1]["pose"]
            pose2: Pose = textmap.G.nodes[node2]["pose"]
            line = draw_line(pose1.pose[:3, 3], pose2.pose[:3, 3], color="red", line_width=2)
            plotter.add_mesh(line)
        elif isinstance(node1, int) and isinstance(node2, str):
            pose: Pose = textmap.G.nodes[node1]["pose"]
            textbag: TextBag = textmap.G.nodes[node2]["textbag"]
            if textbag.pc is not None:
                for point_3d in textbag.pc:
                    line = draw_line(pose.pose[:3, 3], point_3d, color="green", line_width=2)
                    plotter.add_mesh(line)
        elif isinstance(node1, str) and isinstance(node2, int):
            pose: Pose = textmap.G.nodes[node2]["pose"]
            textbag: TextBag = textmap.G.nodes[node1]["textbag"]
            if textbag.pc is not None:
                mean_pc = textbag.pc.mean(axis=0)
                text_list = list(textbag.text_dict.keys())
                texts = "/".join(text_list)
                line = draw_line(pose.pose[:3, 3], mean_pc, color="magenta", line_width=2)
                plotter.add_mesh(line)

    ground_point_down = downsample_point_cloud(ground_pc_rs, voxel_size=0.2)
    ground_point_down_np = np.asarray(ground_point_down.points)
    print(ground_point_down_np.shape)
    plotter = draw_point_cloud(plotter, ground_point_down_np, color=None, point_size=2.0)
    ground_mesh = points_to_mesh(ground_point_down, voxel_size=0.2)
    ground_mesh_pyvista = convert_open3d_mesh_to_pyvista(ground_mesh)
    plotter.add_mesh(ground_mesh_pyvista, color="lightgray", opacity=0.5, show_edges=True)

    plotter.show()


if __name__ == "__main__":
    main()
