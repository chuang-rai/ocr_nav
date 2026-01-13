from pathlib import Path
from dataclasses import dataclass
import re
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple

from ocr_nav.utils.levenshtein_utils import levenshtein_distance
from ocr_nav.utils.ocr_utils import re_match, compute_text_freq, select_points_in_bbox
from ocr_nav.utils.mapping_utils import project_points, points_to_mesh, downsample_point_cloud
from ocr_nav.utils.io_utils import load_depth, load_image, load_intrinsics, load_pose, load_lidar, load_masks

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


def main():
    ocr_dir = Path(
        "/home/chuang/hcg/projects/ocr/data/giorgio_data/gtonetti_data_zurich_hb_3_merged_pose_zed_gen2_depth_fs_zurich_hb_3_5_bag_zurich_hb_3_5_2/left_deepseek_ocr_results_gundam"
    )
    root_dir = ocr_dir.parent
    debug = False
    if debug:
        debug_image_dir = root_dir / "debug_lidar_proj"
        debug_image_dir.mkdir(exist_ok=True)

    intr_mat = load_intrinsics(root_dir / "intrinsics.txt")
    robosense2zed_left_mat_path = root_dir / "tf_robosense_e1r_to_zed_left_camera_optical_frame.txt"
    tf_rs2zed_left = load_pose(robosense2zed_left_mat_path)
    textmap = PoseGraph()
    ground_pc = o3d.geometry.PointCloud()
    for pi, p in enumerate(sorted(ocr_dir.iterdir())):
        print(f"Processing frame {pi}")
        frame_id = int(p.stem[6:])

        if pi == 0 or debug:
            image_path = root_dir / "left" / f"image_{frame_id}.jpg"
            img = load_image(image_path)
            assert img is not None
            h, w, _ = np.array(img).shape

        pose_path = root_dir / "pose" / f"pose_{frame_id}.txt"
        depth_path = root_dir / "depth" / f"depth_{frame_id}.png"
        lidar_path = root_dir / "rslidar" / f"rslidar_{frame_id}.npy"
        mask_path = root_dir / "masks_sam2_s" / f"mask_{frame_id}.npy"
        pose_np = load_pose(pose_path)
        masks = load_masks(mask_path)
        pc_rs = load_lidar(lidar_path)  # (N, 3)

        # transform point cloud from robosense frame to zed left camera frame
        pc_rs_hom = np.hstack((pc_rs, np.ones((pc_rs.shape[0], 1))))  # (N, 4)
        pc_hom_zed = tf_rs2zed_left @ pc_rs_hom.T
        pc_zed = pc_hom_zed[:3, :].T  # (N, 3)

        # project points to image plane
        pc_image_2d, pc_image_2d_depth, pc_image_3d = project_points(pc_zed, intr_mat, w, h)  # (N, 2)

        if debug:
            depth_scale = (pc_image_2d_depth - np.min(pc_image_2d_depth)) / (
                np.max(pc_image_2d_depth) - np.min(pc_image_2d_depth)
            ).reshape((-1, 1))
            depth_scale_gray = (depth_scale * 255).astype(np.uint8)
            # get depth color map jet
            depth_color = cv2.applyColorMap(depth_scale_gray, cv2.COLORMAP_JET)[0]  # (N, 3)
            for i in range(pc_image_2d.shape[0]):
                u, v = int(pc_image_2d[i, 0]), int(pc_image_2d[i, 1])
                # img.putpixel((u, v), tuple(depth_color[i]))
                # draw a small circle at the projected point
                for du in range(-2, 3):
                    for dv in range(-2, 3):
                        if 0 <= u + du < w and 0 <= v + dv < h:
                            img.putpixel((u + du, v + dv), tuple(depth_color[i]))

            assert isinstance(debug_image_dir, Path)
            debug_image_path = debug_image_dir / f"lidar_proj_{frame_id}.png"
            img.save(debug_image_path)

        ground_mask = masks[0][0]
        valid_indices = np.where(ground_mask[pc_image_2d[:, 1].astype(int), pc_image_2d[:, 0].astype(int)] > 0)[0]
        ground_pc_image_3d = pc_image_3d[valid_indices, :]  # (K, 3)
        ground_pc_image_3d_hom = np.hstack((ground_pc_image_3d, np.ones((ground_pc_image_3d.shape[0], 1))))  # (K, 4)
        ground_pc_world = (pose_np @ ground_pc_image_3d_hom.T).T[:, :3]  # (K, 3)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(ground_pc_world)
        ground_pc += pc_o3d

        # add pose to the textmap
        pc_world = (pose_np @ pc_hom_zed)[:3, :].T  # (N, 3)
        pose_id = textmap.add_pose(pi, pose_np, lidar=pc_world)
        if pi > 0:
            textmap.add_pose_edge(pi - 1, pi)

        # read texts from ocr, can be wrapped in a function call
        result_path = p / "result.mmd"
        with open(result_path, "r") as f:
            texts = f.read()

        # extract texts
        matches, matches_image, matches_other = re_match(texts)
        print(matches_other)

        text_freq = compute_text_freq(matches_other)
        integrated_text_count = defaultdict(int)

        for mi, m in enumerate(matches_other):
            # skip those repeated misdetections
            _, text, bbox = m
            freq = text_freq[text]
            integrated_count = integrated_text_count[text]
            if freq > 10:
                continue
            bbox = eval(bbox)[0]  # bbox format: [x1, y1, x2, y2]
            bbox = [bbox[0] / 999 * w, bbox[1] / 999 * h, bbox[2] / 999 * w, bbox[3] / 999 * h]

            # centroid = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            # centroid_3d = backproject_point(depth, intr_mat, centroid[0], centroid[1])
            box_pc = select_points_in_bbox(pc_image_2d, bbox, pc_image_3d)  # (M, 3)
            centroid_3d = box_pc.mean(axis=0) if box_pc.shape[0] > 0 else None
            if centroid_3d is None:  # or np.linalg.norm(centroid_3d) > 10.0:
                # skip those points with too large depth values
                print(f"Skipping text {text} due to no valid 3D points")
                print(centroid_3d)
                continue

            # transform to world coordinate
            cam2robot = np.eye(4)
            cam2robot[:3, :3] = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
            centroid_3d_hom = np.ones((4,))
            centroid_3d_hom[:3] = centroid_3d
            centroid_3d_world_hom = pose_np @ centroid_3d_hom
            centroid_3d_world = centroid_3d_world_hom[:3] / centroid_3d_world_hom[3]

            most_similar_node_and_dist = textmap.find_similar_text(text, dist_threshold=2)
            # if similar text is found in the textmap, fuse them (add them to the text_list)
            if most_similar_node_and_dist:
                matched_node, d = most_similar_node_and_dist
                textmap.G.nodes[matched_node]["textbag"].text_dict[text] = []
                textmap.G.nodes[matched_node]["textbag"].text_dict[text].append(pi)
                prev_pc = textmap.G.nodes[matched_node]["textbag"].pc
                print("prev_pc:", prev_pc)
                textmap.G.nodes[matched_node]["textbag"].pc = (
                    np.concat((prev_pc, centroid_3d_world.reshape(1, 3)), axis=0)
                    if prev_pc is not None
                    else centroid_3d_world.reshape(1, 3)
                )
                textmap.link_old_text_node_to_new_pose(pi, matched_node)
            else:
                # initialize new text in the textmap
                textmap.add_text_to_pose(pose_id, text, centroid_3d_world)
                print(f"added text node {text}")
    # total pose number
    total_pose_nodes = sorted([node for node, att in textmap.G.nodes(data=True) if isinstance(node, int)])
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
    num_pose_nodes = len(total_pose_nodes)
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
            # plotter = draw_point_cloud(plotter, pose_node.lidar, point_size=2.0)
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
                texts = "/".join(list(textbag.text_dict.keys()))
                line = draw_line(pose.pose[:3, 3], mean_pc, color="magenta", line_width=2)
                plotter.add_mesh(line)

    ground_point_down = downsample_point_cloud(ground_pc, voxel_size=0.5)
    ground_point_down_np = np.asarray(ground_point_down.points)
    print(ground_point_down_np.shape)
    plotter = draw_point_cloud(plotter, ground_point_down_np, color=None, point_size=2.0)
    ground_mesh = points_to_mesh(ground_point_down, voxel_size=0.5)
    ground_mesh_pyvista = convert_open3d_mesh_to_pyvista(ground_mesh)
    plotter.add_mesh(ground_mesh_pyvista, color="lightgray", opacity=0.5, show_edges=True)

    plotter.show()


if __name__ == "__main__":
    main()
