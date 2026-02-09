import argparse
import rclpy
import cv2
import open3d as o3d
from tqdm import tqdm
import pyvista as pv
import numpy as np
from pathlib import Path
from ocr_nav.utils.segmentation_utils import GroundingDinoSamSegmenter
from ocr_nav.utils.io_utils import BagIO
from ocr_nav.scene_graph.floor_graph import FloorGraph
from ocr_nav.utils.mapping_utils import downsample_point_cloud, points_to_mesh
from ocr_nav.utils.pyvista_vis_utils import (
    draw_cube,
    draw_line,
    draw_sphere,
    draw_point_cloud,
    draw_coordinate,
    create_plotter,
    convert_open3d_mesh_to_pyvista,
)


def main():
    config_dir = Path(__file__).parent.parent / "config"
    grounding_dino_config_path = config_dir / "detection" / "grounding_dino.yaml"
    sam_config_path = config_dir / "segmentation" / "segment_anything.yaml"

    parser = argparse.ArgumentParser(description="Bag file path")
    parser.add_argument(
        "--bag_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_factory/rosbag2_2025_12_16-17_09_00_perception_suite_fixed_and_trimmed_1",
    )
    args = parser.parse_args()
    segmenter = GroundingDinoSamSegmenter(
        grounding_dino_config_path=grounding_dino_config_path.as_posix(),
        sam_config_path=sam_config_path.as_posix(),
        device="cuda",
    )

    rclpy.init()
    bag_path = Path(args.bag_path)
    bagio = BagIO(bag_path.as_posix(), sample_every=10)
    bagio.init_reader()

    ground_mesh = FloorGraph(voxel_size=0.2)
    ground_mesh.build_floor_graph_with_bagio(bagio, segmenter, vis=False)
    stairs_pc_full = o3d.geometry.PointCloud()
    for stair_pc in ground_mesh.stairs_pc_list:
        floor_stairs = o3d.geometry.PointCloud()
        floor_stairs.points = o3d.utility.Vector3dVector(np.asarray(stair_pc))
        stairs_pc_full += floor_stairs

    ground_point_down = downsample_point_cloud(ground_mesh.full_ground_pc, voxel_size=0.2)
    ground_point_down_np = np.asarray(ground_point_down.points)
    plotter = create_plotter()
    plotter = draw_point_cloud(plotter, ground_point_down_np, color=None, point_size=2.0)
    floor_mesh = points_to_mesh(ground_point_down, voxel_size=0.2)
    floor_mesh_pyvista = convert_open3d_mesh_to_pyvista(floor_mesh)
    plotter.add_mesh(floor_mesh_pyvista, color="lightgray", opacity=0.5, show_edges=True)

    total_pose_nodes = sorted(
        [node for node, att in ground_mesh.pose_graph.G.nodes(data=True) if isinstance(node, int)]
    )
    num_pose_nodes = total_pose_nodes[-1] + 1
    graymap = (np.linspace(0, 1, num_pose_nodes) * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(graymap, cv2.COLORMAP_JET).reshape((num_pose_nodes, 3))
    print("Plotting pose nodes...")
    for node, att in ground_mesh.pose_graph.G.nodes(data=True):
        if isinstance(node, int):
            pose_node = att["pose"]
            pose_id = pose_node.pose_id
            cam_cube = draw_cube(pose_node.pose[:3, 3], size=0.5, color="blue")
            plotter.add_mesh(cam_cube, color=colormap[pose_id] / 255.0)
            if node == 0:
                axis = draw_coordinate(pose_node.pose[:3, 3], size=2)
                plotter.add_actor(axis)  # type: ignore
    print("Plotting pose edges...")
    for edge in ground_mesh.pose_graph.G.edges():
        node1, node2 = edge
        if isinstance(node1, int) and isinstance(node2, int):
            pose1 = ground_mesh.pose_graph.G.nodes[node1]["pose"]
            pose2 = ground_mesh.pose_graph.G.nodes[node2]["pose"]
            line = draw_line(pose1.pose[:3, 3], pose2.pose[:3, 3])
            plotter.add_mesh(line, color="red", line_width=2, render_lines_as_tubes=True)

    # plot voronoi graph
    print("Plotting voronoi graphs...")
    for edge in tqdm(ground_mesh.floor_graph.edges()):
        src, tar = edge
        src_pos = ground_mesh.floor_graph.nodes[src]["pos"]
        tar_pos = ground_mesh.floor_graph.nodes[tar]["pos"]
        line = draw_line(
            np.array([src_pos[0], src_pos[1], src_pos[2]]),
            np.array([tar_pos[0], tar_pos[1], tar_pos[2]]),
        )
        plotter.add_mesh(line, line_width=4, color="green", render_lines_as_tubes=True)
    for node in tqdm(ground_mesh.floor_graph.nodes()):
        node_pos = ground_mesh.floor_graph.nodes[node]["pos"]
        sphere = draw_sphere(
            np.array([node_pos[0], node_pos[1], node_pos[2]]),
            radius=0.1,
        )
        plotter.add_mesh(sphere, color="green")

    meshes_to_combine = plotter.meshes
    combined = pv.merge(meshes_to_combine)
    if not isinstance(combined, pv.PolyData):
        combined = combined.extract_surface()
    combined.save("/tmp/ground_mesh_and_graph.ply")
    plotter.show()
    ground_mesh.save_floor_graph("/tmp/ground_fused_graph.json")
    fused_graph = ground_mesh.load_floor_graph("/tmp/ground_fused_graph.json")

    bagio.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
