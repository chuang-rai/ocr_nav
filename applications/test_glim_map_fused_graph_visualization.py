import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pyvista as pv
import networkx as nx
from plyfile import PlyData
from typing import Optional


def create_plotter() -> pv.Plotter:
    plotter = pv.Plotter()
    plotter.set_background((1.0, 1.0, 1.0))  # type: ignore
    plotter.window_size = [1920, 1080]
    return plotter


def load_ply_point_cloud(ply_path: Path) -> np.ndarray:

    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"].data
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    return points  # (N, 3)


def draw_point_cloud(
    plotter: pv.Plotter, points: np.ndarray, color: Optional[np.ndarray] = None, point_size: float = 5.0
) -> pv.Plotter:
    point_cloud = pv.PolyData(points)
    point_cloud["colors"] = color if color is not None else np.array([[0.0, 0.0, 1.0]] * points.shape[0])
    plotter.add_points(point_cloud, scalars="colors", rgb=True, point_size=point_size)
    return plotter


def draw_line(
    point1: np.ndarray,
    point2: np.ndarray,
) -> pv.PolyData:
    line = pv.Line(point1, point2)
    return line


def main():
    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--glim_map_path",
        type=str,
        default="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/glim_map.ply",
        help="Root path to the dataset folder",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rosbag2_2026_02_02-17_46_19_perception_suite_fix_trimmed_glim/ground_fused_graph.json",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()
    glim_map_path = args.glim_map_path
    graph_path = args.graph_path

    # floor_graph = FloorGraph()
    # floor_graph.load_floor_graph(graph_path)
    with open(graph_path, "r") as f:
        graph_json = json.load(f)
    floor_graph = nx.node_link_graph(graph_json)

    plotter = create_plotter()
    glim_map_pc = load_ply_point_cloud(glim_map_path)
    plotter = draw_point_cloud(plotter, glim_map_pc, color=None, point_size=2.0)

    print("Plotting voronoi graphs...")
    for edge in tqdm(floor_graph.edges()):
        src, tar = edge
        src_pos = floor_graph.nodes[src]["pos"]
        tar_pos = floor_graph.nodes[tar]["pos"]
        line = draw_line(
            np.array([src_pos[0], src_pos[1], src_pos[2]]),
            np.array([tar_pos[0], tar_pos[1], tar_pos[2]]),
        )
        plotter.add_mesh(line, line_width=4, color="green", render_lines_as_tubes=True)

    plotter.show()


if __name__ == "__main__":
    main()
