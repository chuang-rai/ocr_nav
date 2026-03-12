import os
import time
import argparse
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import pyvista as pv
import cv2
from ocr_nav.utils.io_utils import FolderIO
from ocr_nav.scene_graph.floor_graph_refactor import FloorGraph
from ocr_nav.rag.graph_rag import SimpleObjectFrameGraphRAG

from ocr_nav.utils.mapping_utils import project_points, downsample_point_cloud, points_to_mesh, segment_floor
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np
from ocr_nav.utils.floor_graph_utils import visualize_floor_graph
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


def main():
    config_path = Path(__file__).parent.parent.parent / "config" / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    rgb_dir = bag_path.parent / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))

    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = SimpleObjectFrameGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")

    # graph_path = bag_path.parent / "ground_fused_graph.json"
    graph_path = bag_path.parent / "ground_fused_graph.json"
    floor_graph = FloorGraph()
    floor_graph.load_floor_graph(graph_path)

    plotter = create_plotter()
    plotter = visualize_floor_graph(plotter, floor_graph)
    # plotter.show()

    k = 1
    plotter.show(interactive_update=True)
    while k != ord("q"):
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text == "q":
            break

        img_ids, box_infos, img_poses = graph_rag.find_images_by_concept(query_text, top_k=5)

        for img_id, box_info, img_pose in zip(img_ids, box_infos, img_poses):
            img_np = np.array(cv2.imread(rgb_paths[img_id].as_posix())).astype(np.uint8)
            img_with_boxes = draw_bounding_boxes_on_image_np(img_np, [box_info], normalize_max=None)
            cv2.imshow("Retrieved Image", img_with_boxes)
            k = cv2.waitKey(0)
            if k == ord("q"):
                break
            print(img_pose)
            img_pos = np.array(img_pose).reshape((4, 4))[:3, 3].tolist()

            path_pos_quat = floor_graph.plan_global_path(np.array([0, 0, 0]), np.array(img_pos))
            path_pos = path_pos_quat[:, :3]

            cube = draw_cube(np.array(img_pos), size=0.4)
            plotter.add_mesh(cube, name="goal", color="red")

            cube = draw_cube(np.array([0, 0, 0]), size=0.4)
            plotter.add_mesh(cube, name="start", color="green")
            for i in range(len(path_pos) - 1):
                line = draw_line(path_pos[i], path_pos[i + 1])
                plotter.add_mesh(line, line_width=10, color="blue", render_lines_as_tubes=True)
                plotter.update()

            plotter.update()
        plotter.update()
    plotter.show()


if __name__ == "__main__":
    main()
