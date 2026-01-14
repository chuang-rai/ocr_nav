from pathlib import Path
from dataclasses import dataclass
import re
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple

from ocr_nav.utils.levenshtein_utils import levenshtein_distance
from ocr_nav.utils.ocr_utils import re_match, compute_text_freq, select_points_in_bbox
from ocr_nav.utils.mapping_utils import project_points, downsample_point_cloud, points_to_mesh, segment_floor

from ocr_nav.utils.io_utils import (
    load_ply_point_cloud,
    load_depth,
    load_image,
    load_intrinsics,
    load_pose,
    load_lidar,
    load_masks,
    load_livox_poses_timestamps,
    search_latest_poses_within_timestamp_range,
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
from ocr_nav.scene_graph.pose_graph import PoseGraph, Pose
from ocr_nav.scene_graph.text_graph import TextBag


def main():
    ply_path = Path(
        "/home/chuang/hcg/projects/ocr/data/Flexoffice/rosbag2_2025_12_10-10_25_57_perception_suite_glim_map_bin.ply"
    )
    res = 0.1
    pc = load_ply_point_cloud(ply_path)  # (N, 3)
    heights = segment_floor(pc, resolution=0.1, vis=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    min_bound, max_bound = (
        pcd.get_axis_aligned_bounding_box().get_min_bound(),
        pcd.get_axis_aligned_bounding_box().get_max_bound(),
    )
    grid = np.ceil((max_bound - min_bound) / res).astype(int) + 1

    height_ranges = zip(heights - 0.2, heights + 3)
    for floor_i, (low, high) in enumerate(height_ranges):
        # low = heights[-2]
        # high = heights[-1]

        mask = (pc[:, 2] >= low) & (pc[:, 2] < high)
        pc_floor = pc[mask, :]

        voxels = np.round((pc_floor - min_bound) / res).astype(int)

        hist, _, _ = np.histogram2d(
            voxels[:, 0],
            voxels[:, 1],
            bins=(grid[0], grid[1]),
            range=[[0, grid[0]], [0, grid[1]]],
        )
        plt.figure()
        plt.imshow(hist, interpolation="nearest", origin="lower", cmap="jet")
        plt.colorbar()
        plt.show()

        hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist = cv2.GaussianBlur(hist, (5, 5), 1)
        hist_threshold = 0.25 * np.max(hist)
        _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)
        walls_skeleton = cv2.dilate(walls_skeleton, np.ones((3, 3), np.uint8), iterations=1)
        cv2.imshow("walls_skeleton", walls_skeleton)
        cv2.waitKey()

        plotter = create_plotter()

        draw_point_cloud(plotter, pc_floor, point_size=2.0)
        plotter.show()

        heights = heights[:-2]


if __name__ == "__main__":
    main()
