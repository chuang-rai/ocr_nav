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
import open3d as o3d

from ocr_nav.utils.io_utils import load_pose, load_depth, load_intrinsics
from ocr_nav.utils.mapping_utils import backproject_depth_map


pose_dir = Path(
    "/home/chuang/hcg/projects/ocr/data/giorgio_data/gtonetti_data_zurich_hb_3_merged_pose_zed_gen2_depth_fs_zurich_hb_3_5_bag_zurich_hb_3_5_2/pose"
)
root_dir = pose_dir.parent


intr_mat = load_intrinsics(root_dir / "intrinsics.txt")


pcd_list = []
for pi, p in enumerate(sorted(pose_dir.iterdir())):
    if pi > 10:
        break
    print(f"Processing frame {pi}")
    frame_id = int(p.stem.split("_")[-1])
    pose_path = root_dir / "pose" / f"pose_{frame_id}.txt"
    depth_path = root_dir / "depth" / f"depth_{frame_id}.png"
    pose = load_pose(pose_path)
    depth = load_depth(depth_path) / 1000.0

    points_3d_cam = backproject_depth_map(depth, intr_mat)
    ones = np.ones((points_3d_cam.shape[0], 1))
    points_3d_cam_hom = np.hstack((points_3d_cam, ones))
    points_3d_world_hom = (pose @ points_3d_cam_hom.T).T
    points_3d_world = points_3d_world_hom[:, :3]  # (N, 3)

    # visualize it with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d_world)
    # sample point cloud uniformly with voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([pcd])
    pcd_list.append(pcd)
o3d.visualization.draw_geometries(pcd_list)  # type: ignore
