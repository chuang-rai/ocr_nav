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

from ocr_nav.utils.io_utils import load_pose, load_depth, load_intrinsics, load_lidar
from ocr_nav.utils.mapping_utils import backproject_depth_map


def main():
    pose_dir = Path(
        "/home/chuang/hcg/projects/ocr/data/giorgio_data/gtonetti_data_zurich_hb_3_merged_pose_zed_gen2_depth_fs_zurich_hb_3_5_bag_zurich_hb_3_5_2/pose"
    )
    root_dir = pose_dir.parent
    debug = False
    if debug:
        debug_image_dir = root_dir / "debug_lidar_proj"
        debug_image_dir.mkdir(exist_ok=True)

    intr_mat = load_intrinsics(root_dir / "intrinsics.txt")
    robosense2zed_left_mat_path = root_dir / "tf_robosense_e1r_to_zed_left_camera_optical_frame.txt"
    tf_zed_left2rs = load_pose(robosense2zed_left_mat_path)
    global_pcd = o3d.geometry.PointCloud()
    for pi, p in enumerate(sorted(pose_dir.iterdir())):
        print(f"Processing frame {pi}")
        frame_id = int(p.stem.split("_")[-1])

        pose_path = root_dir / "pose" / f"pose_{frame_id}.txt"
        lidar_path = root_dir / "rslidar" / f"rslidar_{frame_id}.npy"
        pose = load_pose(pose_path)
        pc_rs = load_lidar(lidar_path)  # (N, 3)

        # transform point cloud from robosense frame to zed left camera frame
        pc_rs_hom = np.hstack((pc_rs, np.ones((pc_rs.shape[0], 1))))  # (N, 4)
        pc_world_hom = pose @ tf_zed_left2rs @ pc_rs_hom.T  # (N, 4)

        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(pc_world_hom[:3, :].T)
        global_pcd += pcd_frame

        # voxel sampling
        global_pcd = global_pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([global_pcd])  # type: ignore


if __name__ == "__main__":
    main()
