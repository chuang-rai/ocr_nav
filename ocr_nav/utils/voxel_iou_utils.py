import open3d as o3d
import numpy as np


def compute_voxel_iou(voxel_grid_a: set[tuple[int, int, int]], voxel_grid_b: set[tuple[int, int, int]]) -> float:
    # 1. Convert indices to a set of tuples for fast set operations
    # 2. Calculate Intersection and Union
    intersection = len(voxel_grid_a.intersection(voxel_grid_b))
    union = len(voxel_grid_a.union(voxel_grid_b))

    iou = intersection / union if union > 0 else 0
    return iou
