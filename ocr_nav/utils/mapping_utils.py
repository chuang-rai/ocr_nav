import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, Union, List


def backproject_point(depth: np.ndarray, intrinsics: np.ndarray, u: int, v: int) -> np.ndarray:
    """Backproject a single pixel with a depth image

    Args:
        depth (np.ndarray): (H, W) depth image.
        intrinsics (np.ndarray): (3, 3) Camera intrinsics.
        u (int): Pixel x-coordinate.
        v (int): Pixel y-coordinate.
    Returns:
        np.ndarray: (3,) 3D point in camera coordinates.
    """
    z = depth[v, u]
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.array([x, y, z])


def backproject_depth_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Backproject all pixels in the depth image.

    Args:
        depth (np.ndarray): (H, W) depth image.
        intrinsics (np.ndarray): (3, 3) Camera intrinsics.

    Returns:
        np.ndarray: (N, 3) 3D points in camera coordinates.
    """
    h, w = depth.shape
    i_coords = np.arange(w)
    j_coords = np.arange(h)
    u_coords, v_coords = np.meshgrid(i_coords, j_coords)
    u_coords = u_coords.flatten()
    v_coords = v_coords.flatten()
    z_coords = depth.flatten()
    x_coords = (u_coords - intrinsics[0, 2]) * z_coords / intrinsics[0, 0]
    y_coords = (v_coords - intrinsics[1, 2]) * z_coords / intrinsics[1, 1]
    points_3d = np.vstack((x_coords, y_coords, z_coords)).T
    return points_3d


def project_points(pc: np.ndarray, intrinsics: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D points to 2D image plane.

    Args:
        pc (np.ndarray): (N, 3) 3D points.
        intrinsics (np.ndarray): (3, 3) Camera intrinsics.
        w (int): Image width.
        h (int): Image height.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (N, 2) 2D projected points, (N,) depth values, (N, 3) filtered 3D points.
    """
    # filter points behind the camera
    pc = pc[pc[:, 2] > 0]

    pc_2d_d = intrinsics @ pc.T  # (3, N)
    pc_2d = pc_2d_d[:2, :] / pc_2d_d[2, :]
    # filter points outside the image
    mask = (pc_2d[0, :] >= 0) & (pc_2d[0, :] < w) & (pc_2d[1, :] >= 0) & (pc_2d[1, :] < h)
    pc = pc[mask, :]
    pc_2d = pc_2d[:, mask]
    return pc_2d[:2, :].T, pc_2d_d[2, :], pc  # (N, 2), (N,), (N, 3)


def downsample_point_cloud(
    pc: Union[np.ndarray, o3d.geometry.PointCloud], voxel_size: float = 0.05
) -> o3d.geometry.PointCloud:
    """Apply voxel downsampling to a point cloud.

    Args:
        pc (Union[np.ndarray, o3d.geometry.PointCloud]): (N, 3) point cloud or Open3D PointCloud object.
        voxel_size (float, optional): voxel size for downsampling. Defaults to 0.05.

    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud.
    """
    if isinstance(pc, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
    else:
        pcd = pc
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def points_to_mesh(
    pc: Union[np.ndarray, o3d.geometry.PointCloud], voxel_size: float = 0.05
) -> o3d.geometry.TriangleMesh:
    """Create a mesh with the point cloud based on ball pivoting algorithm.

    Args:
        pc (Union[np.ndarray, o3d.geometry.PointCloud]): (N, 3) point cloud or Open3D PointCloud object.
        voxel_size (float, optional): The voxel size for mesh generation. Defaults to 0.05.

    Returns:
        o3d.geometry.TriangleMesh: Generated mesh.
    """
    if isinstance(pc, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
    else:
        pcd = pc
    pcd.estimate_normals()

    mean_distance = pcd.compute_nearest_neighbor_distance()
    voxel_size = voxel_size if voxel_size is not None else mean_distance.mean()
    radius = voxel_size * 2
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, 2 * radius])
    )

    return mesh


def segment_floor(pc: np.ndarray, resolution: float = 0.05, vis: bool = False) -> np.ndarray:
    """Segment floor heights from a point cloud based on histogram analysis.

    Args:
        pc (np.ndarray): (N, 3) point cloud.
        resolution (float, optional): Resolution for histogram bins. Defaults to 0.05.
        vis (bool, optional): Whether to visualize the histogram and detected peaks. Defaults to False.

    Returns:
        np.ndarray: Detected floor heights.
    """

    z = pc[:, 2]
    bins_num = (np.max(z) - np.min(z)) / resolution
    z_hist = np.histogram(z, bins=int(bins_num))
    z_hist_smooth = gaussian_filter1d(z_hist[0], sigma=2)
    distance = 0.2 / resolution
    min_peak_height = np.percentile(z_hist_smooth, 90)
    peaks, _ = find_peaks(z_hist_smooth, distance=distance, height=min_peak_height)
    heights = peaks * resolution + np.min(z)
    if vis:
        plt.figure()
        plt.plot(z_hist[1][:-1], z_hist_smooth)
        plt.plot(z_hist[1][peaks], z_hist_smooth[peaks], "x")
        plt.hlines(min_peak_height, np.min(z_hist[1]), np.max(z_hist[1]), colors="r")
        plt.show()
    return np.array(heights)
