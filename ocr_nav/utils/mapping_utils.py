import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, Union, List


def backproject_point(depth: np.ndarray, intrinsics: np.ndarray, u: int, v: int) -> np.ndarray:
    z = depth[v, u]
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.array([x, y, z])


def backproject_depth_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
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
    """
    Docstring for project_points

    :param pc: (N, 3)
    :type pc: np.ndarray
    :param intrinsics: (3, 3)
    :type intrinsics: np.ndarray
    :return: 2D points (N, 2)
    :rtype: ndarray float
    :return: depth (N,)
    :rtype: ndarray float
    :return: 3D points (N, 3)
    :rtype: ndarray float
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
    """
    Docstring for downsample_point_cloud

    :param pc: (N, 3)
    :type pc: np.ndarray
    :param voxel_size: voxel size for downsampling
    :type voxel_size: float
    :return: downsampled point cloud
    :rtype: o3d.geometry.PointCloud
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
    """
    Docstring for points_to_mesh

    :param pc: (N, 3)
    :type pc: np.ndarray
    :param voxel_size: voxel size for downsampling
    :type voxel_size: float
    :return: mesh
    :rtype: o3d.geometry.TriangleMesh
    """
    if isinstance(pc, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
    else:
        pcd = pc
    pcd.estimate_normals()
    # mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    mean_distance = pcd.compute_nearest_neighbor_distance()
    voxel_size = voxel_size if voxel_size is not None else mean_distance.mean()
    radius = voxel_size * 2
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, 2 * radius])
    )

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=voxel_size)

    return mesh


def segment_floor(pc: np.ndarray, resolution: float = 0.05, vis: bool = False) -> np.ndarray:

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
