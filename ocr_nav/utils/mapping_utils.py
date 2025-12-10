import numpy as np
from typing import Tuple


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
