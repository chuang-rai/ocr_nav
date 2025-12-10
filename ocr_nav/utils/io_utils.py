from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def load_image(image_path: Path) -> Image.Image:

    try:
        image = Image.open(image_path)

        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image

    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def load_depth(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32)
    return depth


def load_intrinsics(intrinsics_path: Path):
    with open(intrinsics_path, "r") as f:
        line = f.readline()
        line = [float(x.strip()) for x in line.strip().split(",")]

    fx, fy, cx, cy = line[0], line[4], line[2], line[5]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsics


def load_pose(pose_path: Path) -> np.ndarray:
    with open(pose_path, "r") as f:
        line = f.readline()
        line = [float(x) for x in line.strip().split()]
    assert len(line) == 7
    pose = np.eye(4)
    pose[:3, 3] = np.array([float(x) for x in line[0:3]])
    rot = R.from_quat([float(x) for x in line[3:7]])
    pose[:3, :3] = rot.as_matrix()
    return pose


def load_lidar(lidar_path: Path) -> np.ndarray:
    """
    Docstring for load_lidar

    :param lidar_path: lidar path
    :type lidar_path: Path
    :return: (N, 3)
    :rtype: ndarray
    """
    return np.load(lidar_path)
