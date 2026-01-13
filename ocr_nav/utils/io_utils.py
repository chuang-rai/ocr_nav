from PIL import Image, ImageOps
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData
from typing import List, Tuple, Union


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    corrected_image = ImageOps.exif_transpose(image)
    return corrected_image


def load_depth(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_UNCHANGED)
    assert depth is not None
    depth = depth.astype(np.float32)
    return depth


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """Load the camera intrinsics from a text file. The content in the file
    is expected to be in the format:
    fx, 0, cx, 0, fy, cy, 0, 0, 1

    Args:
        intrinsics_path (Path): Path to the intrinsics file.

    Returns:
        np.ndarray: (3, 3) Camera intrinsics matrix.
    """
    with open(intrinsics_path, "r") as f:
        line = f.readline()
        line = [float(x.strip()) for x in line.strip().split(",")]

    fx, fy, cx, cy = line[0], line[4], line[2], line[5]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsics


def load_pose(pose_path: Path) -> np.ndarray:
    """Load the 6DOF pose from a txt file. The content of the file
    is expected to be in the format:
    tx, ty, tz, qx, qy, qz, qw


    Args:
        pose_path (Path): Path to the pose file.

    Returns:
        np.ndarray: (4, 4) Pose matrix.
    """
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
    """Load the lidar point cloud from a numpy file. The content of the file
    is expected to be in the format:
    (N, 3) array of point coordinates.

    Args:
        lidar_path (Path): Path to the lidar .npy file.

    Returns:
        np.ndarray: (N, 3) Lidar point cloud.
    """
    return np.load(lidar_path)


def load_livox_poses_timestamps(poses_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Livox poses and corresponding timestamps from an npy file.
    Each row in the array is expected to be:
    (tx, ty, tz, qx, qy, qz, qw, timestamp)

    Args:
        poses_path (Path): Path to the poses .npy file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing an array of poses (N, 4, 4) and an array of timestamps (N,).
    """
    poses = []
    poses_and_timestamps = np.load(poses_path)
    for line in poses_and_timestamps:
        pose = np.eye(4)
        pose[:3, 3] = line[:3]
        rot = R.from_quat(line[3:7])
        pose[:3, :3] = rot.as_matrix()
        poses.append(pose)
    return np.array(poses), np.array(poses_and_timestamps[:, -1])


def load_masks(masks_path: Path) -> np.ndarray:
    """Load the segmentation mask of a certain object from an npy file.
    The content of the file is expected to be: an (H, W) numpy array

    Args:
        masks_path (Path): Path to the masks .npy file.

    Returns:
        np.ndarray: (H, W) Segmentation mask.
    """
    masks = np.load(masks_path)
    return masks


def find_closest(sorted_arr: np.ndarray, target: float) -> int:
    """Find the index of the closest value to the target in a sorted array.

    Args:
        sorted_arr (np.ndarray): Sorted array of values.
        target (float): Target value to find the closest to.

    Returns:
        int: Index of the closest value in the array.
    """

    # Find the index where 'target' would be inserted to maintain order
    idx = np.searchsorted(sorted_arr, target)

    # If the target is smaller than the first element
    if idx == 0:
        return 0

    # If the target is larger than the last element
    if idx == len(sorted_arr):
        return len(sorted_arr) - 1

    # Otherwise, compare the neighbors
    before = sorted_arr[idx - 1]
    after = sorted_arr[idx]

    if abs(after - target) < abs(before - target):
        return idx
    else:
        return idx - 1


def search_latest_poses_within_timestamp_range(
    poses: np.ndarray, timestamps: np.ndarray, start_timestamp_sec_nano: str
) -> Union[Tuple[np.ndarray, np.int64], Tuple[None, None]]:
    """Search for the latest pose within a given timestamp range.

    Args:
        poses (np.ndarray): a list of poses (N, 4, 4)
        timestamps (np.ndarray): a list of timestamps (N,)
        start_timestamp_sec_nano (str): Start timestamp in the format "seconds_nanoseconds".

    Returns:
        Union[Tuple[np.ndarray, np.int64], Tuple[None, None]]: Tuple containing the latest pose and its timestamp, or (None, None) if not found.
    """
    ids = np.argsort(timestamps)
    sorted_timestamps = timestamps[ids]
    sorted_poses = poses[ids]

    start_sec, start_nano = [int(x) for x in start_timestamp_sec_nano.split("_")]
    start_timestamp = float(start_sec) + float(start_nano) * 1e-9
    id = find_closest(sorted_timestamps, start_timestamp)
    found_timestamp = sorted_timestamps[id] * 1e9
    if found_timestamp - start_sec * 1e9 - start_nano > 1e8:
        return None, None
    return sorted_poses[id], sorted_timestamps[id]


def load_ply_point_cloud(ply_path: Path) -> np.ndarray:

    plydata = PlyData.read(ply_path)
    vertex_data = plydata["vertex"].data
    points = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T
    return points  # (N, 3)


class FolderIO:
    """Class to handle I/O operations for a folder containing various sensor data.
    The folder structure is expected to be:
    root_dir/
        left/                # images
            image_*.jpg
        pose/                # camera poses
            pose_*.txt
        depth/               # depth maps
            depth_*.png
        livox/              # livox point clouds
            livox_*.npy
        rslidar/            # rslidar point clouds
            rslidar_*.npy
        masks_sam2_s/       # segmentation masks
            mask_*.npy
        intrinsics.txt      # camera intrinsics
        tf_livox_mid360_to_zed_left_camera_optical_frame.txt
        tf_robosense_e1r_to_zed_left_camera_optical_frame.txt
    The timestamps in filenames of different sensorsare expected to align with each other.
    """

    def __init__(
        self,
        root_dir: Path,
        img_name: str = "left",
        camera_pose_name: str = "pose",  # set folder name to "" if there is no such data
        depth_name: str = "depth",
        livox_name: str = "livox",
        rslidar_name: str = "rslidar",
        mask_name: str = "masks_sam2_s",
    ):
        self.root_dir = root_dir
        self.img_dir = root_dir / img_name
        self.camera_pose_dir = root_dir / camera_pose_name
        self.depth_dir = root_dir / depth_name
        self.livox_dir = root_dir / livox_name
        self.rslidar_dir = root_dir / rslidar_name
        self.mask_dir = root_dir / mask_name

        self.timestamp_list = ["_".join(x.stem.split("_")[-2:]) for x in sorted(self.img_dir.iterdir())]
        self.len = len(self.timestamp_list)

    def check_timestamp_consistency(self, folder: Path) -> bool:
        assert hasattr(self, "timestamp_list"), "FolderIO not initialized properly."
        folder_timestamp_set = set("_".join(x.stem.split("_")[-2:]) for x in sorted(folder.iterdir()))
        img_timestamp_set = set(self.timestamp_list)
        if img_timestamp_set != folder_timestamp_set:
            diff_set_1 = folder_timestamp_set - img_timestamp_set
            diff_set_2 = img_timestamp_set - folder_timestamp_set
            for ts in diff_set_1:
                print(f"Timestamp {ts} exists in {folder} folder but not in image folder.")
            for ts in diff_set_2:
                print(f"Timestamp {ts} exists in image folder but not in {folder} folder.")
            return False
        else:
            return True

    def get_image(self, index: int) -> Image.Image:
        timestamp = self.timestamp_list[index]
        image_path = self.img_dir / f"image_{timestamp}.jpg"
        image = load_image(image_path)
        return image

    def get_camera_pose(self, index: int) -> np.ndarray:
        timestamp = self.timestamp_list[index]
        pose_path = self.camera_pose_dir / f"{self.camera_pose_dir.stem}_{timestamp}.txt"
        pose = load_pose(pose_path)
        return pose

    def get_depth(self, index: int) -> np.ndarray:
        timestamp = self.timestamp_list[index]
        depth_path = self.depth_dir / f"{self.depth_dir.stem}_{timestamp}.png"
        depth = load_depth(depth_path)
        return depth

    def get_livox(self, index: int) -> np.ndarray:
        timestamp = self.timestamp_list[index]
        livox_path = self.livox_dir / f"{self.livox_dir.stem}_{timestamp}.npy"
        livox = load_lidar(livox_path)
        return livox  # (N, 3)

    def get_rslidar(self, index: int) -> np.ndarray:
        timestamp = self.timestamp_list[index]
        rslidar_path = self.rslidar_dir / f"{self.rslidar_dir.stem}_{timestamp}.npy"
        rslidar = load_lidar(rslidar_path)
        return rslidar  # (N, 3)

    def get_mask(self, index: int) -> np.ndarray:
        timestamp = self.timestamp_list[index]
        mask_path = self.mask_dir / f"mask_{timestamp}.npy"
        mask = load_masks(mask_path)
        return mask

    def get_intrinsics(self) -> np.ndarray:
        intrinsics_path = self.root_dir / "intrinsics.txt"
        intrinsics = load_intrinsics(intrinsics_path)
        return intrinsics

    def get_livox2left_camera_tf(self) -> np.ndarray:
        tf_path = self.root_dir / "tf_livox_mid360_to_zed_left_camera_optical_frame.txt"
        tf = load_pose(tf_path)
        return tf

    def get_rslidar2left_camera_tf(self) -> np.ndarray:
        tf_path = self.root_dir / "tf_robosense_e1r_to_zed_left_camera_optical_frame.txt"
        tf = load_pose(tf_path)
        return tf


class BagIO:
    def __init__(self, bag_path: Path):
        self.bag_path = bag_path
