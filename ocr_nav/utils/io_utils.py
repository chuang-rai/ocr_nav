import os
import yaml
import json
from PIL import Image, ImageOps
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
from tqdm import tqdm
import cv2
from plyfile import PlyData

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2
from sensor_msgs.msg import Image as ROSImage
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from message_filters import Cache, Subscriber, ApproximateTimeSynchronizer
from collections import deque

# TF2 Imports
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from ocr_nav.utils.mapping_utils import transform_point_cloud

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


def encode_image_to_bytes(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed.")
    return buffer.tobytes()


def encode_image_to_base64_string(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed.")
    img_bytes = buffer.tobytes()
    import base64

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


def waypoints_to_yaml(path: Path, waypoints: np.ndarray) -> None:
    waypoints_list = []
    for waypoint in waypoints:
        position = waypoint[:3].tolist()
        orientation = waypoint[3:].tolist()  # Assuming quaternion (w, x, y, z)
        waypoints_list.append(
            {
                "position": position,
                "orientation": orientation,
                "tags": [],
            }
        )

    yaml_dict = {
        "interpolation_method": "none",
        "max_distance": 0.1,
        "waypoints": waypoints_list,
    }

    with open(path, "w") as f:
        yaml.dump(yaml_dict, f)


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
        annotation_name: str = "qwen3vl_annotations",
    ):
        self.root_dir = root_dir
        self.img_dir = root_dir / img_name
        self.camera_pose_dir = root_dir / camera_pose_name
        self.depth_dir = root_dir / depth_name
        self.livox_dir = root_dir / livox_name
        self.rslidar_dir = root_dir / rslidar_name
        self.mask_dir = root_dir / mask_name
        self.annotation_dir = root_dir / annotation_name
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

    def get_image(self, index: int, prefix: str = "image_") -> Image.Image:
        timestamp = self.timestamp_list[index]
        image_path = self.img_dir / f"{prefix}{timestamp}.jpg"
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

    def get_annotation(self, index: int) -> dict:
        timestamp = self.timestamp_list[index]
        annotation_path = self.annotation_dir / f"qwen3vl_{timestamp}.json"
        if not annotation_path.exists():
            return {}

        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        return annotation

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


def convert_pc2_to_numpy(cloud_msg: PointCloud2) -> np.ndarray:
    # Returns a structured NumPy array, typically (N,) with fields 'x', 'y', 'z', etc.
    structured_array = pc2.read_points(cloud_msg, field_names=("x", "y", "z"))

    # You may need to restructure it into an N x 3 array:
    points = np.stack(
        [structured_array["x"], structured_array["y"], structured_array["z"]],
        axis=-1,
    )

    return points


def convert_numpy_to_pc2(points: np.ndarray, frame_id: str, stamp: Time) -> PointCloud2:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    pc2_msg = pc2.create_cloud_xyz32(header, points.tolist())
    return pc2_msg


def convert_compressed_image_to_numpy(compressed_image_msg: ROSImage) -> np.ndarray:
    try:
        np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)

        # 2. Decode the image into a color (BGR) numpy array
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except CvBridgeError as e:
        print(f"CvBridge Error: {e}")
        return None
    return cv_image


def convert_image_to_numpy(image_msg: ROSImage) -> np.ndarray:
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(f"CvBridge Error: {e}")
        return None
    return cv_image


def msg_time_within_latency(msg_time: Time, reference_time: Time, max_latency_s: float) -> bool:
    time_diff = abs((msg_time.sec - reference_time.sec) + (msg_time.nanosec - reference_time.nanosec) / 1_000_000_000)
    return time_diff <= max_latency_s


class BagIO(Node):
    """BagIO is a ROS2 node for reading and processing data from a rosbag.
    Provide camera and lidar data topics to the class for synchronized data reading.
    The synchronization is based on the anchor lidar topic specified by the user.

    Usage:
        import rclpy
        rclpy.init()
        bag_io_node = BagIO(
            bag_path=Path("/path/to/your.bag"),
            rgb_topic="/zed/zed_node/left/color/rect/image/compressed",
            camera_info_topic="/zed/zed_node/left/color/rect/image/camera_info",
            anchor_lidar_id=0,
            lidar_topic_list=["/glim_ros/points_corrected", "/rslidar_points"],
            lidar_frame_ids=["livox_mid360_imu", "robosense_e1r"],
            camera_frame_id="zed_left_camera_optical_frame",
            max_bag_total_time=3600,
            sample_every=1,
        )
        bag_io_node.init_reader()
        while bag_io_node.has_next():
            data = bag_io_node.get_next_sync_data(max_latency_s=0.2)
            if data is not None:
                lidar_pc_list, img_np, anchor_lidar_pose, timenano = data
            # Process the synchronized data as needed


    Args:
        Node (_type_): _description_
    """

    def __init__(
        self,
        bag_path: Path,
        rgb_topic: str = "/zed/zed_node/left/color/rect/image/compressed",
        camera_info_topic: str = "/zed/zed_node/left/color/rect/image/camera_info",
        camera_frame_id: str = "zed_left_camera_optical_frame",
        anchor_lidar_id: int = 0,
        lidar_topic_list: list[str] = ["/glim_ros/points_corrected", "/rslidar_points"],
        lidar_frame_ids: list[str] = ["livox_mid360_imu", "robosense_e1r"],
        world_frame_id: str = "odom_glim",
        max_bag_total_time: int = 3600,
        sample_every: int = 1,
    ):
        super().__init__("bagio_node")
        self.bag_path = bag_path
        self.sample_every = sample_every
        self.sync_msg_count = 0

        # 1. Define the topics we care about
        self.rgb_topic = rgb_topic
        self.camera_info_topic = camera_info_topic
        self.camera_frame_id = camera_frame_id

        self.anchor_lidar_id = anchor_lidar_id
        self.lidar_topic_list = lidar_topic_list
        self.non_anchor_lidar_topic_list = [topic for i, topic in enumerate(lidar_topic_list) if i != anchor_lidar_id]
        self.lidar_frame_ids = lidar_frame_ids

        self.world_frame_id = world_frame_id

        self.target_topics = {
            self.camera_info_topic,
            self.rgb_topic,
            *self.lidar_topic_list,
        }

        # 2. Set up storage and converter options
        self.storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="mcap")
        self.converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )

        # 3. Set up some flags for one-time saving
        self.intrinsic_saved = False

        # 4. Preload TF Buffer
        self._preload_tf_buffer(buffer_time_sec=max_bag_total_time)

    def _preload_tf_buffer(self, buffer_time_sec: int = 3600):
        # 1. Initialize TF Buffer and Listener (requires the node)
        unlimited_duration = rclpy.duration.Duration(seconds=buffer_time_sec)
        self.tf_buffer = Buffer(node=self, cache_time=unlimited_duration)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. Setup a bag reader
        reader = rosbag2_py.SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        # 3. Get topic information from the bag
        topic_types = reader.get_all_topics_and_types()
        self.type_map = {topic.name: topic.type for topic in topic_types}

        # 4. Go through the whole bag file to load TFs into the buffer
        search_time_list = []
        pbar = tqdm(total=buffer_time_sec)
        self.total_duration_sec = 0
        self.start_sec = None
        while reader.has_next():
            topic, data, t_nanosec = reader.read_next()
            timestamp_sec = t_nanosec // 1_000_000_000
            if self.start_sec is None:
                self.start_sec = timestamp_sec

            self.total_duration_sec = timestamp_sec - self.start_sec
            if topic not in {"/tf", "/tf_static", self.camera_info_topic}:
                continue

            # add /tf and /tf_static messages to the tf_buffer
            if topic == "/tf":
                msg = deserialize_message(data, self.get_msg_type(topic))
                assert isinstance(msg, TFMessage)
                for transform in msg.transforms:
                    test_time = transform.header.stamp
                    self.tf_buffer.set_transform(transform, "default_authority")
                search_time_list.append(test_time)
            elif topic == "/tf_static":
                msg = deserialize_message(data, self.get_msg_type(topic))
                assert isinstance(msg, TFMessage)
                for transform in msg.transforms:
                    self.tf_buffer.set_transform_static(transform, "default_authority")
                    print(f"Saving static transform from {transform.header.frame_id} to {transform.child_frame_id}")
            elif not self.intrinsic_saved and topic == self.camera_info_topic:
                msg = deserialize_message(data, self.get_msg_type(topic))
                assert isinstance(msg, CameraInfo)
                self.save_intrinsics(msg)
            pbar.update(timestamp_sec - self.start_sec - pbar.n)
            pbar.set_description(f"Loading TFs for time: {timestamp_sec}s")

        # save the tf from lidar to camera frame
        self.save_lidar2camera_tfs()

    # Helper function to get the message type (Python class) from its name
    def get_msg_type(self, topic_name):
        assert hasattr(self, "type_map")
        try:
            return get_message(self.type_map[topic_name])
        except Exception:
            return None

    def save_intrinsics(self, camera_info: CameraInfo) -> None:
        assert isinstance(camera_info, CameraInfo)
        k = [float(x) for x in camera_info.k]
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.h = camera_info.height
        self.w = camera_info.width
        self.intrinsic_saved = True

    def get_intrinsics(self) -> np.ndarray:
        return self.intrinsics

    def get_image_size(self) -> tuple[int, int]:
        return self.h, self.w

    def save_lidar2camera_tfs(self) -> None:
        self.lidar2camera_tfs = []
        for lidar_frame_id in self.lidar_frame_ids:
            tf = self.lookup_tfs(self.camera_frame_id, lidar_frame_id, rclpy.time.Time())
            self.lidar2camera_tfs.append(tf)

    def get_lidar2camera_tfs(self) -> list[np.ndarray]:
        return self.lidar2camera_tfs

    def init_reader(self) -> None:
        self.get_logger().info(f"Opening bag file at: {self.bag_path}")
        # 1. Define a new rosbag reader
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(self.storage_options, self.converter_options)

        topic_types = self.reader.get_all_topics_and_types()
        self.type_map = {topic.name: topic.type for topic in topic_types}
        self.pbar = tqdm(total=self.total_duration_sec)

    def has_next(self) -> bool:
        return self.reader.has_next()

    def get_next_sync_data(
        self, max_latency_s: float = 0.2
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray] | None:
        """Get the next synchronized data from lidar_topic_list and anchor lidar's pose in Glim frame

        Returns:
            Union[Tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray], None]:
            Tuple containing:
                - lidar_pc_list (list[np.ndarray]): List of point clouds from the lidars. Each element is (N, 3).
                - img_np (np.ndarray): (H, W, 3) RGB image.
                - anchor_lidar_pose (np.ndarray): (4, 4) anchor lidar pose in Glim frame.
                - timenano (np.ndarray): Timestamp in nanoseconds.
        """
        if not self.reader.has_next():
            return None

        latest_msgs = {
            self.rgb_topic: None,
            **{
                lidar_topic: None
                for lidar_i, lidar_topic in enumerate(self.lidar_topic_list)
                if lidar_i != self.anchor_lidar_id
            },
        }

        while self.reader.has_next():
            topic, data, t_nanosec = self.reader.read_next()
            timestamp_sec = t_nanosec // 1_000_000_000
            self.pbar.update(timestamp_sec - self.start_sec - self.pbar.n)

            if topic not in self.target_topics:
                continue

            msg_type = self.get_msg_type(topic)
            if not msg_type:
                continue

            # 5. Deserialization
            msg = deserialize_message(data, msg_type)

            # Update the 'latest' buffer for non-anchor topics
            if topic in latest_msgs:
                latest_msgs[topic] = msg
                continue  # Don't save yet, wait for the trigger (anchor lidar)

            # if the message is from the anchor lidar, trigger synchronization
            elif topic == self.lidar_topic_list[self.anchor_lidar_id]:
                sync_time = msg.header.stamp

                # initialize placeholder for lidar pcs
                lidar_pc_list = [None] * len(self.lidar_topic_list)

                # save anchor lidar pc
                lidar_pc_list[self.anchor_lidar_id] = convert_pc2_to_numpy(msg)  # (N, 3)

                if not latest_msgs[self.rgb_topic] or any(
                    [not latest_msgs[lidar_topic] for lidar_topic in self.non_anchor_lidar_topic_list]
                ):
                    continue

                if latest_msgs[self.rgb_topic]:
                    img_msg = latest_msgs[self.rgb_topic]
                    if not msg_time_within_latency(img_msg.header.stamp, sync_time, max_latency_s):
                        print(
                            "Skipping due to latency: anchor lidar time = "
                            + str(sync_time.sec)
                            + "."
                            + str(sync_time.nanosec)
                            + ", Image time = "
                            + str(img_msg.header.stamp.sec)
                            + "."
                            + str(img_msg.header.stamp.nanosec)
                        )
                        continue

                    # latest_msgs[self.rgb_topic] = None  # Clear after use
                    assert isinstance(img_msg, CompressedImage), f"Expected CompressedImage, got {type(img_msg)}"
                    img_bgr_np = convert_compressed_image_to_numpy(img_msg)  # (H, W, 3)
                    img_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                for lidar_i, lidar_topic in enumerate(self.lidar_topic_list):
                    if lidar_i == self.anchor_lidar_id:
                        continue
                    if latest_msgs[lidar_topic]:
                        lidar_msg = latest_msgs[lidar_topic]
                        if not msg_time_within_latency(lidar_msg.header.stamp, sync_time, max_latency_s):
                            continue
                        assert isinstance(lidar_msg, PointCloud2)

                        # Convert the message into numpy array
                        lidar_pc = convert_pc2_to_numpy(lidar_msg)  # (N, 3)
                        lidar_pc_list[lidar_i] = lidar_pc

                # --- The crucial fix: Use the message's timestamp for the lookup ---
                lookup_time_rclpy = rclpy.time.Time.from_msg(sync_time)

                try:
                    # Look up the transform from the pose frame to the map frame at the pose time
                    # Note: Using the message's timestamp (lookup_time) is crucial for accurate TF.
                    anchor_lidar_pose = self.lookup_tfs(
                        self.world_frame_id, self.lidar_frame_ids[self.anchor_lidar_id], lookup_time_rclpy
                    )

                except Exception as e:
                    self.get_logger().warn(
                        f"TF Lookup Error for synchronized pose at time {msg.header.stamp.sec}."
                        f"{msg.header.stamp.nanosec}: {e}"
                    )
                    continue
                self.sync_msg_count += 1
                if self.sync_msg_count % self.sample_every != 0:
                    continue
                return lidar_pc_list, img_np, anchor_lidar_pose, t_nanosec

        return None

    def lookup_tfs(self, tar_frame_id: str, src_frame_id: str, time: rclpy.time.Time) -> np.ndarray:
        assert hasattr(self, "tf_buffer")
        stamped_tf = self.tf_buffer.lookup_transform(tar_frame_id, src_frame_id, time)
        t = stamped_tf.transform.translation
        rot = stamped_tf.transform.rotation
        pose = np.eye(4)
        pose[0, 3] = t.x
        pose[1, 3] = t.y
        pose[2, 3] = t.z
        r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        pose[:3, :3] = r.as_matrix()
        return pose


class SubscriberIO(Node):

    def __init__(self, tf_buffer_sec: int = 600):
        super().__init__("floor_graph_subscriber")

        # 1. Setup TF2 Buffer and Listener
        self.tf_buffer = Buffer(node=self, cache_time=rclpy.duration.Duration(seconds=tf_buffer_sec))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 2. Standard Topic Subscription
        self.sub_camera_info = self.create_subscription(
            CameraInfo, "/zed/zed_node/left/color/rect/image/camera_info", self.camera_info_callback, 10
        )
        self.sub_livox = Subscriber(self, PointCloud2, "/glim_ros/points_corrected")
        self.sub_rslidar = Subscriber(self, PointCloud2, "/rslidar_points")
        self.sub_mask = Subscriber(self, ROSImage, "/segmentation_mask")
        self.pub_acc_livox = self.create_publisher(PointCloud2, "/acc_livox", 10)

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.sub_livox, self.sub_rslidar, self.sub_mask], queue_size=10, slop=0.1
        )
        self.synchronizer.registerCallback(self.synchronized_callback)

        # 3. Buffer
        self.sync_buffer = deque(maxlen=20)

    def camera_info_callback(self, msg: CameraInfo):
        k = [float(x) for x in msg.k]
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        self.h = msg.height
        self.w = msg.width
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def synchronized_callback(self, livox_msg: PointCloud2, rslidar_msg: PointCloud2, mask_msg: ROSImage):
        try:
            # lookup_transform(target_frame, source_frame, time)
            listened_time = rclpy.time.Time.from_msg(livox_msg.header.stamp)
            tf_livox_to_odom = self.lookup_tfs("odom_glim", "livox_mid360_imu", listened_time)  # (4, 4)
            livox_pc_np = self.convert_pc2_to_numpy(livox_msg)  # (N, 3)
            rslidar_pc_np = self.convert_pc2_to_numpy(rslidar_msg)  # (N, 3)
            mask_np = self.convert_image_to_numpy(mask_msg)  # (H, W, C) or (H, W)
            global_pc_numpy = transform_point_cloud(livox_pc_np, tf_livox_to_odom)
            pub_msg = self.convert_numpy_to_pc2(global_pc_numpy, "odom_glim", listened_time.to_msg())
            self.pub_acc_livox.publish(pub_msg)

            self.get_logger().info(f"Found TF at: \n{tf_livox_to_odom}")
            self.sync_buffer.append((livox_pc_np, rslidar_pc_np, mask_np, tf_livox_to_odom))

        except TransformException as ex:
            self.get_logger().info(f"Could not transform /odom_glim to /livox_mid360_imu: {ex}")

    def get_latest_sync_data(self) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
        """Get the latest synchronized data containing: Livox Lidar, RS Lidar, Ground segmentation
        mask, and a tf (4,4) that transforms point cloud from the Livox frame to the Global frame

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
            (livox_pc_np (N, 3), rslidar_pc_np (M, 3), mask_np (H, W), tf_livox_to_odom (4, 4))
        """
        if len(self.sync_buffer) == 0:
            self.get_logger().warning("No synchronized data available yet.")
            return None
        return self.sync_buffer[-1]

    def get_intrinsics(self) -> np.ndarray:
        return self.intrinsics

    def get_image_size(self) -> Tuple[int, int]:
        return self.h, self.w

    def get_livox2left_camera_tf(self) -> np.ndarray:
        try:
            self.livox2left_camera_tf = self.lookup_tfs(
                "zed_left_camera_optical_frame", "livox_mid360", rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().info(f"Could not transform zed_left_camera_optical_frame to livox_mid360: {ex}")
            return None
        return self.livox2left_camera_tf

    def get_rslidar2left_camera_tf(self) -> np.ndarray:
        try:
            self.rslidar2left_camera_tf = self.lookup_tfs(
                "zed_left_camera_optical_frame", "robosense_e1r", rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().info(f"Could not transform zed_left_camera_optical_frame to robosense_e1r: {ex}")
            return None
        return self.rslidar2left_camera_tf

    def lookup_tfs(self, tar_frame_id: str, src_frame_id: str, time: rclpy.time.Time) -> np.ndarray:
        assert hasattr(self, "tf_buffer")
        stamped_tf = self.tf_buffer.lookup_transform(tar_frame_id, src_frame_id, time)
        t = stamped_tf.transform.translation
        rot = stamped_tf.transform.rotation
        pose = np.eye(4)
        pose[0, 3] = t.x
        pose[1, 3] = t.y
        pose[2, 3] = t.z
        r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        pose[:3, :3] = r.as_matrix()
        return pose

    def convert_pc2_to_numpy(self, cloud_msg: PointCloud2) -> np.ndarray:
        # Returns a structured NumPy array, typically (N,) with fields 'x', 'y', 'z', etc.
        structured_array = pc2.read_points(cloud_msg, field_names=("x", "y", "z"))

        # You may need to restructure it into an N x 3 array:
        points = np.stack(
            [structured_array["x"], structured_array["y"], structured_array["z"]],
            axis=-1,
        )

        return points

    def convert_numpy_to_pc2(self, points: np.ndarray, frame_id: str, stamp: Time) -> PointCloud2:
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        pc2_msg = pc2.create_cloud_xyz32(header, points.tolist())
        return pc2_msg

    def convert_image_to_numpy(self, image_msg: ROSImage) -> np.ndarray:
        bridge = CvBridge()
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return None
        return cv_image
