import os
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
    def __init__(self, bag_path: Path, buffer_time_sec: int = 3600, sample_every: int = 1):
        self.bag_path = bag_path

        # 1. Define the topics we care about
        # self.pose_topic = "/zed/zed_node/pose"
        # self.rgb_topic = "/zed/zed_node/left/image_rect_color/compressed"
        self.rgb_topic = "/zed/zed_node/left/color/rect/image/compressed"
        # self.depth_topic = "/foundation_stereo/depth/image"
        # self.camera_info_topic = "/zed/zed_node/left/camera_info"
        self.camera_info_topic = "/zed/zed_node/left/color/rect/image/camera_info"
        # self.livox_lidar_topic = "/lidar_3389171904/test_points"
        self.livox_lidar_topic = "/glim_ros/points_corrected"
        self.robosense_lidar_topic = "/rslidar_points"  # tf: robosense_e1r -> zed_left_camera_optical_frame

        self.target_topics = {
            self.camera_info_topic,
            self.pose_topic,
            self.rgb_topic,
            # self.depth_topic,
            self.livox_lidar_topic,
            self.robosense_lidar_topic,
        }

        # 2. Set up storage and converter options
        self.storage_options = rosbag2_py.StorageOptions(uri=self.bag_file_path, storage_id="mcap")
        self.converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        )

        # 3. Set up some flags for one-time saving
        self.intrinsic_saved = False
        self.rslidar2left_img_tf_saved = False
        self.livox2left_img_tf_saved = False

        # 4. Preload TF Buffer
        self.preload_tf_buffer(buffer_time_sec=buffer_time_sec)

        # 5. Start reading the bag and saving per-frame information
        self.read_rosbag(sample_every=sample_every)

    def preload_tf_buffer(self, buffer_time_sec=3600):
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
        while reader.has_next():

            topic, data, t_nanosec = reader.read_next()
            timestamp_sec = t_nanosec // 1_000_000_000
            timestamp_nanosec = t_nanosec % 1_000_000_000
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
                self.save_intrinsics(self.output_dir, msg)
            pbar.update(timestamp_sec - self.start_time_sec - pbar.n)
            pbar.set_description(f"Loading TFs for time: {timestamp_sec}s")

        # save the tf from lidar to zed left camera frame
        self.save_livox2left_img_tf()
        self.save_rslidar2left_img_tf()

    # Helper function to get the message type (Python class) from its name
    def get_msg_type(self, topic_name):
        assert hasattr(self, "type_map")
        try:
            return get_message(self.type_map[topic_name])
        except Exception:
            return None

    def convert_pc2_to_numpy(self, cloud_msg: PointCloud2) -> np.ndarray:
        # Returns a structured NumPy array, typically (N,) with fields 'x', 'y', 'z', etc.
        structured_array = pc2.read_points(cloud_msg, field_names=("x", "y", "z"))

        # You may need to restructure it into an N x 3 array:
        points = np.stack(
            [structured_array["x"], structured_array["y"], structured_array["z"]],
            axis=-1,
        )

        return points

    def lookup_static_tfs(self, tar_frame_id: str, src_frame_id: str) -> np.ndarray:
        assert hasattr(self, "tf_buffer")
        stamped_tf = self.tf_buffer.lookup_transform(tar_frame_id, src_frame_id, rclpy.time.Time())
        t = stamped_tf.transform.translation
        rot = stamped_tf.transform.rotation
        pose = np.eye(4)
        pose[0, 3] = t.x
        pose[1, 3] = t.y
        pose[2, 3] = t.z
        r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        pose[:3, :3] = r.as_matrix()
        return pose

    def save_intrinsics(self, output_dir: str, camera_info: CameraInfo) -> None:
        assert isinstance(camera_info, CameraInfo)
        intrinsics_path = os.path.join(output_dir, "intrinsics.txt")
        k = [float(x) for x in camera_info.k]
        with open(intrinsics_path, "w") as f:
            f.write(",".join([str(x) for x in k]))
        print("Saved intrinsics to ", intrinsics_path)
        self.intrinsic_saved = True

    def save_livox2left_img_tf(self) -> None:
        if self.livox2left_img_tf_saved:
            return
        self.livox2left_img_tf = self.lookup_static_tfs("zed_left_camera_optical_frame", "livox_mid360")
        self.livox2left_img_tf_saved = True

    def save_rslidar2left_img_tf(self) -> None:
        if self.rslidar2left_img_tf_saved:
            return
        self.rslidar2left_img_tf = self.lookup_static_tfs("zed_left_camera_optical_frame", "robosense_e1r")
        self.rslidar2left_img_tf_saved = True

    def read_rosbag(self, sample_every: int = 1) -> None:

        self.get_logger().info(f"Opening bag file at: {self.bag_file_path}")
        # 1. Define a new rosbag reader
        reader = rosbag2_py.SequentialReader()
        reader.open(self.storage_options, self.converter_options)

        topic_types = reader.get_all_topics_and_types()
        self.type_map = {topic.name: topic.type for topic in topic_types}

        # 2. Setup output directories
        # self.left_dir = os.path.join(self.output_dir, "left")
        # self.depth_dir = os.path.join(self.output_dir, "depth")
        # self.pose_dir = os.path.join(self.output_dir, "pose")
        # self.rslidar_dir = os.path.join(self.output_dir, "rslidar")
        # self.livox_dir = os.path.join(self.output_dir, "livox")
        # os.makedirs(self.left_dir, exist_ok=True)
        # os.makedirs(self.depth_dir, exist_ok=True)
        # os.makedirs(self.pose_dir, exist_ok=True)
        # os.makedirs(self.rslidar_dir, exist_ok=True)
        # os.makedirs(self.livox_dir, exist_ok=True)
        bridge = CvBridge()

        # 3. Load the tf buffer first by iterating through all messages
        self.get_logger().info("Starting message iteration...")

        # 4. Initialize the SequentialReader for the second time
        self.get_logger().info(f"Opening bag file again at: {self.bag_file_path}")
        # Dictionary to store the most recent message of each type
        latest_msgs = {
            # self.livox_lidar_topic: None,
            self.rgb_topic: None,
            self.robosense_lidar_topic: None,
            self.pose_topic: None,
            self.depth_topic: None,
        }

        message_count = 0
        pbar = tqdm(total=self.end_time_sec - self.start_time_sec)
        while reader.has_next():
            topic, data, t_nanosec = reader.read_next()

            # --- PROCESS DATA MESSAGES ---
            timestamp_sec = t_nanosec // 1_000_000_000
            if timestamp_sec < self.start_time_sec:
                continue
            if timestamp_sec > self.end_time_sec:
                break
            if topic not in self.target_topics:
                continue

            msg_type = self.get_msg_type(topic)
            if not msg_type:
                continue

            # 5. Deserialization
            msg = deserialize_message(data, msg_type)
            msg_time = msg.header.stamp.sec

            # Update the 'latest' buffer for asynchronous sensors
            if topic in latest_msgs:
                latest_msgs[topic] = msg
                continue  # Don't save yet, wait for the trigger (RGB)

            if topic == self.livox_lidar_topic:
                sync_time = msg.header.stamp
                sync_ts_str = f"{sync_time.sec}_{sync_time.nanosec:09d}"
                # save livox
                pc2_np = self.convert_pc2_to_numpy(msg)  # (N, 3)
                pc_path = os.path.join(self.livox_dir, f"livox_{sync_ts_str}.npy")
                np.save(pc_path, pc2_np)

                if latest_msgs[self.rgb_topic]:
                    msg = latest_msgs[self.rgb_topic]
                    # latest_msgs[self.rgb_topic] = None  # Clear after use
                    assert isinstance(msg, CompressedImage), f"Expected CompressedImage, got {type(msg)}"
                    image_output_path = os.path.join(self.left_dir, f"image_{sync_ts_str}.jpg")
                    with open(image_output_path, "wb") as img_file:
                        img_file.write(msg.data)
                    self.get_logger().debug(f"Saved synchronized image to {image_output_path}")
                if latest_msgs[self.depth_topic]:
                    msg = latest_msgs[self.depth_topic]
                    # latest_msgs[self.depth_topic] = None  # Clear after use
                    assert isinstance(msg, ROSImage)
                    depth_output_path = os.path.join(self.depth_dir, f"depth_{sync_ts_str}.png")
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                        if msg.encoding == "32FC1":
                            cv_image = (cv_image * 1000).astype(cv2.CV_16UC1)

                        cv2.imwrite(depth_output_path, cv_image)
                        self.get_logger().debug(f"Saved synchronized depth to {depth_output_path}")

                    except (CvBridgeError, Exception) as e:
                        self.get_logger().error(f"Depth processing error: {e}")

                if latest_msgs[self.pose_topic]:
                    msg = latest_msgs[self.pose_topic]
                    # latest_msgs[self.pose_topic] = None  # Clear after use
                    assert isinstance(msg, PoseStamped)
                    pose_output_path = os.path.join(self.pose_dir, f"pose_{sync_ts_str}.txt")

                    # --- The crucial fix: Use the message's timestamp for the lookup ---
                    lookup_time = Time(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec)

                    try:
                        # Look up the transform from the pose frame to the map frame at the pose time
                        # Note: Using the message's timestamp (lookup_time) is crucial for accurate TF.
                        transform_stamped = self.tf_buffer.lookup_transform(
                            "map",
                            "zed_left_camera_optical_frame",
                            lookup_time,
                        )
                        self.get_logger().debug(f"Saving pose at {pose_output_path}")
                        with open(pose_output_path, "w") as pose_file:
                            T = transform_stamped.transform.translation
                            R = transform_stamped.transform.rotation
                            pose_file.write(f"{T.x} {T.y} {T.z} {R.x} {R.y} {R.z} {R.w}")

                        self.get_logger().debug(f"Saved synchronized pose to {pose_output_path}")

                    except Exception as e:
                        self.get_logger().warn(
                            f"TF Lookup Error for synchronized pose at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}: {e}"
                        )
                        continue
                if latest_msgs[self.robosense_lidar_topic]:
                    msg = latest_msgs[self.robosense_lidar_topic]
                    # latest_msgs[self.robosense_lidar_topic] = None  # Clear after use
                    assert isinstance(msg, PointCloud2)
                    # Convert the message into numpy array
                    pc2_np = self.convert_pc2_to_numpy(msg)  # (N, 3)
                    pc_path = os.path.join(self.rslidar_dir, f"rslidar_{sync_ts_str}.npy")
                    np.save(pc_path, pc2_np)

            pbar.update(msg_time - self.start_time_sec - pbar.n)
            pbar.set_description(f"Saving data for time: {msg_time}s")

            message_count += 1
            if message_count % 1000 == 0:
                self.get_logger().info(f"Processed {message_count} messages...")

        self.get_logger().info(f"\nFinished reading. Total messages processed: {message_count}")


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
