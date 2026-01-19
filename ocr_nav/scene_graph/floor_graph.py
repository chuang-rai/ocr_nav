import json
import threading
from pathlib import Path
import numpy as np
import networkx as nx
from tqdm import tqdm
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi
from scipy.ndimage import binary_erosion, binary_closing
from ocr_nav.scene_graph.pose_graph import PoseGraph
from ocr_nav.utils.mapping_utils import (
    project_points,
    segment_floor,
    select_points_in_masks,
    select_points_near_heights,
    transform_point_cloud,
    to_o3d_pc,
    get_largest_region,
)
from ocr_nav.utils.io_utils import (
    load_livox_poses_timestamps,
    search_latest_poses_within_timestamp_range,
    FolderIO,
    SubscriberIO,
)
import rclpy
from typing import List, Tuple


class FloorGraph:
    """Class for building a traversable floor graph for a building incrementally
    with data recorded with Perception-Suite.  It assumes rgb image, segmentation
    mask of the ground for each image, Livox Lidar, RoboSense Lidar, pose
    of Livox Lidar, camera intrinsics, camera-Livox extrinsics, and
    camera-RoboSense extrinsics are available.

    Attributes:
    pose_graph: PoseGraph object storing the pose nodes and edges.
    cell_size: voxel size for downsampling the ground point cloud.
    voronoi_graphs: dict mapping from floor id to its Voronoi graph.
    """

    def __init__(self, voxel_size: float = 0.1):
        self.pose_graph = PoseGraph()
        self.cell_size = voxel_size
        self.voronoi_graphs = {}

    def build_floor_graph_with_folder(
        self,
        folderio: FolderIO,
        floor_seg_resolution: float = 0.05,
        obstacle_height_range: Tuple[float, float] = (0.6, 2.7),
        sample_rate: int = 10,
        vis: bool = False,
    ) -> None:

        # load intrinsics
        intr_mat = folderio.get_intrinsics()

        # load some static transforms
        # the naming convention: tf_source2target means tf that can transform
        # points from source frame to target frame
        # pc_target_frame_homo = tf_source2target @ pc_source_frame_homo
        tf_livox2zed_left = folderio.get_livox2left_camera_tf()
        tf_zed_left2livox = np.linalg.inv(tf_livox2zed_left)
        tf_rs2zed_left = folderio.get_rslidar2left_camera_tf()
        livox_poses_timestamps_path = folderio.root_dir / "glim_livox_poses_timestamps.npy"

        # load livox poses and timestamps of glim results
        livox_poses, livox_timestamps = load_livox_poses_timestamps(livox_poses_timestamps_path)

        self.full_ground_pc_livox = o3d.geometry.PointCloud()
        self.full_ground_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_livox = o3d.geometry.PointCloud()
        self.pose_list = []
        pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
        last_pi = -1
        # Go through the data stream to construct full point clouds of the ground from
        # different LiDAR source (Livox, RoboSense), and build the pose graph
        for pi, p in pbar:
            if pi % sample_rate != 0:
                continue

            pbar.set_description(f"Processing frame {pi}")
            frame_id = p

            # get image size
            if pi == 0:
                img = folderio.get_image(pi)
                h, w, _ = np.array(img).shape
                static_params = {
                    "tf_livox2zed_left": tf_livox2zed_left,
                    "tf_zed_left2livox": tf_zed_left2livox,
                    "tf_rs2zed_left": tf_rs2zed_left,
                    "intr_mat": intr_mat,
                    "w": w,
                    "h": h,
                }

            masks = folderio.get_mask(pi)
            pc_livox = folderio.get_livox(pi)  # (N, 3)
            pc_rslidar = folderio.get_rslidar(pi)  # (N, 3)

            # load livox poses computed with glim
            livox_pose, livox_timestamp = search_latest_poses_within_timestamp_range(
                livox_poses,
                livox_timestamps,
                frame_id,
            )
            if livox_pose is None:
                print(f"No livox pose found for frame {frame_id}, skipping...")
                continue

            mask_np = masks[0][0]

            self.accumulate_ground_points(
                pc_livox,
                pc_rslidar,
                mask_np,
                livox_pose,
                static_params,
                pi,
                last_pi,
            )

            last_pi = pi

        self.fused_floor_graph, self.full_ground_pc = self.build_floor_graph_with_ground_points(
            floor_seg_resolution, obstacle_height_range, vis
        )
        return self.fused_floor_graph, self.full_ground_pc

    def build_floor_graph_with_rosio(
        self,
        subscriber_io: SubscriberIO,
        floor_seg_resolution: float = 0.05,
        obstacle_height_range: Tuple[float, float] = (0.6, 2.7),
        sample_rate: int = 10,
        vis: bool = False,
    ):

        one_time_read_flag = False
        self.full_ground_pc_livox = o3d.geometry.PointCloud()
        self.full_ground_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_livox = o3d.geometry.PointCloud()
        self.pose_list = []
        pi = 0

        try:
            while rclpy.ok():
                if not one_time_read_flag:
                    tf_livox2zed_left = subscriber_io.get_livox2left_camera_tf()
                    tf_rs2zed_left = subscriber_io.get_rslidar2left_camera_tf()
                    if tf_livox2zed_left is None or tf_rs2zed_left is None:
                        print("Waiting for static transforms...")
                        continue
                    tf_zed_left2livox = np.linalg.inv(tf_livox2zed_left)
                    intr_mat = subscriber_io.get_intrinsics()
                    h, w = subscriber_io.get_image_size()
                    static_params = {
                        "tf_livox2zed_left": tf_livox2zed_left,
                        "tf_zed_left2livox": tf_zed_left2livox,
                        "tf_rs2zed_left": tf_rs2zed_left,
                        "intr_mat": intr_mat,
                        "w": w,
                        "h": h,
                    }
                    one_time_read_flag = True

                data = subscriber_io.get_latest_sync_data()
                if data is None:
                    continue
                pc_livox, pc_rslidar, mask_np, livox_pose = data

                self.accumulate_ground_points(
                    pc_livox,
                    pc_rslidar,
                    mask_np,
                    livox_pose,
                    static_params,
                    pi,
                    last_pi,
                )

                last_pi = pi
                pi += 1

        except KeyboardInterrupt:
            pass

        self.fused_floor_graph, self.full_ground_pc = self.build_floor_graph_with_ground_points(
            floor_seg_resolution, obstacle_height_range, vis
        )
        return self.fused_floor_graph, self.full_ground_pc

    def accumulate_ground_points(
        self,
        pc_livox,
        pc_rslidar,
        mask_np,
        livox_pose,
        static_params,
        pi,
        last_pi,
    ):
        tf_livox2zed_left = static_params["tf_livox2zed_left"]
        tf_zed_left2livox = static_params["tf_zed_left2livox"]
        tf_rs2zed_left = static_params["tf_rs2zed_left"]
        intr_mat = static_params["intr_mat"]
        w = static_params["w"]
        h = static_params["h"]

        self.pose_list.append(livox_pose)

        # transform point cloud from robosense/livox frames to zed left camera frame
        pc_livox_in_zed_frame = transform_point_cloud(pc_livox, tf_livox2zed_left)  # (N, 3)
        pc_rslidar_in_zed_frame = transform_point_cloud(pc_rslidar, tf_rs2zed_left)  # (N, 3)

        # project points to image plane
        pc_livox_image_2d, pc_livox_image_2d_depth, filtered_pc_livox_in_zed_frame = project_points(
            pc_livox_in_zed_frame, intr_mat, w, h
        )  # (N, 2), (N,), (N, 3)
        pc_rs_image_2d, pc_rs_image_2d_depth, filtered_pc_rslidar_in_zed_frame = project_points(
            pc_rslidar_in_zed_frame, intr_mat, w, h
        )  # (N, 2), (N,), (N, 3)

        # Select Livox ground points using segmentation masks
        ground_mask = mask_np == 1
        ground_pc_livox_o3d = select_points_in_masks(ground_mask, pc_livox_image_2d, filtered_pc_livox_in_zed_frame)
        ground_pc_livox_o3d = ground_pc_livox_o3d.transform(livox_pose @ tf_zed_left2livox)
        self.full_ground_pc_livox += ground_pc_livox_o3d

        # Select RoboSense ground points using segmentation masks
        ground_pc_rs_o3d = select_points_in_masks(ground_mask, pc_rs_image_2d, filtered_pc_rslidar_in_zed_frame)
        ground_pc_rs_o3d = ground_pc_rs_o3d.transform(livox_pose @ tf_zed_left2livox)
        self.full_ground_pc_rs += ground_pc_rs_o3d

        # Accumulate full Livox point clouds
        pc_livox_world_np = transform_point_cloud(filtered_pc_livox_in_zed_frame, livox_pose @ tf_zed_left2livox)
        pc_livox_world_o3d = to_o3d_pc(pc_livox_world_np)
        self.full_pc_livox += pc_livox_world_o3d

        # Accumulate full RoboSense point clouds
        pc_rs_world_np = transform_point_cloud(filtered_pc_rslidar_in_zed_frame, livox_pose @ tf_zed_left2livox)
        pc_rs_world_o3d = to_o3d_pc(pc_rs_world_np)
        self.full_pc_rs += pc_rs_world_o3d
        # Add pose node to the pose graph
        pose_id = self.pose_graph.add_pose(pi, livox_pose, lidar=pc_rs_world_o3d)
        if last_pi != -1 and self.pose_graph.G.has_node(last_pi):
            self.pose_graph.add_pose_edge(last_pi, pi)

    def build_floor_graph_with_ground_points(
        self,
        floor_seg_resolution: float = 0.1,
        obstacle_height_range: Tuple[float, float] = (0.6, 2.7),
        vis: bool = False,
    ):
        heights = segment_floor(np.array(self.full_ground_pc_livox.points), resolution=floor_seg_resolution, vis=vis)
        full_ground_pc_numpy_list = select_points_near_heights(
            np.array((self.full_ground_pc_livox + self.full_ground_pc_rs).points), heights, threshold=0.2
        )
        full_ground_pc_np = np.vstack(full_ground_pc_numpy_list)  # (N, 3)
        self.full_ground_pc = to_o3d_pc(full_ground_pc_np)
        self.full_ground_pc = self.full_ground_pc.voxel_down_sample(voxel_size=self.cell_size)

        fused_floor_graph, full_ground_pc_np = self.build_connected_floor_voronois(
            self.full_pc_livox,
            self.full_ground_pc,
            np.array(heights),
            voxel_res=self.cell_size,
            high_threshold=obstacle_height_range[1],
            low_threshold=obstacle_height_range[0],
            vis=vis,
        )
        self.full_ground_pc = to_o3d_pc(full_ground_pc_np)
        return fused_floor_graph, self.full_ground_pc

    @staticmethod
    def save_floor_graph(graph: nx.Graph, graph_path: Path) -> None:
        """Save the floor graph to a json file.

        Args:
            graph (nx.Graph): The floor graph.
            graph_path (Path): The path where the graph will be saved.
        """
        graph_json = nx.node_link_data(graph)
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=4)

    @staticmethod
    def load_floor_graph(graph_path: Path) -> nx.Graph:
        """Load the floor graph saved as a json file.

        Args:
            graph_path (Path): The path to the graph json file.

        Returns:
            nx.Graph: The loaded floor graph.
        """
        with open(graph_path, "r") as f:
            graph_json = json.load(f)
        graph = nx.node_link_graph(graph_json)
        return graph

    def compute_grid_parameters(self, full_pc: o3d.geometry.PointCloud, voxel_res: float = 0.05) -> None:
        """Given a point cloud of the whole scene, compute the voxel grid
        boundaries as well as the assignment of points to grid id.

        Args:
            full_pc (o3d.geometry.PointCloud): Full point cloud of the scene.
            voxel_res (float, optional): Voxel resolution. Defaults to 0.05.
        """
        self.min_bound, self.max_bound = full_pc.get_min_bound(), full_pc.get_max_bound()
        self.in_bound = self.min_bound + np.array([0, 0, -1.0])
        self.full_voxel_grid = np.round((np.array(full_pc.points) - self.min_bound) / voxel_res)
        self.x_range, self.y_range, self.z_range = (
            np.max(self.full_voxel_grid[:, 0]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 1]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 2]).astype(int) + 1,
        )

    def build_connected_floor_voronois(
        self,
        full_pc: o3d.geometry.PointCloud,
        ground_pc: o3d.geometry.PointCloud,
        heights: np.ndarray,
        voxel_res: float = 0.05,
        high_threshold: float = 1.7,
        low_threshold: float = 0.2,
        close_kernel_size: int = 5,
        vis: bool = False,
    ) -> Tuple[nx.Graph, o3d.geometry.PointCloud]:
        """build a Voronoi graph for each floor and connect them with trajectories on stairs.
        The method assumes the z coordinate of the point clouds is aligned with the gravity direction.
        An example of parameters:

        z-axis
        |                  |
        |                  |
        -----Floor i+1------------ -> heights[i+1]
        |                  |
        |------------------|----------------------------
        |                  |  |                        |
        |                  |  |                        |
        |                  |  | ----> high_threshold   | -> Obstacle pc range
        |                  |  |                        |
        |                  |  |                        |
        |---------------------|-------------------------
        |                  |  |  | -> low_threshold    |
        |                  |  |  |                     |
        ------Floor i------------- -> heights[i]       | -> Ground pc range
        |                  |                           |
        |------------------|---------------------------- -> heights[i-1] + high_threshold
        |                  |

        Args:
            full_pc (o3d.geometry.PointCloud): Full point cloud of the scene.
            ground_pc (o3d.geometry.PointCloud): Ground point cloud of the scene.
            heights (np.ndarray): Heights of the floors.
            voxel_res (float, optional): Voxel resolution. Defaults to 0.05.
            high_threshold (float, optional): High threshold for obstacle height. Defaults to 1.7.
            low_threshold (float, optional): Low threshold for obstacle height. Defaults to 0.2.
            close_kernel_size (int, optional): Kernel size for morphological closing. Defaults to 5.
            vis (bool, optional): Flag to visualize the process. Defaults to False.

        Returns:
            Tuple[nx.Graph, o3d.geometry.PointCloud]: A tuple containing the fused Voronoi graph
            and the filtered ground point cloud.
        """
        self.compute_grid_parameters(full_pc, voxel_res=voxel_res)
        obstacle_height_ranges = zip(heights + low_threshold, heights + high_threshold)
        ground_voxel_grid = np.round((np.array(ground_pc.points) - self.min_bound) / voxel_res)

        traversable_ground_voxel_grid = []
        self.stairs_pc_list = []
        self.stairs_graph_list = []
        last_high = 0
        for floor_i, obstacle_height_range in enumerate(obstacle_height_ranges):
            low, high = np.floor((np.array(obstacle_height_range) - self.min_bound[2]) / voxel_res).astype(int)
            obstacle_mask = (self.full_voxel_grid[:, 2] >= low) & (self.full_voxel_grid[:, 2] < high)
            ground_mask = (ground_voxel_grid[:, 2] >= last_high) & (ground_voxel_grid[:, 2] < low)
            floor_obstacle_voxels = self.full_voxel_grid[obstacle_mask, :]
            floor_ground_voxels = ground_voxel_grid[ground_mask, :]

            # BEV binary maps for obstacle and ground
            floor_obstacle_map = np.zeros((self.x_range, self.y_range), dtype=bool)
            floor_obstacle_map[floor_obstacle_voxels[:, 0].astype(int), floor_obstacle_voxels[:, 1].astype(int)] = True
            floor_ground_map = np.zeros((self.x_range, self.y_range), dtype=bool)
            floor_ground_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)] = True

            if vis:
                cv2.imshow("floor_obstacle_map", floor_obstacle_map.astype(np.uint8) * 255)
                cv2.imshow("floor_ground_map", floor_ground_map.astype(np.uint8) * 255)
                cv2.waitKey()

            # project poses onto a BEV and treat their vicinity as traversable
            pose_map_floor = self.get_pose_map_on_floor(
                (heights[floor_i], obstacle_height_range[1]),
                self.min_bound,
                voxel_res=voxel_res,
                camera_height_range=0.2,
                radius=1,
                vis=vis,
            )

            # traversable BEV = ground BEV + pose BEV - obstacle BEV
            traversable_map = (floor_ground_map.astype(bool) | pose_map_floor) & (~floor_obstacle_map.astype(bool))
            traversable_map = binary_closing(
                traversable_map, structure=np.ones((close_kernel_size, close_kernel_size)), iterations=1
            )

            traversable_map = get_largest_region(traversable_map)
            if vis:
                cv2.imshow("traversable_map", traversable_map.astype(np.uint8) * 255)
                cv2.waitKey()

            height_map = np.ones_like(traversable_map, dtype=np.float32) * (heights[floor_i])
            voronoi_graph = self.get_voronoi_graph(
                traversable_map, floor_id=str(floor_i), height_map=height_map, vis=vis
            )
            sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
            self.voronoi_graphs[floor_i] = sparse_voronoi_graph

            traversable_x, traversable_y = np.where(traversable_map)
            traversable_ground_voxels = np.concatenate(
                [
                    traversable_x.reshape((-1, 1)),
                    traversable_y.reshape((-1, 1)),
                    np.ones((len(traversable_x), 1)) * (heights[floor_i] - self.min_bound[2]) / voxel_res,
                ],
                axis=1,
            )  # (N, 3)
            traversable_ground_voxel_grid.append(traversable_ground_voxels)
            last_high = high

            # build stairs graph between this floor and the next floor
            if floor_i + 1 < len(heights):
                stairs_graph, stair_pos_list = self.get_stairs_graph(floor_i, margin=0.5, height_offset=1.0, vis=vis)
                self.stairs_pc_list.append(stair_pos_list)
                self.stairs_graph_list.append(stairs_graph)

        # connect floor voronois with the stairs graph
        self.fused_graph = self.voronoi_graphs[0]
        for floor_i, stair_graph in enumerate(self.stairs_graph_list):
            self.fused_graph = self.connect_voronoi_graphs(self.stairs_graph_list[floor_i], self.fused_graph)
            self.fused_graph = self.connect_voronoi_graphs(self.fused_graph, self.voronoi_graphs[floor_i + 1])

        traversable_ground_voxel_grid = np.vstack(traversable_ground_voxel_grid)
        traversable_ground_points = traversable_ground_voxel_grid * voxel_res + self.min_bound
        return self.fused_graph, traversable_ground_points

    def get_pose_map_on_floor(
        self,
        floor_height_range: Tuple[float, float],
        min_bound: np.ndarray,
        voxel_res: float = 0.1,
        camera_height_range: float = 0.3,
        radius: float = 0.5,
        vis: bool = False,
    ) -> np.ndarray:
        """Generate a binary BEV map where projected pose locations and their
        vicinity within a radius are 1. Otherwise pixel values are 0.

        Args:
            floor_height_range (Tuple[float, float]): Height range for
                selecting poses on the floor.
            min_bound (np.ndarray): Minimum bound of the grid.
            voxel_res (float, optional): Voxel resolution. Defaults to 0.1.
            camera_height_range (float, optional): Height range for filtering
                poses around the major height. Defaults to 0.3.
            radius (float, optional): Radius around each pose to mark on the
                map. Defaults to 0.5.
            vis (bool, optional): Whether to visualize the pose map. Defaults
                to False.
        Returns:
            np.ndarray: Binary BEV map with pose locations marked.
        """
        poses_list_on_floor = [
            pose
            for pose in self.pose_list
            if (pose[2, 3] >= floor_height_range[0]) and (pose[2, 3] <= floor_height_range[1])
        ]
        pose_heights = np.array([pose[2, 3] for pose in poses_list_on_floor])
        clusters = DBSCAN(eps=0.2).fit(pose_heights.reshape(-1, 1))
        labels, counts = np.unique(clusters.labels_, return_counts=True)
        id = np.argmax(counts)
        mask = clusters.labels_ == labels[id]
        major_height = np.mean(pose_heights[mask])
        filtered_poses_list = [
            pose for pose in poses_list_on_floor if np.abs(pose[2, 3] - major_height) < camera_height_range
        ]

        assert hasattr(self, "x_range") and hasattr(self, "y_range"), "Grid parameters not computed yet."
        assert hasattr(self, "min_bound"), "Grid parameters not computed yet."

        pose_map = np.zeros((self.x_range, self.y_range), dtype=np.uint8)
        for filtered_pose in filtered_poses_list:
            voxel_x = int((filtered_pose[0, 3] - min_bound[0]) / voxel_res)
            voxel_y = int((filtered_pose[1, 3] - min_bound[1]) / voxel_res)
            # mark the position on the grid
            pose_map = cv2.circle(pose_map, (voxel_y, voxel_x), radius=int(radius / voxel_res), color=255, thickness=-1)
        if vis:
            cv2.imshow("pose_map", pose_map)
            cv2.waitKey()
        return pose_map

    def get_stairs_graph(
        self, floor_id: int, margin: float = 0.1, height_offset: float = 1.0, vis: bool = False
    ) -> Tuple[nx.Graph, np.ndarray]:
        """Generate a graph representing the exploration trajectory on the stairs
        between two floors.

        Args:
            floor_id (int): The floor id. For example, 0 for the first floor.
            margin (float, optional): A range of height for selecting poses.
                Defaults to 0.1.
            height_offset (float, optional): Height offset for adjusting the
                z-coordinate of poses in the stairs graph. Defaults to 1.0.
            vis (bool, optional): Whether to visualize the stairs graph.
                Defaults to False.

        Returns:
            Tuple[nx.Graph, np.ndarray]: A tuple containing the stairs graph
                and the list of stair positions.
        """

        pose_heights = np.array([pose[2, 3] for pose in self.pose_list])
        frequency, bin_edges = np.histogram(pose_heights, bins=100)  # len(bin_edges) = len(frequency) + 1
        min_peak_height = 0.3 * np.max(frequency)
        peaks, _ = find_peaks(
            frequency, distance=20, height=min_peak_height
        )  # return the indices of the peak position (x direction)

        if vis:
            plt.figure()
            plt.plot(bin_edges[:-1], frequency)
            plt.plot(bin_edges[peaks], frequency[peaks], "x")
            plt.show()

        low = bin_edges[peaks[floor_id]] + margin
        high = bin_edges[peaks[floor_id + 1]] - margin

        stair_pose_list = [pose for pose in self.pose_list if (pose[2, 3] >= low) and (pose[2, 3] < high)]
        stair_pos_list = np.concatenate([pose[:3, 3].reshape((1, 3)) for pose in stair_pose_list], axis=0) - np.array(
            [0, 0, height_offset]
        )
        stairs_graph = nx.Graph()
        last_node = None
        last_pos = None
        for i, pos in enumerate(stair_pos_list):
            stairs_graph.add_node((pos[0], pos[1], str(floor_id)), pos=(pos[0], pos[1], pos[2]), floor_id=floor_id)
            if i > 0:
                stairs_graph.add_edge(
                    last_node,
                    (pos[0], pos[1], str(floor_id)),
                    dist=np.linalg.norm(pos - last_pos),
                )
            last_node = (pos[0], pos[1], str(floor_id))
            last_pos = pos

        return stairs_graph, stair_pos_list

    def get_voronoi_graph(
        self,
        main_free_map: np.ndarray,
        floor_id: str,
        height_map: np.ndarray = None,
        vis: bool = False,
    ) -> nx.Graph:
        """Generate the Voronoi Graph of the floor based on the free space map substracting obstacle map.

        Args:
            main_free_map (np.ndarray): Free space map.
            floor_id (str): The floor id. For example, "0" for the first floor.
            height_map (np.ndarray, optional): Height map used for getting the 3D positional attributes of
                the Voronoi node. Defaults to None.

        Returns:
            nx.Graph: Resulting Voronoi graph.
        """
        boundary_map = binary_erosion(main_free_map, iterations=1).astype(np.uint8)
        boundary_map = main_free_map - boundary_map
        rows, cols = np.where(boundary_map == 1)
        boundaries = np.array(list(zip(rows, cols)))
        voronoi = Voronoi(boundaries)

        fig_free = main_free_map.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)
        vertices = []
        if height_map is None:
            height_map = np.ones_like(boundary_map) * self.min_bound[2]
        voronoi_graph = nx.Graph()
        for simplex in voronoi.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                continue
            src, tar = voronoi.vertices[simplex]
            if (
                src[0] < 0
                or src[0] >= fig_free.shape[0]
                or src[1] < 0
                or src[1] >= fig_free.shape[1]
                or tar[0] < 0
                or tar[0] >= fig_free.shape[0]
                or tar[1] < 0
                or tar[1] >= fig_free.shape[1]
            ):
                continue
            if main_free_map[int(src[0]), int(src[1])] == 0 or main_free_map[int(tar[0]), int(tar[1])] == 0:
                continue
            cv2.line(
                fig_free,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.circle(fig_free, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)

            # check if src and tar already exist in the graph
            if (src[0], src[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(src[0]), int(src[1])]
                voronoi_graph.add_node(
                    (src[0], src[1], floor_id),
                    pos=(
                        src[0] * self.cell_size + self.min_bound[0],
                        src[1] * self.cell_size + self.min_bound[1],
                        height,
                    ),
                    floor_id=floor_id,
                )
            if (tar[0], tar[1], floor_id) not in voronoi_graph.nodes:
                height = height_map[int(tar[0]), int(tar[1])]
                voronoi_graph.add_node(
                    (tar[0], tar[1], floor_id),
                    pos=(
                        tar[0] * self.cell_size + self.min_bound[0],
                        tar[1] * self.cell_size + self.min_bound[1],
                        height,
                    ),
                    floor_id=floor_id,
                )
            # check if the edge already exists
            if (src[0], src[1], floor_id) not in voronoi_graph[(tar[0], tar[1], floor_id)]:
                voronoi_graph.add_edge(
                    (src[0], src[1], floor_id),
                    (tar[0], tar[1], floor_id),
                    dist=np.linalg.norm(src - tar),
                )
        if vis:
            cv2.imshow("voronoi_free", fig_free)
            cv2.waitKey()
        return voronoi_graph

    def sparsify_graph(self, floor_graph: nx.Graph, resampling_dist: float = 0.4) -> nx.Graph:
        """Sparsify the graph by removing nodes within a certain distance
        using chain traversal instead of all-pairs shortest paths.

        Args:
            floor_graph (nx.Graph): The input floor graph to be sparsified.
            resampling_dist (float, optional): The distance threshold for resampling
                nodes. Defaults to 0.4.

        Returns:
            nx.Graph: The sparsified graph.
        """
        if len(floor_graph.nodes) < 10:
            return floor_graph.copy()

        # 1. Identify "Junction" nodes (endpoints and intersections)
        junction_nodes = [n for n in floor_graph.nodes if floor_graph.degree(n) != 2]

        new_graph = nx.Graph()
        # Pre-add junctions to the new graph
        for n in junction_nodes:
            new_graph.add_node(n, **floor_graph.nodes[n])

        visited_edges = set()

        # 2. Traverse from each junction to find paths/chains
        for start_node in junction_nodes:
            for neighbor in floor_graph.neighbors(start_node):
                edge = tuple(sorted((start_node, neighbor)))
                if edge in visited_edges:
                    continue

                # Start tracing the path
                path = [start_node, neighbor]
                visited_edges.add(edge)

                curr = neighbor
                # Follow the chain of degree-2 nodes
                while floor_graph.degree(curr) == 2:
                    # Find the next node in the chain that isn't the one we just came from
                    next_node = [n for n in floor_graph.neighbors(curr) if n != path[-2]][0]
                    path.append(next_node)
                    visited_edges.add(tuple(sorted((curr, next_node))))
                    curr = next_node
                    if curr in junction_nodes:
                        break

                # 3. Apply Resampling Logic on the discovered path
                self._add_resampled_path(new_graph, floor_graph, path, resampling_dist)

        return new_graph

    def _add_resampled_path(self, new_graph: nx.Graph, original_graph: nx.Graph, path: List, resampling_dist: float):
        """
        Helper to subdivide a long chain of nodes based on resampling_dist.
        """
        predecessor = path[0]
        accumulated_dist = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Get edge weight (default to Euclidean if 'dist' key is missing)
            edge_data = original_graph.edges[u, v]
            d = edge_data.get(
                "dist",
                np.linalg.norm(np.array(original_graph.nodes[u]["pos"]) - np.array(original_graph.nodes[v]["pos"])),
            )

            accumulated_dist += d

            # If we exceed resampling distance, create a node in the new graph
            # Note: self.cell_size included as per your original logic
            if (accumulated_dist * self.cell_size) > resampling_dist:
                if v not in new_graph:
                    new_graph.add_node(v, **original_graph.nodes[v])

                dist_val = np.linalg.norm(
                    np.array(new_graph.nodes[predecessor]["pos"]) - np.array(new_graph.nodes[v]["pos"])
                )
                new_graph.add_edge(predecessor, v, dist=dist_val)

                predecessor = v
                accumulated_dist = 0

        # Always connect the final segment to the last junction
        last_node = path[-1]
        if last_node != predecessor:
            if last_node not in new_graph:
                new_graph.add_node(last_node, **original_graph.nodes[last_node])

            dist_val = np.linalg.norm(
                np.array(new_graph.nodes[predecessor]["pos"]) - np.array(new_graph.nodes[last_node]["pos"])
            )
            new_graph.add_edge(predecessor, last_node, dist=dist_val)

    def connect_voronoi_graphs(self, src_graph: nx.Graph, tar_graph: nx.Graph) -> nx.Graph:
        """Connect two graphs by finding the closest node in the source graph to the target graph.

        Args:
            src_graph (nx.Graph): The source graph.
            tar_graph (nx.Graph): The target graph.

        Returns:
            tar_graph (nx.Graph): The resulting graph.
        """
        # find closest node from src_graph to tar_graph
        floor_nodes = [
            (i, tar_graph.nodes[node]["pos"]) for i, node in enumerate(tar_graph.nodes) if tar_graph.degree(node) > 1
        ]
        floor_node_ids = [node[0] for node in floor_nodes]
        floor_nodes = np.array([node[1] for node in floor_nodes])
        stairs_nodes = np.array(
            [src_graph.nodes[node]["pos"] for node in src_graph.nodes]  # if src_graph.degree(node) == 1]
        )

        dist_mat = cdist(stairs_nodes, floor_nodes)
        row, col = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
        col = floor_node_ids[col]
        stair_node = list(src_graph.nodes)[row]
        floor_node = list(tar_graph.nodes)[col]

        tar_graph = nx.compose(tar_graph, src_graph)
        tar_graph.add_edge(
            stair_node,
            floor_node,
            dist=np.linalg.norm(np.array(stair_node[:2]) - np.array(floor_node[:2])),
        )
        return tar_graph
