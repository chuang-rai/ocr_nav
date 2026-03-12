import json
import os
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing

from ocr_nav.scene_graph.pose_graph import PoseGraph
from ocr_nav.utils.floor_graph_utils import (
    compute_waypoint_orientations,
    connect_graphs,
    get_delaunay_graph,
    get_free_space_map,
    get_obstacle_map,
    get_pose_map_on_floor,
    get_stairs_graph,
    get_voronoi_graph,
    plan_global_path,
    sparsify_graph,
)
from ocr_nav.utils.io_utils import (
    BagIO,
    waypoints_to_yaml,
)
from ocr_nav.utils.mapping_utils import (
    get_largest_region,
    project_points,
    segment_floor,
    select_points_in_masks,
    select_points_near_heights,
    to_o3d_pc,
    transform_point_cloud,
)
from ocr_nav.utils.segmentation_utils import GroundingDinoSamSegmenter


class FloorGraph:
    """Class for building a traversable floor graph for a building incrementally
    with data recorded with Perception-Suite.  It assumes rgb image, segmentation
    mask of the ground for each image, Livox Lidar, RoboSense Lidar, pose
    of Livox Lidar, camera intrinsics, camera-Livox extrinsics, and
    camera-RoboSense extrinsics are available.

    Usage:
    1. Call build_floor_graph_with_*() methods to build the floor graph. * can be
    'folder', 'bagio' or 'rosio' depending on the data source (currently only bagio).
    The outcome of the method is a floor graph stored in self.floor_graph, where
    each node represents a traversable waypoint whose position is in "pos" attribute,
    and each edge represents a traversable path between two waypoints with their
    distance in "dist" attribute.

    2. Call save_floor_graph() to save the floor graph to a json file.

    3. Call load_floor_graph() to load a saved floor graph from a json file.

    4. Call plan_global_path() to plan a global path between two 3D points on the
    floor graph.

    Attributes:
    pose_graph: PoseGraph object storing the pose nodes and edges.
    cell_size: voxel size for downsampling the ground point cloud.
    floor_graphs: dict mapping from floor id to its floor graph.
    """

    def __init__(self):
        self.pose_graph = PoseGraph()
        self.floor_graphs = {}

    def build_floor_graph_with_bagio(
        self,
        bagio: BagIO,
        segmenter: GroundingDinoSamSegmenter,
        floor_seg_resolution: float = 0.1,
        obstacle_height_range: tuple[float, float] = (0.6, 2.7),
        cache_segmentation: bool = True,
        cache_dir: Path = Path("/control_suite/temp/segmentation_cache"),
        min_floor_points_num: int = 0,
        relative_percentile: float = 90,
        use_relative: bool = True,
        cell_size: float = 0.1,
        pose_height_margin: float = 0.2,
        pose_radius_as_traversable: float = 1,
        floor_graph_type: str = "voronoi",
        delaunay_distance_threshold: float = 0.5,
        max_sync_latency_sec: float = 0.2,
        vis: bool = False,
    ) -> None:
        if cache_segmentation:
            bag_name = Path(bagio.bag_path).stem
            cache_dir = cache_dir / bag_name
            os.makedirs(cache_dir, exist_ok=True)
        one_time_read_flag = False
        self.full_ground_pc_list = [o3d.geometry.PointCloud() for _ in range(len(bagio.lidar_frame_ids))]
        self.full_pc_list = [o3d.geometry.PointCloud() for _ in range(len(bagio.lidar_frame_ids))]
        self.pose_list = []
        pi = 0
        last_pi = -1

        while bagio.has_next():
            if not one_time_read_flag:
                lidar2camera_tfs = bagio.get_lidar2camera_tfs()
                if any([tf is None for tf in lidar2camera_tfs]):
                    print("Waiting for static transforms...")
                    continue
                tf_camera2anchor_lidar = np.linalg.inv(lidar2camera_tfs[bagio.anchor_lidar_id])
                intr_mat = bagio.get_intrinsics()
                h, w = bagio.get_image_size()
                static_params = {
                    "lidar2camera_extrinsics_list": lidar2camera_tfs,
                    "tf_camera2anchor_lidar": tf_camera2anchor_lidar,
                    "intr_mat": intr_mat,
                    "w": w,
                    "h": h,
                }
                one_time_read_flag = True

            data = bagio.get_next_sync_data(max_latency_s=max_sync_latency_sec)
            if data is None:
                continue
            lidar_pc_list, img_np, anchor_lidar_pose, t_nanosec = data

            if cache_segmentation:
                mask_path = cache_dir / f"mask_{t_nanosec}.npy"
                if not mask_path.exists():
                    mask_np = segmenter.segment(img_np, text_prompt="ground")
                    np.save(mask_path, mask_np)
                else:
                    mask_np = np.load(mask_path)
            else:
                mask_np = segmenter.segment(img_np, text_prompt="ground")

            self._accumulate_ground_points(
                lidar_pc_list,
                bagio.anchor_lidar_id,
                mask_np,
                anchor_lidar_pose,
                static_params,
                pi,
                last_pi,
            )

            last_pi = pi
            pi += 1

        self.floor_graph, self.full_ground_pc = self._build_floor_graph_with_ground_points(
            anchor_lidar_id=bagio.anchor_lidar_id,
            floor_seg_resolution=floor_seg_resolution,
            obstacle_height_range=obstacle_height_range,
            min_floor_points_num=min_floor_points_num,
            relative_percentile=relative_percentile,
            use_relative=use_relative,
            considered_lidar_ids=[bagio.anchor_lidar_id],
            cell_size=cell_size,
            pose_height_margin=pose_height_margin,
            pose_radius_as_traversable=pose_radius_as_traversable,
            floor_graph_type=floor_graph_type,
            delaunay_distance_threshold=delaunay_distance_threshold,
            vis=vis,
        )
        return self.floor_graph, self.full_ground_pc

    def _accumulate_ground_points(
        self,
        pc_lidar_list: list[np.ndarray],
        anchor_lidar_id: int,
        mask_np: np.ndarray,
        anchor_lidar_pose: np.ndarray,
        static_params: dict,
        pi: int,
        last_pi: int,
    ) -> None:
        """Project multiple LiDARs' pionts into the image plane, select ground points using segmentation masks,
        and accumulate the ground points and full point clouds in the world frame. This method
        updates self.full_ground_pc_list, self.full_ground_pc_list, self.pose_list.
        At the same time, it also adds pose nodes and edges to the pose graph (self.pose_graph).

        Args:
            pc_lidar_list (list[np.ndarray]): List of (N, 3) arrays of LiDAR points.
            anchor_lidar_id (int): Index of the anchor LiDAR in the pc_lidar_list.
            mask_np (np.ndarray): (H, W) segmentation mask array of the camera. 1 indicates ground. Otherwise 0.
            anchor_lidar_pose (np.ndarray): (4, 4) transformation matrix representing the pose of the anchor
                LiDAR sensor in the world frame.
            static_params (Dict): Dictionary containing static transformation matrices and camera intrinsics.
                static_params = {
                    "lidar2camera_extrinsics_list": [lidar2camera_extrinsics_1 (4x4),
                                                       lidar2camera_extrinsics_2 (4x4),
                                                       ...
                                                      ],
                    "tf_camera2anchor_lidar": tf_camera2anchor_lidar (4x4),
                    "intr_mat": intr_mat,
                    "w": w,
                    "h": h,
                }
            pi (int): Current pose index.
            last_pi (int): Previous pose index.
        """
        tf_camera2anchor_lidar = static_params["tf_camera2anchor_lidar"]
        intr_mat = static_params["intr_mat"]
        w = static_params["w"]
        h = static_params["h"]

        assert hasattr(self, "full_ground_pc_list"), "full_ground_pc_list not initialized."
        assert hasattr(self, "full_pc_list"), "full_pc_list not initialized."
        assert hasattr(self, "pose_list"), "pose_list not initialized."

        assert (
            len(pc_lidar_list) == len(self.full_ground_pc_list) == len(self.full_pc_list)
        ), "LiDAR lists length mismatch."
        assert isinstance(
            self.full_ground_pc_list[0], o3d.geometry.PointCloud
        ), "full_ground_pc_list not initialized properly."
        assert isinstance(self.full_pc_list[0], o3d.geometry.PointCloud), "full_pc_list not initialized properly."

        self.pose_list.append(anchor_lidar_pose)

        ground_mask = mask_np == 1
        # transform point cloud from robosense/livox frames to camera frame
        for i, pc_lidar in enumerate(pc_lidar_list):
            if i == anchor_lidar_id:
                self.pose_graph.add_pose(pi, anchor_lidar_pose, lidar=pc_lidar)
                if last_pi != -1 and self.pose_graph.G.has_node(last_pi):
                    self.pose_graph.add_pose_edge(last_pi, pi)

            pc_lidar_in_camera_frame = transform_point_cloud(pc_lidar, static_params["lidar2camera_extrinsics_list"][i])

            pc_image_2d, pc_lidar_image_2d_depth, filtered_pc_lidar_in_camera_frame = project_points(
                pc_lidar_in_camera_frame, intr_mat, w, h
            )  # (N, 2), (N,), (N, 3)

            # accumulate ground points for each LiDAR
            ground_pc_lidar_o3d = select_points_in_masks(ground_mask, pc_image_2d, filtered_pc_lidar_in_camera_frame)
            ground_pc_lidar_o3d = ground_pc_lidar_o3d.transform(anchor_lidar_pose @ tf_camera2anchor_lidar)
            self.full_ground_pc_list[i] += ground_pc_lidar_o3d

            # accumulate full point clouds for each LiDAR
            pc_lidar_world_np = transform_point_cloud(
                filtered_pc_lidar_in_camera_frame, anchor_lidar_pose @ tf_camera2anchor_lidar
            )
            pc_lidar_world_o3d = to_o3d_pc(pc_lidar_world_np)
            self.full_pc_list[i] += pc_lidar_world_o3d

    def _build_floor_graph_with_ground_points(
        self,
        anchor_lidar_id: int,
        floor_seg_resolution: float = 0.1,
        obstacle_height_range: tuple[float, float] = (0.6, 2.7),
        min_floor_points_num: int = 0,
        relative_percentile: float = 90,
        use_relative: bool = True,
        floor_height_margin: float = 0.2,
        considered_lidar_ids: list[int] = [0],
        cell_size: float = 0.1,
        pose_height_margin: float = 0.2,
        pose_radius_as_traversable: float = 1,
        floor_graph_type: str = "voronoi",
        delaunay_distance_threshold: float = 0.5,
        vis: bool = False,
    ) -> tuple[nx.Graph, o3d.geometry.PointCloud]:
        """Build the multi-floor floor graph with ground point clouds.

        Args:
            floor_seg_resolution (float, optional): The height histogram resolution. Defaults to 0.1 meters.
            obstacle_height_range (tuple[float, float], optional): The range of obstacle heights to consider.
                Defaults to (0.6, 2.7) meters.
            min_floor_points_num (int, optional): Minimum number of points in a bin to consider a floor. Defaults to 0.
            relative_percentile (float, optional): The Peak of the histogram should be at least above this percentile
                of the top heights. Defaults to 90.
            use_relative (bool, optional): Whether to use relative thresholding. Defaults to True.
            delaunay_distance_threshold (float, optional): The distance threshold for connecting nodes in the
                Delaunay graph. Defaults to 0.5.
            vis (bool, optional): Whether to visualize the process. Defaults to False.

        Returns:
            tuple[nx.Graph, o3d.geometry.PointCloud]: A tuple containing the fused floor graph and the full
                ground point cloud. The fused floor graph contains Voronoi graphs of all floors connected by
                stair trajectories. Each node in the graph has an attribute "pos" indicating its 3D position.
                Each edge represents a traversable connection between nodes, containing a "weight" attribute
                indicating the Euclidean distance between the nodes.
        """
        assert hasattr(self, "full_ground_pc_list"), "full_ground_pc_list not initialized."
        assert hasattr(self, "full_pc_list"), "full_pc_list not initialized."
        floor_heights = segment_floor(
            np.array(self.full_ground_pc_list[anchor_lidar_id].points),
            resolution=floor_seg_resolution,
            min_floor_points_num=min_floor_points_num,
            relative_percentile=relative_percentile,
            use_relative=use_relative,
            vis=vis,
        )
        considered_ground_lidar = o3d.geometry.PointCloud()
        considered_full_lidar = o3d.geometry.PointCloud()
        for lidar_id in considered_lidar_ids:
            considered_ground_lidar += self.full_ground_pc_list[lidar_id]
            considered_full_lidar += self.full_pc_list[lidar_id]
        full_ground_pc_numpy_list = select_points_near_heights(
            np.array(considered_ground_lidar.points), floor_heights, threshold=floor_height_margin
        )
        full_ground_pc_np = np.vstack(full_ground_pc_numpy_list)  # (N, 3)
        self.full_ground_pc = to_o3d_pc(full_ground_pc_np)
        self.full_ground_pc = self.full_ground_pc.voxel_down_sample(voxel_size=cell_size)

        fused_floor_graph, full_ground_pc_np = self._build_connected_floor_graphs(
            considered_full_lidar,
            self.full_ground_pc,
            np.array(floor_heights),
            pose_height_margin=pose_height_margin,
            pose_radius_as_traversable=pose_radius_as_traversable,
            voxel_res=cell_size,
            high_threshold=obstacle_height_range[1],
            low_threshold=obstacle_height_range[0],
            floor_graph_type=floor_graph_type,
            delaunay_distance_threshold=delaunay_distance_threshold,
            vis=vis,
        )
        self.full_ground_pc = to_o3d_pc(full_ground_pc_np)
        return fused_floor_graph, self.full_ground_pc

    def save_floor_graph(self, graph_path: Path) -> None:
        """Save the floor graph to a json file.

        Args:
            graph (nx.Graph): The floor graph.
            graph_path (Path): The path where the graph will be saved.
        """
        assert hasattr(self, "floor_graph"), "Floor graph not built yet! Call"
        "build_floor_graph_with_*() methods first."
        graph_json = nx.node_link_data(self.floor_graph)
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=4)

    def load_floor_graph(self, graph_path: Path) -> nx.Graph:
        """Load the floor graph saved as a json file.

        Args:
            graph_path (Path): The path to the graph json file.

        Returns:
            nx.Graph: The loaded floor graph.
        """
        with open(graph_path, "r") as f:
            graph_json = json.load(f)
        self.floor_graph = nx.node_link_graph(graph_json)
        return self.floor_graph

    def compute_grid_parameters(self, full_pc: o3d.geometry.PointCloud, voxel_res: float = 0.05) -> None:
        """Given a point cloud of the whole scene, compute the voxel grid
        boundaries as well as the assignment of points to grid id.

        Args:
            full_pc (o3d.geometry.PointCloud): Full point cloud of the scene.
            voxel_res (float, optional): Voxel resolution. Defaults to 0.05.
        """
        self.min_bound, self.max_bound = full_pc.get_min_bound(), full_pc.get_max_bound()
        self.full_voxel_grid = np.round((np.array(full_pc.points) - self.min_bound) / voxel_res)
        self.x_range, self.y_range, self.z_range = (
            np.max(self.full_voxel_grid[:, 0]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 1]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 2]).astype(int) + 1,
        )
        self.cell_size = voxel_res

    def _build_connected_floor_graphs(
        self,
        full_pc: o3d.geometry.PointCloud,
        ground_pc: o3d.geometry.PointCloud,
        floor_heights: np.ndarray,
        floor_graph_type: str = "voronoi",
        pose_height_margin: float = 0.2,
        pose_radius_as_traversable: float = 1,
        voxel_res: float = 0.05,
        high_threshold: float = 1.7,
        low_threshold: float = 0.2,
        close_kernel_size: int = 5,
        graph_sparsify_dist: float = 0.4,
        delaunay_distance_threshold: float = 0.5,
        vis: bool = False,
    ) -> tuple[nx.Graph, o3d.geometry.PointCloud]:
        """build a graph for each floor and connect them with trajectories on stairs.
        The method assumes the z coordinate of the point clouds is aligned with the gravity direction.
        The method classifies obstacle points and ground points for each floor based on their heights.
        Given the height of a floor h_i, points with height in [h_{i-1} + high_threshold,
        h_i + low_threshold) are classified as floor points, and points with height in [h_i + low_threshold,
        h_i + high_threshold) are classified as obstacle points.  The traversable area on each floor is computed
        by subtracting the obstacle area from the union of the ground area and exploration pose area.
        A graph is built on the traversable area of each floor. Finally, the graphs of each floor
        are connected with the stairs graphs built from the exploration poses between two floors.

        An example of parameters:

        z-axis
        |                  |
        |                  |
        -----Floor i+1------------ -> floor_heights[i+1]
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
        ------Floor i------------- -> floor_heights[i] | -> Ground pc range
        |                  |                           |
        |------------------|---------------------------- -> floor_heights[i-1] + high_threshold
        |                  |

        Args:
            full_pc (o3d.geometry.PointCloud): Full point cloud of the scene.
            ground_pc (o3d.geometry.PointCloud): Ground point cloud of the scene.
            floor_heights (np.ndarray): Heights of the floors.
            floor_graph_type (str, optional): The type of floor graph to build ["voronoi", "delaunay"].
                Defaults to "voronoi".
            voxel_res (float, optional): Voxel resolution. Defaults to 0.05.
            high_threshold (float, optional): High threshold for obstacle height. Defaults to 1.7.
            low_threshold (float, optional): Low threshold for obstacle height. Defaults to 0.2.
            close_kernel_size (int, optional): Kernel size for morphological closing. Defaults to 5.
            graph_sparsify_dist (float, optional): Distance for sparsifying the Voronoi graph. Defaults to 0.4.
            vis (bool, optional): Flag to visualize the process. Defaults to False.

        Returns:
            Tuple[nx.Graph, o3d.geometry.PointCloud]: A tuple containing the floor graph
            and the traversable ground point cloud. Before this step, the ground point clouds are
            obtained by projecting lidar points into the ground segmentation mask in the image,
            which contains regions under tables etc. that are not traversable. Those untraversable
            points are removed from the output point cloud.
        """
        self.compute_grid_parameters(full_pc, voxel_res=voxel_res)
        obstacle_height_ranges = zip(floor_heights + low_threshold, floor_heights + high_threshold)

        traversable_ground_voxel_grid = []
        self.stairs_pc_lists: list[list[np.ndarray]] = []  # first index: floor i, second index: stair segment j
        self.stairs_graphs_list: list[list[nx.Graph]] = []  # first index: floor i, second index: stair segment j

        last_obstacle_height_range = (self.min_bound[2] - 1, self.min_bound[2] - 1)
        for floor_i, obstacle_height_range in enumerate(obstacle_height_ranges):
            floor_height_range = (last_obstacle_height_range[1], obstacle_height_range[0])
            map_shape = (self.x_range, self.y_range)
            floor_ground_map = get_free_space_map(ground_pc, self.min_bound, voxel_res, floor_height_range, map_shape)

            floor_obstacle_map = get_obstacle_map(
                self.full_voxel_grid, self.min_bound, voxel_res, obstacle_height_range, map_shape
            )

            # project poses onto a BEV and treat their vicinity as traversable
            pose_map_floor = get_pose_map_on_floor(
                self.pose_list,
                (floor_heights[floor_i], obstacle_height_range[1]),
                self.min_bound,
                pose_height_margin,
                pose_radius_as_traversable,
                voxel_res,
                map_shape,
            )

            # traversable BEV = ground BEV + pose BEV - obstacle BEV
            traversable_map = (floor_ground_map.astype(bool) | pose_map_floor) & (~floor_obstacle_map.astype(bool))
            traversable_map = binary_closing(
                traversable_map, structure=np.ones((close_kernel_size, close_kernel_size)), iterations=1
            )
            traversable_map = get_largest_region(traversable_map)
            if vis:
                cv2.imshow("floor_obstacle_map", floor_obstacle_map.astype(np.uint8) * 255)
                cv2.imshow("floor_ground_map", floor_ground_map.astype(np.uint8) * 255)
                cv2.imshow("pose_map", pose_map_floor)
                cv2.imshow("traversable_map", traversable_map.astype(np.uint8) * 255)
                cv2.waitKey()

            if floor_graph_type == "voronoi":
                floor_graph = get_voronoi_graph(
                    traversable_map,
                    str(floor_i),
                    self.min_bound,
                    voxel_res,
                    floor_height=floor_heights[floor_i],
                    vis=vis,
                )
                floor_graph = sparsify_graph(floor_graph, voxel_res, resampling_dist=graph_sparsify_dist)
            elif floor_graph_type == "delaunay":
                floor_graph = get_delaunay_graph(
                    traversable_map,
                    str(floor_i),
                    self.min_bound,
                    voxel_res,
                    floor_height=floor_heights[floor_i],
                    delaunay_distance_threshold=delaunay_distance_threshold,
                    vis=vis,
                )
            else:
                raise ValueError(f"Unsupported floor_graph_type: {floor_graph_type}")
            self.floor_graphs[floor_i] = floor_graph

            traversable_x, traversable_y = np.where(traversable_map)
            traversable_ground_voxels = np.concatenate(
                [
                    traversable_x.reshape((-1, 1)),
                    traversable_y.reshape((-1, 1)),
                    np.ones((len(traversable_x), 1)) * (floor_heights[floor_i] - self.min_bound[2]) / voxel_res,
                ],
                axis=1,
            )  # (N, 3)
            traversable_ground_voxel_grid.append(traversable_ground_voxels)
            last_obstacle_height_range = obstacle_height_range

            # build stairs graph between this floor and the next floor
            if floor_i + 1 < len(floor_heights):
                stairs_graphs_list, stair_pos_lists = get_stairs_graph(
                    self.pose_list, floor_i, floor_heights[floor_i], margin=0.5, vis=vis
                )
                self.stairs_pc_lists.append(stair_pos_lists)
                self.stairs_graphs_list.append(stairs_graphs_list)

        # connect floor graphs with the stairs graph
        fused_floor_graph = self.floor_graphs[0]
        for floor_i, stair_graphs in enumerate(self.stairs_graphs_list):
            for segment_j, stair_graph in enumerate(stair_graphs):
                fused_floor_graph = connect_graphs(stair_graph, fused_floor_graph)
            fused_floor_graph = connect_graphs(self.floor_graphs[floor_i + 1], fused_floor_graph)
        traversable_ground_voxel_grid = np.vstack(traversable_ground_voxel_grid)
        traversable_ground_points = traversable_ground_voxel_grid * voxel_res + self.min_bound
        return fused_floor_graph, traversable_ground_points

    def plan_global_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, yaml_path: Path = None) -> None:
        """Plan a global path from start to goal position and save it as a yaml file.

        Args:
            start_pos (np.ndarray): Start position (x, y, z).
            goal_pos (np.ndarray): Goal position (x, y, z).
            yaml_path (Path): Path to save the yaml file.
        """
        path = plan_global_path(self.floor_graph, start_pos, goal_pos)
        if not path:
            print("No path found. YAML file will not be created.")
            return
        path_np = np.concatenate([pos.reshape((1, 3)) for pos in path], axis=0)  # (N, 3)
        waypoints = compute_waypoint_orientations(path_np)
        if yaml_path is not None:
            waypoints_to_yaml(yaml_path, waypoints)
        return waypoints
