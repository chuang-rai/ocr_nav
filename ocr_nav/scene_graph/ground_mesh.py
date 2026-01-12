import os
import json
from pathlib import Path
import copy
import time
import numpy as np
import networkx as nx
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from scipy.ndimage import binary_erosion
from ocr_nav.scene_graph.text_graph import TextMap
from ocr_nav.utils.mapping_utils import project_points, downsample_point_cloud, points_to_mesh, segment_floor
from ocr_nav.utils.io_utils import (
    load_livox_poses_timestamps,
    search_latest_poses_within_timestamp_range,
    FolderIO,
)
from typing import Tuple


class GroundMesh:
    def __init__(self, voxel_size: float = 0.1):
        self.textmap = TextMap()
        self.cell_size = voxel_size
        self.voronoi_graphs = {}

    def build_ground_mesh_with_folder(
        self, folderio: FolderIO, filter_obs: bool = True, debug: bool = False, vis: bool = False
    ) -> None:
        if debug:
            debug_image_dir = folderio.root_dir / "debug_lidar_proj"
            debug_image_dir.mkdir(exist_ok=True)

        # load intrinsics
        intr_mat = folderio.get_intrinsics()

        # load some static transforms
        tf_livox2zed_left = folderio.get_livox2left_camera_tf()
        tf_zed_left2livox = np.linalg.inv(tf_livox2zed_left)
        tf_rs2zed_left = folderio.get_rslidar2left_camera_tf()
        livox_poses_timestamps_path = folderio.root_dir / "glim_livox_poses_timestamps.npy"

        # load livox poses and timestamps of glim results
        livox_poses, livox_timestamps = load_livox_poses_timestamps(livox_poses_timestamps_path)

        ground_pc = o3d.geometry.PointCloud()
        ground_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_rs = o3d.geometry.PointCloud()
        self.full_pc_livox = o3d.geometry.PointCloud()
        self.pose_list = []
        pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
        last_pi = -1
        for pi, p in pbar:
            if pi % 10 != 0:
                continue

            pbar.set_description(f"Processing frame {pi}")
            frame_id = p

            if pi == 0 or debug:
                img = folderio.get_image(pi)
                h, w, _ = np.array(img).shape

            masks = folderio.get_mask(pi)
            pc_livox = folderio.get_livox(pi)  # (N, 3)
            pc_rslidar = folderio.get_rslidar(pi)  # (N, 3)

            # transform point cloud from robosense frame to zed left camera frame
            livox_pose, livox_timestamp = search_latest_poses_within_timestamp_range(
                livox_poses,
                livox_timestamps,
                frame_id,
            )
            if livox_pose is None:
                print(f"No livox pose found for frame {frame_id}, skipping...")
                continue

            self.pose_list.append(livox_pose)

            pc_livox_hom = np.hstack((pc_livox, np.ones((pc_livox.shape[0], 1))))  # (N, 4)
            pc_hom_zed = tf_livox2zed_left @ pc_livox_hom.T
            pc_zed = pc_hom_zed[:3, :].T  # (N, 3)

            pc_rslidar_hom = np.hstack((pc_rslidar, np.ones((pc_rslidar.shape[0], 1))))  # (N, 4)
            pc_rs_hom_zed = tf_rs2zed_left @ pc_rslidar_hom.T
            pc_rs_zed = pc_rs_hom_zed[:3, :].T  # (N, 3)

            # project points to image plane
            pc_image_2d, pc_image_2d_depth, pc_image_3d = project_points(pc_zed, intr_mat, w, h)  # (N, 2)
            pc_rs_image_2d, pc_rs_image_2d_depth, pc_rs_image_3d = project_points(pc_rs_zed, intr_mat, w, h)  # (N, 2)

            if debug:
                depth_scale = (pc_image_2d_depth - np.min(pc_image_2d_depth)) / (
                    np.max(pc_image_2d_depth) - np.min(pc_image_2d_depth)
                ).reshape((-1, 1))
                depth_scale_gray = (depth_scale * 255).astype(np.uint8)
                # get depth color map jet
                depth_color = cv2.applyColorMap(depth_scale_gray, cv2.COLORMAP_JET)[0]  # (N, 3)
                for i in range(pc_image_2d.shape[0]):
                    u, v = int(pc_image_2d[i, 0]), int(pc_image_2d[i, 1])
                    # img.putpixel((u, v), tuple(depth_color[i]))
                    # draw a small circle at the projected point
                    for du in range(-2, 3):
                        for dv in range(-2, 3):
                            if 0 <= u + du < w and 0 <= v + dv < h:
                                img.putpixel((u + du, v + dv), tuple(depth_color[i]))

                assert isinstance(debug_image_dir, Path)
                debug_image_path = debug_image_dir / f"lidar_proj_{frame_id}.png"
                img.save(debug_image_path)

            ground_mask = masks[0][0]
            pc_o3d = self.select_points_in_masks(
                ground_mask, pc_image_2d, pc_image_3d, transform=livox_pose @ tf_zed_left2livox
            )
            ground_pc += pc_o3d
            pc_o3d_rs = self.select_points_in_masks(
                ground_mask,
                pc_rs_image_2d,
                pc_rs_image_3d,
                transform=livox_pose @ tf_zed_left2livox,
            )
            ground_pc_rs += pc_o3d_rs

            pc_o3d_livox_full_np = self.transform_point_cloud(pc_zed, livox_pose @ tf_zed_left2livox)
            pc_o3d_livox_full = o3d.geometry.PointCloud()
            pc_o3d_livox_full.points = o3d.utility.Vector3dVector(pc_o3d_livox_full_np)
            self.full_pc_livox += pc_o3d_livox_full

            pc_o3d_rs_full_np = self.transform_point_cloud(pc_rs_image_3d, livox_pose @ tf_zed_left2livox)
            pc_o3d_rs_full = o3d.geometry.PointCloud()
            pc_o3d_rs_full.points = o3d.utility.Vector3dVector(pc_o3d_rs_full_np)
            self.full_pc_rs += pc_o3d_rs_full

            pose_id = self.textmap.add_pose(pi, livox_pose, lidar=pc_o3d_rs_full)
            if last_pi != -1 and self.textmap.G.has_node(last_pi):
                self.textmap.add_pose_edge(last_pi, pi)

            last_pi = pi
            # add pose to the textmap
            pc_world = (livox_pose @ tf_zed_left2livox @ pc_hom_zed)[:3, :].T  # (N, 3)

        heights = segment_floor(np.array(ground_pc.points), resolution=0.05, vis=vis)
        ground_pc_rs_numpy_list = self.select_points_near_heights(
            np.array((ground_pc + ground_pc_rs).points), heights, threshold=0.2
        )
        ground_pc_rs_numpy = np.vstack(ground_pc_rs_numpy_list)
        self.ground_pc_rs = o3d.geometry.PointCloud()
        self.ground_pc_rs.points = o3d.utility.Vector3dVector(ground_pc_rs_numpy)
        self.ground_pc_rs = self.ground_pc_rs.voxel_down_sample(voxel_size=self.cell_size)

        if filter_obs:
            ground_pc_rs_numpy = self.filter_ground_points_with_obstacles_in_height_range(
                self.full_pc_livox,
                self.ground_pc_rs,
                np.array(heights),
                voxel_res=self.cell_size,
                high_threshold=2.7,
                low_threshold=0.6,
                vis=vis,
            )
            self.ground_pc_rs = o3d.geometry.PointCloud()
            self.ground_pc_rs.points = o3d.utility.Vector3dVector(ground_pc_rs_numpy)

    @staticmethod
    def save_voronoi_graph(graph: nx.Graph, graph_path: Path) -> None:
        """Save the Voronoi graph to a json file.

        Args:
            graph (nx.Graph): The Voronoi graph.
            floor_dir (str): The directory where the intermediate results are stored.
            name (str): The name of the file.
        """
        graph_json = nx.node_link_data(graph)
        with open(graph_path, "w") as f:
            json.dump(graph_json, f, indent=4)

    @staticmethod
    def load_voronoi_graph(graph_path: Path):
        with open(graph_path, "r") as f:
            graph_json = json.load(f)
        graph = nx.node_link_graph(graph_json)
        return graph

    def select_points_in_masks(
        self, mask: np.ndarray, pc_image_2d: np.ndarray, pc_image_3d: np.ndarray, transform: np.ndarray = np.eye(4)
    ) -> o3d.geometry.PointCloud:
        valid_indices = np.where(mask[pc_image_2d[:, 1].astype(int), pc_image_2d[:, 0].astype(int)] > 0)[0]
        ground_pc_image_3d = pc_image_3d[valid_indices, :]  # (K, 3)
        ground_pc_world = self.transform_point_cloud(ground_pc_image_3d, transform)
        # ground_pc_image_3d_hom = np.hstack((ground_pc_image_3d, np.ones((ground_pc_image_3d.shape[0], 1))))  # (K, 4)
        # ground_pc_world = (transform @ ground_pc_image_3d_hom.T).T[:, :3]  # (K, 3)
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(ground_pc_world)
        return pc_o3d

    def transform_point_cloud(self, pc: np.ndarray, transform: np.ndarray) -> np.ndarray:
        pc_hom = np.hstack((pc, np.ones((pc.shape[0], 1))))  # (N, 4)
        pc_transformed_hom = (transform @ pc_hom.T).T
        return pc_transformed_hom[:, :3]

    def select_points_near_heights(self, pc: np.ndarray, heights: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        mask = np.zeros(pc.shape[0], dtype=bool)
        pc_list = []
        for h in heights:
            mask = (pc[:, 2] >= h - threshold) & (pc[:, 2] <= h + threshold)
            pc_list.append(pc[mask, :])
        return pc_list

    def compute_grid_parameters(self, full_pc: o3d.geometry.PointCloud, voxel_res: float = 0.05) -> None:
        self.min_bound, self.max_bound = full_pc.get_min_bound(), full_pc.get_max_bound()
        self.in_bound = self.min_bound + np.array([0, 0, -1.0])
        self.full_voxel_grid = np.round((np.array(full_pc.points) - self.min_bound) / voxel_res)
        self.x_range, self.y_range, self.z_range = (
            np.max(self.full_voxel_grid[:, 0]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 1]).astype(int) + 1,
            np.max(self.full_voxel_grid[:, 2]).astype(int) + 1,
        )

    def filter_ground_points_with_obstacles_in_height_range(
        self,
        full_pc: o3d.geometry.PointCloud,
        ground_pc: o3d.geometry.PointCloud,
        heights: np.ndarray,
        voxel_res: float = 0.05,
        high_threshold: float = 1.7,
        low_threshold: float = 0.2,
        vis: bool = False,
    ) -> o3d.geometry.PointCloud:
        self.compute_grid_parameters(full_pc, voxel_res=voxel_res)
        height_ranges = zip(heights + low_threshold, heights + high_threshold)
        ground_voxel_grid = np.round((np.array(ground_pc.points) - self.min_bound) / voxel_res)

        traversable_ground_voxel_grid = []
        self.stairs_pc_list = []
        self.stairs_graph_list = []
        last_high = 0
        for floor_i, height_range in enumerate(height_ranges):
            low, high = np.floor((np.array(height_range) - self.min_bound[2]) / voxel_res).astype(int)
            obstacle_mask = (self.full_voxel_grid[:, 2] >= low) & (self.full_voxel_grid[:, 2] < high)
            ground_mask = (ground_voxel_grid[:, 2] >= last_high) & (ground_voxel_grid[:, 2] < low)
            floor_obstacle_voxels = self.full_voxel_grid[obstacle_mask, :]
            floor_ground_voxels = ground_voxel_grid[ground_mask, :]

            floor_obstacle_map = np.zeros((self.x_range, self.y_range), dtype=bool)
            floor_obstacle_map[floor_obstacle_voxels[:, 0].astype(int), floor_obstacle_voxels[:, 1].astype(int)] = True
            floor_ground_map = np.zeros((self.x_range, self.y_range), dtype=bool)
            floor_ground_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)] = True

            floor_obstacle_map_cv2 = floor_obstacle_map.astype(np.uint8) * 255

            floor_obstacle_map_close_cv2 = (
                cv2.morphologyEx(floor_obstacle_map_cv2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            ).astype(np.uint8) * 255
            floor_ground_map_cv2 = floor_ground_map.astype(np.uint8) * 255
            floor_ground_map_close_cv2 = (
                cv2.morphologyEx(floor_ground_map_cv2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            ).astype(np.uint8) * 255
            if vis:
                cv2.imshow("floor_obstacle_map", floor_obstacle_map_cv2)
                cv2.imshow("floor_obstacle_map_close", floor_obstacle_map_close_cv2)
                cv2.imshow("floor_ground_map", floor_ground_map_cv2)
                cv2.imshow("floor_ground_map_close", floor_ground_map_close_cv2)
                cv2.waitKey()

            pose_map_floor = self.get_pose_map_on_floor(
                (height_range[0], height_range[1]),
                self.min_bound,
                voxel_res=voxel_res,
                camera_height_range=0.4,
                radius=1,
                vis=vis,
            )

            traversable_map = (floor_ground_map | pose_map_floor) & (~floor_obstacle_map)
            traversable_map = (
                cv2.morphologyEx(traversable_map.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            )

            traversable_map = self.get_largest_region(traversable_map)
            if vis:
                cv2.imshow("traversable_map", traversable_map.astype(np.uint8) * 255)
                cv2.waitKey()

            if floor_i + 1 < len(heights):
                stairs_graph, stair_pos_list = self.get_stairs_graph(floor_i, margin=0.5, height_offset=1.0, vis=vis)
                self.stairs_pc_list.append(stair_pos_list)
                self.stairs_graph_list.append(stairs_graph)

            traversable_map_rgb = cv2.cvtColor(traversable_map.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            height_map = np.ones_like(traversable_map, dtype=np.float32) * (heights[floor_i])
            voronoi_graph = self.get_voronoi_graph(
                traversable_map, traversable_map_rgb, floor_id=str(floor_i), height_map=height_map, vis=vis
            )
            sparse_voronoi_graph = self.sparsify_graph(voronoi_graph, resampling_dist=0.4)
            self.voronoi_graphs[floor_i] = sparse_voronoi_graph

            traversable_x, traversable_y = np.where(traversable_map)
            # traversable_ground_indices = np.where(
            #     traversable_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)]
            # )[0]

            # traversable_ground_voxels = floor_ground_voxels[traversable_ground_indices, :]
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

        for floor_i, stair_graph in enumerate(self.stairs_graph_list):
            fused_graph = self.connect_voronoi_graphs(self.stairs_graph_list[floor_i], self.voronoi_graphs[floor_i])
            fused_graph = self.connect_voronoi_graphs(fused_graph, self.voronoi_graphs[floor_i + 1])
        self.fused_graph = fused_graph

        traversable_ground_voxel_grid = np.vstack(traversable_ground_voxel_grid)
        traversable_ground_points = traversable_ground_voxel_grid * voxel_res + self.min_bound
        return traversable_ground_points

    def get_pose_map_on_floor(
        self,
        floor_height_range: Tuple[float, float],
        min_bound: np.ndarray,
        voxel_res: float = 0.1,
        camera_height_range: float = 0.3,
        radius: float = 0.5,
        vis: bool = False,
    ) -> np.ndarray:
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

    def get_largest_region(self, binary_map: np.ndarray) -> np.ndarray:
        """Get the largest disconnected island region in the binary map.

        Args:
            binary_map (np.ndarray): The binary map.

        Returns:
            np.ndarray: the largest region in the binary map.
        """
        # Threshold it so it becomes binary
        input = (binary_map > 0).astype(np.uint8)
        output = cv2.connectedComponentsWithStats(input, 8, cv2.CV_8UC1)
        areas = output[2][:, -1]
        # TODO: the top region is 0 region, so we need to sort the areas and get the second largest
        # but I am not sure if the largest region is always the background
        id = np.argsort(areas)[::-1][1]
        return output[1] == id

    def get_voronoi_graph(
        self,
        main_free_map: np.ndarray,
        map_rgb: np.ndarray,
        floor_id: str,
        height_map: np.ndarray = None,
        vis: bool = False,
    ) -> nx.Graph:
        """Generate the Voronoi Graph of the floor based on the free space map substracting obstacle map.

        Args:
            main_free_map (np.ndarray): Free space map.
            map_rgb (np.ndarray): RGB map of the floor.
            floor_dir (str): Directory where the intermediate results will be stored.
            floor_id (str): The floor id. For example, "0" for the first floor.
            name (str, optional): The name is used for saving the intermediate results. Defaults to "".
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
        map_bgr = cv2.cvtColor(map_rgb, cv2.COLOR_RGB2BGR)
        fig = map_bgr.copy()
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
                or src[0] >= fig.shape[0]
                or src[1] < 0
                or src[1] >= fig.shape[1]
                or tar[0] < 0
                or tar[0] >= fig.shape[0]
                or tar[1] < 0
                or tar[1] >= fig.shape[1]
            ):
                continue
            if main_free_map[int(src[0]), int(src[1])] == 0 or main_free_map[int(tar[0]), int(tar[1])] == 0:
                continue
            cv2.line(
                fig,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.line(
                fig_free,
                tuple(np.int32(src[::-1])),
                tuple(np.int32(tar[::-1])),
                (0, 0, 255),
                1,
            )
            cv2.circle(fig, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)
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
            cv2.imshow("voronoi", fig)
            cv2.imshow("voronoi_free", fig_free)
            cv2.waitKey()
        # cv2.imwrite(os.path.join(floor_dir, f"vor_{name}.png"), fig)
        # cv2.imwrite(os.path.join(floor_dir, f"vor_free_{name}.png"), fig_free)
        # vertices = np.array(vertices)
        return voronoi_graph

    def sparsify_graph(self, floor_graph: nx.Graph, resampling_dist: float = 0.4):
        """
        Optimized sparsification using chain traversal instead of all-pairs shortest paths.
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

    def _add_resampled_path(self, new_graph, original_graph, path, resampling_dist):
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
