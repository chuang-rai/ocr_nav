# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion
from scipy.signal import find_peaks
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from ocr_nav.utils.pyvista_vis_utils import (
    draw_coordinate,
    draw_cube,
    draw_line,
    draw_sphere,
)

if TYPE_CHECKING:
    from ocr_nav.scene_graph.floor_graph import FloorGraph
    from ocr_nav.scene_graph.pose_graph import PoseGraph


def get_free_space_map(
    ground_pc: np.ndarray,
    min_bound: np.ndarray,
    cell_size: float,
    free_space_height_range: tuple[float, float],
    free_map_shape: tuple[int, int],
) -> np.ndarray:
    """Get the free space map by filtering the full voxel grid with a height range.

    Args:
        ground_pc (np.ndarray): Ground point cloud as a numpy array of shape (N, 3).
        min_bound (np.ndarray): Minimum bound of the floor map in world coordinates.
        cell_size (float): Cell size of the floor map.
        free_space_height_range (tuple[float, float]): Height range (min_height, max_height) to consider as free space.
        free_map_shape (tuple[int, int]): Shape of the resulting free space map (height, width).

    Returns:
        np.ndarray: Free space map of shape (H, W) where 1 indicates free space and 0 indicates occupied space.
    """
    low, high = np.floor((np.array(free_space_height_range) - min_bound[2]) / cell_size).astype(int)
    ground_voxel_grid = np.round((np.array(ground_pc.points) - min_bound) / cell_size).astype(int)
    ground_mask = (ground_voxel_grid[:, 2] >= low) & (ground_voxel_grid[:, 2] < high)
    floor_ground_voxels = ground_voxel_grid[ground_mask, :]
    floor_ground_map = np.zeros(free_map_shape, dtype=bool)
    floor_ground_map[floor_ground_voxels[:, 0].astype(int), floor_ground_voxels[:, 1].astype(int)] = True
    return floor_ground_map


def get_obstacle_map(
    full_voxel_grid: np.ndarray,
    min_bound: np.ndarray,
    cell_size: float,
    obstacle_height_range: tuple[float, float],
    map_shape: tuple[int, int],
) -> np.ndarray:
    """Get the obstacle map by filtering the full voxel grid with a height range.

    Args:
        full_voxel_grid (np.ndarray): Full voxel grid as a numpy array of shape (N, 3).
        min_bound (np.ndarray): Minimum bound of the floor map in world coordinates.
        cell_size (float): Cell size of the floor map.
        obstacle_height_range (tuple[float, float]): Height range (min_height, max_height) to consider as obstacles.
        map_shape (tuple[int, int]): Shape of the resulting obstacle map (height, width).

    Returns:
        np.ndarray: Obstacle map of shape (H, W) where 1 indicates occupied space and 0 indicates free space.
    """
    low, high = np.floor((np.array(obstacle_height_range) - min_bound[2]) / cell_size).astype(int)
    obstacle_mask = (full_voxel_grid[:, 2] >= low) & (full_voxel_grid[:, 2] < high)
    floor_obstacle_voxels = full_voxel_grid[obstacle_mask, :]
    floor_obstacle_map = np.zeros(map_shape, dtype=bool)
    floor_obstacle_map[floor_obstacle_voxels[:, 0].astype(int), floor_obstacle_voxels[:, 1].astype(int)] = True

    return floor_obstacle_map


def get_pose_map_on_floor(
    pose_list: list[np.ndarray],
    floor_height_range: tuple[float, float],
    min_bound: np.ndarray,
    pose_height_margin: float,
    radius: float,
    voxel_res: float,
    map_shape: tuple[int, int],
) -> np.ndarray:
    """Generate a binary BEV map where projected pose locations and their
    vicinity within a radius are 1. Otherwise pixel values are 0.

    Args:
        pose_list (list[np.ndarray]): List of poses as numpy arrays of shape (4, 4).
        floor_height_range (tuple[float, float]): Height range for
            selecting poses on the floor.
        min_bound (np.ndarray): Minimum bound of the grid.
        voxel_res (float, optional): Voxel resolution. Defaults to 0.1.
        pose_height_margin (float, optional): Height range for filtering
            poses around the major height. Defaults to 0.3.
        radius (float, optional): Radius around each pose to mark on the
            map. Defaults to 0.5.
        map_shape (tuple[int, int]): Shape of the resulting pose map (height, width).
    Returns:
        np.ndarray: Binary BEV map with pose locations marked.
    """
    poses_list_on_floor = [
        pose for pose in pose_list if (pose[2, 3] >= floor_height_range[0]) and (pose[2, 3] <= floor_height_range[1])
    ]
    pose_heights = np.array([pose[2, 3] for pose in poses_list_on_floor])
    clusters = DBSCAN(eps=0.2).fit(pose_heights.reshape(-1, 1))
    labels, counts = np.unique(clusters.labels_, return_counts=True)
    id = np.argmax(counts)
    mask = clusters.labels_ == labels[id]
    major_height = np.mean(pose_heights[mask])
    filtered_poses_list = [
        pose for pose in poses_list_on_floor if np.abs(pose[2, 3] - major_height) < pose_height_margin
    ]

    pose_map = np.zeros(map_shape, dtype=np.uint8)
    for filtered_pose in filtered_poses_list:
        voxel_x = int((filtered_pose[0, 3] - min_bound[0]) / voxel_res)
        voxel_y = int((filtered_pose[1, 3] - min_bound[1]) / voxel_res)
        # mark the position on the grid
        pose_map = cv2.circle(pose_map, (voxel_y, voxel_x), radius=int(radius / voxel_res), color=255, thickness=-1)
    return pose_map


def get_voronoi_graph(
    main_free_map: np.ndarray,
    floor_id: str,
    min_bound: np.ndarray,
    cell_size: float,
    floor_height: float,
    height_map: np.ndarray = None,
    vis: bool = False,
) -> nx.Graph:
    """Generate the Voronoi Graph of the floor based on the free space map substracting obstacle map.
    In main_free_map, 1 indicates free space and 0 indicates occupied space.

    Args:
        main_free_map (np.ndarray): Free space map.
        floor_id (str): The floor id. For example, "0" for the first floor.
        min_bound (np.ndarray): Minimum bound of the floor map in world coordinates.
        cell_size (float): Cell size of the floor map.
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

    if vis:
        fig_free = main_free_map.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)
    if height_map is None:
        height_map = np.ones_like(boundary_map) * floor_height
    voronoi_graph = nx.Graph()
    for simplex in voronoi.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            continue
        src, tar = voronoi.vertices[simplex]
        shape = np.array(main_free_map.shape)
        if not (index_in_bound(src, shape) and index_in_bound(tar, shape)):
            continue
        if main_free_map[int(src[0]), int(src[1])] == 0 or main_free_map[int(tar[0]), int(tar[1])] == 0:
            continue
        if vis:
            cv2.line(fig_free, tuple(np.int32(src[::-1])), tuple(np.int32(tar[::-1])), (0, 0, 255), 1)
            cv2.circle(fig_free, tuple(np.int32(src[::-1])), 2, (255, 0, 0), -1)
        src_node_id = (src[0], src[1], floor_id)
        tar_node_id = (tar[0], tar[1], floor_id)
        src_pos_2d = src * cell_size + min_bound[:2]
        tar_pos_2d = tar * cell_size + min_bound[:2]
        src_int, tar_int = np.round(src).astype(int), np.round(tar).astype(int)

        # add src node
        if src_node_id not in voronoi_graph.nodes:
            src_pos_3d = np.array([src_pos_2d[0], src_pos_2d[1], height_map[src_int[0], src_int[1]]]).astype(float)
            voronoi_graph.add_node(src_node_id, pos=(src_pos_3d[0], src_pos_3d[1], src_pos_3d[2]), floor_id=floor_id)

        # add tar node
        if tar_node_id not in voronoi_graph.nodes:
            tar_pos_3d = np.array([tar_pos_2d[0], tar_pos_2d[1], height_map[tar_int[0], tar_int[1]]]).astype(float)
            voronoi_graph.add_node(tar_node_id, pos=(tar_pos_3d[0], tar_pos_3d[1], tar_pos_3d[2]), floor_id=floor_id)

        # add edge
        if src_node_id not in voronoi_graph[tar_node_id]:
            src_pos = np.array(voronoi_graph.nodes[src_node_id]["pos"])
            tar_pos = np.array(voronoi_graph.nodes[tar_node_id]["pos"])
            voronoi_graph.add_edge(src_node_id, tar_node_id, dist=float(np.linalg.norm(src_pos - tar_pos)))
    if vis:
        cv2.imshow("voronoi_free", fig_free)
        cv2.waitKey()
    return voronoi_graph


def get_delaunay_graph(
    main_free_map: np.ndarray,
    floor_id: str,
    min_bound: np.ndarray,
    cell_size: float,
    floor_height: float,
    height_map: np.ndarray = None,
    delaunay_distance_threshold: float = 0.5,
    vis: bool = False,
) -> nx.Graph:
    delaunay_pixel_distance = int(delaunay_distance_threshold / cell_size)
    map_shape = np.array(main_free_map.shape)
    rows, cols = np.meshgrid(np.arange(map_shape[0]), np.arange(map_shape[1]), indexing="ij")
    rows = rows[::delaunay_pixel_distance, ::delaunay_pixel_distance]
    cols = cols[::delaunay_pixel_distance, ::delaunay_pixel_distance]
    points_2d = np.array([rows.flatten(), cols.flatten()]).T

    if height_map is None:
        height_map = (np.ones_like(main_free_map) * floor_height).astype(float)

    if vis:
        fig_free = main_free_map.copy().astype(np.uint8) * 255
        fig_free = cv2.cvtColor(fig_free, cv2.COLOR_GRAY2BGR)

    delaunay_graph = nx.Graph()
    delaunay = Delaunay(points_2d)
    for simplex in delaunay.simplices:
        simplex = np.asarray(simplex).astype(int)
        three_points = points_2d[simplex]
        if main_free_map[three_points[:, 0], three_points[:, 1]].sum() < 3:
            continue

        for p_2d in three_points:
            p_2d = p_2d.astype(int)
            if delaunay_graph.has_node((int(p_2d[0]), int(p_2d[1]), floor_id)):
                continue
            p_3d = np.array(
                [p_2d[0] * cell_size + min_bound[0], p_2d[1] * cell_size + min_bound[1], height_map[p_2d[0], p_2d[1]]]
            ).astype(float)
            delaunay_graph.add_node(
                (int(p_2d[0]), int(p_2d[1]), floor_id), pos=(p_3d[0], p_3d[1], p_3d[2]), floor_id=floor_id
            )
            if vis:
                cv2.circle(fig_free, tuple(p_2d[::-1]), 2, (255, 0, 0), -1)

        for i, j in zip([0, 1, 2], [1, 2, 0]):
            src_2d = three_points[i]
            tar_2d = three_points[j]
            src_node_id = (int(src_2d[0]), int(src_2d[1]), floor_id)
            tar_node_id = (int(tar_2d[0]), int(tar_2d[1]), floor_id)
            if src_node_id not in delaunay_graph[tar_node_id]:
                src_pos = np.array(delaunay_graph.nodes[src_node_id]["pos"]).astype(float)
                tar_pos = np.array(delaunay_graph.nodes[tar_node_id]["pos"]).astype(float)
                delaunay_graph.add_edge(src_node_id, tar_node_id, dist=float(np.linalg.norm(src_pos - tar_pos)))
                if vis:
                    cv2.line(fig_free, tuple(src_2d[::-1]), tuple(tar_2d[::-1]), (0, 0, 255), 1)
    if vis:
        cv2.imshow("delaunay_free", fig_free)
        cv2.waitKey()

    return delaunay_graph


def get_stairs_graph(
    pose_list: list[np.ndarray],
    floor_id: int,
    floor_height: float,
    margin: float = 0.1,
    buffer_pose_num: int = 5,
    vis: bool = False,
) -> tuple[list[nx.Graph], list[np.ndarray]]:
    """Generate a graph representing the exploration trajectory on the stairs
    between two floors.

    Args:
        floor_id (int): The floor id. For example, 0 for the first floor.
        floor_height (float): The height of the current floor.
        margin (float, optional): A range of height for selecting poses.
            Defaults to 0.1.
        buffer_pose_num (int, optional): Number of extra poses to include at the beginning and
            the end of the stair trajectory for smoother connection. Defaults to 5.
        vis (bool, optional): Whether to visualize the stairs graph.
            Defaults to False.

    Returns:
        tuple[list[nx.Graph], list[np.ndarray]]: A tuple containing the list of stairs graphs
            and the list of stair positions (N, 3). Each stairs graph corresponds to a trajectory segment
            between two floors, and each stair position array contains the 3D positions of the poses
            in that segment.
    """

    pose_heights = np.array([pose[2, 3] for pose in pose_list])
    frequency, bin_edges = np.histogram(pose_heights, bins=100)  # len(bin_edges) = len(frequency) + 1
    min_peak_height = 0.3 * np.max(frequency)
    peaks, _ = find_peaks(
        frequency, distance=20, height=min_peak_height
    )  # return the indices of the peak position (x direction)

    if vis:
        plt.figure()
        plt.title("Pose Height Histogram with Detected Peaks")
        plt.plot(bin_edges[:-1], frequency)
        plt.plot(bin_edges[peaks], frequency[peaks], "x")
        plt.show()

    low = bin_edges[peaks[floor_id]] + margin
    high = bin_edges[peaks[floor_id + 1]]

    trajectory_segments, trajectory_segment_pose_indices = filter_trajectory_segments_within_height_range(
        pose_list, [low, high]
    )
    valid_trajectory_segments = []
    valid_trajectory_segment_pose_indices = []
    for segment, segment_pose_indices in zip(trajectory_segments, trajectory_segment_pose_indices):
        if check_if_trajectory_segment_can_reach_height_range_limits(segment, [low, high], margin=margin):
            valid_trajectory_segments.append(segment)
            valid_trajectory_segment_pose_indices.append(segment_pose_indices)

    if len(valid_trajectory_segments) == 0:
        print(
            "No valid trajectory segment found for stairs between floors " + str(floor_id) + " and " + str(floor_id + 1)
        )
        print("The height range is [" + str(low) + ", " + str(high) + "]")
        return [nx.Graph()], [np.array([])]

    stairs_graphs = []
    stair_pos_lists = []
    for segment, segment_pose_indices in zip(valid_trajectory_segments, valid_trajectory_segment_pose_indices):
        segment_pose_list = get_extended_segment_pose_list(
            pose_list, segment_pose_indices[0], segment_pose_indices[-1], buffer_pose_num
        )

        lowest_pose_height = min([pose[2, 3] for pose in segment_pose_list])
        height_offset = lowest_pose_height - floor_height
        segment_pos_list = get_positions_with_z_offset(segment_pose_list, -height_offset)

        # build stairs graph
        segment_stairs_graph = nx.Graph()
        last_node = None
        last_pos = None
        for i, pos in enumerate(segment_pos_list):
            segment_stairs_graph.add_node(
                (pos[0], pos[1], str(floor_id)), pos=(pos[0], pos[1], pos[2]), floor_id=floor_id
            )
            if i > 0:
                segment_stairs_graph.add_edge(
                    last_node,
                    (pos[0], pos[1], str(floor_id)),
                    dist=float(np.linalg.norm(pos - last_pos)),
                )
            last_node = (pos[0], pos[1], str(floor_id))
            last_pos = pos

        stairs_graphs.append(segment_stairs_graph)
        stair_pos_lists.append(segment_pos_list)

    return stairs_graphs, stair_pos_lists


def filter_trajectory_segments_within_height_range(
    pose_list: list[np.ndarray], height_range: tuple[float, float]
) -> tuple[list[list[np.ndarray]], list[list[int]]]:
    """Filter trajectory segments whose z values are within a certain height range.
    Each segment is a list of poses with consecutive indices.

    Args:
        pose_list (list[np.ndarray]): List of poses as numpy arrays of shape (4, 4).
        height_range (tuple[float, float]): Height range (min_height, max_height).

    Returns:
        tuple[list[list[np.ndarray]], list[list[int]]]: A tuple containing a list of
            trajectory segments within the height range and a list of their corresponding
            pose indices in the pose_list.
    """
    min_height, max_height = height_range
    segments = []
    segment_pose_indices = []
    current_segment = []
    current_segment_pose_indices = []

    for pose_id, pose in enumerate(pose_list):
        z_value = pose[2, 3]
        if min_height <= z_value <= max_height:
            current_segment.append(pose)
            current_segment_pose_indices.append(pose_id)
        else:
            if current_segment:
                segments.append(current_segment)
                segment_pose_indices.append(current_segment_pose_indices)
                current_segment = []
                current_segment_pose_indices = []
    if current_segment:
        segments.append(current_segment)
        segment_pose_indices.append(current_segment_pose_indices)

    return segments, segment_pose_indices


def check_if_trajectory_segment_can_reach_height_range_limits(
    trajectory_segment: list[np.ndarray],
    height_range: tuple[float, float],
    margin: float = 0.2,
) -> bool:
    """Check if a trajectory segment can reach both limits of a height range.

    Args:
        trajectory_segment (list[np.ndarray]): List of poses as numpy arrays of shape (4, 4).
        height_range (tuple[float, float]): Height range (min_height, max_height).
        margin (float, optional): Margin to consider when checking if the trajectory can reach
            the limits. Defaults to 0.2 meters.

    Returns:
        bool: True if the segment can reach both limits, False otherwise.
    """
    min_height, max_height = height_range
    z_values = [pose[2, 3] for pose in trajectory_segment]

    can_reach_min = any(z <= min_height + margin for z in z_values)
    can_reach_max = any(z >= max_height - margin for z in z_values)

    return can_reach_min and can_reach_max


def get_extended_segment_pose_list(
    pose_list: list[np.ndarray], start_idx: int, end_idx: int, buffer_pose_num: int
) -> list[np.ndarray]:
    """Get an extended segment of the pose list by adding buffer poses at both ends.

    Args:
        pose_list (list[np.ndarray]): List of poses as numpy arrays of shape (4, 4).
        start_idx (int): Start index of the segment.
        end_idx (int): End index of the segment.
        buffer_pose_num (int): Number of buffer poses to add at both ends.

    Returns:
        list[np.ndarray]: Extended segment of poses.
    """
    extended_start_idx = max(start_idx - buffer_pose_num, 0)
    extended_end_idx = min(end_idx + buffer_pose_num, len(pose_list) - 1)
    return pose_list[extended_start_idx : extended_end_idx + 1]


def get_positions_with_z_offset(pose_list: list[np.ndarray], z_offset: float) -> np.ndarray:
    """Get positions from a list of poses with a z offset applied.

    Args:
        pose_list (list[np.ndarray]): List of poses as numpy arrays of shape (4, 4).
        z_offset (float): Z offset to apply to each position.

    Returns:
        np.ndarray: Array of positions with shape (N, 3) after applying the z offset.
    """
    positions = np.array([pose[:3, 3] for pose in pose_list])
    positions[:, 2] += z_offset
    return positions


def index_in_bound(index: np.ndarray, shape: np.ndarray) -> bool:
    """Check if a pixel is within the image bounds with a margin.

    Args:
        pixel (np.ndarray): Pixel coordinates as a numpy array of shape (2,).
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        margin (int, optional): Margin in pixels to consider when checking bounds. Defaults to 10.

    Returns:
        bool: True if the pixel is within bounds, False otherwise.
    """
    return all(0 <= index) and all(index < shape)


def visualize_pose_graph(plotter: pv.Plotter, pose_graph: PoseGraph) -> None:
    """Visualize the pose graph using Open3D.

    Args:
        pose_graph (PoseGraph): The pose graph to visualize.
    """
    total_pose_nodes = sorted([node for node, att in pose_graph.G.nodes(data=True) if isinstance(node, int)])
    num_pose_nodes = total_pose_nodes[-1] + 1
    graymap = (np.linspace(0, 1, num_pose_nodes) * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(graymap, cv2.COLORMAP_JET).reshape((num_pose_nodes, 3))
    print("Plotting pose nodes...")
    for node, att in pose_graph.G.nodes(data=True):
        if isinstance(node, int):
            pose_node = att["pose"]
            pose_id = pose_node.pose_id
            cam_cube = draw_cube(pose_node.pose[:3, 3], size=0.5, color="blue")
            plotter.add_mesh(cam_cube, color=colormap[pose_id] / 255.0)
            if node == 0:
                axis = draw_coordinate(pose_node.pose[:3, 3], size=2)
                plotter.add_actor(axis)  # type: ignore
    print("Plotting pose edges...")
    for edge in pose_graph.G.edges():
        node1, node2 = edge
        if isinstance(node1, int) and isinstance(node2, int):
            pose1 = pose_graph.G.nodes[node1]["pose"]
            pose2 = pose_graph.G.nodes[node2]["pose"]
            line = draw_line(pose1.pose[:3, 3], pose2.pose[:3, 3])
            plotter.add_mesh(line, color="red", line_width=2, render_lines_as_tubes=True)
    return plotter


def visualize_floor_graph(plotter: pv.Plotter, floor_graph: FloorGraph) -> None:
    """Visualize the floor graph using Open3D.

    Args:
        floor_graph (FloorGraph): The floor graph to visualize.
    """

    print("Plotting voronoi graphs...")
    for edge in tqdm(floor_graph.floor_graph.edges()):
        src, tar = edge
        src_pos = floor_graph.floor_graph.nodes[src]["pos"]
        tar_pos = floor_graph.floor_graph.nodes[tar]["pos"]
        line = draw_line(
            np.array([src_pos[0], src_pos[1], src_pos[2]]),
            np.array([tar_pos[0], tar_pos[1], tar_pos[2]]),
        )
        plotter.add_mesh(line, line_width=4, color="green", render_lines_as_tubes=True)
    for node in tqdm(floor_graph.floor_graph.nodes()):
        node_pos = floor_graph.floor_graph.nodes[node]["pos"]
        sphere = draw_sphere(
            np.array([node_pos[0], node_pos[1], node_pos[2]]),
            radius=0.1,
        )
        plotter.add_mesh(sphere, color="green")
    return plotter


def sparsify_graph(floor_graph: nx.Graph, cell_size: float, resampling_dist: float = 0.4) -> nx.Graph:
    """Sparsify the graph by removing nodes with only two neighbors and within the proximity of
    resampling_dist. It will keep junction nodes (endpoints and intersections) and resample nodes
    between junctions.

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
            _add_resampled_path(new_graph, floor_graph, path, resampling_dist, cell_size)

    return new_graph


def _add_resampled_path(
    new_graph: nx.Graph, original_graph: nx.Graph, path: list, resampling_dist: float, cell_size: float
):
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
            float(np.linalg.norm(np.array(original_graph.nodes[u]["pos"]) - np.array(original_graph.nodes[v]["pos"]))),
        )

        accumulated_dist += d

        # If we exceed resampling distance, create a node in the new graph
        # Note: self.cell_size included as per your original logic
        if (accumulated_dist * cell_size) > resampling_dist:
            if v not in new_graph:
                new_graph.add_node(v, **original_graph.nodes[v])

            dist_val = float(
                np.linalg.norm(np.array(new_graph.nodes[predecessor]["pos"]) - np.array(new_graph.nodes[v]["pos"]))
            )
            new_graph.add_edge(predecessor, v, dist=dist_val)

            predecessor = v
            accumulated_dist = 0

    # Always connect the final segment to the last junction
    last_node = path[-1]
    if last_node != predecessor:
        if last_node not in new_graph:
            new_graph.add_node(last_node, **original_graph.nodes[last_node])

        dist_val = float(
            np.linalg.norm(np.array(new_graph.nodes[predecessor]["pos"]) - np.array(new_graph.nodes[last_node]["pos"]))
        )
        new_graph.add_edge(predecessor, last_node, dist=dist_val)


def connect_graphs(src_graph: nx.Graph, tar_graph: nx.Graph, max_dist: float = 1) -> nx.Graph:
    """Connect two graphs by finding the closest node in the source graph to the target graph. The graphs' nodes
    should contain a "pos" attribute indicating their 3D position. The nodes should be tuples of (x, y, floor_id).

    Args:
        src_graph (nx.Graph): The source graph.
        tar_graph (nx.Graph): The target graph.
        max_dist (float, optional): Maximum distance to consider for connection. Defaults to 1 meters.

    Returns:
        tar_graph (nx.Graph): The resulting graph.
    """
    target_node_ids = [node for node in tar_graph.nodes if tar_graph.degree(node) > 1]
    if not target_node_ids:
        print("No valid target nodes with degree > 1 found in target graph. No connection made.")
        return nx.compose(tar_graph, src_graph)

    target_pos_array = np.array([tar_graph.nodes[node]["pos"] for node in target_node_ids])

    src_node_ids = list(src_graph.nodes)
    src_pos_array = np.array([src_graph.nodes[node]["pos"] for node in src_node_ids])
    dist_mat = cdist(src_pos_array, target_pos_array)

    rows, cols = np.where(dist_mat < max_dist)
    if len(rows) == 0:
        print("No nodes in source graph are within max_dist to target graph. No connection made.")
        print("Minimum distance is " + str(np.min(dist_mat)))
        return nx.compose(tar_graph, src_graph)

    fused_graph = nx.compose(tar_graph, src_graph)
    for row, col in zip(rows, cols):
        src_node = src_node_ids[row]
        tar_node = target_node_ids[col]
        src_pos = np.array(src_graph.nodes[src_node]["pos"])
        tar_pos = np.array(tar_graph.nodes[tar_node]["pos"])
        fused_graph.add_edge(
            src_node,
            tar_node,
            dist=float(np.linalg.norm(src_pos - tar_pos)),
        )
    return fused_graph


def plan_global_path(floor_graph: nx.Graph, start_pos: np.ndarray, goal_pos: np.ndarray) -> list[np.ndarray]:
    """Plan a global path from start to goal position using the fused floor graph.

    Args:
        start_pos (np.ndarray): Start position (x, y, z).
        goal_pos (np.ndarray): Goal position (x, y, z).
    Returns:
        list[np.ndarray]: List of waypoints [x, y, z, qw, qx, qy, qz] representing the planned path.
    """

    floor_nodes = [(i, floor_graph.nodes[node]) for i, node in enumerate(floor_graph.nodes)]
    floor_node_ids = [node[0] for node in floor_nodes]
    floor_node_pos = np.array([node[1]["pos"] for node in floor_nodes])
    dist_to_start = np.linalg.norm(floor_node_pos - start_pos.reshape((1, 3)), axis=1)
    dist_to_goal = np.linalg.norm(floor_node_pos - goal_pos.reshape((1, 3)), axis=1)
    closest_start_node_id = floor_node_ids[np.argmin(dist_to_start)]
    closest_goal_node_id = floor_node_ids[np.argmin(dist_to_goal)]
    start_node = list(floor_graph.nodes)[closest_start_node_id]
    goal_node = list(floor_graph.nodes)[closest_goal_node_id]

    try:
        dijkstra_path = nx.dijkstra_path(floor_graph, start_node, goal_node, weight="dist")
    except nx.NetworkXNoPath:
        print("No path found between start and goal.")
        return []

    dijkstra_path_pos = [np.array(floor_graph.nodes[node]["pos"]) for node in dijkstra_path]
    if np.linalg.norm(np.array(start_pos) - dijkstra_path_pos[0]) > 0.1:
        dijkstra_path_pos = [start_pos] + dijkstra_path_pos
    if np.linalg.norm(np.array(goal_pos) - dijkstra_path_pos[-1]) > 0.1:
        dijkstra_path_pos = dijkstra_path_pos + [goal_pos]

    return dijkstra_path_pos


def compute_waypoint_orientations(path: list[np.ndarray]) -> np.ndarray:
    """Compute orientations for each waypoint in the path based on the direction to the next waypoint.

    Args:
        path (list[np.ndarray]): List of waypoints [x, y, z] representing the planned path.

    Returns:
        np.ndarray: (N, 7) Waypoints array where each row is[x, y, z, qw, qx, qy, qz].
    """
    waypoints_with_orientations = []
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        direction = end - start
        yaw = np.arctan2(direction[1], direction[0])
        # Convert yaw to quaternion (assuming roll and pitch are 0)
        qw = np.cos(yaw / 2)
        qx = 0
        qy = 0
        qz = np.sin(yaw / 2)
        waypoints_with_orientations.append(np.array([start[0], start[1], start[2], qw, qx, qy, qz]))
    # Add orientation for the last waypoint (same as the second last)
    waypoints_with_orientations.append(np.array([path[-1][0], path[-1][1], path[-1][2], qw, qx, qy, qz]))
    return np.array(waypoints_with_orientations)
