from pathlib import Path
import time
from tqdm import tqdm
from visualization_msgs.msg import Marker, MarkerArray
import rclpy
from rclpy.node import Node
import numpy as np
from typing import List

from ocr_nav.scene_graph.floor_graph import FloorGraph


def draw_cube(
    center: np.ndarray, frame_id: str = "map", size: float = 0.1, color: List[float] = [1.0, 0.0, 0.0]
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    marker.scale.x = size  # Width
    marker.scale.y = size  # Depth
    marker.scale.z = size  # Height
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Don't forget this!
    return marker


def draw_sphere(
    center: np.ndarray, frame_id: str = "map", radius: float = 0.1, color: List[float] = [0.0, 1.0, 0.0]
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = center[0]
    marker.pose.position.y = center[1]
    marker.pose.position.z = center[2]
    marker.scale.x = radius * 2  # Diameter
    marker.scale.y = radius * 2  # Diameter
    marker.scale.z = radius * 2  # Diameter
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Don't forget this!
    return marker


def draw_line(
    point1: np.ndarray,
    point2: np.ndarray,
    frame_id: str = "map",
    line_width: float = 0.05,
    color: List[float] = [0.0, 0.0, 1.0],
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = line_width  # Line width
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 1.0  # Don't forget this!

    start_point = point1
    end_point = point2

    marker.points.append(type("Point", (object,), {"x": start_point[0], "y": start_point[1], "z": start_point[2]})())
    marker.points.append(type("Point", (object,), {"x": end_point[0], "y": end_point[1], "z": end_point[2]})())

    return marker


def main():
    root_path = Path(
        # "/home/chuang/hcg/projects/ocr/data/Flexoffice_extracted/rosbag2_2025_12_10-10_25_57_perception_suite"
        "/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite"
    )
    fused_graph = FloorGraph.load_floor_graph(root_path / "ground_fused_graph.json")
    marker_array = MarkerArray()
    print("Plotting voronoi graphs...")
    marker_id = 0
    for edge in tqdm(fused_graph.edges()):
        node1, node2 = edge
        pos1 = fused_graph.nodes[node1]["pos"]
        pos2 = fused_graph.nodes[node2]["pos"]
        line_marker = draw_line(
            pos1,
            pos2,
            line_width=0.1,
            color=[0.0, 1.0, 0.0],
        )
        line_marker.id = marker_id
        marker_id += 1
        marker_array.markers.append(line_marker)
    print(f"Total markers: {len(marker_array.markers)}")

    for node in tqdm(fused_graph.nodes()):
        pos = fused_graph.nodes[node]["pos"]
        sphere_marker = draw_sphere(
            pos,
            radius=0.1,
            color=[0.0, 0.0, 1.0],
        )
        sphere_marker.id = marker_id
        marker_id += 1
        marker_array.markers.append(sphere_marker)
    print(f"Total markers: {len(marker_array.markers)}")

    rclpy.init()
    node = Node("traversable_graph_visualizer")
    publisher = node.create_publisher(MarkerArray, "traversable_graph_markers", 10)
    publisher.publish(marker_array)
    timer = node.create_timer(2.0, lambda: publisher.publish(marker_array))

    print("Publishing graph every 2 seconds. Press Ctrl+C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
