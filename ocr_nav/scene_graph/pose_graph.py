from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from ocr_nav.utils.levenshtein_utils import levenshtein_distance
from typing import List, Optional
import networkx as nx


@dataclass
class Pose:
    """Class for storing pose information. It serves as a node in the pose graph class.
    Attributes:
    pose_id: unique identifier for the pose.
    pose: (4, 4) homogeneous transformation matrix representing the pose.
    lidar: (N, 3) optional point cloud data associated with the pose.
    """

    pose_id: int
    pose: np.ndarray
    lidar: Optional[np.ndarray] = None  # (N, 3)


class PoseGraph:
    """Class for storing a graph where nodes denote poses and edges connect
    poses from consecutive frames. It also stores text observations as nodes
    and connects them to the poses where they were observed.
    """

    def __init__(self):
        self.G = nx.Graph()

    def add_pose(self, pose_id: int, pose: np.ndarray, lidar: Optional[np.ndarray] = None) -> int:
        p_node = Pose(pose_id, pose, lidar)
        assert pose_id not in self.G.nodes, f"Pose id {pose_id} already exists in the graph."
        self.G.add_node(pose_id, pose=p_node)
        return pose_id

    def add_pose_edge(self, pose_id1: int, pose_id2: int):
        self.G.add_edge(pose_id1, pose_id2)
