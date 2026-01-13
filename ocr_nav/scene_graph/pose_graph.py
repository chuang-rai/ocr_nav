from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from ocr_nav.utils.levenshtein_utils import levenshtein_distance
from typing import List, Optional
import networkx as nx


@dataclass
class Pose:
    pose_id: int
    pose: np.ndarray
    lidar: Optional[np.ndarray] = None  # (N, 3)


@dataclass
class TextBag:
    text_dict: defaultdict[str, List[int]]  # map from text to the frame id
    pc: Optional[np.ndarray] = None  # (N, 3)


class PoseGraph:
    def __init__(self):
        self.G = nx.Graph()

    def add_pose(self, pose_id: int, pose: np.ndarray, lidar: Optional[np.ndarray] = None):
        p_node = Pose(pose_id, pose, lidar)
        self.G.add_node(pose_id, pose=p_node)
        return pose_id

    def add_pose_edge(self, pose_id1: int, pose_id2: int):
        self.G.add_edge(pose_id1, pose_id2)

    def add_text_to_pose(self, pose_id: int, text: str, pos_3d: Optional[np.ndarray] = None):
        text_dict = defaultdict(list)
        text_dict[text] = [pose_id]
        new_text_node = TextBag(text_dict=text_dict, pc=pos_3d.reshape((1, 3)) if pos_3d is not None else None)
        self.G.add_node(text, textbag=new_text_node)
        self.G.add_edge(pose_id, text)
        return text, new_text_node

    def link_old_text_node_to_new_pose(self, pose_id: int, text_node: str):
        self.G.add_edge(pose_id, text_node)

    def find_similar_text(self, text: str, dist_threshold: int):
        text_nodes = [(node, att) for node, att in self.G.nodes(data=True) if isinstance(node, str)]
        similar_node_and_dist = []
        for node, att in text_nodes:
            print(node, att)
            min_dist = 1000
            for node_text, pose_ids in att["textbag"].text_dict.items():
                dist = levenshtein_distance(node_text, text)
                if dist <= dist_threshold and dist < min_dist:
                    min_dist = dist
            if min_dist < 1000:
                similar_node_and_dist.append((node, min_dist))

        if len(similar_node_and_dist) == 0:
            return None
        most_similar_node_id_and_dist = sorted(similar_node_and_dist, key=lambda x: x[1])[0]
        return most_similar_node_id_and_dist
