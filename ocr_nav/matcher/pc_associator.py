import numpy as np
import open3d as o3d
from ocr_nav.utils.voxel_iou_utils import compute_voxel_iou
from ocr_nav.utils.mapping_utils import to_numpy_pc, to_o3d_pc


class GlobalFragmentAssociator:
    def __init__(self, threshold: float = 0.5, voxel_size_list: list = [0.05]):
        self.threshold = threshold
        self.voxel_size_list = sorted(voxel_size_list)[::-1]  # sort from large to small
        self.global_fragment_voxel_grids_history: dict[int, list[set[tuple[int, int, int]]]] = {}
        self.global_fragment_pc_history: dict[int, o3d.geometry.PointCloud] = {}
        pass

    def convert_pc_to_voxel_grid(self, pc: o3d.geometry.PointCloud, voxel_size: float) -> set[tuple[int, int, int]]:
        pc_np = to_numpy_pc(pc)
        voxel_grid = np.round(pc_np / voxel_size).astype(np.int32)
        return set(map(tuple, voxel_grid))

    def convert_pc_to_hierarchical_voxels(self, pc: o3d.geometry.PointCloud) -> list[set[tuple[int, int, int]]]:
        voxel_grids = []
        for voxel_size in self.voxel_size_list:
            voxel_grid = self.convert_pc_to_voxel_grid(pc, voxel_size)
            voxel_grids.append(voxel_grid)
        return voxel_grids

    def compute_voxel_iou(
        self, voxel_grid1: set[tuple[int, int, int]], voxel_grid2: set[tuple[int, int, int]]
    ) -> float:
        iou = compute_voxel_iou(voxel_grid1, voxel_grid2)
        return iou

    def compute_hierarchical_voxel_iou(
        self,
        voxel_grids1: list[set[tuple[int, int, int]]],
        voxel_grids2: list[set[tuple[int, int, int]]],
        early_stop_threshold: float = 0.0,
    ) -> float | None:
        for level_i, (vg1, vg2) in enumerate(zip(voxel_grids1, voxel_grids2)):
            iou = self.compute_voxel_iou(vg1, vg2)
            if level_i < len(self.voxel_size_list) - 1 and iou <= early_stop_threshold:
                return 0.0
        return iou

    def add_new_fragment(self, pc: o3d.geometry.PointCloud, id: int):
        voxel_grids = self.convert_pc_to_hierarchical_voxels(pc)
        self.global_fragment_voxel_grids_history[id] = voxel_grids
        self.global_fragment_pc_history[id] = pc
        self.global_fragment_pc_history[id] = self.global_fragment_pc_history[id].voxel_down_sample(
            self.voxel_size_list[-1]
        )

    def update_fragment(
        self, fragment_id: int, pc: o3d.geometry.PointCloud, hierarchical_voxel_grids: list[set[tuple[int, int, int]]]
    ):
        for level_i, voxel_grid in enumerate(hierarchical_voxel_grids):
            previous_set = self.global_fragment_voxel_grids_history[fragment_id][level_i]
            current_set = voxel_grid
            merged_set = previous_set.union(current_set)
            self.global_fragment_voxel_grids_history[fragment_id][level_i] = merged_set
        self.global_fragment_pc_history[fragment_id] += pc
        self.global_fragment_pc_history[fragment_id] = self.global_fragment_pc_history[fragment_id].voxel_down_sample(
            self.voxel_size_list[-1]
        )

    def associate_with_pc(self, pc: o3d.geometry.PointCloud, early_stop_threshold: float = 0.1) -> float:
        query_voxel_grids = self.convert_pc_to_hierarchical_voxels(pc)
        for fragment_id, hier_fragment_voxel_grids in self.global_fragment_voxel_grids_history.items():
            iou = self.compute_hierarchical_voxel_iou(
                query_voxel_grids, hier_fragment_voxel_grids, early_stop_threshold=early_stop_threshold
            )
            if iou > self.threshold:
                self.update_fragment(fragment_id, pc, query_voxel_grids)
                return fragment_id
        # create new fragment
        new_id = max(self.global_fragment_voxel_grids_history.keys(), default=-1) + 1
        self.add_new_fragment(pc, new_id)
        return new_id

    def associate_fragment_with_id(self, id: int, early_stop_threshold: float = 0.1) -> float:
        if id not in self.global_fragment_pc_history:
            return None
        other_ids = [k for k in self.global_fragment_pc_history.keys() if k != id]
        for other_id in other_ids:
            iou = self.compute_hierarchical_voxel_iou(
                self.global_fragment_voxel_grids_history[id],
                self.global_fragment_voxel_grids_history[other_id],
                early_stop_threshold=early_stop_threshold,
            )
            if iou > self.threshold:
                self.update_fragment(
                    other_id, self.global_fragment_pc_history[id], self.global_fragment_voxel_grids_history[id]
                )
                # remove the old fragment
                del self.global_fragment_pc_history[id]
                del self.global_fragment_voxel_grids_history[id]
                return other_id
        return None
