import numpy as np
import cv2
import torch


class XFeatMatcher:
    def __init__(self, top_k=4096):
        self.xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k)

    def match(self, img1, img2, top_k=4096) -> tuple[np.ndarray, np.ndarray]:
        """
        Match features between two images using XFeat.

        Args:
            img1 (np.ndarray): First input image in BGR format.
            img2 (np.ndarray): Second input image in BGR format.

        Returns:
            tuple[np.ndarray, np.ndarray]: Matched keypoints from img1 and img2.
        """
        mkpts1, mkpts2 = self.xfeat.match_xfeat(img1, img2, top_k=top_k)
        return mkpts1, mkpts2

    def compute_bbox_similarities(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        box_list_1: list[np.ndarray],
        box_list_2: list[np.ndarray],
        w: int,
        h: int,
        normalize_max: int = 1000,
    ) -> np.ndarray:
        """
        Compute similarity scores between bounding boxes from two lists based on matched keypoints.

        Args:
            img1 (np.ndarray): First input image in BGR format.
            img2 (np.ndarray): Second input image in BGR format.
            box_list_1 (list[np.ndarray]): List of bounding boxes from the first image.
            box_list_2 (list[np.ndarray]): List of bounding boxes from the second image.
            w (int): Width of the images.
            h (int): Height of the images.
            normalize_max (int): Maximum value for normalization of bounding box coordinates.
        Returns:
            np.ndarray: Similarity score matrix between bounding boxes.
        """
        kps1, kps2 = self.match(img1, img2)
        sim_matrix = np.zeros((len(box_list_1), len(box_list_2)), dtype=np.float32)
        for i, box1 in enumerate(box_list_1):
            x1_min, y1_min, x1_max, y1_max = box1
            x1_min = int(x1_min / w * normalize_max)
            y1_min = int(y1_min / h * normalize_max)
            x1_max = int(x1_max / w * normalize_max)
            y1_max = int(y1_max / h * normalize_max)
            for j, box2 in enumerate(box_list_2):
                x2_min, y2_min, x2_max, y2_max = box2
                x2_min = int(x2_min / w * normalize_max)
                y2_min = int(y2_min / h * normalize_max)
                x2_max = int(x2_max / w * normalize_max)
                y2_max = int(y2_max / h * normalize_max)

                # Count keypoints in each bounding box
                mask1 = (
                    (kps1[:, 0] >= x1_min) & (kps1[:, 0] <= x1_max) & (kps1[:, 1] >= y1_min) & (kps1[:, 1] <= y1_max)
                )
                mask2 = (
                    (kps2[:, 0] >= x2_min) & (kps2[:, 0] <= x2_max) & (kps2[:, 1] >= y2_min) & (kps2[:, 1] <= y2_max)
                )

                count1 = np.sum(mask1)
                count2 = np.sum(mask2)

                if count1 + count2 > 0:
                    sim_matrix[i, j] = 2 * np.sum(mask1 & mask2) / (count1 + count2)
        return sim_matrix
