import numpy as np
import cv2
import torch
from ocr_nav.utils.visualization_utils import warp_corners_and_draw_matches


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
            tuple[np.ndarray, np.ndarray]: Matched keypoints (N, 2) from img1 and img2.
        """
        self.kps1, self.kps2 = self.xfeat.match_xfeat(img1, img2, top_k=top_k)
        H, mask = cv2.findHomography(self.kps1, self.kps2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        mask = mask.flatten()
        self.kps1 = self.kps1[mask == 1]
        self.kps2 = self.kps2[mask == 1]
        return self.kps1, self.kps2

    def compute_bbox_similarities(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        box_list_1: list[np.ndarray] | list[list[int]],
        box_list_2: list[np.ndarray] | list[list[int]],
        w: int,
        h: int,
        normalize_max: int | None = None,
        vis: bool = False,
        box_label_list_1: list[str] | None = None,
        save_vis_path: str | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
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
            vis (bool): Whether to visualize the matched keypoints and bounding boxes.
        Returns:
            tuple[np.ndarray, list[np.ndarray]]: Similarity score matrix between bounding boxes and list of
                kpts masks for each object bounding boxfor the first image.
        """
        kps1, kps2 = self.match(img1, img2)
        sim_matrix = np.zeros((len(box_list_1), len(box_list_2)), dtype=np.float32)
        kpts_mask1_list = []
        if vis or save_vis_path is not None:
            img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        for i, box1 in enumerate(box_list_1):
            x1_min, y1_min, x1_max, y1_max = box1
            if normalize_max is not None:
                x1_min = int(x1_min * w / normalize_max)
                y1_min = int(y1_min * h / normalize_max)
                x1_max = int(x1_max * w / normalize_max)
                y1_max = int(y1_max * h / normalize_max)
            mask1 = (kps1[:, 0] >= x1_min) & (kps1[:, 0] <= x1_max) & (kps1[:, 1] >= y1_min) & (kps1[:, 1] <= y1_max)
            kpts_mask1_list.append(mask1)
            if vis or save_vis_path is not None:
                cv2.rectangle(
                    img1_bgr,
                    (x1_min, y1_min),
                    (x1_max, y1_max),
                    color=(0, 255, 0),
                    thickness=2,
                )

            for j, box2 in enumerate(box_list_2):
                x2_min, y2_min, x2_max, y2_max = box2
                if normalize_max is not None:
                    x2_min = int(x2_min * w / normalize_max)
                    y2_min = int(y2_min * h / normalize_max)
                    x2_max = int(x2_max * w / normalize_max)
                    y2_max = int(y2_max * h / normalize_max)

                # Count keypoints in each bounding box
                mask2 = (
                    (kps2[:, 0] >= x2_min) & (kps2[:, 0] <= x2_max) & (kps2[:, 1] >= y2_min) & (kps2[:, 1] <= y2_max)
                )

                count1 = np.sum(mask1)
                count2 = np.sum(mask2)

                if count1 + count2 > 0:
                    sim_matrix[i, j] = 2 * np.sum(mask1 & mask2) / (count1 + count2)
                if (vis or save_vis_path is not None) and j == 0:
                    cv2.rectangle(
                        img2_bgr,
                        (x2_min, y2_min),
                        (x2_max, y2_max),
                        color=(0, 255, 0),
                        thickness=2,
                    )
        if vis or save_vis_path is not None:
            img2_with_matches = warp_corners_and_draw_matches(kps1, kps2, img1_bgr, img2_bgr)

        for i, box1 in enumerate(box_list_1):
            x1_min, y1_min, x1_max, y1_max = box1
            if normalize_max is not None:
                x1_min = int(x1_min * w / normalize_max)
                y1_min = int(y1_min * h / normalize_max)
                x1_max = int(x1_max * w / normalize_max)
                y1_max = int(y1_max * h / normalize_max)
            if vis or save_vis_path is not None:
                cv2.rectangle(
                    img2_with_matches,
                    (x1_min, y1_min),
                    (x1_max, y1_max),
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.putText(
                    img2_with_matches,
                    box_label_list_1[i] if box_label_list_1 is not None else str(i),
                    (x1_min, y1_min + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        if vis:
            cv2.imshow("Matches", img2_with_matches)
            cv2.waitKey(0)
        if save_vis_path is not None:
            cv2.imwrite(save_vis_path, img2_with_matches)

        return sim_matrix, kpts_mask1_list
