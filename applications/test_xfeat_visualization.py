import torch
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import cv2


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2_with_corners, keypoints2, matches, None, matchColor=(0, 255, 0), flags=2
    )

    return img_matches


def main():

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        # default="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/",
        default="/home/chuang/hcg/projects/ocr/data/ETH_Bauhalle_long_run_1/",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()
    xfeat = torch.hub.load("verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096)

    rgb_dir = Path(args.root_path) / "rgb"
    anno_dir = Path(args.root_path) / "qwen3vl_annotations_fast_8b"
    rgb_paths = sorted(rgb_dir.iterdir())
    anno_paths = sorted(anno_dir.iterdir())
    pbar = tqdm(enumerate(zip(rgb_paths[:-1], anno_paths[:-1])), total=len(rgb_paths) - 1)
    normalize_max = 1000
    for i, (rgb_path, anno_path) in pbar:
        img = cv2.imread(rgb_path.as_posix())
        width, height = img.shape[1], img.shape[0]
        with open(anno_path, "r") as f:
            annotation = json.load(f)
        if "objects" not in annotation or len(annotation["objects"]) == 0:
            continue
        next_img = cv2.imread(rgb_paths[i + 1].as_posix())
        img_cp, img_cp_next = img.copy(), next_img.copy()
        # img_cp = draw_bounding_boxes_on_image_np(img_cp, annotation["objects"], normalize_max=1000)
        # cv2.imshow("Current Image with Annotations", img_cp)
        # cv2.waitKey(1)

        for obj in annotation["objects"]:
            bbox = obj["bounding_box"]
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min = int(x_min / normalize_max * width), int(y_min / normalize_max * height)
            x_max, y_max = int(x_max / normalize_max * width), int(y_max / normalize_max * height)

            # mkpts_0, mkpts_1 = xfeat.match_xfeat(img[y_min:y_max, x_min:x_max, :3], next_img, top_k=4096)
            mkpts_0, mkpts_1 = xfeat.match_xfeat(img, next_img, top_k=4096)
            # mkpts_0 += np.array([x_min, y_min])  # Adjust keypoints to original image coordinates
            canvas_bgr = warp_corners_and_draw_matches(mkpts_0, mkpts_1, img, next_img)
            canvas_bgr = cv2.rectangle(
                canvas_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2
            )  # Draw original bbox in blue
            cv2.imshow("Matches with Warped Corners", canvas_bgr)
            key = cv2.waitKey()


if __name__ == "__main__":
    main()
