import torch
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import cv2
from ocr_nav.matcher.xfeat_matcher import XFeatMatcher


def main():

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        # default="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rosbag2_2026_02_02-17_46_19_perception_suite_fix_trimmed_glim",
        default="/home/chuang/hcg/projects/ocr/data/ETH_Bauhalle_long_run_1/",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()
    xfeat = XFeatMatcher()

    rgb_dir = Path(args.root_path) / "rgb"
    anno_dir = Path(args.root_path) / "qwen3vl_annotations_fast_8b"
    rgb_paths = sorted(rgb_dir.iterdir())
    anno_paths = sorted(anno_dir.iterdir())
    sim_matrix_dir = Path(args.root_path) / "bbox_similarity_matrices"
    sim_matrix_dir.mkdir(exist_ok=True)
    pbar = tqdm(enumerate(zip(rgb_paths[:-1], anno_paths[:-1])), total=len(rgb_paths) - 1)
    normalize_max = 1000
    for i, (rgb_path, anno_path) in pbar:
        img = cv2.imread(rgb_path.as_posix())
        width, height = img.shape[1], img.shape[0]
        with open(anno_path, "r") as f:
            annotation = json.load(f)
        with open(anno_paths[i + 1], "r") as f:
            next_annotation = json.load(f)
        if "objects" not in annotation or len(annotation["objects"]) == 0:
            continue
        next_img = cv2.imread(rgb_paths[i + 1].as_posix())

        img_time_nano = int(rgb_path.stem.split("_")[-1])
        next_img_time_nano = int(rgb_paths[i + 1].stem.split("_")[-1])
        bbox_list_1 = [obj["bounding_box"] for obj in annotation["objects"]]
        bbox_list_2 = [obj["bounding_box"] for obj in next_annotation["objects"]]
        sim_matrix = xfeat.compute_bbox_similarities(
            img, next_img, bbox_list_1, bbox_list_2, width, height, normalize_max=normalize_max
        )

        sim_matrix_path = sim_matrix_dir / f"sim_matrix_{img_time_nano}_to_{next_img_time_nano}.npy"
        np.save(sim_matrix_path.as_posix(), sim_matrix)


if __name__ == "__main__":
    main()
