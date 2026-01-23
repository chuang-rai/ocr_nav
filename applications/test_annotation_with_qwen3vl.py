import os
from pathlib import Path
import json
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np
from ocr_nav.vlm.qwen3_vl import QWen3VLQueryInterface
from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes, encode_image_to_base64_string
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    qwen3_vl = QWen3VLQueryInterface(model_name="Qwen/Qwen3-VL-8B-Instruct-FP8")

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite",
        help="Root path to the dataset folder",
    )
    parser.add_argument("--sample_every", type=int, default=10, help="Sample every N images for annotation")
    args = parser.parse_args()

    root_path = Path(args.root_path)
    annotation_dir = root_path / "qwen3vl_annotations"
    os.makedirs(annotation_dir, exist_ok=True)

    folderio = FolderIO(root_path, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")
    pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
    for i, timestamp in pbar:
        if i % args.sample_every != 0:
            continue
        print(timestamp)
        img_np = np.array(folderio.get_image(i)).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        prompt = (
            "Describe the scene in the image based on the highlighted areas in the mask. "
            "Also draw bounding boxes around any objects of interest. "
            "Please normalize the coordinates to be between 0 and 1000. "
            "The output format should be in JSON:"
            """
        {
            "description": "A brief description of the scene.",
            "objects": [
                {
                    "label": "object label",
                    "bounding_box": [x_min, y_min, x_max, y_max],
                    "attributes": {
                        "color": "color of the object",
                        "material": "material of the object",
                        "function": "function or use of the object",
                        ...
                    }
                },
                ...
        }
        """
        )
        response = qwen3_vl.query(prompt, encode_image_to_base64_string(img_bgr))
        print(response)
        response = response.replace("```json", "").replace("```", "")
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for timestamp {timestamp}. Skipping...")
            continue

        img_bgr = draw_bounding_boxes_on_image_np(img_bgr, data.get("objects", []), normalize_max=1000)

        cv2.imshow("Image", img_bgr)
        cv2.waitKey(1)
        pbar.set_description(f"Timestamp: {timestamp}")
        annotation_path = annotation_dir / f"qwen3vl_{timestamp}.json"
        with open(annotation_path, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
