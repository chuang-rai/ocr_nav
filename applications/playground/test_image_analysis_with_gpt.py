import os
import json
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2

import yaml
from rai_ai_core_library.utils import dynamic_model

from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    config_dir = Path(__file__).parent.parent / "config"
    gpt_config_path = config_dir / "llm" / "gpt.yaml"
    with open(gpt_config_path, "r") as file:
        gpt_config = yaml.safe_load(file)
    device = "cuda"
    gpt = dynamic_model(gpt_config, device=device)

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()

    root_path = Path(args.root_path)
    folderio = FolderIO(root_path, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")
    pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
    for i, timestamp in pbar:
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
        response = gpt.query(prompt, encode_image_to_bytes(img_bgr))
        print(response)
        response = response.replace("```json", "").replace("```", "")
        data = json.loads(response)
        img_bgr = draw_bounding_boxes_on_image_np(img_bgr, data.get("objects", []), normalize_max=1000)

        cv2.imshow("Image", img_bgr)
        cv2.waitKey()
        pbar.set_description(f"Timestamp: {timestamp}")
        break


if __name__ == "__main__":
    main()
