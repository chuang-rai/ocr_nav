import os
from pathlib import Path
import json
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import rclpy
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np
from ocr_nav.vlm.qwen3_vl import QWen3VLQueryInterface, QWen3VLvLLMQueryInterface
from ocr_nav.utils.io_utils import BagIO, encode_image_to_bytes, encode_image_to_base64_string
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    # qwen3_vl = QWen3VLQueryInterface(model_name="Qwen/Qwen3-VL-8B-Instruct")
    # qwen3_vl = QWen3VLvLLMQueryInterface(model_name="Qwen/Qwen3-VL-4B-Instruct")
    qwen3_vl = QWen3VLvLLMQueryInterface(model_name="Qwen/Qwen3-VL-8B-Instruct")
    # qwen3_vl = QWen3VLQueryInterface(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--bag_path",
        type=str,
        # default="/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rosbag2_2026_02_02-17_46_19_perception_suite_fix_trimmed_glim",
        default="/home/chuang/hcg/projects/ocr/data/ETH_Bauhalle_long_run_1/rosbag2_2026_01_21-15_13_10_perception_suite_fixed_and_trimmed_glim_tf_static_fix",
        help="Root path to the ROS Bag file (folder)",
    )
    parser.add_argument("--sample_every", type=int, default=1, help="Sample every N images for annotation")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for vLLM inference")
    args = parser.parse_args()

    root_path = Path(args.bag_path)
    annotation_dir = root_path / "qwen3vl_annotations_fast_8b"
    rgb_dir = root_path / "rgb"
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    bs = args.batch_size
    rclpy.init()
    bagio = BagIO(root_path.as_posix(), sample_every=args.sample_every)
    bagio.init_reader()
    batch_images = []
    pi = 0
    while bagio.has_next():

        data = bagio.get_next_sync_data()
        if data is None:
            continue
        pc_livox, pc_rslidar, img_np, livox_pose, t_nanosec = data
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
        new_prompt = "detect everything in the image."
        "Also draw bounding boxes around any objects of interest. "
        "Please normalize the coordinates to be between 0 and 1000. "
        "The output format should be in JSON:"
        """
        {
            "objects": [
                {"bounding_box": [x_min, y_min, x_max, y_max]},
                ...
            ]
        }
        """
        image_bytes = encode_image_to_base64_string(img_bgr)
        batch_images.append((pi, t_nanosec, prompt, image_bytes, img_bgr))
        if len(batch_images) == bs:
            responses = qwen3_vl.batch_query([item[2] for item in batch_images], [item[3] for item in batch_images])
            for response_i, response in enumerate(responses):
                response = response.replace("```json", "").replace("```", "")
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON for timestamp {batch_images[response_i][1]}. Skipping...")
                    print(response)
                    continue

                img_bgr = draw_bounding_boxes_on_image_np(
                    batch_images[response_i][4], data.get("objects", []), normalize_max=1000
                )

                cv2.imshow("Image", img_bgr)
                cv2.waitKey(1)
                annotation_path = annotation_dir / f"qwen3vl_{batch_images[response_i][1]}.json"
                with open(annotation_path, "w") as f:
                    json.dump(data, f, indent=4)

                rgb_path = rgb_dir / f"rgb_{batch_images[response_i][1]}.png"
                cv2.imwrite(rgb_path.as_posix(), batch_images[response_i][4])
            batch_images = []
        pi += 1

    bagio.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
