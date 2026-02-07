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
from ocr_nav.vlm.deepseek_ocr import DeepSeekOCRvLLMQueryInterface
from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes, encode_image_to_base64_string
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    # qwen3_vl = QWen3VLQueryInterface(model_name="Qwen/Qwen3-VL-8B-Instruct")
    deepseek_ocr = DeepSeekOCRvLLMQueryInterface(model_name="deepseek-ai/DeepSeek-OCR")
    # qwen3_vl = QWen3VLQueryInterface(model_name="Qwen/Qwen2.5-VL-3B-Instruct")

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        # default="/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite",
        default="/home/chuang/hcg/projects/ocr/data/giorgio_data/gtonetti_data_zurich_hb_3_merged_pose_zed_gen2_depth_fs_zurich_hb_3_5_bag_zurich_hb_3_5_2",
        help="Root path to the dataset folder",
    )
    parser.add_argument("--sample_every", type=int, default=1, help="Sample every N images for annotation")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for vLLM inference")
    args = parser.parse_args()

    root_path = Path(args.root_path)
    annotation_dir = root_path / "deepseek_ocr_annotations"
    os.makedirs(annotation_dir, exist_ok=True)

    bs = args.batch_size
    folderio = FolderIO(root_path, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")
    pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
    batch_images = []
    for i, timestamp in pbar:
        if i % args.sample_every != 0:
            continue
        img = folderio.get_image(i, prefix="").convert("RGB")
        img_np = np.array(img).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # prompt = "<image>\nIdentify all objects in the image and output them in bounding boxes."
        # prompt = "<image>\nFree OCR."
        prompt = "<image>\n<|grounding|>OCR this image. "
        batch_images.append((i, timestamp, prompt, img, img_bgr))
        if len(batch_images) == bs:
            responses = deepseek_ocr.batch_query([item[2] for item in batch_images], [item[3] for item in batch_images])
            for response_i, response in enumerate(responses):
                print(response)

                # img_bgr = draw_bounding_boxes_on_image_np(
                #     batch_images[response_i][4], data.get("objects", []), normalize_max=1000
                # )

                # cv2.imshow("Image", img_bgr)
                # cv2.waitKey(1)
                # annotation_path = annotation_dir / f"qwen3vl_{batch_images[response_i][1]}.json"
                # with open(annotation_path, "w") as f:
                #     json.dump(data, f, indent=4)
            batch_images = []
        pbar.set_description(f"Timestamp: {timestamp}")


if __name__ == "__main__":
    main()
