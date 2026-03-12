import os
import json
import numpy as np
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2

import yaml
from ocr_nav.vlm.gemini_plus import GeminiPlusQueryInterface
from rai_ai_core_library.utils import dynamic_model

from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes, numpy_img2bytes_pil
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    config_dir = Path(__file__).parent.parent.parent / "config"
    gemini_config_path = config_dir / "llm" / "gemini_plus.yaml"
    with open(gemini_config_path, "r") as file:
        gemini_config = yaml.safe_load(file)
    device = "cuda"
    gemini = dynamic_model(gemini_config, device=device)

    annotator_prompt = (
        "You are visualization coder which use Python OpenCV library to draw bounding boxes around objects in images. "
        "You will receive a JSON file with object descriptions, labels, and normalized bounding box coordinates (values are 0-1000)."
        "Please write a Python script that takes an image array (img_bgr) and the JSON string (json_str) as input and uses OpenCV to draw "
        "the bounding boxes on the image based on the provided JSON data. The code should save the resulting image with bounding boxes drawn on it. "
        "Output only the Python code that can be directly run with eval() without any explanation, additional text, and markdown wrapper. "
    )
    json_path = "/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite/gemini_annotations/gemini_1765901341_903371776.json"
    rgb_path = "/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite/left/image_1765901341_903371776.jpg"
    with open(json_path, "r") as f:
        annotation_data = json.load(f)
    img_bgr = cv2.imread(rgb_path)
    json_str = json.dumps(annotation_data)
    # answers = gemini.query(json_str, annotator_prompt)
    # print(answers)
    # exec(answers)

    critic_history = []
    critic_prompt = (
        "You are a careful and precise observer of images. "
        "Your task is to check what information you can get from the image beyond the provided JSON annotations, "
        "which contain the image descriptions and object bounding boxes, attributes, and labels. "
        "Be creative and insightful in analyzing the image and go beyond object level, and provide any new observations that are not included in the JSON annotations. "
        "Please analyze the image and the JSON annotations, and provide the new observations in JSON format with normalized bounding boxes. "
    )

    # answer = gemini.query(json_str, critic_prompt, encode_image_to_bytes(img_bgr))
    # print(answer)

    google_map_screenshot_path = "/home/chuang/Downloads/google_map_screenshot.jpg"
    google_map_img_bgr = cv2.imread(google_map_screenshot_path)
    answer = gemini.query(
        "What are surrounding the user in the map?",
        "Analyze the image and provide any insights you can get from the image. Also provide the bounding boxes to indicate the source of your insights. "
        "The bounding box format should be [row_min, col_min, row_max, col_max], and the coordinates should be normalized to be between 0 and 1000. "
        "Output the insights and bounding boxes in JSON format without markdown wrapper.",
        encode_image_to_bytes(google_map_img_bgr),
    )

    print(answer)
    # json_path = "/home/chuang/Downloads/google_map_screenshot.jpg"
    rgb_path = "/home/chuang/Downloads/google_map_screenshot.jpg"
    with open(json_path, "r") as f:
        annotation_data = json.load(f)
    img_bgr = cv2.imread(rgb_path)
    json_str = answer
    answers = gemini.query(json_str, annotator_prompt)
    print(answers)
    exec(answers)


if __name__ == "__main__":
    main()
