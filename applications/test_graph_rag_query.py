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
from ocr_nav.rag.graph_rag import EmbodiedGraphRAG
from ocr_nav.vlm.qwen3_vl import QWen3VLQueryInterface
from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes, encode_image_to_base64_string
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()

    root_path = Path(args.root_path)
    annotation_dir = root_path / "qwen3vl_annotations"
    os.makedirs(annotation_dir, exist_ok=True)

    graph_rag_path = root_path / "graph_rag.db"
    graph_rag = EmbodiedGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")
    folderio = FolderIO(root_path, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")

    k = 1
    while k != ord("q"):
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text == "q":
            break

        img_ids, box_infos = graph_rag.find_images_by_concept(query_text, top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("computer", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I feel lonely, where should I go?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to go upstairs", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to do some machinery stuff", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I wash my hand?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I find a hanger?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where are shelf with wooden boxes?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to reach the yellow line", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I see green color?", top_k=5)

        for img_id, box_info in zip(img_ids, box_infos):
            img_np = np.array(folderio.get_image(img_id)).astype(np.uint8)
            annotation = folderio.get_annotation(img_id)
            if len(annotation) == 0:
                continue
            img_with_boxes = draw_bounding_boxes_on_image_np(img_np, [box_info], normalize_max=1000)
            cv2.imshow("Retrieved Image", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(0)
            if k == ord("q"):
                break


if __name__ == "__main__":
    main()
