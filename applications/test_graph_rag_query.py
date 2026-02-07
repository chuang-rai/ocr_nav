import os
from pathlib import Path
from omegaconf import OmegaConf
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
    config_path = Path(__file__).parent.parent / "config" / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    rgb_dir = bag_path.parent / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))

    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = EmbodiedGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")

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
            img_np = np.array(cv2.imread(rgb_paths[img_id].as_posix())).astype(np.uint8)
            img_with_boxes = draw_bounding_boxes_on_image_np(img_np, [box_info], normalize_max=None)
            cv2.imshow("Retrieved Image", img_with_boxes)
            k = cv2.waitKey(0)
            if k == ord("q"):
                break


if __name__ == "__main__":
    main()
