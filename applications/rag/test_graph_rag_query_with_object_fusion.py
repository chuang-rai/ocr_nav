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
from ocr_nav.rag.graph_rag import BaseGraphRAG
from ocr_nav.utils.rag_utils import visualize_graphrag, visualize_graphrag_query_result
from ocr_nav.vlm.qwen3_vl import QWen3VLQueryInterface
from ocr_nav.utils.io_utils import FolderIO, encode_image_to_bytes, encode_image_to_base64_string
from ocr_nav.utils.visualization_utils import draw_bounding_boxes_on_image_np


def main():
    config_path = Path(__file__).parent.parent.parent / "config" / "floor_graph_config.yaml"
    args = OmegaConf.load(config_path.as_posix())
    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    rgb_dir = bag_path.parent / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))

    graph_rag_path = bag_path.parent / "graph_rag_new"
    graph_rag = BaseGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")
    # visualize_graphrag(graph_rag, graph_rag_path.parent)

    graph_rag.build_node_index("Object", "embedding", metric="cosine", mu=16, efc=200)
    k = 1
    while k != ord("q"):
        query_text = input("Enter your query (or 'q' to quit): ")
        if query_text == "q":
            break

        obj_score_tuples = graph_rag.retrieve_node_and_score_by_query("Object", query_text, "embedding", top_k=20)
        obj_nodes = [x[0] for x in obj_score_tuples]
        scores = [x[1] for x in obj_score_tuples]
        print([x["labels"] for x in obj_nodes])
        print(scores)
        visualize_graphrag_query_result(graph_rag, bag_path.parent, query_text, obj_nodes)

        # for obj_node, distance in obj_nodes:
        #     print(f"Retrieved Object node {obj_node['id']} with label '{obj_node['label']}' and distance {distance}")
        #     obj_id = obj_node["id"]
        #     related_src_rel_tar_tuples = graph_rag.retrieve_related_nodes_with_tar_node("Object", obj_id)
        #     if len(related_src_rel_tar_tuples) == 0:
        #         print(f"No related Bbox nodes found for Object node {obj_id}")
        #         continue
        #     bbox_list = []
        #     label = obj_node["label"]
        #     for id, related_tuple in enumerate(related_src_rel_tar_tuples):

        #         bbox = related_tuple[0]
        #         bbox_id = bbox["id"]
        #         bbox_data = bbox["bbox"]

        #         related_src_rel_tar_tuples = graph_rag.retrieve_related_nodes_with_tar_node("Bbox", bbox_id)
        #         if len(related_src_rel_tar_tuples) == 0:
        #             print(f"No related Frame nodes found for Bbox node {bbox_id}")
        #             continue
        #         frame_id = related_src_rel_tar_tuples[0][0]["id"]

        #         img_np = np.array(cv2.imread(rgb_paths[frame_id].as_posix())).astype(np.uint8)
        #         img_with_boxes = draw_bounding_boxes_on_image_np(
        #             img_np, [{"bounding_box": bbox_data, "label": label}], normalize_max=None
        #         )
        #         cv2.imshow(f"Associated Bbox {id}", img_with_boxes)
        #     k = cv2.waitKey(0)
        #     if k == ord("q"):
        #         break

        # img_ids, box_infos = graph_rag.find_images_by_concept("computer", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I feel lonely, where should I go?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to go upstairs", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to do some machinery stuff", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I wash my hand?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I find a hanger?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where are shelf with wooden boxes?", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("I want to reach the yellow line", top_k=5)
        # img_ids, box_infos = graph_rag.find_images_by_concept("Where can I see green color?", top_k=5)

        # for img_id, box_info in zip(img_ids, box_infos):
        #     img_np = np.array(cv2.imread(rgb_paths[img_id].as_posix())).astype(np.uint8)
        #     img_with_boxes = draw_bounding_boxes_on_image_np(img_np, [box_info], normalize_max=None)
        #     cv2.imshow("Retrieved Image", img_with_boxes)
        #     k = cv2.waitKey(0)
        #     if k == ord("q"):
        #         break


if __name__ == "__main__":
    main()
