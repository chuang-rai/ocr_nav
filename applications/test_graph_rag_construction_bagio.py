import os
import json
from turtle import st
import numpy as np
import rclpy
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import argparse
from ocr_nav.rag.graph_rag import EmbodiedGraphRAG
from ocr_nav.utils.io_utils import FolderIO, BagIO


def compute_object_embedding_with_annotation(annotation: dict, embedding_model) -> np.ndarray:
    obj = annotation["label"]
    attrs = annotation.get("attributes", {})
    color = attrs.get("color", "")
    material = attrs.get("material", "")
    material_str = f"made of {material}" if material else ""
    function = attrs.get("function", "")
    function_str = f" for {function}" if function else ""
    narrative = f"A {color} {obj} {material_str}{function_str}."
    embedding = embedding_model.encode(narrative)
    return embedding


def main():

    config_path = Path(__file__).parent.parent / "config" / "floor_graph_config.yaml"

    args = OmegaConf.load(config_path.as_posix())

    bag_path = Path(args.bag_path)
    annotation_dir = bag_path.parent / "qwen3vl_annotations_fast_8b"
    rgb_dir = bag_path.parent / "rgb"
    rgb_paths = sorted(list(rgb_dir.iterdir()))

    anno_paths = sorted(list(annotation_dir.iterdir()))
    timelist = [int(anno_path.stem.split("_")[1]) for anno_path in anno_paths]

    os.makedirs(annotation_dir, exist_ok=True)
    rclpy.init()
    bagio = BagIO(
        bag_path.as_posix(),
        rgb_topic=args.bagio.rgb_topic,
        camera_info_topic=args.bagio.camera_info_topic,
        camera_frame_id=args.bagio.camera_frame_id,
        anchor_lidar_id=args.bagio.anchor_lidar_id,
        lidar_topic_list=args.bagio.lidar_topic_list,
        lidar_frame_ids=args.bagio.lidar_frame_ids,
        world_frame_id=args.bagio.world_frame_id,
        max_bag_total_time=args.bagio.max_bag_total_time_sec,
        sample_every=args.bagio.sample_every,
    )
    bagio.init_reader()

    graph_rag_path = bag_path.parent / "graph_rag"
    graph_rag = EmbodiedGraphRAG(graph_rag_path.as_posix(), overwrite=True, embedding_model_name="BAAI/bge-m3")
    graph_rag.define_node_type(
        "Object", {"id": int, "label": str, "embedding": (float, 1024), "attributes": [str], "bbox": (int, 4)}
    )
    graph_rag.define_node_type("Frame", {"id": int, "timestamp": str, "caption": str, "pose": (float, 16)})
    graph_rag.define_relationship_type("FrameContainsObject", "Frame", "Object", {"bbox": (int, 4)})

    i = 0
    while bagio.has_next():
        data = bagio.get_next_sync_data()
        lidar_list, img_np, anchor_lidar_pose, t_nanosec = data
        w, h = img_np.shape[1], img_np.shape[0]
        if t_nanosec not in timelist:
            continue
        id = timelist.index(t_nanosec)
        with open(anno_paths[id], "r") as f:
            annotation = json.load(f)

        timestamp = str(t_nanosec)
        graph_rag.add_node(
            "Frame",
            {
                "id": id,
                "timestamp": timestamp,
                "caption": annotation.get("description", ""),
                "pose": anchor_lidar_pose.flatten().tolist(),
            },
        )
        for obj_ann in annotation["objects"]:
            embedding = compute_object_embedding_with_annotation(obj_ann, graph_rag.embedding_model)
            embedding_list = embedding.tolist()
            attrs_list = [f"{k}: {v}" for k, v in obj_ann["attributes"].items()]
            bbox = np.array(obj_ann.get("bounding_box", []))
            bbox[0] = int(bbox[0] * w / 1000)
            bbox[1] = int(bbox[1] * h / 1000)
            bbox[2] = int(bbox[2] * w / 1000)
            bbox[3] = int(bbox[3] * h / 1000)
            graph_rag.add_node(
                "Object",
                {
                    "id": graph_rag.obj_id,
                    "label": obj_ann["label"],
                    "embedding": embedding_list,
                    "attributes": attrs_list,
                    "bbox": bbox.tolist(),
                },
            )
            graph_rag.add_relationship(
                "FrameContainsObject",
                "Frame",
                id,
                "Object",
                graph_rag.obj_id,
                {"bbox": bbox.tolist()},
            )
            graph_rag.obj_id += 1

    # graph_rag.ingest_json_frame(i, timestamp, annotation["description"], annotation["objects"])
    i += 1


if __name__ == "__main__":
    main()
