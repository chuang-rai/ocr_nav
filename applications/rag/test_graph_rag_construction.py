import os
from pathlib import Path
from tqdm import tqdm
import argparse
from ocr_nav.rag.graph_rag import SimpleObjectFrameGraphRAG
from ocr_nav.utils.io_utils import FolderIO


def main():

    parser = argparse.ArgumentParser(description="Ground Mesh Construction with Folder")
    parser.add_argument(
        "--root_path",
        type=str,
        # default="/home/chuang/hcg/projects/ocr/data/eth_extracted_sync/rosbag2_2025_12_16-17_09_00_perception_suite",
        default="",
        help="Root path to the dataset folder",
    )
    args = parser.parse_args()

    root_path = Path(args.root_path)
    annotation_dir = root_path / "qwen3vl_annotations"
    os.makedirs(annotation_dir, exist_ok=True)

    graph_rag_path = root_path / "graph_rag.db"
    graph_rag = SimpleObjectFrameGraphRAG(graph_rag_path.as_posix(), embedding_model_name="BAAI/bge-m3")

    folderio = FolderIO(root_path, depth_name="", camera_pose_name="", mask_name="masks_gd_sam2_s")
    pbar = tqdm(enumerate(folderio.timestamp_list), total=folderio.len)
    for i, timestamp in pbar:
        annotation = folderio.get_annotation(i)
        if len(annotation) == 0:
            continue
        graph_rag.ingest_json_frame(i, timestamp, annotation["description"], annotation["objects"])


if __name__ == "__main__":
    main()
