import argparse
import rclpy
import cv2
import numpy as np
from pathlib import Path
from ocr_nav.utils.segmentation_utils import GroundingDinoSamSegmenter
from ocr_nav.utils.io_utils import BagIO


def main():
    config_dir = Path(__file__).parent.parent / "config"
    grounding_dino_config_path = config_dir / "detection" / "grounding_dino.yaml"
    sam_config_path = config_dir / "segmentation" / "segment_anything.yaml"

    parser = argparse.ArgumentParser(description="Bag file path")
    parser.add_argument(
        "--bag_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_factory/rosbag2_2025_12_16-17_09_00_perception_suite_fixed_and_trimmed_1",
    )
    args = parser.parse_args()
    segmenter = GroundingDinoSamSegmenter(
        grounding_dino_config_path=grounding_dino_config_path.as_posix(),
        sam_config_path=sam_config_path.as_posix(),
        device="cuda",
    )

    rclpy.init()
    bag_path = Path(args.bag_path)
    bagio = BagIO(bag_path.as_posix())
    bagio.init_reader()
    while bagio.has_next():
        data = bagio.get_next_sync_data()
        if data is None:
            break
        livox_pc_np, rslidar_pc_np, img_np, livox_pose = data

        mask = segmenter.segment(img_np, text_prompt="ground")
        mask = (mask > 0).astype(np.uint8) * 255
        cv2.imshow("Image", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

    bagio.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
