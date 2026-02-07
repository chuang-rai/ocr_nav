import os
import cv2
import rclpy
import argparse
from pathlib import Path
from ocr_nav.utils.io_utils import BagIO


def main():
    parser = argparse.ArgumentParser(description="Bag file path")
    parser.add_argument(
        "--bag_path",
        type=str,
        default="/home/chuang/hcg/projects/ocr/data/eth_factory/rosbag2_2025_12_16-17_09_00_perception_suite_fixed_and_trimmed_1",
    )
    args = parser.parse_args()
    rclpy.init()
    bag_path = Path(args.bag_path)
    bagio = BagIO(bag_path.as_posix())
    bagio.init_reader()
    while bagio.has_next():
        data = bagio.get_next_sync_data()
        if data is None:
            continue
        pc_livox, pc_rslidar, img_np, livox_pose, t_nanosec = data
        cv2.imshow("Image", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    bagio.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
