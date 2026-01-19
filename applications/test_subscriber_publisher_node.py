import rclpy
from ocr_nav.utils.io_utils import SubscriberIO


def main():
    rclpy.init()
    node = SubscriberIO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
