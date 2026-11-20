"""Publish RealSense D415 aligned RGB-D as ROS topics (host-read-time stamps)."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("FDPOSE_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(ROOT / "src/fdpose_camera"))

import numpy as np

from fdpose_camera.rgbd import open_d415


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--namespace", default="/d415")
    parser.add_argument("--frame-id", default="camera_color_optical_frame")
    parser.add_argument("--stream-width", type=int, default=640)
    parser.add_argument("--stream-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("fdpose_d415", enable_rosout=False)
    namespace = '/' + args.namespace.strip('/')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    color_pub = node.create_publisher(Image, namespace + "/color/image_raw", qos)
    depth_pub = node.create_publisher(Image, namespace + "/depth/image_raw", qos)
    color_info_pub = node.create_publisher(CameraInfo, namespace + "/color/camera_info", qos)
    depth_info_pub = node.create_publisher(CameraInfo, namespace + "/depth/camera_info", qos)
    try:
        with open_d415(args.serial, args.stream_width, args.stream_height, args.fps) as stream:
            K = stream.K
            distortion = np.asarray(stream.metadata.get("dist_coeffs", []), dtype=float)
            width, height = stream.metadata["rgb_resolution"]
            while rclpy.ok():
                rgb, depth_m, _frame_number, _device_ms = stream.read_frame()
                stamp = node.get_clock().now().to_msg()
                depth_mm = np.rint(depth_m * 1000).astype(np.uint16)

                color = Image()
                color.header.stamp = stamp
                color.header.frame_id = args.frame_id
                color.height, color.width = rgb.shape[:2]
                color.encoding = "rgb8"
                color.is_bigendian = False
                color.step = rgb.shape[1] * 3
                color.data = rgb.tobytes()

                depth = Image()
                depth.header.stamp = stamp
                depth.header.frame_id = args.frame_id
                depth.height, depth.width = depth_mm.shape
                depth.encoding = "16UC1"
                depth.is_bigendian = False
                depth.step = depth_mm.shape[1] * 2
                depth.data = depth_mm.tobytes()

                info = CameraInfo()
                info.header.stamp = stamp
                info.header.frame_id = args.frame_id
                info.height, info.width = height, width
                info.distortion_model = "plumb_bob"
                info.d = distortion.tolist()
                info.k = K.ravel().tolist()
                info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.p = [K[0, 0], 0.0, K[0, 2], 0.0,
                          0.0, K[1, 1], K[1, 2], 0.0,
                          0.0, 0.0, 1.0, 0.0]

                color_pub.publish(color)
                depth_pub.publish(depth)
                color_info_pub.publish(info)
                depth_info_pub.publish(info)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()