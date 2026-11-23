"""Headless ROS 2 node: Gemini RGB-D in, FoundationPose tracking, PoseStamped out."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("FDPOSE_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(ROOT / "src/fdpose_perception"))
sys.path.insert(0, str(ROOT / "src/fdpose_camera"))

from fdpose_camera.orbbec_ros import OrbbecRosSource, RosRgbdSource
from fdpose_perception.adapters.foundationpose import draw_pose
from fdpose_perception.adapters.ros2 import fill_pose_stamped
from fdpose_perception.frontend import (add_common_args, bootstrap, build_summary, make_pipeline,
                                        make_ui, make_validate, validate_inputs, write_summary)
from fdpose_perception.pipeline import load_cad_from_target
from fdpose_perception.recording import Recorder


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, ROOT, camera_choices=("gemini", "d415"), camera_default="gemini",
                    frames_default=0)
    parser.add_argument("--pose-topic", default="/fdpose/pose")
    args = parser.parse_args(argv)
    if args.class_id < 0:
        parser.error("--class-id must be nonnegative")
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if args.init_width <= 0:
        parser.error("--init-width must be positive")
    if args.est_refine_iter < 1:
        parser.error("--est-refine-iter must be positive")
    if not args.pose_topic.startswith('/'):
        parser.error("--pose-topic must be an absolute ROS topic")
    validate_inputs(parser, args)
    return args


def main():
    args = parse_args()
    cad_from_target = load_cad_from_target(args.geometry)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import PoseStamped

    model = bootstrap(args, ROOT)
    rclpy.init()
    node = rclpy.create_node("fdpose_perception", enable_rosout=False)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    publisher = node.create_publisher(PoseStamped, args.pose_topic, 10)
    try:
        if args.camera == "gemini":
            source = OrbbecRosSource(node, executor, args.serial, args.camera_namespace)
        else:
            source = RosRgbdSource(node, executor, args.serial, args.camera_namespace, "RealSense D415")
        K = source.K
        frame_id = source.metadata["frame_id"]
        frame_stamp_ns = 0

        def read_frame():
            nonlocal frame_stamp_ns
            frame = source.read_frame()
            frame_stamp_ns = source.ros_stamp_ns()
            return frame

        def publish(target_pose):
            publisher.publish(fill_pose_stamped(PoseStamped(), target_pose, frame_stamp_ns, frame_id))

        print(json.dumps(source.metadata), flush=True)

        recorder = Recorder(args.output, K, cad_from_target, publish=publish)
        pipeline = make_pipeline(model, args, K, make_validate(model.corners, model.estimator, K))
        ui = make_ui(
            read_frame=read_frame, present=lambda bgr, status: True,
            draw=lambda bgr, cad_pose, target_pose: draw_pose(K, bgr, cad_pose, target_pose, model.mesh.bounds),
            interactive=False, closed=lambda: not rclpy.ok())
        pipeline.run(ui, recorder, args.frames)
        write_summary(args.output, build_summary(args, source.metadata, recorder, model.segmenter,
                                                 extra={"pose_topic": args.pose_topic}))
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()