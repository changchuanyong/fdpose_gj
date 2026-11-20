"""Headless ROS 2 node: Gemini RGB-D in, FoundationPose tracking, PoseStamped out."""

import argparse
from functools import partial
from itertools import product
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(os.environ.get("FDPOSE_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(ROOT / "fdpose_ws/src/fdpose_perception"))
sys.path.insert(0, str(ROOT / "fdpose_ws/src/fdpose_camera"))

import numpy as np

from fdpose_camera.orbbec_ros import OrbbecRosSource, RosRgbdSource
from fdpose_perception.adapters.detector import Sam2Segmenter, TargetDetector
from fdpose_perception.adapters.foundationpose import draw_pose, render_depth
from fdpose_perception.adapters.ros2 import rotation_matrix_to_quaternion
from fdpose_perception.pipeline import (PosePipeline, TrackingSession, UNIT_TO_M,
                                        load_cad_from_target)
from fdpose_perception.recording import Recorder
from fdpose_perception.validation import depth_alignment, validate_pose as validate_pose_result


CHECK_PERIOD_S = 0.6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose", type=Path, default=ROOT / "third_party/FoundationPose")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-units", choices=tuple(UNIT_TO_M), default="mm")
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--mask-backend", choices=("sam2", "yolo-seg"), default="sam2")
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "weights/sam2.1_hiera_small.pt")
    parser.add_argument("--camera", choices=("gemini", "d415"), default="gemini")
    parser.add_argument("--camera-namespace", default="/gemini")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=0, help="Tracking frames after initialization, 0 for continuous")
    parser.add_argument("--init-width", type=int, default=640)
    parser.add_argument("--est-refine-iter", type=int, default=3)
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
    if not (args.foundationpose / "estimater.py").is_file():
        parser.error(f"Project FoundationPose source not found: {args.foundationpose}")
    for path in (args.mesh, args.geometry, args.weights):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.mask_backend == "sam2" and not args.sam2_checkpoint.is_file():
        parser.error(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    return args


def main():
    args = parse_args()
    cad_from_target = load_cad_from_target(args.geometry)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from geometry_msgs.msg import PoseStamped
    import torch
    import trimesh

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output.parent / "ultralytics"))
    os.environ.setdefault("FDONPOSE_TMP", str(args.output / "tmp"))
    sys.path.insert(0, str(args.foundationpose.resolve()))
    from estimater import FoundationPose, set_seed

    logging.getLogger().setLevel(logging.WARNING)
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose requires CUDA")
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_scale(UNIT_TO_M[args.mesh_units])
    corners = np.asarray(list(product(*mesh.bounds.T)))
    detector = TargetDetector(args.weights, args.mask_backend, args.class_id)
    segmenter = None
    if args.mask_backend == "sam2":
        segmenter = Sam2Segmenter(args.sam2_checkpoint, ROOT / "third_party/sam2")
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                               mesh=mesh, debug=0, debug_dir=str(args.output))
    session = TrackingSession(estimator, torch.cuda.synchronize)

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
            message = PoseStamped()
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(frame_stamp_ns, 1_000_000_000)
            message.header.frame_id = frame_id
            message.pose.position.x = float(target_pose[0, 3])
            message.pose.position.y = float(target_pose[1, 3])
            message.pose.position.z = float(target_pose[2, 3])
            message.pose.orientation.x, message.pose.orientation.y, \
                message.pose.orientation.z, message.pose.orientation.w = \
                rotation_matrix_to_quaternion(target_pose[:3, :3])
            publisher.publish(message)

        print(json.dumps(source.metadata), flush=True)

        def validate_pose(pose, detection_box, measured_depth, evaluation_K=None):
            evaluation_K = K if evaluation_K is None else evaluation_K
            return validate_pose_result(corners, pose, detection_box, measured_depth,
                                        evaluation_K, partial(render_depth, estimator), depth_alignment)

        segment_box = segmenter.segment_box if segmenter else None
        recorder = Recorder(args.output, K, cad_from_target, publish=publish)
        pipeline = PosePipeline(estimator, session, detector, segment_box, corners, K, validate_pose,
                                init_width=args.init_width, est_refine_iter=args.est_refine_iter,
                                mask_backend=args.mask_backend, check_period_s=CHECK_PERIOD_S)
        ui = SimpleNamespace(
            read_frame=read_frame, present=lambda bgr, status: True,
            emit=lambda obj: print(json.dumps(obj, flush=True)),
            log=lambda msg: print(msg, flush=True),
            draw=lambda bgr, cad_pose, target_pose: draw_pose(K, bgr, cad_pose, target_pose, mesh.bounds),
            interactive=False, closed=lambda: not rclpy.ok())
        pipeline.run(ui, recorder, args.frames)
        summary = {"camera_serial": source.metadata["serial"], "camera_backend": args.camera,
                   "camera_metadata": source.metadata, "resolution": source.metadata["rgb_resolution"],
                   "initialization_mask": "SAM2.1_hiera_small" if segmenter else "yolo-seg",
                   "class_id": args.class_id, "mesh_units": args.mesh_units,
                   "est_refine_iter": args.est_refine_iter, "pose_topic": args.pose_topic,
                   "register_ms": recorder.init_ms, "registrations": recorder.registrations,
                   "tracked_poses": recorder.tracked_poses, "tracking_checks": recorder.checks,
                   "tracking_recoveries": recorder.recoveries, "tracking_resets": recorder.resets,
                   "tracking_fps": recorder.tracked_poses / recorder.duration if recorder.tracked_poses else None,
                   "calibration_independently_validated": False, "pose_accuracy_validated": False}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()