"""Desktop CLI: track a configured part with D415 or Gemini RGB-D via FoundationPose."""

import argparse
from contextlib import ExitStack
from functools import partial
from itertools import product
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FDPOSE_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / "src/fdpose_perception"))
sys.path.insert(0, str(CODE_ROOT / "src/fdpose_camera"))
from fdpose_camera.rgbd import open_d415
from fdpose_perception.adapters.detector import Sam2Segmenter, TargetDetector
from fdpose_perception.adapters.foundationpose import draw_pose, render_depth
from fdpose_perception.adapters.ros2 import PosePublisher
from fdpose_perception.pipeline import PosePipeline, TrackingSession, UNIT_TO_M, load_cad_from_target
from fdpose_perception.preview import GeminiPreview, PreviewWindow
from fdpose_perception.recording import Recorder
from fdpose_perception.validation import depth_alignment, validate_pose as validate_pose_result


CHECK_PERIOD_S = 0.6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundationpose", type=Path, default=ROOT / "third_party/FoundationPose")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-units", choices=tuple(UNIT_TO_M), default="mm")
    parser.add_argument("--geometry", type=Path, required=True, help="CAD-to-target transform JSON")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO weights for the selected backend")
    parser.add_argument("--class-id", type=int, default=0, help="Target class ID in YOLO weights")
    parser.add_argument("--mask-backend", choices=("sam2", "yolo-seg"), default="sam2")
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "weights/sam2.1_hiera_small.pt")
    parser.add_argument("--camera", choices=("d415", "orbbec-ros"), default="d415")
    parser.add_argument("--camera-namespace", default="/gemini", help="ROS 2 namespace for Orbbec")
    parser.add_argument("--serial", required=True, help="Expected camera serial number")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60, help="Tracking frames after initialization, 0 for continuous")
    parser.add_argument("--stream-width", type=int, default=640, help="D415 stream width")
    parser.add_argument("--stream-height", type=int, default=480, help="D415 stream height")
    parser.add_argument("--init-width", type=int, default=640,
                        help="Downscale RGB, depth and mask for registration to this width")
    parser.add_argument("--est-refine-iter", type=int, default=3,
                        help="Registration refinement iterations")
    parser.add_argument("--show", action="store_true", help="Show live RGB and pose overlays on the Jetson desktop")
    parser.add_argument("--ros-pose-topic", help="Publish target poses as geometry_msgs/PoseStamped")
    args = parser.parse_args(argv)
    if args.class_id < 0:
        parser.error("--class-id must be nonnegative")
    if args.frames < 0:
        parser.error("--frames must be nonnegative")
    if args.stream_width <= 0 or args.stream_height <= 0 or args.init_width <= 0:
        parser.error("--stream-width, --stream-height and --init-width must be positive")
    if args.est_refine_iter < 1:
        parser.error("--est-refine-iter must be positive")
    if args.ros_pose_topic is not None and not args.ros_pose_topic.startswith('/'):
        parser.error("--ros-pose-topic must be an absolute ROS topic")
    if not (args.foundationpose / "estimater.py").is_file():
        parser.error(f"Project FoundationPose source not found: {args.foundationpose}")
    if args.show and not os.environ.get("DISPLAY"):
        parser.error("--show needs a desktop display. Run from a terminal on the Jetson desktop.")
    for path in (args.mesh, args.geometry, args.weights):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.mask_backend == "sam2" and not args.sam2_checkpoint.is_file():
        parser.error(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
    return args


def main():
    args = parse_args()
    cad_from_target = load_cad_from_target(args.geometry)
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
    with ExitStack() as stack:
        if args.camera == "orbbec-ros":
            from fdpose_camera.orbbec_ros import open_orbbec_ros

            camera = stack.enter_context(open_orbbec_ros(args.serial, args.camera_namespace))
        else:
            camera = stack.enter_context(open_d415(args.serial, args.stream_width, args.stream_height))
        live_preview = None
        if args.show and args.camera == "orbbec-ros":
            live_preview = GeminiPreview(camera.metadata["camera"])
        preview = live_preview or (PreviewWindow(camera.metadata["camera"]) if args.show else None)
        if preview:
            stack.callback(preview.close)
        K = camera.K
        frame_stamp_ns = None
        pose_publisher = None
        if args.ros_pose_topic:
            frame_id = camera.metadata.get("frame_id", "camera_color_optical_frame")
            pose_publisher = PosePublisher(args.ros_pose_topic, frame_id)
            stack.callback(pose_publisher.close)

        def read_frame():
            nonlocal frame_stamp_ns
            frame = camera.read_frame()
            if pose_publisher:
                frame_stamp_ns = camera.ros_stamp_ns() if camera.ros_stamp_ns else pose_publisher.now_ns()
            if live_preview:
                live_preview.source = frame[2:4]
            return frame
        print(json.dumps(camera.metadata), flush=True)

        def validate_pose(pose, detection_box, measured_depth, evaluation_K=None):
            evaluation_K = K if evaluation_K is None else evaluation_K
            return validate_pose_result(corners, pose, detection_box, measured_depth,
                                        evaluation_K, partial(render_depth, estimator), depth_alignment)

        segment_box = segmenter.segment_box if segmenter else None
        publish = (lambda target_pose: pose_publisher.publish(target_pose, frame_stamp_ns)) if pose_publisher else None
        recorder = Recorder(args.output, K, cad_from_target, publish=publish)
        pipeline = PosePipeline(estimator, session, detector, segment_box, corners, K, validate_pose,
                                init_width=args.init_width, est_refine_iter=args.est_refine_iter,
                                mask_backend=args.mask_backend, check_period_s=CHECK_PERIOD_S)
        ui = SimpleNamespace(
            read_frame=read_frame,
            present=lambda bgr, status: preview.present(bgr, status) if preview else True,
            emit=lambda obj: print(json.dumps(obj, flush=True)),
            log=lambda msg: print(msg, flush=True),
            draw=lambda bgr, cad_pose, target_pose: draw_pose(K, bgr, cad_pose, target_pose, mesh.bounds),
            interactive=bool(preview),
            closed=lambda: preview.closed if preview else False)
        pipeline.run(ui, recorder, args.frames)
        summary = {"camera_serial": camera.metadata["serial"], "camera_backend": args.camera,
                   "camera_metadata": camera.metadata,
                   "resolution": camera.metadata["rgb_resolution"],
                   "requested_stream_fps": 30 if args.camera == "d415" else None,
                   "initialization_mask": "SAM2.1_hiera_small" if segmenter else "yolo-seg",
                   "class_id": args.class_id, "mesh_units": args.mesh_units,
                   "est_refine_iter": args.est_refine_iter,
                   "register_ms": recorder.init_ms, "registrations": recorder.registrations,
                   "tracked_poses": recorder.tracked_poses,
                   "tracking_checks": recorder.checks, "tracking_recoveries": recorder.recoveries,
                   "tracking_resets": recorder.resets,
                   "tracking_fps": recorder.tracked_poses / recorder.duration if recorder.tracked_poses else None,
                   "statistics_sample_frames": len(recorder.times),
                   "tracking_latency_p50_ms": float(np.median(recorder.times)) if recorder.times else None,
                   "tracking_latency_p95_ms": float(np.percentile(recorder.times, 95)) if recorder.times else None,
                   "position_std_xyz_mm": (np.std(recorder.translations, axis=0) * 1000).tolist() if recorder.translations else None,
                   "last_position_xyz_m": recorder.translations[-1].tolist() if recorder.translations else None,
                   "calibration_independently_validated": False, "pose_accuracy_validated": False}
        if live_preview:
            summary.update(live_preview.measurements())
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()