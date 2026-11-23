"""Desktop CLI: track a configured part with D415 or Gemini RGB-D via FoundationPose."""

import argparse
from contextlib import ExitStack
from itertools import product
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("FDPOSE_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / "src/fdpose_perception"))
from fdpose_perception.camera import open_d415
from fdpose_perception.adapters.detector import Sam2Segmenter, TargetDetector
from fdpose_perception.adapters.foundationpose import draw_pose, make_validate
from fdpose_perception.adapters.ros2 import PosePublisher
from fdpose_perception.tracking import PosePipeline, TrackingSession, UNIT_TO_M, load_cad_from_target
from fdpose_perception.preview import make_preview
from fdpose_perception.recording import Recorder, write_json


def add_common_args(parser, root):
    parser.add_argument("--foundationpose", type=Path, default=root / "third_party/FoundationPose")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-units", choices=tuple(UNIT_TO_M), default="mm")
    parser.add_argument("--geometry", type=Path, required=True, help="CAD-to-target transform JSON")
    parser.add_argument("--weights", type=Path, required=True, help="YOLO weights for the selected backend")
    parser.add_argument("--class-id", type=int, default=0, help="Target class ID in YOLO weights")
    parser.add_argument("--mask-backend", choices=("sam2", "yolo-seg"), default="sam2")
    parser.add_argument("--sam2-checkpoint", type=Path, default=root / "weights/sam2.1_hiera_small.pt")
    parser.add_argument("--camera", choices=("d415", "orbbec-ros"), default="d415")
    parser.add_argument("--camera-namespace", default="/gemini", help="ROS 2 namespace for Orbbec")
    parser.add_argument("--serial", required=True, help="Expected camera serial number")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60,
                        help="Tracking frames after initialization, 0 for continuous")
    parser.add_argument("--init-width", type=int, default=640,
                        help="Downscale RGB, depth and mask for registration to this width")
    parser.add_argument("--est-refine-iter", type=int, default=3,
                        help="Registration refinement iterations")


def validate_inputs(parser, args):
    """File-existence checks; numeric checks stay in the CLI."""
    if not (args.foundationpose / "estimater.py").is_file():
        parser.error(f"Project FoundationPose source not found: {args.foundationpose}")
    for path in (args.mesh, args.geometry, args.weights):
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
    if args.mask_backend == "sam2" and not args.sam2_checkpoint.is_file():
        parser.error(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")


def bootstrap(args, root):
    """Prepare the output directory and build the mesh, detector, estimator and session."""
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
        segmenter = Sam2Segmenter(args.sam2_checkpoint, root / "third_party/sam2")
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                               mesh=mesh, debug=0, debug_dir=str(args.output))
    session = TrackingSession(estimator, torch.cuda.synchronize)
    return SimpleNamespace(mesh=mesh, corners=corners, detector=detector, segmenter=segmenter,
                           estimator=estimator, session=session)


def tracking_statistics(recorder):
    return {"statistics_sample_frames": len(recorder.times),
            "tracking_latency_p50_ms": float(np.median(recorder.times)) if recorder.times else None,
            "tracking_latency_p95_ms": float(np.percentile(recorder.times, 95)) if recorder.times else None,
            "position_std_xyz_mm": (np.std(recorder.translations, axis=0) * 1000).tolist() if recorder.translations else None,
            "last_position_xyz_m": recorder.translations[-1].tolist() if recorder.translations else None}


def build_summary(args, metadata, recorder, segmenter, *, extra=None):
    summary = {"camera_serial": metadata["serial"], "camera_backend": args.camera,
               "camera_metadata": metadata, "resolution": metadata["rgb_resolution"],
               "initialization_mask": "SAM2.1_hiera_small" if segmenter else "yolo-seg",
               "class_id": args.class_id, "mesh_units": args.mesh_units,
               "est_refine_iter": args.est_refine_iter,
               "register_ms": recorder.init_ms, "registrations": recorder.registrations,
               "tracked_poses": recorder.tracked_poses, "tracking_checks": recorder.checks,
               "tracking_recoveries": recorder.recoveries, "tracking_resets": recorder.resets,
               "tracking_fps": recorder.tracked_poses / recorder.duration if recorder.tracked_poses else None,
               "calibration_independently_validated": False, "pose_accuracy_validated": False}
    if extra:
        summary.update(extra)
    return summary


def write_summary(output, summary):
    write_json(Path(output) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, ROOT)
    parser.add_argument("--stream-width", type=int, default=640, help="D415 stream width")
    parser.add_argument("--stream-height", type=int, default=480, help="D415 stream height")
    parser.add_argument("--show", action="store_true",
                        help="Show live RGB and pose overlays on the Jetson desktop")
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
    if args.show and not os.environ.get("DISPLAY"):
        parser.error("--show needs a desktop display. Run from a terminal on the Jetson desktop.")
    validate_inputs(parser, args)
    return args


def main():
    args = parse_args()
    cad_from_target = load_cad_from_target(args.geometry)
    model = bootstrap(args, ROOT)
    with ExitStack() as stack:
        if args.camera == "orbbec-ros":
            from fdpose_perception.orbbec_ros import open_orbbec_ros

            camera = stack.enter_context(open_orbbec_ros(args.serial, args.camera_namespace))
        else:
            camera = stack.enter_context(open_d415(args.serial, args.stream_width, args.stream_height))
        live_preview, preview = make_preview(camera.metadata["camera"], show=args.show,
                                             spawned=args.camera == "orbbec-ros")
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

        publish = (lambda target_pose: pose_publisher.publish(target_pose, frame_stamp_ns)) if pose_publisher else None
        recorder = Recorder(args.output, K, cad_from_target, publish=publish)
        pipeline = PosePipeline(model.session, model.detector,
                                model.segmenter.segment_box if model.segmenter else None,
                                model.corners, K, make_validate(model.corners, model.estimator, K),
                                init_width=args.init_width, est_refine_iter=args.est_refine_iter,
                                mask_backend=args.mask_backend)
        ui = SimpleNamespace(
            read_frame=read_frame,
            present=lambda bgr, status: preview.present(bgr, status) if preview else True,
            emit=lambda obj: print(json.dumps(obj), flush=True),
            log=lambda msg: print(msg, flush=True),
            draw=lambda bgr, cad_pose, target_pose: draw_pose(K, bgr, cad_pose, target_pose, model.mesh.bounds),
            interactive=bool(preview),
            closed=lambda: preview.closed if preview else False)
        pipeline.run(ui, recorder, args.frames)
        extra = {"requested_stream_fps": 30 if args.camera == "d415" else None}
        extra.update(tracking_statistics(recorder))
        if live_preview:
            extra.update(live_preview.measurements())
        write_summary(args.output, build_summary(args, camera.metadata, recorder, model.segmenter,
                                                 extra=extra))


if __name__ == "__main__":
    main()
