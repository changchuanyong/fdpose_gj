"""Shared CLI surface, model bootstrap and summary for the live pose frontends."""

from functools import partial
from itertools import product
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from fdpose_perception.adapters.detector import Sam2Segmenter, TargetDetector
from fdpose_perception.adapters.foundationpose import render_depth
from fdpose_perception.jsonio import write_json
from fdpose_perception.pipeline import TrackingSession, UNIT_TO_M
from fdpose_perception.validation import depth_alignment, validate_pose


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


def make_validate(corners, estimator, default_K):
    def validate(pose, detection_box, measured_depth, evaluation_K=None):
        evaluation_K = default_K if evaluation_K is None else evaluation_K
        return validate_pose(corners, pose, detection_box, measured_depth, evaluation_K,
                             partial(render_depth, estimator), depth_alignment)
    return validate


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