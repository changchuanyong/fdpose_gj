"""Replay a saved drift RGB-D pair at native resolution and optional downscaled widths."""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import trimesh


PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src/fdpose_perception"))
sys.path.insert(0, str(PROJECT / "third_party/FoundationPose"))
from estimater import FoundationPose
from fdpose_perception.adapters.foundationpose import make_validate
from fdpose_perception.tracking import TrackingSession, pose_delta
from fdpose_perception.adapters.detector import TargetDetector
from fdpose_perception.recording import write_json


def detect_box(detector, rgb):
    detection = detector.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return detection[0] if detection else None


def scaled_input(rgb, depth, K, width):
    if width == rgb.shape[1]:
        return rgb, depth, K
    scale = width / rgb.shape[1]
    height = round(rgb.shape[0] * scale)
    resized_rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    resized_K = K.copy()
    resized_K[0] *= scale
    resized_K[1] *= scale
    return resized_rgb, resized_depth, resized_K


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--mesh", type=Path, default=PROJECT / "config/mesh/socket_mod.PLY")
    parser.add_argument("--weights", type=Path, default=PROJECT / "weights/best.pt")
    parser.add_argument("--widths", type=int, nargs="+", default=[1280, 640])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    rgb = cv2.cvtColor(cv2.imread(str(directory / "after_rgb.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(directory / "after_depth.npy", allow_pickle=False)
    K = np.loadtxt(directory / "cam_K.txt")
    before_pose = np.loadtxt(directory / "before_pose.txt")
    mesh = trimesh.load(args.mesh, force="mesh")
    mesh.apply_scale(0.001)
    detector = TargetDetector(args.weights, backend="sam2", class_id=0)
    rows = []
    for width in args.widths:
        if width > rgb.shape[1]:
            raise ValueError(f"Cannot create a real {width}px result from {rgb.shape[1]}px input")
        current_rgb, current_depth, current_K = scaled_input(rgb, depth, K, width)
        estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                   mesh=mesh, debug=0, debug_dir=str(args.output.parent))
        to_center = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        estimator.pose_last = torch.as_tensor(before_pose @ np.linalg.inv(to_center),
                                              device="cuda", dtype=torch.float32)
        session = TrackingSession(estimator, torch.cuda.synchronize)
        started = time.perf_counter()
        pose = session.track(current_rgb, current_depth, current_K)
        elapsed_ms = (time.perf_counter() - started) * 1000
        detection_box = detect_box(detector, current_rgb)
        corners = np.asarray(np.meshgrid(*estimator.mesh_ori.bounds.T)).reshape(3, -1).T
        validation = make_validate(corners, estimator, current_K)(pose, detection_box, current_depth)
        rows.append({"width": width, "height": current_rgb.shape[0], "track_ms": elapsed_ms,
                     "change_from_before": pose_delta(before_pose, pose), "validation": validation})
        estimator.glctx = None
        del session, estimator
        torch.cuda.empty_cache()
    write_json(args.output, {"source": str(directory), "runs": rows})
    print(json.dumps({"source": str(directory), "runs": rows}), flush=True)


if __name__ == "__main__":
    main()
