"""Track a configured part with live RGB-D and FoundationPose on Jetson."""

import argparse
from collections import deque
from contextlib import ExitStack
from functools import partial
from itertools import product
import json
import logging
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FDPOSE_ROOT", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / "fdpose_ws/src/fdpose_perception"))
sys.path.insert(0, str(CODE_ROOT / "fdpose_ws/src/fdpose_camera"))
from fdpose_camera.rgbd import open_d415
from fdpose_perception.adapters.detector import Sam2Segmenter, TargetDetector
from fdpose_perception.adapters.foundationpose import render_depth
from fdpose_perception.adapters.ros2 import PosePublisher
from fdpose_perception.geometry import box_in_view, projected_box
from fdpose_perception.pipeline import RejectedMask, TrackingSession, prepare_registration
from fdpose_perception.preview import GeminiPreview, PreviewWindow
from fdpose_perception.validation import (MAX_REVIEW_ROTATION_DEG, MAX_REVIEW_TRANSLATION_M,
                                          depth_alignment, pose_delta,
                                          validate_pose as validate_pose_result)


CHECK_PERIOD_S = 0.6


UNIT_TO_M = {"mm": 0.001, "m": 1.0}


def load_cad_from_target(path):
    geometry = json.loads(path.read_text(encoding="utf-8-sig"))
    units = geometry.get("units", "mm")
    if units not in UNIT_TO_M:
        raise ValueError(f"Unsupported geometry units {units!r}: {path}")
    target_from_cad = np.asarray(geometry["cad_to_target"], dtype=float)
    if (target_from_cad.shape != (4, 4) or not np.isfinite(target_from_cad).all()
            or not np.allclose(target_from_cad[3], [0, 0, 0, 1])):
        raise ValueError(f"cad_to_target must be a finite 4x4 homogeneous transform: {path}")
    rotation = target_from_cad[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError(f"cad_to_target must contain a proper rotation: {path}")
    target_from_cad[:3, 3] *= UNIT_TO_M[units]
    return np.linalg.inv(target_from_cad)


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
    from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, set_seed

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

        def draw_pose(bgr_frame, cad_pose, target_pose):
            boxed = draw_posed_3d_box(K, bgr_frame, cad_pose, mesh.bounds)
            scale = 0.04
            axes = np.vstack((np.zeros((1, 3)), np.eye(3) * scale))
            bounds = projected_box(K, axes, target_pose)
            if bounds is None:
                return boxed
            height, width = boxed.shape[:2]
            x0, y0 = np.maximum(np.floor(bounds[:2]).astype(int) - 4, 0)
            x1, y1 = np.minimum(np.ceil(bounds[2:]).astype(int) + 5, (width, height))
            if x1 > x0 and y1 > y0:
                local_K = K.copy()
                local_K[:2, 2] -= [x0, y0]
                crop = boxed[y0:y1, x0:x1]
                crop[:] = draw_xyz_axis(crop, target_pose, scale=scale, K=local_K)
            return boxed

        def validate_pose(pose, detection_box, measured_depth, evaluation_K=None):
            evaluation_K = K if evaluation_K is None else evaluation_K

            return validate_pose_result(corners, pose, detection_box, measured_depth,
                                        evaluation_K, partial(render_depth, estimator), depth_alignment)

        def save_drift(before, after, rejected_pose, recovery_pose, rejected_validation,
                       recovery_validation, change):
            directory = args.output / "drift" / f"{after[2]:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            for name, frame in (("before", before), ("after", after)):
                rgb, depth, number, timestamp, pose = frame
                cv2.imwrite(str(directory / f"{name}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(directory / f"{name}_depth.npy", depth, allow_pickle=False)
                cv2.imwrite(str(directory / f"{name}_depth_mm.png"),
                            np.rint(depth * 1000).astype(np.uint16))
                np.savetxt(directory / f"{name}_pose.txt", pose)
            np.savetxt(directory / "cam_K.txt", K)
            np.savetxt(directory / "rejected_pose.txt", rejected_pose)
            np.savetxt(directory / "recovery_pose.txt", recovery_pose)
            metadata = {"before_frame": before[2], "after_frame": after[2],
                        "before_device_time_ms": before[3], "after_device_time_ms": after[3],
                        "change_from_confirmed": {"translation_m": change[0], "rotation_deg": change[1]},
                        "rejected_validation": rejected_validation,
                        "recovery_validation": recovery_validation}
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                       encoding="utf-8")
            print(json.dumps({"event": "drift_rgbd_saved", "path": str(directory)}), flush=True)

        def write_pose(stream, pose, frame, device_ms, stage, elapsed_ms, validation=None, change=None):
            target_pose = np.asarray(pose, dtype=float) @ cad_from_target
            rotation = target_pose[:3, :3]
            if (not np.isfinite(target_pose).all() or target_pose[2, 3] <= 0
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                    or abs(np.linalg.det(rotation) - 1) > 1e-3):
                raise RuntimeError("FoundationPose returned an invalid rigid transform")
            record = {"frame": frame, "device_time_ms": device_ms, "stage": stage,
                      "elapsed_ms": elapsed_ms, "validation": validation,
                      "change_from_confirmed": change, "T_camera_target_m": target_pose.tolist()}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if pose_publisher:
                pose_publisher.publish(target_pose, frame_stamp_ns)
            return target_pose

        segment_box = segmenter.segment_box if segmenter else None

        def seek_pose():
            deadline = float("inf") if preview else time.monotonic() + 25
            status = "Aim at the target"
            while time.monotonic() < deadline:
                started = time.perf_counter()
                rgb, depth, number, timestamp = read_frame()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detection = detector.predict(bgr)
                if detection is None or not box_in_view(detection[0], bgr.shape):
                    status = "Aim at the target" if detection is None else "Target clipped | Move it fully into view"
                    if preview and not preview.present(bgr, f"{status} | Q or Esc to quit"):
                        return None
                    continue
                box, confidence, mask = detection
                action = session.seek_action(time.monotonic())
                if action == "recover":
                    recovered = session.track(rgb, depth, K, recover=True)
                    validation = validate_pose(recovered, box, depth)
                    elapsed = (time.perf_counter() - started) * 1000
                    if validation["accepted"]:
                        print(json.dumps({"event": "tracking_recovered", "frame": number,
                                          "validation": validation,
                                          "backend": "last_confirmed_pose"}), flush=True)
                        return (recovered, rgb, depth, bgr, bgr, number, timestamp,
                                elapsed, validation, "recover")
                    status = "Recovering from last confirmed pose"
                    if preview and not preview.present(bgr, status):
                        return None
                    continue
                if action == "wait":
                    status = "Waiting before registration retry"
                    if preview and not preview.present(bgr, status):
                        return None
                    continue
                session.begin_registration(time.monotonic())
                try:
                    candidate = prepare_registration(rgb, depth, K, args.init_width, box, mask, segment_box)
                except RejectedMask as exc:
                    status = str(exc)
                    if preview and not preview.present(bgr, status):
                        return None
                    continue
                mask_info = {"mask_backend": args.mask_backend}
                if candidate.score is not None:
                    mask_info["sam2_score"] = candidate.score
                overlay = bgr.copy()
                overlay_mask = cv2.resize(candidate.mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay[overlay_mask] = (overlay[overlay_mask] * 0.55
                                         + np.array([0, 255, 0]) * 0.45).astype(np.uint8)
                x0, y0, x1, y1 = np.rint(box).astype(int)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 255), 2)
                cv2.imwrite(str(args.output / "candidate_mask.jpg"), overlay)
                if preview and not preview.present(overlay, "Target mask | Initializing FoundationPose"):
                    return None
                started = time.perf_counter()
                initial = session.register(candidate.rgb, candidate.depth, candidate.K, candidate.mask,
                                           args.est_refine_iter)
                elapsed = (time.perf_counter() - started) * 1000
                validation = validate_pose(initial, candidate.box, candidate.depth, evaluation_K=candidate.K)
                if estimator.pose_last is None or not validation["accepted"]:
                    estimator.pose_last = None
                    status = "FoundationPose pose failed projection/depth validation"
                    print(json.dumps({"event": "registration_rejected", "validation": validation}), flush=True)
                    continue
                print(json.dumps({"event": "registration_accepted", "frame": number,
                                  "detector_confidence": confidence, **mask_info,
                                  "validation": validation, "register_ms": elapsed}), flush=True)
                return (initial, rgb, depth, bgr, overlay, number, timestamp,
                        elapsed, validation, "register")
            cv2.imwrite(str(args.output / "camera.jpg"), bgr)
            print(f"No aligned target pose before timeout: {status}", flush=True)
            return None

        for _ in range(12):
            read_frame()
        times, translations = deque(maxlen=600), deque(maxlen=600)
        tracked_poses = registrations = checks = recoveries = resets = 0
        duration = 0.0
        init_ms = image = last_overlay = None
        with (args.output / "poses.jsonl").open("x", encoding="utf-8") as stream:
            try:
                while not args.frames or tracked_poses < args.frames:
                    registration = seek_pose()
                    if registration is None:
                        break
                    (pose, rgb, depth, bgr, mask_overlay, number, timestamp,
                     elapsed, validation, stage) = registration
                    target_pose = write_pose(stream, pose, number, timestamp, stage, elapsed,
                                             validation=validation)
                    if stage == "register":
                        registrations += 1
                    else:
                        recoveries += 1
                    session.confirm(rgb, depth, number, timestamp, pose)
                    last_overlay = bgr, pose, target_pose
                    image = draw_pose(bgr.copy(), pose, target_pose)
                    if registrations == 1 and stage == "register":
                        init_ms = elapsed
                        cv2.imwrite(str(args.output / "initial_mask.jpg"), mask_overlay)
                        cv2.imwrite(str(args.output / "first_pose.jpg"), image)
                    if preview and not preview.present(image, "Tracking | Q or Esc to quit"):
                        break
                    next_check_at = 0.0
                    while not args.frames or tracked_poses < args.frames:
                        started = time.perf_counter()
                        rgb, depth, new_number, timestamp = read_frame()
                        if new_number <= number:
                            raise RuntimeError("Camera frame number did not advance")
                        number = new_number
                        stage, validation = "track", None
                        pose = session.track(rgb, depth, K)
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        projection = projected_box(K, corners, pose)
                        translation_change, rotation_change = pose_delta(session.confirmed_frame[4], pose)
                        change = {"translation_m": translation_change, "rotation_deg": rotation_change}
                        sudden_change = (translation_change > MAX_REVIEW_TRANSLATION_M
                                         or rotation_change > MAX_REVIEW_ROTATION_DEG)
                        if projection is None or sudden_change or started >= next_check_at:
                            detection = detector.predict(bgr)
                            detection_box = detection[0] if detection and box_in_view(detection[0], bgr.shape) else None
                            validation = validate_pose(pose, detection_box, depth)
                            checks += 1
                            next_check_at = started + CHECK_PERIOD_S
                            if not validation["accepted"]:
                                rejected_pose = pose.copy()
                                rejected_validation = validation
                                recovered = session.track(rgb, depth, K, recover=True)
                                recovery_validation = validate_pose(recovered, detection_box, depth)
                                save_drift(session.confirmed_frame,
                                           (rgb, depth, number, timestamp, rejected_pose),
                                           rejected_pose, recovered, rejected_validation,
                                           recovery_validation, (translation_change, rotation_change))
                                if recovery_validation["accepted"]:
                                    pose, validation, stage = recovered, recovery_validation, "recover"
                                    translation_change, rotation_change = pose_delta(session.confirmed_frame[4], pose)
                                    change = {"translation_m": translation_change,
                                              "rotation_deg": rotation_change}
                                    recoveries += 1
                                    print(json.dumps({"event": "tracking_recovered", "frame": number,
                                                      "validation": validation,
                                                      "backend": "last_confirmed_pose"}), flush=True)
                                else:
                                    resets += 1
                                    session.tracking_lost(time.monotonic())
                                    if preview:
                                        preview.present(bgr, "Pose rejected | Recovering target")
                                    duration += time.perf_counter() - started
                                    print(json.dumps({"event": "tracking_lost", "frame": number,
                                                      "rejected_validation": rejected_validation,
                                                      "recovery_validation": recovery_validation,
                                                      "fallback": "cached_pose_then_throttled_register"}), flush=True)
                                    break
                        ms = (time.perf_counter() - started) * 1000
                        target_pose = write_pose(stream, pose, number, timestamp, stage, ms,
                                                 validation=validation, change=change)
                        times.append(ms)
                        translations.append(target_pose[:3, 3])
                        tracked_poses += 1
                        if validation is not None and validation["accepted"]:
                            session.confirm(rgb, depth, number, timestamp, pose)
                        last_overlay = bgr, pose, target_pose
                        if preview:
                            image = draw_pose(bgr, pose, target_pose)
                            preview.present(image, "Tracking | Q or Esc to quit")
                        duration += time.perf_counter() - started
                        if preview and preview.closed:
                            break
                    if preview and preview.closed:
                        break
            except KeyboardInterrupt:
                pass
        if not preview and last_overlay is not None:
            image = draw_pose(*last_overlay)
        if image is not None:
            cv2.imwrite(str(args.output / "last_pose.jpg"), image)
        summary = {"camera_serial": camera.metadata["serial"], "camera_backend": args.camera,
                   "camera_metadata": camera.metadata,
                   "resolution": camera.metadata["rgb_resolution"],
                   "requested_stream_fps": 30 if args.camera == "d415" else None,
                   "initialization_mask": "SAM2.1_hiera_small" if segmenter else "yolo-seg",
                   "class_id": args.class_id, "mesh_units": args.mesh_units,
                   "est_refine_iter": args.est_refine_iter,
                   "register_ms": init_ms, "registrations": registrations, "tracked_poses": tracked_poses,
                   "tracking_checks": checks, "tracking_recoveries": recoveries, "tracking_resets": resets,
                   "tracking_fps": tracked_poses / duration if tracked_poses else None,
                   "statistics_sample_frames": len(times),
                   "tracking_latency_p50_ms": float(np.median(times)) if times else None,
                   "tracking_latency_p95_ms": float(np.percentile(times, 95)) if times else None,
                   "position_std_xyz_mm": (np.std(translations, axis=0) * 1000).tolist() if translations else None,
                   "last_position_xyz_m": translations[-1].tolist() if translations else None,
                   "calibration_independently_validated": False, "pose_accuracy_validated": False}
        if live_preview:
            summary.update(live_preview.measurements())
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
