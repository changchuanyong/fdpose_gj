"""Capture aligned RGB-D frames from an Intel RealSense D415 for reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from rgbd_dataset import write_json, write_png


def camera_matrix(intrinsic) -> list[list[float]]:
    return [
        [float(intrinsic.fx), 0.0, float(intrinsic.ppx)],
        [0.0, float(intrinsic.fy), float(intrinsic.ppy)],
        [0.0, 0.0, 1.0],
    ]


def distortion_coefficients(intrinsic) -> list[float]:
    return [float(value) for value in intrinsic.coeffs]


def profile_description(profile) -> dict:
    return {
        "width": int(profile.width()),
        "height": int(profile.height()),
        "fps": int(profile.fps()),
        "format": str(profile.format()),
    }


def depth_scale_supported(scale_m: float) -> bool:
    # pyrealsense2 返回 float32：1 mm 读回为 0.0010000000474974513
    return 0 < scale_m <= 0.001 * (1 + 1e-6)


def aligned_pair(pipeline, align, errors=None):
    try:
        frames = pipeline.wait_for_frames(500)
    except RuntimeError as exc:
        if errors is not None:
            errors.append(str(exc))
        return None
    aligned = align.process(frames)
    color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
    if not color or not depth:
        return None
    if (color.get_width(), color.get_height()) != (depth.get_width(), depth.get_height()):
        raise RuntimeError("RealSense align returned different RGB and depth dimensions")
    return color, depth


def run(args: argparse.Namespace) -> int:
    if args.frames < 0 or args.warmup < 0:
        raise ValueError("--frames and --warmup must be nonnegative")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "capture requires OpenCV, NumPy and the pyrealsense2 package"
        ) from exc

    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height,
                         rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                         rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    output_created = False
    index_stream = None
    try:
        device = profile.get_device()
        serial = str(device.get_info(rs.camera_info.serial_number))
        name = str(device.get_info(rs.camera_info.name))
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsic = color_profile.get_intrinsics()
        matrix = camera_matrix(intrinsic)
        coefficients = distortion_coefficients(intrinsic)
        rectify = None
        if intrinsic.model != rs.distortion.none and any(abs(v) > 1e-12 for v in coefficients):
            if intrinsic.model != rs.distortion.brown_conrady:
                raise RuntimeError(f"unsupported RealSense color distortion model: {intrinsic.model}")
            rectify = cv2.initUndistortRectifyMap(
                np.asarray(matrix), np.asarray(coefficients), None, np.asarray(matrix),
                (intrinsic.width, intrinsic.height), cv2.CV_32FC1,
            )
        scale_m = float(device.first_depth_sensor().get_depth_scale())
        if not depth_scale_supported(scale_m):
            raise RuntimeError(f"unsupported RealSense depth scale: {scale_m} m/unit")
        align = rs.align(rs.stream.color)
        sdk_errors = []
        for _ in range(args.warmup):
            deadline = time.monotonic() + 5
            while aligned_pair(pipeline, align, sdk_errors) is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("no aligned RealSense RGB-D pair during warmup; "
                                       f"last SDK error: {sdk_errors[-1] if sdk_errors else 'missing color/depth frame'}")
        metadata = {
            "schema_version": 1,
            "camera": "Intel RealSense",
            "device_name": name,
            "serial": serial,
            "alignment": "realsense_align_to_color",
            "rectified": True,
            "width": int(intrinsic.width),
            "height": int(intrinsic.height),
            "fps": int(color_profile.fps()),
            "color_format": str(color_profile.format()),
            "camera_matrix": matrix,
            "source_distortion": coefficients,
            "distortion_model": str(intrinsic.model),
            "color_profile": profile_description(color_profile),
            "depth_profile": profile_description(depth_profile),
            "depth_scale_m": scale_m,
        }
        count = 0
        last_saved = 0.0
        last_pair = time.monotonic()
        while args.frames == 0 or count < args.frames:
            pair = aligned_pair(pipeline, align, sdk_errors)
            if pair is None:
                if time.monotonic() - last_pair > 5:
                    raise TimeoutError("no aligned RealSense RGB-D pair for 5 seconds; "
                                       f"last SDK error: {sdk_errors[-1] if sdk_errors else 'missing color/depth frame'}")
                continue
            last_pair = time.monotonic()
            now = time.monotonic()
            if count and now - last_saved < args.interval_s:
                continue
            color_frame, depth_frame = pair
            rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
                color_frame.get_height(), color_frame.get_width(), 3)
            bgr = rgb[:, :, ::-1].copy()
            raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()).copy()
            if rectify is not None:
                bgr = cv2.remap(bgr, *rectify, cv2.INTER_LINEAR)
                raw_depth = cv2.remap(raw_depth, *rectify, cv2.INTER_NEAREST)
            if not output_created:
                args.output.mkdir(parents=True, exist_ok=False)
                write_json(args.output / "metadata.json", metadata)
                index_stream = (args.output / "frames.jsonl").open("x", encoding="utf-8")
                output_created = True
            stem = f"{count:06d}"
            rgb_name, depth_name = f"frames/{stem}_rgb.png", f"frames/{stem}_depth.png"
            write_png(args.output / rgb_name, bgr)
            write_png(args.output / depth_name, raw_depth)
            record = {
                "index": count,
                "rgb": rgb_name,
                "depth": depth_name,
                "color_device_time_us": int(color_frame.get_timestamp() * 1000),
                "depth_device_time_us": int(depth_frame.get_timestamp() * 1000),
                "valid_depth_pixels": int(np.count_nonzero(raw_depth)),
            }
            index_stream.write(json.dumps(record, allow_nan=False) + "\n")
            index_stream.flush()
            count += 1
            last_saved = now
            print(json.dumps({"saved": count, "output": str(args.output),
                              "valid_depth_pixels": record["valid_depth_pixels"]}), flush=True)
        if index_stream is not None:
            index_stream.close()
        if count == 0:
            raise RuntimeError("capture ended before any frame was saved")
        return 0
    finally:
        if index_stream is not None and not index_stream.closed:
            index_stream.close()
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial", help="RealSense serial number; default selects the only/default device")
    parser.add_argument("--frames", type=int, default=100, help="0 captures until Ctrl+C")
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    args = parser.parse_args()
    if min(args.fps, args.color_width, args.color_height, args.depth_width, args.depth_height) <= 0:
        parser.error("stream dimensions and --fps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())