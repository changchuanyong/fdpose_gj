"""Capture aligned RGB-D frames from an Intel RealSense D415 for reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fdpose_camera"))
from fdpose_camera.calibration import rectify_maps
from fdpose_tools.capture_common import (add_capture_args, capture_loop, require_modules,
                                         validate_capture_args, warmup)


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
    validate_capture_args(args)
    cv2, np, rs = require_modules(
        "capture requires OpenCV, NumPy and the pyrealsense2 package",
        "cv2", "numpy", "pyrealsense2")

    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.color_width, args.color_height,
                         rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.depth_width, args.depth_height,
                         rs.format.z16, args.fps)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
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
        if intrinsic.model != rs.distortion.none:
            if any(abs(v) > 1e-12 for v in coefficients) and intrinsic.model != rs.distortion.brown_conrady:
                raise RuntimeError(f"unsupported RealSense color distortion model: {intrinsic.model}")
            rectify = rectify_maps(matrix, coefficients, (intrinsic.width, intrinsic.height),
                                   zero_tol=1e-12)
        scale_m = float(device.first_depth_sensor().get_depth_scale())
        if not depth_scale_supported(scale_m):
            raise RuntimeError(f"unsupported RealSense depth scale: {scale_m} m/unit")
        align = rs.align(rs.stream.color)
        sdk_errors = []

        def next_pair():
            return aligned_pair(pipeline, align, sdk_errors)

        def sdk_suffix():
            return f"; last SDK error: {sdk_errors[-1] if sdk_errors else 'missing color/depth frame'}"

        warmup(next_pair, args.warmup,
               lambda: "no aligned RealSense RGB-D pair during warmup" + sdk_suffix())
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

        def decode(pair):
            color_frame, depth_frame = pair
            rgb = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
                color_frame.get_height(), color_frame.get_width(), 3)
            raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()).copy()
            return (rgb[:, :, ::-1].copy(), raw_depth,
                    int(color_frame.get_timestamp() * 1000),
                    int(depth_frame.get_timestamp() * 1000))

        return capture_loop(
            args, metadata=metadata, next_pair=next_pair, decode=decode, rectify=rectify,
            timeout_message=lambda: "no aligned RealSense RGB-D pair for 5 seconds" + sdk_suffix(),
            cv2=cv2, np=np)
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_args(parser, depth_height_default=480,
                     serial_help="RealSense serial number; default selects the only/default device")
    args = parser.parse_args()
    if min(args.fps, args.color_width, args.color_height, args.depth_width, args.depth_height) <= 0:
        parser.error("stream dimensions and --fps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())