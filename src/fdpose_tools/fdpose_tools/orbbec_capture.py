"""Capture aligned RGB-D frames from an Orbbec Gemini camera for reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "fdpose_camera"))
from fdpose_camera.calibration import rectify_maps
from fdpose_tools.capture_common import (add_capture_args, capture_loop, require_modules,
                                         validate_capture_args, warmup)


def camera_matrix(intrinsic) -> list[list[float]]:
    return [
        [float(intrinsic.fx), 0.0, float(intrinsic.cx)],
        [0.0, float(intrinsic.fy), float(intrinsic.cy)],
        [0.0, 0.0, 1.0],
    ]


def distortion_coefficients(distortion) -> list[float]:
    return [float(getattr(distortion, name))
            for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]


def profile_description(profile) -> dict:
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": str(profile.get_format()),
    }


def choose_profile(profile_list, width: int, height: int, fps: int, formats: tuple) -> object:
    errors = []
    for frame_format in formats:
        try:
            return profile_list.get_video_stream_profile(width, height, frame_format, fps)
        except Exception as exc:
            errors.append(str(exc))
    available = [profile_description(profile_list[index]) for index in range(len(profile_list))]
    raise RuntimeError("requested stream profile is unavailable; "
                       f"available={json.dumps(available)}; errors={errors}")


def decode_color(frame, ob, cv2, np):
    width, height = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if frame.get_format() == ob.OBFormat.RGB:
        return data.reshape(height, width, 3)[:, :, ::-1].copy()
    if frame.get_format() == ob.OBFormat.MJPG:
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("could not decode Orbbec MJPG color frame")
        return decoded
    raise RuntimeError(f"unsupported Orbbec color format: {frame.get_format()}")


def decode_depth(frame, ob, np):
    if frame.get_format() != ob.OBFormat.Y16:
        raise RuntimeError(f"unsupported Orbbec depth format: {frame.get_format()}")
    return np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
        frame.get_height(), frame.get_width()).copy()


def aligned_pair(pipeline, align_filter, ob):
    frames = pipeline.wait_for_frames(500)
    if frames is None:
        return None
    aligned = align_filter.process(frames)
    if not aligned:
        return None
    frames = aligned.as_frame_set()
    color, depth = frames.get_color_frame(), frames.get_depth_frame()
    if color is None or depth is None:
        return None
    if (color.get_width(), color.get_height()) != (depth.get_width(), depth.get_height()):
        raise RuntimeError("software D2C returned different RGB and depth dimensions")
    return frames, color, depth


def run(args: argparse.Namespace) -> int:
    validate_capture_args(args)
    cv2, np, ob = require_modules(
        "capture requires OpenCV, NumPy and the official pyorbbecsdk v2 package",
        "cv2", "numpy", "pyorbbecsdk")

    if args.serial:
        context = ob.Context()
        device = context.query_devices().get_device_by_serial_number(args.serial)
        pipeline = ob.Pipeline(device)
    else:
        pipeline = ob.Pipeline()
    config = ob.Config()
    color_formats = (ob.OBFormat.RGB, ob.OBFormat.MJPG)
    color_profile = choose_profile(
        pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
        args.color_width, args.color_height, args.fps, color_formats,
    )
    depth_profile = choose_profile(
        pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR),
        args.depth_width, args.depth_height, args.fps, (ob.OBFormat.Y16,),
    )
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync()
    pipeline.start(config)
    try:
        align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

        def next_pair():
            return aligned_pair(pipeline, align_filter, ob)

        warmup(next_pair, args.warmup,
               lambda: "no synchronized Orbbec RGB-D pair during warmup")
        intrinsic = color_profile.get_intrinsic()
        matrix = camera_matrix(intrinsic)
        coefficients = distortion_coefficients(color_profile.get_distortion())
        rectify = rectify_maps(matrix, coefficients,
                               (color_profile.get_width(), color_profile.get_height()),
                               zero_tol=1e-12)
        device_info = pipeline.get_device().get_device_info()
        metadata = {
            "schema_version": 1,
            "camera": "Orbbec",
            "device_name": str(device_info.get_name()),
            "serial": str(device_info.get_serial_number()),
            "alignment": "software_depth_to_color",
            "rectified": True,
            "width": int(color_profile.get_width()),
            "height": int(color_profile.get_height()),
            "fps": int(color_profile.get_fps()),
            "color_format": str(color_profile.get_format()),
            "camera_matrix": matrix,
            "source_distortion": coefficients,
            "color_profile": profile_description(color_profile),
            "depth_profile": profile_description(depth_profile),
        }

        def decode(pair):
            frames, color_frame, depth_frame = pair
            bgr = decode_color(color_frame, ob, cv2, np)
            raw_depth = decode_depth(depth_frame, ob, np)
            scale_m = float(depth_frame.get_depth_scale()) / 1000.0
            if not 0 < scale_m <= 0.001:
                raise RuntimeError(f"unsupported Orbbec depth scale: {scale_m} m/unit")
            if "depth_scale_m" in metadata and abs(scale_m - metadata["depth_scale_m"]) > 1e-9:
                raise RuntimeError("Orbbec depth scale changed during capture")
            metadata["depth_scale_m"] = scale_m
            return (bgr, raw_depth, int(color_frame.get_timestamp_us()),
                    int(depth_frame.get_timestamp_us()))

        return capture_loop(
            args, metadata=metadata, next_pair=next_pair, decode=decode, rectify=rectify,
            timeout_message=lambda: "no synchronized Orbbec RGB-D pair for 5 seconds",
            cv2=cv2, np=np)
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_args(parser, depth_height_default=400,
                     serial_help="Orbbec serial number; default selects the only/default device")
    args = parser.parse_args()
    if min(args.fps, args.color_width, args.color_height, args.depth_width, args.depth_height) <= 0:
        parser.error("stream dimensions and --fps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())