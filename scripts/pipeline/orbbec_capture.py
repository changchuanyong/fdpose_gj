"""Capture aligned RGB-D frames from an Orbbec Gemini camera for reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from rgbd_dataset import write_json, write_png


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
    if args.frames < 0 or args.warmup < 0:
        raise ValueError("--frames and --warmup must be nonnegative")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    try:
        import cv2
        import numpy as np
        import pyorbbecsdk as ob
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "capture requires OpenCV, NumPy and the official pyorbbecsdk v2 package"
        ) from exc

    context = None
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
    output_created = False
    index_stream = None
    try:
        align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        for _ in range(args.warmup):
            deadline = time.monotonic() + 5
            while aligned_pair(pipeline, align_filter, ob) is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("no synchronized Orbbec RGB-D pair during warmup")
        intrinsic = color_profile.get_intrinsic()
        matrix = camera_matrix(intrinsic)
        coefficients = distortion_coefficients(color_profile.get_distortion())
        rectify = None
        if any(abs(value) > 1e-12 for value in coefficients):
            rectify = cv2.initUndistortRectifyMap(
                np.asarray(matrix), np.asarray(coefficients), None, np.asarray(matrix),
                (color_profile.get_width(), color_profile.get_height()), cv2.CV_32FC1,
            )
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
        count = 0
        last_saved = 0.0
        last_pair = time.monotonic()
        index_stream = None
        while args.frames == 0 or count < args.frames:
            pair = aligned_pair(pipeline, align_filter, ob)
            if pair is None:
                if time.monotonic() - last_pair > 5:
                    raise TimeoutError("no synchronized Orbbec RGB-D pair for 5 seconds")
                continue
            last_pair = time.monotonic()
            now = time.monotonic()
            if count and now - last_saved < args.interval_s:
                continue
            frames, color_frame, depth_frame = pair
            bgr = decode_color(color_frame, ob, cv2, np)
            raw_depth = decode_depth(depth_frame, ob, np)
            if rectify is not None:
                bgr = cv2.remap(bgr, *rectify, cv2.INTER_LINEAR)
                raw_depth = cv2.remap(raw_depth, *rectify, cv2.INTER_NEAREST)
            scale_m = float(depth_frame.get_depth_scale()) / 1000.0
            if not 0 < scale_m <= 0.001:
                raise RuntimeError(f"unsupported Orbbec depth scale: {scale_m} m/unit")
            if "depth_scale_m" in metadata and abs(scale_m - metadata["depth_scale_m"]) > 1e-9:
                raise RuntimeError("Orbbec depth scale changed during capture")
            metadata["depth_scale_m"] = scale_m
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
                "color_device_time_us": int(color_frame.get_timestamp_us()),
                "depth_device_time_us": int(depth_frame.get_timestamp_us()),
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
    parser.add_argument("--serial", help="Orbbec serial number; default selects the only/default device")
    parser.add_argument("--frames", type=int, default=100, help="0 captures until Ctrl+C")
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=400)
    args = parser.parse_args()
    if min(args.fps, args.color_width, args.color_height, args.depth_width, args.depth_height) <= 0:
        parser.error("stream dimensions and --fps must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
