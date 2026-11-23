"""Shared capture validation, warmup and frame-saving loop for the RGB-D capture tools."""

from __future__ import annotations

import importlib
import json
import time

from fdpose_tools.rgbd_dataset import write_json, write_png


def add_capture_args(parser, *, serial_help, depth_height_default):
    from pathlib import Path

    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial", help=serial_help)
    parser.add_argument("--frames", type=int, default=100, help="0 captures until Ctrl+C")
    parser.add_argument("--interval-s", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=depth_height_default)


def validate_capture_args(args):
    if args.frames < 0 or args.warmup < 0:
        raise ValueError("--frames and --warmup must be nonnegative")
    if args.interval_s < 0:
        raise ValueError("--interval-s must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")


def require_modules(message, *names):
    try:
        return tuple(importlib.import_module(name) for name in names)
    except ModuleNotFoundError as exc:
        raise RuntimeError(message) from exc


def warmup(next_pair, count, timeout_message):
    for _ in range(count):
        deadline = time.monotonic() + 5
        while next_pair() is None:
            if time.monotonic() > deadline:
                raise TimeoutError(timeout_message())


def capture_loop(args, *, metadata, next_pair, decode, rectify, timeout_message, cv2, np):
    """Save aligned pairs until the frame budget is met.

    decode(pair) returns (bgr, raw_depth, color_us, depth_us) and owns any
    per-frame device checks; the caller keeps pipeline lifetime in its own finally.
    """
    output_created = False
    index_stream = None
    count = 0
    last_saved = 0.0
    last_pair = time.monotonic()
    try:
        while args.frames == 0 or count < args.frames:
            pair = next_pair()
            if pair is None:
                if time.monotonic() - last_pair > 5:
                    raise TimeoutError(timeout_message())
                continue
            last_pair = time.monotonic()
            now = time.monotonic()
            if count and now - last_saved < args.interval_s:
                continue
            bgr, raw_depth, color_us, depth_us = decode(pair)
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
                "color_device_time_us": color_us,
                "depth_device_time_us": depth_us,
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