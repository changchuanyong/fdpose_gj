"""Propagate one SAM2 box prompt into per-frame object masks for reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/fdpose_tools"))
from fdpose_tools.rgbd_dataset import load_frame_index, load_metadata, read_png, write_json, write_png


def run(args: argparse.Namespace) -> int:
    root = args.capture.resolve()
    metadata = load_metadata(root)
    records = load_frame_index(root)
    if args.max_frames:
        records = records[:args.max_frames]
    if not 0 <= args.prompt_frame < len(records):
        raise ValueError("--prompt-frame must index the selected frames")
    if args.box[0] >= args.box[2] or args.box[1] >= args.box[3]:
        raise ValueError("--box must be x0 y0 x1 y1 with x0<x1 and y0<y1")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    try:
        import cv2
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "masking requires OpenCV, NumPy, torch and the project SAM2 source"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 video propagation requires CUDA")
    sys.path.insert(0, str(ROOT / "third_party/sam2"))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        args.sam2_config, str(args.sam2_checkpoint), device="cuda",
    )
    scratch = Path(tempfile.mkdtemp(prefix="sam2-jpg-"))
    try:
        for record in records:
            bgr = read_png(root / record["rgb"], cv2.IMREAD_COLOR)
            if not cv2.imwrite(str(scratch / f"{record['index']:05d}.jpg"), bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"could not write JPEG for frame {record['index']}")
        state = predictor.init_state(video_path=str(scratch), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=args.prompt_frame,
                                        obj_id=1, box=np.asarray(args.box, dtype=np.float32))
        args.output.mkdir(parents=True)
        height, width = int(metadata["height"]), int(metadata["width"])
        counts = {}
        for reverse in (False, True):
            for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=reverse):
                mask = (np.squeeze(mask_logits[0].cpu().numpy()) > 0.0).astype(np.uint8) * 255
                if mask.shape != (height, width):
                    raise RuntimeError(f"unexpected mask shape {mask.shape} for frame {frame_idx}")
                index = records[frame_idx]["index"]
                write_png(args.output / f"{index:06d}_mask.png", mask)
                counts[index] = int(np.count_nonzero(mask))
        ordered = sorted(counts.values())
        summary = {
            "frames": len(counts),
            "prompt_frame": args.prompt_frame,
            "mask_pixels_min": ordered[0],
            "mask_pixels_median": ordered[len(ordered) // 2],
            "frames_below_128": sorted(i for i, n in counts.items() if n < 128),
        }
        write_json(args.output / "masks_summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Mask directory written as 000000_mask.png")
    parser.add_argument("--prompt-frame", type=int, required=True,
                        help="Frame index whose box prompt is propagated both directions")
    parser.add_argument("--box", type=float, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--max-frames", type=int, default=0, help="Use only the first N capture frames")
    parser.add_argument("--sam2-checkpoint", type=Path, default=ROOT / "weights/sam2.1_hiera_small.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())