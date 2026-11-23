"""Shared reader for aligned, rectified RGB-D capture directories."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = 1


def load_metadata(root: Path) -> dict:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    required = {"schema_version", "width", "height", "camera_matrix", "depth_scale_m", "rectified"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"capture metadata missing: {', '.join(missing)}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported capture schema: {metadata['schema_version']}")
    if not metadata["rectified"]:
        raise ValueError("RGB-D reconstruction requires rectified capture data")
    if int(metadata["width"]) <= 0 or int(metadata["height"]) <= 0:
        raise ValueError("capture dimensions must be positive")
    if float(metadata["depth_scale_m"]) <= 0:
        raise ValueError("depth_scale_m must be positive")
    matrix = metadata["camera_matrix"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("camera_matrix must be 3x3")
    if matrix[0][0] <= 0 or matrix[1][1] <= 0 or matrix[2][2] != 1:
        raise ValueError("camera_matrix has invalid focal lengths or homogeneous scale")
    return metadata


def load_frame_index(root: Path) -> list[dict]:
    path = Path(root) / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records:
        raise ValueError("capture contains no frames")
    indices = [int(record["index"]) for record in records]
    if indices != list(range(len(records))):
        raise ValueError("capture frame indices must be contiguous and ordered")
    for record in records:
        if not record.get("rgb") or not record.get("depth"):
            raise ValueError(f"frame {record['index']} has no RGB-D paths")
    return records


def read_png(path: Path, flags: int):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"image is unreadable: {path}")
    return image


def read_rgbd(root: Path, record: dict, width: int, height: int):
    import cv2

    root = Path(root)
    bgr = read_png(root / record["rgb"], cv2.IMREAD_COLOR)
    depth = read_png(root / record["depth"], cv2.IMREAD_UNCHANGED)
    if bgr.shape[:2] != (height, width) or depth.shape != (height, width):
        raise ValueError(f"frame {record['index']} dimensions do not match metadata")
    if depth.dtype.name != "uint16":
        raise ValueError(f"frame {record['index']} depth is not uint16")
    return bgr[:, :, ::-1].copy(), depth


def write_png(path: Path, image) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"could not encode PNG: {path}")
    encoded.astype(np.uint8).tofile(str(path))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
