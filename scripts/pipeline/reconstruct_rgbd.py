"""Reconstruct a millimetre PLY mesh from an aligned multi-view RGB-D capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rgbd_dataset import load_frame_index, load_metadata, read_png, read_rgbd, write_json


def make_rgbd(o3d, rgb: np.ndarray, depth: np.ndarray, depth_scale_m: float,
              max_depth_m: float):
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth),
        depth_scale=1.0 / depth_scale_m, depth_trunc=max_depth_m,
        convert_rgb_to_intensity=False,
    )


def masked_depth(depth: np.ndarray, mask: np.ndarray, min_depth_m: float,
                 max_depth_m: float, depth_scale_m: float) -> np.ndarray:
    selected = depth.copy()
    valid = mask > 0
    selected[~valid] = 0
    values = selected.astype(np.float32) * depth_scale_m
    selected[(values < min_depth_m) | (values > max_depth_m)] = 0
    if np.count_nonzero(selected) < 128:
        raise ValueError("object mask has fewer than 128 valid depth pixels")
    return selected


def reconstruct(args: argparse.Namespace) -> int:
    try:
        import cv2
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("reconstruction requires OpenCV, NumPy and Open3D") from exc

    root = args.capture.resolve()
    metadata = load_metadata(root)
    records = load_frame_index(root)[::args.frame_stride]
    if args.max_frames:
        records = records[:args.max_frames]
    if len(records) < 2:
        raise ValueError("reconstruction needs at least two frames")
    if not args.mask_dir.is_dir():
        raise FileNotFoundError(f"object mask directory not found: {args.mask_dir}")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("output mesh or report already exists")

    height, width = int(metadata["height"]), int(metadata["width"])
    scale_m = float(metadata["depth_scale_m"])
    matrix = np.asarray(metadata["camera_matrix"], dtype=float)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
    )
    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = args.odometry_max_depth_diff
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_mm / 1000.0,
        sdf_trunc=args.sdf_trunc_mm / 1000.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    previous = None
    first_from_previous = np.eye(4)
    pose_records = []
    integrated = 0
    for record in records:
        rgb, raw_depth = read_rgbd(root, record, width, height)
        mask_path = args.mask_dir / f"{record['index']:06d}_mask.png"
        mask = read_png(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != (height, width):
            raise ValueError(f"mask dimensions do not match frame {record['index']}: {mask_path}")
        selected_depth = masked_depth(
            raw_depth, mask, args.min_depth_m, args.max_depth_m, scale_m,
        )
        full_rgbd = make_rgbd(o3d, rgb, raw_depth, scale_m, args.max_depth_m)
        object_rgbd = make_rgbd(o3d, rgb, selected_depth, scale_m, args.max_depth_m)
        if previous is not None:
            success, previous_from_current, information = (
                o3d.pipelines.odometry.compute_rgbd_odometry(
                    full_rgbd, previous, intrinsic, np.eye(4), jacobian, option,
                )
            )
            if not success or not np.isfinite(previous_from_current).all():
                pose_records.append({"index": record["index"], "integrated": False,
                                     "reason": "rgbd_odometry_failed"})
                print(json.dumps(pose_records[-1]), flush=True)
                continue
            first_from_previous = first_from_previous @ previous_from_current
        volume.integrate(object_rgbd, intrinsic, np.linalg.inv(first_from_previous))
        previous = full_rgbd
        integrated += 1
        pose_records.append({
            "index": record["index"],
            "integrated": True,
            "T_first_camera_current": first_from_previous.tolist(),
        })
        print(json.dumps({"integrated": integrated, "index": record["index"]}), flush=True)

    if integrated < 2:
        raise RuntimeError("fewer than two frames could be registered; recapture with slower motion")
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise RuntimeError("TSDF integration produced an empty mesh")
    bounds_m = [np.asarray(mesh.get_min_bound()).tolist(), np.asarray(mesh.get_max_bound()).tolist()]
    mesh.scale(1000.0, center=np.zeros(3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh):
        raise RuntimeError(f"could not write mesh: {args.output}")
    report = {
        "capture": str(root),
        "mesh": str(args.output),
        "mesh_unit": "mm",
        "coordinate_frame": "first_camera",
        "requested_frames": len(records),
        "integrated_frames": integrated,
        "voxel_mm": args.voxel_mm,
        "sdf_trunc_mm": args.sdf_trunc_mm,
        "depth_scale_m": scale_m,
        "mesh_bounds_m_before_export": bounds_m,
        "geometry_validated": False,
        "frames": pose_records,
    }
    write_json(args.report, report)
    print(json.dumps({"mesh": str(args.output), "vertices": len(mesh.vertices),
                      "triangles": len(mesh.triangles), "geometry_validated": False}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True,
                        help="Binary PNG masks named 000000_mask.png, matching capture indices")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY, exported in millimetres")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--voxel-mm", type=float, default=1.5)
    parser.add_argument("--sdf-trunc-mm", type=float, default=6.0)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--odometry-max-depth-diff", type=float, default=0.07)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if args.report is None:
        args.report = args.output.with_suffix(".json")
    if min(args.voxel_mm, args.sdf_trunc_mm, args.odometry_max_depth_diff) <= 0:
        parser.error("voxel, SDF truncation and odometry depth difference must be positive")
    if not 0 <= args.min_depth_m < args.max_depth_m:
        parser.error("depth range must satisfy 0 <= min < max")
    if args.frame_stride <= 0 or args.max_frames < 0:
        parser.error("--frame-stride must be positive and --max-frames nonnegative")
    return reconstruct(args)


if __name__ == "__main__":
    raise SystemExit(main())
