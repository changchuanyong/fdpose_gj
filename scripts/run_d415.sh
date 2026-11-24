#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

FDPOSE_PYTHON="${FDPOSE_PYTHON:-$HOME/miniforge3/envs/fdpose/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
exec "$FDPOSE_PYTHON" -u -B src/fdpose_perception/fdpose_perception/live_pose.py \
  --serial 349222063070 \
  --mesh config/mesh/socket_mod.PLY \
  --geometry config/geometry/socket_openings_ply.json \
  --weights weights/best.pt \
  --output "artifacts/d415-live-$(date +%Y%m%d-%H%M%S)" \
  --show --frames 0 "$@"
