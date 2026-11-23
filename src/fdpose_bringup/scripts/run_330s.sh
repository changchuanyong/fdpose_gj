#!/usr/bin/env bash
set -eo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

source /opt/ros/humble/setup.bash
source install/local_setup.bash
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-117}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

serial="${ORBBEC_SERIAL:-CPCD7530009J}"
run_dir="artifacts/gemini-330s-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"

setsid ros2 launch orbbec_camera gemini_330_series.launch.py \
  camera_name:=gemini serial_number:="$serial" \
  color_width:=1280 color_height:=720 color_fps:=30 \
  depth_width:=1280 depth_height:=720 depth_fps:=30 \
  depth_registration:=true align_mode:=SW enable_frame_sync:=true \
  frame_aggregate_mode:=full_frame enable_depth_scale:=true \
  publish_tf:=false trigger_out_enabled:=false \
  >"$run_dir/camera.log" 2>&1 &
camera_pid=$!

stop_camera() {
  kill -INT -- "-$camera_pid" 2>/dev/null || true
  for ((i=0; i<50; i++)); do
    kill -0 -- "-$camera_pid" 2>/dev/null || break
    sleep 0.1
  done
  kill -TERM -- "-$camera_pid" 2>/dev/null || true
  wait "$camera_pid" 2>/dev/null || true
}
trap stop_camera EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=false
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  if ! kill -0 "$camera_pid" 2>/dev/null; then
    echo "Orbbec camera launch exited; see $run_dir/camera.log" >&2
    exit 1
  fi
  if timeout 6s ros2 service list --no-daemon --spin-time 2 2>/dev/null \
    | grep -Fx /gemini/get_device_info >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "Timed out waiting for /gemini/get_device_info; see $run_dir/camera.log" >&2
  exit 1
fi

"$HOME/venvs/ev-charging-d415-gpu/bin/python" -u -B apps/live_pose.py \
  --camera orbbec-ros --camera-namespace /gemini --serial "$serial" \
  --mesh config/mesh/socket_mod.PLY \
  --geometry config/geometry/socket_openings_ply.json \
  --weights weights/best.pt \
  --output "$run_dir/poses" \
  --show --frames 0 "$@"
