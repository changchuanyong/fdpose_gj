# fdpose_bringup

在仓库根目录构建感知与启动包。`fdpose_live.launch.py` 是唯一 launch，运行 `fdpose_perception/live_pose.py`：`camera` 选 `orbbec-ros`（默认）或 `d415`，不传 `--show`，所以经 launch 启动默认就是无头位姿节点。权重、网格与 `third_party/` 从 `project_root` 读取。Jetson 上的 `python` 启动参数默认指向 `~/venvs/ev-charging-d415-gpu/bin/python`。

```bash
source /opt/ros/humble/setup.bash
export FDPOSE_ROOT="$PWD"
colcon build --symlink-install --base-paths src third_party/OrbbecSDK_ROS2 --packages-up-to fdpose_bringup
source install/local_setup.bash
ros2 launch fdpose_bringup fdpose_live.launch.py --show-args
```

Gemini 模式需先启动 Orbbec 驱动（完整启动参数见 [run_330s.sh](scripts/run_330s.sh)），在同一 ROS 域中运行。每次指定新的输出目录，`frames:=0` 为持续运行。

```bash
ros2 launch fdpose_bringup fdpose_live.launch.py \
  serial:=CPCD7530009J frames:=0 \
  output:="artifacts/gemini-ros-$(date +%Y%m%d-%H%M%S)"
ros2 topic echo /fdpose/pose --once
```

D415 由 CLI 直连取流，无需任何相机驱动工作区：

```bash
ros2 launch fdpose_bringup fdpose_live.launch.py \
  camera:=d415 serial:=349222063070 frames:=0 \
  output:="artifacts/d415-ros-$(date +%Y%m%d-%H%M%S)"
```

默认话题 `/fdpose/pose` 发布 `geometry_msgs/msg/PoseStamped`，与同帧 JSON 的 `T_camera_target_m` 对应。平移单位为米，四元数按 x、y、z、w 排列。Gemini 使用彩色图像头的光学坐标系和采集时间。D415 使用 `camera_color_optical_frame` 和主机读帧时刻，设备毫秒保存在 JSON。丢失期间停止发布，订阅端应检查时间戳判断数据是否过期。

桌面预览走 [run_330s.sh](scripts/run_330s.sh) 与 [run_d415.sh](scripts/run_d415.sh)，它们给同一个 CLI 加 `--show`。离线回放工具是 `fdpose_perception/replay_drift.py`，与实时共用同一套注册/跟踪/验证原子。