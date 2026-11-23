# fdpose_bringup

在仓库根目录构建相机、感知与启动包。`fdpose_bringup` 组合三种入口：桌面 CLI（`fdpose_live`）、无头 Gemini 节点（`fdpose_node`）、D415 发布节点加位姿节点（`fdpose_d415`）。权重、网格与 `third_party/` 从 `project_root` 读取。Jetson 上的 `python` 启动参数默认指向 `~/venvs/ev-charging-d415-gpu/bin/python`。

```bash
source /opt/ros/humble/setup.bash
export FDPOSE_ROOT="$PWD"
colcon build --symlink-install --base-paths src --packages-up-to fdpose_bringup
source install/local_setup.bash
ros2 launch fdpose_bringup fdpose_live.launch.py --show-args
```

Gemini 模式需先按[零件接入指南](../../docs/part_pose.md)启动驱动，在同一 ROS 域中运行。每次指定新的输出目录。

```bash
ros2 launch fdpose_bringup fdpose_live.launch.py \
  serial:=CPCD7530009J \
  output:="artifacts/gemini-ros-$(date +%Y%m%d-%H%M%S)"
```

默认话题 `/fdpose/pose` 发布 `geometry_msgs/msg/PoseStamped`，与同帧 JSON 的 `T_camera_target_m` 对应。平移单位为米，四元数按 x、y、z、w 排列。Gemini 使用彩色图像头的光学坐标系和采集时间。D415 使用 `camera_color_optical_frame` 和主机读帧时刻，设备毫秒保存在 JSON。丢失期间停止发布，订阅端应检查时间戳判断数据是否过期。旧 CLI 通过 `--ros-pose-topic /fdpose/pose` 开启输出。

## 无头 ROS 节点

`fdpose_node.launch.py` 运行 `fdpose_perception/node.py`：订阅 Gemini RGB-D 话题，复用与 CLI 相同的 `PosePipeline` 跟踪流程，用节点自身的 publisher 发布位姿（无桌面预览）。需先启动 Orbbec 驱动。

```bash
ros2 launch fdpose_bringup fdpose_node.launch.py \
  serial:=CPCD7530009J \
  output:="artifacts/gemini-node-$(date +%Y%m%d-%H%M%S)"
ros2 topic echo /fdpose/pose --once
```

`fdpose_d415.launch.py` 同时启动 `fdpose_camera/d415_node.py`（把 D415 对齐 RGB-D 发布为 `{ns}/color/image_raw`、`{ns}/depth/image_raw` 与 CameraInfo，主机读帧时刻打时间戳）和位姿节点（`--camera d415`），无需 Orbbec 驱动。

```bash
ros2 launch fdpose_bringup fdpose_d415.launch.py \
  serial:=349222063070 \
  output:="artifacts/d415-node-$(date +%Y%m%d-%H%M%S)"
```

离线回放工具位于 `fdpose_tools`（`replay_drift.py`），与实时共用同一套注册/跟踪/验证原子。
