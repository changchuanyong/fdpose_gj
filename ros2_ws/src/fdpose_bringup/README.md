# fdpose_bringup

在仓库根目录构建两个包。启动时 `fdpose_bringup` 使用安装后的 `fdpose_perception` 代码，权重、网格与 `third_party/` 从 `project_root` 读取。Jetson 上的 `python` 启动参数默认指向 `~/venvs/ev-charging-d415-gpu/bin/python`。

```bash
source /opt/ros/humble/setup.bash
export FDPOSE_ROOT="$PWD"
cd ros2_ws
colcon build --base-paths src --packages-up-to fdpose_bringup
source install/local_setup.bash
ros2 launch fdpose_bringup fdpose_live.launch.py --show-args
```

Gemini 模式需先按[零件接入指南](../../../docs/part_pose.md)启动驱动，在同一 ROS 域中运行。每次指定新的输出目录。

```bash
ros2 launch fdpose_bringup fdpose_live.launch.py \
  serial:=CPCD7530009J \
  output:="artifacts/gemini-ros-$(date +%Y%m%d-%H%M%S)"
```

默认话题 `/fdpose/pose` 发布 `geometry_msgs/msg/PoseStamped`，与同帧 JSON 的 `T_camera_target_m` 对应。平移单位为米，四元数按 x、y、z、w 排列。Gemini 使用彩色图像头的光学坐标系和采集时间。D415 使用 `camera_color_optical_frame` 和主机读帧时刻，设备毫秒保存在 JSON。丢失期间停止发布，订阅端应检查时间戳判断数据是否过期。旧 CLI 通过 `--ros-pose-topic /fdpose/pose` 开启输出。
