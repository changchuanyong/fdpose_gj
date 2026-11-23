# 接入新零件的实时位姿估计

每次运行跟踪一个指定类别的一件零件。同类有多个实例时，使用置信度最高的实例。运行前准备该零件的分割权重、网格和目标坐标系配置。

## 训练零件分割权重

1. 采集零件在不同视角、光照和背景下的 RGB 图。用实例轮廓标注零件，每个实例对应一条记录。在 Ultralytics 分割数据集中，每条标签为 `class_id x1 y1 x2 y2 ...`，坐标归一化到 `[0, 1]`。
2. 在数据集 YAML 中填写 `train`、`val` 和 `names`。按采集批次划分数据，同一录像的相邻帧放在同一个集合。用独立实物序列检查掩膜边界以及注册、遮挡和恢复表现。
3. 在训练环境中从分割模型微调。下面的 `data.yaml` 是自行准备的数据集配置，训练结束后把 `best.pt` 放到设备的 `weights/`。

   ```bash
   yolo segment train model=yolo11n-seg.pt data=data.yaml imgsz=640
   ```

## 选择网格和目标坐标系

`--mesh` 指向对应零件的网格，`--mesh-units mm` 或 `--mesh-units m` 指定顶点单位。`--geometry` 指向含 `cad_to_target` 的 JSON。该刚体变换把网格坐标转换到目标坐标。JSON 的 `units` 指定变换平移量的单位，支持 `mm` 和 `m`，默认 `mm`，与网格单位分别设置。若以网格坐标系为目标坐标系，可使用单位矩阵。

```json
{
  "units": "mm",
  "cad_to_target": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

## 启动 D415

从仓库根目录运行。示例中的权重、网格和坐标配置均需用实际零件文件替换，`2` 改成权重内对应的类别 ID。

```bash
bash src/fdpose_bringup/scripts/run_d415.sh \
  --mask-backend yolo-seg --weights weights/part-seg.pt --class-id 2 \
  --mesh config/mesh/part.PLY --mesh-units mm --geometry config/geometry/part.json
```

切换到 `--mask-backend sam2` 时使用该零件的 YOLO **检测**权重，并准备 SAM2.1 checkpoint。也可用 `--sam2-checkpoint` 指定自定义路径。初次接入时查看输出目录的 `initial_mask.jpg`、位姿叠加图和 `summary.json`。`poses.jsonl` 的 `T_camera_target_m` 表示目标坐标系到相机光学坐标系的变换，平移单位为米。

## 启动 Gemini 335L / 336L（Jetson）

两款相机共用 [Orbbec ROS 2 适配器](../src/fdpose_camera/fdpose_camera/orbbec_ros.py)，D415 启动不变。Jetson 需要 ROS 2 Humble；Orbbec 驱动源码 vendored 在 `third_party/OrbbecSDK_ROS2`，**随本项目一起 colcon 构建、只装进项目 `install/`**，不写 `/opt/ros/humble`、也不依赖外部驱动工作区；此路径**不依赖** `pyorbbecsdk`。录制重建数据的 [Orbbec SDK 采集脚本](../src/fdpose_tools/fdpose_tools/orbbec_capture.py) 是另一条路径。

若 `third_party/OrbbecSDK_ROS2` 尚未就位：从上游 clone v1.5.21，或从已有编译工作区复制源码 `cp -a <驱动工作区>/src/OrbbecSDK_ROS2 third_party/`，然后按 README 安装节构建。

在两个 Jetson 终端分别执行以下环境初始化。将序列号替换成实际值；两个终端的 ROS 域与序列号保持一致。

```bash
source /opt/ros/humble/setup.bash
source <仓库根>/install/local_setup.bash
export ROS_DOMAIN_ID=117 ROS_LOCALHOST_ONLY=1
read -r -p 'Gemini serial: ' SERIAL
```

终端一运行相机节点，开启帧同步和深度到彩色的软件对齐：

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py \
  camera_name:=gemini serial_number:="$SERIAL" \
  color_width:=1280 color_height:=720 color_fps:=30 \
  depth_width:=1280 depth_height:=720 depth_fps:=30 \
  depth_registration:=true align_mode:=SW enable_frame_sync:=true \
  frame_aggregate_mode:=full_frame enable_depth_scale:=true \
  publish_tf:=false trigger_out_enabled:=false
```

终端二在 `fdpose` 仓库根目录启动位姿程序。将示例的三个零件文件路径替换为实际路径。`--show` 仅在 Jetson 桌面终端有显示服务时使用。

```bash
~/venvs/ev-charging-d415-gpu/bin/python -u -B apps/live_pose.py \
  --camera orbbec-ros --camera-namespace /gemini --serial "$SERIAL" \
  --mask-backend yolo-seg --weights weights/part-seg.pt --class-id 0 \
  --mesh config/mesh/part.PLY --geometry config/geometry/part.json \
  --output "artifacts/gemini-live-$(date +%Y%m%d-%H%M%S)" --frames 60
```

适配器查询并核对实际型号和序列号，匹配 15 ms 内的 RGB-D 帧，检查对齐后的分辨率、光学坐标系和内参。驱动已经按设备的深度比例把 `16UC1` 转为毫米，适配器再换算成米，并用同一去畸变映射处理彩色和深度。位姿程序退出后，在相机终端按 Ctrl+C 停止驱动。

2026-11-18 在 Jetson 上用本适配器读取了 Gemini 336L 的 10 组配对帧。输入为 640×480 彩色及 640×400 原始深度，对齐后均为 640×480。设备身份、帧号与时间戳递增、米制深度输出均通过检查；日志与摘要位于 Jetson 的 `~/fdonpose-git/artifacts/gemini336l-adapter-20261118-165712/`。335L 需按实际流配置验证，两款相机的内参不共用。此次仅验证相机到适配器，不涉及已知距离的深度校验、FoundationPose 实物位姿或长期稳定性。
