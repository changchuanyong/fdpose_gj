# fdonpose

在 Jetson AGX Orin 上用 RealSense D415 或 Orbbec Gemini 336L 获取 RGB-D，估计零件的 6D 位姿。默认目标是充电口：YOLO 检测加 SAM2.1 生成掩膜，FoundationPose 完成注册与跟踪；也可换用 YOLO-seg。以下入口只运行视觉估计，不启动机械臂控制器，也不发送运动指令。

## 快速启动

**使用新相机 Gemini 336L：** 彩色和深度默认都以原生 1280×720、30 fps 采集，深度再配准到彩色光学坐标系。把相机接到已部署的 Jetson，让充电口完整入镜，在 **Jetson 桌面终端**运行：

```bash
bash ~/fdonpose/scripts/run_330s.sh
```

这一条命令会同时启动 330 系列相机驱动和 FoundationPose，无需另开位姿程序。只有使用 **RealSense D415** 时才运行下面的 `run_d415.sh`。无显示器时见下文的 SSH 入口；首次部署见 [Jetson 安装与验证](docs/jetson_setup.md)。

### RealSense D415

```bash
bash ~/fdonpose/scripts/run_d415.sh
```

[D415 启动脚本](scripts/run_d415.sh) 默认使用这台 D415 的 SDK 序列号 `349222063070`，从项目目录读取模型、坐标配置和权重，并打开桌面窗口持续跟踪。按 **Q** 或 **Esc** 退出。

| 用途 | 参数 |
| --- | --- |
| 默认取流 | 640×480、30 fps |
| 以 720p 采集 | 追加 `--stream-width 1280 --stream-height 720` |
| 更换 D415 | 追加 `--serial <SDK 序列号>` |
| 按官方默认迭代次数初始化 | 追加 `--est-refine-iter 5`，本项目默认 3 轮 |

窗口显示 `Aim at the target` 时，让目标完整入镜。出现 `Target clipped` 时，把目标移离画面边缘。

如果 `FileNotFoundError` 指向 `~/ev-charging-d415/`，请改用上面的启动脚本。YOLO 权重位于 `~/fdonpose/weights/best.pt`，SAM2 权重位于 `~/fdonpose/weights/sam2.1_hiera_small.pt`。

### Gemini 336L / 335L

[Gemini 启动脚本](scripts/run_330s.sh) 需要已部署的 ROS 2 Humble、`~/HKR_WorkSpace/ros2_ws` Orbbec 驱动工作区和项目 Python 环境。脚本启动相机，等待设备服务就绪，再打开桌面窗口持续跟踪。默认使用 336L 序列号 `CPCD7530009J`；换用 335L 或其他 336L 时，在命令前设置 `ORBBEC_SERIAL`。按 **Q** 或 **Esc** 退出，脚本会停止相机驱动。

**实际调用链：** `run_330s.sh` 先启动 `orbbec_camera` 的 `gemini_330_series.launch.py`；等 `/gemini/get_device_info` 就绪后，再调用[共用位姿程序](apps/jetson_foundationpose_d415.py)。虽然文件名带 `d415`，脚本显式传入 `--camera orbbec-ros --camera-namespace /gemini`，因此读取的是 Gemini 的 RGB-D。

**SSH / 无显示器入口：** 桌面脚本固定传入 `--show`，无显示器时不要直接运行它。按[零件接入指南的 Gemini 双终端步骤](docs/part_pose.md)初始化两个 Jetson 终端，在终端一启动相机驱动。终端二使用相同 ROS 域和序列号，在仓库根目录运行以下充电口测试；结束后，在终端一按 **Ctrl+C** 停止相机。

```bash
cd ~/fdonpose
~/venvs/ev-charging-d415-gpu/bin/python -u -B apps/jetson_foundationpose_d415.py \
  --camera orbbec-ros --camera-namespace /gemini --serial "$SERIAL" \
  --weights weights/best.pt --mask-backend sam2 --mesh assets/mesh/socket_mod.PLY \
  --geometry config/geometry/socket_openings_ply.json \
  --output "artifacts/gemini-headless-$(date +%Y%m%d-%H%M%S)" --frames 60
```

两个脚本默认使用 YOLO 检测加 SAM2，并支持追加位姿参数。已有零件分割权重时，追加 `--mask-backend yolo-seg --weights <分割权重>`。更换零件网格、坐标配置，或手动启动 Orbbec 驱动，见[零件接入指南](docs/part_pose.md)。

## 运行与输出

Gemini 桌面窗口左侧独立刷新实时彩色图，右侧缩略图显示推理所用的源帧、位姿或状态，以及对应帧号和 ROS 时间戳。旧位姿不会叠到新的实时图上。`summary.json` 中 `preview_fps` 表示实时画面的刷新速度，`tracking_fps` 表示跟踪位姿输出速度。

应用将深度对齐到彩色画面。首帧初始化后，FoundationPose 每帧精化 2 轮，窗口显示 3D 框和 XYZ 坐标轴。首个跟踪帧、之后约每 0.6 秒以及姿态突变时，都会核对投影、检测框和模型表面深度。只有完整复核通过的位姿才用于恢复缓存。失配时窗口撤掉位姿框，先尝试从最后确认位姿恢复，失败后重新检测、分割并注册。

D415 结果保存在 `artifacts/d415-live-*`，Gemini 结果保存在 `artifacts/gemini-330s-*/poses/`，相机节点日志在同次运行目录的 `camera.log`。网格 `socket_mod.PLY` 与 `config/geometry/socket_openings_ply.json` 配套使用，后者定义从网格坐标到充电口目标坐标的变换。

| 文件 | 内容 |
| --- | --- |
| `poses.jsonl` | 输出帧的 `T_camera_target_m`，将目标坐标转换到相机光学坐标，平移单位为米 |
| `initial_mask.jpg` | 首次通过注册检查的目标掩膜叠加图 |
| `first_pose.jpg`、`last_pose.jpg` | 首尾位姿叠加图 |
| `summary.json` | 首帧注册耗时、实际跟踪帧率和恢复、重置次数 |
| `drift/<帧号>/` | 复核失配时的前后 RGB-D、内参、候选/恢复位姿及验证指标 |

可用 `src/fdpose_tools/fdpose_tools/replay_drift.py <漂移目录> --output <结果.json>` 回放同一段输入；默认在原生分辨率和 640 像素宽度下分别跟踪、复核并计时。

## 实测结果

| Jetson 测试 | 结果 |
| --- | --- |
| Gemini 336L，2026-11-19，双路原生 1280×720、30 fps，静止实物充电口，桌面窗口 | 预览 28.37 fps（531 帧）；首次注册 7.46 s，随后 60 个跟踪位姿，6.22 fps，跟踪延迟 P95 为 196.77 ms。15 次跟踪复核，无恢复或重置 |
| Gemini 336L，2026-11-19，640×480 RGB-D 采集 | 对齐后彩色和深度均为 640×480；60/60 组通过，29.96 组/秒，最大配对时差 0.0192 ms |
| Gemini 336L，2026-11-19，640×480 静止实物充电口 | 首次注册 7.09 s；连续跟踪 60 帧，8.93 fps。11 次跟踪复核全部通过，无恢复或重置，首尾叠加框均落在充电口上。此结果不代表绝对精度，也不覆盖 335L 或机械臂联动 |
| D415 实物充电口，1280×720，桌面窗口 | 首帧 5 轮注册约 10.1 s。连续输出 60 个跟踪位姿，约 5.9 fps。15 次投影检查通过，首尾叠加图贴合充电口 |
| 同帧注册对照，640×480，同一组 RGB-D 和 SAM2 掩膜 | 5 轮约 6–7 s，3 轮约 4.4 s，2 轮约 3.6 s。相对实时脚本的 5 轮结果，3 轮最大差异约 0.12 mm / 0.28°，2 轮最大旋转差约 1.83° |
| FoundationPose 官方 `mustard0` 序列 | 737/737 帧，单次耗时 139–142 s，见 [Jetson 验证记录](docs/jetson_setup.md#实测结果) |

同帧对照衡量不同设置之间的位姿差异。绝对 6D 精度与实物插装仍待独立验证。实物窗口与同帧对照的报告目录分别为 `d415-sam2-final-20261111-163114` 和 `d415-official-compare-20261111-172556`，位于实验环境的 `artifacts/` 下。

## 文档与目录

- [ROS 2 Humble 启动与位姿话题](src/fdpose_bringup/README.md) · 包构建、启动参数、光学坐标系和时间戳

- [Jetson 安装与验证](docs/jetson_setup.md) · 从零部署、固定版本、平台问题和官方样例复现
- [充电口离线回放](docs/charging_replay.md) · 从 `robot-ev-charging` 数据记录运行 YOLO、SAM2 和 FoundationPose
- [管件 RGB-D 重建](docs/pipe_rgbd.md) · 奥比中光 335L / RealSense D415 采集、网格重建与接入条件（尚未真机验证）
- [标定板 v2](docs/tag_mount.md) · 双 M6 螺栓安装、两孔试装件和打印说明
- [Windows 原生验证归档](docs/env/windows-validation-20261105.md) · 历史环境、验证数据和编译排障
- [文档索引](docs/index.md) · [上游归属与许可](NOTICE.md)

```text
src/
  fdpose_camera/      D415 与 Gemini RGB-D 接入 · D415 发布节点
  fdpose_perception/  几何 · 流程 · 验证 · 记录 · 适配层 · 无头 ROS 节点
  fdpose_bringup/     组合启动（CLI / 节点 / D415）
  fdpose_tools/       漂移回放与离线评估
apps/        实时 CLI 入口与桌面预览（薄前端，共用 PosePipeline）
scripts/     setup/ 部署 · pipeline/ 采集重建 · eval/ 评估 · probes/ 环境探针
config/      运行时约束、上游版本和几何配置
assets/      mesh/ 网格模型 · print/ 打印件
docs/        部署、流程和历史记录
patches/     上游适配补丁
tests/       冒烟测试
```

`third_party/` 的上游源码按 [固定版本清单](config/upstream.json) 部署，运行输出写入 `artifacts/`。两者按 `.gitignore` 管理。适配补丁见 [patches/README.md](patches/README.md)。

