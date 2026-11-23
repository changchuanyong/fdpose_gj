# 335L 管件 RGB-D 采集与重建使用说明

本文面向当前无 CAD 的 PVC-U 弯头，使用奥比中光 Gemini 335L 采集多角度 RGB-D，再生成 FoundationPose 使用的 PLY 网格。也可改用 RealSense D415 采集（见第三节第 5 小节），重建流程相同。

> 当前状态：采集和重建代码已完成离线检查，但 335L 与 D415 采集脚本均尚未连接相机真机运行，也没有用实物数据验证重建精度。现阶段先完成环境检查和样例采集，不代表已经获得可用模型。

## 一、使用流程

```text
安装并检查 335L 环境
        ↓
固定弯头，环绕采集 RGB-D
        ↓
检查彩色图、原始深度和相机参数
        ↓
生成并人工检查管件掩膜
        ↓
重建 PLY 网格并核对实物尺寸
        ↓
配置目标坐标、检测权重和实时位姿程序
```

如果只是想先准备数据，完成下面的环境检查和样例采集即可。

## 二、安装并检查环境

在连接 335L 的 Windows 电脑或 Jetson 上安装：

- 与主机系统、CPU 架构匹配的 OrbbecSDK v2 和官方 `pyorbbecsdk` Python 包
- NumPy、OpenCV、Open3D

`pyorbbecsdk` 按厂商 SDK 安装，不加入项目的通用 pip 约束，不要直接假定 `pip install pyorbbecsdk` 适用于当前系统。采集接口依据[官方 Python SDK 应用指南](https://orbbec.github.io/pyorbbecsdk/source/4_Application_Guide/ApplicationGuide.html)：Y16 原始深度、帧同步、软件 D2C 和彩色流内参。

先在 Orbbec Viewer 中确认相机能正常显示彩色和深度画面，再检查 Python 包：

```bash
python -c "import pyorbbecsdk; print('Orbbec SDK OK')"
```

如果这里失败，先处理 SDK 安装，不需要启动采集脚本。

## 三、采集弯头 RGB-D

### 1. 摆放和拍摄要求

- 将弯头固定放在桌面，不要手持；拍摄期间不要移动弯头。
- 缓慢移动相机环绕弯头，完整拍到外表面和两个承口，相邻视角保持较大重叠。
- 保留静止的桌面和背景。程序使用完整场景估计相机运动，不使用背景生成管件模型。
- 避免运动模糊和强光直射。白色反光表面出现深度空洞时，调整拍摄角度和照明后重拍。

### 2. 先采集 10 帧样例

在项目根目录运行：

```bash
python src/fdpose_tools/fdpose_tools/orbbec_capture.py \
  --output artifacts/elbow-rgbd-sample \
  --frames 10 \
  --interval-s 0.5
```

Windows CMD 或 PowerShell 可以写成一行：

```powershell
python src/fdpose_tools/fdpose_tools/orbbec_capture.py --output artifacts/elbow-rgbd-sample --frames 10 --interval-s 0.5
```

样例用于检查取流、RGB／深度对齐和深度完整性，不用于完整重建。

### 3. 采集完整环绕序列

样例正常后，换一个新的输出目录：

```bash
python src/fdpose_tools/fdpose_tools/orbbec_capture.py \
  --output artifacts/elbow-rgbd-001 \
  --frames 100 \
  --interval-s 0.2
```

默认保存 100 组 RGB-D，目标采集时间约 20 秒。持续采集可使用 `--frames 0`，按 `Ctrl+C` 停止。

输出目录必须不存在，防止两次采集混入同一序列；再次采集时使用 `elbow-rgbd-002` 等新目录。多相机环境使用 `--serial <序列号>` 指定设备。

默认流参数为彩色 640×480、深度 640×400、30 fps，深度通过软件对齐到彩色图。设备不支持默认档位时，程序会列出可用档位，再显式设置：

- `--color-width`、`--color-height`
- `--depth-width`、`--depth-height`
- `--fps`

### 4. 检查输出

```text
artifacts/elbow-rgbd-001/
├── metadata.json
├── frames.jsonl
└── frames/
    ├── 000000_rgb.png
    ├── 000000_depth.png
    ├── 000001_rgb.png
    ├── 000001_depth.png
    └── ...
```

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 实际流参数、3×3 彩色内参、深度比例、设备信息 |
| `frames.jsonl` | 连续帧编号、RGB／深度路径和设备时间戳 |
| `frames/*_rgb.png` | 对齐并去畸变后的 8 位彩色图 |
| `frames/*_depth.png` | 对齐并去畸变后的原始 16 位深度图 |

深度比例从设备帧读取，不假定 1 个深度单位一定是 1 mm。深度 PNG 是原始 16 位数据，普通看图软件显示很暗或接近黑色不一定是异常；同时检查终端输出的 `valid_depth_pixels` 和 RGB／深度轮廓是否对齐。

### 5. 改用 RealSense D415 采集

没有 335L 时可以用 RealSense D415 采集，输出目录格式与上面完全一致，后续掩膜和重建步骤不变。依赖 `pyrealsense2`（仓库实时位姿程序已在用），取流方式与 `apps/live_pose.py` 一致：彩色 rgb8、深度 z16、`rs.align` 对齐到彩色、brown_conrady 去畸变。

```bash
python src/fdpose_tools/fdpose_tools/realsense_capture.py \
  --output artifacts/elbow-rgbd-d415-001 \
  --frames 100 \
  --interval-s 0.2
```

默认彩色 640×480、深度 640×480、30 fps，可用 `--color-width`、`--color-height`、`--depth-width`、`--depth-height`、`--fps` 调整；多相机环境用 `--serial <序列号>` 指定设备。注意：

- D415 的最小深度距离较远（以官方数据手册为准），拍小管件不能凑得太近，否则深度无效；先采 10 帧样例确认 `valid_depth_pixels` 充足。
- 白色反光表面的深度空洞问题与 335L 相同，调整拍摄角度和照明后重拍。
- 该脚本同样尚未真机验证。

## 四、生成管件掩膜

在采集目录下建立 `masks/`，为每个用于重建的帧准备同名黑白 PNG：

```text
artifacts/elbow-rgbd-001/masks/
├── 000000_mask.png
├── 000001_mask.png
└── ...
```

要求：

- 管件为白色，手、桌面和其他背景为黑色。
- 分辨率与对应 RGB 图相同。
- 承口边缘尽量准确，不要把背景并入管件。
- 可使用图像标注工具或 SAM2 辅助生成，但需要人工抽查。

SAM2 视频传播可批量生成：选一帧管件尽量完整的帧给框提示，向前向后传播整段序列（需要 CUDA 与 `weights/sam2.1_hiera_small.pt`）：

```bash
python src/fdpose_tools/fdpose_tools/make_masks_sam2.py \
  --capture artifacts/elbow-rgbd-001 \
  --output artifacts/elbow-rgbd-001/masks \
  --prompt-frame 250 --box 240 210 437 479
```

`--box` 为提示帧上管件的外接框 `x0 y0 x1 y1`；手持或移动管件的尾段用 `--max-frames` 截掉。生成后必须目视抽查掩膜（管件白、背景黑），确认承口边缘无误再重建。

掩膜只控制融合进模型的区域；采集时必须保留完整场景，不能只保存掩膜内的 RGB-D。

## 五、重建三维网格

```bash
python src/fdpose_tools/fdpose_tools/reconstruct_rgbd.py \
  --capture artifacts/elbow-rgbd-001 \
  --mask-dir artifacts/elbow-rgbd-001/masks \
  --output config/mesh/elbow_dn50.PLY
```

成功后得到：

- `config/mesh/elbow_dn50.PLY`：三维网格，单位为**毫米**，坐标系暂时是第一帧相机坐标系。
- `config/mesh/elbow_dn50.json`：融合帧数、配准失败帧、深度比例和网格包围盒。

程序对相邻完整 RGB-D 帧做里程计，再将掩膜内的管件深度融合为 TSDF 网格。默认 1.5 mm 体素只是 dn50 管件的初始参数，不代表重建精度。Open3D 0.20 的 ScalableTSDFVolume 有深度缩放回归（融合结果为空网格），脚本改用首帧管件中心居中的 UniformTSDFVolume；`--volume-length-m` 设定积分立方体边长（默认 1 m），分辨率由体素尺寸导出。每步里程计另用掩膜点云对齐残差校验（`--odometry-max-residual-mm`，默认 10 mm），超差即跳帧并保留前一锚点。手持采集必须慢到相邻帧无运动模糊（建议 ≤0.1 m/s、整段 20–30 s 以上）：快动会同时造成深度大面积缺失与里程计崩溃，残差门控会拒掉绝大多数帧，可用帧不足则重建失败。

如果大量出现 `rgbd_odometry_failed`，重新采集时降低移动速度、增加相邻视角重叠，并保留有纹理的静止背景。如果网格包含背景或内壁破洞，先修正掩膜和原始深度，不要只靠网格平滑掩盖问题。

## 六、检查模型能否使用

使用卡尺实测承口内径、外径、承口深度和总长，与网格逐项核对。误差未确认前，不能把模型用于插装控制。重建报告固定写入：

```json
"geometry_validated": false
```

只有另有实测验收记录后，才能把几何视为已验证。

## 七、接入位姿估计

重建网格只是替换充电口的第一步，正式运行还需要：

- 根据选定的承口中心和插入方向建立新的 `cad_to_target` 几何配置。
- 将 YOLO 检测权重换成该管件的权重，或先用手工框做离线验证。
- Orbbec 实时取流已由 `fdpose_camera/orbbec_ros.py` 实现；仍需用 335L 检查对齐、深度比例和投影（两款相机内参不共用）。

目前不能仅凭一张照片或未采集的数据声称已经得到可用的三维模型。
