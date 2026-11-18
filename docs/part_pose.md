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
bash scripts/run_d415.sh \
  --mask-backend yolo-seg --weights weights/part-seg.pt --class-id 2 \
  --mesh assets/mesh/part.PLY --mesh-units mm --geometry config/geometry/part.json
```

切换到 `--mask-backend sam2` 时使用该零件的 YOLO **检测**权重，并准备 SAM2.1 checkpoint。也可用 `--sam2-checkpoint` 指定自定义路径。初次接入时查看输出目录的 `initial_mask.jpg`、位姿叠加图和 `summary.json`。`poses.jsonl` 的 `T_camera_target_m` 表示目标坐标系到相机光学坐标系的变换，平移单位为米。

## 后续接入 Gemini 335L

[相机模块](../apps/rgbd_camera.py) 定义实时取流接口。适配器提供与 RGB 像素对应的内参 `K`、设备元数据，并按帧返回 RGB8、对齐到彩色的米制深度、帧号和毫秒时间戳。可以从[现有 Orbbec 采集脚本](../scripts/pipeline/orbbec_capture.py) 复用同步取流、彩色解码和对齐处理。

切换默认相机前，在部署设备上检查 335L 实际使用的彩色内参、RGB 与深度的对齐、已知距离的深度读数，以及同一零件的注册和跟踪恢复。当前实时入口为 D415，Gemini 335L 实时适配待接入。
