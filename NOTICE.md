# NOTICE

本仓库是一个围绕 NVlabs FoundationPose 的充电口位姿工程，包含上游源码的适配补丁与部署脚本。

## 上游组件

- **FoundationPose**（NVlabs，CVPR 2024）：以固定提交拉取到 `third_party/FoundationPose`（不随仓库分发，
  版本见 `config/upstream.json`）。许可见其目录内 `LICENSE`，为 NVIDIA 版权声明，保留所有权利。
- **pytorch3d、nvdiffrast、warp-lang**：由 `scripts/setup/jetson_setup.sh` 从上游源码或 PyPI 安装，
  各自遵循其上游许可，本仓库不重复声明。
- **SAM2**：以固定提交拉取到 `third_party/sam2`（版本见 `config/upstream.json`），经 `sys.path`
  以源码方式使用，部署脚本不安装它。许可见其目录内 `LICENSE` 与 `LICENSE_cctorch`。

## 补丁说明

`patches/0001-platform-compat.patch` 修改 FoundationPose 的 5 个文件
（`Utils.py`、`estimater.py`、`mycpp/CMakeLists.txt`、`mycpp/include/Utils.h`、`run_demo.py`），
用于在官方未支持的平台组合上构建与运行。该补丁属于 FoundationPose 的衍生修改，
再分发时必须同时保留上游 LICENSE 与版权声明。

## 本仓库许可

尚未声明。在明确选择许可证之前，不应假定本仓库代码可被再分发或用于商业用途。
