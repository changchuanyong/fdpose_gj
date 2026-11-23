# patches

上游适配补丁。目标仓库与固定提交见 `config/upstream.json`。命名沿用 git format-patch / quilt 的编号系列（`0001-<目的>.patch`），编号即应用顺序；多上游时再分 `patches/<upstream>/` 子目录。

| 补丁 | 目标 | 改动文件 | 用途 |
| --- | --- | --- | --- |
| `0001-platform-compat.patch` | FoundationPose `a1b694b8` | 5 个：`Utils.py`、`estimater.py`、`mycpp/CMakeLists.txt`、`mycpp/include/Utils.h`、`run_demo.py` | warp 内核缓存目录可配置、`FDONPOSE_TMP`/`FDONPOSE_WARP_FALLBACK` 开关、mycpp 的 MSVC 与 aarch64/GCC 构建分支等跨平台适配；Jetson 部署同样应用 |

手工应用：

```bash
cd third_party/FoundationPose
git apply ../../patches/0001-platform-compat.patch
```

Jetson 侧由 `scripts/jetson_setup.sh repo` 自动应用，并校验恰好 5 个文件被修改。
该补丁是上游的衍生修改，再分发时必须保留上游 LICENSE 与版权声明。
