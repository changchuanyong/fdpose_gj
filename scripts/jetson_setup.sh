#!/usr/bin/env bash
# FoundationPose 在 Jetson AGX Orin（L4T R36 / CUDA 12.6 或 R39 / CUDA 13.2）上的部署脚本。
#
# 在 Jetson 本地执行，只写 $HOME，不安装或修改任何系统组件。
# 用法：
#   bash jetson_setup.sh repo    # 固定提交并应用补丁
#   bash jetson_setup.sh conda   # 创建 Conda 环境，装 torch/torchvision 与 requirements
#   bash jetson_setup.sh ext     # 构建 pytorch3d 与 nvdiffrast
#   bash jetson_setup.sh mycpp   # 构建 mycpp 扩展
#   bash jetson_setup.sh all     # 依次执行以上四步
#
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
REPO="${REPO:-$PROJECT_ROOT/third_party/FoundationPose}"
CONDA_EXE="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-$HOME/miniforge3/envs/fdpose}"
WORK="${WORK:-$PROJECT_ROOT/.runtime/jetson-setup}"
PATCH="${PATCH:-$PROJECT_ROOT/patches/0001-platform-compat.patch}"
REQ="${REQ:-$PROJECT_ROOT/config/runtime-jetson.txt}"
# 与 config/upstream.json 的 foundationpose.commit 一致；stage_repo 不依赖 Python。
COMMIT="a1b694b83e633c2cb6115b9063d940a687759392"
# torch 2.9.1 Jetson 轮子的 libtorch_cuda.so 依赖 libcudss.so.0，而 JetPack 6.2 不自带 cuDSS。
CUDSS_VERSION="${CUDSS_VERSION:-0.8.0.10}"
# 可选下载代理，例如 http://127.0.0.1:7897（本机 Clash）。直连 jetson 轮子源只有 20–75 KB/s。
PROXY="${PROXY:-}"
PY="$CONDA_ENV/bin/python"
STAGE="${1:-all}"
export PYTHONNOUSERSITE=1

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
select_platform() {
  local release
  release="$(sed -n 's/^# R\([0-9][0-9]*\) (release).*/\1/p' /etc/nv_tegra_release 2>/dev/null || true)"
  case "$release" in
    36)
      PYTHON_VERSION=3.10
      CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
      TORCH_INDEX="${TORCH_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
      TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
      ;;
    39)
      PYTHON_VERSION=3.12
      CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
      TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu132}"
      TORCH_VERSION="${TORCH_VERSION:-2.12.0+cu132}"
      TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.27.0+cu132}"
      ;;
    *) die "unsupported L4T release '${release:-unknown}'; expected R36 or R39" ;;
  esac
}

check_python() {
  [ -x "$PY" ] || die "Python not found: $PY (run the conda stage first)"
  "$PY" -c 'import sys; sys.exit(0 if f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1] else 1)' \
    "$PYTHON_VERSION" || die "$CONDA_ENV needs Python $PYTHON_VERSION; use a separate CONDA_ENV"
}

if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY"
  log "using proxy $PROXY"
fi

stage_repo() {
  [ -f "$PATCH" ] || die "patch not found: $PATCH"
  if [ ! -d "$REPO/.git" ]; then
    log "cloning NVlabs/FoundationPose into $REPO"
    git clone https://github.com/NVlabs/FoundationPose.git "$REPO"
  fi
  log "checking out $COMMIT"
  if [ "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    git -C "$REPO" fetch --all --quiet
    git -C "$REPO" checkout --force "$COMMIT"
  fi
  [ "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT" ] || die "HEAD is not $COMMIT"
  if ! git -C "$REPO" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    git -C "$REPO" apply "$PATCH" || die "patch failed to apply"
  fi
  local changed
  changed="$(git -C "$REPO" diff --name-only | wc -l)"
  [ "$changed" -eq 5 ] || die "expected 5 patched files, got $changed"
  log "repo ready: $(git -C "$REPO" rev-parse --short HEAD) + patch (5 files)"
}

install_r36_torch() {
  log "downloading Jetson torch wheels from $TORCH_INDEX"
  mkdir -p "$WORK/wheels"
  if ! compgen -G "$WORK/wheels/torch-$TORCH_VERSION-*.whl" >/dev/null \
      || ! compgen -G "$WORK/wheels/torchvision-$TORCHVISION_VERSION-*.whl" >/dev/null; then
    "$PY" -m pip download --no-deps --dest "$WORK/wheels" --index-url "$TORCH_INDEX" \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  fi
  log "installing the downloaded Jetson torch wheels without pulling non-Jetson PyPI builds"
  "$PY" -m pip install --no-deps "$WORK/wheels"/torch-"$TORCH_VERSION"-*.whl \
    "$WORK/wheels"/torchvision-"$TORCHVISION_VERSION"-*.whl
  log "installing cuDSS $CUDSS_VERSION and linking it into torch/lib"
  "$PY" -m pip install -q "nvidia-cudss-cu12==$CUDSS_VERSION"
  local site
  site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  for lib in "$site"/nvidia/*/lib/libcudss*.so*; do
    [ -e "$lib" ] && ln -sf "$lib" "$site/torch/lib/$(basename "$lib")"
  done
  build_cusolver_shim "$site"
}

stage_conda() {
  [ -f "$REQ" ] || die "requirements not found: $REQ"
  [ -x "$CONDA_EXE" ] || die "conda not found: $CONDA_EXE"
  if [ ! -x "$PY" ]; then
    log "creating Conda environment $CONDA_ENV"
    "$CONDA_EXE" create --yes --prefix "$CONDA_ENV" "python=$PYTHON_VERSION" pip tk
  fi
  check_python
  local req_pattern='^[[:space:]]*ultralytics([[:space:]<=>].*)?$'
  if [ "$PYTHON_VERSION" = 3.12 ]; then
    # R39 uses the wheel's CUDA 13 libraries, not the R36 cuDSS/cuSolver shims.
    "$PY" -m pip install -q --upgrade pip wheel
    "$PY" -m pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    req_pattern+='|^[[:space:]]*nvidia-cudss-cu12=='
  else
    "$PY" -m pip install -q --upgrade pip setuptools wheel
    install_r36_torch
  fi
  "$PY" -m pip install -q pybind11 ninja
  log "installing requirements from $REQ"
  grep -Ev "$req_pattern" "$REQ" > "$WORK/runtime-requirements.txt"
  printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$WORK/torch-constraints.txt"
  "$PY" -m pip install -c "$WORK/torch-constraints.txt" -r "$WORK/runtime-requirements.txt"
  # Ultralytics declares opencv-python; this project intentionally provides headless OpenCV 5.x instead.
  "$PY" -m pip install --no-deps "ultralytics==8.4.210"
  if "$PY" -m pip show opencv-python >/dev/null 2>&1; then
    die "remove the conflicting opencv-python package; use opencv-python-headless only"
  fi
  "$PY" -c 'import cv2; assert cv2.__version__.startswith("5."), cv2.__version__'
  "$PY" -m pip freeze > "$WORK/logs/pip-freeze.txt"
  log "Conda environment ready: $("$PY" -c 'import torch;print(torch.__version__)')"
}

# torch 2.9.1 的 Jetson 轮子按 CUDA 12.8/12.9 构建，libtorch_cuda_linalg.so 需要
# cusolverDnXsyevBatched*；JetPack 6.2 的 CUDA 12.6 没有这两个符号，dlopen 会失败。
# 这里只补这两个符号（FoundationPose 只用 inv/solve，不调用 batched eigh），
# 并用 .pth 在解释器启动时以 RTLD_GLOBAL 预加载，避免依赖 LD_LIBRARY_PATH。
build_cusolver_shim() {
  local site="$1"
  local dir="$WORK/shim"
  mkdir -p "$dir"
  cat > "$dir/jetpack_cusolver_compat.c" <<'EOS'
#include <stdio.h>
#include <stdlib.h>

void cusolverDnXsyevBatched(void) {
  fprintf(stderr, "cusolverDnXsyevBatched unavailable on JetPack 6.2 CUDA 12.6\n");
  abort();
}

int cusolverDnXsyevBatched_bufferSize(void) {
  return 1;
}
EOS
  cat > "$dir/jetpack_cusolver_compat.map" <<'EOS'
libcusolver.so.11 {
  global:
    cusolverDnXsyevBatched;
    cusolverDnXsyevBatched_bufferSize;
  local: *;
};
EOS
  log "building cuSolver compatibility shim"
  gcc -shared -fPIC -O2 -o "$site/libcusolver_jetpack_compat.so" \
    "$dir/jetpack_cusolver_compat.c" -Wl,--version-script="$dir/jetpack_cusolver_compat.map"
  cat > "$site/zz_jetpack_cusolver_preload.py" <<'EOS'
import ctypes
import os

_shim = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcusolver_jetpack_compat.so")
if os.path.exists(_shim):
    try:
        ctypes.CDLL(_shim, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
EOS
  printf 'import zz_jetpack_cusolver_preload\n' > "$site/zz_jetpack_cusolver.pth"
}

stage_ext() {
  check_python
  export CUDA_HOME
  # Conda bin 必须在 PATH 前面，torch 的 BuildExtension 才能找到 pip 装的 ninja 并行编译。
  export PATH="$CONDA_ENV/bin:$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}"
  export MAX_JOBS="${MAX_JOBS:-10}"
  [ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc not found under $CUDA_HOME"
  command -v ninja >/dev/null || die "ninja not on PATH (pip install ninja)"
  # 当前网络下 git://https 大传输会 GnuTLS 中断，改用 codeload tarball（可走 PROXY）。
  local src="$WORK/src"
  local p3d_sha="e4ad6e5687a360ccdc6ebd91348e8dbbac97bf31"
  local ndr_sha="253ac4fcea7de5f396371124af597e6cc957bfae"
  mkdir -p "$src"
  local curl_cmd=(curl -fsSL --retry 5 --retry-delay 5)
  if [ -n "$PROXY" ]; then curl_cmd+=(--proxy "$PROXY"); fi
  fetch_tarball() {
    local org="$1" name="$2" sha="$3"
    [ -d "$src/$name-$sha" ] && return 0
    log "downloading $org/$name@$sha"
    "${curl_cmd[@]}" -o "$src/$name.tar.gz" \
      "https://codeload.github.com/$org/$name/tar.gz/$sha" \
      || die "tarball download failed: $name"
    tar -xzf "$src/$name.tar.gz" -C "$src"
    rm -f "$src/$name.tar.gz"
  }
  local p3d_src="$PROJECT_ROOT/third_party/pytorch3d"
  local ndr_src="$PROJECT_ROOT/third_party/nvdiffrast"
  if [ ! -f "$p3d_src/setup.py" ]; then
    fetch_tarball facebookresearch pytorch3d "$p3d_sha"
    p3d_src="$src/pytorch3d-$p3d_sha"
  fi
  if [ ! -f "$ndr_src/setup.py" ]; then
    fetch_tarball NVlabs nvdiffrast "$ndr_sha"
    ndr_src="$src/nvdiffrast-$ndr_sha"
  fi
  log "building pytorch3d (CUDA_HOME=$CUDA_HOME arch=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS)"
  "$PY" -m pip install --no-build-isolation "$p3d_src"
  log "building nvdiffrast"
  "$PY" -m pip install --no-build-isolation "$ndr_src"
  log "extensions installed"
}

stage_mycpp() {
  check_python
  log "building mycpp"
  if "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import mycpp' "$REPO/mycpp/build" >/dev/null 2>&1; then
    log "reusing the compatible Python $PYTHON_VERSION mycpp extension"
    return
  fi
  local build="$WORK/mycpp-build"
  local site
  site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$build"
  cmake -S "$REPO/mycpp" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PY" \
    -DPYTHON_EXECUTABLE="$PY" \
    -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)"
  cmake --build "$build" -j"${MAX_JOBS:-4}"
  local extension
  extension="$(find "$build" -maxdepth 1 -name 'mycpp*.so' -print -quit)"
  [ -n "$extension" ] || die "mycpp extension missing"
  install -m 0644 "$extension" "$site/"
  log "mycpp ready: $site/$(basename "$extension")"
}

case "$STAGE" in
  conda|ext|mycpp|all) select_platform ;;
esac
mkdir -p "$WORK/logs"
case "$STAGE" in
  repo)  stage_repo ;;
  conda) stage_conda ;;
  ext)   stage_ext ;;
  mycpp) stage_mycpp ;;
  all)   stage_repo; stage_conda; stage_ext; stage_mycpp ;;
  *)     die "unknown stage: $STAGE (repo|conda|ext|mycpp|all)" ;;
esac
log "stage '$STAGE' done"
