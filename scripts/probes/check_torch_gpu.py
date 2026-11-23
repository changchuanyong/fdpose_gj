"""Exercise actual CUDA matrix multiplication and torchvision NMS on this host."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from probe_report import checked_at_utc, run_probe


def check(expect_torch: str, expect_torchvision: str) -> dict:
    import torch
    import torchvision

    if torch.__version__ != expect_torch:
        raise RuntimeError(f"Unexpected PyTorch candidate: {torch.__version__} (expected {expect_torch})")
    if torchvision.__version__ != expect_torchvision:
        raise RuntimeError(
            f"Unexpected torchvision candidate: {torchvision.__version__} (expected {expect_torchvision})"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    matrix = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    product = matrix @ matrix.T
    reference = matrix.cpu() @ matrix.cpu().T
    torch.cuda.synchronize()
    if not torch.equal(product.cpu(), reference):
        raise RuntimeError("CUDA matrix product differs from CPU reference")
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32, device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    selected = torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    if selected.cpu().tolist() != [0, 2]:
        raise RuntimeError("Unexpected CUDA NMS result")
    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "matrix_product_passed": True,
        "torchvision_nms_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-torch", default="2.9.1", help="Expected torch.__version__ string.")
    parser.add_argument(
        "--expect-torchvision", default="0.24.1", help="Expected torchvision.__version__ string."
    )
    args = parser.parse_args()
    return run_probe(
        args.output,
        {"checked_at_utc": checked_at_utc(), "python": sys.executable, "passed": False},
        lambda: check(args.expect_torch, args.expect_torchvision),
        error_keys=("error_type", "error", "traceback"))


if __name__ == "__main__":
    raise SystemExit(main())
