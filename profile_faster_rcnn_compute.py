#!/usr/bin/env python3
"""
Compare parameter count and estimated FLOPs across Faster R-CNN checkpoints (no training).

Uses a single dummy forward at fixed resolution (default 640×480) with
``torch.utils.flop_counter.FlopCounterMode`` (counts aten mm/bmm/conv, etc.).

Notes:
- FLOPs are **one forward estimate**; detection models include RPN/ROI work that can depend on
  proposals and is **not** identical to “backbone only,” but the number is comparable across
  checkpoints of the **same resolution** and architecture family.
- **Pruned** models must be loaded so channel shapes match the checkpoint (same as inference).

Examples:
  python profile_faster_rcnn_compute.py \\
    --resnet50_ckpt ./trained_models97.06pct/final_model.pth \\
    --resnet18_ckpt ./models_resnet18_v2/final_model.pth \\
    --pruned_ckpt ./models_resnet18_v3_structured/structured_pruned.pth

  python profile_faster_rcnn_compute.py --resnet50_ckpt ./trained_models97.06pct/final_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.flop_counter import FlopCounterMode

from inference_resnet_50 import load_model as load_resnet50
from inference_resnet_18 import load_model as load_resnet18_plain
from inference_structured_pruning_resnet_18 import load_model as load_resnet18_structured


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _flops_one_forward(model: torch.nn.Module, device: torch.device, height: int, width: int) -> int:
    model.eval()
    x = torch.randn(3, height, width, device=device, dtype=torch.float32)
    with FlopCounterMode(display=False) as fc:
        with torch.inference_mode():
            model([x])
    return fc.get_total_flops()


def _maybe_profile(
    name: str,
    ckpt: Optional[Path],
    loader: Callable[[Path, torch.device], torch.nn.Module],
    device: torch.device,
    height: int,
    width: int,
) -> None:
    if ckpt is None:
        return
    if not ckpt.is_file():
        print(f"[{name}] SKIP (missing file): {ckpt}")
        return
    print(f"\n--- {name} ---")
    print(f"Checkpoint: {ckpt}")
    model = loader(ckpt, device)
    n_param = _count_params(model)
    flops = _flops_one_forward(model, device, height, width)
    gflops = flops / 1e9
    print(f"Parameters: {n_param:,}")
    print(f"FLOPs (1 forward @ {width}x{height}, estimate): {flops:,}  (~{gflops:.2f} GFLOPs)")


def main() -> None:
    p = argparse.ArgumentParser(description="Parameter + FLOP comparison for Faster R-CNN checkpoints")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (FLOP count is usually run on CPU)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--resnet50_ckpt", type=str, default=None, help="ResNet-50 Faster R-CNN .pth")
    p.add_argument("--resnet18_ckpt", type=str, default=None, help="Plain ResNet-18 .pth")
    p.add_argument("--pruned_ckpt", type=str, default=None, help="Structured-pruned ResNet-18 .pth")
    args = p.parse_args()

    device = torch.device(args.device)
    print("FLOP counting: torch.utils.flop_counter.FlopCounterMode (PyTorch 2.x)")
    print("Compare at equal --width / --height for apples-to-apples.")

    _maybe_profile("ResNet-50", Path(args.resnet50_ckpt) if args.resnet50_ckpt else None, load_resnet50, device, args.height, args.width)
    _maybe_profile("ResNet-18 (full)", Path(args.resnet18_ckpt) if args.resnet18_ckpt else None, load_resnet18_plain, device, args.height, args.width)
    _maybe_profile("ResNet-18 (structured-pruned)", Path(args.pruned_ckpt) if args.pruned_ckpt else None, load_resnet18_structured, device, args.height, args.width)

    if not any([args.resnet50_ckpt, args.resnet18_ckpt, args.pruned_ckpt]):
        print("Provide at least one of: --resnet50_ckpt, --resnet18_ckpt, --pruned_ckpt")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
