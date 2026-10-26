#!/usr/bin/env python3
"""
Prune + fine-tune a trained Faster R-CNN (ResNet-18 FPN) checkpoint.

Uses global magnitude pruning (L1 unstructured) on Conv2d weights so a single
`amount` applies across all listed layers. This maximizes parameter sparsity
(zeros in weight tensors). Note: standard dense CUDA convolutions do not
accelerate from unstructured zeros; for real latency wins you typically need
structured/channel pruning with a narrowed model, quantization, or sparse
libraries. This script still gives smaller .pth with optional compression and
can recover accuracy via fine-tuning.

Example (single pass + 20 epochs, low LR):
  python prune_finetune_resnet18.py \\
    --checkpoint ./models_resnet18_v2/final_model.pth \\
    --output_dir ./models_resnet18_v2_pruned \\
    --prune_amount 0.25 \\
    --finetune_epochs 20 \\
    --lr 1e-4

Example (iterative: 25% then 25%, 10+10 epochs):
  python prune_finetune_resnet18.py \\
    --checkpoint ./models_resnet18_v2/final_model.pth \\
    --output_dir ./models_resnet18_v2_pruned_iter \\
    --iterative \\
    --prune_rounds 2 \\
    --prune_amount 0.25 \\
    --finetune_epochs_per_round 10 \\
    --lr 1e-4
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, random_split

from train import GraspDataset, collate_fn, train_one_epoch
from train_resnet_18 import evaluate_with_diagnostics, get_model, print_model_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _collect_conv_weights(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """All Conv2d weight tensors suitable for global pruning."""
    out: List[Tuple[nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight is not None:
            out.append((m, "weight"))
    return out


def _total_and_nonzero_params(model: nn.Module) -> Tuple[int, int]:
    total = 0
    nonzero = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def _conv2d_weight_elems_and_nonzero(model: nn.Module) -> Tuple[int, int]:
    elems = 0
    nz = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight is not None:
            w = m.weight
            elems += w.numel()
            nz += int(torch.count_nonzero(w).item())
    return elems, nz


def print_parameter_report(model: nn.Module, label: str = "") -> dict:
    """
    Log full parameter counts. Pruning zeros Conv2d weights only; tensor shapes
    (and total parameter slots) stay the same — effective size is reflected in nonzero counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nz_total = sum(int(torch.count_nonzero(p).item()) for p in model.parameters())
    conv_elems, conv_nz = _conv2d_weight_elems_and_nonzero(model)
    zeros_total = total - nz_total
    zeros_conv = conv_elems - conv_nz
    sparsity_total = 100.0 * zeros_total / total if total else 0.0
    sparsity_conv = 100.0 * zeros_conv / conv_elems if conv_elems else 0.0
    approx_mb = total * 4 / (1024 * 1024)

    prefix = f"{label}: " if label else ""
    logger.info(f"{prefix}========== PARAMETER REPORT ==========")
    logger.info(f"{prefix}Total parameter elements (all layers): {total:,}")
    logger.info(f"{prefix}Trainable parameter elements:          {trainable:,}")
    logger.info(f"{prefix}Nonzero elements (entire model):      {nz_total:,}")
    logger.info(f"{prefix}Zero elements (entire model):         {zeros_total:,}  ({sparsity_total:.2f}% sparse)")
    logger.info(f"{prefix}Approx. FP32 size if dense stored:    {approx_mb:.2f} MB")
    logger.info(f"{prefix}--- Conv2d weights (what L1 pruning targets) ---")
    logger.info(f"{prefix}Conv2d weight elements:                 {conv_elems:,}")
    logger.info(f"{prefix}Conv2d nonzero weights:               {conv_nz:,}")
    logger.info(f"{prefix}Conv2d zero weights:                  {zeros_conv:,}  ({sparsity_conv:.2f}% sparse)")
    logger.info(f"{prefix}========================================")
    return {
        "total_parameter_elements": total,
        "trainable_parameter_elements": trainable,
        "nonzero_elements": nz_total,
        "zero_elements": zeros_total,
        "sparsity_percent_total": round(sparsity_total, 4),
        "conv2d_weight_elements": conv_elems,
        "conv2d_nonzero": conv_nz,
        "conv2d_zero": zeros_conv,
        "sparsity_percent_conv2d_only": round(sparsity_conv, 4),
        "approx_fp32_size_mb": round(approx_mb, 4),
    }


def apply_global_l1_prune(model: nn.Module, amount: float) -> None:
    """Apply one shot of global magnitude pruning across all Conv2d weights."""
    params = _collect_conv_weights(model)
    if not params:
        raise RuntimeError("No Conv2d layers found to prune.")
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for module, name in params:
        prune.remove(module, name)


def benchmark_inference(model: nn.Module, device: torch.device, batch_size: int = 1, runs: int = 20) -> float:
    """Rough mean forward time per image (eval, no targets)."""
    model.eval()
    dummy = [torch.rand(3, 480, 640, device=device) for _ in range(batch_size)]
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(sum(times) / len(times) / batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune + fine-tune ResNet-18 Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth (e.g. final_model.pth)")
    parser.add_argument("--output_dir", type=str, default="./models_pruned_resnet18", help="Where to save pruned+fine-tuned weights")

    parser.add_argument("--prune_amount", type=float, default=0.25, help="Fraction of weights to prune per round (global L1 on Conv2d)")
    parser.add_argument("--iterative", action="store_true", help="Multiple prune rounds with fine-tune between")
    parser.add_argument("--prune_rounds", type=int, default=2, help="Number of prune rounds if --iterative")
    parser.add_argument("--finetune_epochs", type=int, default=20, help="Total fine-tune epochs (single-round mode)")
    parser.add_argument("--finetune_epochs_per_round", type=int, default=10, help="Epochs after each prune round (iterative mode)")

    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate (smaller than pre-prune training)")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine", "plateau", "exponential", "none"])
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--eval_conf_threshold", type=float, default=0.5)
    parser.add_argument("--eval_iou_threshold", type=float, default=0.25)
    parser.add_argument("--eval_top_k", type=int, default=1)

    parser.add_argument("--augmented_cornell_path", type=str, default="./augmented_dataset")
    parser.add_argument("--cornell_path", type=str, default="./cornell_dataset")
    parser.add_argument("--jacquard_path", type=str, default="./jacquard_dataset")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark", action="store_true", help="Log inference time before/after (same device)")

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flops_log_path = output_dir / "flops.jsonl"

    # Data
    data_sources = []
    if os.path.exists(args.augmented_cornell_path):
        data_sources.append({"type": "augmented_cornell", "path": args.augmented_cornell_path})
    if os.path.exists(args.cornell_path):
        data_sources.append({"type": "original_cornell", "path": args.cornell_path})
    if os.path.exists(args.jacquard_path):
        data_sources.append({"type": "original_jacquard", "path": args.jacquard_path})
    if not data_sources:
        logger.error("No dataset paths found.")
        return

    dataset = GraspDataset(data_sources, target_size=(640, 480))
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Model + checkpoint
    model = get_model(num_classes=2, freeze_backbone=args.freeze_backbone)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)

    logger.info("=== Before pruning (loaded checkpoint) ===")
    print_model_summary(model)
    stats_before = print_parameter_report(model, label="BEFORE prune")
    if args.benchmark:
        t = benchmark_inference(model, device)
        logger.info(f"Benchmark mean time per image (approx): {t*1000:.2f} ms")

    # Prune schedule
    if args.iterative:
        rounds = max(1, args.prune_rounds)
        epochs_each = max(1, args.finetune_epochs_per_round)
    else:
        rounds = 1
        epochs_each = max(1, args.finetune_epochs)

    global_epoch = 0

    for r in range(rounds):
        logger.info(f"=== Prune round {r + 1}/{rounds}: global L1 amount={args.prune_amount} ===")
        apply_global_l1_prune(model, args.prune_amount)
        print_parameter_report(model, label=f"AFTER prune round {r + 1}")

        params = [p for p in model.parameters() if p.requires_grad]
        if args.optimizer == "sgd":
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

        # Scheduler over current segment
        if args.scheduler == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_each, eta_min=args.lr * 0.01)
        elif args.scheduler == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == "plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
        elif args.scheduler == "exponential":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            lr_scheduler = None

        logger.info(f"Fine-tuning for {epochs_each} epochs (lr={args.lr})...")
        for _ in range(epochs_each):
            global_epoch += 1
            train_metrics = train_one_epoch(model, optimizer, train_loader, device, global_epoch, grad_clip=args.grad_clip, flops_log_path=flops_log_path)
            val_metrics = evaluate_with_diagnostics(
                model,
                val_loader,
                device,
                confidence_threshold=args.eval_conf_threshold,
                iou_threshold=args.eval_iou_threshold,
                top_k=max(1, args.eval_top_k),
            )
            if lr_scheduler is not None:
                if args.scheduler == "plateau":
                    lr_scheduler.step(val_metrics["avg_loss"])
                else:
                    lr_scheduler.step()
            logger.info(
                f"Round {r+1} Ep {global_epoch}: train_loss={train_metrics['total_loss']:.4f} "
                f"val_loss={val_metrics['avg_loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

        torch.save(
            {
                "epoch": global_epoch,
                "prune_round": r + 1,
                "prune_amount_per_round": args.prune_amount,
                "iterative": args.iterative,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_dir / f"pruned_finetuned_round_{r + 1}.pth",
        )

    logger.info("=== After pruning + fine-tune (final) ===")
    stats_after = print_parameter_report(model, label="AFTER prune+finetune")
    if args.benchmark:
        t = benchmark_inference(model, device)
        logger.info(f"Benchmark mean time per image (approx): {t*1000:.2f} ms")

    torch.save(
        {
            "epoch": global_epoch,
            "checkpoint_source": str(args.checkpoint),
            "prune_amount_per_round": args.prune_amount,
            "prune_rounds": rounds,
            "iterative": args.iterative,
            "finetune_epochs_total": global_epoch,
            "parameter_stats_before_prune": stats_before,
            "parameter_stats_after_prune_finetune": stats_after,
            "model_state_dict": model.state_dict(),
        },
        output_dir / "final_pruned_finetuned.pth",
    )
    logger.info(f"Saved: {output_dir / 'final_pruned_finetuned.pth'}")


if __name__ == "__main__":
    main()
