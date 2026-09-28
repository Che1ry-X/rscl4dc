"""Train a topology-aware ResUNet with balanced 512 x 512 patches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import (
        BalancedPatchDataset,
        RandomPatchDataset,
        discover_pairs,
        find_shape_mismatches,
        split_pairs,
    )
    from .losses import BCEDiceCLDiceLoss, binary_counts, counts_to_metrics
    from .model import ResUNet
except ImportError:  # Allows: cd resunetcl && python train.py
    from dataset import (
        BalancedPatchDataset,
        RandomPatchDataset,
        discover_pairs,
        find_shape_mismatches,
        split_pairs,
    )
    from losses import BCEDiceCLDiceLoss, binary_counts, counts_to_metrics
    from model import ResUNet


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a topology-aware ResUNet with BCE + Dice + clDice"
    )
    parser.add_argument("--data-root", type=Path, default=Path("/CR/train"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "resunet_cl",
        help="Training outputs (default: resunetcl/runs/resunet_cl)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of 512 x 512 patches per optimizer step (default: 2)",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--samples-per-epoch", type=int, default=580)
    parser.add_argument("--validation-samples", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--foreground-probability", type=float, default=0.7)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--cldice-weight", type=float, default=0.4)
    parser.add_argument("--cldice-iterations", type=int, default=10)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--norm-low", type=float, default=1.0)
    parser.add_argument("--norm-high", type=float, default=99.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_epoch(
    model: ResUNet,
    loader: DataLoader,
    criterion: BCEDiceCLDiceLoss,
    device: torch.device,
    threshold: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_cldice_loss = 0.0
    total_items = 0
    tp = predicted = actual = 0.0
    description = "train" if training else "valid"

    for images, masks in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss, bce_loss, dice_loss, cldice_loss = criterion.components(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_bce_loss += float(bce_loss.detach().item()) * batch_size
        total_dice_loss += float(dice_loss.detach().item()) * batch_size
        total_cldice_loss += float(cldice_loss.detach().item()) * batch_size
        total_items += batch_size
        batch_tp, batch_predicted, batch_actual = binary_counts(
            logits.detach(), masks, threshold
        )
        tp += batch_tp
        predicted += batch_predicted
        actual += batch_actual

    metrics = counts_to_metrics(tp, predicted, actual)
    denominator = max(total_items, 1)
    metrics["loss"] = total_loss / denominator
    metrics["bce_loss"] = total_bce_loss / denominator
    metrics["dice_loss"] = total_dice_loss / denominator
    metrics["cldice_loss"] = total_cldice_loss / denominator
    metrics["soft_cldice"] = 1.0 - metrics["cldice_loss"]
    return metrics


def append_history(path: Path, row: Dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: ResUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_dice": validation_metrics["dice"],
        "val_soft_cldice": validation_metrics["soft_cldice"],
        "threshold": args.threshold,
        "model_config": {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": args.base_channels,
        },
        "normalization": {
            "lower_percentile": args.norm_low,
            "upper_percentile": args.norm_high,
        },
        "loss_config": {
            "bce_weight": args.bce_weight,
            "dice_weight": args.dice_weight,
            "cldice_weight": args.cldice_weight,
            "cldice_iterations": args.cldice_iterations,
        },
        "training_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if args.patch_size < 32 or args.patch_size % 16 != 0:
        raise ValueError("patch-size must be >= 32 and divisible by 16")

    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data_root)
    mismatches = find_shape_mismatches(pairs)
    if mismatches:
        print(
            "WARNING: image/mask shape mismatches were found; "
            "masks will be nearest-neighbor resized:"
        )
        for item in mismatches:
            print(
                f"  {item['name']}: image={item['image_hw']}, "
                f"mask={item['mask_hw']}"
            )

    train_pairs, validation_pairs = split_pairs(
        pairs, args.validation_fraction, args.seed
    )
    split_record = {
        "seed": args.seed,
        "train": [pair.name for pair in train_pairs],
        "validation": [pair.name for pair in validation_pairs],
        "shape_mismatches": mismatches,
    }
    (args.output_dir / "split.json").write_text(
        json.dumps(split_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = BalancedPatchDataset(
        train_pairs,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        foreground_probability=args.foreground_probability,
        augment=True,
        seed=args.seed,
        lower_percentile=args.norm_low,
        upper_percentile=args.norm_high,
    )
    validation_dataset = RandomPatchDataset(
        validation_pairs,
        patch_size=args.patch_size,
        samples_per_epoch=args.validation_samples,
        foreground_probability=0.5,
        augment=False,
        deterministic=True,
        seed=args.seed + 10_000,
        lower_percentile=args.norm_low,
        upper_percentile=args.norm_high,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    # The balanced schedule is already shuffled and rebuilt every epoch.
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )

    model = ResUNet(
        in_channels=1, out_channels=1, base_channels=args.base_channels
    ).to(device)
    criterion = BCEDiceCLDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        cldice_weight=args.cldice_weight,
        cldice_iterations=args.cldice_iterations,
        pos_weight=args.pos_weight,
    ).to(device)
    optimizer_class = getattr(torch.optim, "AdamW", torch.optim.Adam)
    optimizer = optimizer_class(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    first_summary = train_dataset.sampling_summary()
    (args.output_dir / "sampling_plan_epoch_001.json").write_text(
        json.dumps(first_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Device={device}; images={len(pairs)} "
        f"(train={len(train_pairs)}, valid={len(validation_pairs)}); "
        f"patches={first_summary['total']} x {args.patch_size}; "
        f"batch={args.batch_size}; precision=FP32; "
        f"optimizer={optimizer_class.__name__}"
    )
    per_image_totals = list(first_summary["per_image"].values())
    print(
        f"Balanced sampling per image: min={min(per_image_totals)}, "
        f"max={max(per_image_totals)}"
    )
    print(
        "Loss = "
        f"{args.bce_weight}*BCE + {args.dice_weight}*Dice "
        f"+ {args.cldice_weight}*clDice"
    )

    best_dice = -1.0
    stale_epochs = 0
    history_path = args.output_dir / "history.csv"

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, args.threshold, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device, args.threshold, optimizer=None
        )
        scheduler.step(validation_metrics["dice"])
        row: Dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 2),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        append_history(history_path, row)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train loss={train_metrics['loss']:.4f}, "
            f"dice={train_metrics['dice']:.4f}, "
            f"soft-clDice={train_metrics['soft_cldice']:.4f}; "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"dice={validation_metrics['dice']:.4f}, "
            f"soft-clDice={validation_metrics['soft_cldice']:.4f}, "
            f"iou={validation_metrics['iou']:.4f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            validation_metrics,
            args,
        )
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            stale_epochs = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                validation_metrics,
                args,
            )
            print(
                f"  Saved new best.pt (validation Dice={best_dice:.4f}, "
                f"soft-clDice={validation_metrics['soft_cldice']:.4f})"
            )
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(
                    "Early stopping: validation Dice did not improve for "
                    f"{stale_epochs} epochs"
                )
                break

    print(f"Training complete. Best validation Dice={best_dice:.4f}")
    print(f"Checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()




