"""Fast smoke test for clDice, balanced 512 patches, inference and metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

try:
    from .dataset import (
        BalancedPatchDataset,
        discover_pairs,
        load_normalized_image,
    )
    from .losses import BCEDiceCLDiceLoss
    from .metrics import binary_segmentation_metrics
    from .model import ResUNet
    from .postprocess import (
        make_color_diagrams,
        reverse_fill_closed_regions,
        save_postprocess_outputs,
        skeletonize_prediction,
    )
    from .predict import sliding_window_predict
except ImportError:
    from dataset import (
        BalancedPatchDataset,
        discover_pairs,
        load_normalized_image,
    )
    from losses import BCEDiceCLDiceLoss
    from metrics import binary_segmentation_metrics
    from model import ResUNet
    from postprocess import (
        make_color_diagrams,
        reverse_fill_closed_regions,
        save_postprocess_outputs,
        skeletonize_prediction,
    )
    from predict import sliding_window_predict


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short synthetic topology-aware ResUNet smoke test"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "quick_test_output"
    )
    return parser.parse_args()


def make_synthetic_data(root: Path) -> None:
    original_dir = root / "original"
    mask_dir = root / "mask"
    predict_dir = root / "predict"
    original_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    predict_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    for index in range(1, 5):
        height = width = 320
        base = rng.normal(45, 6, size=(height, width)).clip(0, 255).astype(np.uint8)
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        offset = index % 3
        draw.rectangle((36 + offset, 40, 144 + offset, 152), outline=255, width=5)
        draw.polygon(
            [(184, 56), (284, 80), (264, 210), (172, 184)],
            outline=255,
            width=5,
        )
        mask = np.asarray(mask_image)
        synthetic = np.clip(base + (mask > 0) * 150, 0, 255).astype(np.uint8)
        Image.fromarray(synthetic).save(original_dir / f"{index}.tif")
        mask_image.save(mask_dir / f"{index}.tif")
        if index == 1:
            Image.fromarray(synthetic).save(
                predict_dir / "synthetic_predict.tif"
            )


def main() -> None:
    args = parse_args()
    data_root = args.output_dir / "synthetic_data"
    result_dir = args.output_dir / "postprocess_result"
    make_synthetic_data(data_root)

    pairs = discover_pairs(data_root)
    dataset = BalancedPatchDataset(
        pairs,
        patch_size=128,
        samples_per_epoch=12,
        foreground_probability=1.0,
        augment=True,
        seed=42,
    )
    summary = dataset.sampling_summary()
    totals = list(summary["per_image"].values())
    assert summary["patch_size"] == 128
    assert summary["total"] == 12
    assert min(totals) == max(totals) == 3, totals

    samples = [dataset[index] for index in (0, 1)]
    images = torch.stack([sample[0] for sample in samples])
    masks = torch.stack([sample[1] for sample in samples])

    model = ResUNet(in_channels=1, out_channels=1, base_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceCLDiceLoss(
        bce_weight=0.3,
        dice_weight=0.3,
        cldice_weight=0.4,
        cldice_iterations=3,
    )
    logits = model(images)
    loss, bce_loss, dice_loss, cldice_loss = criterion.components(logits, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert logits.shape == masks.shape
    assert torch.isfinite(loss), "Training smoke-test loss is not finite"
    expected = 0.3 * bce_loss + 0.3 * dice_loss + 0.4 * cldice_loss
    assert torch.allclose(loss, expected), "Loss weights were not applied exactly"

    prediction_path = data_root / "predict" / "synthetic_predict.tif"
    normalized = load_normalized_image(prediction_path)
    tiled_probability = sliding_window_predict(
        model,
        normalized,
        tile_size=128,
        overlap=32,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert tiled_probability.shape == normalized.shape
    assert np.isfinite(tiled_probability).all()
    assert (
        0.0
        <= float(tiled_probability.min())
        <= float(tiled_probability.max())
        <= 1.0
    )

    with Image.open(data_root / "mask" / "1.tif") as mask_image:
        known_binary = np.asarray(mask_image) > 0
    perfect_metrics = binary_segmentation_metrics(known_binary, known_binary)
    assert abs(perfect_metrics["dice"] - 1.0) < 1e-7
    assert abs(perfect_metrics["cldice"] - 1.0) < 1e-7
    assert abs(perfect_metrics["recall"] - 1.0) < 1e-7

    cleaned, skeleton = skeletonize_prediction(
        known_binary, closing_kernel=3, minimum_line_component=1
    )
    structure_ids, measurements = reverse_fill_closed_regions(
        skeleton,
        pixels_per_mm=8.4,
        minimum_closed_area=100,
        source_image=prediction_path.name,
    )
    assert len(measurements) == 2, (
        f"Expected 2 closed structures, found {len(measurements)}"
    )

    with Image.open(prediction_path) as image:
        original_rgb = np.asarray(image.convert("RGB"))
    diagram, overlay = make_color_diagrams(
        structure_ids, skeleton, measurements, original_rgb=original_rgb
    )
    save_postprocess_outputs(
        result_dir,
        prediction_path.name,
        tiled_probability,
        known_binary,
        cleaned,
        skeleton,
        structure_ids,
        measurements,
        diagram,
        overlay,
    )
    torch.save(
        {
            "format_version": 2,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
            },
            "normalization": {
                "lower_percentile": 1.0,
                "upper_percentile": 99.5,
            },
            "training_config": {"patch_size": 128},
            "loss_config": {
                "bce_weight": 0.3,
                "dice_weight": 0.3,
                "cldice_weight": 0.4,
                "cldice_iterations": 3,
            },
            "threshold": 0.2,
        },
        args.output_dir / "quick_test_checkpoint.pt",
    )

    print(
        "PASS: balanced single-size sampling "
        f"(patch=128, total={summary['total']}, "
        f"per-image={min(totals)}..{max(totals)})"
    )
    print(
        "PASS: 0.3 BCE + 0.3 Dice + 0.4 clDice "
        f"forward/backward loss={loss.detach().item():.4f}"
    )
    print("PASS: sliding-window prediction shape and range")
    print("PASS: per-image Dice/clDice/recall metrics")
    print(
        f"PASS: skeletonization and reverse filling found "
        f"{len(measurements)} structures"
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()



