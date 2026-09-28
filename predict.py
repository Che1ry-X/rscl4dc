"""Sliding-window topology-aware ResUNet prediction, metrics and measurement."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

try:
    from .dataset import SUPPORTED_SUFFIXES, load_normalized_image
    from .metrics import (
        METRIC_NAMES,
        binary_segmentation_metrics,
        load_binary_ground_truth,
    )
    from .model import ResUNet
    from .postprocess import (
        MEASUREMENT_COLUMNS,
        make_color_diagrams,
        reverse_fill_closed_regions,
        save_postprocess_outputs,
        skeletonize_prediction,
    )
except ImportError:  # Allows: cd resunetcl && python predict.py
    from dataset import SUPPORTED_SUFFIXES, load_normalized_image
    from metrics import (
        METRIC_NAMES,
        binary_segmentation_metrics,
        load_binary_ground_truth,
    )
    from model import ResUNet
    from postprocess import (
        MEASUREMENT_COLUMNS,
        make_color_diagrams,
        reverse_fill_closed_regions,
        save_postprocess_outputs,
        skeletonize_prediction,
    )


PROJECT_DIR = Path(__file__).resolve().parent
COUNT_METRIC_NAMES = (
    "true_positive_px",
    "false_positive_px",
    "false_negative_px",
    "true_negative_px",
    "predicted_foreground_px",
    "ground_truth_foreground_px",
)
ALL_METRIC_NAMES = METRIC_NAMES + COUNT_METRIC_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict trajectory lines, measure closed structures and optionally "
            "evaluate against same-stem ground-truth masks"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/CR/train"))
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help=(
            "Optional same-stem mask directory. Dice/clDice/recall cannot be "
            "calculated without ground truth."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "predict_results_resunet_cl",
        help="Prediction outputs (default: resunetcl/predict_results_resunet_cl)",
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--pixels-per-mm", type=float, default=8.4)
    parser.add_argument("--closing-kernel", type=int, default=3)
    parser.add_argument("--min-line-component-px", type=int, default=20)
    parser.add_argument("--min-closed-area-px", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    return torch.device(requested)


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, object]:
    # Only load checkpoints you trust; PyTorch checkpoints are pickle-based.
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, length - tile_size + 1, stride))
    if values[-1] != length - tile_size:
        values.append(length - tile_size)
    return values


@torch.no_grad()
def sliding_window_predict(
    model: ResUNet,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D normalized image, got {image.shape}")
    if tile_size < 32 or tile_size % 16 != 0:
        raise ValueError("tile_size must be >= 32 and divisible by 16")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    original_height, original_width = image.shape
    pad_bottom = max(0, tile_size - original_height)
    pad_right = max(0, tile_size - original_width)
    if pad_bottom or pad_right:
        mode = "reflect" if original_height > 1 and original_width > 1 else "edge"
        image = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode=mode)
    height, width = image.shape
    stride = tile_size - overlap
    y_starts = _starts(height, tile_size, stride)
    x_starts = _starts(width, tile_size, stride)

    window_1d = np.hanning(tile_size).astype(np.float32)
    blend_window = np.maximum(np.outer(window_1d, window_1d), 0.05)
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    locations = [(top, left) for top in y_starts for left in x_starts]
    model.eval()
    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start : start + batch_size]
        patches = np.stack(
            [
                image[top : top + tile_size, left : left + tile_size]
                for top, left in batch_locations
            ]
        )
        tensor = torch.from_numpy(patches[:, None]).float().to(
            device, non_blocking=True
        )
        probabilities = torch.sigmoid(model(tensor)).squeeze(1).cpu().numpy()
        for probability, (top, left) in zip(probabilities, batch_locations):
            probability_sum[
                top : top + tile_size, left : left + tile_size
            ] += probability * blend_window
            weight_sum[
                top : top + tile_size, left : left + tile_size
            ] += blend_window
    result = probability_sum / np.maximum(weight_sum, 1e-7)
    return result[:original_height, :original_width]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def build_same_stem_map(folder: Optional[Path]) -> Dict[str, Path]:
    if folder is None:
        return {}
    if not folder.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {folder}")
    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        key = path.stem.lower()
        if key in result:
            raise ValueError(
                f"Duplicate ground-truth stem '{path.stem}' in {folder}: "
                f"{result[key].name} and {path.name}"
            )
        result[key] = path
    return result


def unavailable_metric_row(
    source_image: str, threshold: float
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "source_image": source_image,
        "ground_truth_available": False,
        "ground_truth_file": "",
        "threshold": threshold,
    }
    for prefix in ("raw", "cleaned"):
        for name in ALL_METRIC_NAMES:
            row[f"{prefix}_{name}"] = np.nan
    return row


def evaluate_prediction(
    source_image: str,
    threshold: float,
    ground_truth_path: Optional[Path],
    binary: np.ndarray,
    cleaned: np.ndarray,
) -> Dict[str, object]:
    if ground_truth_path is None:
        return unavailable_metric_row(source_image, threshold)

    target = load_binary_ground_truth(ground_truth_path, binary.shape)
    raw_metrics = binary_segmentation_metrics(binary, target)
    cleaned_metrics = binary_segmentation_metrics(cleaned, target)
    row: Dict[str, object] = {
        "source_image": source_image,
        "ground_truth_available": True,
        "ground_truth_file": ground_truth_path.name,
        "threshold": threshold,
    }
    row.update({f"raw_{key}": value for key, value in raw_metrics.items()})
    row.update({f"cleaned_{key}": value for key, value in cleaned_metrics.items()})
    return row


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model_config = dict(checkpoint.get("model_config", {}))
    model_config.setdefault("in_channels", 1)
    model_config.setdefault("out_channels", 1)
    model_config.setdefault("base_channels", 32)
    model = ResUNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    training_config = dict(checkpoint.get("training_config", {}))
    normalization = dict(checkpoint.get("normalization", {}))
    norm_low = float(normalization.get("lower_percentile", 1.0))
    norm_high = float(normalization.get("upper_percentile", 99.5))
    default_tile_size = int(training_config.get("patch_size", 512))
    tile_size = args.tile_size or default_tile_size
    threshold = args.threshold
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")

    input_dir = args.input_dir or (args.data_root / "predict")
    output_dir = args.output_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction input directory does not exist: {input_dir}"
        )
    images = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise ValueError(f"No supported images found in {input_dir}")
    ground_truth_map = build_same_stem_map(args.ground_truth_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ground_truth_dir is None:
        print(
            "NOTE: --ground-truth-dir was not supplied. Predictions and geometry "
            "will be saved, but Dice/clDice/recall require masks and will be NaN."
        )
    print(
        f"Device={device}; images={len(images)}; tile={tile_size}; "
        f"overlap={args.overlap}; threshold={threshold}; "
        f"scale={args.pixels_per_mm} px/mm"
    )

    all_measurements: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    for image_path in tqdm(images, desc="predict"):
        normalized = load_normalized_image(image_path, norm_low, norm_high)
        probability = sliding_window_predict(
            model,
            normalized,
            tile_size=tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device=device,
        )
        binary = probability >= threshold
        cleaned, skeleton = skeletonize_prediction(
            binary,
            closing_kernel=args.closing_kernel,
            minimum_line_component=args.min_line_component_px,
        )
        structure_ids, measurements = reverse_fill_closed_regions(
            skeleton,
            pixels_per_mm=args.pixels_per_mm,
            minimum_closed_area=args.min_closed_area_px,
            source_image=image_path.name,
        )
        diagram, overlay = make_color_diagrams(
            structure_ids,
            skeleton,
            measurements,
            original_rgb=load_rgb(image_path),
        )
        image_output_dir = output_dir / image_path.stem
        save_postprocess_outputs(
            image_output_dir,
            image_path.name,
            probability,
            binary,
            cleaned,
            skeleton,
            structure_ids,
            measurements,
            diagram,
            overlay,
        )

        ground_truth_path = ground_truth_map.get(image_path.stem.lower())
        metric_row = evaluate_prediction(
            image_path.name,
            threshold,
            ground_truth_path,
            binary,
            cleaned,
        )
        pd.DataFrame([metric_row]).to_csv(
            image_output_dir / "metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        all_measurements.append(measurements)
        metric_rows.append(metric_row)
        summary_row: Dict[str, object] = {
            "source_image": image_path.name,
            "closed_structure_count": len(measurements),
        }
        summary_row.update(
            {
                key: value
                for key, value in metric_row.items()
                if key not in ("source_image",)
            }
        )
        summary_rows.append(summary_row)

    combined = (
        pd.concat(all_measurements, ignore_index=True)
        if all_measurements
        else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    )
    summary = pd.DataFrame(summary_rows)
    metrics = pd.DataFrame(metric_rows)

    combined.to_csv(
        output_dir / "all_measurements.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output_dir / "all_summary.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        output_dir / "all_metrics.csv", index=False, encoding="utf-8-sig"
    )
    with pd.ExcelWriter(
        output_dir / "all_measurements.xlsx", engine="openpyxl"
    ) as writer:
        combined.to_excel(writer, sheet_name="measurements", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)
    with pd.ExcelWriter(output_dir / "all_metrics.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)

    available_count = int(metrics["ground_truth_available"].sum())
    missing_count = len(metrics) - available_count
    print(f"Done. Measurements: {output_dir / 'all_measurements.xlsx'}")
    print(
        f"Metrics: {output_dir / 'all_metrics.xlsx'} "
        f"(ground truth available={available_count}, missing={missing_count})"
    )


if __name__ == "__main__":
    main()


