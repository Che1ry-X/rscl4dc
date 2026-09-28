"""Per-image binary and topology metrics for trajectory-line predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


METRIC_NAMES = (
    "dice",
    "cldice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def load_binary_ground_truth(
    path: Path, expected_shape: Tuple[int, int]
) -> np.ndarray:
    with Image.open(path) as image:
        mask_image = image.convert("L")
        original_shape = (mask_image.height, mask_image.width)
        if original_shape != expected_shape:
            warnings.warn(
                f"Ground-truth shape {original_shape} does not match prediction "
                f"shape {expected_shape} for {path.name}; resizing with nearest-neighbor.",
                UserWarning,
            )
            mask_image = mask_image.resize(
                (expected_shape[1], expected_shape[0]), resample=Image.NEAREST
            )
        array = np.asarray(mask_image)
    return array > 0


def _ratio(numerator: float, denominator: float, empty_value: float) -> float:
    if denominator <= 0.0:
        return float(empty_value)
    return float(numerator / denominator)


def hard_cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)
    predicted_skeleton = skeletonize(prediction_bool)
    target_skeleton = skeletonize(target_bool)

    predicted_skeleton_count = float(predicted_skeleton.sum())
    target_skeleton_count = float(target_skeleton.sum())
    both_empty = predicted_skeleton_count == 0.0 and target_skeleton_count == 0.0

    topology_precision = _ratio(
        float(np.logical_and(predicted_skeleton, target_bool).sum()),
        predicted_skeleton_count,
        1.0 if both_empty else 0.0,
    )
    topology_sensitivity = _ratio(
        float(np.logical_and(target_skeleton, prediction_bool).sum()),
        target_skeleton_count,
        1.0 if both_empty else 0.0,
    )
    denominator = topology_precision + topology_sensitivity
    if denominator <= 0.0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


def binary_segmentation_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> Dict[str, float]:
    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)
    if prediction_bool.shape != target_bool.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction_bool.shape} vs {target_bool.shape}"
        )

    tp = float(np.logical_and(prediction_bool, target_bool).sum())
    fp = float(np.logical_and(prediction_bool, ~target_bool).sum())
    fn = float(np.logical_and(~prediction_bool, target_bool).sum())
    tn = float(np.logical_and(~prediction_bool, ~target_bool).sum())

    both_foregrounds_empty = (tp + fp + fn) == 0.0
    dice = _ratio(2.0 * tp, 2.0 * tp + fp + fn, 1.0)
    iou = _ratio(tp, tp + fp + fn, 1.0)
    precision = _ratio(tp, tp + fp, 1.0 if both_foregrounds_empty else 0.0)
    recall = _ratio(tp, tp + fn, 1.0)
    specificity = _ratio(tn, tn + fp, 1.0)
    accuracy = _ratio(tp + tn, tp + tn + fp + fn, 1.0)

    return {
        "dice": dice,
        "cldice": hard_cldice(prediction_bool, target_bool),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "true_positive_px": tp,
        "false_positive_px": fp,
        "false_negative_px": fn,
        "true_negative_px": tn,
        "predicted_foreground_px": tp + fp,
        "ground_truth_foreground_px": tp + fn,
    }
