"""BCE, Dice and topology-preserving clDice losses and metrics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _flatten_sum(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(tensor.size(0), -1).sum(dim=1)


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = _flatten_sum(probabilities * targets)
    denominator = _flatten_sum(probabilities) + _flatten_sum(targets)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_erode(image: torch.Tensor) -> torch.Tensor:
    """Differentiable 2-D erosion used by soft skeletonization."""
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(vertical, horizontal)


def soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=(3, 3), stride=1, padding=(1, 1))


def soft_open(image: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Approximate a skeleton while retaining gradients for clDice."""
    opened = soft_open(image)
    skeleton = F.relu(image - opened)
    eroded = image
    for _ in range(iterations):
        eroded = soft_erode(eroded)
        opened = soft_open(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_score(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft clDice score based on skeleton precision and topology sensitivity."""
    predicted_skeleton = soft_skeletonize(probabilities, iterations)
    target_skeleton = soft_skeletonize(targets, iterations)
    topology_precision = (
        _flatten_sum(predicted_skeleton * targets) + eps
    ) / (_flatten_sum(predicted_skeleton) + eps)
    topology_sensitivity = (
        _flatten_sum(target_skeleton * probabilities) + eps
    ) / (_flatten_sum(target_skeleton) + eps)
    score = (
        2.0 * topology_precision * topology_sensitivity + eps
    ) / (topology_precision + topology_sensitivity + eps)
    return score.mean()


def soft_cldice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    iterations: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    return 1.0 - soft_cldice_score(torch.sigmoid(logits), targets, iterations, eps)


class BCEDiceCLDiceLoss(nn.Module):
    """Topology-aware objective: 0.3 BCE + 0.3 Dice + 0.4 clDice by default.

    The coefficients sum to 1.0 and are applied directly.
    """

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        cldice_weight: float = 0.4,
        cldice_iterations: int = 10,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("bce_weight", bce_weight),
            ("dice_weight", dice_weight),
            ("cldice_weight", cldice_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if bce_weight + dice_weight + cldice_weight <= 0.0:
            raise ValueError("At least one loss weight must be positive")
        if cldice_iterations < 1:
            raise ValueError("cldice_iterations must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.cldice_weight = float(cldice_weight)
        self.cldice_iterations = int(cldice_iterations)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.tensor(float(pos_weight)),
        )

    def components(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        dice = soft_dice_loss(logits, targets)
        cldice = soft_cldice_loss(logits, targets, self.cldice_iterations)
        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.cldice_weight * cldice
        )
        return total, bce, dice, cldice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total, _, _, _ = self.components(logits, targets)
        return total


@torch.no_grad()
def binary_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.2,
) -> Tuple[float, float, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    true_positive = (predictions & truth).sum().item()
    predicted_positive = predictions.sum().item()
    actual_positive = truth.sum().item()
    return float(true_positive), float(predicted_positive), float(actual_positive)


def counts_to_metrics(
    tp: float, predicted: float, actual: float, eps: float = 1e-7
) -> Dict[str, float]:
    dice = (2.0 * tp + eps) / (predicted + actual + eps)
    union = predicted + actual - tp
    iou = (tp + eps) / (union + eps)
    precision = (tp + eps) / (predicted + eps)
    recall = (tp + eps) / (actual + eps)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}



