"""TIFF pairing, normalization and foreground-aware patch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
from typing import Dict, List, Tuple, Union
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


SUPPORTED_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path

    @property
    def name(self) -> str:
        return self.image_path.stem


def _image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def discover_pairs(data_root: Union[str, Path]) -> List[ImageMaskPair]:
    data_root = Path(data_root)
    originals = _image_files(data_root / "original")
    masks = _image_files(data_root / "mask")
    mask_by_stem = {path.stem: path for path in masks}
    original_stems = {path.stem for path in originals}

    missing_masks = [path.name for path in originals if path.stem not in mask_by_stem]
    extra_masks = [path.name for path in masks if path.stem not in original_stems]
    if missing_masks or extra_masks:
        raise ValueError(
            "Image/mask pairing failed. "
            f"Missing masks for: {missing_masks or 'none'}; masks without images: {extra_masks or 'none'}"
        )
    if not originals:
        raise ValueError(f"No supported images found in {data_root / 'original'}")
    return [ImageMaskPair(path, mask_by_stem[path.stem]) for path in originals]


def split_pairs(
    pairs: List[ImageMaskPair],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[ImageMaskPair], List[ImageMaskPair]]:
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, min(len(pairs) - 1, round(len(pairs) * validation_fraction)))
    return shuffled[validation_count:], shuffled[:validation_count]


def image_shape(path: Union[str, Path]) -> Tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def find_shape_mismatches(pairs: List[ImageMaskPair]) -> List[Dict[str, object]]:
    mismatches: List[Dict[str, object]] = []
    for pair in pairs:
        image_hw = image_shape(pair.image_path)
        mask_hw = image_shape(pair.mask_path)
        if image_hw != mask_hw:
            mismatches.append({"name": pair.name, "image_hw": image_hw, "mask_hw": mask_hw})
    return mismatches


def _as_grayscale_array(path: Union[str, Path]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        rgb = array[..., :3].astype(np.float32)
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise ValueError(f"Unsupported image shape {array.shape} in {path}")


def normalize_image(
    array: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Robustly map a grayscale image to [0, 1]."""
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Normalization percentiles must satisfy 0 <= low < high <= 100")
    array = array.astype(np.float32, copy=False)
    low, high = np.percentile(array, (lower_percentile, upper_percentile))
    if high - low < 1e-6:
        low = float(array.min())
        high = float(array.max())
    if high - low < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def load_normalized_image(
    path: Union[str, Path],
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    return normalize_image(_as_grayscale_array(path), lower_percentile, upper_percentile)


@lru_cache(maxsize=16)
def _load_pair_cached(
    image_path_string: str,
    mask_path_string: str,
    lower_percentile: float,
    upper_percentile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    image = load_normalized_image(image_path_string, lower_percentile, upper_percentile)
    mask_array = _as_grayscale_array(mask_path_string)
    mask = mask_array > 0

    if mask.shape != image.shape:
        warnings.warn(
            f"Mask shape {mask.shape} does not match image shape {image.shape} for "
            f"{Path(image_path_string).name}; resizing the mask with nearest-neighbor interpolation. "
            "Please verify that the source annotation is spatially aligned.",
            stacklevel=2,
        )
        target_height, target_width = image.shape
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(
            mask_image.resize((target_width, target_height), resample=Image.Resampling.NEAREST)
        ) > 0
    return image, mask.astype(np.float32)


def _pad_to_patch(array: np.ndarray, patch_size: int, is_mask: bool) -> np.ndarray:
    height, width = array.shape
    pad_bottom = max(0, patch_size - height)
    pad_right = max(0, patch_size - width)
    if pad_bottom == 0 and pad_right == 0:
        return array
    if is_mask:
        return np.pad(array, ((0, pad_bottom), (0, pad_right)), mode="constant")
    # Reflect requires at least two pixels along a padded axis; edge is a safe fallback.
    mode = "reflect" if height > 1 and width > 1 else "edge"
    return np.pad(array, ((0, pad_bottom), (0, pad_right)), mode=mode)


class RandomPatchDataset(Dataset):
    """Sample native-resolution patches so thin trajectory lines are not downscaled away."""

    def __init__(
        self,
        pairs: List[ImageMaskPair],
        patch_size: int = 512,
        samples_per_epoch: int = 510,
        foreground_probability: float = 0.7,
        augment: bool = True,
        deterministic: bool = False,
        seed: int = 42,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
    ) -> None:
        if not pairs:
            raise ValueError("pairs cannot be empty")
        if patch_size < 32 or patch_size % 16 != 0:
            raise ValueError("patch_size must be >= 32 and divisible by 16")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0, 1]")
        self.pairs = pairs
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.foreground_probability = foreground_probability
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = (
            np.random.default_rng(self.seed + index)
            if self.deterministic
            else np.random.default_rng(np.random.randint(0, 2**31 - 1))
        )
        pair_index = index % len(self.pairs) if self.deterministic else int(rng.integers(len(self.pairs)))
        pair = self.pairs[pair_index]
        image, mask = _load_pair_cached(
            str(pair.image_path.resolve()),
            str(pair.mask_path.resolve()),
            self.lower_percentile,
            self.upper_percentile,
        )
        image = _pad_to_patch(image, self.patch_size, is_mask=False)
        mask = _pad_to_patch(mask, self.patch_size, is_mask=True)
        height, width = image.shape

        if rng.random() < self.foreground_probability and np.any(mask > 0.5):
            foreground_y, foreground_x = np.nonzero(mask > 0.5)
            chosen = int(rng.integers(len(foreground_y)))
            center_y, center_x = int(foreground_y[chosen]), int(foreground_x[chosen])
            offset_y = int(rng.integers(self.patch_size // 4, 3 * self.patch_size // 4 + 1))
            offset_x = int(rng.integers(self.patch_size // 4, 3 * self.patch_size // 4 + 1))
            top = int(np.clip(center_y - offset_y, 0, height - self.patch_size))
            left = int(np.clip(center_x - offset_x, 0, width - self.patch_size))
        else:
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))

        image_patch = image[top : top + self.patch_size, left : left + self.patch_size].copy()
        mask_patch = mask[top : top + self.patch_size, left : left + self.patch_size].copy()

        if self.augment:
            if rng.random() < 0.5:
                image_patch = np.fliplr(image_patch)
                mask_patch = np.fliplr(mask_patch)
            if rng.random() < 0.5:
                image_patch = np.flipud(image_patch)
                mask_patch = np.flipud(mask_patch)
            rotations = int(rng.integers(4))
            if rotations:
                image_patch = np.rot90(image_patch, rotations)
                mask_patch = np.rot90(mask_patch, rotations)
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.08, 0.08))
            image_patch = np.clip(image_patch * contrast + brightness, 0.0, 1.0)
            if rng.random() < 0.25:
                noise = rng.normal(0.0, 0.015, size=image_patch.shape).astype(np.float32)
                image_patch = np.clip(image_patch + noise, 0.0, 1.0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image_patch[None])).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_patch[None])).float()
        return image_tensor, mask_tensor

class BalancedPatchDataset(Dataset):
    """Sample one fixed patch size while balancing counts across source images."""

    def __init__(
        self,
        pairs: List[ImageMaskPair],
        patch_size: int = 512,
        samples_per_epoch: int = 580,
        foreground_probability: float = 0.7,
        augment: bool = True,
        seed: int = 42,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.5,
    ) -> None:
        if not pairs:
            raise ValueError("pairs cannot be empty")
        if patch_size < 32 or patch_size % 16 != 0:
            raise ValueError("patch_size must be >= 32 and divisible by 16")
        if samples_per_epoch < len(pairs):
            raise ValueError(
                "samples_per_epoch must be at least the number of training images "
                "so every image is sampled in every epoch"
            )
        if not 0.0 <= foreground_probability <= 1.0:
            raise ValueError("foreground_probability must be in [0, 1]")

        self.pairs = pairs
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.foreground_probability = float(foreground_probability)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.epoch = 0
        self.schedule: List[int] = []
        self.set_epoch(0)

    def _build_schedule(self, epoch: int) -> List[int]:
        rng = np.random.default_rng(self.seed + epoch * 100_003)
        image_count = len(self.pairs)
        counts = np.full(
            image_count, self.samples_per_epoch // image_count, dtype=np.int64
        )
        remainder = self.samples_per_epoch % image_count
        if remainder:
            extra_images = rng.permutation(image_count)[:remainder]
            counts[extra_images] += 1

        schedule: List[int] = []
        for image_index, count in enumerate(counts):
            schedule.extend([image_index] * int(count))
        rng.shuffle(schedule)
        return schedule

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.schedule = self._build_schedule(self.epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def sampling_summary(self) -> Dict[str, object]:
        per_image: Dict[str, int] = {
            pair.name: 0 for pair in self.pairs
        }
        for image_index in self.schedule:
            per_image[self.pairs[image_index].name] += 1
        return {
            "epoch": self.epoch,
            "total": len(self.schedule),
            "patch_size": self.patch_size,
            "per_image": per_image,
        }

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_index = self.schedule[index]
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + index * 97
        )
        pair = self.pairs[pair_index]
        image, mask = _load_pair_cached(
            str(pair.image_path.resolve()),
            str(pair.mask_path.resolve()),
            self.lower_percentile,
            self.upper_percentile,
        )
        image = _pad_to_patch(image, self.patch_size, is_mask=False)
        mask = _pad_to_patch(mask, self.patch_size, is_mask=True)
        height, width = image.shape

        if rng.random() < self.foreground_probability and np.any(mask > 0.5):
            foreground_y, foreground_x = np.nonzero(mask > 0.5)
            chosen = int(rng.integers(len(foreground_y)))
            center_y = int(foreground_y[chosen])
            center_x = int(foreground_x[chosen])
            offset_y = int(
                rng.integers(
                    self.patch_size // 4,
                    3 * self.patch_size // 4 + 1,
                )
            )
            offset_x = int(
                rng.integers(
                    self.patch_size // 4,
                    3 * self.patch_size // 4 + 1,
                )
            )
            top = int(
                np.clip(
                    center_y - offset_y,
                    0,
                    height - self.patch_size,
                )
            )
            left = int(
                np.clip(
                    center_x - offset_x,
                    0,
                    width - self.patch_size,
                )
            )
        else:
            top = int(rng.integers(0, height - self.patch_size + 1))
            left = int(rng.integers(0, width - self.patch_size + 1))

        image_patch = image[
            top : top + self.patch_size,
            left : left + self.patch_size,
        ].copy()
        mask_patch = mask[
            top : top + self.patch_size,
            left : left + self.patch_size,
        ].copy()

        if self.augment:
            if rng.random() < 0.5:
                image_patch = np.fliplr(image_patch)
                mask_patch = np.fliplr(mask_patch)
            if rng.random() < 0.5:
                image_patch = np.flipud(image_patch)
                mask_patch = np.flipud(mask_patch)
            rotations = int(rng.integers(4))
            if rotations:
                image_patch = np.rot90(image_patch, rotations)
                mask_patch = np.rot90(mask_patch, rotations)
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.08, 0.08))
            image_patch = np.clip(
                image_patch * contrast + brightness, 0.0, 1.0
            )
            if rng.random() < 0.25:
                noise = rng.normal(
                    0.0, 0.015, size=image_patch.shape
                ).astype(np.float32)
                image_patch = np.clip(image_patch + noise, 0.0, 1.0)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_patch[None])
        ).float()
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_patch[None])
        ).float()
        return image_tensor, mask_tensor
