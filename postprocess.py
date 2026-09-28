"""Skeletonize trajectory lines, reverse-fill closed cells and measure them."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.morphology import skeletonize


MEASUREMENT_COLUMNS = [
    "source_image",
    "structure_id",
    "top_y_px",
    "bottom_y_px",
    "left_x_px",
    "right_x_px",
    "top_to_bottom_px",
    "top_to_bottom_mm",
    "left_to_right_px",
    "left_to_right_mm",
    "area_px2",
    "area_mm2",
    "perimeter_px",
    "perimeter_mm",
    "center_x_px",
    "center_y_px",
]


def remove_small_components(binary: np.ndarray, minimum_area: int) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    if minimum_area <= 1:
        return binary_u8
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    cleaned = np.zeros_like(binary_u8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            cleaned[labels == label] = 1
    return cleaned


def skeletonize_prediction(
    binary: np.ndarray,
    closing_kernel: int = 3,
    minimum_line_component: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean the binary prediction and return (cleaned mask, one-pixel skeleton)."""
    cleaned = remove_small_components(binary, minimum_line_component)
    if closing_kernel > 0:
        if closing_kernel % 2 == 0:
            raise ValueError("closing_kernel must be 0 or an odd positive integer")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(cleaned > 0).astype(np.uint8)
    return cleaned, skeleton


def _interior_label_point(component: np.ndarray, left: int, top: int) -> Tuple[int, int]:
    """Choose the most interior pixel, which keeps the ID inside concave structures."""
    padded = cv2.copyMakeBorder(component.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, _, _, maximum_location = cv2.minMaxLoc(distance)
    return left + int(maximum_location[0]) - 1, top + int(maximum_location[1]) - 1


def reverse_fill_closed_regions(
    skeleton: np.ndarray,
    pixels_per_mm: float = 8.4,
    minimum_closed_area: int = 50,
    source_image: str = "",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Label inverse-skeleton components that do not touch an image edge.

    Four-connectivity is deliberately used for the background while the skeleton is
    eight-connected. This is the standard complementary-connectivity choice that
    allows diagonal one-pixel boundaries to enclose a digital region.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    if skeleton.ndim != 2:
        raise ValueError(f"Expected a 2-D skeleton, got shape {skeleton.shape}")
    background = (skeleton == 0).astype(np.uint8)
    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=4
    )

    border_labels = set(np.unique(component_labels[0, :]).tolist())
    border_labels.update(np.unique(component_labels[-1, :]).tolist())
    border_labels.update(np.unique(component_labels[:, 0]).tolist())
    border_labels.update(np.unique(component_labels[:, -1]).tolist())

    candidates: List[Tuple[int, int, int]] = []
    for component_label in range(1, count):
        if component_label in border_labels:
            continue
        area = int(stats[component_label, cv2.CC_STAT_AREA])
        if area < minimum_closed_area:
            continue
        top = int(stats[component_label, cv2.CC_STAT_TOP])
        left = int(stats[component_label, cv2.CC_STAT_LEFT])
        candidates.append((top, left, component_label))
    candidates.sort()

    structure_ids = np.zeros_like(component_labels, dtype=np.int32)
    rows: List[Dict[str, Union[float, int, str]]] = []
    for structure_id, (_, _, component_label) in enumerate(candidates, start=1):
        component = component_labels == component_label
        structure_ids[component] = structure_id
        ys, xs = np.nonzero(component)
        top_y, bottom_y = int(ys.min()), int(ys.max())
        left_x, right_x = int(xs.min()), int(xs.max())
        height_px = bottom_y - top_y + 1
        width_px = right_x - left_x + 1
        area_px = int(component.sum())

        crop = component[top_y : bottom_y + 1, left_x : right_x + 1].astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter_px = float(sum(cv2.arcLength(contour, True) for contour in contours))
        center_x, center_y = _interior_label_point(crop, left_x, top_y)

        rows.append(
            {
                "source_image": source_image,
                "structure_id": structure_id,
                "top_y_px": top_y,
                "bottom_y_px": bottom_y,
                "left_x_px": left_x,
                "right_x_px": right_x,
                "top_to_bottom_px": height_px,
                "top_to_bottom_mm": height_px / pixels_per_mm,
                "left_to_right_px": width_px,
                "left_to_right_mm": width_px / pixels_per_mm,
                "area_px2": area_px,
                "area_mm2": area_px / (pixels_per_mm**2),
                "perimeter_px": perimeter_px,
                "perimeter_mm": perimeter_px / pixels_per_mm,
                "center_x_px": center_x,
                "center_y_px": center_y,
            }
        )
    return structure_ids, pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)


def _structure_color(structure_id: int) -> Tuple[int, int, int]:
    # Golden-ratio hue stepping gives stable, well-separated colors.
    hue = (0.17 + structure_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def _draw_ids(canvas: np.ndarray, measurements: pd.DataFrame) -> None:
    height, width = canvas.shape[:2]
    font_scale = max(0.45, min(1.2, min(height, width) / 1200.0))
    thickness = max(1, round(font_scale * 2))
    for row in measurements.itertuples(index=False):
        text = str(int(row.structure_id))
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        origin = (
            int(row.center_x_px - text_size[0] / 2),
            int(row.center_y_px + text_size[1] / 2),
        )
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def make_color_diagrams(
    structure_ids: np.ndarray,
    skeleton: np.ndarray,
    measurements: pd.DataFrame,
    original_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    height, width = structure_ids.shape
    diagram = np.full((height, width, 3), 255, dtype=np.uint8)
    overlay = None
    if original_rgb is not None:
        if original_rgb.shape[:2] != (height, width):
            raise ValueError("original_rgb and structure_ids must have the same spatial shape")
        overlay = original_rgb[..., :3].astype(np.uint8, copy=True)

    for structure_id in range(1, int(structure_ids.max()) + 1):
        region = structure_ids == structure_id
        color = np.asarray(_structure_color(structure_id), dtype=np.uint8)
        diagram[region] = color
        if overlay is not None:
            overlay[region] = (0.42 * overlay[region] + 0.58 * color).astype(np.uint8)

    diagram[skeleton > 0] = (20, 20, 20)
    if overlay is not None:
        overlay[skeleton > 0] = (255, 40, 40)
    _draw_ids(diagram, measurements)
    if overlay is not None:
        _draw_ids(overlay, measurements)
    return diagram, overlay


def save_postprocess_outputs(
    output_dir: Union[str, Path],
    source_name: str,
    probability: np.ndarray,
    binary: np.ndarray,
    cleaned: np.ndarray,
    skeleton: np.ndarray,
    structure_ids: np.ndarray,
    measurements: pd.DataFrame,
    diagram: np.ndarray,
    overlay: Optional[np.ndarray],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_u16 = np.clip(probability * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(probability_u16).save(output_dir / "probability_uint16.tif")
    Image.fromarray((binary > 0).astype(np.uint8) * 255).save(output_dir / "binary_mask.tif")
    Image.fromarray((cleaned > 0).astype(np.uint8) * 255).save(output_dir / "cleaned_mask.tif")
    Image.fromarray((skeleton > 0).astype(np.uint8) * 255).save(output_dir / "skeleton_1px.tif")
    Image.fromarray(structure_ids.astype(np.int32), mode="I").save(output_dir / "closed_structure_ids.tif")
    Image.fromarray(diagram).save(output_dir / "closed_structures_color.png")
    if overlay is not None:
        Image.fromarray(overlay).save(output_dir / "closed_structures_overlay.png")
    measurements.to_csv(output_dir / "measurements.csv", index=False, encoding="utf-8-sig")
    metadata = pd.DataFrame(
        [
            {
                "source_image": source_name,
                "closed_structure_count": len(measurements),
            }
        ]
    )
    metadata.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
