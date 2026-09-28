import argparse
import csv
import fnmatch
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import filters, measure, morphology, segmentation

try:
    from .train_plusresunet import IMAGE_EXTS, PlusResUNet, list_images, pad_to_patch, read_image
except ImportError:
    from train_plusresunet import IMAGE_EXTS, PlusResUNet, list_images, pad_to_patch, read_image


ROOT = Path(__file__).resolve().parent


def save_u8(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_u16(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint16)).save(path)


def color_table(n: int):
    rng = np.random.default_rng(2026)
    colors = [np.array([0, 0, 0], dtype=np.uint8)]
    for _ in range(n):
        colors.append(rng.integers(45, 245, size=3, dtype=np.uint8))
    return colors


def colorize_labels(labels: np.ndarray, draw_ids=True) -> Image.Image:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    colors = color_table(int(labels.max()))
    for lab in range(1, int(labels.max()) + 1):
        out[labels == lab] = colors[lab]
    img = Image.fromarray(out)
    if draw_ids:
        draw_label_ids(img, labels)
    return img


def draw_label_ids(img: Image.Image, labels: np.ndarray) -> None:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for prop in measure.regionprops(labels):
        if prop.area < 20:
            continue
        y, x = prop.centroid
        text = str(int(prop.label))
        draw.text((int(x), int(y)), text, fill=(255, 255, 255), font=font, anchor="mm", stroke_width=2, stroke_fill=(0, 0, 0))


def blend_overlay(image_rgb: np.ndarray, mask: np.ndarray, labels: np.ndarray) -> Image.Image:
    base = image_rgb.astype(np.float32).copy()
    mask_bool = mask > 0
    base[mask_bool] = base[mask_bool] * 0.40 + np.array([255, 255, 255], dtype=np.float32) * 0.60
    label_img = np.asarray(colorize_labels(labels, draw_ids=False)).astype(np.float32)
    label_bool = labels > 0
    base[label_bool] = base[label_bool] * 0.45 + label_img[label_bool] * 0.55
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw_label_ids(img, labels)
    return img


def apply_prediction_boost(prob, white_boost, smooth_sigma):
    boosted = np.clip(prob, 0.0, 1.0)
    if white_boost > 1.0:
        boosted = 1.0 - np.power(1.0 - boosted, white_boost)
    if smooth_sigma > 0:
        smoothed = filters.gaussian(boosted, sigma=smooth_sigma, preserve_range=True)
        boosted = np.maximum(boosted, smoothed * 0.9)
    return np.clip(boosted, 0.0, 1.0)


def binarize_probability(prob, high_threshold, low_threshold, mode, min_line_area):
    if mode == "hard":
        mask = prob >= high_threshold
    elif mode == "hysteresis":
        mask = filters.apply_hysteresis_threshold(prob, low_threshold, high_threshold)
    else:
        raise ValueError(f"unknown binarize mode: {mode}")
    if min_line_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=min_line_area)
    return mask


def smooth_binary_mask(mask, close_radius, open_radius, hole_area, min_object_area):
    out = mask.astype(bool)
    if close_radius > 0:
        out = morphology.binary_closing(out, morphology.disk(close_radius))
    if hole_area > 0:
        out = morphology.remove_small_holes(out, area_threshold=hole_area)
    if open_radius > 0:
        out = morphology.binary_opening(out, morphology.disk(open_radius))
    if min_object_area > 0:
        out = morphology.remove_small_objects(out, min_size=min_object_area)
    return out


def skeletonize_mask(mask, method, min_distance):
    if method == "medial":
        skeleton, distance = morphology.medial_axis(mask, return_distance=True)
        if min_distance > 0:
            skeleton &= distance >= min_distance
        return skeleton
    if method == "skeletonize":
        return morphology.skeletonize(mask)
    raise ValueError(f"unknown skeleton method: {method}")


def neighbor_count(skel):
    padded = np.pad(skel.astype(np.uint8), 1)
    count = np.zeros_like(skel, dtype=np.uint8)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            count += padded[1 + dy : 1 + dy + skel.shape[0], 1 + dx : 1 + dx + skel.shape[1]]
    return count


def trace_from_endpoint(skel, start, max_len):
    h, w = skel.shape
    path = [start]
    prev = None
    cur = start
    for _ in range(max_len + 1):
        y, x = cur
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and (ny, nx) != prev:
                    neighbors.append((ny, nx))
        if len(neighbors) != 1:
            return path, len(neighbors) >= 2
        prev, cur = cur, neighbors[0]
        path.append(cur)
    return path, False


def prune_skeleton_spurs(skeleton, max_spur_length, iterations):
    skel = skeleton.copy()
    for _ in range(max(1, iterations)):
        counts = neighbor_count(skel)
        endpoints = list(map(tuple, np.argwhere(skel & (counts == 1))))
        removed = 0
        for ep in endpoints:
            if not skel[ep]:
                continue
            path, ended_at_branch = trace_from_endpoint(skel, ep, max_spur_length)
            if ended_at_branch and len(path) <= max_spur_length:
                for p in path[:-1]:
                    if skel[p]:
                        skel[p] = False
                        removed += 1
        if removed == 0:
            break
    return skel


@torch.no_grad()
def predict_image(model, image_rgb, patch_size, overlap, device):
    padded, (orig_h, orig_w) = pad_to_patch(image_rgb, patch_size, value=0)
    h, w = padded.shape[:2]
    stride = patch_size - overlap
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    prob_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y in ys:
        for x in xs:
            patch = padded[y : y + patch_size, x : x + patch_size].astype(np.float32) / 255.0
            tensor = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device)
            prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            prob_sum[y : y + patch_size, x : x + patch_size] += prob
            count[y : y + patch_size, x : x + patch_size] += 1.0
    prob = prob_sum / np.maximum(count, 1e-6)
    return prob[:orig_h, :orig_w]


def analyze_closed_structures(binary_bool, args):
    clean_mask = smooth_binary_mask(
        binary_bool,
        close_radius=args.pre_skeleton_close_radius,
        open_radius=args.pre_skeleton_open_radius,
        hole_area=args.pre_skeleton_hole_area,
        min_object_area=args.min_line_area,
    )
    skeleton_raw = skeletonize_mask(clean_mask, args.skeleton_method, args.medial_min_distance)
    skeleton = prune_skeleton_spurs(skeleton_raw, args.spur_prune_length, args.spur_prune_iterations)
    barrier = morphology.binary_dilation(skeleton, morphology.disk(max(1, args.barrier_radius)))
    filled = segmentation.clear_border(~barrier)
    min_area_px = max(1, int(round(args.min_area_mm2 * args.pixel_per_mm * args.pixel_per_mm)))
    filled = morphology.remove_small_objects(filled, min_size=min_area_px)
    labels = measure.label(filled, connectivity=2).astype(np.uint16)

    records = []
    for prop in measure.regionprops(labels):
        minr, minc, maxr, maxc = prop.bbox
        height_px = maxr - minr
        width_px = maxc - minc
        height_mm = height_px / args.pixel_per_mm
        width_mm = width_px / args.pixel_per_mm
        records.append(
            {
                "id": int(prop.label),
                "top_to_bottom_mm": height_mm,
                "left_to_right_mm": width_mm,
                "height_width_ratio": float(height_mm / width_mm) if width_mm else float("inf"),
                "area_mm2": float(prop.area) / (args.pixel_per_mm * args.pixel_per_mm),
            }
        )
    return clean_mask, skeleton_raw, skeleton, labels, records


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "top_to_bottom_mm", "left_to_right_mm", "height_width_ratio", "area_mm2"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "top_to_bottom_mm": f"{row['top_to_bottom_mm']:.4f}",
                    "left_to_right_mm": f"{row['left_to_right_mm']:.4f}",
                    "height_width_ratio": f"{row['height_width_ratio']:.6f}",
                    "area_mm2": f"{row['area_mm2']:.4f}",
                }
            )


def load_model(model_path: Path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = PlusResUNet(base=ckpt.get("base_channels", 32)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    dirs = {
        "mask": out_dir / "binary_mask",
        "prob": out_dir / "probability",
        "boost": out_dir / "probability_boosted",
        "clean": out_dir / "cleaned_mask",
        "raw_skel": out_dir / "skeleton_raw",
        "skel": out_dir / "skeleton_1px",
        "label": out_dir / "structure_id",
        "color": out_dir / "colored_mask",
        "overlay": out_dir / "overlay",
        "table": out_dir / "measurements",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(Path(args.model), device)
    threshold = args.threshold if args.threshold is not None else float(ckpt.get("threshold", 0.5))
    low_threshold = args.low_threshold if args.low_threshold is not None else float(ckpt.get("low_threshold", 0.2))
    overlap = min(args.overlap, args.patch_size - 1)
    print(
        f"device={device}, patch_size={args.patch_size}, threshold={threshold}, "
        f"low_threshold={low_threshold}, pixel_per_mm={args.pixel_per_mm}"
    )

    images = list_images(data_dir / "predict")
    if args.image_glob:
        images = [p for p in images if fnmatch.fnmatch(p.name, args.image_glob)]
    if args.max_images:
        images = images[: args.max_images]

    all_rows = []
    for img_path in images:
        print(f"processing: {img_path.name}")
        image = read_image(img_path, "RGB")
        prob = predict_image(model, image, args.patch_size, overlap, device)
        boosted = apply_prediction_boost(prob, args.white_boost, args.smooth_sigma)
        binary = binarize_probability(boosted, threshold, low_threshold, args.binarize_mode, args.min_line_area)
        clean_mask, raw_skeleton, skeleton, labels, records = analyze_closed_structures(binary, args)
        stem = img_path.stem
        save_u8(dirs["mask"] / f"{stem}_mask.png", binary.astype(np.uint8) * 255)
        save_u8(dirs["prob"] / f"{stem}_prob.png", np.clip(prob * 255.0, 0, 255).astype(np.uint8))
        save_u8(dirs["boost"] / f"{stem}_prob_boosted.png", np.clip(boosted * 255.0, 0, 255).astype(np.uint8))
        save_u8(dirs["clean"] / f"{stem}_cleaned_mask.png", clean_mask.astype(np.uint8) * 255)
        save_u8(dirs["raw_skel"] / f"{stem}_skeleton_raw.png", raw_skeleton.astype(np.uint8) * 255)
        save_u8(dirs["skel"] / f"{stem}_skeleton.png", skeleton.astype(np.uint8) * 255)
        save_u16(dirs["label"] / f"{stem}_structure_id.tif", labels)
        colorize_labels(labels, draw_ids=True).save(dirs["color"] / f"{stem}_colored_mask.png")
        blend_overlay(image, binary.astype(np.uint8) * 255, labels).save(dirs["overlay"] / f"{stem}_overlay.png")
        write_csv(dirs["table"] / f"{stem}_measurements.csv", records)
        for row in records:
            row = dict(row)
            row["image"] = img_path.name
            all_rows.append(row)

    summary_path = out_dir / "all_measurements.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "id", "top_to_bottom_mm", "left_to_right_mm", "height_width_ratio", "area_mm2"],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "id": row["id"],
                    "top_to_bottom_mm": f"{row['top_to_bottom_mm']:.4f}",
                    "left_to_right_mm": f"{row['left_to_right_mm']:.4f}",
                    "height_width_ratio": f"{row['height_width_ratio']:.6f}",
                    "area_mm2": f"{row['area_mm2']:.4f}",
                }
            )
    print(f"done: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PlusResUNet prediction and real-scale cell measurements.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--model", default=str(ROOT / "weights" / "best_model.pt"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--low-threshold", type=float, default=None)
    parser.add_argument("--white-boost", type=float, default=1.05)
    parser.add_argument("--smooth-sigma", type=float, default=0.4)
    parser.add_argument("--binarize-mode", choices=["hysteresis", "hard"], default="hysteresis")
    parser.add_argument("--min-line-area", type=int, default=24)
    parser.add_argument("--pre-skeleton-close-radius", type=int, default=2)
    parser.add_argument("--pre-skeleton-open-radius", type=int, default=1)
    parser.add_argument("--pre-skeleton-hole-area", type=int, default=96)
    parser.add_argument("--skeleton-method", choices=["medial", "skeletonize"], default="medial")
    parser.add_argument("--medial-min-distance", type=float, default=1.5)
    parser.add_argument("--spur-prune-length", type=int, default=28)
    parser.add_argument("--spur-prune-iterations", type=int, default=4)
    parser.add_argument("--barrier-radius", type=int, default=1)
    parser.add_argument("--min-area-mm2", type=float, default=5.0)
    parser.add_argument("--pixel-per-mm", type=float, default=8.4)
    parser.add_argument("--image-glob", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
