import argparse
import math
import os
import random
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import morphology, segmentation
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert(mode))


def resize_to(arr: np.ndarray, size_wh, resample) -> np.ndarray:
    if arr.shape[1] == size_wh[0] and arr.shape[0] == size_wh[1]:
        return arr
    return np.asarray(Image.fromarray(arr).resize(size_wh, resample=resample))


def pad_to_patch(arr: np.ndarray, patch_size: int, value=0):
    h, w = arr.shape[:2]
    ph = max(patch_size, int(math.ceil(h / patch_size) * patch_size))
    pw = max(patch_size, int(math.ceil(w / patch_size) * patch_size))
    if arr.ndim == 2:
        out = np.full((ph, pw), value, dtype=arr.dtype)
        out[:h, :w] = arr
    else:
        out = np.full((ph, pw, arr.shape[2]), value, dtype=arr.dtype)
        out[:h, :w, :] = arr
    return out, (h, w)


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(max(1, radius))
    if radius >= 16 and hasattr(morphology, "isotropic_dilation"):
        return morphology.isotropic_dilation(mask, radius=radius)
    return morphology.binary_dilation(mask, morphology.disk(radius))


def load_triplet(image_path: Path, mask_path: Path, triple_path: Path):
    image = read_image(image_path, "RGB")
    mask = read_image(mask_path, "L")
    triple = read_image(triple_path, "L")
    size_wh = (image.shape[1], image.shape[0])
    mask = resize_to(mask, size_wh, Image.Resampling.NEAREST)
    triple = resize_to(triple, size_wh, Image.Resampling.BILINEAR)
    return image, mask, triple


def match_training_files(data_dir: Path):
    originals = list_images(data_dir / "original")
    masks = {p.name: p for p in list_images(data_dir / "mask")}
    triples = {p.name: p for p in list_images(data_dir / "triplepoint")}
    items = []
    missing = []
    for img in originals:
        if img.name in masks and img.name in triples:
            items.append((img, masks[img.name], triples[img.name]))
        else:
            missing.append(img.name)
    if not items:
        raise RuntimeError("No paired original/mask/triplepoint images were found.")
    if missing:
        print("Warning: skipped images without paired mask or triplepoint:", ", ".join(missing))
    return items


def make_soft_mask(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_bool.astype(np.float32)
    soft = mask_bool.astype(np.float32)
    band1 = dilate_bool(mask_bool, max(1, radius // 2))
    band2 = dilate_bool(mask_bool, radius)
    soft = np.maximum(soft, band1.astype(np.float32) * 0.55)
    soft = np.maximum(soft, band2.astype(np.float32) * 0.25)
    return soft.astype(np.float32)


def make_cell_interior(mask_bool: np.ndarray, barrier_radius: int, min_area: int) -> np.ndarray:
    barrier = dilate_bool(mask_bool, barrier_radius)
    interior = segmentation.clear_border(~barrier)
    interior = morphology.remove_small_objects(interior, min_size=max(1, min_area))
    return interior.astype(np.float32)


def make_supervision_maps(
    mask_u8: np.ndarray,
    triple_u8: np.ndarray,
    soft_radius: int,
    line_radius: int,
    triple_near_radius: int,
    triple_threshold: float,
    interior_barrier_radius: int,
    interior_min_area: int,
):
    mask = mask_u8 > 127
    triple = np.clip(triple_u8.astype(np.float32) / 255.0, 0.0, 1.0)
    soft_mask = make_soft_mask(mask, soft_radius)
    skeleton = morphology.skeletonize(mask)
    line_band = dilate_bool(skeleton, line_radius)
    triple_seed = triple >= triple_threshold
    if triple_seed.any():
        triple_neighborhood = dilate_bool(triple_seed, triple_near_radius)
        connection = line_band & triple_neighborhood
    else:
        triple_neighborhood = np.zeros_like(mask, dtype=bool)
        connection = np.zeros_like(mask, dtype=bool)
    interior = make_cell_interior(mask, interior_barrier_radius, interior_min_area)
    return {
        "mask": mask.astype(np.float32),
        "soft_mask": soft_mask,
        "triple": triple.astype(np.float32),
        "line_band": line_band.astype(np.float32),
        "connection": connection.astype(np.float32),
        "triple_neighborhood": triple_neighborhood.astype(np.float32),
        "interior": interior.astype(np.float32),
    }


class MixedPatchDataset(Dataset):
    def __init__(
        self,
        items,
        patch_size,
        patches_per_image,
        augment=True,
        focus_prob=0.7,
        soft_radius=2,
        line_radius=3,
        triple_near_radius=64,
        triple_threshold=0.2,
        interior_barrier_radius=2,
        interior_min_area=128,
    ):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.focus_prob = focus_prob
        self.samples = []
        for paths in items:
            image, mask_u8, triple_u8 = load_triplet(*paths)
            maps = make_supervision_maps(
                mask_u8,
                triple_u8,
                soft_radius=soft_radius,
                line_radius=line_radius,
                triple_near_radius=triple_near_radius,
                triple_threshold=triple_threshold,
                interior_barrier_radius=interior_barrier_radius,
                interior_min_area=interior_min_area,
            )
            image, _ = pad_to_patch(image, patch_size, value=0)
            padded_maps = {}
            for key, value in maps.items():
                padded_maps[key], _ = pad_to_patch(value, patch_size, value=0)
            priority = (
                (padded_maps["connection"] > 0)
                | (padded_maps["line_band"] > 0)
                | (padded_maps["triple_neighborhood"] > 0)
            )
            self.samples.append(
                {
                    "image": image,
                    "maps": padded_maps,
                    "priority_yx": np.argwhere(priority),
                }
            )

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def _choose_crop(self, sample):
        ps = self.patch_size
        h, w = sample["maps"]["mask"].shape
        if sample["priority_yx"].size and random.random() < self.focus_prob:
            cy, cx = sample["priority_yx"][random.randrange(len(sample["priority_yx"]))]
            y = int(np.clip(cy - random.randint(ps // 4, ps * 3 // 4), 0, max(0, h - ps)))
            x = int(np.clip(cx - random.randint(ps // 4, ps * 3 // 4), 0, max(0, w - ps)))
        else:
            y = 0 if h == ps else random.randint(0, h - ps)
            x = 0 if w == ps else random.randint(0, w - ps)
        return y, x

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        ps = self.patch_size
        y, x = self._choose_crop(sample)
        image = sample["image"][y : y + ps, x : x + ps]
        maps = {k: v[y : y + ps, x : x + ps] for k, v in sample["maps"].items()}

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, 1)
                maps = {k: np.flip(v, 1) for k, v in maps.items()}
            if random.random() < 0.5:
                image = np.flip(image, 0)
                maps = {k: np.flip(v, 0) for k, v in maps.items()}
            k_rot = random.randint(0, 3)
            if k_rot:
                image = np.rot90(image, k_rot)
                maps = {k: np.rot90(v, k_rot) for k, v in maps.items()}
            if random.random() < 0.3:
                image = image.astype(np.float32)
                image = image * random.uniform(0.9, 1.12) + random.uniform(-8, 8)
                image = np.clip(image, 0, 255).astype(np.uint8)

        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        tensors = {"image": torch.from_numpy(image.transpose(2, 0, 1))}
        for key in ["mask", "soft_mask", "triple", "line_band", "connection", "interior"]:
            tensors[key] = torch.from_numpy(np.ascontiguousarray(maps[key]).astype(np.float32)[None, ...])
        return tensors


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class PlusResUNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.e1 = ResidualBlock(in_ch, base)
        self.e2 = ResidualBlock(base, base * 2)
        self.e3 = ResidualBlock(base * 2, base * 4)
        self.e4 = ResidualBlock(base * 4, base * 8)
        self.b = ResidualBlock(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = ResidualBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ResidualBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ResidualBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ResidualBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def weighted_dice_loss(prob, target, weight, smooth=1.0):
    dims = (1, 2, 3)
    inter = (prob * target * weight).sum(dims)
    denom = (prob * weight).sum(dims) + (target * weight).sum(dims)
    return 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()


def focal_tversky_loss(prob, target, weight, alpha=0.45, beta=0.55, gamma=0.75, smooth=1.0):
    dims = (1, 2, 3)
    tp = (prob * target * weight).sum(dims)
    fp = (prob * (1.0 - target) * weight).sum(dims)
    fn = ((1.0 - prob) * target * weight).sum(dims)
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.pow(1.0 - ti, gamma).mean()


class BoundaryLoss(nn.Module):
    def __init__(
        self,
        bce_weight=0.8,
        focal_tversky_weight=0.8,
        soft_dice_weight=0.35,
        interior_fp_weight=0.35,
        positive_weight=1.3,
        triple_point_weight=1.0,
        connection_weight=2.0,
        line_weight=1.0,
        tversky_alpha=0.45,
        tversky_beta=0.55,
        tversky_gamma=0.75,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.soft_dice_weight = soft_dice_weight
        self.interior_fp_weight = interior_fp_weight
        self.positive_weight = positive_weight
        self.triple_point_weight = triple_point_weight
        self.connection_weight = connection_weight
        self.line_weight = line_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma

    def make_weight(self, batch):
        return (
            1.0
            + self.triple_point_weight * batch["triple"]
            + self.connection_weight * batch["connection"]
            + self.line_weight * batch["line_band"]
        )

    def forward(self, logits, batch):
        mask = batch["mask"]
        prob = torch.sigmoid(logits)
        weight = self.make_weight(batch)
        positive_bias = 1.0 + (self.positive_weight - 1.0) * mask
        bce_raw = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        bce = weighted_mean(bce_raw, weight * positive_bias)
        ft = focal_tversky_loss(
            prob,
            mask,
            weight,
            alpha=self.tversky_alpha,
            beta=self.tversky_beta,
            gamma=self.tversky_gamma,
        )
        soft_dice = weighted_dice_loss(prob, batch["soft_mask"], weight)
        interior_weight = batch["interior"]
        interior_fp = (prob * interior_weight).sum() / interior_weight.sum().clamp_min(1.0)
        total = (
            self.bce_weight * bce
            + self.focal_tversky_weight * ft
            + self.soft_dice_weight * soft_dice
            + self.interior_fp_weight * interior_fp
        )
        parts = {
            "loss": total.detach(),
            "bce": bce.detach(),
            "focal_tversky": ft.detach(),
            "soft_dice": soft_dice.detach(),
            "interior_fp": interior_fp.detach(),
        }
        return total, parts


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loaders, criterion, device, threshold=0.5):
    model.eval()
    losses, dices, precisions, recalls, interior_fps = [], [], [], [], []
    for loader in loaders:
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["image"])
            loss, _ = criterion(logits, batch)
            prob = torch.sigmoid(logits)
            pred = (prob > threshold).float()
            tp = (pred * batch["mask"]).sum(dim=(1, 2, 3))
            pred_sum = pred.sum(dim=(1, 2, 3))
            target_sum = batch["mask"].sum(dim=(1, 2, 3))
            dice = ((2 * tp + 1.0) / (pred_sum + target_sum + 1.0)).mean()
            precision = ((tp + 1.0) / (pred_sum + 1.0)).mean()
            recall = ((tp + 1.0) / (target_sum + 1.0)).mean()
            interior = batch["interior"]
            interior_fp = (prob * interior).sum() / interior.sum().clamp_min(1.0)
            losses.append(float(loss.item()))
            dices.append(float(dice.item()))
            precisions.append(float(precision.item()))
            recalls.append(float(recall.item()))
            interior_fps.append(float(interior_fp.item()))
    score = 0.55 * np.mean(dices) + 0.25 * np.mean(precisions) + 0.20 * np.mean(recalls) - 0.25 * np.mean(interior_fps)
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "interior_fp": float(np.mean(interior_fps)),
        "score": float(score),
    }


def kfold_indices(n, k, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = min(k, n)
    return [([i for i in idx if i not in idx[fold::k]], idx[fold::k]) for fold in range(k)]


def make_loaders(items, args, train=True):
    common = dict(
        items=items,
        augment=train,
        focus_prob=args.focus_prob if train else 0.0,
        soft_radius=args.soft_radius,
        line_radius=args.line_radius,
        triple_near_radius=args.triple_near_radius,
        triple_threshold=args.triple_threshold,
        interior_barrier_radius=args.interior_barrier_radius,
        interior_min_area=args.interior_min_area,
    )
    ds256 = MixedPatchDataset(
        patch_size=256,
        patches_per_image=args.patches_256_per_image if train else max(4, args.patches_256_per_image // 8),
        **common,
    )
    ds512 = MixedPatchDataset(
        patch_size=512,
        patches_per_image=args.patches_512_per_image if train else max(2, args.patches_512_per_image // 8),
        **common,
    )
    return (
        DataLoader(ds256, batch_size=args.batch_size_256, shuffle=train, num_workers=args.workers, pin_memory=True),
        DataLoader(ds512, batch_size=args.batch_size_512, shuffle=train, num_workers=args.workers, pin_memory=True),
    )


def train_one_loader(model, loader, criterion, optimizer, device):
    model.train()
    records = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = criterion(model(batch["image"]), batch)
        loss.backward()
        optimizer.step()
        records.append({k: float(v.item()) for k, v in parts.items()})
    return records


def mean_records(records, key):
    return float(np.mean([r[key] for r in records])) if records else 0.0


def train(args):
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = match_training_files(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device: {device}")
    print(f"samples={len(items)}, folds={min(args.folds, len(items))}")

    fold_records = []
    for fold, (train_idx, val_idx) in enumerate(kfold_indices(len(items), args.folds, args.seed), start=1):
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]
        print(f"\nFold {fold}: train={len(train_items)}, val={len(val_items)}")
        train256, train512 = make_loaders(train_items, args, train=True)
        val256, val512 = make_loaders(val_items, args, train=False)
        model = PlusResUNet(base=args.base_channels).to(device)
        criterion = BoundaryLoss(
            bce_weight=args.bce_loss_weight,
            focal_tversky_weight=args.focal_tversky_loss_weight,
            soft_dice_weight=args.soft_dice_loss_weight,
            interior_fp_weight=args.interior_fp_loss_weight,
            positive_weight=args.positive_weight,
            triple_point_weight=args.triple_point_weight,
            connection_weight=args.connection_weight,
            line_weight=args.line_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_score = -1e9
        best_path = out_dir / f"plusresunet_fold{fold}.pt"

        for epoch in range(1, args.epochs + 1):
            if random.random() < 0.5:
                records = train_one_loader(model, train256, criterion, optimizer, device)
                records += train_one_loader(model, train512, criterion, optimizer, device)
            else:
                records = train_one_loader(model, train512, criterion, optimizer, device)
                records += train_one_loader(model, train256, criterion, optimizer, device)
            scheduler.step()
            val = evaluate(model, [val256, val512], criterion, device, threshold=args.threshold)
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train={mean_records(records, 'loss'):.4f} "
                f"bce={mean_records(records, 'bce'):.4f} ft={mean_records(records, 'focal_tversky'):.4f} "
                f"soft={mean_records(records, 'soft_dice'):.4f} ifp={mean_records(records, 'interior_fp'):.4f} "
                f"val_dice={val['dice']:.4f} val_precision={val['precision']:.4f} "
                f"val_recall={val['recall']:.4f} val_ifp={val['interior_fp']:.4f} score={val['score']:.4f}"
            )
            if val["score"] > best_score:
                best_score = val["score"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "base_channels": args.base_channels,
                        "fold": fold,
                        "val": val,
                        "threshold": args.threshold,
                        "low_threshold": args.low_threshold,
                        "architecture": "PlusResUNet",
                        "loss_args": vars(args),
                    },
                    best_path,
                )
        fold_records.append((fold, best_score, best_path))

    best = max(fold_records, key=lambda x: x[1])
    best_model = out_dir / "best_model.pt"
    best_model.write_bytes(best[2].read_bytes())
    print(f"\nbest fold={best[0]}, score={best[1]:.4f}")
    print(f"saved: {best_model}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PlusResUNet for detonation cell boundary segmentation.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size-256", type=int, default=6)
    parser.add_argument("--batch-size-512", type=int, default=2)
    parser.add_argument("--patches-256-per-image", type=int, default=70)
    parser.add_argument("--patches-512-per-image", type=int, default=30)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focus-prob", type=float, default=0.7)
    parser.add_argument("--soft-radius", type=int, default=2)
    parser.add_argument("--line-radius", type=int, default=3)
    parser.add_argument("--triple-near-radius", type=int, default=64)
    parser.add_argument("--triple-threshold", type=float, default=0.2)
    parser.add_argument("--interior-barrier-radius", type=int, default=2)
    parser.add_argument("--interior-min-area", type=int, default=128)
    parser.add_argument("--bce-loss-weight", type=float, default=0.8)
    parser.add_argument("--focal-tversky-loss-weight", type=float, default=0.8)
    parser.add_argument("--soft-dice-loss-weight", type=float, default=0.35)
    parser.add_argument("--interior-fp-loss-weight", type=float, default=0.35)
    parser.add_argument("--positive-weight", type=float, default=1.3)
    parser.add_argument("--triple-point-weight", type=float, default=1.0)
    parser.add_argument("--connection-weight", type=float, default=2.0)
    parser.add_argument("--line-weight", type=float, default=1.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.45)
    parser.add_argument("--tversky-beta", type=float, default=0.55)
    parser.add_argument("--tversky-gamma", type=float, default=0.75)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--low-threshold", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
