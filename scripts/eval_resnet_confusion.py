#!/usr/bin/env python3
"""Evaluate a ResNet18 checkpoint on knot mosaics and summarise confusions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_mosaic_images import KnotMosaicDataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_model(ckpt_path: Path, num_classes: int) -> Tuple[nn.Module, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    label_names = checkpoint.get("label_names")
    if label_names is None:
        raise RuntimeError("Checkpoint is missing label_names metadata")
    recorded_classes = checkpoint.get("num_classes", len(label_names))
    if recorded_classes != num_classes:
        raise RuntimeError(
            f"Checkpoint expects {recorded_classes} classes but dataset supplies {num_classes}"
        )
    weights = None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, label_names


def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    dataset_names: Sequence[str],
    label_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    targets: list[int] = []
    preds: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            batch_preds = outputs.argmax(dim=1).cpu().numpy()
            for dataset_label, pred_idx in zip(labels.tolist(), batch_preds.tolist()):
                knot_name = dataset_names[dataset_label]
                target_idx = label_to_idx.get(knot_name)
                if target_idx is None:
                    raise RuntimeError(f"Label {knot_name} not present in checkpoint metadata")
                targets.append(target_idx)
                preds.append(pred_idx)
    return np.array(targets, dtype=np.int64), np.array(preds, dtype=np.int64)


def summarise_confusions(cm: np.ndarray, label_names: Sequence[str], top_k: int) -> List[Tuple[str, str, int, float]]:
    mis = cm.copy()
    np.fill_diagonal(mis, 0)
    flat = mis.flatten()
    order = np.argsort(flat)[::-1]
    num_classes = cm.shape[0]
    results: list[Tuple[str, str, int, float]] = []
    taken = 0
    for idx in order:
        count = int(flat[idx])
        if count <= 0:
            break
        true_idx = idx // num_classes
        pred_idx = idx % num_classes
        row_total = cm[true_idx].sum()
        frac = float(count / row_total) if row_total else 0.0
        results.append((label_names[true_idx], label_names[pred_idx], count, frac))
        taken += 1
        if taken >= top_k:
            break
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise confusion pairs for a ResNet knot classifier")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/resnet18_knot.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset_split"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=20, help="Number of misclassification pairs to display")
    parser.add_argument("--csv", type=Path, help="Optional CSV destination for the reported confusion pairs")
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    if not split_dir.exists():
        raise SystemExit(f"Split directory {split_dir} is missing")

    transform = build_transform(args.image_size)
    dataset = KnotMosaicDataset(split_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_label_names = dataset.label_names
    model, label_names = load_model(args.checkpoint, num_classes=len(dataset_label_names))
    targets, preds = gather_predictions(model, loader, dataset_label_names, label_names)

    accuracy = (targets == preds).mean() if len(targets) else 0.0
    print(f"Split {args.split} accuracy: {accuracy:.4f}")

    cm = confusion_matrix(targets, preds, labels=list(range(len(label_names))))
    top_pairs = summarise_confusions(cm, label_names, args.top_k)

    if top_pairs:
        print("Top misclassified pairs (true -> predicted):")
        for true_name, pred_name, count, frac in top_pairs:
            print(f"  {true_name} -> {pred_name}: count={count} row_share={frac:.2%}")
    else:
        print("No misclassifications recorded.")

    if args.csv and top_pairs:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as fh:
            fh.write("true_label,predicted_label,count,row_fraction\n")
            for true_name, pred_name, count, frac in top_pairs:
                fh.write(f"{true_name},{pred_name},{count},{frac:.6f}\n")
        print(f"Wrote confusion summary to {args.csv}")


if __name__ == "__main__":
    main()
