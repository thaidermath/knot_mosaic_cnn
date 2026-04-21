#!/usr/bin/env python3
"""Plot confusion matrices for a shared subset of knot labels across two experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

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
    model = models.resnet18(weights=None)
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


def count_by_label(dataset: KnotMosaicDataset) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in dataset.label_names}
    for _, label_idx in dataset.samples:
        counts[dataset.label_names[label_idx]] += 1
    return counts


def select_shared_labels(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    counts_a: Dict[str, int],
    counts_b: Dict[str, int],
    top_n: int,
) -> List[str]:
    shared = set(labels_a) & set(labels_b)
    if not shared:
        raise RuntimeError("No shared labels found between the two datasets")
    scored = []
    for name in shared:
        score = min(counts_a.get(name, 0), counts_b.get(name, 0))
        scored.append((score, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:top_n]]


def build_confusion_for_labels(
    targets: np.ndarray,
    preds: np.ndarray,
    label_names: Sequence[str],
    subset: Sequence[str],
) -> np.ndarray:
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    subset_indices = [label_to_idx[name] for name in subset]
    mask = np.isin(targets, subset_indices)
    filtered_targets = targets[mask]
    filtered_preds = preds[mask]
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=subset_indices)
    return cm


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return normalized


def plot_confusion(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
    normalize: str,
) -> None:
    if normalize == "row":
        plot_cm = normalize_rows(cm)
    else:
        plot_cm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(plot_cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if normalize == "row":
                text = f"{plot_cm[i, j]:.2f}"
            else:
                text = f"{int(plot_cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_dataset(split_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[KnotMosaicDataset, DataLoader]:
    transform = build_transform(image_size)
    dataset = KnotMosaicDataset(split_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrices for shared knot labels")
    parser.add_argument("--ckpt-a", type=Path, required=True, help="Checkpoint for experiment A")
    parser.add_argument("--ckpt-b", type=Path, required=True, help="Checkpoint for experiment B")
    parser.add_argument("--data-root-a", type=Path, required=True, help="Data root for experiment A")
    parser.add_argument("--data-root-b", type=Path, required=True, help="Data root for experiment B")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--top-n", type=int, default=5, help="Number of shared labels to plot")
    parser.add_argument("--labels", type=str, help="Comma-separated label list to use instead of auto-select")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize", choices=["none", "row"], default="row")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/confusion_subset"))
    args = parser.parse_args()

    split_dir_a = args.data_root_a / args.split
    split_dir_b = args.data_root_b / args.split
    if not split_dir_a.exists():
        raise SystemExit(f"Split directory {split_dir_a} is missing")
    if not split_dir_b.exists():
        raise SystemExit(f"Split directory {split_dir_b} is missing")

    dataset_a, loader_a = build_dataset(split_dir_a, args.image_size, args.batch_size, args.num_workers)
    dataset_b, loader_b = build_dataset(split_dir_b, args.image_size, args.batch_size, args.num_workers)

    if args.labels:
        subset = [name.strip() for name in args.labels.split(",") if name.strip()]
    else:
        counts_a = count_by_label(dataset_a)
        counts_b = count_by_label(dataset_b)
        subset = select_shared_labels(dataset_a.label_names, dataset_b.label_names, counts_a, counts_b, args.top_n)

    print("Selected labels:")
    for name in subset:
        print(f"  {name}")

    model_a, labels_a = load_model(args.ckpt_a, num_classes=len(dataset_a.label_names))
    targets_a, preds_a = gather_predictions(model_a, loader_a, dataset_a.label_names, labels_a)
    cm_a = build_confusion_for_labels(targets_a, preds_a, labels_a, subset)

    model_b, labels_b = load_model(args.ckpt_b, num_classes=len(dataset_b.label_names))
    targets_b, preds_b = gather_predictions(model_b, loader_b, dataset_b.label_names, labels_b)
    cm_b = build_confusion_for_labels(targets_b, preds_b, labels_b, subset)

    tag_a = args.ckpt_a.stem
    tag_b = args.ckpt_b.stem
    out_a = args.output_dir / f"confusion_{tag_a}_{args.split}.png"
    out_b = args.output_dir / f"confusion_{tag_b}_{args.split}.png"

    plot_confusion(cm_a, subset, f"{tag_a} ({args.split})", out_a, args.normalize)
    plot_confusion(cm_b, subset, f"{tag_b} ({args.split})", out_b, args.normalize)

    print(f"Saved: {out_a}")
    print(f"Saved: {out_b}")


if __name__ == "__main__":
    main()
