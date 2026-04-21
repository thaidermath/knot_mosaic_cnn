#!/usr/bin/env python3
"""Plot a full confusion matrix for a ResNet18 knot checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def load_model(ckpt_path: Path, num_classes: int) -> Tuple[nn.Module, List[str], dict]:
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
    return model, label_names, checkpoint.get("args", {})


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


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)


def select_top_labels(cm: np.ndarray, label_names: Sequence[str], top_n: int) -> Tuple[np.ndarray, List[str]]:
    if top_n <= 0 or top_n >= len(label_names):
        return cm, list(label_names)
    supports = cm.sum(axis=1)
    top_indices = np.argsort(supports)[::-1][:top_n]
    top_indices = np.sort(top_indices)
    subset_cm = cm[np.ix_(top_indices, top_indices)]
    subset_labels = [label_names[idx] for idx in top_indices]
    return subset_cm, subset_labels


def build_title(default_title: str, checkpoint_name: str, split: str, accuracy: float, args: dict) -> str:
    if default_title:
        return default_title
    weight_decay = args.get("weight_decay")
    label_smoothing = args.get("label_smoothing")
    return (
        f"{checkpoint_name} | {split} | acc={accuracy:.3f}\n"
        f"wd={weight_decay} ls={label_smoothing}"
    )


def plot_confusion(
    cm: np.ndarray,
    labels: Sequence[str],
    output_path: Path,
    title: str,
    normalize: bool,
    dpi: int,
) -> None:
    plot_cm = normalize_rows(cm) if normalize else cm.astype(float)
    n_labels = len(labels)
    fig_size = max(7, min(14, 0.28 * n_labels + 3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(plot_cm, cmap="Blues", vmin=0.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    tick_fontsize = 6 if n_labels > 40 else 7 if n_labels > 25 else 8
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(labels, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(labels, fontsize=tick_fontsize)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    annotate = n_labels <= 20
    if annotate:
        for i in range(n_labels):
            for j in range(n_labels):
                value = plot_cm[i, j]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot confusion matrix for a ResNet knot classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--normalize", action="store_true", help="Normalize each row to fractions")
    parser.add_argument("--top-n", type=int, default=20, help="Plot only the top-N labels by support; use 0 for all labels")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--title", default="", help="Optional custom figure title")
    parser.add_argument("--output", type=Path, help="Destination PNG path")
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

    model, label_names, ckpt_args = load_model(args.checkpoint, num_classes=len(dataset.label_names))
    targets, preds = gather_predictions(model, loader, dataset.label_names, label_names)
    accuracy = float((targets == preds).mean()) if len(targets) else 0.0
    cm = confusion_matrix(targets, preds, labels=list(range(len(label_names))))
    cm, label_names = select_top_labels(cm, label_names, args.top_n)

    if args.output is None:
        suffix = "_row_norm" if args.normalize else ""
        top_n_suffix = f"_top{args.top_n}" if args.top_n > 0 else "_all"
        output = Path("plots") / f"{args.checkpoint.stem}_{args.split}_confusion{suffix}{top_n_suffix}.png"
    else:
        output = args.output

    title = build_title(args.title, args.checkpoint.stem, args.split, accuracy, ckpt_args)
    plot_confusion(cm, label_names, output, title, args.normalize, args.dpi)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved confusion matrix to {output}")


if __name__ == "__main__":
    main()
