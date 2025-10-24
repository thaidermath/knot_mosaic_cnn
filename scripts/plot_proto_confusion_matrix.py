"""Plot a confusion matrix for a trained prototypical network checkpoint.

Usage example:
  python scripts/plot_proto_confusion_matrix.py \
      --ckpt checkpoints/proto_20way_3shot/proto_epoch_10.pt \
      --output plots/confusion_proto_20way3shot.png

The script builds class prototypes from rotated training folders and evaluates the
specified checkpoint on canonical (non-rotated) examples, mirroring the evaluation
logic used during episodic training.  Results are saved as a PNG image and, if
requested, as a CSV file containing the raw counts.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_tile_probs import TileProbsDataset
from model_tileprob_cnn import TileProbCNN


class EmbeddingNet(torch.nn.Module):
    """Wrap `TileProbCNN` to expose embedding vectors matching training."""

    def __init__(self, base: TileProbCNN):
        super().__init__()
        self.features = base.features
        self.proj = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 2, 128),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.proj(x)
        return x


def load_checkpoint(path: Path, device: torch.device) -> EmbeddingNet:
    dataset = TileProbsDataset()
    base = TileProbCNN(in_channels=11, num_classes=len(dataset.name2id), hidden=128)
    net = EmbeddingNet(base).to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state["model_state"])
    net.eval()
    return net


def build_prototypes(net: EmbeddingNet, dataset: TileProbsDataset, device: torch.device) -> Dict[str, torch.Tensor]:
    pools: Dict[str, List[Path]] = {}
    for sample_path, label in dataset.samples:
        name = dataset.id2name[label]
        folder = sample_path.parent
        key = folder
        if "rot" in folder.name:
            pools.setdefault(name, []).append(folder)
    # fallback: ensure every class has at least one folder
    for name in dataset.name2id.keys():
        if name not in pools or len(pools[name]) == 0:
            pools.setdefault(name, [])
            for sample_path, label in dataset.samples:
                if dataset.id2name[label] == name:
                    pools[name].append(sample_path.parent)
                    break

    prototypes: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, folders in pools.items():
            embeddings = []
            for folder in folders:
                tile_path = folder / "tile_probs.npy"
                if not tile_path.exists():
                    continue
                arr = np.load(str(tile_path)).astype(np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
                embedding = net(tensor).squeeze(0).cpu()
                embeddings.append(embedding)
            if embeddings:
                prototypes[name] = torch.stack(embeddings, dim=0).mean(dim=0)
    return prototypes


def gather_predictions(net: EmbeddingNet, dataset: TileProbsDataset, prototypes: Dict[str, torch.Tensor], device: torch.device):
    proto_names = sorted(prototypes.keys())
    if not proto_names:
        raise RuntimeError("No prototypes were generated; confirm the dataset folders are populated.")
    proto_matrix = torch.stack([prototypes[name] for name in proto_names], dim=0).to(device)
    name_to_index = {name: idx for idx, name in enumerate(proto_names)}

    grouped: Dict[str, Dict[str, List[Path]]] = {}
    for sample_path, label in dataset.samples:
        name = dataset.id2name[label]
        entry = grouped.setdefault(name, {"canonical": [], "rotated": []})
        if "rot" in sample_path.parent.name:
            entry["rotated"].append(sample_path)
        else:
            entry["canonical"].append(sample_path)

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for name, buckets in grouped.items():
            if name not in name_to_index:
                continue
            eval_paths = buckets["canonical"] or buckets["rotated"]
            for sample_path in eval_paths:
                arr = np.load(str(sample_path)).astype(np.float32)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
                embedding = net(tensor)
                dists = torch.cdist(embedding, proto_matrix, p=2).squeeze(0)
                pred_idx = torch.argmin(dists).item()
                y_true.append(name_to_index[name])
                y_pred.append(pred_idx)

    return np.array(y_true, dtype=int), np.array(y_pred, dtype=int), proto_names


def plot_confusion(cm: np.ndarray, labels: List[str], output_path: Path, normalize: bool, title: str):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
    else:
        cm_display = cm.astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    n_labels = len(labels)
    if n_labels <= 20:
        tick_positions = np.arange(n_labels)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlim(-0.5, cm_display.shape[1] - 0.5)
    ax.set_ylim(cm_display.shape[0] - 0.5, -0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_csv(cm: np.ndarray, labels: List[str], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as f:
        header = ",".join(["label"] + labels)
        f.write(header + "\n")
        for idx, row in enumerate(cm):
            values = ",".join(str(int(v)) for v in row)
            f.write(f"{labels[idx]},{values}\n")


def main():
    parser = argparse.ArgumentParser(description="Plot a confusion matrix for a prototypical checkpoint.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the prototypical checkpoint (.pt).")
    parser.add_argument("--output", type=Path, default=Path("plots/confusion_proto.png"), help="Destination PNG path.")
    parser.add_argument("--csv", type=Path, help="Optional CSV path to save raw confusion counts.")
    parser.add_argument("--device", default="cpu", help="Torch device identifier (e.g., cpu or cuda:0).")
    parser.add_argument("--normalize", action="store_true", help="Row-normalize the confusion matrix before plotting.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = TileProbsDataset()
    net = load_checkpoint(args.ckpt, device)
    prototypes = build_prototypes(net, dataset, device)
    y_true, y_pred, proto_names = gather_predictions(net, dataset, prototypes, device)

    if y_true.size == 0:
        raise RuntimeError("No evaluation samples were collected; ensure canonical folders exist.")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(proto_names))))
    plot_confusion(cm, proto_names, args.output, args.normalize,
                   title=f"Confusion matrix: {args.ckpt.parent.name} (epoch {args.ckpt.stem.split('_')[-1]})")

    if args.csv:
        save_csv(cm, proto_names, args.csv)

    print(f"Wrote confusion matrix figure to {args.output}")
    if args.csv:
        print(f"Wrote raw confusion counts to {args.csv}")


if __name__ == "__main__":
    main()
