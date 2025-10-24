"""Evaluate classical baselines (k-NN and SVM) on tile-probability mosaics.

The script mirrors the prototypical-network setup:
- Training uses rotated variants when available and falls back to originals otherwise.
- Evaluation is performed on the canonical (non-rotated) folders, again falling back to
  any available sample if no canonical example exists.

Outputs overall top-1/top-2/top-3 accuracy for each classifier and, optionally,
serialises the results as JSON for later reporting.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
import json
import sys

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_tile_probs import TileProbsDataset


@dataclass
class SplitEntry:
    label: int
    label_name: str
    folder: str
    vector: np.ndarray
    is_rotated: bool


def load_tile_prob_vectors(include_rotations: bool = True) -> Tuple[List[SplitEntry], int, int]:
    """Load and pad all tile-probability tensors, returning flattened vectors.

    Returns a tuple of (entries, max_height, max_width).
    """
    dataset = TileProbsDataset()
    entries: List[SplitEntry] = []
    max_h = 0
    max_w = 0

    # First pass: load arrays and track maximum spatial size
    raw_arrays: List[Tuple[int, str, str, np.ndarray]] = []
    for path, label in dataset.samples:
        arr = np.load(str(path)).astype(np.float32)
        c, h, w = arr.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        folder_name = path.parent.name
        label_name = dataset.id2name[label]
        raw_arrays.append((label, label_name, folder_name, arr))

    def pad_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        c, h, w = arr.shape
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Encountered tile array larger than the globally computed maximum.")
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

    for label, label_name, folder_name, arr in raw_arrays:
        padded = pad_to(arr, max_h, max_w)
        vector = padded.reshape(-1)
        entry = SplitEntry(
            label=label,
            label_name=label_name,
            folder=folder_name,
            vector=vector,
            is_rotated="rot" in folder_name,
        )
        entries.append(entry)

    return entries, max_h, max_w


def build_splits(entries: List[SplitEntry]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[str, int]], List[str], List[str]]:
    """Create train/test splits grouped by label.

    Training prefers rotated variants; evaluation prefers canonical (non-rotated) samples.
    Falls back to whatever is available when a class lacks the preferred variant.
    Returns numpy arrays for train/test along with per-class statistics and notes on fallbacks.
    """
    by_label: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for entry in entries:
        info = by_label.setdefault(entry.label, {"train": [], "test": [], "name": entry.label_name})
        if entry.is_rotated:
            info["train"].append(entry.vector)
        else:
            info["test"].append(entry.vector)

    train_vectors: List[np.ndarray] = []
    train_labels: List[int] = []
    test_vectors: List[np.ndarray] = []
    test_labels: List[int] = []
    class_stats: Dict[int, Dict[str, int]] = {}
    missing_train: List[str] = []
    missing_test: List[str] = []

    for label, info in by_label.items():
        name = info["name"]
        if len(info["train"]) == 0:
            missing_train.append(name)
            info["train"].extend(info["test"])
        if len(info["test"]) == 0:
            missing_test.append(name)
            info["test"].extend(info["train"])
        class_stats[label] = {
            "name": name,
            "train_count": len(info["train"]),
            "test_count": len(info["test"]),
        }
        for vec in info["train"]:
            train_vectors.append(vec)
            train_labels.append(label)
        for vec in info["test"]:
            test_vectors.append(vec)
            test_labels.append(label)

    train_X = np.stack(train_vectors)
    test_X = np.stack(test_vectors)
    train_y = np.array(train_labels)
    test_y = np.array(test_labels)
    return train_X, train_y, test_X, test_y, class_stats, missing_train, missing_test


def compute_topk_metrics(scores: np.ndarray, classes: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    order = np.argsort(scores, axis=1)[:, ::-1]
    n = true_labels.shape[0]
    top1 = 0
    top2 = 0
    top3 = 0
    for idx in range(n):
        label = true_labels[idx]
        ranking = classes[order[idx]]
        if ranking.size == 0:
            continue
        if label == ranking[0]:
            top1 += 1
        if label in ranking[: min(2, ranking.size)]:
            top2 += 1
        if label in ranking[: min(3, ranking.size)]:
            top3 += 1
    denom = float(n) if n > 0 else 1.0
    return {
        "top1": top1 / denom,
        "top2": top2 / denom,
        "top3": top3 / denom,
    }


def evaluate_model(name: str, estimator, train_X, train_y, test_X, test_y) -> Dict[str, float]:
    estimator.fit(train_X, train_y)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(test_X)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(test_X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
    else:
        # As a last resort, fall back to hard predictions
        preds = estimator.predict(test_X)
        acc = (preds == test_y).mean() if len(test_y) else 0.0
        return {"top1": acc, "top2": acc, "top3": acc}
    classes = estimator.classes_
    return compute_topk_metrics(scores, classes, test_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_metrics.json"),
                        help="Optional JSON destination for the aggregated results.")
    parser.add_argument("--neighbors", type=int, default=3, help="k value for the k-NN classifier.")
    parser.add_argument("--svm-c", type=float, default=1.0, help="Regularisation strength for the SVM baseline.")
    parser.add_argument("--svm-gamma", type=str, default="scale", help="Gamma parameter for the RBF SVM kernel.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the JSON results file.")
    args = parser.parse_args()

    entries, max_h, max_w = load_tile_prob_vectors()
    train_X, train_y, test_X, test_y, class_stats, missing_train, missing_test = build_splits(entries)

    print(f"Loaded {len(entries)} samples padded to spatial size ({max_h}, {max_w}).")
    print(f"Train set: {train_X.shape[0]} samples; Test set: {test_X.shape[0]} samples.")
    if missing_train:
        print("Classes without rotated training examples (used originals instead):")
        print(", ".join(sorted(missing_train)))
    if missing_test:
        print("Classes without canonical evaluation examples (fell back to training samples):")
        print(", ".join(sorted(missing_test)))

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=args.neighbors, weights="distance")),
    ])
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=args.svm_c, gamma=args.svm_gamma, probability=True, decision_function_shape="ovr")),
    ])

    results: Dict[str, Dict[str, float]] = {}
    for name, estimator in [("knn", knn), ("svm", svm)]:
        metrics = evaluate_model(name, estimator, train_X, train_y, test_X, test_y)
        results[name] = metrics
        print(f"{name.upper()} metrics: top1={metrics['top1']:.4f} top2={metrics['top2']:.4f} top3={metrics['top3']:.4f}")

    aggregate = {
        "train_samples": int(train_X.shape[0]),
        "test_samples": int(test_X.shape[0]),
        "missing_rotated_classes": sorted(missing_train),
        "missing_original_classes": sorted(missing_test),
        "metrics": results,
    }

    if not args.no_save:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Wrote baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
