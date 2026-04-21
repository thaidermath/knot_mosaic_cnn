#!/usr/bin/env python3
"""Split knot mosaic dataset into train/val/test with per-knot minimums."""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
MIN_COUNTS = {"val": 1, "test": 1}


def compute_split_counts(num_items: int) -> Tuple[int, int, int]:
    """Return counts for train, val, test while respecting minimums."""
    desired = {name: num_items * TARGET_RATIOS[name] for name in SPLITS}
    counts = {name: int(round(desired[name])) for name in SPLITS}
    for name, minimum in MIN_COUNTS.items():
        if counts[name] < minimum:
            counts[name] = minimum
    total = sum(counts.values())
    # Trim down if we overshoot because of the minimums or rounding.
    while total > num_items:
        for name in SPLITS:
            limit = MIN_COUNTS.get(name, 0)
            if counts[name] > limit:
                counts[name] -= 1
                total -= 1
                if total == num_items:
                    break
        else:
            raise ValueError("Could not reduce counts to match available items")
    # Backfill any leftover items into the training split.
    while total < num_items:
        counts["train"] += 1
        total += 1
    if sum(counts.values()) != num_items:
        raise ValueError("Split counts do not sum to the number of items")
    return counts["train"], counts["val"], counts["test"]


def prepare_destination(dest_root: Path, overwrite: bool) -> None:
    if dest_root.exists():
        if overwrite:
            shutil.rmtree(dest_root)
        elif any(dest_root.iterdir()):
            raise SystemExit(f"Destination {dest_root} is not empty; use --overwrite to replace it")
    dest_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (dest_root / split).mkdir(parents=True, exist_ok=True)


def collect_pngs(knot_dir: Path) -> List[Path]:
    pngs = sorted(knot_dir.glob("*.png"))
    return [p for p in pngs if p.is_file()]


def copy_pair(png_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, target_dir / png_path.name)
    matrix_path = png_path.with_name(f"{png_path.stem}_matrix.npy")
    if matrix_path.exists():
        shutil.copy2(matrix_path, target_dir / matrix_path.name)
    else:
        print(f"Warning: missing matrix for {png_path}")


def split_knot(knot_dir: Path, split_counts: Tuple[int, int, int], rng: random.Random, dest_root: Path) -> Dict[str, int]:
    pngs = collect_pngs(knot_dir)
    rng.shuffle(pngs)
    train_cnt, val_cnt, test_cnt = split_counts
    train_end = train_cnt
    val_end = train_cnt + val_cnt
    partitions = {
        "train": pngs[:train_end],
        "val": pngs[train_end:val_end],
        "test": pngs[val_end:val_end + test_cnt],
    }
    tallies = {name: 0 for name in SPLITS}
    for split_name, files in partitions.items():
        if not files:
            continue
        target_dir = dest_root / split_name / knot_dir.name
        for png_path in files:
            copy_pair(png_path, target_dir)
            tallies[split_name] += 1
    return tallies


def main() -> None:
    parser = argparse.ArgumentParser(description="Split knot mosaics into train/val/test folds per folder")
    parser.add_argument("--source", default="dataset_filtered", help="Path to filtered dataset root")
    parser.add_argument("--dest", default="dataset_split", help="Output directory for the split dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    parser.add_argument("--overwrite", action="store_true", help="Replace destination if it already exists")
    args = parser.parse_args()

    source_root = Path(args.source).expanduser().resolve()
    dest_root = Path(args.dest).expanduser().resolve()

    if not source_root.exists():
        raise SystemExit(f"Source directory {source_root} does not exist")

    prepare_destination(dest_root, args.overwrite)

    rng = random.Random(args.seed)
    summary = {name: 0 for name in SPLITS}
    processed = 0
    skipped = 0

    for knot_dir in sorted(source_root.iterdir()):
        if not knot_dir.is_dir():
            continue
        pngs = collect_pngs(knot_dir)
        num_items = len(pngs)
        if num_items == 0:
            skipped += 1
            continue
        try:
            split_counts = compute_split_counts(num_items)
        except ValueError as exc:
            print(f"Skipping {knot_dir.name}: {exc}")
            skipped += 1
            continue
        tallies = split_knot(knot_dir, split_counts, rng, dest_root)
        for name in SPLITS:
            summary[name] += tallies[name]
        processed += 1
        print(f"Split {knot_dir.name}: train={tallies['train']} val={tallies['val']} test={tallies['test']}")

    print("---")
    print(f"Processed folders: {processed}")
    if skipped:
        print(f"Skipped folders: {skipped}")
    print("Images per split:")
    for name in SPLITS:
        print(f"  {name}: {summary[name]}")


if __name__ == "__main__":
    main()
