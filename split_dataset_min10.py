#!/usr/bin/env python3
"""Split filtered knot mosaics with per-class minimum counts into train/val/test."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from split_dataset import collect_pngs, copy_pair

SPLITS = ("train", "val", "test")


def compute_split_counts_min10(num_items: int) -> Tuple[int, int, int]:
    """Return counts with 10% val/test (floored), minimum 1 each, remainder train."""
    if num_items < 10:
        raise ValueError("Expected at least 10 images per knot")
    val_cnt = max(1, int(num_items * 0.1))
    test_cnt = max(1, int(num_items * 0.1))
    train_cnt = num_items - val_cnt - test_cnt
    if train_cnt < 1:
        raise ValueError("Not enough images to allocate train/val/test")
    return train_cnt, val_cnt, test_cnt


def prepare_destination(dest_root: Path, overwrite: bool) -> None:
    if dest_root.exists():
        if overwrite:
            shutil.rmtree(dest_root)
        elif any(dest_root.iterdir()):
            raise SystemExit(f"Destination {dest_root} already exists and is not empty; use --overwrite to replace it")
    for split in SPLITS:
        (dest_root / split).mkdir(parents=True, exist_ok=True)


def split_knot(
    knot_dir: Path,
    split_counts: Tuple[int, int, int],
    rng,
    dest_root: Path,
) -> Dict[str, int]:
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
    parser = argparse.ArgumentParser(description="Split mosaic dataset with minimum 10 images per knot")
    parser.add_argument("--source", default="dataset_filtered_10", help="Root directory containing filtered knot folders")
    parser.add_argument("--dest", default="dataset_split_10", help="Destination root for train/val/test splits")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    parser.add_argument("--overwrite", action="store_true", help="Replace destination directory if it exists")
    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve()

    if not source_root.exists():
        raise SystemExit(f"Source directory {source_root} does not exist")

    import random

    rng = random.Random(args.seed)
    prepare_destination(dest_root, args.overwrite)

    summary = {name: 0 for name in SPLITS}
    processed = 0

    for knot_dir in sorted(source_root.iterdir()):
        if not knot_dir.is_dir():
            continue
        pngs = collect_pngs(knot_dir)
        num_items = len(pngs)
        if num_items < 10:
            continue
        split_counts = compute_split_counts_min10(num_items)
        tallies = split_knot(knot_dir, split_counts, rng, dest_root)
        for name in SPLITS:
            summary[name] += tallies[name]
        processed += 1
        print(
            f"Split {knot_dir.name}: train={tallies['train']} val={tallies['val']} test={tallies['test']}"
        )

    print("---")
    print(f"Processed knots: {processed}")
    print("Images per split:")
    for name in SPLITS:
        print(f"  {name}: {summary[name]}")


if __name__ == "__main__":
    main()
