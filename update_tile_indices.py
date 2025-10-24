"""
update_tile_indices.py

Regenerate tile_index.npy from updated tile_matrix.csv files.

Inputs:
- --matrices-root: folder containing <knot_name>/tile_matrix.csv

Behavior:
- Parses labels like 'tile_01'..'tile_11' or numeric '1'..'11'.
- Empty cells are mapped to 'tile_01' (1) unless --blank-zero is set.
- Writes tile_index.npy (N x N, uint8) alongside each CSV.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import numpy as np


def parse_cell(cell: str, blank_zero: bool) -> int:
    s = (cell or '').strip()
    if not s:
        return 0 if blank_zero else 1
    if s.startswith('tile_'):
        s = s.split('_', 1)[1]
    try:
        v = int(float(s))
    except Exception:
        v = 0 if blank_zero else 1
    return max(0, min(11, v))


def csv_to_index(csv_path: Path, blank_zero: bool) -> np.ndarray | None:
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    idx = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for i, r in enumerate(rows):
        for j in range(n_cols):
            val = rows[i][j] if j < len(r) else ''
            idx[i, j] = parse_cell(val, blank_zero)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrices-root', type=Path, default=Path('tile_matrices'))
    ap.add_argument('--blank-zero', action='store_true', help='Map empty cells to 0 instead of tile_01')
    args = ap.parse_args()

    root = args.matrices_root
    folders = [d for d in root.iterdir() if d.is_dir()]
    updated = 0
    skipped = 0
    for d in sorted(folders):
        csv_path = d / 'tile_matrix.csv'
        if not csv_path.exists():
            skipped += 1
            continue
        arr = csv_to_index(csv_path, args.blank_zero)
        if arr is None:
            skipped += 1
            continue
        np.save(d / 'tile_index.npy', arr)
        updated += 1
        print(f"Updated: {d.name} -> {arr.shape}")
    print(f"Done. Updated {updated}, skipped {skipped}.")


if __name__ == '__main__':
    main()

