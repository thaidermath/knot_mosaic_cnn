"""
build_probs_from_matrices.py

Create per-mosaic tile probability tensors (11,H,W) from tile_matrices by converting indices to one-hot probabilities with optional label smoothing.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrices-root', type=Path, default=Path('tile_matrices'))
    ap.add_argument('--out-root', type=Path, default=Path('tile_probs_from_matrices'))
    ap.add_argument('--smooth', type=float, default=0.0, help='Label smoothing epsilon (e.g., 0.05)')
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    folders = [d for d in args.matrices_root.iterdir() if d.is_dir()]
    for i, d in enumerate(sorted(folders), 1):
        idx_path = d / 'tile_index.npy'
        if not idx_path.exists():
            # try to infer from CSV if needed
            continue
        idx = np.load(idx_path)  # (H,W) ints in 0..11
        H, W = idx.shape
        probs = np.full((11, H, W), fill_value=args.smooth / 10.0, dtype=np.float32) if args.smooth > 0 else np.zeros((11, H, W), dtype=np.float32)
        for k in range(1, 12):
            mask = (idx == k)
            if args.smooth > 0:
                probs[k-1][mask] = 1.0 - args.smooth
            else:
                probs[k-1][mask] = 1.0
        out_dir = args.out_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'tile_probs.npy', probs)
        print(f"[{i}] {d.name} -> {out_dir / 'tile_probs.npy'}")


if __name__ == '__main__':
    main()

