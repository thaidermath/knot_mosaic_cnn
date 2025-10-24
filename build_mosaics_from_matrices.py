"""
build_mosaics_from_matrices.py

Reconstruct mosaic images from per-knot tile matrices using template tiles.

Inputs:
- --matrices-root: directory containing subfolders <name>/tile_matrix.csv
- --tiles-dir: directory with tile_01.png .. tile_11.png (RGBA preferred)

Output:
- One PNG per matrix written to --out/<name>.png

Notes:
- Empty/unknown cells ('', '0') are rendered with tile_01.png by default.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
from typing import List
from PIL import Image


def load_matrix_csv(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append([c.strip() for c in r])
    # Normalize row lengths (pad shorter rows)
    if not rows:
        return rows
    N = max(len(r) for r in rows)
    for i in range(len(rows)):
        if len(rows[i]) < N:
            rows[i] = rows[i] + [''] * (N - len(rows[i]))
    return rows


def compose_one(name: str, matrix_csv: Path, tiles_dir: Path, out_dir: Path) -> Path | None:
    labels = load_matrix_csv(matrix_csv)
    if not labels:
        return None
    n_rows = len(labels)
    n_cols = max((len(r) for r in labels), default=0)
    if n_rows == 0 or n_cols == 0:
        return None
    # Determine tile size from tile_01 (fallback to first existing tile)
    tile_path = tiles_dir / 'tile_01.png'
    if not tile_path.exists():
        for k in range(1, 12):
            p = tiles_dir / f'tile_{k:02d}.png'
            if p.exists():
                tile_path = p
                break
    with Image.open(tile_path).convert('RGBA') as t0:
        tile_w, tile_h = t0.size
    mosaic = Image.new('RGBA', (tile_w * n_cols, tile_h * n_rows), (255, 255, 255, 255))
    # Paste per cell
    for r in range(n_rows):
        row = labels[r]
        for c in range(n_cols):
            lab = row[c] if c < len(row) else ''
            if not lab:
                lab = 'tile_01'
            # Accept forms like 'tile_3' or '3'
            if lab.startswith('tile_'):
                suffix = lab.split('_', 1)[1]
            else:
                suffix = lab
            try:
                k = int(suffix)
            except Exception:
                k = 1
            k = max(1, min(11, k))
            tpath = tiles_dir / f'tile_{k:02d}.png'
            if not tpath.exists():
                tpath = tiles_dir / 'tile_01.png'
            with Image.open(tpath).convert('RGBA') as ti:
                # Paste at cell position
                x = c * tile_w
                y = r * tile_h
                mosaic.paste(ti, (x, y), ti)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.png'
    mosaic.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrices-root', type=Path, default=Path('tile_matrices'))
    ap.add_argument('--tiles-dir', type=Path, default=Path('tiles'))
    ap.add_argument('--out', type=Path, default=Path('mosaics_from_matrices'))
    ap.add_argument('--preview', type=int, default=0)
    args = ap.parse_args()

    folders = [d for d in args.matrices_root.iterdir() if d.is_dir()]
    created = 0
    for i, d in enumerate(sorted(folders), 1):
        mcsv = d / 'tile_matrix.csv'
        if not mcsv.exists():
            continue
        outp = compose_one(d.name, mcsv, args.tiles_dir, args.out)
        if outp is not None:
            print(f'[{i}] Wrote {outp}')
            created += 1
            if args.preview and created >= args.preview:
                break
    print(f'Done. Created {created} mosaics.')


if __name__ == '__main__':
    main()
