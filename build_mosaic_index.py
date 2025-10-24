"""
build_mosaic_index.py

Scan mosaics_from_matrices and write an index CSV of mosaics (original + rotations).

Output columns:
- name: file stem (e.g., 10_165 or 10_165_rot90)
- path: relative path to PNG
- base_id: base knot id without rotation suffix (e.g., 10_165)
- is_rotation: 0/1
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import re


def base_knot_id(name: str) -> str:
    return re.sub(r'_rot(90|180|270)$', '', name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mosaics-root', type=Path, default=Path('mosaics_from_matrices'))
    p.add_argument('--out', type=Path, default=Path('mosaics_from_matrices/index.csv'))
    args = p.parse_args()

    rows = []
    for img in sorted(args.mosaics_root.glob('*.png')):
        name = img.stem
        if name == 'cells':
            continue
        base = base_knot_id(name)
        is_rot = 1 if name != base else 0
        rows.append({'name': name, 'path': str(img), 'base_id': base, 'is_rotation': is_rot})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'path', 'base_id', 'is_rotation'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote {len(rows)} entries to {args.out}')


if __name__ == '__main__':
    main()

