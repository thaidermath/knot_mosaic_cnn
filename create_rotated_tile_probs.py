"""
Create rotated variants of tile_probs.npy files.
For each folder under tile_probs_from_matrices that contains tile_probs.npy, this script will create
three new folders with suffixes _rot90, _rot180, _rot270 and save the rotated arrays there, unless they
already exist.

This lets the existing training split logic (which looks for 'rot' in folder names) find training samples.
"""
from pathlib import Path
import numpy as np
import shutil

ROOT = Path(__file__).parent
TP_DIR = ROOT / 'tile_probs_from_matrices'
ANGLES = [90, 180, 270]

if not TP_DIR.exists():
    print("Directory not found:", TP_DIR)
    raise SystemExit(1)

folders = [p for p in TP_DIR.iterdir() if p.is_dir()]
print(f"Found {len(folders)} folders in {TP_DIR}")

created = 0
skipped = 0
for f in folders:
    src = f / 'tile_probs.npy'
    if not src.exists():
        # skip folders without tile_probs.npy
        continue
    try:
        arr = np.load(src)
    except Exception as e:
        print(f"Failed to load {src}: {e}")
        continue

    for angle in ANGLES:
        rot_name = f"{f.name}_rot{angle}"
        rot_folder = TP_DIR / rot_name
        rot_file = rot_folder / 'tile_probs.npy'
        if rot_file.exists():
            skipped += 1
            continue
        # compute k for np.rot90: k=1 for 90, k=2 for 180, k=3 for 270
        k = angle // 90
        # rotate on last two axes
        try:
            rot_arr = np.rot90(arr, k=k, axes=(-2, -1))
        except Exception as e:
            print(f"Failed to rot90 for {src} angle {angle}: {e}")
            continue
        rot_folder.mkdir(parents=True, exist_ok=True)
        np.save(rot_file, rot_arr)
        created += 1

print(f"Created {created} rotated tile_probs files (skipped {skipped} that already existed)")
# extra: report how many folders now contain 'rot' in their name
all_folders = [p.name for p in TP_DIR.iterdir() if p.is_dir()]
rot_count = sum(1 for n in all_folders if 'rot' in n)
print(f"Total folders after creation: {len(all_folders)}; folders with 'rot' in name: {rot_count}")
