"""
rotate_mosaics.py

Regenerate 90/180/270-degree rotated versions of specific mosaic images.

Default targets (based on your note about blank rotations):
- 10_037.png
- 10_130.png
- 10_165.png

Usage examples (PowerShell):
  # Regenerate the three known images in mosaics/ (overwrite if exist)
  python rotate_mosaics.py

  # Specify explicit files
  python rotate_mosaics.py --files mosaics/10_037.png mosaics/10_130.png mosaics/10_165.png

  # Provide a directory and pattern
  python rotate_mosaics.py --dir mosaics --names 10_037.png 10_130.png 10_165.png

Rotated files are saved alongside the originals as:
  <stem>_rot90.png, <stem>_rot180.png, <stem>_rot270.png
"""
from pathlib import Path
import argparse
from PIL import Image


def rotate_save(src: Path, angle: int) -> Path:
    stem = src.stem
    out = src.with_name(f"{stem}_rot{angle}.png")
    im = Image.open(src)
    # Use transpose for exact 90-degree rotations without resampling
    if angle == 90:
        rim = im.transpose(Image.ROTATE_90)
    elif angle == 180:
        rim = im.transpose(Image.ROTATE_180)
    elif angle == 270:
        rim = im.transpose(Image.ROTATE_270)
    else:
        # Fallback to general rotation for non-right-angle (not used here)
        rim = im.rotate(angle, expand=True)
    # Ensure PNG is saved; overwrite if exists
    rim.save(out)
    return out


def regenerate_rotations(files: list[Path]) -> None:
    angles = (90, 180, 270)
    for src in files:
        if not src.exists():
            print(f"Skip missing: {src}")
            continue
        try:
            for a in angles:
                out = rotate_save(src, a)
                print(f"Wrote: {out}")
        except Exception as e:
            print(f"Failed for {src}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=Path, default=Path('mosaics'), help='Base directory containing images')
    p.add_argument('--names', nargs='*', help='Image basenames to process (e.g., 10_037.png)')
    p.add_argument('--files', nargs='*', type=Path, help='Explicit image file paths')
    p.add_argument('--all-in-dir', action='store_true', help='Rotate all PNG/JPG images found in --dir')
    args = p.parse_args()

    files: list[Path] = []
    if args.all_in_dir:
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        for pat in patterns:
            for fp in sorted(args.dir.glob(pat)):
                # skip already-rotated outputs
                if '_rot90' in fp.stem or '_rot180' in fp.stem or '_rot270' in fp.stem:
                    continue
                files.append(fp)
    elif args.files:
        files.extend(args.files)
    elif args.names:
        files.extend([args.dir / n for n in args.names])
    else:
        # Default list requested
        files = [
            args.dir / '10_037.png',
            args.dir / '10_130.png',
            args.dir / '10_165.png',
        ]

    regenerate_rotations(files)


if __name__ == '__main__':
    main()
