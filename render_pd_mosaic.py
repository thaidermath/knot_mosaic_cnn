import argparse
from pathlib import Path

from pd_to_mosaic_3 import mosaic_from_pd_auto, render_mosaic_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a knot mosaic image from a PD code using tile sprites."
    )
    parser.add_argument(
        "--pd",
        required=True,
        help="PD string, e.g. PD[X[1, 6, 2, 7], X[3, 8, 4, 9], ...]",
    )
    parser.add_argument(
        "--tiles-dir",
        default="tiles",
        help="Directory containing tile PNGs (default: ./tiles)",
    )
    parser.add_argument(
        "--output",
        default="rendered_mosaic.png",
        help="Output image filepath (default: rendered_mosaic.png)",
    )
    parser.add_argument(
        "--filler-tile",
        type=int,
        default=1,
        help="Tile ID to use for empty cells (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = mosaic_from_pd_auto(args.pd)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_mosaic_image(
        matrix,
        tiles_dir=args.tiles_dir,
        out_path=str(output_path),
        filler_tile=args.filler_tile,
    )
    print(f"Saved mosaic to {output_path.resolve()}")


if __name__ == "__main__":
    main()
