# Knot Mosaic CNN

End-to-end toolkit for aligning KnotInfo metadata with mosaic images and training CNN-based classifiers on tile-probability arrays.

## Data merge overview

The `merge_knot_data.py` workflow joins `knotinfo.csv` with the downloaded mosaics in `mosaics/` and writes `mosaics/merged_knotinfo.csv`. Snake_case columns capture the parsed mosaic and tile counts alongside the matching image reference.

### Key files

- `knotinfo.csv` — original KnotInfo metadata (includes `Mosaic/Tile-Number` values like `{ 7 ; 27 }`).
- `mosaics/index.csv` — index for downloaded mosaic images (`name`, `filename`, `url`).
- `mosaics/merged_knotinfo.csv` — merged output produced by `merge_knot_data.py`.
- `merge_knot_data.py` — matching script that builds the merged CSV.

### Columns in `mosaics/merged_knotinfo.csv`

Important fields (snake_case):

- `Name` — original knot name (e.g. `10_1`).
- `crossing_number`, `jones_polynomial`, `hyperbolic_volume`, `meridian_length` — selected KnotInfo columns.
- `mosaic_num`, `tile_num` — parsed values from `Mosaic/Tile-Number` or inferred from the matched filename.
- `image_filename_matched`, `image_url_matched` — chosen mosaic reference.

Notes
- `Mosaic/Tile-Number` is dropped once `mosaic_num` and `tile_num` are split out.
- Matching strategy: exact filename → normalized ID (e.g. `10_001` ↔ `10_1`) → normalized name → fuzzy fallback.

## Tile-prob dataset and experiments

Pipeline components for training classifiers on tile-probability tensors:

- `tile_matrices/` — source tile-matrix images.
- `tile_probs_from_matrices/` — per-knot directories with `tile_probs.npy` inputs.
- `create_rotated_tile_probs.py` — generates `_rot90`, `_rot180`, `_rot270` augments.
- `dataset_tile_probs.py` — dataset loader for numpy arrays.
- `model_tileprob_cnn.py` — shared CNN feature extractor.
- `train_cnn.py` — supervised baseline (train on rotations, validate on originals).
- `train_prototypical.py` — episodic prototypical trainer (N-way K-shot with augmentation).

## Evaluation and plotting

- `scripts/parse_proto_logs_and_plot.py` — parse training logs and emit CSVs plus plots (e.g. `plots/compare_top3_3shot.png`).

## Outputs

- `logs/` — training logs.
- `checkpoints/` — saved model weights.
- `plots/` — generated CSV/PNG artifacts.
- `reports/` — LaTeX write-ups (not tracked in git).

## Quick start

```powershell
./knotenv/Scripts/Activate.ps1
python merge_knot_data.py
python create_rotated_tile_probs.py
python train_prototypical.py --epochs 10 --episodes-per-epoch 200 --N 20 --K 5 --Q 5 --eval --save-every 1 --save-dir checkpoints/proto_run
python scripts/parse_proto_logs_and_plot.py
```
