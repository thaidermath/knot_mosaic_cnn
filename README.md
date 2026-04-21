# Flype-Equivalent Knot Mosaic Experiments

This repository contains the code used to train and evaluate image classifiers on
flype-equivalent knot mosaic diagrams. The current workflow uses PNG mosaics
grouped by knot label, filters classes by the number of available flype variants,
splits the dataset into train/validation/test folds, and fine-tunes ResNet18
classifiers on those images.

Large generated datasets, checkpoints, logs, plots, and legacy rotated-image
artifacts are intentionally ignored by Git. Keep those files locally or publish
them through a data release mechanism rather than committing them to the source
repository.

## Repository Contents

- `dataset_mosaic_images.py` - PyTorch dataset and dataloader utilities for
  split PNG mosaic datasets.
- `split_dataset.py` - creates train/validation/test splits from
  `dataset_filtered/`.
- `split_dataset_min10.py` - creates splits from `dataset_filtered_10/`, keeping
  classes with at least 10 flype-equivalent examples.
- `train_resnet18.py` - fine-tunes a ResNet18 classifier with class-balanced
  sampling.
- `scripts/eval_resnet_confusion.py` - evaluates a checkpoint and reports the
  most common confusion pairs.
- `scripts/plot_resnet_confusion_matrix.py` - plots full or top-N confusion
  matrices for a trained ResNet18 checkpoint.
- `scripts/plot_confusion_subset_dual.py` - compares confusion matrices across
  two ResNet experiments on a shared label subset.
- `scripts/plot_resnet18_knot_training_log.py` and
  `scripts/plot_resnet18_knot_min10_training_log.py` - reproduce saved training
  metric plots from fixed epoch tables.
- `pd_to_mosaic_3.py` and `render_pd_mosaic.py` - utilities for rendering mosaic
  images from PD codes.

## Expected Local Data Layout

The training scripts expect generated data to exist locally in one of these
ignored folders:

```text
dataset_filtered/
  3_1/
    3_1_01.png
    3_1_01_matrix.npy
    ...
dataset_split/
  train/<knot_label>/*.png
  val/<knot_label>/*.png
  test/<knot_label>/*.png
```

The min-10 experiment uses the same layout under `dataset_filtered_10/` and
`dataset_split_10/`.

## Basic Workflow

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a split dataset:

```powershell
python split_dataset.py --source dataset_filtered --dest dataset_split --overwrite
```

Or create the min-10 split:

```powershell
python split_dataset_min10.py --source dataset_filtered_10 --dest dataset_split_10 --overwrite
```

Train ResNet18:

```powershell
python train_resnet18.py --data-root dataset_split --epochs 50 --output checkpoints/resnet18_knot.pt --evaluate-test
```

Evaluate the most common confusion pairs:

```powershell
python scripts/eval_resnet_confusion.py --checkpoint checkpoints/resnet18_knot.pt --data-root dataset_split --split test --csv plots/resnet18_confusions.csv
```

Plot a normalized confusion matrix:

```powershell
python scripts/plot_resnet_confusion_matrix.py --checkpoint checkpoints/resnet18_knot.pt --data-root dataset_split --split test --normalize --top-n 20
```

## Notes

- Generated data is not committed because the flype-equivalent image datasets
  are multi-gigabyte local artifacts.
- The previous rotated-image and tile-probability workflow has been removed
  from the tracked project surface so GitHub reflects only the current
  flype-equivalent mosaic experiments.
