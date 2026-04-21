"""Utilities for loading mosaic PNG datasets with balanced sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

__all__ = [
    "KnotMosaicDataset",
    "compute_sample_weights",
    "make_weighted_sampler",
    "build_image_dataloaders",
]


class KnotMosaicDataset(Dataset):
    """Dataset reading knot-specific PNG tiles from a split folder."""

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root!s} does not exist")
        self.transform = transform

        self.label_names: List[str] = []
        self.name_to_index: Dict[str, int] = {}
        self.samples: List[Tuple[Path, int]] = []

        for label_name in sorted(p.name for p in self.root.iterdir() if p.is_dir()):
            self.name_to_index[label_name] = len(self.label_names)
            self.label_names.append(label_name)

        for label_name in self.label_names:
            label_dir = self.root / label_name
            for png_path in sorted(label_dir.glob("*.png")):
                if png_path.is_file():
                    label = self.name_to_index[label_name]
                    self.samples.append((png_path, label))

        if not self.samples:
            raise RuntimeError(f"No PNG files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                np_img = np.asarray(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(np_img).permute(2, 0, 1)
        return tensor, label


def compute_sample_weights(samples: Sequence[Tuple[Path, int]]) -> torch.DoubleTensor:
    counts = Counter(label for _, label in samples)
    if not counts:
        raise ValueError("Cannot compute weights for empty sample list")
    weights = [1.0 / counts[label] for _, label in samples]
    return torch.DoubleTensor(weights)


def make_weighted_sampler(
    dataset: KnotMosaicDataset,
    *,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    weights = compute_sample_weights(dataset.samples)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def build_image_dataloaders(
    root: Path | str,
    batch_size: int,
    *,
    num_workers: int = 0,
    transforms_map: Optional[Dict[str, Callable[[Image.Image], torch.Tensor]]] = None,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Tuple[KnotMosaicDataset, DataLoader]]:
    root_path = Path(root)
    transforms_map = transforms_map or {}
    splits = {}
    for split in ("train", "val", "test"):
        split_dir = root_path / split
        if not split_dir.exists():
            continue
        transform = transforms_map.get(split)
        dataset = KnotMosaicDataset(split_dir, transform=transform)
        if split == "train":
            sampler = make_weighted_sampler(dataset, generator=generator)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        splits[split] = (dataset, loader)
    if not splits:
        raise FileNotFoundError(f"No dataset splits found under {root_path}")
    return splits
