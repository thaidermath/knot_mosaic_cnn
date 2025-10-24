"""
dataset_tile_probs.py

PyTorch Dataset that loads `tile_probs_from_matrices/*/tile_probs.npy` and maps them to labels via `merged_knotinfo.csv`.

Each sample is returned as a torch.FloatTensor of shape (C, H, W) where C is number of tile classes (11) and HxW is the grid size (e.g., 6x6 for mosaic_num 6). We'll use padding or adaptive pooling in the model to handle variable sizes.

"""
from pathlib import Path
import csv
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
import numpy as np

ROOT = Path('.')
MERGED = ROOT / 'merged_knotinfo.csv'
TILE_PROBS_DIR = ROOT / 'tile_probs_from_matrices'


def build_label_map(merged_csv: Path) -> Tuple[Dict[str,int], Dict[int,str]]:
    # map image_name (without path) -> label id (0..K-1)
    names = []
    with open(merged_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            names.append(r['Name'])
    unique = sorted(set(names))
    name2id = {n:i for i,n in enumerate(unique)}
    id2name = {i:n for n,i in name2id.items()}
    return name2id, id2name


class TileProbsDataset(Dataset):
    def __init__(self, root: Path = TILE_PROBS_DIR, merged_csv: Path = MERGED, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.name2id, self.id2name = build_label_map(merged_csv)
        # gather samples: assume folder structure root/{knot_name}/tile_probs.npy
        self.samples = []  # list of (path, label)
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            fname = d.name
            # name in merged CSV uses e.g. '10_1' vs folder '10_001' - normalize by removing leading zeros in second part
            # try variants: direct name, or remove leading zeros in middle
            name_variant = fname
            # try matching by prefix (e.g., 10_001 -> 10_1)
            parts = fname.split('_')
            if len(parts) >= 2:
                a = parts[0]
                b = str(int(parts[1]))
                name_variant = f"{a}_{b}"
            label = self.name2id.get(name_variant)
            if label is None:
                # try with full folder name
                label = self.name2id.get(fname)
            if label is None:
                # unknown label; skip
                continue
            p = d / 'tile_probs.npy'
            if not p.exists():
                continue
            self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        arr = np.load(str(p))  # shape (C, H, W)
        # ensure float32
        arr = arr.astype('float32')
        # convert to tensor
        tensor = torch.from_numpy(arr)
        # reorder to (C, H, W) if needed (assuming saved as (C, H, W))
        if tensor.ndim == 3:
            pass
        elif tensor.ndim == 2:
            # single channel
            tensor = tensor.unsqueeze(0)
        else:
            # unexpected
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2])
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label


if __name__ == '__main__':
    ds = TileProbsDataset()
    print('samples:', len(ds))
    x,y = ds[0]
    print('example shape', x.shape, 'label', y)
