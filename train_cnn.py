"""
train_cnn.py

Train the TileProbCNN on `tile_probs_from_matrices` using the dataset class.
Saves best model to `checkpoints/tileprob_cnn.pt`.

Usage:
    python train_cnn.py --epochs 20 --batch-size 16

"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import optim
from dataset_tile_probs import TileProbsDataset
from model_tileprob_cnn import TileProbCNN

ROOT = Path('.')
CKPT_DIR = ROOT / 'checkpoints'
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def collate_fn(batch):
    # batch: list of (tensor, label) with variable HxW
    # We'll pad to the max H,W in the batch
    import torch
    max_h = max([t.shape[1] for t,_ in batch])
    max_w = max([t.shape[2] for t,_ in batch])
    padded = []
    labels = []
    for t,l in batch:
        c,h,w = t.shape
        if h != max_h or w != max_w:
            pad_h = max_h - h
            pad_w = max_w - w
            # pad (left,right,top,bottom) = (0,pad_w,0,pad_h)
            t = F.pad(t, (0,pad_w,0,pad_h), 'constant', 0.0)
        padded.append(t)
        labels.append(l)
    xs = torch.stack(padded)
    ys = torch.tensor(labels, dtype=torch.long)
    return xs, ys


def train(args):
    ds = TileProbsDataset()
    # Create train/val split where ORIGINAL (non-rotated) samples are used for validation
    # and ROTATED samples (folder name contains 'rot') are used for training.
    from collections import defaultdict
    samples = ds.samples  # list of (Path, label)
    label_to_indices = defaultdict(list)
    for idx, (p, label) in enumerate(samples):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    for label, idxs in label_to_indices.items():
        # prefer a non-rotated sample for validation
        orig_idx = None
        for i in idxs:
            folder_name = samples[i][0].parent.name.lower()
            if 'rot' not in folder_name:
                orig_idx = i
                break
        if orig_idx is None:
            # fallback: if no non-rotated, pick the first as validation
            orig_idx = idxs[0]
        val_indices.append(orig_idx)
        for i in idxs:
            if i != orig_idx:
                train_indices.append(i)

    # Build subsets
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    name2id = ds.name2id
    num_classes = len(name2id)
    model = TileProbCNN(in_channels=11, num_classes=num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    print(f'Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}, Classes: {len(ds.name2id)}')
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct = 0, 0
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        train_acc = correct / total if total>0 else 0.0

        # validation
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        val_acc = correct / total if total>0 else 0.0
        print(f'Epoch {epoch} train_acc={train_acc:.3f} val_acc={val_acc:.3f}')
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'name2id': ds.name2id}, CKPT_DIR / 'tileprob_cnn.pt')
    print('Done. Best val', best_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
