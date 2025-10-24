"""
Prototypical network training for tile-prob matrices.

- Uses `dataset_tile_probs.TileProbsDataset` to access samples.
- On each episode, samples N ways, K support examples (with augmentation) and Q query examples per way.
- Uses the `TileProbCNN` (without final classifier) as an embedding network producing d-dimensional embeddings.
- Uses prototypical loss (squared Euclidean distance) and standard training loop.

Run a short smoke test with: python train_prototypical.py --epochs 1 --episodes-per-epoch 20
"""
from pathlib import Path
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import default_collate
import os
from collections import defaultdict

from dataset_tile_probs import TileProbsDataset
from model_tileprob_cnn import TileProbCNN

ROOT = Path(__file__).parent


def augment_tile_probs(arr: np.ndarray):
    """Small augmentations on the tile-prob numpy array (C,H,W).
    - add small gaussian noise
    - small translations (roll)
    - multiplicative jitter and renormalize per-channel
    Returns a float32 numpy array same shape.
    """
    a = arr.copy()
    # gaussian noise
    if random.random() < 0.8:
        a = a + np.random.normal(0, 0.01, size=a.shape)
    # small roll shifts
    if random.random() < 0.5:
        shift_x = random.randint(-1, 1)
        shift_y = random.randint(-1, 1)
        a = np.roll(a, shift_x, axis=-2)
        a = np.roll(a, shift_y, axis=-1)
    # multiplicative channel jitter
    if random.random() < 0.5:
        mul = 1.0 + np.random.normal(0, 0.05, size=(a.shape[0],1,1))
        a = a * mul
    # clip and ensure float32
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return a


class EpisodicSampler:
    def __init__(self, samples_by_label: dict[str, list[str]]):
        self.samples_by_label = samples_by_label
        self.labels = list(samples_by_label.keys())

    def sample_episode(self, N, K, Q):
        ways = random.sample(self.labels, N)
        support = []  # tuples (label, path)
        query = []
        for w in ways:
            pool = list(self.samples_by_label[w])
            need = K + Q
            if len(pool) >= need:
                paths = random.sample(pool, need)
            else:
                # not enough unique examples: sample with replacement and allow duplicates
                paths = [random.choice(pool) for _ in range(need)]
            sup_p = paths[:K]
            qry_p = paths[K:K+Q]
            support.extend([(w, p) for p in sup_p])
            query.extend([(w, p) for p in qry_p])
        return ways, support, query


def compute_prototypes(embeddings, labels, N, K):
    # embeddings: (N*K, D), labels in order grouped per-way
    D = embeddings.size(1)
    prototypes = embeddings.view(N, K, D).mean(dim=1)  # (N, D)
    return prototypes


def prototypical_loss(prototypes, query_embeddings, query_targets):
    # prototypes: (N, D)
    # query_embeddings: (N*Q, D)
    # query_targets: (N*Q,) with indices 0..N-1
    # distances: (N*Q, N)
    dists = torch.cdist(query_embeddings, prototypes, p=2) ** 2
    # negative distances as logits
    log_p = torch.log_softmax(-dists, dim=1)
    loss = -log_p[range(len(query_targets)), query_targets].mean()
    preds = (-dists).argmax(dim=1)
    acc = (preds == query_targets).float().mean().item()
    return loss, acc


def load_np(path: Path):
    arr = np.load(path)
    return arr.astype(np.float32)


def make_samples_by_label(ds: TileProbsDataset):
    # Build mapping: label_name -> list of folder Path objects (each containing tile_probs.npy)
    d = {}
    for p, label in ds.samples:
        # p is a Path to tile_probs.npy; parent is the folder
        name = ds.id2name[label]
        folder_path = p.parent
        d.setdefault(name, []).append(folder_path)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--episodes-per-epoch', type=int, default=200)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Q', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke', action='store_true')
    # checkpointing / evaluation args
    parser.add_argument('--save-dir', default=str(ROOT / 'checkpoints' / 'prototypical'))
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--eval', action='store_true', help='Evaluate on originals using rotated prototypes after each epoch')
    args = parser.parse_args()

    device = torch.device(args.device)

    ds = TileProbsDataset()
    samples_by_label = make_samples_by_label(ds)

    sampler = EpisodicSampler(samples_by_label)

    # embedding network: reuse TileProbCNN but remove final classifier
    # create a small wrapper to extract embeddings (the layer before classifier)
    num_classes = len(ds.name2id)
    base = TileProbCNN(in_channels=11, num_classes=num_classes, hidden=128)
    # adjust: we only need up to classifier's first linear output (hidden)
    class EmbeddingNet(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.features = base.features
            # take the first linear layer from classifier to project to embedding dim
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128*2, 128),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.proj(x)
            return x

    net = EmbeddingNet(base).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir) if hasattr(args, 'save_dir') else ROOT / 'checkpoints' / 'prototypical'
    save_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_prototypes(net, ds, device):
        net.eval()
        # build per-class rotated pools
        pools = defaultdict(list)
        originals = defaultdict(list)
        for p, label in ds.samples:
            name = ds.id2name[label]
            folder = p.parent
            if 'rot' in folder.name:
                pools[name].append(folder)
            else:
                originals[name].append(folder)

        # fallback: if no rotated pools for a class, use any folder for that class
        for name in ds.name2id.keys():
            if len(pools[name]) == 0:
                for p,label in ds.samples:
                    if ds.id2name[label] == name:
                        pools[name].append(p.parent)

        # compute prototypes
        prototypes = {}
        with torch.no_grad():
            for name, folders in pools.items():
                embs = []
                for folder in folders:
                    fp = folder / 'tile_probs.npy'
                    if not fp.exists():
                        continue
                    arr = load_np(fp)
                    t = torch.from_numpy(arr).unsqueeze(0).to(device)
                    e = net(t)
                    embs.append(e.squeeze(0))
                if len(embs) == 0:
                    continue
                prototypes[name] = torch.stack(embs, dim=0).mean(dim=0)

            # evaluate on originals
            ys = []
            preds = []
            top2_correct = 0
            top3_correct = 0
            total = 0
            proto_names = list(prototypes.keys())
            proto_matrix = torch.stack([prototypes[n] for n in proto_names], dim=0) if len(proto_names)>0 else None
            for name, folders in originals.items():
                if len(folders) == 0:
                    continue
                for folder in folders:
                    fp = folder / 'tile_probs.npy'
                    if not fp.exists():
                        continue
                    arr = load_np(fp)
                    t = torch.from_numpy(arr).unsqueeze(0).to(device)
                    e = net(t).squeeze(0)
                    if proto_matrix is None:
                        continue
                    dists = torch.cdist(e.unsqueeze(0), proto_matrix, p=2).squeeze(0)
                    ranking = torch.argsort(dists)
                    pred_idx = ranking[0].item()
                    pred_name = proto_names[pred_idx]
                    preds.append(pred_name)
                    ys.append(name)
                    total += 1
                    # compute top-2 and top-3
                    top2 = [proto_names[i] for i in ranking[:2].tolist()]
                    top3 = [proto_names[i] for i in ranking[:3].tolist()]
                    if name in top2:
                        top2_correct += 1
                    if name in top3:
                        top3_correct += 1
            top1 = sum(1 for y,p in zip(ys,preds) if y==p) / (total if total>0 else 1)
            top2 = top2_correct / (total if total>0 else 1)
            top3 = top3_correct / (total if total>0 else 1)
        net.train()
        return top1, top2, top3, total

    # smoke test: if smoke, reduce episodes and N/K/Q
    if args.smoke:
        args.episodes_per_epoch = 10
        args.N = 5
        args.K = 1
        args.Q = 2

    for epoch in range(args.epochs):
        net.train()
        tot_loss = 0.0
        tot_acc = 0.0
        for ep in range(args.episodes_per_epoch):
            N, K, Q = args.N, args.K, args.Q
            ways, support, query = sampler.sample_episode(N, K, Q)
            # build support and query numpy arrays (keep as numpy to pad variable sizes)
            sup_arrays = []
            for (label, folder) in support:
                if isinstance(folder, str):
                    folder = Path(folder)
                path = folder / 'tile_probs.npy'
                arr = load_np(path)
                arr = augment_tile_probs(arr)
                sup_arrays.append(arr)
            qry_arrays = []
            qry_targets = []
            for (label, folder) in query:
                if isinstance(folder, str):
                    folder = Path(folder)
                path = folder / 'tile_probs.npy'
                arr = load_np(path)
                arr = augment_tile_probs(arr)
                qry_arrays.append(arr)
                qry_targets.append(ways.index(label))

            # pad arrays in each set to same HxW (max among support+query) so we can batch
            all_arrays = sup_arrays + qry_arrays
            chans = all_arrays[0].shape[0]
            max_h = max(a.shape[-2] for a in all_arrays)
            max_w = max(a.shape[-1] for a in all_arrays)

            def pad_to(a, H, W):
                c, h, w = a.shape
                pad_h = H - h
                pad_w = W - w
                pad = ((0,0),(0,pad_h),(0,pad_w))
                return np.pad(a, pad, mode='constant', constant_values=0.0)

            sup_padded = [pad_to(a, max_h, max_w) for a in sup_arrays]
            qry_padded = [pad_to(a, max_h, max_w) for a in qry_arrays]

            sup_batch = torch.stack([torch.from_numpy(a) for a in sup_padded]).to(device)
            qry_batch = torch.stack([torch.from_numpy(a) for a in qry_padded]).to(device)
            # forward
            sup_emb = net(sup_batch)
            qry_emb = net(qry_batch)
            prototypes = compute_prototypes(sup_emb, None, N, K)
            loss, acc = prototypical_loss(prototypes, qry_emb, torch.tensor(qry_targets, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            tot_acc += acc
        print(f"Epoch {epoch+1} loss={tot_loss/args.episodes_per_epoch:.4f} acc={tot_acc/args.episodes_per_epoch:.4f}")
        # save checkpoint
        if (epoch+1) % args.save_every == 0:
            ckpt = {
                'epoch': epoch+1,
                'model_state': net.state_dict(),
                'opt_state': opt.state_dict(),
                'args': vars(args)
            }
            fname = save_dir / f'proto_epoch_{epoch+1}.pt'
            torch.save(ckpt, str(fname))
            print('Saved checkpoint', fname)
        # optional evaluation on originals
        if args.eval:
            top1, top2, top3, n = evaluate_prototypes(net, ds, device)
            print(f"Eval on originals: n={n} top1={top1:.4f} top2={top2:.4f} top3={top3:.4f}")


if __name__ == '__main__':
    main()
