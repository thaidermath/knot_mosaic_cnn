"""
eval_baseline_nn.py

Compute leave-one-out nearest-neighbor accuracy using simple pooled features
extracted from `tile_probs_from_matrices/*/tile_probs.npy`.

Features: per-channel mean over HxW -> 11-dimensional vector.

Usage:
    python eval_baseline_nn.py

"""
from pathlib import Path
import numpy as np
from dataset_tile_probs import TileProbsDataset
from collections import defaultdict


def build_features(dataset):
    X = []
    y = []
    files = []
    for p, label in dataset.samples:
        arr = np.load(str(p)).astype('float32')  # (C,H,W)
        # global average pool per channel
        feat = arr.mean(axis=(1,2))
        X.append(feat)
        y.append(label)
        files.append(str(p))
    X = np.stack(X)
    y = np.array(y)
    return X, y, files


def loo_nn(X, y, metric='euclidean'):
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = X[i:i+1]
        others = np.concatenate([X[:i], X[i+1:]], axis=0)
        labels = np.concatenate([y[:i], y[i+1:]], axis=0)
        if metric == 'euclidean':
            d = np.linalg.norm(others - xi, axis=1)
            idx = d.argmin()
        else:
            # cosine similarity
            norms = np.linalg.norm(others, axis=1) * np.linalg.norm(xi)
            sims = (others @ xi.T).flatten() / (norms + 1e-12)
            idx = sims.argmax()
        pred = labels[idx]
        if pred == y[i]:
            correct += 1
    return correct / n


if __name__ == '__main__':
    ds = TileProbsDataset()
    X,y,files = build_features(ds)
    print('samples', X.shape)
    eu = loo_nn(X,y,'euclidean')
    cos = loo_nn(X,y,'cosine')
    print('LOO NN euclidean:', eu)
    print('LOO NN cosine:', cos)
