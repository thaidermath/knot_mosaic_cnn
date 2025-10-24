"""
eval_retrieval_nn.py

Evaluate a retrieval baseline: pad all `tile_probs.npy` to a common size, flatten, normalize, and compute
leave-one-out nearest-neighbor accuracy using cosine similarity.

This doesn't train a classifier and is appropriate when there's only one example per class.

Usage:
    python eval_retrieval_nn.py

"""
from dataset_tile_probs import TileProbsDataset
import numpy as np
from tqdm import tqdm


def pad_to_max(arrs, pad_value=0.0):
    # arrs: list of np arrays with shape (C,H,W)
    max_h = max(a.shape[1] for a in arrs)
    max_w = max(a.shape[2] for a in arrs)
    C = arrs[0].shape[0]
    out = np.full((len(arrs), C, max_h, max_w), pad_value, dtype='float32')
    for i,a in enumerate(arrs):
        c,h,w = a.shape
        out[i,:c,:h,:w] = a
    return out


def loo_nn_cosine(X, y):
    # X: (N,D)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-12)
    n = X.shape[0]
    correct = 0
    for i in range(n):
        xi = Xn[i:i+1]
        others = np.concatenate([Xn[:i], Xn[i+1:]], axis=0)
        labels = np.concatenate([y[:i], y[i+1:]], axis=0)
        sims = (others @ xi.T).flatten()
        pred = labels[sims.argmax()]
        if pred == y[i]:
            correct += 1
    return correct / n


def main():
    ds = TileProbsDataset()
    print('samples:', len(ds))
    arrs = []
    labels = []
    for p,l in ds.samples:
        a = np.load(str(p)).astype('float32')
        arrs.append(a)
        labels.append(l)
    padded = pad_to_max(arrs)
    N,C,H,W = padded.shape
    print('padded shape', padded.shape)
    # flatten
    X = padded.reshape(N, -1)
    y = np.array(labels)
    # run LOO cosine
    acc = loo_nn_cosine(X, y)
    print('LOO NN cosine (flatten padded):', acc)
    # also try per-channel mean
    feats = padded.mean(axis=(2,3))
    acc2 = loo_nn_cosine(feats, y)
    print('LOO NN cosine (per-channel mean):', acc2)

if __name__ == '__main__':
    main()
