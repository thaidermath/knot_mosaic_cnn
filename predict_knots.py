"""
predict_knots.py

Load a trained checkpoint and run inference on a `.npy` file or a folder of `.npy` files.

Usage:
    python predict_knots.py --model checkpoints/tileprob_cnn.pt --input tile_probs_from_matrices/9_9/tile_probs.npy

"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from model_tileprob_cnn import TileProbCNN
from dataset_tile_probs import TileProbsDataset
import numpy as np


def load_checkpoint(path):
    d = torch.load(path, map_location='cpu')
    name2id = d.get('name2id')
    num_classes = len(name2id)
    model = TileProbCNN(in_channels=11, num_classes=num_classes)
    model.load_state_dict(d['model_state_dict'])
    return model, name2id


def predict_single(model, path):
    arr = np.load(str(path)).astype('float32')
    import torch
    x = torch.from_numpy(arr).unsqueeze(0)
    if x.ndim == 4:
        pass
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    logits = model(x)
    probs = F.softmax(logits, dim=1).numpy()[0]
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    model, name2id = load_checkpoint(args.model)
    id2name = {v:k for k,v in name2id.items()}
    p = Path(args.input)
    if p.is_file():
        probs = predict_single(model, p)
        topk = probs.argsort()[::-1][:5]
        for k in topk:
            print(id2name[k], probs[k])
    else:
        for f in sorted(p.glob('**/*.npy')):
            probs = predict_single(model, f)
            topk = probs.argsort()[::-1][:3]
            print(f.name, [ (id2name[k], float(probs[k])) for k in topk ])
