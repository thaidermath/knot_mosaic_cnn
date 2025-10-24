"""Compute per-checkpoint top-1/top-2/top-3 for checkpoints in checkpoints/proto_run and write CSV compatible with parser.
Usage: python scripts/compute_proto_run_metrics.py --ckpt-dir checkpoints/proto_run --out plots/proto_run_metrics.csv
"""
from pathlib import Path
import argparse
import torch
import numpy as np
from collections import defaultdict

# import local modules
import sys
sys.path.insert(0, '.')
from dataset_tile_probs import TileProbsDataset
from model_tileprob_cnn import TileProbCNN


def build_prototypes(net, ds, device='cpu'):
    pools = defaultdict(list)
    for p,label in ds.samples:
        name = ds.id2name[label]
        folder = p.parent
        if 'rot' in folder.name:
            pools[name].append(folder)
    # fallback: use any folder
    for name in ds.name2id.keys():
        if len(pools[name])==0:
            for p,label in ds.samples:
                if ds.id2name[label]==name:
                    pools[name].append(p.parent)
    prototypes = {}
    net.eval()
    with torch.no_grad():
        for name,folders in pools.items():
            embs=[]
            for folder in folders:
                fp=folder/'tile_probs.npy'
                if not fp.exists():
                    continue
                arr=np.load(fp).astype('float32')
                t=torch.from_numpy(arr).unsqueeze(0)
                if t.ndim==3:
                    t=t.unsqueeze(0)
                e=net(t.to(device))
                embs.append(e.squeeze(0).cpu())
            if len(embs):
                prototypes[name]=torch.stack(embs,dim=0).mean(dim=0).cpu()
    return prototypes


def evaluate_checkpoint(net, prototypes, ds, device='cpu'):
    proto_names=list(prototypes.keys())
    proto_matrix=torch.stack([prototypes[n] for n in proto_names],dim=0)
    ys=[]
    preds=[]
    for name in ds.name2id.keys():
        # find folders for this name (originals)
        folders=[p.parent for p,l in ds.samples if ds.id2name[l]==name]
        for folder in folders:
            fp=folder/'tile_probs.npy'
            if not fp.exists():
                continue
            arr=np.load(fp).astype('float32')
            t=torch.from_numpy(arr).unsqueeze(0)
            if t.ndim==3:
                t=t.unsqueeze(0)
            with torch.no_grad():
                e=net(t.to(device)).squeeze(0).cpu()
            dists = torch.cdist(e.unsqueeze(0), proto_matrix, p=2).squeeze(0)
            # get top-2/top-3 indices
            k2 = min(2, dists.numel())
            k3 = min(3, dists.numel())
            top2_idx = torch.topk(-dists, k=k2).indices.tolist()
            top3_idx = torch.topk(-dists, k=k3).indices.tolist()
            pred_idx = dists.argmin().item()
            ys.append(name)
            preds.append((proto_names[pred_idx], [proto_names[i] for i in top2_idx], [proto_names[i] for i in top3_idx]))
    # compute top1/top2/top3
    total=len(ys)
    top1=sum(1 for y,(p,top2,top3) in zip(ys,preds) if y==p)
    top2=sum(1 for y,(p,top2,top3) in zip(ys,preds) if y in top2)
    top3=sum(1 for y,(p,top2,top3) in zip(ys,preds) if y in top3)
    return top1/total if total>0 else 0.0, top2/total if total>0 else 0.0, top3/total if total>0 else 0.0, total


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    args=parser.parse_args()

    ds=TileProbsDataset()
    num_classes=len(ds.name2id)
    base=TileProbCNN(in_channels=11,num_classes=num_classes,hidden=128)
    # embedding net
    class EmbeddingNet(torch.nn.Module):
        def __init__(self,base):
            super().__init__()
            self.features=base.features
            self.proj=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(128*2,128), torch.nn.ReLU())
        def forward(self,x):
            x=self.features(x)
            x=self.proj(x)
            return x
    net=EmbeddingNet(base).to(args.device)

    ckpt_dir=Path(args.ckpt_dir)
    rows=[]
    for ckpt in sorted(ckpt_dir.glob('proto_epoch_*.pt')):
        print('Processing',ckpt)
        d=torch.load(ckpt,map_location=args.device)
        net.load_state_dict(d['model_state'])
        prototypes=build_prototypes(net,ds,device=args.device)
        top1,top2,top3,n=evaluate_checkpoint(net,prototypes,ds)
        rows.append((ckpt.stem,top1,top2,top3,n))
    # write CSV with columns: epoch,top1,top5,n
    outp=Path(args.out)
    outp.parent.mkdir(parents=True,exist_ok=True)
    with outp.open('w',encoding='utf-8') as f:
        f.write('epoch,top1,top2,top3,n\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]}\n")
    print('Wrote',outp)

if __name__=='__main__':
    main()
