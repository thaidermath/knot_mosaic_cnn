"""
Evaluate saved prototypical checkpoints on the originals and produce per-checkpoint confusion matrices and per-class CSVs.

Usage: python scripts/eval_checkpoints_and_save_confusion.py --ckpt-dir checkpoints/proto_run --out plots/conf_proto_run
"""
from pathlib import Path
import torch
import numpy as np
import argparse
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

from dataset_tile_probs import TileProbsDataset
from model_tileprob_cnn import TileProbCNN

ROOT = Path(__file__).parent.parent


def load_checkpoint(fp, device='cpu'):
    d = torch.load(fp, map_location=device)
    return d


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
                e=net(t)
                embs.append(e.squeeze(0))
            if len(embs):
                prototypes[name]=torch.stack(embs,dim=0).mean(dim=0).cpu()
    return prototypes


def evaluate(net, prototypes, ds):
    proto_names=list(prototypes.keys())
    proto_matrix=torch.stack([prototypes[n] for n in proto_names],dim=0)
    ys=[]
    preds=[]
    for name,folders in [(n,[p.parent for p,l in ds.samples if ds.id2name[l]==n]) for n in ds.name2id.keys()]:
        for folder in folders:
            fp=Path(folder)/'tile_probs.npy'
            if not fp.exists():
                continue
            arr=np.load(fp).astype('float32')
            t=torch.from_numpy(arr).unsqueeze(0)
            with torch.no_grad():
                e=net(t).squeeze(0).cpu()
            dists=torch.cdist(e.unsqueeze(0),proto_matrix,p=2).squeeze(0)
            pred_idx=dists.argmin().item()
            ys.append(name)
            preds.append(proto_names[pred_idx])
    return ys,preds,proto_names


def save_confusion(ys,preds,out_prefix):
    labels=sorted(list(set(ys)))
    idx={l:i for i,l in enumerate(labels)}
    N=len(labels)
    mat=[[0]*N for _ in range(N)]
    for y,p in zip(ys,preds):
        if y not in idx or p not in idx:
            continue
        mat[idx[y]][idx[p]]+=1
    # write csv per-class
    out_csv=Path(out_prefix+'.perclass.csv')
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['class','pred_class','count'])
        for i,y in enumerate(labels):
            for j,p in enumerate(labels):
                if mat[i][j]>0:
                    w.writerow([y,p,mat[i][j]])
    # save heatmap
    import numpy as np
    arr=np.array(mat)
    plt.figure(figsize=(8,6))
    plt.imshow(arr, interpolation='nearest', cmap='viridis')
    plt.colorbar()
    plt.title('Confusion matrix (rows=true, cols=pred)')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.tight_layout()
    plt.savefig(out_prefix+'.png')
    plt.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--out-prefix', default='plots/conf')
    parser.add_argument('--device', default='cpu')
    args=parser.parse_args()

    ds=TileProbsDataset()
    # build embedding net architecture similar to train_prototypical
    num_classes=len(ds.name2id)
    base=TileProbCNN(in_channels=11,num_classes=num_classes,hidden=128)
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
    for ckpt in sorted(ckpt_dir.glob('proto_epoch_*.pt')):
        outp=Path(args.out_prefix+'_'+ckpt.stem)
        print('Processing',ckpt,'->',outp)
        d=torch.load(ckpt,map_location=args.device)
        net.load_state_dict(d['model_state'])
        prototypes=build_prototypes(net,ds,device=args.device)
        ys,preds,proto_names=evaluate(net,prototypes,ds)
        save_confusion(ys,preds,str(outp))

if __name__=='__main__':
    main()
