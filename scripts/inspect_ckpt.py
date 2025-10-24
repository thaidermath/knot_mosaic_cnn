import torch,sys
p='checkpoints/proto_run/proto_epoch_10.pt'
try:
    d=torch.load(p,map_location='cpu')
    print('keys=',list(d.keys()))
    for k in ['epoch','train_loss','train_acc','eval_top1','eval_top2','eval_top3','model_state']:
        print(k, 'in d?', k in d)
    if 'stats' in d:
        print('stats keys:', d['stats'].keys())
except Exception as e:
    print('error',e)
