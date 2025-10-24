import sys
sys.path.insert(0, '.')
import runpy
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--ckpt-dir', required=True)
parser.add_argument('--out-prefix', default='plots/conf')
parser.add_argument('--device', default='cpu')
args=parser.parse_args()
runpy.run_path('scripts/eval_checkpoints_and_save_confusion.py', run_name='__main__')
