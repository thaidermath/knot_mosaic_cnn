from pathlib import Path
import re

LOG_DIR = Path(__file__).parent.parent / 'logs'
files = ['proto_10way.log','proto_5way.log']
pattern_epoch = re.compile(r"Epoch\s*(\d+)\s+loss=([0-9.]+)\s+acc=([0-9.]+)")
pattern_eval_old = re.compile(r"Eval on originals: n=(\d+) top1=([0-9.]+) top5=([0-9.]+)")
pattern_eval_new = re.compile(r"Eval on originals: n=(\d+) top1=([0-9.]+) top2=([0-9.]+) top3=([0-9.]+)")

for fname in files:
    fp = LOG_DIR / fname
    print('\n---', fp)
    raw = fp.read_text(encoding='utf-8', errors='ignore')
    print('RAW repr start:', repr(raw[:400]))
    # handle fences as parser does
    if raw.startswith('```') and raw.rstrip().endswith('```'):
        parts = raw.splitlines()
        if parts[0].strip().startswith('```'):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith('```'):
            parts = parts[:-1]
        lines = [l + '\n' for l in parts]
    else:
        lines = raw.splitlines(True)
    print('Total lines:', len(lines))
    matches = 0
    evals = 0
    for i,line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith('```'):
            continue
        me = pattern_epoch.search(s)
        if me:
            matches += 1
            print('EPOCH MATCH', i, s)
        me2 = pattern_eval_new.search(s)
        if me2:
            evals += 1
            print('EVAL MATCH (new)', i, s)
        else:
            mo = pattern_eval_old.search(s)
            if mo:
                evals += 1
                print('EVAL MATCH (old)', i, s)
    print('Found epoch matches:', matches, 'eval matches:', evals)
print('\nDone')
