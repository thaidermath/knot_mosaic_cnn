"""
Parse prototypical training logs in `logs/` and produce CSV + comparison plots for top1/top2/top3.

Usage: python scripts/parse_proto_logs_and_plot.py
It will scan `logs/` for files named `proto_*.log` and output CSVs and PNGs to `plots/`.
"""
from pathlib import Path
import re
import csv
import codecs
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / 'logs'
PLOT_DIR = ROOT / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

pattern_epoch = re.compile(r"Epoch\s*(\d+)\s+loss=([0-9.]+)\s+acc=([0-9.]+)")
# support new format: top1/top2/top3; fallback to old top5 if present
pattern_eval_new = re.compile(r"Eval on originals: n=(\d+) top1=([0-9.]+) top2=([0-9.]+) top3=([0-9.]+)")
pattern_eval_old = re.compile(r"Eval on originals: n=(\d+) top1=([0-9.]+) top5=([0-9.]+)")

results = {}
meta = {}
for fp in sorted(LOG_DIR.glob('proto_*.log')):
    name = fp.stem  # proto_10way
    epochs = []
    # read as binary and decode robustly (some logs redirected by PowerShell are UTF-16 LE)
    raw_bytes = fp.read_bytes()
    # detect BOM or null bytes to guess utf-16-le
    if raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(codecs.BOM_UTF16_BE) or b'\x00' in raw_bytes[:2000]:
        try:
            raw = raw_bytes.decode('utf-16')
        except Exception:
            raw = raw_bytes.decode('utf-16-le', errors='ignore')
    else:
        try:
            raw = raw_bytes.decode('utf-8')
        except Exception:
            raw = raw_bytes.decode('latin-1', errors='ignore')
    # remove common code-fence wrappers (```...```) that some logs contain
    if raw.startswith('```') and raw.rstrip().endswith('```'):
        # strip the first line if it's a fence marker
        parts = raw.splitlines()
        # drop leading/trailing fence lines
        if parts[0].strip().startswith('```'):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith('```'):
            parts = parts[:-1]
        lines = [l + '\n' for l in parts]
    else:
        lines = raw.splitlines(True)
    epoch_info = {}
    for i,line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        m = pattern_epoch.search(line)
        if m:
            ep = int(m.group(1))
            loss = float(m.group(2))
            acc = float(m.group(3))
            epoch_info.setdefault(ep, {})['train_loss'] = loss
            epoch_info[ep]['train_acc'] = acc
            # check following lines for Eval
            # scan next up to 5 lines
            for j in range(i+1, min(i+6, len(lines))):
                ln = lines[j]
                me = pattern_eval_new.search(ln)
                if me:
                    n = int(me.group(1))
                    top1 = float(me.group(2))
                    top2 = float(me.group(3))
                    top3 = float(me.group(4))
                    epoch_info[ep]['eval_n'] = n
                    epoch_info[ep]['eval_top1'] = top1
                    epoch_info[ep]['eval_top2'] = top2
                    epoch_info[ep]['eval_top3'] = top3
                    break
                mo = pattern_eval_old.search(ln)
                if mo:
                    n = int(mo.group(1))
                    top1 = float(mo.group(2))
                    top5 = float(mo.group(3))
                    # best-effort mapping: treat old top5 as >= top3; store as top3 if top5==1.0 else leave top2/top3 empty
                    epoch_info[ep]['eval_n'] = n
                    epoch_info[ep]['eval_top1'] = top1
                    # when only old top5 is available, set top3 to top5 for visibility (conservative)
                    epoch_info[ep]['eval_top3'] = top5
                    break
    # write CSV per log
    out_csv = PLOT_DIR / f'{name}_metrics.csv'
    with out_csv.open('w', newline='') as csvf:
        w = csv.writer(csvf)
        header = ['epoch','train_loss','train_acc','eval_n','eval_top1','eval_top2','eval_top3']
        w.writerow(header)
        for ep in sorted(epoch_info.keys()):
            row = [ep,
                   epoch_info[ep].get('train_loss',''),
                   epoch_info[ep].get('train_acc',''),
                   epoch_info[ep].get('eval_n',''),
                   epoch_info[ep].get('eval_top1',''),
                   epoch_info[ep].get('eval_top2',''),
                   epoch_info[ep].get('eval_top3','')]
            w.writerow(row)
    # infer way/shot metadata from filename pattern
    parts = name.split('_')
    way = next((p for p in parts if p.endswith('way')), None)
    shot = next((p for p in parts if p.endswith('shot')), None)
    # skip deprecated settings (5-shot runs) to keep plots focused on current experiments
    if shot == '5shot':
        continue

    meta[name] = {'way': way, 'shot': shot}
    results[name] = epoch_info

# combined plots with consistent styling
if results:
    # style helpers
    way_styles = {
        '20way': dict(color='#1f77b4', marker='s'),
        '10way': dict(color='#ff7f0e', marker='o'),
        '5way' : dict(color='#2ca02c', marker='D'),
    }
    shot_styles = {
        '5shot': dict(linestyle='-'),
        '3shot': dict(linestyle='--'),
        '2shot': dict(linestyle='-.'),
    }
    way_order = ['20way', '10way', '5way']
    shot_order = ['3shot', '2shot']

    def format_way(w):
        return w.replace('way', '-way') if w else 'unknown'

    def format_shot(s):
        return s.replace('shot', '-shot') if s else ''

    # determine plotting order for combined figure
    preferred = []
    for shot in shot_order:
        for way in way_order:
            key = f'proto_{way}_{shot}'
            if key in results:
                preferred.append(key)
    # include any remaining logs
    preferred += [n for n in results.keys() if n not in preferred]

    # Top-1
    fig,ax = plt.subplots(figsize=(8,5))
    # iterate preferred first, then any remaining
    names = preferred
    for name in names:
        info = results[name]
        eps = sorted(info.keys())
        top1 = [info[e].get('eval_top1', 0) for e in eps]
        info_meta = meta.get(name, {})
        way = info_meta.get('way')
        shot = info_meta.get('shot')
        base_style = way_styles.get(way, {})
        shot_style = shot_styles.get(shot, {})
        style = {
            'color': base_style.get('color', '#333333'),
            'marker': base_style.get('marker', 'o'),
            'linestyle': shot_style.get('linestyle', '-')
        }
        label = f"{format_way(way)} {format_shot(shot)}".strip()
        ax.plot(eps, top1, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('eval top1')
    ax.set_title('Eval Top-1 comparison')
    ax.grid(True)
    leg = ax.legend(frameon=True)
    # make legend box consistent
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor('black')
    # axis box
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'compare_top1.png')

    # Top-2 comparison
    fig,ax = plt.subplots(figsize=(8,5))
    for name in names:
        info = results[name]
        eps = sorted(info.keys())
        top2 = [info[e].get('eval_top2', 0) for e in eps]
        info_meta = meta.get(name, {})
        way = info_meta.get('way')
        shot = info_meta.get('shot')
        base_style = way_styles.get(way, {})
        shot_style = shot_styles.get(shot, {})
        style = {
            'color': base_style.get('color', '#333333'),
            'marker': base_style.get('marker', 'o'),
            'linestyle': shot_style.get('linestyle', '-')
        }
        label = f"{format_way(way)} {format_shot(shot)}".strip()
        ax.plot(eps, top2, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('eval top2')
    ax.set_title('Eval Top-2 comparison')
    ax.grid(True)
    leg = ax.legend(frameon=True)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor('black')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'compare_top2.png')

    # Top-3 (preferred secondary metric for K=3)
    fig,ax = plt.subplots(figsize=(8,5))
    for name in names:
        info = results[name]
        eps = sorted(info.keys())
        top3 = [info[e].get('eval_top3', 0) for e in eps]
        info_meta = meta.get(name, {})
        way = info_meta.get('way')
        shot = info_meta.get('shot')
        base_style = way_styles.get(way, {})
        shot_style = shot_styles.get(shot, {})
        style = {
            'color': base_style.get('color', '#333333'),
            'marker': base_style.get('marker', 'o'),
            'linestyle': shot_style.get('linestyle', '-')
        }
        label = f"{format_way(way)} {format_shot(shot)}".strip()
        ax.plot(eps, top3, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('eval top3')
    ax.set_title('Eval Top-3 comparison')
    ax.grid(True)
    leg = ax.legend(frameon=True)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor('black')
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'compare_top3.png')

    # per-shot plots
    groups = defaultdict(list)
    for name, info_meta in meta.items():
        shot = info_meta.get('shot', 'unspecified')
        groups[shot].append(name)

    for shot, shot_names in groups.items():
        shot_label = format_shot(shot) or 'unspecified'
        shot_names = sorted(shot_names, key=lambda n: way_order.index(meta[n]['way']) if meta[n]['way'] in way_order else len(way_order))

        # top-1 per shot
        fig, ax = plt.subplots(figsize=(8,5))
        for name in shot_names:
            info = results[name]
            eps = sorted(info.keys())
            top1 = [info[e].get('eval_top1', 0) for e in eps]
            way = meta[name].get('way')
            base_style = way_styles.get(way, {})
            style = {
                'color': base_style.get('color', '#333333'),
                'marker': base_style.get('marker', 'o'),
                'linestyle': '-'
            }
            label = format_way(way)
            ax.plot(eps, top1, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                    linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
        ax.set_xlabel('epoch')
        ax.set_ylabel('eval top1')
        ax.set_title(f'Eval Top-1 comparison ({shot_label})')
        ax.grid(True)
        leg = ax.legend(frameon=True)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor('black')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.tight_layout()
        out_name = f'compare_top1_{shot}.png'
        fig.savefig(PLOT_DIR / out_name)

        # top-2 per shot
        fig, ax = plt.subplots(figsize=(8,5))
        for name in shot_names:
            info = results[name]
            eps = sorted(info.keys())
            top2 = [info[e].get('eval_top2', 0) for e in eps]
            way = meta[name].get('way')
            base_style = way_styles.get(way, {})
            style = {
                'color': base_style.get('color', '#333333'),
                'marker': base_style.get('marker', 'o'),
                'linestyle': '-'
            }
            label = format_way(way)
            ax.plot(eps, top2, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                    linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
        ax.set_xlabel('epoch')
        ax.set_ylabel('eval top2')
        ax.set_title(f'Eval Top-2 comparison ({shot_label})')
        ax.grid(True)
        leg = ax.legend(frameon=True)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor('black')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.tight_layout()
        out_name = f'compare_top2_{shot}.png'
        fig.savefig(PLOT_DIR / out_name)

        # top-3 per shot
        fig, ax = plt.subplots(figsize=(8,5))
        for name in shot_names:
            info = results[name]
            eps = sorted(info.keys())
            top3 = [info[e].get('eval_top3', 0) for e in eps]
            way = meta[name].get('way')
            base_style = way_styles.get(way, {})
            style = {
                'color': base_style.get('color', '#333333'),
                'marker': base_style.get('marker', 'o'),
                'linestyle': '-'
            }
            label = format_way(way)
            ax.plot(eps, top3, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                    linewidth=2.0, markersize=6, markeredgecolor='k', markeredgewidth=0.8, label=label)
        ax.set_xlabel('epoch')
        ax.set_ylabel('eval top3')
        ax.set_title(f'Eval Top-3 comparison ({shot_label})')
        ax.grid(True)
        leg = ax.legend(frameon=True)
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_edgecolor('black')
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        fig.tight_layout()
        out_name = f'compare_top3_{shot}.png'
        fig.savefig(PLOT_DIR / out_name)

    # remove stale figures from deprecated settings
    for stale_name in ['compare_top1_5shot.png', 'compare_top2_5shot.png', 'compare_top3_5shot.png']:
        stale_path = PLOT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    print('Wrote CSVs and plots to', PLOT_DIR)
else:
    print('No proto logs found in', LOG_DIR)
