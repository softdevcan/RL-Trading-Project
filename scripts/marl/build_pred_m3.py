"""Faz 10 M3: PC060-K12 pozitif kontrol tahmin bloklari (M2 tasarim §9.1; M3 on kaydi) -> ortamin okudugu .npz.

PC060-K12: `prediction.pit_table.positive_control_table` (AUC 0,60), taban tablo `pit_v1_seed42` (maske ve olcek),
etiket 1[Of_{t+2} > Of_{t+1}] (`scripts/marl/pc_alignment_m2.py::k12_labels`, ayni tanim). BILEREK GELECEKTEN:
yalniz boru hatti kontrolu; tez sonucu degil. `meta_ok` olmayan hucre NaN -> ortamda pred_ok = 0 (maske).
Cikti panel tarih/sembol sirasiyla hizali [T, N]; `marl.benchmarl_task.load_pred` hizayi denetler.
`--shuffle`: KARISTIRILMIS kontrol (M3 on kaydi v0.2, hakem-rl M3a #2): ayni blok, SATIRLAR (gunler) tohum = uretec
tohumuyla rastgele permute edilir (uc alan + maske birlikte; kesit yapisi ve marjinal dagilim ayni, zaman
hizasi yok -> sinyal yok). Dosya `pred_pc060_k12_g{tohum}_shuf.npz`.

    python scripts/marl/build_pred_m3.py --seeds 20261115,1,2,3,4 --out-dir data/marl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
PANEL_SHA = 'a5dcf9a607e0447df490f0a47c5187555f428faddc62b47d5eef81819e5e147b'


def _pc():
    spec = importlib.util.spec_from_file_location('pc_alignment_m2', os.path.join(ROOT, 'scripts', 'marl',
                                                                                   'pc_alignment_m2.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def provenance() -> dict:
    import subprocess
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def to_arrays(table: pd.DataFrame, dates, symbols) -> dict:
    t = table.copy()
    t['date'] = pd.to_datetime(t['date']).dt.tz_localize(None).dt.normalize().dt.strftime('%Y-%m-%d')
    out = {}
    ok = t.pivot_table(index='date', columns='symbol', values='meta_ok', aggfunc='first')
    ok = ok.reindex(index=list(dates), columns=list(symbols)).fillna(False).to_numpy(dtype=bool)
    for k in ('p_up_cal', 'pred_logret', 'model_disagreement'):
        v = t.pivot_table(index='date', columns='symbol', values=k, aggfunc='first')
        v = v.reindex(index=list(dates), columns=list(symbols)).to_numpy(dtype=np.float64)
        out[k] = np.where(ok, v, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='20261115,1,2,3,4')
    ap.add_argument('--shuffle', action='store_true', help='karistirilmis kontrol bloklari (gunler permute)')
    ap.add_argument('--out-dir', default=os.path.join('data', 'marl'))
    ap.add_argument('--json', default=None, help="varsayilan measurements/phase10-m3/pred_blocks[_shuf].json")
    a = ap.parse_args()
    from app.auth import workspace as ws
    from marl import accounting as A
    from marl import panel as PN
    from prediction.pit_table import load_table, positive_control_table

    pc = _pc()
    arrays = PN.load_npz(PANEL)
    if arrays['sha256'] != PANEL_SHA:
        raise SystemExit(f"panel SHA {arrays['sha256']} != {PANEL_SHA}")
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(arrays['dates']))
    side = json.load(open(os.path.join(ws.results_dir(), 'predictions_pit', 'pc_auc060.json'), encoding='utf-8'))
    base, binfo = load_table(side['base_table']['run_id'])
    if binfo['sha256'] != side['base_table']['sha256']:
        raise SystemExit('taban tablo SHA uyusmuyor')
    base = base.copy()
    base['date'] = pd.to_datetime(base['date']).dt.tz_localize(None).dt.normalize()
    labels = pc.k12_labels(arrays, px)
    from evaluation.calibration import auc as _auc
    _lab = labels.copy()
    _lab['date'] = _lab['date'].dt.strftime('%Y-%m-%d')
    Y = (_lab.pivot_table(index='date', columns='symbol', values='y_up', aggfunc='first')
         .reindex(index=list(arrays['dates']), columns=list(arrays['symbols'])).to_numpy(dtype=np.float64))
    rec = {'panel_sha256': arrays['sha256'], 'base_table': side['base_table'], 'target_auc': side['target_auc'],
           'label': '1[Of_{t+2} > Of_{t+1}] (pc_alignment_m2.k12_labels)', **provenance(), 'blocks': []}
    os.makedirs(a.out_dir, exist_ok=True)
    for sd in [int(s) for s in a.seeds.split(',')]:
        tab, info = positive_control_table(base, labels, side['target_auc'], seed=sd)
        blk = to_arrays(tab, arrays['dates'], arrays['symbols'])
        suffix = ''
        if a.shuffle:
            perm = np.random.default_rng(sd).permutation(len(arrays['dates']))
            blk = {k: v[perm] for k, v in blk.items()}
            suffix = '_shuf'
        path = os.path.join(a.out_dir, f'pred_pc060_k12_g{sd}{suffix}.npz')
        np.savez_compressed(path, dates=arrays['dates'], symbols=arrays['symbols'], **blk)
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        cov = float(np.isfinite(blk['p_up_cal']).mean())
        m = np.isfinite(blk['p_up_cal']) & np.isfinite(Y)
        auc_block = float(_auc(blk['p_up_cal'][m], Y[m]))   # dosyadaki blogun K12 etiketiyle AUC'si
        rec['blocks'].append({'seed': sd, 'shuffled_rows': bool(a.shuffle), 'path': path.replace(os.sep, '/'), 'sha256': sha,
                              'generator_auc_meta_ok': info['achieved_auc_meta_ok'], 'auc_block_vs_label': auc_block,
                              'coverage': cov})
        print(f"tohum {sd}: {path} uretec AUC {info['achieved_auc_meta_ok']:.4f} blok AUC {auc_block:.4f} "
              f"kapsam {cov:.3f} sha {sha[:8]}")
    a.json = a.json or os.path.join('docs', 'development', 'measurements', 'phase10-m3',
                                    'pred_blocks_shuf.json' if a.shuffle else 'pred_blocks.json')
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
