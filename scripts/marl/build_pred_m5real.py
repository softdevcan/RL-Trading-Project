"""Faz 10 M5 gercek sinyal kollari: `pit_v2` P tablosu -> ortamin okudugu .npz (M5 tasarimi §2.3, §5). Ana venv.

Gercek sinyal: `results/predictions_pit/pit_v2_seed42.parquet` (P5; hedef open_to_open = K12 ufku Oₜ₊₁ -> Oₜ₊₂; SHA
on kayitta sabit). Bilgi kapisindan gecmedi (0/30; M5 §2.1): bu kollar SAGLAMLIK sorusunu sinar, mekanizma degil.
Bicim `build_pred_m3.to_arrays` ile ayni (uc alan + meta_ok maskesi; maskeli hucre NaN -> ortamda pred_ok = 0).

Bloklar (panel tarih/sembol sirasiyla [T, N]):
  pred_pitv2.npz                  gercek blok (tum tohumlar ayni blogu okur; tablo tek)
  pred_pitv2_g{s}_shuf.npz        satirlar (gunler) tohum s ile permute: karistirilmis kontrol (M3 S1 yontemi)
  pred_pitv2_g{s}_shuf2.npz       ikinci, bagimsiz permutasyon (tohum 1000 + s): gurultu tabani (iki bilgisiz kol)
Kayit JSON'u: tablo SHA, blok SHA'lari, kapsam (egitim satirlarinda), K12 etiketiyle AUC (betimsel).

  python scripts/marl/build_pred_m5real.py --seeds 9,10,11 --json docs/development/measurements/phase10-m5/pred_blocks_real.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
os.chdir(ROOT)

RUN_ID = 'pit_v2_seed42'
TABLE_SHA = '9116c571c474'   # gate_pit_v2_sensitivity.json -> table_sha256 oneki (P5; M5 §2.1)
TRAIN_END = '2018-12-31'
SHUF2_OFFSET = 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='9,10,11')
    ap.add_argument('--out-dir', default=os.path.join('data', 'marl'))
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    from build_pred_m3 import PANEL, PANEL_SHA, _pc, provenance, to_arrays
    from evaluation.calibration import auc as _auc
    from marl import accounting as A
    from marl import obs_spec as OS
    from marl import panel as PN
    from prediction.pit_table import load_table

    arrays = PN.load_npz(PANEL)
    if arrays['sha256'] != PANEL_SHA:
        raise SystemExit(f"panel SHA {arrays['sha256']} != {PANEL_SHA}")
    table, info = load_table(RUN_ID)
    if not info['sha256'].startswith(TABLE_SHA):
        raise SystemExit(f"{RUN_ID} SHA {info['sha256'][:12]} != {TABLE_SHA} (on kayittaki tablo degil)")
    dates, symbols = arrays['dates'], arrays['symbols']
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    rows = np.zeros(len(dates), dtype=bool)
    rows[OS.WARMUP_ROWS:end_row + 1] = True
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(dates))
    lab = _pc().k12_labels(arrays, px)
    lab['date'] = lab['date'].dt.strftime('%Y-%m-%d')
    Y = (lab.pivot_table(index='date', columns='symbol', values='y_up', aggfunc='first')
         .reindex(index=list(dates), columns=list(symbols)).to_numpy(dtype=np.float64))
    base = to_arrays(table, dates, symbols)
    rec = {'note': 'M5 gercek sinyal bloklari (pit_v2); kapidan gecmedi, saglamlik sorusu', **provenance(),
           'panel_sha256': arrays['sha256'], 'table': {'run_id': RUN_ID, 'sha256': info['sha256']},
           'label': '1[Of_{t+2} > Of_{t+1}] (pc_alignment_m2.k12_labels)', 'shuf2_seed_offset': SHUF2_OFFSET,
           'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])], 'blocks': []}
    os.makedirs(a.out_dir, exist_ok=True)

    def write(name, blk, kind, seed):
        path = os.path.join(a.out_dir, name)
        np.savez_compressed(path, dates=dates, symbols=symbols, **blk)
        ok = np.isfinite(blk['p_up_cal']) & np.isfinite(blk['pred_logret']) & np.isfinite(blk['model_disagreement'])
        m = ok & np.isfinite(Y)
        mt = m & rows[:, None]
        row = {'kind': kind, 'seed': seed, 'path': path.replace(os.sep, '/'),
               'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
               'coverage_all': float(ok.mean()), 'coverage_train_rows': float(ok[rows].mean()),
               'auc_vs_k12_all': float(_auc(blk['p_up_cal'][m], Y[m])),
               'auc_vs_k12_train': float(_auc(blk['p_up_cal'][mt], Y[mt]))}
        rec['blocks'].append(row)
        print(f"{name}: kapsam egitim {row['coverage_train_rows']:.3f}  AUC egitim {row['auc_vs_k12_train']:.4f}  "
              f"sha {row['sha256'][:8]}")

    write('pred_pitv2.npz', base, 'real', None)
    for sd in [int(s) for s in a.seeds.split(',')]:
        for suffix, gen in (('_shuf', sd), ('_shuf2', SHUF2_OFFSET + sd)):
            perm = np.random.default_rng(gen).permutation(len(dates))
            write(f'pred_pitv2_g{sd}{suffix}.npz', {k: v[perm] for k, v in base.items()}, suffix[1:], sd)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
