"""Faz 10 M5 kesif olcumu (M5 tasarimi §3.2; ana venv): PC060-K12 guveni yon isabetiyle iliskili mi, ve iki guven
biciminin (prob, zlogodds) dinamik araligi ne? KOSU SONUCU DEGIL: sinyal ozelligi; politika egitilmez.

Hucreler: egitim satirlari (WARMUP_ROWS .. 2018-12-31), tahmini olan (uc alan sonlu = ortamin pred_ok'u) ve etiketi
olan hucreler. Etiket 1[Of_{t+2} > Of_{t+1}] (`pc_alignment_m2.k12_labels`; build_pred_m3 ile ayni).
  guven c = |p - 0,5|; isabet = 1[(p > 0,5) == y] (p = 0,5 -> 0,5)
  besliklere gore isabet (beslik sinirlari blogun kendi c dagilimindan), corr(c, isabet)
  prob terimi log clip(0,5 + c, 0,5, 0,99) ve zlogodds terimi |logit p| / std: yuzdelikler ve p90 - p10 araligi
Ilk olcum 2026-11-19'da M4a bloklarinda (tohum 20261115, 1, 2) scratchpad betigiyle yapildi; bu betik onu depodan
yeniden uretir. Karistirilmis bloklar (`_shuf`) kontroldur: isabet ~0,5, corr ~0 beklenir.

  python scripts/marl/conf_informative_m5.py --seeds 20261115,1,2 \\
      --out docs/development/measurements/phase10-m5/conf_informative.json
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

TRAIN_END = '2018-12-31'
Q = (10, 25, 50, 75, 90)


def summary(p: np.ndarray, y: np.ndarray) -> dict:
    """p, y: tahminli ve etiketli hucreler (1-B)."""
    c = np.abs(p - 0.5)
    hit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y > 0.5)).astype(float))
    edges = np.quantile(c, [0.2, 0.4, 0.6, 0.8])
    qi = np.searchsorted(edges, c, side='right')
    quint = [float(hit[qi == k].mean()) for k in range(5)]
    b_prob = np.log(np.clip(0.5 + c, 0.5, 0.99))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lz = np.abs(np.log(pc / (1 - pc)))
    b_z = lz / lz.std(ddof=1)
    pct = lambda v: {f'p{q}': float(np.percentile(v, q)) for q in Q}   # noqa: E731
    return {'n_cells': int(p.size), 'hit_rate': float(hit.mean()), 'hit_by_conf_quintile': quint,
            'corr_conf_hit': float(np.corrcoef(c, hit)[0, 1]),
            'abs_p_minus_half': pct(c), 'prob_term': {**pct(b_prob), 'p90_minus_p10': float(
                np.percentile(b_prob, 90) - np.percentile(b_prob, 10))},
            'zlogodds_term': {**pct(b_z), 'p90_minus_p10': float(np.percentile(b_z, 90) - np.percentile(b_z, 10)),
                              'z_scale': float(lz.std(ddof=1))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='20261115,1,2')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from build_pred_m3 import PANEL, PANEL_SHA, _pc, provenance
    from marl import accounting as A
    from marl import obs_spec as OS
    from marl import panel as PN

    arrays = PN.load_npz(PANEL)
    if arrays['sha256'] != PANEL_SHA:
        raise SystemExit(f"panel SHA {arrays['sha256']} != {PANEL_SHA}")
    dates, symbols = arrays['dates'], [str(s) for s in arrays['symbols']]
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    rows = np.zeros(len(dates), dtype=bool)
    rows[OS.WARMUP_ROWS:end_row + 1] = True
    px = A.fill_prices(arrays['open'], arrays['close'], A.LISTING_LOCKOUT, A.limit_down_gaps(dates))
    lab = _pc().k12_labels(arrays, px)
    lab['date'] = lab['date'].dt.strftime('%Y-%m-%d')
    Y = (lab.pivot_table(index='date', columns='symbol', values='y_up', aggfunc='first')
         .reindex(index=list(dates), columns=symbols).to_numpy(dtype=np.float64))
    with open(os.path.abspath(__file__), 'rb') as fh:
        rec = {'note': 'M5 kesif olcumu: sinyal ozelligi, kosu sonucu degil; BILEREK GELECEKTEN etiketli PC060-K12',
               **provenance(), 'script_sha256': hashlib.sha256(fh.read()).hexdigest(),
               'panel_sha256': arrays['sha256'], 'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])],
               'label': '1[Of_{t+2} > Of_{t+1}] (pc_alignment_m2.k12_labels)', 'blocks': []}
    for sd in [int(s) for s in a.seeds.split(',')]:
        for suffix in ('', '_shuf'):
            path = os.path.join('data', 'marl', f'pred_pc060_k12_g{sd}{suffix}.npz')
            d = np.load(path, allow_pickle=False)
            if not (np.array_equal(d['dates'], dates) and np.array_equal(d['symbols'], arrays['symbols'])):
                raise SystemExit(f'blok panel ile hizali degil: {path}')
            ok = np.isfinite(d['p_up_cal']) & np.isfinite(d['pred_logret']) & np.isfinite(d['model_disagreement'])
            m = ok & rows[:, None] & np.isfinite(Y)
            s = summary(d['p_up_cal'][m], Y[m])
            rec['blocks'].append({'seed': sd, 'shuffled_rows': bool(suffix), 'path': path.replace(os.sep, '/'),
                                  'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(), **s})
            print(f"tohum {sd}{suffix}: besliklere gore isabet {[round(v, 3) for v in s['hit_by_conf_quintile']]} "
                  f"corr {s['corr_conf_hit']:+.3f}  |p-0,5| medyan {s['abs_p_minus_half']['p50']:.4f}  "
                  f"aralik p90-p10: prob {s['prob_term']['p90_minus_p10']:.4f}  "
                  f"zlogodds {s['zlogodds_term']['p90_minus_p10']:.3f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
