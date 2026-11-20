"""Faz 10 M5 on kayit oncesi tasarim olcumleri (venv-marl; hakem-rl M5 #3, #4). KOSU SONUCU DEGIL, politika egitilmez.

Tohum basina (6, 7, 8; komsu sinyali bloklari `pred_m4nb_g{seed}.npz`, egitim satirlari WARMUP_ROWS..2018-12-31):
  GLS: derangement pi (tohum = egitim tohumu) kac gondericiyi gat komsusuna esliyor (rastgele beklenti 2E / N(N-1))
  GLM: gonderici sicakliklari T_j (varsayilan [0,25; 4]; --temp-range ile alternatifler) ve bozmanin gucu: gunluk kesitsel Spearman(b_GL, b_GLM) (b = |logit p_j|, b_GLM =
       T_j |logit p_j|; en az 3 tahminli gonderici olan gunler). On kayit kurali: ortalama > 0,9 ise H3'un gucu
       bastan dusuk beyan edilir (M5 tasarimi §4.7).

  venv-marl\\Scripts\\python scripts/marl/m5_design_checks.py --out docs/development/measurements/phase10-m5/design_checks.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
GRAPH = os.path.join('docs', 'development', 'measurements', 'phase10-m4', 'graph_rules.json')
TRAIN_END = '2018-12-31'
SPEARMAN_LOW_POWER = 0.9


def rank(v: np.ndarray) -> np.ndarray:
    """Ortalama rank (esitlikler ortalama rank alir; 1 tabanli)."""
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='6,7,8')
    ap.add_argument('--rule', default='sector+mst')
    ap.add_argument('--temp-range', default='0.25,4.0', help='GLM T_j log-tekduze [lo, hi] (on kayit: comm_models.GLM_TEMP_RANGE)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    from marl import obs_spec as OS
    from marl import panel as PN
    from marl.benchmarl_task import load_pred
    from marl.comm_models import derangement, sender_temperatures

    arrays = PN.load_npz(PANEL)
    dates, symbols = arrays['dates'], [str(s) for s in arrays['symbols']]
    N = len(symbols)
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    rows = slice(OS.WARMUP_ROWS, end_row + 1)
    g = json.load(open(GRAPH, encoding='utf-8'))
    idx = {s: i for i, s in enumerate(symbols)}
    E = {frozenset((idx[u], idx[v])) for u, v in g['edges'][a.rule]}
    rec = {'note': 'M5 on kayit oncesi tasarim olcumu (hakem-rl M5 #3, #4); kosu sonucu degil', **provenance(),
           'panel_sha256': arrays['sha256'], 'graph_rule': a.rule, 'n_edges': len(E),
           'neighbor_rate_random': 2 * len(E) / (N * (N - 1)), 'spearman_low_power_rule': SPEARMAN_LOW_POWER, 'temp_range': a.temp_range,
           'train_window': [str(dates[OS.WARMUP_ROWS]), str(dates[end_row])], 'seeds': []}
    for sd in [int(s) for s in a.seeds.split(',')]:
        pi = [int(i) for i in derangement(N, sd)]
        nb = sum(frozenset((j, pi[j])) in E for j in range(N))
        path = os.path.join('data', 'marl', f'pred_m4nb_g{sd}.npz')
        pred = load_pred(path, arrays)
        ok = (np.isfinite(pred['p_up_cal']) & np.isfinite(pred['pred_logret'])
              & np.isfinite(pred['model_disagreement']))[rows]
        p = np.clip(pred['p_up_cal'][rows], 1e-6, 1 - 1e-6)
        b = np.abs(np.log(p / (1 - p)))
        lo, hi = (float(x) for x in a.temp_range.split(','))
        T = np.asarray(sender_temperatures(N, sd, lo, hi))
        bt = b * T[None, :]
        daily = [spearman(b[t, ok[t]], bt[t, ok[t]]) for t in range(b.shape[0]) if ok[t].sum() >= 3]
        daily = np.asarray([x for x in daily if np.isfinite(x)])
        rec['seeds'].append({
            'seed': sd, 'pred_path': path.replace(os.sep, '/'), 'pred_sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
            'GLS_derangement': pi, 'GLS_neighbor_pairs': int(nb), 'GLS_neighbor_rate': nb / N,
            'GLM_temperatures': [float(t) for t in T],
            'GLM_temp_log_spread': float(np.log(T).std(ddof=1)),
            'GLM_daily_spearman_mean': float(daily.mean()), 'GLM_daily_spearman_median': float(np.median(daily)),
            'GLM_daily_spearman_p10': float(np.percentile(daily, 10)), 'GLM_n_days': int(daily.size),
            'GLM_low_power': bool(daily.mean() > SPEARMAN_LOW_POWER)})
        s = rec['seeds'][-1]
        print(f"tohum {sd}: GLS komsu esleme {nb}/{N} (rastgele {rec['neighbor_rate_random']:.3f})  "
              f"GLM gunluk Spearman ort {s['GLM_daily_spearman_mean']:.3f} medyan {s['GLM_daily_spearman_median']:.3f} "
              f"p10 {s['GLM_daily_spearman_p10']:.3f}  T [{min(T):.2f}, {max(T):.2f}]")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
