"""Faz 10 M4: `gat` kolu graf kenar kurali olcumu (plan §3.2.5; hakem-2 #14: kural M4 oncesi sabitlenir,
yogunluk raporlanir). Ana venv, saf numpy. Performans sayisi YOK; yalniz graf yapisi.

Graf yalniz EGITIM penceresinden kurulur (B0 mantigi; plan §3.2.5): satirlar WARMUP_ROWS .. 2018-12-31.
Getiri: log(C_t / C_{t-1}), iki gun de gecerliyse; korelasyon cift bazinda tam gozlemli gunlerle (Pearson).
Uzaklik d = sqrt(2 (1 - rho)) (Mantegna 1999). Sektor kenarlari E2 gruplarindan (`DIGER` grup sayilmaz).

Kurallar: sector | thr_<tau> | mst | topk_<k> (i-j kenari: j, i'nin en korele k komsusundaysa YA DA tersi) |
sector+mst | sector+topk_<k>. Kararlilik: egitim penceresinin iki yarisinda ayri kurulan graflarin Jaccard'i.

  python scripts/marl/graph_m4.py --out docs/development/measurements/phase10-m4/graph_rules.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from data.bist30_symbols import SECTORS_FILE, load_sector_groups  # noqa: E402
from marl import obs_spec as OS  # noqa: E402
from marl import panel as PN  # noqa: E402

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
TRAIN_END = '2018-12-31'
THRESHOLDS = (0.3, 0.4, 0.5)
TOPK = (2, 3, 5)


def log_returns(close: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(close[1:] / close[:-1])
    r[~np.isfinite(r)] = np.nan
    return r


def pairwise_corr(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cift bazinda tam gozlemli Pearson; donus (rho [N,N], ortak gun sayisi [N,N])."""
    n = r.shape[1]
    rho = np.eye(n)
    cnt = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(r[:, i]) & np.isfinite(r[:, j])
            cnt[i, j] = cnt[j, i] = int(m.sum())
            rho[i, j] = rho[j, i] = float(np.corrcoef(r[m, i], r[m, j])[0, 1])
    np.fill_diagonal(cnt, [int(np.isfinite(r[:, i]).sum()) for i in range(n)])
    return rho, cnt


def sector_edges(symbols, groups) -> set:
    return {(i, j) for i in range(len(symbols)) for j in range(i + 1, len(symbols))
            if groups[symbols[i]] == groups[symbols[j]] and groups[symbols[i]] != 'DIGER'}


def thr_edges(rho, tau) -> set:
    n = rho.shape[0]
    return {(i, j) for i in range(n) for j in range(i + 1, n) if rho[i, j] > tau}


def mst_edges(rho) -> set:
    """Kruskal, d = sqrt(2(1-rho)); esitlikte (i, j) sirasi (deterministik)."""
    n = rho.shape[0]
    d = np.sqrt(np.clip(2.0 * (1.0 - rho), 0.0, None))
    cand = sorted((d[i, j], i, j) for i in range(n) for j in range(i + 1, n))
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = set()
    for _, i, j in cand:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
            out.add((i, j))
            if len(out) == n - 1:
                break
    return out


def topk_edges(rho, k) -> set:
    n = rho.shape[0]
    out = set()
    for i in range(n):
        order = [j for j in np.argsort(-rho[i], kind='stable') if j != i][:k]
        for j in order:
            out.add((min(i, j), max(i, j)))
    return out


def components(n, edges) -> int:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def build_rules(rho, sec) -> dict:
    rules = {'sector': sec, 'mst': mst_edges(rho)}
    for t in THRESHOLDS:
        rules[f'thr_{t}'] = thr_edges(rho, t)
    for k in TOPK:
        rules[f'topk_{k}'] = topk_edges(rho, k)
    rules['sector+mst'] = sec | rules['mst']
    for k in TOPK:
        rules[f'sector+topk_{k}'] = sec | rules[f'topk_{k}']
    return rules


def describe(n, edges, sec) -> dict:
    deg = np.zeros(n, dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    m = len(edges)
    return {'n_edges': m, 'density': round(m / (n * (n - 1) / 2), 4), 'n_components': components(n, edges),
            'n_isolated': int((deg == 0).sum()), 'degree_min': int(deg.min()), 'degree_median': float(np.median(deg)),
            'degree_max': int(deg.max()), 'frac_edges_in_sector': round(len(edges & sec) / m, 4) if m else None}


def jaccard(a: set, b: set) -> float | None:
    u = a | b
    return round(len(a & b) / len(u), 4) if u else None


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(SECTORS_FILE, 'rb') as fh:
        sec_sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha, 'sectors_csv_sha256': sec_sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arrays = PN.load_npz(PANEL)
    dates = arrays['dates']
    symbols = [str(s) for s in arrays['symbols']]
    end_row = int(np.flatnonzero(dates <= TRAIN_END)[-1])
    lo = OS.WARMUP_ROWS
    close = arrays['close'][lo:end_row + 1]
    r = log_returns(close)
    groups = load_sector_groups()
    sec = sector_edges(symbols, groups)
    n = len(symbols)

    rho, cnt = pairwise_corr(r)
    off = rho[np.triu_indices(n, 1)]
    rules = build_rules(rho, sec)
    half = r.shape[0] // 2
    rho_a, _ = pairwise_corr(r[:half])
    rho_b, _ = pairwise_corr(r[half:])
    ra, rb = build_rules(rho_a, sec), build_rules(rho_b, sec)

    out = {
        'note': 'M4 gat graf kenar kurali olcumu; yalniz egitim penceresi; performans sayisi YOK',
        **provenance(), 'panel_sha256': arrays['sha256'],
        'window': [str(dates[lo]), str(dates[end_row])], 'n_return_days': int(r.shape[0]),
        'halves': [[str(dates[lo + 1]), str(dates[lo + half])], [str(dates[lo + half + 1]), str(dates[end_row])]],
        'pair_days_min': int(cnt[np.triu_indices(n, 1)].min()),
        'rho': {'mean': round(float(off.mean()), 4), 'median': round(float(np.median(off)), 4),
                'min': round(float(off.min()), 4), 'max': round(float(off.max()), 4)},
        'sector_groups': {g: sorted(s for s in symbols if groups[s] == g) for g in sorted(set(groups.values()))},
        'rules': {k: {**describe(n, e, sec), 'stability_jaccard_halves': jaccard(ra[k], rb[k])}
                  for k, e in rules.items()},
        'edges': {k: sorted([symbols[i], symbols[j]] for i, j in e) for k, e in rules.items()
                  if k in ('sector', 'mst', 'topk_3', 'sector+mst', 'sector+topk_3')},
    }
    width = max(len(k) for k in out['rules'])
    print(f"pencere {out['window']}, getiri gunu {out['n_return_days']}, ort. rho {out['rho']['mean']}")
    for k, v in out['rules'].items():
        print(f"{k:<{width}}  kenar {v['n_edges']:>3}  yogunluk {v['density']:.3f}  bilesen {v['n_components']}"
              f"  derece {v['degree_min']}/{v['degree_median']:.0f}/{v['degree_max']}"
              f"  sektor-ici {v['frac_edges_in_sector']}  Jaccard {v['stability_jaccard_halves']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
