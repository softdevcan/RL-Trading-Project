"""Faz 10 M4b okuma betigi (ana venv) — M4 tasarimi §6 ON KAYDI (PILOT); sonuctan ONCE yazildi.

Soru: gat, mimari ve parametre bakimindan es iletisimsiz kola (self) gore komsudaki bilgiyi tasiyor mu?
Maliyetsiz X, SR yillik; tohumlar 3, 4, 5. D_X = ort_s [SR(G-X_s) - SR(Z-X_s)], D_S = ayni S kollarinda, G = D_X - D_S.

  (i)   egitim: D_X > 0 VE ortak HAC tek yonlu p < 0,05
  (ii)  egitim: G > 0 VE z = G / sqrt(se_X^2 + se_S^2) >= 1,645 (bagimsizlik varsayimi, tutucu)
  (iii) dogrulama: D_X > 0 VE en az 2/3 tohumda Delta > 0
PILOT_ISARETI = ucu birden (hakem-rl M4b #1: karar kapisi degil, isaret). KISMI = (i) var, ama ucu birden degil.
Aksi GOZLENMEDI. PILOT: guc dusuk ((ii) tek basina ~%35; uc kosulun bileşik gucu daha dusuk). GOZLENMEDI
"etki yok" diye okunmaz. Hicbir etiket M5 tabanini tek basina secmez; karar kullanicida.

BETIMSEL (kararlari degistirmez): uzun kosu icin gereken tohum sayisi (tek yonlu %5, guc %80):
  k = 3 * ((1,645 + 0,842) * se / etki)^2; G ve D_X icin gozlenen etkiyle, G'nin %95 GA uclariyla ve M4a capasiyla
  (G = 0,17, SE = 0,19; M4 tasarimi §3.7). Etki <= 0 ise None. Varsayim: SE tohumla 1/sqrt(k) kuculur; HAC SE
  agirlikla piyasa gunu gurultusu tasidigindan bu varsayim iyimserdir (hakem-rl M4b #2). Tahmin kabadir.

  python scripts/marl/analyze_m4b.py --out docs/development/measurements/phase10-m4/m4b_results.json
  python scripts/marl/analyze_m4b.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_m3 import ALPHA, pooled, sr  # noqa: E402
from analyze_m3b import provenance  # noqa: E402

SEEDS = (3, 4, 5)
Z_ONE_SIDED = 1.645
Z_POWER80 = 0.842
PREREG = 'phase-10-m4-design.md §6 v0.6'
M4A_ANCHOR = {'G': 0.171, 'se_G': 0.188}   # m4a_stage2.json -> arms.G.conditions (gap, se_gap)


def load(root, arm, window, seeds=SEEDS):
    out = {}
    for k in seeds:
        p = os.path.join(root, arm, f's{k}', f'eval_{window}.npz')
        if os.path.exists(p):
            d = np.load(p, allow_pickle=False)
            out[k] = {'X': d['policy_X'], 'date': d['policy_date'], 'turnover': d['policy_turnover']}
    return out


def pair(root, a_arm, b_arm, window, seeds=SEEDS):
    a, b = load(root, a_arm, window, seeds), load(root, b_arm, window, seeds)
    if any(k not in a or k not in b for k in seeds):
        return None
    for k in seeds:
        if not np.array_equal(a[k]['date'], b[k]['date']):
            raise SystemExit(f'{window}: {a_arm}/{b_arm} tohum {k} tarihleri eslesmiyor')
    sa, sb = [sr(a[k]['X']) for k in seeds], [sr(b[k]['X']) for k in seeds]
    return {**pooled([(a[k]['X'], b[k]['X']) for k in seeds]), 'sharpe_a': sa, 'sharpe_b': sb,
            'delta_per_seed': [x - y for x, y in zip(sa, sb)],
            'turnover_mean': {a_arm: float(np.mean([a[k]['turnover'].mean() for k in seeds])),
                              b_arm: float(np.mean([b[k]['turnover'].mean() for k in seeds]))}}


def seeds_needed(effect: float, se: float, k_now: int = len(SEEDS)):
    if effect <= 0 or se <= 0:
        return None
    return int(math.ceil(k_now * ((Z_ONE_SIDED + Z_POWER80) * se / effect) ** 2))


def read(root) -> dict:
    res = {'preregistration': PREREG, 'seeds': list(SEEDS), 'pilot': True}
    tx, ts, vx = pair(root, 'G-X', 'Z-X', 'train'), pair(root, 'G-S', 'Z-S', 'train'), pair(root, 'G-X', 'Z-X', 'val')
    if tx is None or ts is None or vx is None:
        return {**res, 'label': 'EKSIK'}
    gap = tx['D_annual'] - ts['D_annual']
    se_gap = float(np.hypot(tx['se_annual'], ts['se_annual']))
    z = gap / se_gap if se_gap > 0 else float('inf')
    cond = {'i_train_DX_gt_0_p': bool(tx['D_annual'] > 0 and tx['p_one_sided'] < ALPHA),
            'ii_train_G_gt_0_z': bool(gap > 0 and z >= Z_ONE_SIDED),
            'iii_val_DX_gt_0_2of3': bool(vx['D_annual'] > 0 and sum(d > 0 for d in vx['delta_per_seed']) >= 2)}
    label = ('PILOT_ISARETI' if all(cond.values()) else ('KISMI' if cond['i_train_DX_gt_0_p'] else 'GOZLENMEDI'))
    lo, hi = gap - 1.96 * se_gap, gap + 1.96 * se_gap
    return {**res, 'train_X': tx, 'train_S': ts, 'val_X': vx, 'G': gap, 'se_G': se_gap, 'z_G': z,
            'G_ci95': [lo, hi], 'conditions': cond, 'label': label,
            'sizing_betimsel': {'seeds_for_80pct_G_observed': seeds_needed(gap, se_gap),
                                'seeds_for_80pct_G_ci_low': seeds_needed(lo, se_gap),
                                'seeds_for_80pct_G_ci_high': seeds_needed(hi, se_gap),
                                'seeds_for_80pct_G_m4a_anchor': seeds_needed(M4A_ANCHOR['G'], M4A_ANCHOR['se_G']),
                                'seeds_for_80pct_DX_observed': seeds_needed(tx['D_annual'], tx['se_annual']),
                                'note': 'SE tohumla 1/sqrt(k) kuculur varsayimi iyimser (HAC gun gurultusu); kaba'}}


def self_test() -> bool:
    rng = np.random.default_rng(0)
    T, ok, total = 400, 0, 0
    dates = np.array([f'2015-01-{i:04d}' for i in range(T)])

    def check(name, good):
        nonlocal ok, total
        total += 1
        ok += bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name}")

    def build(edges):
        root = tempfile.mkdtemp(prefix='m4b_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            for arm, e in edges.items():
                for k in SEEDS:
                    d = os.path.join(root, arm, f's{k}')
                    os.makedirs(d, exist_ok=True)
                    np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=mkt + rng.normal(0, 0.003, T) + e,
                             policy_date=dates, policy_turnover=np.zeros(T), policy_cash_weight=np.zeros(T))
        return root

    r = read(build({'G-X': 0.003, 'Z-X': 0.0, 'G-S': 0.0, 'Z-S': 0.0}))
    check('bilgi tasiniyor: PILOT_ISARETI, tohum gereksinimi (gozlenen + capa) hesaplanir',
          r['label'] == 'PILOT_ISARETI' and r['sizing_betimsel']['seeds_for_80pct_G_observed'] is not None
          and r['sizing_betimsel']['seeds_for_80pct_G_m4a_anchor'] == 23)
    r = read(build({'G-X': 0.003, 'Z-X': 0.0, 'G-S': 0.003, 'Z-S': 0.0}))
    check('avantaj karistirilmis sinyalde de var: KISMI ((ii) eler)',
          r['label'] == 'KISMI' and not r['conditions']['ii_train_G_gt_0_z'])
    r = read(build({'G-X': 0.0, 'Z-X': 0.0, 'G-S': 0.0, 'Z-S': 0.0}))
    check('bos: GOZLENMEDI', r['label'] == 'GOZLENMEDI')
    r = read(build({'G-X': 0.003, 'Z-X': 0.0}))
    check('eksik kol: EKSIK', r['label'] == 'EKSIK')
    check('tohum gereksinimi: etki <= 0 -> None; etki = 2 SE -> 3*(2,487/2)^2 = 5', seeds_needed(-0.1, 0.1) is None
          and seeds_needed(0.2, 0.1) == 5)
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m4b'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res = {**read(a.root), 'provenance': provenance()}
    print(json.dumps({k: res.get(k) for k in ('label', 'conditions', 'G', 'z_G', 'sizing_betimsel')}, indent=1,
                     ensure_ascii=False, default=float))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
