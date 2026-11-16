"""Faz 10 M3a okuma betigi (ana venv) — M3 tasarim §5 ON KAYDI'ndaki kural; sonuctan ONCE yazildi.

Girdi duzeni (eval_m3.py ciktilari):
  <root>/<kol>/s<k>/eval_<pencere>.npz   kol in {A0, A1, B0, B1}, k in {0..4}, pencere in {train, val}
  <root>/baselines_<pencere>_<bps>.npz   (ew, bh, cash; ayni muhasebe)
Birincil: egitim penceresi, D = ort_s [SR(A1_s) - SR(A0_s)] (yillik), ortak HAC (`sharpe_diff_hac_pooled`),
tek yonlu p. GECTI: D > 0 ve p_tek < 0,05. Ikincil/betimsel kalemler kapi kararini degistirmez.

  python scripts/marl/analyze_m3.py --root results/marl_m3 --out docs/development/measurements/phase10-m3/m3a_results.json
  python scripts/marl/analyze_m3.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ANN = np.sqrt(252.0)
ARMS = {'A': ('A1', 'A0', 0.0), 'B': ('B1', 'B0', 20.0)}   # cift: (PC kolu, tahminsiz kol, maliyet bps)
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.05


def sr(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * ANN)


def pooled(pairs):
    from evaluation.sharpe_tests import sharpe_diff_hac_pooled
    r = sharpe_diff_hac_pooled(pairs)
    D, se = r['delta'] * ANN, r['se'] * ANN
    p_one = float(r['p_value'] / 2 if r['delta'] > 0 else 1 - r['p_value'] / 2)
    return {'D_annual': D, 'se_annual': se, 'p_one_sided': p_one, 'p_two_sided': r['p_value'],
            'per_seed_delta_annual': [d * ANN for d in r['deltas']], 'n_days': r['n'], 'k_pairs': r['k_pairs']}


def load_arm(root, arm, window):
    out = {}
    for k in SEEDS:
        p = os.path.join(root, arm, f's{k}', f'eval_{window}.npz')
        if os.path.exists(p):
            d = np.load(p, allow_pickle=False)
            out[k] = {'X': d['policy_X'], 'date': d['policy_date'], 'turnover': d['policy_turnover'],
                      'cash': d['policy_cash_weight']}
    return out


def analyze(root):
    res = {'preregistration': 'phase-10-m3-design.md §5', 'pairs': {}, 'arms': {}}
    for pair, (pc, nop, bps) in ARMS.items():
        for window in ('train', 'val'):
            a, b = load_arm(root, pc, window), load_arm(root, nop, window)
            seeds = [k for k in SEEDS if k in a and k in b]
            key = f'{pair}_{window}'
            if len(seeds) < len(SEEDS):
                res['pairs'][key] = {'status': 'eksik', 'seeds_present': seeds}   # eksik kol okunmaz (§5.1)
                continue
            for k in seeds:
                if not np.array_equal(a[k]['date'], b[k]['date']):
                    raise SystemExit(f'{key}: tohum {k} tarihleri eslesmiyor')
            res['pairs'][key] = {**pooled([(a[k]['X'], b[k]['X']) for k in seeds]), 'bps': bps}
            bl_path = os.path.join(root, f'baselines_{window}_{int(bps)}.npz')
            for arm, data in ((pc, a), (nop, b)):
                rec = {'sharpe_per_seed': [sr(data[k]['X']) for k in seeds],
                       'turnover_mean': float(np.mean([data[k]['turnover'].mean() for k in seeds])),
                       'cash_mean': float(np.mean([data[k]['cash'].mean() for k in seeds]))}
                if os.path.exists(bl_path):
                    bl = np.load(bl_path, allow_pickle=False)
                    from evaluation.sharpe_tests import sharpe_diff_hac
                    xm = np.mean([data[k]['X'] for k in seeds], axis=0)
                    for kind in ('ew', 'bh', 'cash'):
                        xb = bl[f'{kind}_X']
                        if kind == 'cash':
                            rec['vs_cash_sharpe'] = sr(xm)   # nakit X = 0: fark kolun kendi SR'si
                            continue
                        t = sharpe_diff_hac(xm, xb)
                        rec[f'vs_{kind}'] = {'delta_annual': t['delta'] * ANN, 'se_annual': t['se'] * ANN,
                                             'p_two_sided': t['p_value']}
                    rec['baseline_sharpe'] = {kind: sr(bl[f'{kind}_X']) for kind in ('ew', 'bh')}
                res['arms'][f'{arm}_{window}'] = rec
    prim = res['pairs'].get('A_train', {})
    if 'D_annual' not in prim:
        res['reading'] = 'BIRINCIL CIFT EKSIK: okunmaz (§5.1)'
    elif prim['D_annual'] > 0 and prim['p_one_sided'] < ALPHA:
        res['reading'] = ('GECTI: boru hatti bilinen bir sinyali maliyetsiz kosulda eyleme aktariyor '
                          '(M3 tasarim §5 kural 1). M3b ve M4 acilir.')
    else:
        res['reading'] = ('KALDI: bu butce ve yapida boru hatti bilinen sinyali aktaramiyor (§5 kural 2). '
                          'Gercek sinyalli MARL kosusu yapilmaz; kesifsel tani sirasi.')
    res['gate_passed'] = bool('D_annual' in prim and prim['D_annual'] > 0 and prim['p_one_sided'] < ALPHA)
    return res


def self_test():
    import tempfile
    rng = np.random.default_rng(0)
    T = 400
    dates = np.array([f'2015-01-{i:04d}' for i in range(T)])
    mkt = rng.normal(0.0003, 0.01, T)
    ok = 0

    def write(root, arm, k, x):
        d = os.path.join(root, arm, f's{k}')
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, 'eval_train.npz'), policy_X=x, policy_date=dates, policy_turnover=np.zeros(T),
                 policy_cash_weight=np.zeros(T))

    for case, edge in (('sinyal', 0.002), ('bos', 0.0)):
        root = tempfile.mkdtemp(prefix=f'm3a_{case}_')
        for k in SEEDS:
            noise = rng.normal(0, 0.003, T)
            write(root, 'A0', k, mkt + noise)
            write(root, 'A1', k, mkt + noise + edge)
        r = analyze(root)
        want = case == 'sinyal'
        good = r['gate_passed'] == want and r['pairs'].get('A_val', {}).get('status') == 'eksik'
        print(f"  [{'OK' if good else 'FAIL'}] {case}: gecti={r['gate_passed']} D={r['pairs']['A_train']['D_annual']:.2f}")
        ok += good
    root = tempfile.mkdtemp(prefix='m3a_eksik_')
    for k in SEEDS[:3]:
        write(root, 'A0', k, mkt)
        write(root, 'A1', k, mkt)
    r = analyze(root)
    good = not r['gate_passed'] and 'EKSIK' in r['reading']
    print(f"  [{'OK' if good else 'FAIL'}] eksik tohum: okunmaz")
    ok += good
    print(f"Sonuc: {ok}/3")
    return ok == 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m3'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res = analyze(a.root)
    print(json.dumps({k: res[k] for k in ('reading', 'gate_passed')}, indent=1, ensure_ascii=False))
    print(json.dumps(res['pairs'], indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
