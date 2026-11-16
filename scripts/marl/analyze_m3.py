"""Faz 10 M3a okuma betigi (ana venv) — M3 tasarim §5 ON KAYDI v0.2'deki kural; sonuctan ONCE yazildi.

Girdi duzeni (eval_m3.py ciktilari):
  <root>/<kol>/s<k>/eval_<pencere>.npz   kol in {A1, S1}, k in {0, 1, 2}, pencere in {train, val}
  <root>/baselines_<pencere>_0.npz       (ew, bh, cash; ayni muhasebe, maliyetsiz)
A1 = PC060-K12, S1 = karistirilmis PC (ayni uretec, gunler permute; sinyal yok). Maliyetsiz, lr 5e-4.

GECTI = uc kosulun HEPSI (hakem-rl M3a #1, #3, #5):
  (i)   egitim penceresi: D = ort_s [SR(A1_s) - SR(S1_s)] >= 0,3 (yillik) VE ortak HAC tek yonlu p < 0,05
  (ii)  egitim penceresi: min_s SR(A1_s) > max_s SR(S1_s) (3'e 3 kesin permutasyon, tek yonlu p = 1/20)
  (iii) dogrulama penceresi ISARET KONTROLU (yetersiz guc; genelleme kaniti degil): D_val > 0 VE >= 2/3 tohumda Delta > 0
KISMI = (i)'de D > 0 ve p < 0,05 var ama GECTI degil. Aksi halde GOZLENMEDI ("bu butce ve lr'de aktarim
gozlenmedi", kanit yoklugu; "aktaramiyor" denmez).

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
PC, CTRL = 'A1', 'S1'
SEEDS = (0, 1, 2)
ALPHA = 0.05
D_MIN = 0.3


def sr(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / x.std(ddof=1) * ANN)


def pooled(pairs):
    from evaluation.sharpe_tests import sharpe_diff_hac_pooled
    r = sharpe_diff_hac_pooled(pairs)
    p_one = float(r['p_value'] / 2 if r['delta'] > 0 else 1 - r['p_value'] / 2)
    return {'D_annual': r['delta'] * ANN, 'se_annual': r['se'] * ANN, 'p_one_sided': p_one,
            'p_two_sided': r['p_value'], 'n_days': r['n'], 'k_pairs': r['k_pairs']}


def load_arm(root, arm, window):
    out = {}
    for k in SEEDS:
        p = os.path.join(root, arm, f's{k}', f'eval_{window}.npz')
        if os.path.exists(p):
            d = np.load(p, allow_pickle=False)
            out[k] = {'X': d['policy_X'], 'date': d['policy_date'], 'turnover': d['policy_turnover'],
                      'cash': d['policy_cash_weight']}
    return out


def window_stats(root, window):
    a, b = load_arm(root, PC, window), load_arm(root, CTRL, window)
    if any(k not in a or k not in b for k in SEEDS):
        return {'status': 'eksik', 'seeds_present': sorted(set(a) & set(b))}   # eksik kol okunmaz (§5.1)
    for k in SEEDS:
        if not np.array_equal(a[k]['date'], b[k]['date']):
            raise SystemExit(f'{window}: tohum {k} tarihleri eslesmiyor')
    sa, sb = [sr(a[k]['X']) for k in SEEDS], [sr(b[k]['X']) for k in SEEDS]
    rec = {**pooled([(a[k]['X'], b[k]['X']) for k in SEEDS]),
           'sharpe_A1': sa, 'sharpe_S1': sb, 'delta_per_seed': [x - y for x, y in zip(sa, sb)],
           'min_A1_gt_max_S1': bool(min(sa) > max(sb)),
           'turnover_mean': {PC: float(np.mean([a[k]['turnover'].mean() for k in SEEDS])),
                             CTRL: float(np.mean([b[k]['turnover'].mean() for k in SEEDS]))},
           'cash_mean': {PC: float(np.mean([a[k]['cash'].mean() for k in SEEDS])),
                         CTRL: float(np.mean([b[k]['cash'].mean() for k in SEEDS]))}}
    bl_path = os.path.join(root, f'baselines_{window}_0.npz')
    if os.path.exists(bl_path):   # betimsel: kollarin tohum ortalamali serisi vs ayni muhasebeli baseline'lar
        from evaluation.sharpe_tests import sharpe_diff_hac
        bl = np.load(bl_path, allow_pickle=False)
        rec['baseline_sharpe'] = {kind: sr(bl[f'{kind}_X']) for kind in ('ew', 'bh')}
        for arm, data in ((PC, a), (CTRL, b)):
            xm = np.mean([data[k]['X'] for k in SEEDS], axis=0)
            rec[f'{arm}_seedmean_vs'] = {kind: {kk: v * (ANN if kk != 'p_value' else 1) for kk, v in
                                                sharpe_diff_hac(xm, bl[f'{kind}_X']).items() if kk in ('delta', 'se', 'p_value')}
                                         for kind in ('ew', 'bh')}
    return rec


def analyze(root):
    res = {'preregistration': 'phase-10-m3-design.md §5 v0.2', 'train': window_stats(root, 'train'),
           'val': window_stats(root, 'val')}
    tr, va = res['train'], res['val']
    if 'D_annual' not in tr or 'D_annual' not in va:
        res.update(reading='EKSIK: okunmaz (§5.1)', label='EKSIK', gate_passed=False)
        return res
    c1 = tr['D_annual'] >= D_MIN and tr['p_one_sided'] < ALPHA
    c2 = tr['min_A1_gt_max_S1']
    c3 = va['D_annual'] > 0 and sum(d > 0 for d in va['delta_per_seed']) >= 2
    res['conditions'] = {'i_train_D_ge_0.3_p_lt_0.05': bool(c1), 'ii_train_min_A1_gt_max_S1': bool(c2),
                         'iii_val_D_gt_0_and_2of3': bool(c3)}
    if c1 and c2 and c3:
        res.update(label='GECTI', reading='GECTI: egitim penceresinde, ayni kapasite ve parmak izine sahip kontrole karsi, '
                                          'maliyetsiz kosulda sinyal aktarimi gozlendi; dogrulamada isaret celismiyor '
                                          '(M3 §5 v0.2). M3b/M4 acilir.')
    elif tr['D_annual'] > 0 and tr['p_one_sided'] < ALPHA:
        res.update(label='KISMI', reading='KISMI: egitim penceresinde anlamli pozitif fark var ama GECTI kosullarinin '
                                          'hepsi saglanmadi (§5 v0.2). Gercek sinyalli MARL kosusu yapilmaz.')
    else:
        res.update(label='GOZLENMEDI', reading='GOZLENMEDI: bu butce ve lr\'de aktarim gozlenmedi (kanit yoklugu; '
                                               '§5 v0.2). Gercek sinyalli MARL kosusu yapilmaz; kesifsel tani.')
    res['gate_passed'] = res['label'] == 'GECTI'
    return res


def self_test():
    import tempfile
    rng = np.random.default_rng(0)
    T = 400
    dates = np.array([f'2015-01-{i:04d}' for i in range(T)])
    ok = 0

    def write(root, arm, k, window, x):
        d = os.path.join(root, arm, f's{k}')
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x, policy_date=dates, policy_turnover=np.zeros(T),
                 policy_cash_weight=np.zeros(T))

    for case, edge, want in (('sinyal', 0.003, 'GECTI'), ('bos', 0.0, 'GOZLENMEDI')):
        root = tempfile.mkdtemp(prefix=f'm3a_{case}_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            for k in SEEDS:
                write(root, CTRL, k, window, mkt + rng.normal(0, 0.003, T))
                write(root, PC, k, window, mkt + rng.normal(0, 0.003, T) + edge)
        r = analyze(root)
        good = r['label'] == want
        print(f"  [{'OK' if good else 'FAIL'}] {case}: {r['label']} D_train={r['train']['D_annual']:.2f}")
        ok += good
    root = tempfile.mkdtemp(prefix='m3a_eksik_')
    for k in SEEDS[:2]:
        write(root, PC, k, 'train', rng.normal(0, 0.01, T))
        write(root, CTRL, k, 'train', rng.normal(0, 0.01, T))
    r = analyze(root)
    good = r['label'] == 'EKSIK' and not r['gate_passed']
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
    print(json.dumps({k: res.get(k) for k in ('label', 'reading', 'conditions')}, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
