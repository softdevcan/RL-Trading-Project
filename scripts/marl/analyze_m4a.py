"""Faz 10 M4a okuma betigi (ana venv) — M4 tasarimi §3 ON KAYDI'ndaki kural; sonuctan ONCE yazildi.

Maliyetsiz fazla getiri X, SR yillik. Girdi: <root>/<kol>/s<k>/eval_<pencere>.npz. Kollar N-X, A-X, G-X (asama 1),
A-S / G-S + N-S (asama 2). Her iletisim kolu k in {A, G} icin D_k = ort_s [SR(k-X_s) - SR(N-X_s)].

ASAMA 1. Kol k icin uc kosul (DELTA = 0,25, kullanici karari K-M4-4):
  (i)   egitim: D_k >= DELTA VE ortak HAC tek yonlu p, iki kol uzerinde Holm duzeltmeli, < 0,05
  (ii)  egitim: min_s SR(k-X_s) > max_s SR(N-X_s)
  (iii) dogrulama: D_k > 0 VE en az 2/3 tohumda Delta > 0
(i)-(iii)'un hepsi -> kol asama 2'ye (`stage2_arms`: k-S ve N-S). KISMI = (i)'de D_k > 0 ve Holm p < 0,05 var ama uc
kosulun hepsi degil. Aksi GOZLENMEDI. Hicbir kol gecmezse asama 2 kosulmaz.

ASAMA 2 (kontrol). Asama 1'i gecen kol k icin:
  (iv)  egitim: D_k^S = ort_s [SR(k-S_s) - SR(N-S_s)] < D_k - DELTA/2   (D_k asama 1 JSON'undan)
(iv) saglanirsa k GECTI; saglanmazsa KISMI (iletisim avantaji tasinan bilgiye baglanamadi).

  python scripts/marl/analyze_m4a.py --stage 1 --out docs/development/measurements/phase10-m4/m4a_stage1.json
  python scripts/marl/analyze_m4a.py --stage 2 --out docs/development/measurements/phase10-m4/m4a_stage2.json
  python scripts/marl/analyze_m4a.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_m3 import ALPHA, SEEDS, sr  # noqa: E402
from analyze_m3b import holm, pair, provenance  # noqa: E402

DELTA = 0.25
COMM_ARMS = ('A', 'G')
CTRL = 'N'
PREREG = 'phase-10-m4-design.md §3 v0.3'
STAGE1_JSON = os.path.join('docs', 'development', 'measurements', 'phase10-m4', 'm4a_stage1.json')


def baselines(root, window):
    path = os.path.join(root, f'baselines_{window}_0.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return {kind: sr(d[f'{kind}_X']) for kind in ('ew', 'bh')}


def arm_conditions(tr: dict, va: dict, p_holm: float) -> dict:
    return {'i_train_D_ge_delta_holm_p': bool(tr['D_annual'] >= DELTA and p_holm < ALPHA),
            'ii_train_min_k_gt_max_N': bool(min(tr['sharpe_a']) > max(tr['sharpe_b'])),
            'iii_val_D_gt_0_and_2of3': bool(va['D_annual'] > 0 and sum(d > 0 for d in va['delta_per_seed']) >= 2)}


def stage1(root) -> dict:
    res = {'preregistration': PREREG, 'stage': 1, 'delta': DELTA, 'arms': {}}
    pairs = {}
    for k in COMM_ARMS:
        tr, va = pair(root, f'{k}-X', f'{CTRL}-X', 'train'), pair(root, f'{k}-X', f'{CTRL}-X', 'val')
        if tr is None or va is None:
            return {**res, 'label': 'EKSIK', 'missing': f'{k}-X', 'stage2_arms': []}
        pairs[k] = (tr, va)
    adj = holm([pairs[k][0]['p_one_sided'] for k in COMM_ARMS])
    stage2 = []
    for k, p_holm in zip(COMM_ARMS, adj):
        tr, va = pairs[k]
        cond = arm_conditions(tr, va, p_holm)
        passed = all(cond.values())
        label = 'ASAMA2' if passed else ('KISMI' if tr['D_annual'] > 0 and p_holm < ALPHA else 'GOZLENMEDI')
        res['arms'][k] = {'train': tr, 'val': va, 'p_holm': p_holm, 'conditions': cond, 'label': label}
        if passed:
            stage2.append(f'{k}-S')
    if stage2:
        stage2.append(f'{CTRL}-S')
    res['baseline_sharpe'] = {w: baselines(root, w) for w in ('train', 'val')}
    res['stage2_arms'] = stage2
    res['label'] = 'ASAMA2' if stage2 else 'GECEN_KOL_YOK'
    return res


def stage2(root, s1: dict) -> dict:
    if s1.get('preregistration') != PREREG or s1.get('stage') != 1:
        raise SystemExit('asama 1 JSON bu on kayda ait degil')
    res = {'preregistration': PREREG, 'stage': 2, 'delta': DELTA, 'arms': {}}
    for k in COMM_ARMS:
        a1 = s1['arms'].get(k, {})
        if a1.get('label') != 'ASAMA2':
            res['arms'][k] = {'label': a1.get('label'), 'note': 'asama 1 etiketi; asama 2 kosulmadi'}
            continue
        tr = pair(root, f'{k}-S', f'{CTRL}-S', 'train')
        if tr is None:
            res['arms'][k] = {'label': 'EKSIK', 'missing': f'{k}-S / {CTRL}-S'}
            continue
        d_x = a1['train']['D_annual']
        c4 = bool(tr['D_annual'] < d_x - DELTA / 2)
        res['arms'][k] = {'train_S': tr, 'D_X_train': d_x, 'conditions': {'iv_train_DS_lt_DX_minus_half_delta': c4},
                          'label': 'GECTI' if c4 else 'KISMI'}
    return res


def self_test() -> bool:
    rng = np.random.default_rng(0)
    T, ok, total = 400, 0, 0
    dates = np.array([f'2015-01-{i:04d}' for i in range(T)])

    def write(root, arm, k, window, x):
        d = os.path.join(root, arm, f's{k}')
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x, policy_date=dates, policy_turnover=np.zeros(T),
                 policy_cash_weight=np.zeros(T))

    def check(name, good):
        nonlocal ok, total
        total += 1
        ok += bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name}")

    def build(edges):
        root = tempfile.mkdtemp(prefix='m4a_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            for arm, e in edges.items():
                for k in SEEDS:
                    write(root, arm, k, window, mkt + rng.normal(0, 0.003, T) + e.get(window, 0.0))
        return root

    both = lambda v: {'train': v, 'val': v}   # noqa: E731
    tr0 = {'D_annual': 0.5, 'sharpe_a': [1.0, 1.1, 1.2], 'sharpe_b': [0.5, 0.6, 0.7]}
    va0 = {'D_annual': 0.4, 'delta_per_seed': [0.3, 0.4, 0.5]}
    check('kosullar: hepsi saglaninca gecer', all(arm_conditions(tr0, va0, 0.01).values()))
    for key, tr, va, p in (('i_train_D_ge_delta_holm_p', {'D_annual': 0.2}, {}, 0.01),
                           ('i_train_D_ge_delta_holm_p', {}, {}, 0.06),
                           ('ii_train_min_k_gt_max_N', {'sharpe_a': [0.6, 1.1, 1.2]}, {}, 0.01),
                           ('iii_val_D_gt_0_and_2of3', {}, {'delta_per_seed': [0.9, -0.1, -0.1]}, 0.01)):
        cond = arm_conditions({**tr0, **tr}, {**va0, **va}, p)
        check(f'kosullar: yalniz {key} eler', [k for k, v in cond.items() if not v] == [key])
    r = stage1(build({'N-X': both(0.0), 'A-X': both(0.003), 'G-X': both(0.0)}))
    check('asama 1: yalniz A gecer, asama 2 kollari A-S + N-S',
          r['arms']['A']['label'] == 'ASAMA2' and r['arms']['G']['label'] == 'GOZLENMEDI'
          and r['stage2_arms'] == ['A-S', 'N-S'])
    r = stage1(build({'N-X': both(0.0), 'A-X': both(0.0), 'G-X': both(0.0)}))
    check('asama 1 bos: gecen kol yok, asama 2 bos', r['label'] == 'GECEN_KOL_YOK' and r['stage2_arms'] == [])
    r = stage1(build({'N-X': both(0.0), 'A-X': both(0.003)}))
    check('asama 1 eksik kol: EKSIK', r['label'] == 'EKSIK')
    r = stage1(build({'N-X': both(0.0), 'A-X': {'train': 0.003, 'val': -0.003}, 'G-X': both(0.0)}))
    check('asama 1 dogrulama celiskisi: KISMI', r['arms']['A']['label'] == 'KISMI' and r['stage2_arms'] == [])
    root = build({'N-X': both(0.0), 'A-X': both(0.003), 'G-X': both(0.003), 'A-S': both(0.0), 'G-S': both(0.005),
                  'N-S': both(0.0)})
    s1 = stage1(root)
    r = stage2(root, s1)
    check('asama 2: kontrolde avantaj yok -> GECTI; kontrolde de avantaj -> KISMI',
          r['arms']['A']['label'] == 'GECTI' and r['arms']['G']['label'] == 'KISMI')
    check('asama 2: baska on kaydin JSON u reddedilir', _raises(lambda: stage2(root, {**s1, 'preregistration': 'x'})))
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def _raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=(1, 2))
    ap.add_argument('--root', default=os.path.join('results', 'marl_m4a'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.stage is None:
        ap.error('--stage ya da --self-test gerekli')
    if a.stage == 1:
        res = stage1(a.root)
    else:
        with open(STAGE1_JSON, encoding='utf-8') as fh:
            res = stage2(a.root, json.load(fh))
    res['provenance'] = provenance()
    print(json.dumps({k: res.get(k) for k in ('label', 'stage2_arms')} |
                     {'arms': {k: v.get('label') for k, v in res['arms'].items()}}, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
