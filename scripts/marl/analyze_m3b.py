"""Faz 10 M3b okuma betigi (ana venv) — M3 tasarim §6 ON KAYDI v1.0'daki kural; sonuctan ONCE yazildi.

Tum seriler 20 bps gidis-donus maliyet sonrasi fazla getiri X. SR yillik. Girdi: <root>/<kol>/s<k>/eval_<pencere>.npz.

ASAMA 1 (A1c PC060-K12 <-> S1c karistirilmis PC). GECTI = uc kosulun HEPSI (M3a kurali, ayni esikler):
  (i)   egitim: D >= 0,3 VE ortak HAC tek yonlu p < 0,05
  (ii)  egitim: min_s SR(A1c_s) > max_s SR(S1c_s)
  (iii) dogrulama isaret kontrolu: D > 0 VE en az 2/3 tohumda Delta > 0
KISMI = (i)'de D > 0 ve p < 0,05 var ama GECTI degil. Aksi GOZLENMEDI. GECTI degilse asama 2 kosulmaz.

ASAMA 2 (referans A1c; adaylar IPPO, STATE, A05; A00 yalniz ablasyon, SECILEMEZ: plan §2.3 alfa > 0).
Aday UYGUN = dorduu birden (D = aday - referans):
  (a) dogrulama: D >= 0,3
  (b) dogrulama: ortak HAC tek yonlu p, 3 aday uzerinde Holm duzeltmeli, < 0,05
  (c) dogrulama: en az 2/3 tohumda Delta > 0
  (d) egitim: D > 0 (isaret tutarliligi)
Secilen = uygun adaylar icinde en buyuk dogrulama D'si; uygun aday yoksa referans kalir. Etkenler birlestirilmez.

  python scripts/marl/analyze_m3b.py --stage 1 --out docs/development/measurements/phase10-m3/m3b_stage1.json
  python scripts/marl/analyze_m3b.py --stage 2 --out docs/development/measurements/phase10-m3/m3b_stage2.json
  python scripts/marl/analyze_m3b.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_m3 import ALPHA, D_MIN, SEEDS, load_arm, pooled, sr  # noqa: E402

REF, CTRL = 'A1c', 'S1c'
SELECTABLE = ('IPPO', 'STATE', 'A05')
ABLATION = ('A00',)
PREREG = 'phase-10-m3-design.md §6 v1.0'


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def holm(pvals) -> list:
    """Holm-Bonferroni duzeltilmis p'ler (girdi sirasinda)."""
    p = np.asarray(pvals, dtype=float)
    m, order, out, running = len(p), np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return [float(x) for x in out]


def pair(root, arm_a, arm_b, window):
    """arm_a - arm_b, tohum eslesik. Eksik tohum varsa None (eksik kol okunmaz)."""
    a, b = load_arm(root, arm_a, window), load_arm(root, arm_b, window)
    if any(k not in a or k not in b for k in SEEDS):
        return None
    for k in SEEDS:
        if not np.array_equal(a[k]['date'], b[k]['date']):
            raise SystemExit(f'{window}: {arm_a}/{arm_b} tohum {k} tarihleri eslesmiyor')
    sa, sb = [sr(a[k]['X']) for k in SEEDS], [sr(b[k]['X']) for k in SEEDS]
    return {**pooled([(a[k]['X'], b[k]['X']) for k in SEEDS]), 'sharpe_a': sa, 'sharpe_b': sb,
            'delta_per_seed': [x - y for x, y in zip(sa, sb)],
            'turnover_mean': {arm_a: float(np.mean([a[k]['turnover'].mean() for k in SEEDS])),
                              arm_b: float(np.mean([b[k]['turnover'].mean() for k in SEEDS]))},
            'cash_mean': {arm_a: float(np.mean([a[k]['cash'].mean() for k in SEEDS])),
                          arm_b: float(np.mean([b[k]['cash'].mean() for k in SEEDS]))}}


def baselines(root, window):
    path = os.path.join(root, f'baselines_{window}_20.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return {kind: sr(d[f'{kind}_X']) for kind in ('ew', 'bh')}


def stage1(root) -> dict:
    tr, va = pair(root, REF, CTRL, 'train'), pair(root, REF, CTRL, 'val')
    res = {'preregistration': PREREG, 'stage': 1, 'train': tr, 'val': va}
    if tr is None or va is None:
        return {**res, 'label': 'EKSIK', 'gate_passed': False}
    c1 = tr['D_annual'] >= D_MIN and tr['p_one_sided'] < ALPHA
    c2 = min(tr['sharpe_a']) > max(tr['sharpe_b'])
    c3 = va['D_annual'] > 0 and sum(d > 0 for d in va['delta_per_seed']) >= 2
    label = 'GECTI' if c1 and c2 and c3 else ('KISMI' if tr['D_annual'] > 0 and tr['p_one_sided'] < ALPHA else 'GOZLENMEDI')
    return {**res, 'conditions': {'i_train_D_ge_0.3_p_lt_0.05': bool(c1), 'ii_train_min_A1c_gt_max_S1c': bool(c2),
                                  'iii_val_D_gt_0_and_2of3': bool(c3)},
            'baseline_sharpe': {w: baselines(root, w) for w in ('train', 'val')},
            'label': label, 'gate_passed': label == 'GECTI'}


def stage2(root) -> dict:
    arms = {}
    for arm in SELECTABLE + ABLATION:
        tr, va = pair(root, arm, REF, 'train'), pair(root, arm, REF, 'val')
        if tr is None or va is None:
            return {'preregistration': PREREG, 'stage': 2, 'label': 'EKSIK', 'missing': arm, 'selected': None}
        arms[arm] = {'train': tr, 'val': va}
    adj = holm([arms[a]['val']['p_one_sided'] for a in SELECTABLE])
    eligible = []
    for arm, p_holm in zip(SELECTABLE, adj):
        tr, va = arms[arm]['train'], arms[arm]['val']
        cond = {'a_val_D_ge_0.3': bool(va['D_annual'] >= D_MIN), 'b_val_holm_p_lt_0.05': bool(p_holm < ALPHA),
                'c_val_2of3_seeds': bool(sum(d > 0 for d in va['delta_per_seed']) >= 2),
                'd_train_D_gt_0': bool(tr['D_annual'] > 0)}
        arms[arm].update(p_holm=p_holm, conditions=cond, eligible=all(cond.values()))
        if all(cond.values()):
            eligible.append(arm)
    for arm in ABLATION:
        arms[arm].update(eligible=False, note='alfa = 0 ablasyon; secilemez (plan §2.3)')
    selected = max(eligible, key=lambda x: arms[x]['val']['D_annual']) if eligible else REF
    return {'preregistration': PREREG, 'stage': 2, 'reference': REF, 'arms': arms, 'eligible': eligible,
            'selected': selected, 'label': 'ADAY_SECILDI' if eligible else 'REFERANS_KALDI'}


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
        """edges: {kol: {pencere: gunluk fark}}; tum kollar ayni piyasa + kendi gurultusu."""
        root = tempfile.mkdtemp(prefix='m3b_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            for arm, e in edges.items():
                for k in SEEDS:
                    write(root, arm, k, window, mkt + rng.normal(0, 0.003, T) + e.get(window, 0.0))
        return root

    both = lambda v: {'train': v, 'val': v}   # noqa: E731
    check('holm bilinen deger', np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
    check('asama 1 sinyal: GECTI', stage1(build({REF: both(0.003), CTRL: both(0.0)}))['label'] == 'GECTI')
    check('asama 1 bos: GOZLENMEDI', stage1(build({REF: both(0.0), CTRL: both(0.0)}))['label'] == 'GOZLENMEDI')
    check('asama 1 eksik kol: EKSIK', stage1(build({REF: both(0.003)}))['label'] == 'EKSIK')
    r = stage1(build({REF: {'train': 0.003, 'val': -0.003}, CTRL: both(0.0)}))
    check('asama 1 dogrulama celiskisi: KISMI', r['label'] == 'KISMI' and not r['gate_passed'])
    r = stage2(build({REF: both(0.0), 'IPPO': both(0.003), 'STATE': both(0.0), 'A00': both(0.006),
                      'A05': {'train': -0.003, 'val': 0.004}}))
    check('asama 2: IPPO secilir', r['selected'] == 'IPPO')
    check('asama 2: alfa 0 en buyuk olsa da secilemez', not r['arms']['A00']['eligible'])
    check('asama 2: egitim isareti ters aday elenir', not r['arms']['A05']['eligible'])
    r = stage2(build({arm: both(0.0) for arm in (REF, *SELECTABLE, *ABLATION)}))
    check('asama 2 bos: referans kalir', r['selected'] == REF and r['label'] == 'REFERANS_KALDI')
    r = stage2(build({REF: both(0.0), 'IPPO': both(0.003)}))
    check('asama 2 eksik kol: EKSIK, secim yok', r['label'] == 'EKSIK' and r['selected'] is None)
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, choices=(1, 2))
    ap.add_argument('--root', default=os.path.join('results', 'marl_m3b'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.stage is None:
        ap.error('--stage ya da --self-test gerekli')
    res = {**(stage1(a.root) if a.stage == 1 else stage2(a.root)), 'provenance': provenance()}
    print(json.dumps({k: res.get(k) for k in ('label', 'conditions', 'eligible', 'selected')}, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
