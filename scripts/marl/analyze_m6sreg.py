"""Faz 10 M6b kisa pilot okumasi (M6 tasarimi §6 ON KAYDI; ana venv). Sonuctan once yazildi; esikler degismez.

D = D(SR-N, R0-N): yillik Sharpe farki, fazla getiri, esli tohum (12, 13, 14), ortak HAC (`analyze_m3.pooled`).
  BOZULMA          : egitimde D < 0, iki yonlu p < 0,05, uc tohumun ucunde fark < 0, dogrulamada D < 0 ve >= 2/3 tohum < 0
  IYILESME_ISARETI : ayni kosullar pozitif yonde (kesifsel)
  FARK_YOK         : ikisi de degil
Eksik kol/tohum -> EKSIK. Betimsel: tohum basina Sharpe, devir, nakit agirligi, egitim kaydinin git izi.

    python scripts/marl/analyze_m6sreg.py --self-test
    python scripts/marl/analyze_m6sreg.py --out docs/development/measurements/phase10-m6/m6sreg_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from analyze_m3 import ALPHA  # noqa: E402
from analyze_m4b import pair  # noqa: E402
from analyze_m5 import need  # noqa: E402

SEEDS = (12, 13, 14)
PREREG = 'phase-10-m6-design.md §6 v0.5 (DONDURULDU, kosudan once)'
A, B = 'SR-N', 'R0-N'


def label(tr: dict, va: dict, k: int) -> str:
    for sign, name in ((-1, 'BOZULMA'), (1, 'IYILESME_ISARETI')):
        if (sign * tr['D_annual'] > 0 and tr['p_two_sided'] < ALPHA
                and all(sign * d > 0 for d in tr['delta_per_seed'])
                and sign * va['D_annual'] > 0 and sum(sign * d > 0 for d in va['delta_per_seed']) >= need(k)):
            return name
    return 'FARK_YOK'


def cash_weight(root, arm, window, seeds):
    out = []
    for s in seeds:
        d = np.load(os.path.join(root, arm, f's{s}', f'eval_{window}.npz'), allow_pickle=False)
        out.append(float(d['policy_cash_weight'].mean()))
    return out


def read(root, seeds=SEEDS) -> dict:
    tr, va = pair(root, A, B, 'train', seeds), pair(root, A, B, 'val', seeds)
    res = {'preregistration': PREREG, 'seeds': list(seeds), 'arms': [A, B], 'alpha': ALPHA}
    if tr is None or va is None:
        return {**res, 'label': 'EKSIK'}
    return {**res, 'label': label(tr, va, len(seeds)), 'train': tr, 'val': va,
            'cash_weight_betimsel': {w: {arm: cash_weight(root, arm, w, seeds) for arm in (A, B)} for w in ('train', 'val')}}


def train_records(root, seeds=SEEDS) -> dict:
    out = {}
    for arm in (A, B):
        for s in seeds:
            p = os.path.join(root, arm, f's{s}', 'train.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    r = json.load(fh)
                out[f'{arm}/s{s}'] = {'commit': r.get('commit'), 'clean': r.get('git_status_porcelain') == '',
                                      'extras_sha256': r.get('extras_sha256')}
    return out


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'), 'script_sha256': sha}


def self_test() -> bool:
    rng = np.random.default_rng(0)
    T, ok, total = 400, 0, 0
    dates = np.array([f'd{i:04d}' for i in range(T)])

    def check(name, good):
        nonlocal ok, total
        total += 1
        ok += bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name}")

    def build(edge_train, edge_val=None, per_seed=None):
        root = tempfile.mkdtemp(prefix='m6sreg_')
        for window, edge in (('train', edge_train), ('val', edge_train if edge_val is None else edge_val)):
            mkt = rng.normal(0.0003, 0.01, T)
            for arm in (A, B):
                for i, s in enumerate(SEEDS):
                    d = os.path.join(root, arm, f's{s}')
                    os.makedirs(d, exist_ok=True)
                    e = (per_seed[i] if per_seed is not None and window == 'train' else edge) if arm == A else 0.0
                    x = mkt + rng.normal(0, 0.003, T) + e
                    np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x, policy_date=dates,
                             policy_turnover=np.full(T, 0.3), policy_cash_weight=np.full(T, 0.1))
        return root

    check('etkisiz: FARK_YOK', read(build(0.0))['label'] == 'FARK_YOK')
    check('S-reg acikken her pencerede belirgin kotu: BOZULMA', read(build(-0.003))['label'] == 'BOZULMA')
    check('belirgin iyi: IYILESME_ISARETI', read(build(0.003))['label'] == 'IYILESME_ISARETI')
    check('egitimde kotu ama dogrulamada iyi: FARK_YOK', read(build(-0.003, 0.003))['label'] == 'FARK_YOK')
    check('egitimde bir tohum ters isaretli: FARK_YOK',
          read(build(-0.003, per_seed=[-0.004, -0.004, 0.002]))['label'] == 'FARK_YOK')
    root = build(0.0)
    shutil.rmtree(os.path.join(root, A, f's{SEEDS[0]}'))
    check('eksik tohum: EKSIK', read(root)['label'] == 'EKSIK')
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m6sreg'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res = {**read(a.root), 'provenance': provenance(), 'train_runs': train_records(a.root)}
    print(json.dumps({k: res[k] for k in ('label',)}, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
