"""Faz 10 M5 gercek sinyal kollari okuma betigi (ana venv) — M5 tasarimi §5 ON KAYDI (TASLAK); sonuctan ONCE yazildi.

Maliyetli X (20 bps), SR yillik; egitim 2013-06..2018-12, dogrulama 2019-2020; tohumlar 9, 10, 11. Yol y in {SO, SB}:
  birincil  D(gercek, karistirilmis) egitim, iki yonlu p, iki yol uzerinde Holm
  taban     N_y = |D(karistirilmis, karistirilmis-2)| egitim (iki bilgisiz kol arasi; M5 §4.11 dersi)
  ikincil   D(gercek, tahminsiz), D(karistirilmis, tahminsiz), her yolda; dort test uzerinde Holm
"Fark var" (A, B): (a) Holm p < 0,05, (b) |D(A, B)| > N_y, (c) dogrulamada ayni isaret ve en az m = ceil(2k/3) tohumda
ayni isaret. Okuma (oncelik sirasi; M5 §5.4 tablosu, §2.4 O1-O4):
  1 gercek<->karistirilmis fark var, D < 0                         -> O2_bozma
  2 ... D > 0 ve gercek<->tahminsiz fark var, D > 0               -> O4_kesifsel
  3 ... D > 0, O4 kosullari eksik                                  -> O4_eksik
  4 gercek<->karistirilmis fark yok; gercek<->tahminsiz ve karistirilmis<->tahminsiz fark var, ayni isaret -> O3_girdi
  5 gercek<->karistirilmis fark yok; gercek<->tahminsiz fark var, D < 0 -> O2_tahminsize_gore
  6 hicbiri                                                        -> O1_fark_yok
Eksik kol -> EKSIK.

  python scripts/marl/analyze_m5real.py --out docs/development/measurements/phase10-m5/m5real_results.json
  python scripts/marl/analyze_m5real.py --self-test
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
from analyze_m3 import ALPHA  # noqa: E402
from analyze_m3b import holm, provenance  # noqa: E402
from analyze_m4b import pair  # noqa: E402
from analyze_m5 import need  # noqa: E402

SEEDS = (9, 10, 11)
PREREG = 'phase-10-m5-design.md §5 v0.9 (TASLAK)'
PATHS = {'SO': {'real': 'SO', 'shuf': 'SO-S', 'shuf2': 'SO-S2', 't0': 'T0-N'},
         'SB': {'real': 'SB', 'shuf': 'SB-S', 'shuf2': 'SB-S2', 't0': 'T0-G'}}
COMPARISONS = {'real_shuf': ('real', 'shuf'), 'shuf_shuf2': ('shuf', 'shuf2'), 'real_t0': ('real', 't0'),
               'shuf_t0': ('shuf', 't0')}
ARMS = sorted({a for p in PATHS.values() for a in p.values()})


def read(root, seeds=SEEDS) -> dict:
    k = len(seeds)
    res = {'preregistration': PREREG, 'seeds': list(seeds), 'm_of_k': [need(k), k], 'alpha': ALPHA}
    P = {f'{y}.{c}.{w}': pair(root, arms[a], arms[b], w, seeds)
         for y, arms in PATHS.items() for c, (a, b) in COMPARISONS.items() for w in ('train', 'val')}
    res['pairs'] = P
    missing = [n for n, v in P.items() if v is None]
    if missing:
        return {**res, 'reading': {y: 'EKSIK' for y in PATHS}, 'missing': missing}
    prim = holm([P[f'{y}.real_shuf.train']['p_two_sided'] for y in PATHS])
    sec_keys = [(y, c) for y in PATHS for c in ('real_t0', 'shuf_t0')]
    sec = holm([P[f'{y}.{c}.train']['p_two_sided'] for y, c in sec_keys])
    padj = {**{(y, 'real_shuf'): p for y, p in zip(PATHS, prim)}, **dict(zip(sec_keys, sec))}
    out, reading = {}, {}
    for y in PATHS:
        floor = abs(P[f'{y}.shuf_shuf2.train']['D_annual'])

        def diff(c):
            tr, va = P[f'{y}.{c}.train'], P[f'{y}.{c}.val']
            s = float(np.sign(tr['D_annual']))
            cond = {'a_holm_p': bool(padj[(y, c)] < ALPHA), 'b_gt_noise_floor': bool(abs(tr['D_annual']) > floor),
                    'c_val_same_sign_m_of_k': bool(s != 0 and va['D_annual'] * s > 0
                                                   and sum(d * s > 0 for d in va['delta_per_seed']) >= need(k))}
            return {'D_train': tr['D_annual'], 'D_val': va['D_annual'], 'p_two_sided': tr['p_two_sided'],
                    'p_holm': padj[(y, c)], 'sign': s, 'conditions': cond, 'diff': all(cond.values())}

        rs, rt, st = diff('real_shuf'), diff('real_t0'), diff('shuf_t0')
        if rs['diff'] and rs['sign'] < 0:
            key = 'O2_bozma'
        elif rs['diff'] and rt['diff'] and rt['sign'] > 0:
            key = 'O4_kesifsel'
        elif rs['diff']:
            key = 'O4_eksik'
        elif rt['diff'] and st['diff'] and rt['sign'] == st['sign']:
            key = 'O3_girdi'
        elif rt['diff'] and rt['sign'] < 0:
            key = 'O2_tahminsize_gore'
        else:
            key = 'O1_fark_yok'
        se = P[f'{y}.real_shuf.train']['se_annual']
        out[y] = {'noise_floor': floor, 'real_shuf': rs, 'real_t0': rt, 'shuf_t0': st,
                  'min_detectable_D_betimsel': (1.96 + 0.84) * se}
        reading[y] = key
    return {**res, 'paths': out, 'reading': reading, 'betas': _betas(root, seeds)}


def _betas(root, seeds):
    b = {}
    for arm in ('SB', 'SB-S', 'SB-S2'):
        vals = []
        for s in seeds:
            p = os.path.join(root, arm, f's{s}', 'train.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    vals.append(((json.load(fh).get('conf_scalars_final') or {}).get('beta')))
            else:
                vals.append(None)
        b[arm] = vals
    return b


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
        """edges[arm] = kenar (float) ya da baska kolun adi (birebir kopya); verilmeyen kol 0 kenar alir."""
        root = tempfile.mkdtemp(prefix='m5real_')
        full = {a: edges.get(a, 0.0) for a in ARMS}
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            series = {}
            for arm, e in sorted(full.items(), key=lambda kv: isinstance(kv[1], str)):
                for s in SEEDS:
                    d = os.path.join(root, arm, f's{s}')
                    os.makedirs(d, exist_ok=True)
                    x = series[(e, s)] if isinstance(e, str) else mkt + rng.normal(0, 0.003, T) + e
                    series[(arm, s)] = x
                    np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x, policy_date=dates,
                             policy_turnover=np.zeros(T), policy_cash_weight=np.zeros(T))
        return root

    r = read(build({}))
    check('bos: iki yolda O1_fark_yok', r['reading'] == {'SO': 'O1_fark_yok', 'SB': 'O1_fark_yok'})
    r = read(build({'SO': -0.003}))
    check('gercek sinyal SO yolunu bozuyor: O2_bozma; SB O1', r['reading'] == {'SO': 'O2_bozma', 'SB': 'O1_fark_yok'})
    r = read(build({'SB': 0.003}))
    check('gercek > karistirilmis ve > tahminsiz: O4_kesifsel', r['reading']['SB'] == 'O4_kesifsel')
    r = read(build({'SB': 0.003, 'T0-G': 'SB'}))
    check('gercek > karistirilmis ama tahminsiz de ayni: O4_eksik', r['reading']['SB'] == 'O4_eksik')
    r = read(build({'SO': 0.003, 'SO-S': 'SO', 'SO-S2': 'SO'}))
    check('gercek = karistirilmis, ikisi de tahminsizden iyi: O3_girdi', r['reading']['SO'] == 'O3_girdi')
    r = read(build({'SO': 0.002, 'SO-S2': 0.004}))
    check('fark gurultu tabanindan kucuk: (b) eler, O1', r['reading']['SO'] == 'O1_fark_yok'
          and not r['paths']['SO']['real_shuf']['conditions']['b_gt_noise_floor'])
    root = build({})
    import shutil
    shutil.rmtree(os.path.join(root, 'SB-S2'))
    r = read(root)
    check('eksik kol: EKSIK', r['reading'] == {'SO': 'EKSIK', 'SB': 'EKSIK'} and r['missing'])
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m5real'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res = {**read(a.root), 'provenance': provenance(), 'train_runs': train_records_all(a.root)}
    print(json.dumps({'reading': res['reading'], 'betas': res.get('betas')}, indent=1, ensure_ascii=False, default=float))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


def train_records_all(root, seeds=SEEDS) -> dict:
    """Egitim kaydi izi (commit, temiz agac) tum kollar icin."""
    out = {}
    for arm in ARMS:
        for s in seeds:
            p = os.path.join(root, arm, f's{s}', 'train.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    r = json.load(fh)
                out[f'{arm}/s{s}'] = {'commit': r.get('commit'), 'clean': r.get('git_status_porcelain') == ''}
    return out


if __name__ == '__main__':
    main()
