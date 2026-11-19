"""Faz 10 M4a okuma betigi (ana venv) — M4 tasarimi §3 ON KAYDI'ndaki kural; sonuctan ONCE yazildi.

Maliyetsiz fazla getiri X, SR yillik. Girdi: <root>/<kol>/s<k>/eval_<pencere>.npz. Kollar N-X, A-X, G-X (asama 1),
A-S / G-S + N-S (asama 2). Her iletisim kolu k in {A, G} icin D_k = ort_s [SR(k-X_s) - SR(N-X_s)].

ASAMA 1. Kol k icin uc kosul (DELTA = 0,25, kullanici karari K-M4-4):
  (i)   egitim: D_k >= DELTA VE ortak HAC tek yonlu p, iki kol uzerinde Holm duzeltmeli, < 0,05
  (ii)  egitim: min_s SR(k-X_s) > max_s SR(N-X_s)
  (iii) dogrulama: D_k > 0 VE en az 2/3 tohumda Delta > 0
(i)-(iii)'un hepsi -> kol asama 2'ye (`stage2_arms`: k-S ve N-S). KISMI = (i)'de D_k > 0 ve Holm p < 0,05 var ama uc
kosulun hepsi degil (delta'ya ulasilamamasi dahil). Aksi GOZLENMEDI.

ASAMA 1b (lr duyarliligi; hakem-rl M4a #1). Asama 1'de gecmeyen kol k icin, durma kararindan ONCE: k-X ve N-X lr = 2e-4
ile (`<root>/lr2e-4/`). Ayni (i)-(iii); Holm ailesi = asama 1 + 1b'deki TUM D_k testleri (en cok 4). Asama 1 kararlari
degismez. 1b'de gecen kol asama 2'ye o lr ile gider. 1b'den sonra da gecmeyen kol icin karar kullaniciya doner.

ASAMA 2 (kontrol). Gecen kol k icin (kolun gectigi lr ile):
  (iv)  egitim: G = D_k - D_k^S >= DELTA/2 VE z = G / sqrt(se_X^2 + se_S^2) >= 1,645 (tek yonlu %5)
        D_k^S = ort_s [SR(k-S_s) - SR(N-S_s)]; se'ler ortak HAC'tan. X ve S kosulari ayri bloklarla oldugundan
        bagimsizlik varsayilir; ortak piyasa gunleri pozitif korelasyon verir -> varsayim SE'yi buyutur (tutucu).
(iv) saglanirsa k GECTI; saglanmazsa KISMI (iletisim avantaji tasinan bilgiye baglanamadi). Betimsel: tohum basina
G_s = Delta_X,s - Delta_S,s (`gap_per_seed`; HAC SE tohum varyansini icermez, z yaklasiktir).
gat GECTI ve attn GECTI degilse: attn sonucu "bu butcede gondericiyi tanimlayamadi" diye okunur (sigma grafa hizali);
attn'in M5'te kosulup kosulmayacagi kullanici karari.

BETIMSEL (kararlari degistirmez): kendi-sinyalli tavan = M3a A1 (ayni bloklar, ayni hiperparametre, maliyetsiz;
`measurements/phase10-m3/m3a_results.json`); ayni maliyetsiz 1/N ve al-tut SR'si.

  python scripts/marl/analyze_m4a.py --stage 1 --out docs/development/measurements/phase10-m4/m4a_stage1.json
  python scripts/marl/analyze_m4a.py --stage 1b --out docs/development/measurements/phase10-m4/m4a_stage1b.json
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
Z_ONE_SIDED = 1.645
COMM_ARMS = ('A', 'G')
CTRL = 'N'
LR_ALT = 'lr2e-4'
PREREG = 'phase-10-m4-design.md §3 v0.3'
MEAS = os.path.join('docs', 'development', 'measurements', 'phase10-m4')
STAGE1_JSON = os.path.join(MEAS, 'm4a_stage1.json')
STAGE1B_JSON = os.path.join(MEAS, 'm4a_stage1b.json')
M3A_JSON = os.path.join('docs', 'development', 'measurements', 'phase10-m3', 'm3a_results.json')


def ceiling():
    """Betimsel tavan: M3a A1 (kendi sinyali, none, ayni hiperparametre, maliyetsiz)."""
    if not os.path.exists(M3A_JSON):
        return None
    with open(M3A_JSON, encoding='utf-8') as fh:
        d = json.load(fh)
    return {w: {'sharpe_A1_own_signal': d[w].get('sharpe_A1'), 'D_A1_minus_S1': d[w].get('D_annual')}
            for w in ('train', 'val') if w in d}


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


def _read_arm(tr, va, p_holm) -> dict:
    cond = arm_conditions(tr, va, p_holm)
    passed = all(cond.values())
    label = 'ASAMA2' if passed else ('KISMI' if tr['D_annual'] > 0 and p_holm < ALPHA else 'GOZLENMEDI')
    return {'train': tr, 'val': va, 'p_holm': p_holm, 'conditions': cond, 'label': label}


def _pairs(root, arms):
    out = {}
    for k in arms:
        tr, va = pair(root, f'{k}-X', f'{CTRL}-X', 'train'), pair(root, f'{k}-X', f'{CTRL}-X', 'val')
        if tr is None or va is None:
            return None, f'{k}-X'
        out[k] = (tr, va)
    return out, None


def _s_arms(passed: list, prefix: str = '') -> list:
    return ([f'{prefix}{k}-S' for k in passed] + [f'{prefix}{CTRL}-S']) if passed else []


def stage1(root) -> dict:
    res = {'preregistration': PREREG, 'stage': 1, 'delta': DELTA, 'arms': {}}
    pairs, missing = _pairs(root, COMM_ARMS)
    if pairs is None:
        return {**res, 'label': 'EKSIK', 'missing': missing, 'stage2_arms': [], 'stage1b_arms': []}
    adj = holm([pairs[k][0]['p_one_sided'] for k in COMM_ARMS])
    for k, p_holm in zip(COMM_ARMS, adj):
        res['arms'][k] = {**_read_arm(*pairs[k], p_holm), 'lr': 'base'}
    passed = [k for k in COMM_ARMS if res['arms'][k]['label'] == 'ASAMA2']
    failed = [k for k in COMM_ARMS if k not in passed]
    res['baseline_sharpe'] = {w: baselines(root, w) for w in ('train', 'val')}
    res['ceiling_m3a_A1'] = ceiling()
    res['stage2_arms'] = _s_arms(passed)
    res['stage1b_arms'] = ([f'{LR_ALT}/{k}-X' for k in failed] + [f'{LR_ALT}/{CTRL}-X']) if failed else []
    res['label'] = 'ASAMA2' if passed else 'GECEN_KOL_YOK'
    return res


def stage1b(root, s1: dict) -> dict:
    """Asama 1'de gecmeyen kollar lr 2e-4'te; Holm ailesi asama 1 + 1b'nin tum D_k testleri."""
    if s1.get('preregistration') != PREREG or s1.get('stage') != 1:
        raise SystemExit('asama 1 JSON bu on kayda ait degil')
    failed = [k for k in COMM_ARMS if s1['arms'][k]['label'] != 'ASAMA2']
    res = {'preregistration': PREREG, 'stage': '1b', 'delta': DELTA, 'lr': LR_ALT, 'arms': {}}
    if not failed:
        return {**res, 'label': 'GEREKMEDI', 'stage2_arms': []}
    pairs, missing = _pairs(os.path.join(root, LR_ALT), failed)
    if pairs is None:
        return {**res, 'label': 'EKSIK', 'missing': f'{LR_ALT}/{missing}', 'stage2_arms': []}
    fam = [s1['arms'][k]['train']['p_one_sided'] for k in COMM_ARMS] + [pairs[k][0]['p_one_sided'] for k in failed]
    adj = holm(fam)[len(COMM_ARMS):]
    for k, p_holm in zip(failed, adj):
        res['arms'][k] = {**_read_arm(*pairs[k], p_holm), 'lr': LR_ALT, 'holm_family_size': len(fam)}
    passed = [k for k in failed if res['arms'][k]['label'] == 'ASAMA2']
    res['stage2_arms'] = _s_arms(passed, f'{LR_ALT}/')
    res['label'] = 'ASAMA2' if passed else 'GECEN_KOL_YOK'
    return res


def control_condition(x: dict, s: dict) -> dict:
    """(iv): G = D_X - D_S >= DELTA/2 ve z = G / sqrt(se_X^2 + se_S^2) >= 1,645 (bagimsizlik varsayimi, tutucu)."""
    gap = float(x['D_annual'] - s['D_annual'])
    se = float(np.hypot(x['se_annual'], s['se_annual']))
    z = gap / se if se > 0 else float('inf')
    return {'gap': gap, 'se_gap': se, 'z': z, 'iv_gap_ge_half_delta_and_z': bool(gap >= DELTA / 2 and z >= Z_ONE_SIDED)}


def stage2(root, s1: dict, s1b=None) -> dict:
    for rec, stage in ((s1, 1), (s1b, '1b')):
        if rec is not None and (rec.get('preregistration') != PREREG or rec.get('stage') != stage):
            raise SystemExit(f'asama {stage} JSON bu on kayda ait degil')
    res = {'preregistration': PREREG, 'stage': 2, 'delta': DELTA, 'arms': {}}
    for k in COMM_ARMS:
        src = s1['arms'].get(k, {})
        if src.get('label') != 'ASAMA2' and s1b is not None:
            src = s1b['arms'].get(k, src)
        if src.get('label') != 'ASAMA2':
            res['arms'][k] = {'label': src.get('label'), 'lr': src.get('lr'), 'note': 'asama 2 kosulmadi'}
            continue
        sub_root = root if src['lr'] == 'base' else os.path.join(root, src['lr'])
        tr = pair(sub_root, f'{k}-S', f'{CTRL}-S', 'train')
        if tr is None:
            res['arms'][k] = {'label': 'EKSIK', 'lr': src['lr'], 'missing': f'{k}-S / {CTRL}-S'}
            continue
        cond = control_condition(src['train'], tr)
        cond['gap_per_seed'] = [float(x - s) for x, s in zip(src['train']['delta_per_seed'], tr['delta_per_seed'])]
        res['arms'][k] = {'train_S': tr, 'D_X_train': src['train']['D_annual'], 'lr': src['lr'], 'conditions': cond,
                          'label': 'GECTI' if cond['iv_gap_ge_half_delta_and_z'] else 'KISMI'}
    labels = {k: v['label'] for k, v in res['arms'].items()}
    if labels.get('G') == 'GECTI' and labels.get('A') != 'GECTI':
        res['note'] = ('gat GECTI, attn degil: boru hatti calisiyor; attn sonucu "bu butcede gondericiyi tanimlayamadi" '
                       'diye okunur (sigma grafa hizali). attn M5te kosulup kosulmayacagi kullanici karari (§3.3)')
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

    def fill(root, edges):
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            for arm, e in edges.items():
                for k in SEEDS:
                    write(root, arm, k, window, mkt + rng.normal(0, 0.003, T) + e.get(window, 0.0))
        return root

    def build(edges):
        return fill(tempfile.mkdtemp(prefix='m4a_'), edges)

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
    check('asama 1: yalniz A gecer; asama 2 A-S + N-S; 1b lr2e-4/G-X + lr2e-4/N-X',
          r['arms']['A']['label'] == 'ASAMA2' and r['arms']['G']['label'] == 'GOZLENMEDI'
          and r['stage2_arms'] == ['A-S', 'N-S'] and r['stage1b_arms'] == ['lr2e-4/G-X', 'lr2e-4/N-X'])
    r = stage1(build({'N-X': both(0.0), 'A-X': both(0.0), 'G-X': both(0.0)}))
    check('asama 1 bos: gecen kol yok, asama 2 bos', r['label'] == 'GECEN_KOL_YOK' and r['stage2_arms'] == [])
    r = stage1(build({'N-X': both(0.0), 'A-X': both(0.003)}))
    check('asama 1 eksik kol: EKSIK', r['label'] == 'EKSIK')
    r = stage1(build({'N-X': both(0.0), 'A-X': {'train': 0.003, 'val': -0.003}, 'G-X': both(0.0)}))
    check('asama 1 dogrulama celiskisi: KISMI', r['arms']['A']['label'] == 'KISMI' and r['stage2_arms'] == [])
    # asama 1b
    root = build({'N-X': both(0.0), 'A-X': both(0.0), 'G-X': both(0.0)})
    s1 = stage1(root)
    fill(os.path.join(root, LR_ALT), {'N-X': both(0.0), 'A-X': both(0.0), 'G-X': both(0.003)})
    r = stage1b(root, s1)
    check('asama 1b: lr2e-4te gecen G asama 2ye o lr ile, Holm ailesi 4',
          r['arms']['G']['label'] == 'ASAMA2' and r['arms']['G']['holm_family_size'] == 4
          and r['arms']['A']['label'] != 'ASAMA2' and r['stage2_arms'] == ['lr2e-4/G-S', 'lr2e-4/N-S'])
    s1_a = {**s1, 'arms': {'A': {**s1['arms']['A'], 'label': 'ASAMA2'}, 'G': s1['arms']['G']}}
    check('asama 1b: yalniz asama 1de gecmeyen kollar', set(stage1b(root, s1_a)['arms']) == {'G'})
    fill(os.path.join(root, LR_ALT), {'G-S': both(0.0), 'N-S': both(0.0)})
    r = stage2(root, s1, stage1b(root, s1))
    check('asama 2: 1bde gecen kol kendi lr sinda okunur', r['arms']['G']['lr'] == LR_ALT
          and r['arms']['G']['label'] == 'GECTI')
    # (iv)
    x0 = {'D_annual': 0.5, 'se_annual': 0.09}
    check('(iv): fark 0,4, z 3,1 -> saglanir',
          control_condition(x0, {'D_annual': 0.1, 'se_annual': 0.09})['iv_gap_ge_half_delta_and_z'])
    check('(iv): fark 0,15 >= DELTA/2 ama z 1,2 -> saglanmaz',
          not control_condition(x0, {'D_annual': 0.35, 'se_annual': 0.09})['iv_gap_ge_half_delta_and_z'])
    # asama 2
    root = build({'N-X': both(0.0), 'A-X': both(0.003), 'G-X': both(0.003), 'A-S': both(0.0), 'G-S': both(0.005),
                  'N-S': both(0.0)})
    s1 = stage1(root)
    r = stage2(root, s1)
    check('asama 2: kontrolde avantaj yok -> GECTI; kontrolde de avantaj -> KISMI',
          r['arms']['A']['label'] == 'GECTI' and r['arms']['G']['label'] == 'KISMI')
    check('asama 2: baska on kaydin JSON u reddedilir', _raises(lambda: stage2(root, {**s1, 'preregistration': 'x'})))
    root = build({'N-X': both(0.0), 'A-X': both(0.0), 'G-X': both(0.003), 'G-S': both(0.0), 'N-S': both(0.0)})
    r = stage2(root, stage1(root))
    check('asama 2: gat GECTI, attn degil -> tanimlama notu', r['arms']['G']['label'] == 'GECTI' and 'note' in r)
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
    ap.add_argument('--stage', choices=('1', '1b', '2'))
    ap.add_argument('--root', default=os.path.join('results', 'marl_m4a'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.stage is None:
        ap.error('--stage ya da --self-test gerekli')
    if a.stage == '1':
        res = stage1(a.root)
    else:
        with open(STAGE1_JSON, encoding='utf-8') as fh:
            s1 = json.load(fh)
        if a.stage == '1b':
            res = stage1b(a.root, s1)
        else:
            s1b = None
            if os.path.exists(STAGE1B_JSON):
                with open(STAGE1B_JSON, encoding='utf-8') as fh:
                    s1b = json.load(fh)
            res = stage2(a.root, s1, s1b)
    res['provenance'] = provenance()
    print(json.dumps({k: res.get(k) for k in ('label', 'stage1b_arms', 'stage2_arms')} |
                     {'arms': {k: v.get('label') for k, v in res['arms'].items()}}, indent=1, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
