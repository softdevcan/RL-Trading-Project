"""Faz 10 M5 gercek sinyal kollari okuma betigi (ana venv) — M5 tasarimi §5 ON KAYDI; sonuctan ONCE yazildi.
v0.10: hakem-finans + hakem-rl 1. okuma (Kucuk rev.) islendi: gurultu tabani tek cekim |D| esigi yerine tohum basina
siralama olcutu; tahminsiz karsilastirmalari tabansiz ikincil; boyut tanisi; brut SR betimsel.

Maliyetli X (20 bps), SR yillik; egitim 2013-06..2018-12, dogrulama 2019-2020; tohumlar 9, 10, 11; k tohum, m = ceil(2k/3).
Yol y in {SO, SB}; kollar gercek (R), karistirilmis (S), karistirilmis-2 (S2), tahminsiz (T0).

Birincil (R <-> S), "fark var" = uc kosul birden:
  (a) egitim D(R, S) iki yonlu ortak HAC p, iki yol uzerinde Holm, < 0,05
  (b) SIRALAMA TABANI: her tohumda SR(R) iki karistirilmis kolun ikisinden de buyuk (ya da ikisinden de kucuk), egitimde.
      Bilgi yokken R, S, S2 degistirilebilir sayilirsa (ayni mimari, ayni girdi boyutu, ayni tohumda ayni baslangic)
      tohum basina olasilik 1/3, k = 3'te yon basina (1/3)^3 ~ 0,037, IKI YONLU ~ 0,074/yol. ALT SINIR: gercek blok
      tohumlar arasinda ortak (siralamalar bagimsiz degil); 'yon bilgisi yok VE hiza etkisi yok' varsayimi (§5.6).
  (c) dogrulamada D(R, S) ayni isaretli ve en az m tohumda ayni isaret
Ikincil (R <-> T0, S <-> T0; dort test Holm): TABANI YOK (girdi boyutu ve SB yolunda GL mimarisi farkli; baslangic
  agirliklari eslesmez). "fark var" = (a') Holm p < 0,05, (b') egitimde tum tohumlarda ayni isaret, (c) dogrulama.
  Bu karsilastirmalara dayanan okumalar (O4_kesifsel'in ikinci kosulu, O3_girdi, O2_tahminsize_gore) "ikincil, karisik".
Okuma (oncelik; M5 §5.4, §2.4 O1-O4):
  1 R<->S fark var, D < 0                                   -> O2_bozma
  2 ... D > 0 ve R<->T0 fark var, D > 0                     -> O4_kesifsel
  3 ... D > 0, O4 kosullari eksik                           -> O4_eksik
  4 R<->S fark yok; R<->T0 ve S<->T0 fark var, ayni isaret  -> O3_girdi (isaret negatifse: ek girdi ogrenmeyi bozuyor)
  5 R<->S fark yok; R<->T0 fark var, D < 0                  -> O2_tahminsize_gore
  6 hicbiri                                                 -> O1_fark_yok (R<->T0 pozitif ya da zit isaretli ikincil
                                                               farklar O1'e duser; `secondary_note` ile betimsel)
Boyut tanisi (betimsel): D(S, S2) ortak HAC p < 0,05 ise ortak HAC bu duzende kalibre degil isareti (iki bilgisiz kol).
Eksik kol -> EKSIK.

  python scripts/marl/analyze_m5real.py --out docs/development/measurements/phase10-m5/m5real_results.json
  python scripts/marl/analyze_m5real.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_m3 import ALPHA, sr  # noqa: E402
from analyze_m3b import holm, provenance  # noqa: E402
from analyze_m4b import load, pair  # noqa: E402
from analyze_m5 import need  # noqa: E402

SEEDS = (9, 10, 11)
PREREG = 'phase-10-m5-design.md §5 v0.11 (okuma kurali DONDURULDU)'
COST_ONE_WAY = 20 / 2 / 1e4
PATHS = {'SO': {'real': 'SO', 'shuf': 'SO-S', 'shuf2': 'SO-S2', 't0': 'T0-N'},
         'SB': {'real': 'SB', 'shuf': 'SB-S', 'shuf2': 'SB-S2', 't0': 'T0-G'}}
COMPARISONS = {'real_shuf': ('real', 'shuf'), 'shuf_shuf2': ('shuf', 'shuf2'), 'real_t0': ('real', 't0'),
               'shuf_t0': ('shuf', 't0')}
ARMS = sorted({a for p in PATHS.values() for a in p.values()})


def rank_floor(root, arms, seeds) -> dict:
    """(b): her tohumda SR(R) > max(SR(S), SR(S2)) ya da < min; egitim."""
    sr_of = {r: {s: sr(v['X']) for s, v in load(root, arms[r], 'train', seeds).items()} for r in ('real', 'shuf', 'shuf2')}
    above = [sr_of['real'][s] > max(sr_of['shuf'][s], sr_of['shuf2'][s]) for s in seeds]
    below = [sr_of['real'][s] < min(sr_of['shuf'][s], sr_of['shuf2'][s]) for s in seeds]
    return {'above_both_per_seed': above, 'below_both_per_seed': below, 'all_above': all(above), 'all_below': all(below),
            'null_prob_per_side_approx': (1 / 3) ** len(seeds), 'null_prob_two_sided_approx': 2 * (1 / 3) ** len(seeds),
            'note': 'alt sinir: gercek blok tohumlar arasinda ortak; hiza etkisi yok varsayimi'}


def gross_sharpe(root, arm, seeds) -> list:
    """Betimsel (hakem-finans #4): maliyet geri eklenmis yaklasik brut SR, X + c*devir (birinci derece)."""
    d = load(root, arm, 'train', seeds)
    return [sr(d[s]['X'] + COST_ONE_WAY * d[s]['turnover']) for s in seeds]


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
    for y, arms in PATHS.items():
        def val_ok(c, s):
            va = P[f'{y}.{c}.val']
            return bool(s != 0 and va['D_annual'] * s > 0 and sum(d * s > 0 for d in va['delta_per_seed']) >= need(k))

        tr = P[f'{y}.real_shuf.train']
        s = float(np.sign(tr['D_annual']))
        rf = rank_floor(root, arms, seeds)
        cond = {'a_holm_p': bool(padj[(y, 'real_shuf')] < ALPHA),
                'b_rank_floor': bool((s > 0 and rf['all_above']) or (s < 0 and rf['all_below'])),
                'c_val_same_sign_m_of_k': val_ok('real_shuf', s)}
        rs = {'D_train': tr['D_annual'], 'D_val': P[f'{y}.real_shuf.val']['D_annual'], 'p_holm': padj[(y, 'real_shuf')],
              'sign': s, 'rank_floor': rf, 'conditions': cond, 'diff': all(cond.values())}

        def secondary(c):
            t = P[f'{y}.{c}.train']
            sg = float(np.sign(t['D_annual']))
            cnd = {'a_holm_p': bool(padj[(y, c)] < ALPHA),
                   'b_all_seeds_same_sign': bool(sg != 0 and all(d * sg > 0 for d in t['delta_per_seed'])),
                   'c_val_same_sign_m_of_k': val_ok(c, sg)}
            return {'D_train': t['D_annual'], 'D_val': P[f'{y}.{c}.val']['D_annual'], 'p_holm': padj[(y, c)],
                    'sign': sg, 'conditions': cnd, 'diff': all(cnd.values()), 'note': 'tabansiz ikincil (karisik)'}

        rt, st = secondary('real_t0'), secondary('shuf_t0')
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
        note = None
        if key == 'O1_fark_yok' and (rt['diff'] or st['diff']):
            note = ('ikincil fark var ama hicbir okuma satirina girmiyor (R<->T0 pozitif ya da zit isaretli); betimsel: '
                    f"R<->T0 {rt['sign']:+.0f} diff={rt['diff']}, S<->T0 {st['sign']:+.0f} diff={st['diff']}")
        if key == 'O3_girdi' and rt['sign'] < 0:
            note = 'O3 negatif: ek girdi ogrenmeyi bozuyor (bilgiden bagimsiz)'
        ss = P[f'{y}.shuf_shuf2.train']
        out[y] = {'real_shuf': rs, 'real_t0': rt, 'shuf_t0': st, 'secondary_note': note,
                  'size_diagnostic_shuf_shuf2': {'D_train': ss['D_annual'], 'p_two_sided': ss['p_two_sided'],
                                                 'hac_uncalibrated_flag': bool(ss['p_two_sided'] < ALPHA)},
                  'min_detectable_D_betimsel': (1.96 + 0.84) * tr['se_annual'],
                  'gross_sharpe_betimsel': {a: gross_sharpe(root, arms[a], seeds) for a in ('real', 'shuf', 'shuf2', 't0')}}
        out[y]['evidence_level'] = ('ikincil_olasi' if key in ('O3_girdi', 'O2_tahminsize_gore') else
                                    ('birincil' if key != 'O1_fark_yok' else None))
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
                    vals.append((json.load(fh).get('conf_scalars_final') or {}).get('beta'))
            else:
                vals.append(None)
        b[arm] = vals
    return b


def train_records_all(root, seeds=SEEDS) -> dict:
    out = {}
    for arm in ARMS:
        for s in seeds:
            p = os.path.join(root, arm, f's{s}', 'train.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    r = json.load(fh)
                out[f'{arm}/s{s}'] = {'commit': r.get('commit'), 'clean': r.get('git_status_porcelain') == ''}
    return out


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
                             policy_turnover=np.full(T, 0.5), policy_cash_weight=np.zeros(T))
        return root

    r = read(build({}))
    check('bos: iki yolda O1_fark_yok', r['reading'] == {'SO': 'O1_fark_yok', 'SB': 'O1_fark_yok'})
    r = read(build({'SO': -0.003}))
    check('gercek sinyal SO yolunu bozuyor: O2_bozma; SB O1', r['reading'] == {'SO': 'O2_bozma', 'SB': 'O1_fark_yok'})
    r = read(build({'SB': 0.003}))
    check('gercek > iki karistirilmis ve > tahminsiz: O4_kesifsel', r['reading']['SB'] == 'O4_kesifsel')
    r = read(build({'SB': 0.003, 'T0-G': 'SB'}))
    check('gercek > karistirilmis ama tahminsiz de ayni: O4_eksik', r['reading']['SB'] == 'O4_eksik')
    r = read(build({'SO': 0.003, 'SO-S': 'SO', 'SO-S2': 'SO'}))
    check('gercek = karistirilmis, ikisi de tahminsizden iyi: O3_girdi', r['reading']['SO'] == 'O3_girdi')
    r = read(build({'SO': 0.002, 'SO-S2': 0.004}))
    check('gercek karistirilmis-2 yi gecemiyor: (b) siralama tabani eler, O1; boyut tanisi isaretli',
          r['reading']['SO'] == 'O1_fark_yok' and not r['paths']['SO']['real_shuf']['conditions']['b_rank_floor']
          and r['paths']['SO']['size_diagnostic_shuf_shuf2']['hac_uncalibrated_flag'])
    r = read(build({'SO': -0.003, 'SO-S': 'SO', 'SO-S2': 'SO'}))
    check("R = S = S2, hepsi tahminsizden kotu: O3_girdi + 'O3 negatif' notu",
          r['reading']['SO'] == 'O3_girdi' and 'negatif' in (r['paths']['SO']['secondary_note'] or ''))
    gross = r['paths']['SB']['gross_sharpe_betimsel']['real']
    net = r['pairs']['SB.real_shuf.train']['sharpe_a']
    check('brut SR betimsel: devir 0,5 ile maliyet geri eklenince her tohumda net SR\'dan buyuk',
          len(gross) == 3 and all(g > n for g, n in zip(gross, net)))
    root = build({})
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


if __name__ == '__main__':
    main()
