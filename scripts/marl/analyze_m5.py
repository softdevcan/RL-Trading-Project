"""Faz 10 M5 mekanizma kollari okuma betigi (ana venv) — M5 tasarimi §4 ON KAYDI (PILOT); sonuctan ONCE yazildi.
v0.6: hakem-rl M5 okumasi (#1 gurultu tabani, #2 beta esigi, #5 tek okuma, #6 kod <-> metin) islendi.

Maliyetsiz X, SR yillik, egitim penceresi (2013-06..2018-12) ve dogrulama (2019-2020); tohumlar 6, 7, 8.
D(a, b) = ortak HAC (sharpe_diff_hac_pooled) ile ort_s [SR(a_s) - SR(b_s)]; ayni tohum ve tarihler eslenir.
k = tohum sayisi, m = ceil(2k/3) (3 tohumda 2).

H2 (birincil; guvenle agirlik > guvensiz dikkat):
  (i)   egitim: D(GL, G0) > 0 VE tek yonlu p < 0,05
  (ii)  egitim: D(GL, GLS) > 0 VE tek yonlu p < 0,05   (kontrol: avantaj guven BILGISINDEN mi; = fark farki
        D(GL, G0) - D(GLS, G0), ayni G0 kosulariyla, eslenik HAC ile dogrudan)
  (iii) dogrulama: D(GL, G0) > 0 VE en az m tohumda Delta > 0
  (iv)  mekanizma kullanildi: en az m tohumda egitim sonu beta_GL >= BETA_MIN = 0,04. Dayanak (sonuctan once):
        zlogodds terimi p90 - p10 ~ 2,5 (conf_informative.json) -> exp(0,04 x 2,5) ~ 1,105, yani en guvenli ile en
        guvensiz gonderici arasinda en az %10 dikkat orani. Adam ust siniri |beta| <~ 5e-4 x 1000 = 0,5.
  (v)   gurultu tabani: egitim D(GL, G0) > |D(GD, G0)|. GD'nin terimi gondericileri ~0,036 ayiriyor (islevi ~G0) ama
        yorungeyi degistiriyor; ayni tohumla G0'i tekrar kosmak GPU'da bit-es cikti (2026-11-19), gurultu vermez.
  Etiket: PILOT_ISARETI = (i)-(v); KISMI = (i) var, hepsi degil; GOZLENMEDI = (i) yok; EKSIK = birincil kol ya da
  GL egitim kaydi (beta) eksik. Karar kapisi degil, isaret. delta (Sharpe etki esigi) YOK (kullanici karari K-i).
  Tek okuma (oncelik sirasi): KISMI -> ilk saglanmayan (iv) 'guvenden_bagimsiz', (ii) 'ek_serbestlik',
  (v) 'gurultu_tabani', (iii) 'dogrulamada_tutmuyor'. GOZLENMEDI -> (iv) varsa 'guven_kullanildi_fark_yok', yoksa
  'guven_kullanilmadi'. PILOT_ISARETI -> 'bilgi_isareti'.
H3 (ikincil, daraltilmis: guvenin gondericiler arasi karsilastirilabilirligi), SABIT SIRA: yalniz H2 = PILOT_ISARETI
  iken etiketlenir; aksi SINANMADI ('h2_isareti_yok'), sayilar betimsel. (i) egitim D(GL, GLM) > 0 VE p < 0,05;
  (ii) dogrulama D(GL, GLM) > 0 VE en az m tohum. PILOT_ISARETI = ikisi ('karsilastirilabilirlik_isareti');
  KISMI = yalniz (i) ('dogrulamada_tutmuyor'); GOZLENMEDI ('bozma_avantaji_silmedi'); GLM eksikse EKSIK.
BETIMSEL (kararlari degistirmez): D(GLS, G0), D(GD, G0) dogrulama, devir, tohum basina beta (GL, GLS, GLM) ve
  beta_GLS_ge_min_m_of_k (GLS de esigi geciyorsa (iv) guvene ozgu degildir; hakem-rl 2. okuma #2), z_scale,
  egitim kaydi git izi, GLM bozma gucu (design_checks.json), tohum gereksinimi (M4b formulu; (i) ve (ii) icin).

  python scripts/marl/analyze_m5.py --out docs/development/measurements/phase10-m5/m5_results.json
  python scripts/marl/analyze_m5.py --self-test
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
from analyze_m3 import ALPHA  # noqa: E402
from analyze_m3b import provenance  # noqa: E402
from analyze_m4b import pair, seeds_needed  # noqa: E402

SEEDS = (6, 7, 8)
ARMS = ('G0', 'GL', 'GLS', 'GLM', 'GD')
PREREG = 'phase-10-m5-design.md §4 v0.7 (DONDURULDU)'
BETA_MIN = 0.04
BETA_ADAM_BOUND = 5e-4 * (600_000 // 6000) * 10
DESIGN_CHECKS = os.path.join('docs', 'development', 'measurements', 'phase10-m5', 'design_checks.json')
PRIMARY = {'GL_G0_train': ('GL', 'G0', 'train'), 'GL_GLS_train': ('GL', 'GLS', 'train'),
           'GL_G0_val': ('GL', 'G0', 'val'), 'GD_G0_train': ('GD', 'G0', 'train')}
H3_PAIRS = {'GL_GLM_train': ('GL', 'GLM', 'train'), 'GL_GLM_val': ('GL', 'GLM', 'val')}
DESCRIPTIVE = {'GLS_G0_train': ('GLS', 'G0', 'train'), 'GD_G0_val': ('GD', 'G0', 'val')}


def need(k: int) -> int:
    return math.ceil(2 * k / 3)


def sig_pos(pr) -> bool:
    return bool(pr['D_annual'] > 0 and pr['p_one_sided'] < ALPHA)


def val_ok(pr, k) -> bool:
    return bool(pr['D_annual'] > 0 and sum(d > 0 for d in pr['delta_per_seed']) >= need(k))


def train_records(root, seeds=SEEDS) -> dict:
    out = {}
    for arm in ARMS:
        for k in seeds:
            p = os.path.join(root, arm, f's{k}', 'train.json')
            if os.path.exists(p):
                with open(p, encoding='utf-8') as fh:
                    r = json.load(fh)
                m = (r.get('build') or {}).get('model') or {}
                out[f'{arm}/s{k}'] = {'commit': r.get('commit'), 'clean': r.get('git_status_porcelain') == '',
                                      'conf_scalars_final': r.get('conf_scalars_final'),
                                      'z_scale': m.get('z_scale'), 'frames': r.get('frames'),
                                      'wall_seconds': r.get('wall_seconds')}
    return out


def h2_reading(label: str, c: dict) -> str:
    if label == 'PILOT_ISARETI':
        return 'bilgi_isareti'
    if label == 'GOZLENMEDI':
        return 'guven_kullanildi_fark_yok' if c['iv_beta_GL_ge_min'] else 'guven_kullanilmadi'
    for key, name in (('iv_beta_GL_ge_min', 'guvenden_bagimsiz'), ('ii_train_GL_gt_GLS_p', 'ek_serbestlik'),
                      ('v_train_GL_G0_gt_noise_floor', 'gurultu_tabani'), ('iii_val_GL_gt_G0_m_of_k', 'dogrulamada_tutmuyor')):
        if not c[key]:
            return name
    raise AssertionError('KISMI ama tum kosullar saglaniyor')


def read(root, seeds=SEEDS, design_checks: str | None = DESIGN_CHECKS) -> dict:
    k = len(seeds)
    res = {'preregistration': PREREG, 'seeds': list(seeds), 'pilot': True, 'm_of_k': [need(k), k],
           'beta_min': BETA_MIN, 'beta_adam_bound': BETA_ADAM_BOUND}
    P = {n: pair(root, a, b, w, seeds) for n, (a, b, w) in {**PRIMARY, **H3_PAIRS, **DESCRIPTIVE}.items()}
    tr = train_records(root, seeds)
    betas = {arm: [((tr.get(f'{arm}/s{s}') or {}).get('conf_scalars_final') or {}).get('beta') for s in seeds]
             for arm in ('GL', 'GLS', 'GLM')}
    res.update({'pairs': P, 'beta_final': betas, 'train_runs': tr,
                'train_runs_all_clean': bool(tr) and all(r['clean'] for r in tr.values()),
                'train_runs_commits': sorted({r['commit'] for r in tr.values() if r['commit']})})
    if design_checks and os.path.exists(design_checks):
        with open(design_checks, encoding='utf-8') as fh:
            dc = json.load(fh)
        res['h3_power_betimsel'] = {'temp_range': dc.get('temp_range'),
                                    'daily_spearman_mean': [s['GLM_daily_spearman_mean'] for s in dc['seeds']],
                                    'low_power_any': any(s['GLM_low_power'] for s in dc['seeds'])}
    missing = [n for n in PRIMARY if P[n] is None] + (['beta_GL'] if any(b is None for b in betas['GL']) else [])
    if missing:
        return {**res, 'label_H2': 'EKSIK', 'reading_H2': None, 'label_H3': 'EKSIK', 'reading_H3': None,
                'missing': missing}
    gl, gd = P['GL_G0_train'], P['GD_G0_train']
    c2 = {'i_train_GL_gt_G0_p': sig_pos(gl), 'ii_train_GL_gt_GLS_p': sig_pos(P['GL_GLS_train']),
          'iii_val_GL_gt_G0_m_of_k': val_ok(P['GL_G0_val'], k),
          'iv_beta_GL_ge_min': sum(b >= BETA_MIN for b in betas['GL']) >= need(k),
          'v_train_GL_G0_gt_noise_floor': bool(gl['D_annual'] > abs(gd['D_annual']))}
    h2 = 'PILOT_ISARETI' if all(c2.values()) else ('KISMI' if c2['i_train_GL_gt_G0_p'] else 'GOZLENMEDI')
    gls_b = [b for b in betas['GLS'] if b is not None]
    out = {**res, 'conditions_H2': c2, 'label_H2': h2, 'reading_H2': h2_reading(h2, c2),
           'noise_floor_D_GD_G0': gd['D_annual'],
           # betimsel (hakem-rl M5 2. okuma #2): GLS de esigi geciyorsa (iv) guvene ozgu degildir
           'beta_GLS_ge_min_m_of_k': (sum(b >= BETA_MIN for b in gls_b) >= need(k)) if len(gls_b) == k else None,
           'sizing_betimsel': {'seeds_for_80pct_i_observed': seeds_needed(gl['D_annual'], gl['se_annual'], k),
                               'seeds_for_80pct_i_ci_low': seeds_needed(gl['D_annual'] - 1.96 * gl['se_annual'],
                                                                        gl['se_annual'], k),
                               'seeds_for_80pct_ii_observed': seeds_needed(P['GL_GLS_train']['D_annual'],
                                                                           P['GL_GLS_train']['se_annual'], k),
                               'note': 'SE tohumla 1/sqrt(k) kuculur varsayimi iyimser (HAC gun gurultusu); kaba'}}
    if P['GL_GLM_train'] is None or P['GL_GLM_val'] is None:
        h3, c3 = ('SINANMADI' if h2 != 'PILOT_ISARETI' else 'EKSIK'), None
    else:
        c3 = {'i_train_GL_gt_GLM_p': sig_pos(P['GL_GLM_train']), 'ii_val_GL_gt_GLM_m_of_k': val_ok(P['GL_GLM_val'], k)}
        h3 = ('SINANMADI' if h2 != 'PILOT_ISARETI' else
              ('PILOT_ISARETI' if all(c3.values()) else ('KISMI' if c3['i_train_GL_gt_GLM_p'] else 'GOZLENMEDI')))
    r3 = {'SINANMADI': 'h2_isareti_yok', 'PILOT_ISARETI': 'karsilastirilabilirlik_isareti',
          'KISMI': 'dogrulamada_tutmuyor', 'GOZLENMEDI': 'bozma_avantaji_silmedi', 'EKSIK': None}[h3]
    return {**out, 'conditions_H3': c3, 'label_H3': h3, 'reading_H3': r3}


def self_test() -> bool:
    rng = np.random.default_rng(0)
    T, ok, total = 400, 0, 0
    dates = np.array([f'2015-01-{i:04d}' for i in range(T)])

    def check(name, good):
        nonlocal ok, total
        total += 1
        ok += bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name}")

    def build(edges, beta=0.2):
        """edges[arm] = gunluk kenar (float), baska bir kolun adi (o kolun serisinin birebir kopyasi) ya da
        {'train': ..., 'val': ...}. beta: GL/GLS/GLM egitim sonu beta'si (None -> egitim kaydi yazilmaz)."""
        root = tempfile.mkdtemp(prefix='m5_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            ew = {a: (e[window] if isinstance(e, dict) else e) for a, e in edges.items()}
            series = {}
            for arm, e in sorted(ew.items(), key=lambda kv: isinstance(kv[1], str)):
                for s in SEEDS:
                    d = os.path.join(root, arm, f's{s}')
                    os.makedirs(d, exist_ok=True)
                    x = series[(e, s)] if isinstance(e, str) else mkt + rng.normal(0, 0.003, T) + e
                    series[(arm, s)] = x
                    np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x, policy_date=dates,
                             policy_turnover=np.zeros(T), policy_cash_weight=np.zeros(T))
                    if beta is not None and window == 'train':
                        cs = None if arm == 'G0' else {'beta': 1.0 if arm == 'GD' else beta}
                        with open(os.path.join(d, 'train.json'), 'w', encoding='utf-8') as fh:
                            json.dump({'commit': 'abc', 'git_status_porcelain': '', 'build': {'model': {}},
                                       'conf_scalars_final': cs}, fh)
        return root

    base = {a: 0.0 for a in ARMS}
    R = lambda edges, **kw: read(build(edges, **kw), design_checks=None)   # noqa: E731
    r = R({**base, 'GL': 0.003})
    check('guven bilgisi isliyor: H2 PILOT_ISARETI/bilgi_isareti; GLM bozuk -> H3 PILOT_ISARETI; kayit temiz',
          (r['label_H2'], r['reading_H2'], r['label_H3']) == ('PILOT_ISARETI', 'bilgi_isareti', 'PILOT_ISARETI')
          and r['train_runs_all_clean'])
    r = R({**base, 'GL': 0.003, 'GLS': 0.003})
    check('avantaj karistirilmis guvende de var: KISMI/ek_serbestlik, H3 SINANMADI',
          (r['label_H2'], r['reading_H2'], r['label_H3']) == ('KISMI', 'ek_serbestlik', 'SINANMADI'))
    r = R({**base, 'GL': 0.003, 'GLM': 'GL'})
    check('GL = GLM (ayni seri): H2 PILOT_ISARETI, H3 GOZLENMEDI/bozma_avantaji_silmedi',
          (r['label_H2'], r['label_H3'], r['reading_H3']) == ('PILOT_ISARETI', 'GOZLENMEDI', 'bozma_avantaji_silmedi'))
    r = R(base, beta=-0.05)
    check('bos + beta <= 0: GOZLENMEDI/guven_kullanilmadi, H3 SINANMADI',
          (r['label_H2'], r['reading_H2'], r['label_H3']) == ('GOZLENMEDI', 'guven_kullanilmadi', 'SINANMADI'))
    r = R({**base, 'GL': 0.003}, beta=0.01)
    check('getiri farki var ama beta < 0,04: KISMI/guvenden_bagimsiz (iv)',
          (r['label_H2'], r['reading_H2']) == ('KISMI', 'guvenden_bagimsiz'))
    r = R({**base, 'GL': {'train': 0.003, 'val': -0.003}})
    check('egitimde var, dogrulamada ters: KISMI/dogrulamada_tutmuyor (iii)',
          (r['label_H2'], r['reading_H2']) == ('KISMI', 'dogrulamada_tutmuyor'))
    r = R({**base, 'GL': 0.003, 'GD': 'GL'})
    check('GD de ayni farki uretiyor: KISMI/gurultu_tabani (v)',
          (r['label_H2'], r['reading_H2']) == ('KISMI', 'gurultu_tabani'))
    r = R({**base, 'GL': 0.003, 'GLM': {'train': 0.0, 'val': 'GL'}})
    check('H3 egitimde var, dogrulamada yok: H3 KISMI/dogrulamada_tutmuyor',
          (r['label_H2'], r['label_H3'], r['reading_H3']) == ('PILOT_ISARETI', 'KISMI', 'dogrulamada_tutmuyor'))
    r1 = R({'G0': 0.0, 'GL': 0.003, 'GLS': 0.0, 'GLM': 0.0})
    r2 = R({'G0': 0.0, 'GL': 0.003, 'GLS': 0.0, 'GD': 0.0})
    r3 = R({**base, 'GL': 0.003}, beta=None)
    check('eksik: GD (gurultu tabani) -> H2 EKSIK; yalniz GLM eksik -> H2 okunur, H3 EKSIK; beta kaydi yok -> EKSIK',
          r1['label_H2'] == 'EKSIK' and 'GD_G0_train' in r1['missing'] and r2['label_H2'] == 'PILOT_ISARETI'
          and r2['label_H3'] == 'EKSIK' and r3['label_H2'] == 'EKSIK' and r3['missing'] == ['beta_GL'])
    check('m = ceil(2k/3): 3 -> 2, 4 -> 3, 5 -> 4', (need(3), need(4), need(5)) == (2, 3, 4))
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m5'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    res = {**read(a.root), 'provenance': provenance()}
    print(json.dumps({k: res.get(k) for k in ('label_H2', 'reading_H2', 'conditions_H2', 'label_H3', 'reading_H3',
                                              'conditions_H3', 'beta_final', 'noise_floor_D_GD_G0',
                                              'train_runs_all_clean', 'sizing_betimsel')},
                     indent=1, ensure_ascii=False, default=float))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
