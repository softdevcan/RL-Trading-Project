"""Faz 10 M5 mekanizma kollari okuma betigi (ana venv) — M5 tasarimi §4 ON KAYDI (PILOT); sonuctan ONCE yazildi.

Maliyetsiz X, SR yillik, egitim penceresi (2013-06..2018-12) ve dogrulama (2019-2020); tohumlar 6, 7, 8.
D(a, b) = ortak HAC (sharpe_diff_hac_pooled) ile ort_s [SR(a_s) - SR(b_s)]; ayni tohum ve tarihler eslenir.

H2 (birincil; guvenle agirlik > guvensiz dikkat):
  (i)   egitim: D(GL, G0) > 0 VE tek yonlu p < 0,05
  (ii)  egitim: D(GL, GLS) > 0 VE tek yonlu p < 0,05   (kontrol: avantaj guven BILGISINDEN mi; = fark farki
        D(GL, G0) - D(GLS, G0), ayni G0 kosulariyla, eslenik HAC ile dogrudan; bagimsizlik varsayimi gerekmez)
  (iii) dogrulama: D(GL, G0) > 0 VE en az 2/3 tohumda Delta > 0
  PILOT_ISARETI = ucu birden; KISMI = (i) var ama ucu birden degil; aksi GOZLENMEDI. delta (etki esigi) YOK
  (kullanici karari K-i, 2026-11-19). Karar kapisi degil, isaret (M4b ile ayni dil).
H3 (ikincil, daraltilmis: guvenin gondericiler arasi karsilastirilabilirligi), SABIT SIRA: yalniz H2 = PILOT_ISARETI
  iken etiketlenir (mekanizma guveni kullanmiyorsa kalibrasyonun degeri olculemez); aksi SINANMADI, sayilar betimsel.
  (i) egitim D(GL, GLM) > 0 VE p < 0,05; (ii) dogrulama D(GL, GLM) > 0 VE en az 2/3 tohum. Ikisi = PILOT_ISARETI,
  (i) tek basina = KISMI, aksi GOZLENMEDI. Sabit sira H2 -> H3 aile hata oranini Holm'suz korur.
BETIMSEL (kararlari degistirmez): D(GD, G0) (DebUnc, prob araligi dar -> ~0 beklenir), D(GLS, G0), devir,
  egitim sonu beta (manipulasyon denetimi: beta_GL ort <= 0 ise "GL guvene agirlik vermedi" okunur; Adam ust sinir
  |beta| <~ lr x guncelleme = 5e-4 x 1000 = 0,5), z_scale, egitim kaydi git izi, tohum gereksinimi (M4b formulu).

  python scripts/marl/analyze_m5.py --out docs/development/measurements/phase10-m5/m5_results.json
  python scripts/marl/analyze_m5.py --self-test
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
from analyze_m3b import provenance  # noqa: E402
from analyze_m4b import pair, seeds_needed  # noqa: E402

SEEDS = (6, 7, 8)
ARMS = ('G0', 'GL', 'GLS', 'GLM', 'GD')
PREREG = 'phase-10-m5-design.md §4 v0.5'
BETA_ADAM_BOUND = 5e-4 * (600_000 // 6000) * 10


def label3(train, val) -> tuple[bool, bool]:
    """(i) egitim D > 0 ve p < ALPHA; (iii) dogrulama D > 0 ve >= 2/3 tohum. (ii) cagirana ait."""
    i = bool(train['D_annual'] > 0 and train['p_one_sided'] < ALPHA)
    v = bool(val['D_annual'] > 0 and sum(d > 0 for d in val['delta_per_seed']) >= 2)
    return i, v


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


def read(root, seeds=SEEDS) -> dict:
    res = {'preregistration': PREREG, 'seeds': list(seeds), 'pilot': True}
    P = {n: pair(root, a, b, w, seeds) for n, (a, b, w) in {
        'GL_G0_train': ('GL', 'G0', 'train'), 'GL_GLS_train': ('GL', 'GLS', 'train'), 'GL_G0_val': ('GL', 'G0', 'val'),
        'GL_GLM_train': ('GL', 'GLM', 'train'), 'GL_GLM_val': ('GL', 'GLM', 'val'),
        'GD_G0_train': ('GD', 'G0', 'train'), 'GLS_G0_train': ('GLS', 'G0', 'train')}.items()}
    if any(v is None for v in P.values()):
        return {**res, 'label_H2': 'EKSIK', 'label_H3': 'EKSIK', 'missing': [k for k, v in P.items() if v is None]}
    i, iii = label3(P['GL_G0_train'], P['GL_G0_val'])
    ii = bool(P['GL_GLS_train']['D_annual'] > 0 and P['GL_GLS_train']['p_one_sided'] < ALPHA)
    cond_h2 = {'i_train_GL_gt_G0_p': i, 'ii_train_GL_gt_GLS_p': ii, 'iii_val_GL_gt_G0_2of3': iii}
    h2 = 'PILOT_ISARETI' if all(cond_h2.values()) else ('KISMI' if i else 'GOZLENMEDI')
    h3_i, h3_ii = label3(P['GL_GLM_train'], P['GL_GLM_val'])
    cond_h3 = {'i_train_GL_gt_GLM_p': h3_i, 'ii_val_GL_gt_GLM_2of3': h3_ii}
    h3 = ('SINANMADI' if h2 != 'PILOT_ISARETI' else
          ('PILOT_ISARETI' if h3_i and h3_ii else ('KISMI' if h3_i else 'GOZLENMEDI')))
    d, se = P['GL_G0_train']['D_annual'], P['GL_G0_train']['se_annual']
    tr = train_records(root, seeds)
    betas = {arm: [((tr.get(f'{arm}/s{k}') or {}).get('conf_scalars_final') or {}).get('beta') for k in seeds]
             for arm in ('GL', 'GLS', 'GLM')}
    bgl = [b for b in betas['GL'] if b is not None]
    manip = {'beta_final': betas, 'beta_adam_bound': BETA_ADAM_BOUND,
             'GL_beta_mean': float(np.mean(bgl)) if bgl else None,
             'GL_used_confidence': (bool(np.mean(bgl) > 0) if bgl else None)}
    return {**res, **P, 'conditions_H2': cond_h2, 'label_H2': h2, 'conditions_H3': cond_h3, 'label_H3': h3,
            'manipulation_check': manip,
            'train_runs': tr, 'train_runs_all_clean': bool(tr) and all(r['clean'] for r in tr.values()),
            'train_runs_commits': sorted({r['commit'] for r in tr.values() if r['commit']}),
            'sizing_betimsel': {'seeds_for_80pct_H2_observed': seeds_needed(d, se),
                                'seeds_for_80pct_H2_ci_low': seeds_needed(d - 1.96 * se, se),
                                'seeds_for_80pct_H2_ci_high': seeds_needed(d + 1.96 * se, se),
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

    def build(edges, beta=None):
        """edges[arm] = gunluk kenar (float) ya da baska bir kolun adi (o kolun serisinin birebir kopyasi)."""
        root = tempfile.mkdtemp(prefix='m5_')
        for window in ('train', 'val'):
            mkt = rng.normal(0.0003, 0.01, T)
            series = {}
            for arm, e in sorted(edges.items(), key=lambda kv: isinstance(kv[1], str)):
                for k in SEEDS:
                    d = os.path.join(root, arm, f's{k}')
                    os.makedirs(d, exist_ok=True)
                    x = series[(e, k)] if isinstance(e, str) else mkt + rng.normal(0, 0.003, T) + e
                    series[(arm, k)] = x
                    np.savez(os.path.join(d, f'eval_{window}.npz'), policy_X=x,
                             policy_date=dates, policy_turnover=np.zeros(T), policy_cash_weight=np.zeros(T))
                    if beta is not None and window == 'train':
                        with open(os.path.join(d, 'train.json'), 'w', encoding='utf-8') as fh:
                            json.dump({'commit': 'abc', 'git_status_porcelain': '', 'build': {'model': {'z_scale': 1.0}},
                                       'conf_scalars_final': ({'beta': beta} if arm != 'G0' else None)}, fh)
        return root

    base = {a: 0.0 for a in ARMS}
    r = read(build({**base, 'GL': 0.003}, beta=0.2))
    check('guven bilgisi isliyor: H2 PILOT_ISARETI; GLM bozuk -> H3 PILOT_ISARETI; beta > 0 okunur, kayit temiz',
          r['label_H2'] == 'PILOT_ISARETI' and r['label_H3'] == 'PILOT_ISARETI'
          and r['manipulation_check']['GL_used_confidence'] is True and r['train_runs_all_clean'])
    r = read(build({**base, 'GL': 0.003, 'GLS': 0.003}))
    check('avantaj karistirilmis guvende de var: H2 KISMI ((ii) eler), H3 SINANMADI',
          r['label_H2'] == 'KISMI' and not r['conditions_H2']['ii_train_GL_gt_GLS_p'] and r['label_H3'] == 'SINANMADI')
    r = read(build({**base, 'GL': 0.003, 'GLM': 'GL'}))
    check('GL = GLM (ayni seri): H2 PILOT_ISARETI, H3 GOZLENMEDI (kalibrasyon farki yok)',
          r['label_H2'] == 'PILOT_ISARETI' and r['label_H3'] == 'GOZLENMEDI')
    r = read(build(base, beta=-0.05))
    check('bos: H2 GOZLENMEDI, H3 SINANMADI, beta <= 0 -> guven kullanilmadi',
          r['label_H2'] == 'GOZLENMEDI' and r['label_H3'] == 'SINANMADI'
          and r['manipulation_check']['GL_used_confidence'] is False)
    r = read(build({'G0': 0.0, 'GL': 0.003, 'GLS': 0.0}))
    check('eksik kol: EKSIK', r['label_H2'] == 'EKSIK' and 'GL_GLM_train' in r['missing'])
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
    print(json.dumps({k: res.get(k) for k in ('label_H2', 'conditions_H2', 'label_H3', 'conditions_H3',
                                              'manipulation_check', 'train_runs_all_clean', 'sizing_betimsel')},
                     indent=1, ensure_ascii=False, default=float))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
