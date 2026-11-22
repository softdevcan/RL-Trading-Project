"""M8 okumasina SONUCTAN SONRA eklenen betimsel saglamlik olcumleri (hakem-2 kapanis okumasi #1, #2, #4, #7, #11).

On kayit (phase-10-m8-design.md §6) ve etiketler DEGISMEZ (§6.8); bu betik yalniz betimsel satirlar uretir:

- Tohum kumelemesi (#1): ayni tohum numarasi alti katta yeniden kullaniliyor (ayni baslangic agirliklari), bu yuzden
  72 es 12 tohum kumesidir. Kumeli tam isaret-cevirme (2^12 = 4096 duzen; tohum basina 6 kat ortalamasi), t GA (df 11),
  tohum etkisi F (iki yonlu, kat etkisi cikarilmis) ve ayni kat/ayni tohumda kollar arasi korelasyon.
- 0 bps (#2): ayni karsitlik kurali (m8_stats.contrast, R = B = 10.000) maliyetsiz ex post serilerle.
- Esli MDE (#4): kat ici tabakali SE ve tohum kumeli SE ile (on kayittaki eslesmesiz mde_recomputed'dan farkli).
- Kural portfoyleri 0 bps'de (#7): ogrenen kollarla simetrik kiyas.
- Bir kati disarida birakma (#11).

Kullanim (ana venv): python scripts/marl/m8_robustness.py --out docs/development/measurements/phase10-m8/m8_robustness.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_m8 as A  # noqa: E402
import m8_stats as ST  # noqa: E402

T975_11 = 2.200985   # t(0,975; 11)
T80_11 = 0.875539    # t(0,80; 11)
Z = 1.959964 + 0.841621
FROZEN = '1f898956c08a9c2f5ca2a41b5f8dd12a7bbff676'   # on kayit v1.0 dondurma commit'i


def seed_cluster(d: np.ndarray) -> dict:
    """d: [6 kat, 12 tohum] es farki. Tohum basina kat ortalamasi -> 12 deger."""
    sm = d.mean(axis=0)
    obs = abs(sm.mean())
    flips = np.array(list(itertools.product([1.0, -1.0], repeat=sm.size)))
    cnt = int((np.abs((flips * sm).mean(axis=1)) >= obs - 1e-12).sum())
    se = sm.std(ddof=1) / np.sqrt(sm.size)
    resid = d - d.mean(axis=1, keepdims=True)
    seed_eff = resid.mean(axis=0)
    e = resid - seed_eff
    nf, ns = d.shape
    ms_seed = nf * np.var(seed_eff, ddof=1)
    ms_err = (e ** 2).sum() / ((nf - 1) * (ns - 1))
    F = float(ms_seed / ms_err)
    try:
        from scipy.stats import f as fdist
        pF = float(fdist.sf(F, ns - 1, (nf - 1) * (ns - 1)))
    except Exception:   # noqa: BLE001
        pF = None
    return {'p_signflip_exact': cnt / flips.shape[0], 'n_patterns': int(flips.shape[0]),
            't_ci95': [float(sm.mean() - T975_11 * se), float(sm.mean() + T975_11 * se)],
            'mde80': float((T975_11 + T80_11) * se),
            'seed_effect_F': F, 'seed_effect_df': [ns - 1, (nf - 1) * (ns - 1)], 'seed_effect_p': pF}


def paired_strat_mde(d: np.ndarray) -> float:
    sdf = d.std(axis=1, ddof=1)
    se = np.sqrt((sdf ** 2 / d.shape[1]).sum()) / d.shape[0]
    return float(Z * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m8'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    os.chdir(ROOT)
    ref = A.reference_shas()
    cells, test_rows = {}, {}
    for f in A.FOLDS:
        for arm in A.ARMS:
            for s in A.seeds_of(arm):
                r = A.load_record(a.root, f, arm, s)
                if r is None:
                    raise SystemExit(f'eksik kayit {f} {arm} {s}')
                c = A.record_checks(r, ref)
                if not (all(c[k] for k in ('clean', 'panel', 'graph', 'windows', 'config'))
                        and c['commit_train'] == c['commit_eval'] == FROZEN):
                    raise SystemExit(f'denetim bozuk {f} {arm} {s}: {c}')
                test_rows.setdefault(f, tuple(r['fe']['rows']['test']))
                for rule in A.RULES:
                    cells[(f, arm, s, rule)] = A.cell_metrics(A.load_series(r, rule))
    M = A.matrices(cells)
    out = {'note': ('SONUCTAN SONRA eklenen betimsel saglamlik (hakem-2 kapanis #1 #2 #4 #7 #11); '
                    'on kayit etiketlerini degistirmez (§6.8)'),
           'provenance': {**A.provenance(), 'robustness_script_sha256': A._sha(os.path.abspath(__file__))},
           'contrasts': {}, 'baselines_0bps': {}}
    for name, ca, cb, *_ in A.CONTRASTS:
        blk = {}
        for rule in A.RULES:
            d = M[rule]['sr'][ca] - M[rule]['sr'][cb]
            x, y = M[rule]['sr'][ca], M[rule]['sr'][cb]
            blk[rule] = {
                'delta_bar': float(d.mean()),
                'seed_cluster': seed_cluster(d),
                'mde80_paired_within_fold': paired_strat_mde(d),
                'within_fold_arm_corr': [float(np.corrcoef(x[i], y[i])[0, 1]) for i in range(d.shape[0])],
                'leave_one_fold_out_delta': {A.FOLDS[i]: float(np.delete(d, i, 0).mean())
                                             for i in range(d.shape[0])},
            }
        d0 = M['argmax']['sr_0'][ca] - M['argmax']['sr_0'][cb]
        c0 = ST.contrast(d0, ca, cb)
        blk['argmax_0bps'] = {**c0, 'seed_cluster': seed_cluster(d0)}
        out['contrasts'][name] = blk
    from marl import panel as PN
    from marl.rule_baselines import rule_series
    arrays = PN.load_npz(A.EXPECTED_ARGS['panel'])
    for f, (ts, te) in test_rows.items():
        out['baselines_0bps'][f] = {}
        for kind in ('ew', 'bh', 'minvar', 'hrp'):
            s = rule_series(arrays, ts, te, kind, 0.0)
            out['baselines_0bps'][f][kind] = A.cell_metrics({**s, 'n_sell_blocked': 0})['sr']
    out['baselines_0bps']['mean'] = {k: float(np.mean([out['baselines_0bps'][f][k] for f in test_rows]))
                                     for k in ('ew', 'bh', 'minvar', 'hrp')}
    out['arms_0bps_mean_sr'] = {arm: float(np.nanmean(M['argmax']['sr_0'][arm])) for arm in A.ARMS}
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
    for name, blk in out['contrasts'].items():
        for rule in A.RULES:
            sc = blk[rule]['seed_cluster']
            print(f"{name} {rule}: d {blk[rule]['delta_bar']:+.3f} kumeli p {sc['p_signflip_exact']:.4f} "
                  f"t-GA [{sc['t_ci95'][0]:+.2f}, {sc['t_ci95'][1]:+.2f}] F {sc['seed_effect_F']:.2f} "
                  f"(p {sc['seed_effect_p']:.3g}) MDE esli {blk[rule]['mde80_paired_within_fold']:.3f} "
                  f"kumeli {sc['mde80']:.3f}")
        z = blk['argmax_0bps']
        print(f"   0 bps: d {z['delta_bar']:+.3f} GA [{z['ci95'][0]:+.2f}, {z['ci95'][1]:+.2f}] p_max {z['p_max']:.4f} "
              f"kumeli p {z['seed_cluster']['p_signflip_exact']:.4f}")
    print('kurallar 0 bps ort.:', {k: round(v, 3) for k, v in out['baselines_0bps']['mean'].items()})
    print('kollar 0 bps ort.:', {k: round(v, 3) for k, v in out['arms_0bps_mean_sr'].items()})


if __name__ == '__main__':
    main()
