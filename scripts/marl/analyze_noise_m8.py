"""Faz 10 M8 gurultu tabani BOYUT OLCUMU okuma betigi (ana venv) — M8 tasarimi §3.1 ON KAYIT v1.0; sonuctan ONCE yazildi.

Kosu: scripts/marl/run_noise_m8.py (venv-marl). Referans MAPPO, panel_m8, sozde kat '2020' (egitim ..2018-12-31,
dogrulama 2019, orneklem disi 2020; 2021+ OKUNMAZ), F = 600.000, 10 kontrol noktasi, n_envs 10, 20 bps, tohum 101-106.
Bu bir kol karsilastirmasi DEGIL; karar yok, cikti asil gurultu tabani olcumunun ve M8 tohum sayisinin BOYUTUDUR.

Okuma (betimsel):
  - tohum basina: secilen noktanin dogrulama (2019) Sharpe'i, 2020 fazla getiri Sharpe'i
    (marl.selection.excess_sharpe(R, rf=R - X); eval_fold_m8 secim metrigiyle ayni tanim), secimi tanimsiz
    (degenerate) tohum sayisi, ayri agirlik sayisi.
  - sigma_tohum: 2020 Sharpe'larinin ornek standart sapmasi (ddof 1).
  - bos dagilim ilk bakis: 6 tohumun 3 / 3 ayrik bolmeleri (C(6,3)/2 = 10), D = ort(A) - ort(B). Isaret keyfi
    (A ilk tohumu icerir); |D|'nin en buyugu ve ceyrekleri raporlanir.
  - MDE tablosu: M8 esli tasarimi, n = k tohum x 6 kat esli fark, tek yonlu %5, guc %80, normal yaklasim:
    MDE = (z_0.95 + z_0.80) * sigma_fark / sqrt(n), sigma_fark = sqrt(2) * sigma_tohum. Varsayim (beyan): kollar
    bagimsiz, katlar arasi ayni varyans, kat ici tohum farklari bagimsiz.
Denetimler (betimsel; sayilari degistirmez): egitim ve degerlendirme kayitlarinin git izi temiz mi, pencereler
  on kayitla ayni mi (dogrulama 2019, orneklem disi 2020), panel SHA panel_m8.json ile ayni mi, egitim
  yapilandirmasi on kayitla ayni mi (F, tohum, maliyet, lr, toplu ortam, n kontrol noktasi, egitim sonu).
Sinir (onceden): tek orneklem disi yil (2020, COVID oynakligi) yayilimi tipik olmayan bicimde buyutebilir.

  python scripts/marl/analyze_noise_m8.py --out docs/development/measurements/phase10-m8/noise_size.json
  python scripts/marl/analyze_noise_m8.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from marl.selection import excess_sharpe  # noqa: E402

PREREG = 'phase-10-m8-design.md §3.1 ON KAYIT v1.0'
SEEDS = (101, 102, 103, 104, 105, 106)
FOLDS_M8 = 6
K_GRID = (6, 8, 12, 16, 24)
Z_ONE_SIDED = 1.6448536269514722     # norm.ppf(0.95)
Z_POWER80 = 0.8416212335729143       # norm.ppf(0.80)
EXPECTED = {'frames': 600_000, 'checkpoints': 10, 'cost_bps': 20.0, 'lr': 5e-4, 'batched_env': True,
            'train_end': '2018-12-31'}
EXPECTED_WINDOWS = {'val': ('2019-01-01', '2019-12-31'), 'test': ('2020-01-01', '2020-12-31')}
PANEL_JSON = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'panel_m8.json')


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'),
            'script_sha256': sha}


def mde(sigma_seed: float, k: int, folds: int = FOLDS_M8) -> float:
    return float((Z_ONE_SIDED + Z_POWER80) * math.sqrt(2.0) * sigma_seed / math.sqrt(k * folds))


def splits_3v3(seeds) -> list:
    """6 tohumun ayrik 3/3 bolmeleri; her bolme bir kez (A ilk tohumu icerir)."""
    seeds = list(seeds)
    return [(list(a), [s for s in seeds if s not in a]) for a in itertools.combinations(seeds, len(seeds) // 2)
            if seeds[0] in a]


def _in_window(dates, lo: str, hi: str) -> bool:
    return bool(dates) and lo <= min(dates) and max(dates) <= hi


def seed_record(run_dir: str, panel_sha: str | None) -> dict | None:
    fe_path = os.path.join(run_dir, 'eval', 'fold_eval.json')
    if not os.path.exists(fe_path):
        return None
    with open(fe_path, encoding='utf-8') as fh:
        fe = json.load(fh)
    tr = {}
    if os.path.exists(os.path.join(run_dir, 'train.json')):
        with open(os.path.join(run_dir, 'train.json'), encoding='utf-8') as fh:
            tr = json.load(fh)
    z = np.load(os.path.join(run_dir, 'eval', fe['tests']['argmax']['file']))
    R, X = np.asarray(z['policy_R'], float), np.asarray(z['policy_X'], float)
    dates = [str(d) for d in z['policy_date']]
    sel = fe['selection']
    args = tr.get('args', {})
    cfg_ok = {
        'frames': args.get('frames') == EXPECTED['frames'],
        'checkpoints': args.get('checkpoints') == EXPECTED['checkpoints'],
        'cost_bps': args.get('cost_bps') is not None and float(args['cost_bps']) == EXPECTED['cost_bps'],
        'lr': args.get('lr') is not None and math.isclose(float(args['lr']), EXPECTED['lr']),
        'batched_env': bool(args.get('batched_env')) == EXPECTED['batched_env'],
        'train_end': args.get('train_end') == EXPECTED['train_end'],
    }
    val_dates = [r for r in fe.get('windows', {}).get('val', [])]
    return {
        'val_sharpe_selected': sel.get('metric'),
        'test_sharpe_2020': excess_sharpe(R, rf=R - X),
        'test_mean_excess_daily': float(np.mean(X)) if X.size else None,
        'test_days': int(X.size),
        'selected_step': sel.get('step'), 'degenerate': bool(sel.get('degenerate')),
        'n_valid': sel.get('n_valid'), 'n_distinct': sel.get('n_distinct'),
        'val_sharpe_last': sel.get('last_metric'),
        'checks': {
            'train_clean': tr.get('git_status_porcelain') == '' if tr else False,
            'eval_clean': fe.get('provenance', {}).get('git_status_porcelain') == '',
            'train_commit': tr.get('commit'), 'eval_commit': fe.get('provenance', {}).get('commit'),
            'val_window_ok': _in_window(val_dates, *EXPECTED_WINDOWS['val']),
            'test_window_ok': _in_window(dates, *EXPECTED_WINDOWS['test']),
            'panel_sha_ok': (panel_sha is not None and fe.get('panel_sha256') == panel_sha
                             and tr.get('panel_sha256') == panel_sha),
            'config_ok': all(cfg_ok.values()), 'config_detail': cfg_ok,
            'seed_ok': args.get('seed') is not None and f"s{args.get('seed')}" == os.path.basename(run_dir),
        },
    }


def read(root: str, seeds=SEEDS, panel_sha: str | None = None) -> dict:
    per = {s: seed_record(os.path.join(root, f's{s}'), panel_sha) for s in seeds}
    found = [s for s in seeds if per[s] is not None]
    res = {'preregistration': PREREG, 'seeds': list(seeds), 'seeds_found': found,
           'missing': [s for s in seeds if per[s] is None], 'per_seed': {f's{s}': per[s] for s in found},
           'label': 'TAMAM' if len(found) == len(seeds) else 'EKSIK'}
    sr = {s: per[s]['test_sharpe_2020'] for s in found}
    usable = [s for s in found if sr[s] is not None and np.isfinite(sr[s])]
    res['n_degenerate'] = int(sum(per[s]['degenerate'] for s in found))
    res['n_test_sharpe_undefined'] = len(found) - len(usable)
    checks = {k: all(per[s]['checks'][k] for s in found) if found else False
              for k in ('train_clean', 'eval_clean', 'val_window_ok', 'test_window_ok', 'panel_sha_ok', 'config_ok',
                        'seed_ok')}
    res['checks_all'] = checks
    res['commits'] = sorted({c for s in found for c in (per[s]['checks']['train_commit'],
                                                        per[s]['checks']['eval_commit']) if c})
    if len(usable) < 2:
        return {**res, 'label': 'EKSIK'}
    x = np.array([sr[s] for s in usable])
    v = [per[s]['val_sharpe_selected'] for s in usable if per[s]['val_sharpe_selected'] is not None]
    sigma = float(x.std(ddof=1))
    res['test_sharpe_2020'] = {'mean': float(x.mean()), 'sd_seed': sigma, 'min': float(x.min()),
                               'max': float(x.max()), 'n': int(x.size)}
    res['val_sharpe_selected'] = ({'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)}
                                  if len(v) >= 2 else None)
    res['sigma_seed'] = sigma
    if len(usable) == len(seeds) == 6:
        D = np.array([np.mean([sr[s] for s in a]) - np.mean([sr[s] for s in b]) for a, b in splits_3v3(usable)])
        absD = np.abs(D)
        res['null_3v3'] = {'n_splits': int(D.size), 'D': D.tolist(), 'abs_max': float(absD.max()),
                           'abs_quartiles': [float(q) for q in np.quantile(absD, [0.25, 0.5, 0.75])],
                           'note': 'isaret keyfi (A ilk tohumu icerir); 10 bolme bagimsiz degil'}
    else:
        res['null_3v3'] = None
    res['mde'] = {'assumption': 'normal yaklasim; sigma_fark = sqrt(2) sigma_tohum; kollar ve katlar bagimsiz, '
                                'katlar arasi ayni varyans (beyan)', 'alpha_one_sided': 0.05, 'power': 0.80,
                  'folds': FOLDS_M8,
                  'table': [{'k': k, 'n_pairs': k * FOLDS_M8, 'mde_sharpe': mde(sigma, k)} for k in K_GRID]}
    if res['label'] == 'TAMAM' and (res['n_test_sharpe_undefined'] or not all(checks.values())):
        res['label'] = 'TAMAM_UYARILI'
    return res


def panel_sha_from_json(path: str = PANEL_JSON) -> str | None:
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as fh:
        return json.load(fh).get('npz', {}).get('sha256')


def self_test() -> bool:
    rng = np.random.default_rng(0)
    ok = total = 0

    def check(name, good):
        nonlocal ok, total
        total += 1
        ok += bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name}")

    val_d = [f'2019-{m:02d}-15' for m in range(1, 13)]
    test_d = np.array([f'2020-{1 + i // 21:02d}-{1 + i % 21:02d}' for i in range(252)])

    def build(edges, degenerate=(), dirty=(), skip=(), flat=(), bad_cfg=(), test_dates=test_d):
        root = tempfile.mkdtemp(prefix='m8noise_')
        for s, e in zip(SEEDS, edges):
            if s in skip:
                continue
            d = os.path.join(root, f's{s}')
            os.makedirs(os.path.join(d, 'eval'))
            X = np.zeros(252) if s in flat else rng.normal(e, 0.01, 252)
            R = np.zeros(252) if s in flat else X + 0.0004
            np.savez(os.path.join(d, 'eval', 'test_argmax.npz'), policy_R=R, policy_X=X, policy_date=test_dates)
            args = {'frames': 600_000, 'checkpoints': 10, 'cost_bps': 20.0, 'lr': 5e-4, 'batched_env': True,
                    'train_end': '2018-12-31', 'seed': s}
            if s in bad_cfg:
                args['frames'] = 300_000
            with open(os.path.join(d, 'train.json'), 'w', encoding='utf-8') as fh:
                json.dump({'commit': 'abc', 'git_status_porcelain': ' M x.py' if s in dirty else '', 'args': args,
                           'panel_sha256': 'P'}, fh)
            sel = {'step': 60000, 'metric': None if s in degenerate else 0.5, 'degenerate': s in degenerate,
                   'n_valid': 0 if s in degenerate else 10, 'n_distinct': 10, 'last_metric': 0.4}
            with open(os.path.join(d, 'eval', 'fold_eval.json'), 'w', encoding='utf-8') as fh:
                json.dump({'windows': {'val': [val_d[0], val_d[-1]]}, 'panel_sha256': 'P', 'selection': sel,
                           'tests': {'argmax': {'file': 'test_argmax.npz'}},
                           'provenance': {'commit': 'abc', 'git_status_porcelain': ''}}, fh)
        return root

    # Sharpe'i bilinen seriler: X'i dogrudan kur (ortalama / sd sabit) -> beklenen Sharpe tam hesaplanir
    def build_exact(sharpes, **kw):
        root = build([0.0] * 6, **kw)
        for s, target in zip(SEEDS, sharpes):
            p = os.path.join(root, f's{s}', 'eval', 'test_argmax.npz')
            if not os.path.exists(p) or s in kw.get('flat', ()):
                continue
            z = rng.normal(0, 1, 252)
            z = (z - z.mean()) / z.std(ddof=1)
            X = 0.01 * z + target * 0.01 / np.sqrt(252)
            np.savez(p, policy_R=X + 0.0004, policy_X=X, policy_date=kw.get('test_dates', test_d))
        return root

    target = [0.2, 0.5, 0.8, 1.1, 1.4, 1.7]
    r = read(build_exact(target), panel_sha='P')
    sig = float(np.std(target, ddof=1))
    check('Sharpe tanimi: tohum basina 2020 Sharpe hedefle ayni (1e-9)',
          all(abs(r['per_seed'][f's{s}']['test_sharpe_2020'] - t) < 1e-9 for s, t in zip(SEEDS, target)))
    check('sigma_tohum = ornek sd (ddof 1)', abs(r['sigma_seed'] - sig) < 1e-9)
    check('3/3 bolme: 10 ayrik bolme, |D| ust siniri dogru (en buyuk = {1.1,1.4,1.7} - {0.2,0.5,0.8} = 0.9)',
          r['null_3v3']['n_splits'] == 10 and abs(r['null_3v3']['abs_max'] - 0.9) < 1e-9)
    m6 = r['mde']['table'][0]
    check('MDE formulu: k 6 -> (z95 + z80) sqrt2 sigma / 6; k arttikca azalir',
          abs(m6['mde_sharpe'] - (Z_ONE_SIDED + Z_POWER80) * math.sqrt(2) * sig / 6) < 1e-12
          and all(a['mde_sharpe'] > b['mde_sharpe'] for a, b in zip(r['mde']['table'], r['mde']['table'][1:])))
    check('temiz kayit + dogru pencere/panel/yapilandirma: TAMAM, tum denetimler dogru',
          r['label'] == 'TAMAM' and all(r['checks_all'].values()))
    check('bolme listesi: her bolme tum tohumlari bir kez kapsar, tekrar yok',
          len({tuple(a) for a, _ in splits_3v3(SEEDS)}) == 10
          and all(sorted(a + b) == list(SEEDS) for a, b in splits_3v3(SEEDS)))
    r = read(build_exact(target, dirty=(103,), degenerate=(104,)), panel_sha='P')
    check('kirli egitim kaydi + 1 degenerate: TAMAM_UYARILI, n_degenerate 1, train_clean False',
          r['label'] == 'TAMAM_UYARILI' and r['n_degenerate'] == 1 and not r['checks_all']['train_clean'])
    r = read(build_exact(target, skip=(106,)), panel_sha='P')
    check('eksik tohum: EKSIK, 3/3 bolme yok, sigma 5 tohumdan',
          r['label'] == 'EKSIK' and r['missing'] == [106] and r['null_3v3'] is None
          and abs(r['sigma_seed'] - float(np.std(target[:5], ddof=1))) < 1e-9)
    r = read(build_exact(target, flat=(102,)), panel_sha='P')
    check('islem yok (duz seri): Sharpe tanimsiz sayilir, 3/3 bolme yok',
          r['n_test_sharpe_undefined'] == 1 and r['null_3v3'] is None)
    r = read(build_exact(target, bad_cfg=(101,)), panel_sha='P')
    check('on kayit disi yapilandirma (F 300K): config_ok False, TAMAM_UYARILI',
          not r['checks_all']['config_ok'] and r['label'] == 'TAMAM_UYARILI')
    r = read(build_exact(target, test_dates=np.array([f'2021-01-{i % 28 + 1:02d}' for i in range(252)])),
             panel_sha='P')
    check('orneklem disi pencere 2020 degil (2021 okundu): test_window_ok False',
          not r['checks_all']['test_window_ok'] and r['label'] == 'TAMAM_UYARILI')
    r = read(build_exact(target), panel_sha='Q')
    check('panel SHA farkli: panel_sha_ok False', not r['checks_all']['panel_sha_ok'])
    print(f'Sonuc: {ok}/{total}')
    return ok == total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join('results', 'marl_m8_noise'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    root = a.root if os.path.isabs(a.root) else os.path.join(ROOT, a.root)
    res = {**read(root, panel_sha=panel_sha_from_json()), 'provenance': provenance()}
    print(json.dumps({k: res.get(k) for k in ('label', 'missing', 'n_degenerate', 'n_test_sharpe_undefined',
                                              'checks_all', 'test_sharpe_2020', 'val_sharpe_selected', 'sigma_seed',
                                              'null_3v3', 'mde')}, indent=1, ensure_ascii=False, default=float))
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
