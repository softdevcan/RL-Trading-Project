"""Faz 10 M7b: dikkat testleri (venv-marl; yalniz degerlendirme, egitim yok). BETIMSEL tani, karar yok (M6 tasarimi §3.2, §7).

Kayitli iletisimli politikalarda (gat/attn; M7a ile ayni 21 politika) deterministik tam pencere degerlendirmesi:
  base     : degisiklik yok; kayitli `eval_<pencere>.npz` ile bit-es olmali (dogru politika + kurulum kaniti)
  uniform  : TEK BICIMLI DIKKAT baseline'i (Wiegreffe & Pinter 2019): alici izinli gondericileri esit agirlikla ortalar
             (`CommPolicy.set_uniform_attention`). Fark kucukse ogrenilen agirliklar kararin aciklamasi sayilamaz.
  perm:<r> : KARISTIRILMIS MESAJ (M7a maskelemenin dagilimi koruyan kontrolu): yuva j, sabit bir derangement'a gore
             gonderici pi(j)'nin mesajini (ve guven terimini) tasir (`set_sender_permutation`). Her alici yine gercek bir
             mesaj gorur; mesaj ile kimlik arasindaki bag kopar. M7a 'all' (kapali mesaj) ile yan yana okunur.
  record   : dikkat agirliklari (basa gore ortalama) kaydedilir -> gunler uzerinden ortalama matris W [N, N],
             normalize entropi (1 = tek bicimli) ve tek bicimliye gore KL.
Kol duzeyi (ayni kolun tohumlari): ortalama dikkat matrislerinin izinli, kosegen disi girislerinde tohum ciftleri
arasi Pearson korelasyonu (TOHUMLAR ARASI DIKKAT VARYANSI). Dusuk korelasyon + benzer basarim = dikkat kararlilik
gostermiyor. Dikkat nedensel bir aciklama sayilmaz (Jain & Wallace 2019; M6 tasarimi §4).

    venv-marl\\Scripts\\python scripts/marl/attn_m7b.py --runs-from docs/development/measurements/phase10-m7/mask_train.json \\
        --window train --out docs/development/measurements/phase10-m7/m7b_train.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'marl'))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from mask_m7 import PANEL, _git, _sha, sharpe  # noqa: E402

N_PERM = 3
PERM_SEED0 = 7000     # derangement tohumlari 7000, 7001, ... (egitim tohumlarindan ayri; sabit)


def attention_summary(W_days: np.ndarray, allowed: np.ndarray) -> dict:
    """W_days [T, N, N] (satirlar 1'e toplanir) -> gunler uzerinden ortalama, normalize entropi, tek bicimliye KL."""
    W = W_days.mean(axis=0)
    k = allowed.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.where(W_days > 0, W_days * np.log(W_days), 0.0).sum(axis=-1)       # [T, N]
        norm_ent = np.where(k > 1, ent / np.log(k), 1.0)
        kl = np.log(k)[None, :] - ent
    return {'mean_matrix': W.round(6).tolist(), 'norm_entropy_mean': float(np.nanmean(norm_ent)),
            'kl_to_uniform_mean': float(np.nanmean(kl))}


def run_one(run_dir: str, window: str, arrays, n_perm: int = N_PERM) -> dict:
    from eval_m3 import window_rows
    from marl import obs_spec as OS
    from marl.comm_models import CommPolicy, derangement
    from marl.m3_runner import deterministic_rollout, load_policy

    s, e = window_rows(arrays['dates'], window, OS.WARMUP_ROWS)
    exp, meta = load_policy(run_dir)
    cfg = {k: v for k, v in meta['task_cfg'].items() if k != 'use_state'}
    cfg.update(mode='eval', start_row=s, end_row=e, norm_stats_path=os.path.join(run_dir, 'norm_stats.json'))
    comm = [m for m in exp.policy.modules() if isinstance(m, CommPolicy)]
    if len(comm) != 1:
        raise ValueError(f"{run_dir}: iletisimli politika degil (CommPolicy {len(comm)})")
    comm = comm[0]
    N = comm.n_agents
    stored = np.load(os.path.join(run_dir, f'eval_{window}.npz'), allow_pickle=True)
    res = {'run': run_dir.replace('\\', '/'), 'arm': os.path.dirname(run_dir).replace('\\', '/'), 'kind': comm.kind,
           'conf_mode': comm.conf_mode, 'window': window, 'rows': [int(s), int(e)],
           'policy_sha256': meta.get('policy_sha256'), 'variants': {}}
    t0 = time.time()

    def roll(name, setup, teardown):
        setup()
        ser = deterministic_rollout(exp.policy, cfg)
        teardown()
        return {'sharpe_X': sharpe(ser['X']), 'turnover_mean': float(np.mean(ser['turnover']))}, ser

    v, ser = roll('base', lambda: None, lambda: None)
    v['bit_equal_stored_eval'] = bool(np.array_equal(ser['X'], stored['policy_X']))
    base_sr = v['sharpe_X']
    res['variants']['base'] = v
    v, _ = roll('uniform', lambda: comm.set_uniform_attention(True), lambda: comm.set_uniform_attention(False))
    v['delta_sharpe_X'] = v['sharpe_X'] - base_sr
    res['variants']['uniform'] = v
    deltas = []
    for r in range(n_perm):
        perm = [int(j) for j in derangement(N, PERM_SEED0 + r)]
        v, _ = roll(f'perm:{r}', lambda p=perm: comm.set_sender_permutation(p), lambda: comm.set_sender_permutation(None))
        v['delta_sharpe_X'] = v['sharpe_X'] - base_sr
        v['perm_seed'] = PERM_SEED0 + r
        res['variants'][f'perm:{r}'] = v
        deltas.append(v['delta_sharpe_X'])
    res['perm_delta_mean'] = float(np.mean(deltas))
    comm.set_attention_record(True)
    deterministic_rollout(exp.policy, cfg)
    W_days = np.concatenate([w.numpy().reshape(-1, N, N) for w in comm.attention_log], axis=0)
    comm.set_attention_record(False)
    bm = comm._bool_mask()
    allowed = np.ones((N, N), dtype=bool) if bm is None else ~bm.cpu().numpy()
    res['attention'] = {**attention_summary(W_days, allowed), 'n_days': int(W_days.shape[0]),
                        'allowed': allowed.astype(int).tolist()}
    res['seconds'] = round(time.time() - t0, 1)
    return res


def cross_seed(runs: list) -> dict:
    """Ayni kolun tohumlari arasinda ortalama dikkat matrislerinin benzerligi (izinli, kosegen disi girisler)."""
    out = {}
    by_arm: dict = {}
    for r in runs:
        by_arm.setdefault(r['arm'], []).append(r)
    for arm, rs in sorted(by_arm.items()):
        allowed = np.array(rs[0]['attention']['allowed'], dtype=bool)
        off = allowed & ~np.eye(len(allowed), dtype=bool)
        mats = [np.array(r['attention']['mean_matrix'])[off] for r in rs]
        corr = [float(np.corrcoef(a, b)[0, 1]) for a, b in itertools.combinations(mats, 2)]
        stack = np.stack(mats)
        cv = stack.std(axis=0, ddof=1) / np.maximum(stack.mean(axis=0), 1e-12) if len(mats) > 1 else None
        out[arm] = {'n_seeds': len(rs), 'pairwise_corr': corr, 'pairwise_corr_mean': float(np.mean(corr)) if corr else None,
                    'entry_cv_median': None if cv is None else float(np.median(cv)),
                    'norm_entropy_mean': [r['attention']['norm_entropy_mean'] for r in rs],
                    'delta_uniform': [r['variants']['uniform']['delta_sharpe_X'] for r in rs],
                    'delta_perm_mean': [r['perm_delta_mean'] for r in rs]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--runs-from', default=None, help='M7a olcum JSON\'u (ayni politika listesi)')
    ap.add_argument('--window', choices=('train', 'val'), default='train')
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from marl import panel as PN

    runs = list(a.runs or [])
    mask_ref = {}
    if a.runs_from:
        with open(a.runs_from, encoding='utf-8') as fh:
            m7a = json.load(fh)
        runs += [r['run'] for r in m7a['runs']]
        mask_ref = {r['run']: r['variants']['all']['delta_sharpe_X'] for r in m7a['runs']}
    if not runs:
        raise SystemExit('--runs ya da --runs-from gerekli')
    arrays = PN.load_npz(PANEL)
    out = {'note': 'M7b dikkat testleri (BETIMSEL tani; egitim yok, karar yok)', 'window': a.window, 'n_perm': a.n_perm,
           'perm_seeds': [PERM_SEED0 + r for r in range(a.n_perm)], 'runs': []}
    for r in runs:
        res = run_one(r, a.window, arrays, a.n_perm)
        res['m7a_delta_all'] = mask_ref.get(res['run'])
        out['runs'].append(res)
        v = res['variants']
        print(res['run'], 'bit-es', v['base']['bit_equal_stored_eval'], 'uniform', round(v['uniform']['delta_sharpe_X'], 3),
              'perm', round(res['perm_delta_mean'], 3), 'M7a all', res['m7a_delta_all'] and round(res['m7a_delta_all'], 3),
              'entropi', round(res['attention']['norm_entropy_mean'], 3), 'sn', res['seconds'], flush=True)
    out['cross_seed'] = cross_seed(out['runs'])
    out['provenance'] = {'commit': _git('rev-parse', 'HEAD'), 'git_status_porcelain': _git('status', '--porcelain'),
                         'script_sha256': _sha(os.path.abspath(__file__)),
                         'comm_models_sha256': _sha(os.path.join(ROOT, 'marl', 'comm_models.py')),
                         'panel_sha256': arrays['sha256']}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
