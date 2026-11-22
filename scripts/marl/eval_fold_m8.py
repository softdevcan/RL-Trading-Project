"""Faz 10 M8 (tasarim A1): bir egitim kosusunun kat degerlendirmesi (venv-marl). Test YALNIZ secilen modelle, BIR KEZ.

Sira: `train_m3.py --checkpoints K` kontrol noktalari dogrulama yilinda deterministik tek gecisle olculur; metrik
`marl.selection.excess_sharpe` (`rl_folds.excess_sharpe` ile parite testli, test_marl_core [8]); secim
`marl.selection.select_checkpoint` (en yuksek, esitlikte en erken; hicbiri tanimli degilse son + degenerate).
Sonra test yili YALNIZ secilen agirliklarla okunur. `--also-last` (K-M8-4 etkeni, on kayitta karar): son nokta
ayri etiketli ikinci test kaydi. Cikti dizininde test dosyasi varsa betik DURUR (test ikinci kez okunmaz).

Pencereler tarihle verilir (wf-v1 kati; tanim ana venv'de `evaluation/splits.py`, kayit okumasinda denetlenir):
dogrulama karar satirlari [val_start'tan onceki son satir, test_start'tan onceki son satir), test ayni kuralla
[test_start, test_end). Karar t kapanisi, yurutme t+1 acilisi (eval_m3 pencereleriyle ayni satir kurali).

  venv-marl\\Scripts\\python scripts/marl/eval_fold_m8.py --run-dir <kosu> --fold 2021 --val-start 2020-01-01 \\
      --test-start 2021-01-01 --test-end 2022-01-01 --out-dir <dizin>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def provenance() -> dict:
    git = lambda *c: subprocess.run(['git', *c], capture_output=True, text=True).stdout.strip()   # noqa: E731
    with open(os.path.abspath(__file__), 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'git_status_porcelain': git('status', '--porcelain'), 'script_sha256': sha}


def decision_rows(dates, start: str, end: str) -> tuple:
    """[start, end) yurutme gunleri icin karar satirlari: (start'tan onceki son satir, end'den onceki son satir)."""
    import numpy as np
    before_s, before_e = np.flatnonzero(dates < start), np.flatnonzero(dates < end)
    if before_s.size == 0 or before_e.size == 0 or before_e[-1] <= before_s[-1]:
        raise ValueError(f"pencere panelde yok: {start}..{end}")
    return int(before_s[-1]), int(before_e[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--fold', required=True, help='kat adi (wf-v1: test yili)')
    ap.add_argument('--val-start', required=True)
    ap.add_argument('--test-start', required=True)
    ap.add_argument('--test-end', required=True)
    ap.add_argument('--also-last', action='store_true', help='K-M8-4: son kontrol noktasi ayri test kaydi')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    import numpy as np
    import torch

    from marl import panel as PN
    from marl.m3_runner import deterministic_rollout, load_policy
    from marl.selection import excess_sharpe, select_checkpoint

    with open(os.path.join(a.run_dir, 'ckpt', 'index.json'), encoding='utf-8') as fh:
        index = json.load(fh)
    tests = ['argmax'] + (['last'] if a.also_last else [])
    os.makedirs(a.out_dir, exist_ok=True)
    for t in tests:
        if os.path.exists(os.path.join(a.out_dir, f'test_{t}.npz')):
            raise SystemExit(f"test zaten okundu: {a.out_dir}/test_{t}.npz (ikinci okuma yok)")
    exp, meta = load_policy(a.run_dir)
    cfg = {k: v for k, v in meta['task_cfg'].items() if k != 'use_state'}
    cfg.update(mode='eval', norm_stats_path=os.path.join(a.run_dir, 'norm_stats.json'))
    arrays = PN.load_npz(cfg['panel_path'])
    vs, ve = decision_rows(arrays['dates'], a.val_start, a.test_start)
    ts, te = decision_rows(arrays['dates'], a.test_start, a.test_end)
    if vs < int(meta['task_cfg']['end_row']):
        raise SystemExit(f"dogrulama egitimle ortusuyor: {arrays['dates'][vs]} < egitim sonu")

    def rollout(file, s, e):
        exp.policy.load_state_dict(torch.load(os.path.join(a.run_dir, 'ckpt', file), map_location='cpu'))
        return deterministic_rollout(exp.policy, {**cfg, 'start_row': s, 'end_row': e})

    records = []
    for c in index:
        ser = rollout(c['file'], vs, ve)
        R, X = np.asarray(ser['R']), np.asarray(ser['X'])
        records.append({'step': c['frames'], 'iter': c['iter'], 'file': c['file'], 'params_sha': c['params_sha'],
                        'metric': excess_sharpe(R, rf=R - X), 'mean_excess': float(np.mean(X))})
    chosen = select_checkpoint(records)
    picks = {'argmax': chosen, 'last': max(records, key=lambda r: r['step'])}
    out = {'fold': a.fold, 'windows': {'val': [str(arrays['dates'][vs + 1]), str(arrays['dates'][ve])],
                                       'test': [str(arrays['dates'][ts + 1]), str(arrays['dates'][te])]},
           'rows': {'val': [vs, ve], 'test': [ts, te]}, 'panel_sha256': arrays['sha256'],
           'policy_sha256_final': meta.get('policy_sha256'), 'validation': records, 'selection': chosen, 'tests': {}}
    for t in tests:
        p = picks[t]
        ser = rollout(p['file'], ts, te)
        path = os.path.join(a.out_dir, f'test_{t}.npz')
        np.savez_compressed(path, **{f'policy_{k}': np.asarray(v) for k, v in ser.items()})
        with open(path, 'rb') as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        out['tests'][t] = {'file': os.path.basename(path), 'sha256': sha, 'ckpt': p['file'], 'step': p['step'],
                           'same_as_argmax': p['file'] == chosen['file']}
    out['provenance'] = provenance()
    with open(os.path.join(a.out_dir, 'fold_eval.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
    print(f"kat {a.fold}: {len(records)} kontrol noktasi, secilen {chosen['file']} (adim {chosen['step']}, "
          f"degenerate {chosen['degenerate']}); test {', '.join(tests)} -> {a.out_dir}")


if __name__ == '__main__':
    main()
