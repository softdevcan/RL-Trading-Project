"""Faz 11 S3: simulasyon defteri seri paritesi (ANA venv). Plan phase-11-integration-plan.md §3 S3 kabul kapisi:

  - Her dondurulmus politika (72) icin 2026 test penceresi `marl.sim_ledger` defterinden yeniden oynatilir; gunluk R
    serisi M8 test kaydiyla (results/marl_m8/2026/<kol>/<tohum>/eval/test_argmax.npz, `eval_fold_m8.py`, venv-marl)
    bit-es ya da <= 1e-12. X, devir ve nakit agirligi da karsilastirilir (bilgi; kapi R).
  - M8 test kaydinin SHA'si arsivdeki fold_eval.json kaydiyla ayni olmali (karsilastirilan seri M8'in okudugu seri).
  - Kural portfoyleri (ew, bh, minvar, hrp): defter serisi `rule_baselines.rule_series` ile bit-es ve Sharpe
    `m8_reading.json` -> baselines.2026 ile ayni (analyze_m8.cell_metrics tanimi: excess_sharpe(R, rf = R - X)).
  - Once arsiv dogrulamasi (verify_archive, on kayit listesi).

    python scripts/marl/s3_parity.py [--arm M-none --seed s201] [--out docs/development/measurements/phase11-s3/s3_parity.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from marl import frozen as FZ  # noqa: E402
from marl import panel as PN  # noqa: E402
from marl import rule_baselines as RB  # noqa: E402
from marl import sim_ledger as SL  # noqa: E402
from marl.selection import excess_sharpe  # noqa: E402

ARCHIVE = os.path.join(FZ.FROZEN_ROOT, 'm8-2026')
M8 = os.path.join('results', 'marl_m8', '2026')
READING = os.path.join('docs', 'development', 'measurements', 'phase10-m8', 'm8_reading.json')
PANEL = os.path.join('data', 'marl', 'panel_m8.npz')
TOL = 1e-12
CMP = ('R', 'X', 'turnover', 'cash_weight')


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def diff(a, b) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {'shape_mismatch': [list(a.shape), list(b.shape)], 'max_abs': float('inf'), 'bitwise_equal': False}
    return {'max_abs': float(np.max(np.abs(a - b))) if a.size else 0.0, 'bitwise_equal': bool(np.array_equal(a, b))}


def policy_parity(mk: SL.Market, p: dict) -> dict:
    d = os.path.join(ARCHIVE, p['arm'], p['seed'])
    with open(os.path.join(d, 'fold_eval.json'), encoding='utf-8') as fh:
        fe = json.load(fh)
    if fe['panel_sha256'] != mk.panel_sha256:
        raise ValueError(f"panel SHA farkli: {p['arm']}/{p['seed']}")
    ts, te = fe['rows']['test']
    t0 = time.perf_counter()
    st = SL.PolicyStrategy(d, p['ckpt'], sid=f"policy/{p['arm']}/{p['seed']}")
    _, rows = SL.run(mk, st, SL.LedgerState.start(ts, mk.N), te)
    sec = time.perf_counter() - t0
    ser = SL.series(rows)
    path = os.path.join(M8, p['arm'], p['seed'], 'eval', 'test_argmax.npz')
    m8 = np.load(path)
    out = {'arm': p['arm'], 'seed': p['seed'], 'kind': st.policy.kind, 'n_days': len(rows),
           'window': [ser['date'][0], ser['date'][-1]], 'seconds': round(sec, 2),
           'm8_test_sha_matches_archive': _sha(path) == fe['tests']['argmax']['sha256'],
           'dates_equal': [str(x) for x in m8['policy_date']] == ser['date']}
    out.update({k: diff(ser[k], m8[f'policy_{k}']) for k in CMP})
    out['pass'] = bool(out['m8_test_sha_matches_archive'] and out['dates_equal'] and out['R']['max_abs'] <= TOL)
    return out


def rule_parity(mk: SL.Market, ts: int, te: int, reading: dict) -> list:
    res = []
    for kind in ('ew', 'bh', 'minvar', 'hrp'):
        _, rows = SL.run(mk, SL.RuleStrategy(kind), SL.LedgerState.start(ts, mk.N), te)
        ser = SL.series(rows)
        ref = RB.rule_series(mk.arrays, ts, te, kind, SL.COST_BPS_ROUNDTRIP)
        sr = excess_sharpe(ser['R'], rf=ser['R'] - ser['X'])
        want = reading['baselines']['2026'][kind]['sr']
        r = {'kind': kind, 'n_days': len(rows), 'sr': sr, 'sr_m8_reading': want, 'sr_equal': sr == want,
             'dates_equal': ref['date'] == ser['date']}
        r.update({k: diff(ser[k], ref[k]) for k in CMP})
        r['pass'] = bool(r['sr_equal'] and r['dates_equal'] and r['R']['max_abs'] <= TOL)
        res.append(r)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default=None)
    ap.add_argument('--seed', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(READING, encoding='utf-8') as fh:
        reading = json.load(fh)
    ver = FZ.verify_archive(ARCHIVE, forward=reading['forward_policies_2026'])
    with open(os.path.join(ARCHIVE, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    mk = SL.Market(PN.load_npz(PANEL))
    todo = [p for p in man['policies'] if (a.arm in (None, p['arm'])) and (a.seed in (None, p['seed']))]
    pol = []
    for p in todo:
        r = policy_parity(mk, p)
        pol.append(r)
        print(f"{r['arm']:7s} {r['seed']} {r['kind']:9s} {r['n_days']} gun | R {r['R']['max_abs']:.1e} "
              f"(bit-es {r['R']['bitwise_equal']}) X {r['X']['max_abs']:.1e} devir {r['turnover']['max_abs']:.1e} | "
              f"{r['seconds']} sn | {'GECTI' if r['pass'] else 'KALDI'}")
    with open(os.path.join(ARCHIVE, todo[0]['arm'], todo[0]['seed'], 'fold_eval.json'), encoding='utf-8') as fh:
        ts, te = json.load(fh)['rows']['test']
    rules = rule_parity(mk, ts, te, reading)
    for r in rules:
        print(f"kural {r['kind']:6s} SR {r['sr']:.6f} (M8 okumasi {r['sr_m8_reading']:.6f}, esit {r['sr_equal']}) "
              f"R {r['R']['max_abs']:.1e} | {'GECTI' if r['pass'] else 'KALDI'}")
    summary = {'n_policies': len(pol), 'n_pass': sum(r['pass'] for r in pol), 'tolerance_R': TOL,
               'max_abs': {k: max(r[k]['max_abs'] for r in pol) for k in CMP},
               'n_bitwise_equal': {k: sum(r[k]['bitwise_equal'] for r in pol) for k in CMP},
               'n_m8_sha_ok': sum(r['m8_test_sha_matches_archive'] for r in pol),
               'by_kind': {k: sum(r['kind'] == k for r in pol) for k in sorted({r['kind'] for r in pol})},
               'seconds_per_policy_mean': round(float(np.mean([r['seconds'] for r in pol])), 2),
               'rules_pass': sum(r['pass'] for r in rules), 'n_rules': len(rules)}
    print('ozet:', summary)
    if a.out:
        import subprocess
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump({'plan': 'phase-11-integration-plan.md §3 S3', 'archive': 'm8-2026', 'panel': 'panel_m8.npz',
                       'panel_sha256': mk.panel_sha256, 'window_rows': [ts, te],
                       'written': datetime.now().isoformat(timespec='seconds'), 'commit': head,
                       'git_dirty': bool(dirty), 'archive_verify': ver,
                       'reference': 'M8 test kaydi (eval_fold_m8.py, venv-marl TorchRL ortami) + rule_series',
                       'summary': summary, 'rules': rules, 'policies': pol}, fh, indent=1, ensure_ascii=False)
    ok = summary['n_pass'] == summary['n_policies'] and summary['rules_pass'] == summary['n_rules']
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
