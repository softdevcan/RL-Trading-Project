"""Faz 11 §3.12: M8 ileri veri deposunun makine devri paritesi (ANA venv ya da Docker imaji; ag yok, salt okuma).

Diskteki defteri (data/marl/forward/m8/ledger) ayni ileri panelden ve dondurulmus arsivden GECICI bir dizinde
bastan kurar ve karsilastirir: her strateji satiri (written_at haric) ve durum JSON'u. Gercek depoya, gunluge ve
kilide dokunmaz. Devirden once yerelde, devirden sonra yeni makinede kosulur; ikisi de 'bit_es' demeli. Bit-es
degilse en buyuk mutlak fark raporlanir (hesaplama ortami farki: torch/numpy surumu, islemci) ve devir durur.

    python scripts/marl/handover_parity.py [--out <olcum.json>]
    docker exec rltrading python scripts/marl/handover_parity.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from marl import forward_job as FJ  # noqa: E402
from marl import forward_panel as FP  # noqa: E402
from marl import frozen as FZ  # noqa: E402
from marl import panel as PN  # noqa: E402


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# Devirde tasinan kume (plan §3.12): dondurulmus arsiv + kaynak, ileri panel/defter/cekimler, 2026 test paneli.
HANDOVER = ('frozen/m8-2026', 'frozen/m8-source', 'forward/m8', 'panel_m8.npz')
SKIP = {'.job.lock'}


def tree_digest(root: str = os.path.join('data', 'marl'), parts=HANDOVER) -> dict:
    """Devir kumesindeki her dosyanin SHA-256'si (data/marl'a goreli, '/' ayracli). Iki makinede ayni olmali."""
    out = {}
    for part in parts:
        top = os.path.join(root, part)
        walk = [(os.path.dirname(top), [], [os.path.basename(top)])] if os.path.isfile(top) else os.walk(top)
        for d, _, fs in walk:
            for f in fs:
                if f in SKIP:
                    continue
                p = os.path.join(d, f)
                out[os.path.relpath(p, root).replace(os.sep, '/')] = _sha(p)
    return dict(sorted(out.items()))


def parity(store_dir: str = FJ.LEDGER_DIR) -> dict:
    with open(FJ.READING, encoding='utf-8') as fh:
        fwd = json.load(fh)['forward_policies_2026']
    arch = {k: v for k, v in FZ.verify_archive(FJ.ARCHIVE, forward=fwd).items() if k != 'files'}
    meta, states, rows = FJ.load_store(store_dir)
    if not states:
        return {'status': 'defter_yok', 'archive': arch}
    path = os.path.join(FP.FWD_ROOT, FP.PANEL_FWD)
    arrays = PN.load_npz(path if os.path.exists(path) else FP.PANEL_M8)
    last = str(rows['date'].max())
    arrays = FJ.truncate(arrays, last)
    sids = FJ.strategy_ids()
    with tempfile.TemporaryDirectory() as tmp:
        FJ.advance(arrays, tmp, sids, verified_until=meta.get('verified_until'), now='parity')
        _, st2, rows2 = FJ.load_store(tmp)
    cols = [c for c in rows.columns if c != 'written_at']
    a = rows[cols].sort_values(['sid', 'date']).reset_index(drop=True)
    b = rows2[cols].sort_values(['sid', 'date']).reset_index(drop=True)
    same_shape = a.shape == b.shape and list(a['sid']) == list(b['sid']) and list(a['date']) == list(b['date'])
    num = [c for c in cols if c not in ('sid', 'date') and np.issubdtype(a[c].dtype, np.number)]
    max_abs = 0.0
    if same_shape and num:
        d = np.abs(a[num].to_numpy(dtype=float) - b[num].to_numpy(dtype=float))
        max_abs = float(np.nanmax(d)) if d.size else 0.0
    rows_eq = same_shape and a.equals(b)
    state_eq = {s: states[s].to_json() for s in sids} == {s: st2[s].to_json() for s in sids}
    try:
        import torch
        tv = torch.__version__
    except ImportError:
        tv = None
    return {'status': 'bit_es' if rows_eq and state_eq else 'fark_var', 'rows_equal': rows_eq, 'state_equal': state_eq,
            'same_shape': same_shape, 'max_abs_diff': max_abs, 'n_rows': len(a), 'n_strategies': len(sids),
            'n_sessions': FJ._sessions(states), 'last_date': last, 'archive': arch,
            'env': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': tv,
                    'machine': platform.machine(), 'git_commit': os.environ.get('GIT_COMMIT') or None}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--digest', action='store_true', help='devir kumesindeki (HANDOVER) her dosyanin SHA-256 ozeti de yazilir')
    a = ap.parse_args()
    res = parity()
    if a.digest:
        tree = tree_digest()
        res['digest'] = {'n_files': len(tree), 'sha256_of_list': hashlib.sha256(
            json.dumps(tree, sort_keys=True).encode()).hexdigest()}
        res['files'] = tree
    show = {k: v for k, v in res.items() if k != 'files'}
    print(json.dumps(show, indent=1, ensure_ascii=False, default=str))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False, default=str)
    sys.exit(0 if res['status'] == 'bit_es' else 1)


if __name__ == '__main__':
    main()
