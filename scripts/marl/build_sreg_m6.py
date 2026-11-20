"""Faz 10 M6b: S-reg yayin blogu (ana venv, ag yok). Tum ajanlara ve elestirmene giden nedensel rejim ozellikleri.

Plan (Faz 10 §3.1.3 S-reg, K10; M6 tasarimi K-M6-3): B6 kural tabanli rejim ozellikleri, yalniz <= t veri:
  sma_ratio : XU100/USDTRY_{t-1} kapanisinin 200 gunluk SMA'ya orani - 1 (egilim; isaret = boga/ayi)
  vol       : 60 gunluk log getiri std'si x sqrt(252)
Kaynak `evaluation.regimes.load_primary_labels` (birincil etiket; ayni seri). As-of hizalama (`asof_labels`): panel gunu t
icin endeksin t'ye kadarki son islem gununun degeri (etiket t kapanisinda bilinir; karar t kapanisinda). Olcek YALNIZ
egitim satirlarindan (WARMUP..2018-12-31; eval_m3 ile ayni pencere), +-5'te kirpilir (obs_spec.CLIP). Degeri olmayan gun
(isinma) -> hata (MARL paneli 2013-06'da basliyor, etiket 2005'ten; bosluk beklenmez).

Cikti: data/marl/sreg_m6.npz (dates, names, extras [T, 2] float64, mean, std) + kayit JSON (SHA, olcek, kapsam, B0).

    python scripts/marl/build_sreg_m6.py --json docs/development/measurements/phase10-m6/sreg_block.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from evaluation import regimes as R  # noqa: E402
from marl.obs_spec import CLIP, WARMUP_ROWS  # noqa: E402

PANEL = os.path.join('data', 'marl', 'panel_m2.npz')
OUT = os.path.join('data', 'marl', 'sreg_m6.npz')
NAMES = ('sma_ratio', 'vol')
TRAIN_END = '2018-12-31'


def raw_features(labels: pd.DataFrame, dates) -> np.ndarray:
    cols = [R.asof_labels(labels, dates, n).astype(float).to_numpy() for n in NAMES]
    return np.stack(cols, axis=1)


def train_rows(dates: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros(len(dates), dtype=bool)
    m[WARMUP_ROWS:] = True
    return m & (dates <= pd.Timestamp(TRAIN_END))


def build(labels: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    raw = raw_features(labels, dates)
    if not np.isfinite(raw).all():
        raise ValueError(f"S-reg ozelliginde deger olmayan panel gunu var: {int((~np.isfinite(raw)).any(1).sum())}")
    tr = train_rows(dates)
    mean, std = raw[tr].mean(axis=0), raw[tr].std(axis=0, ddof=1)
    z = np.clip((raw - mean) / std, -CLIP, CLIP)
    return {'extras': z, 'mean': mean, 'std': std, 'raw': raw, 'train_rows': tr}


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git(*args) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    z = np.load(PANEL, allow_pickle=True)
    dates = pd.DatetimeIndex(pd.to_datetime(z['dates'].astype(str)))
    labels = R.load_primary_labels()
    b = build(labels, dates)
    # B0: etiketi t'de kesilmis seriyle kurulan ham ozellik, t'ye kadar ayni (5 kesim)
    b0 = []
    for c in (dates[100], dates[600], dates[1343], dates[1600], dates[-2]):
        from data.index_fetcher import DEFAULT_INDEX_CSV, DEFAULT_USDTRY_CSV, FX_JUMP_LIMIT, load_index_close
        usd = R.usd_index(load_index_close(DEFAULT_INDEX_CSV), load_index_close(DEFAULT_USDTRY_CSV, jump_limit=FX_JUMP_LIMIT))
        lab_c = R.regime_labels(usd[usd.index <= c])
        d_c = dates[dates <= c]
        b0.append(bool(np.array_equal(raw_features(lab_c, d_c), b['raw'][:len(d_c)])))
    np.savez_compressed(a.out, dates=z['dates'], names=np.array(NAMES), extras=b['extras'], mean=b['mean'], std=b['std'])
    tr = b['train_rows']
    rec = {'note': 'M6b S-reg yayin blogu (tasarim girdisi); performans sayisi YOK', 'names': list(NAMES),
           'out': a.out.replace('\\', '/'), 'out_sha256': _sha(a.out), 'n_dates': int(len(dates)),
           'n_train_rows': int(tr.sum()), 'mean': b['mean'].tolist(), 'std': b['std'].tolist(),
           'clip_frac': {n: float((np.abs(b['extras'][:, j]) >= CLIP).mean()) for j, n in enumerate(NAMES)},
           'z_range_val': {n: [float(b['extras'][~tr & (dates > pd.Timestamp(TRAIN_END)), j].min()),
                               float(b['extras'][~tr & (dates > pd.Timestamp(TRAIN_END)), j].max())]
                           for j, n in enumerate(NAMES)},
           'b0_cut_invariance': b0, 'provenance': {'commit': _git('rev-parse', 'HEAD'),
                                                   'git_status_porcelain': _git('status', '--porcelain'),
                                                   'script_sha256': _sha(os.path.abspath(__file__)),
                                                   'panel_sha256': _sha(PANEL)}}
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print({k: rec[k] for k in ('n_dates', 'n_train_rows', 'mean', 'std', 'clip_frac', 'z_range_val', 'b0_cut_invariance')})
    return 0


if __name__ == '__main__':
    sys.exit(main())
