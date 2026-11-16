"""Veri kapisindan gecmis panel -> M2 ortaminin numpy dizileri (ANA venv'de kosar; tasarim §6).

`venv-marl/` icindeki ortam uretim koduna (data/, app/, pandas tabanli kapi) baglanmaz (K9). Bu modul
ana venv'de paneli `data.integrity.prepare_rl_panel` ciktisindan [T, N] dizilere cevirir ve SHA-256'li
bir .npz yazar; ortam yalniz bu dosyayi okur (`load_npz`). M1 spike'inin kalibi, iki farkla:
  - eksik gunler ATILMAZ: hucre NaN kalir, `accounting.fill_prices` maskeler (M1 69 gunu atiyordu)
  - nakit/fazla getiri icin A8 donem basi risksiz getiri `rf` [T] ayni dosyaya yazilir
    (`evaluation.risk_free.risk_free_series`; seri yoksa ACIK HATA, sabite dusulmez)

Sembol sirasi panelin gorunme sirasidir (CLAUDE.md "Sembol SIRASI onemli"); `sorted()` UYGULANMAZ.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Sequence

import numpy as np

from marl.obs_spec import REQUIRED_ARRAYS


def build_arrays(panel, symbols: Optional[Sequence[str]] = None, rf=None) -> dict:
    """(date, symbol) indeksli panel -> {'dates', 'symbols', <REQUIRED_ARRAYS> [T, N], 'rf' [T]}.

    Args:
        panel: `prepare_rl_panel` ciktisi; ADX 'wilder' ile yeniden hesaplanmis olmali
            (`data.technical_indicators.with_adx`). Legacy ADX (hep 0) reddedilir.
        symbols: sira; None -> panelin gorunme sirasi.
        rf: [T] donem basi risksiz getiri (opsiyonel; verilmezse yazilmaz, ortam acikca ister).
    """
    import pandas as pd

    idx_sym = panel.index.get_level_values('symbol')
    order = list(symbols) if symbols is not None else list(pd.unique(idx_sym))
    dates = pd.to_datetime(panel.index.get_level_values('date'))
    if getattr(dates, 'tz', None) is not None:
        dates = dates.tz_localize(None)
    flat = panel.copy()
    flat.index = pd.MultiIndex.from_arrays([dates.normalize(), idx_sym], names=['date', 'symbol'])
    missing = [c for c in REQUIRED_ARRAYS if c not in flat.columns]
    if missing:
        raise ValueError(f"panelde eksik sutun: {missing}")
    out = {}
    cal = None
    for k in REQUIRED_ARRAYS:
        wide = flat[k].unstack('symbol').sort_index().reindex(columns=order)
        cal = wide.index if cal is None else cal
        out[k] = wide.reindex(cal).to_numpy(dtype=np.float64)
    adx = out['adx']
    if np.nanmax(np.abs(adx)) == 0:
        raise ValueError("ADX her yerde 0: legacy ADX (K18). `with_adx(df, 'wilder')` uygulayin")
    out['dates'] = np.array([str(d)[:10] for d in cal])
    out['symbols'] = np.array(order)
    if rf is not None:
        rf = np.asarray(rf, dtype=np.float64).reshape(-1)
        if rf.shape[0] != len(cal):
            raise ValueError(f"rf uzunlugu {rf.shape[0]} != takvim {len(cal)}")
        out['rf'] = rf
    return out


def risk_free_for(dates) -> np.ndarray:
    """A8 donem basi risksiz getiri; ilk seans NaN (ortam ilk adimi t=0 -> 1 icin rf_1 kullanir)."""
    from evaluation.risk_free import risk_free_series
    return risk_free_series(list(dates)).to_numpy(dtype=np.float64)


def save_npz(arrays: dict, path: str, meta: Optional[dict] = None) -> dict:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    payload = dict(arrays)
    payload['meta_json'] = np.array(json.dumps(meta or {}, default=str))
    np.savez_compressed(path, **payload)
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {'path': path, 'sha256': sha, 'T': int(len(arrays['dates'])), 'N': int(len(arrays['symbols'])),
            'first': str(arrays['dates'][0]), 'last': str(arrays['dates'][-1])}


def load_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != 'meta_json'}
    out['meta'] = json.loads(str(d['meta_json'])) if 'meta_json' in d.files else {}
    with open(path, 'rb') as f:
        out['sha256'] = hashlib.sha256(f.read()).hexdigest()
    return out
