"""Faz 9 / C2 — Deneme kaydi (trial registry).

Neden: Deflated Sharpe'in N'si (denenen yapilandirma sayisi) ve V'si
(denemelerin Sharpe varyansi) ile PBO/CSCV'nin T x N getiri matrisi, YAPILAN
HER denemenin kaydini ister — yalnizca "kazanani" degil (Bailey & Lopez de
Prado 2014; Bailey vd. 2017). Kayit olmadan N=1 varsayilir ve DSR secim
yanliligini duzeltmez.

Konum (kullanici bazli, Faz 7 calisma alani):
    results_dir()/trial_registry.jsonl          her satir bir deneme
    results_dir()/trial_returns/<trial_id>.parquet   gunluk getiri serisi
Kayit yalnizca EKLEMELIDIR; silinen model/kosum kayittan dusulmez (deneme
yine de yapilmistir).

`kind` ornekleri: 'rl_train' (egitim rotasi, test bolumu), 'rl_hpo'
(Optuna denemesi, dogrulama bolumu), 'spike' (M1 fizibilite — rapora girmez,
`load_trials` varsayilan olarak dislar).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGISTRY_FILE = 'trial_registry.jsonl'
RETURNS_DIR = 'trial_returns'
_LOCK = threading.Lock()


def _results_dir() -> str:
    try:
        from app.auth import workspace as ws
        return ws.results_dir()
    except Exception:
        os.makedirs('results', exist_ok=True)
        return 'results'


def registry_path(directory: Optional[str] = None) -> str:
    return os.path.join(directory or _results_dir(), REGISTRY_FILE)


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]


def register_trial(
    kind: str,
    config: dict,
    seed: Optional[int],
    metrics: Optional[dict] = None,
    returns=None,
    dates=None,
    split: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    extra: Optional[dict] = None,
    directory: Optional[str] = None,
) -> str:
    """Bir denemeyi kaydet, trial_id dondur.

    Args:
        kind: deneme turu ('rl_train', 'rl_hpo', 'spike', ...).
        config: denemeyi tanimlayan yapilandirma (tohum HARIC — ayni
            yapilandirmanin farkli tohumlari ayni `config_hash`'i paylasir).
        returns/dates: degerlendirme bolumunun donem getirileri (basit getiri).
        split: getirilerin geldigi bolum ('val', 'test', ...).
    """
    directory = directory or _results_dir()
    os.makedirs(directory, exist_ok=True)
    trial_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    rec = {
        'trial_id': trial_id,
        'kind': kind,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'config_hash': config_hash(config),
        'config': config,
        'seed': seed,
        'split': split,
        'metrics': metrics or {},
        'tags': sorted(set(tags or [])),
        **({'extra': extra} if extra else {}),
    }
    if returns is not None:
        r = np.asarray(returns, dtype=float)
        rdir = os.path.join(directory, RETURNS_DIR)
        os.makedirs(rdir, exist_ok=True)
        idx = pd.to_datetime(pd.Index(dates)) if dates is not None else pd.RangeIndex(len(r))
        if len(idx) != len(r):
            raise ValueError(f"dates ({len(idx)}) ve returns ({len(r)}) ayni uzunlukta olmali")
        frame = pd.DataFrame({'ret': r}, index=idx)
        frame.index.name = 'date'
        path = os.path.join(rdir, f'{trial_id}.parquet')
        try:
            frame.to_parquet(path)
        except Exception:
            path = path[:-len('.parquet')] + '.csv'
            frame.to_csv(path)
        rec['returns_path'] = os.path.relpath(path, directory).replace('\\', '/')
        rec['n_returns'] = int(len(r))
    line = json.dumps(rec, default=str, ensure_ascii=False)
    with _LOCK:
        with open(registry_path(directory), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return trial_id


def load_trials(
    kind: Optional[str] = None,
    exclude_tags: Iterable[str] = ('spike',),
    directory: Optional[str] = None,
) -> List[dict]:
    path = registry_path(directory)
    if not os.path.exists(path):
        return []
    excl = set(exclude_tags or ())
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Bozuk deneme kaydi satiri atlandi: {line[:80]}")
                continue
            if kind and rec.get('kind') != kind:
                continue
            if excl & set(rec.get('tags', [])) or rec.get('kind') in excl:
                continue
            rec['_dir'] = directory or _results_dir()
            out.append(rec)
    return out


def load_returns(trial: dict) -> pd.Series:
    path = os.path.join(trial['_dir'], trial['returns_path'])
    if path.endswith('.parquet'):
        s = pd.read_parquet(path)['ret']
    else:
        s = pd.read_csv(path, index_col=0, parse_dates=True)['ret']
    s.name = trial['trial_id']
    return s


def returns_matrix(trials: List[dict], how: str = 'inner') -> pd.DataFrame:
    """PBO/CSCV icin tarih x deneme matrisi (tarihe gore hizali)."""
    series = [load_returns(t) for t in trials if t.get('returns_path')]
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join=how).sort_index()


def sharpe_trial_stats(trials: List[dict], by: str = 'config') -> dict:
    """DSR icin N ve V (DONEM BASI Sharpe).

    by='config': ayni yapilandirmanin tohumlari tek deneme sayilir, SR'leri
        ortalanir (N = farkli yapilandirma sayisi).
    by='trial' : her kayit ayri deneme.
    """
    rows = []
    for t in trials:
        if not t.get('returns_path'):
            continue
        r = load_returns(t).to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < 3 or r.std(ddof=1) == 0:
            continue
        rows.append((t['config_hash'] if by == 'config' else t['trial_id'], r.mean() / r.std(ddof=1)))
    if not rows:
        return {'n_trials': 0, 'var_trials': None, 'srs': {}}
    df = pd.DataFrame(rows, columns=['key', 'sr']).groupby('key')['sr'].mean()
    return {
        'n_trials': int(len(df)),
        'var_trials': float(df.var(ddof=1)) if len(df) > 1 else 0.0,
        'srs': df.to_dict(),
        'by': by,
    }
