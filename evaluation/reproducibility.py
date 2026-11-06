"""Faz 9 / C1, C3, C4 — RL tohumu, veri anlik goruntusu, ortam kaydi.

Kaynak: Henderson, P. vd. (2018). Deep Reinforcement Learning that Matters.
AAAI 2018 — tohum ve uygulama ayrintisinin sonucu ne kadar degistirdigi.

C1 (tohum): `resolve_seed` istekte tohum yoksa kriptografik rastgele bir tohum
   URETIR ve geri dondurur; tohum her durumda kayda girer ("tohumsuz" kosum
   kalmaz). SB3 modeline `seed=` verilir (SB3 kendi icinde
   `set_random_seed(seed)` + eylem uzayi tohumu uygular); ilk `reset` tohumu
   `VecEnv.seed` ile tasinir.
   Kapsam: ayni tohum + CPU -> ayni egri (tests/test_rl_reproducibility.py).
   GPU'da cuDNN/atomik islemler bit-eslik vermeyebilir; sapma olculup
   belgelenir, gizlenmez.
C3 (veri anlik goruntusu): paneli kanonik CSV baytlarindan SHA-256 ile
   ozetler ve parquet olarak `data/snapshots/<ad>_<sha12>.parquet`'e yazar
   (piyasa verisi kullanicilar arasi ortak oldugu icin depo kokunde; ayni
   hash ikinci kez yazilmaz). Manifest/meta yalnizca ozet kaydini tasir;
   `verify_snapshot` hash'i yeniden hesaplar ve degisiklikte uyarir.
C4 (ortam kaydi): Python/paket surumleri, GPU, torch belirlenimcilik bayraklari.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import secrets
import sys
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = os.path.join('data', 'snapshots')

_PACKAGES = ('numpy', 'pandas', 'scipy', 'torch', 'stable_baselines3', 'gymnasium',
             'sklearn', 'xgboost', 'lightgbm', 'catboost', 'arch', 'statsmodels',
             'rliable', 'pyarrow', 'optuna')


# ----------------------------------------------------------------------------
# C1 — tohum
# ----------------------------------------------------------------------------

def resolve_seed(seed: Optional[int]) -> int:
    """Istekteki tohum ya da yeni bir tohum (0 <= s < 2**31-1)."""
    if seed is None:
        return secrets.randbelow(2 ** 31 - 1)
    seed = int(seed)
    if not 0 <= seed < 2 ** 32:
        raise ValueError("seed 0 <= seed < 2**32 olmali")
    return seed


# ----------------------------------------------------------------------------
# C3 — veri anlik goruntusu
# ----------------------------------------------------------------------------

def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_index().reset_index()
    x = x[sorted(x.columns, key=lambda c: (c not in ('symbol', 'date'), str(c)))]
    return x


def frame_sha256(df: pd.DataFrame) -> str:
    """Kanonik CSV baytlarinin SHA-256'si (sirali indeks, sabit float bicimi).

    Parquet baytlari pyarrow surumune gore degisebildigi icin hash parquet'ten
    degil bu kanonik metinden alinir.
    """
    text = _canonical(df).to_csv(index=False, float_format='%.12g', lineterminator='\n')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def snapshot_frame(df: pd.DataFrame, name: str, directory: Optional[str] = None,
                   write: bool = True) -> dict:
    """Paneli ozetle (ve parquet olarak sakla). Donen sozluk manifest/meta'ya girer."""
    sha = frame_sha256(df)
    directory = directory or SNAPSHOT_DIR
    path = os.path.join(directory, f'{name}_{sha[:12]}.parquet')
    if write and not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        try:
            _canonical(df).to_parquet(path, index=False)
        except Exception as exc:          # parquet motoru yoksa ozet yine kaydedilir
            logger.warning(f"Anlik goruntu yazilamadi ({exc}); yalnizca hash kaydedildi")
            path = None
    dates = pd.to_datetime(df.index.get_level_values('date')) if 'date' in df.index.names else None
    return {
        'name': name,
        'sha256': sha,
        'path': path.replace('\\', '/') if path else None,
        'rows': int(len(df)),
        'columns': [str(c) for c in df.columns],
        'n_symbols': int(df.index.get_level_values('symbol').nunique()) if 'symbol' in df.index.names else None,
        'start': str(dates.min().date()) if dates is not None and len(dates) else None,
        'end': str(dates.max().date()) if dates is not None and len(dates) else None,
    }


def verify_snapshot(entry: dict, df: Optional[pd.DataFrame] = None) -> dict:
    """Kayittaki hash'i yeniden hesapla.

    df verilirse onu (bugunku veri), verilmezse kayitli parquet dosyasini
    olcer. Uyusmazlikta uyari loglanir ve {'ok': False, ...} doner.
    """
    if df is None:
        path = entry.get('path')
        if not path or not os.path.exists(path):
            return {'ok': False, 'reason': 'anlik goruntu dosyasi yok', 'expected': entry.get('sha256')}
        x = pd.read_parquet(path)
        idx = [c for c in ('symbol', 'date') if c in x.columns]
        if 'date' in x.columns:
            x['date'] = pd.to_datetime(x['date'])
        df = x.set_index(idx) if idx else x
    current = frame_sha256(df)
    ok = current == entry.get('sha256')
    if not ok:
        logger.warning(
            f"VERI DEGISTI: '{entry.get('name')}' anlik goruntusu {entry.get('sha256', '')[:12]} "
            f"idi, simdi {current[:12]} — bu kayittaki sonuclar bugunku veriyle yeniden uretilemez."
        )
    return {'ok': ok, 'expected': entry.get('sha256'), 'current': current}


# ----------------------------------------------------------------------------
# C4 — ortam kaydi
# ----------------------------------------------------------------------------

def environment_record() -> dict:
    from importlib import metadata

    pkgs = {}
    for name in _PACKAGES:
        dist = {'sklearn': 'scikit-learn', 'stable_baselines3': 'stable_baselines3'}.get(name, name)
        try:
            pkgs[name] = metadata.version(dist)
        except Exception:
            pkgs[name] = None
    rec = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'packages': pkgs,
    }
    try:
        import torch
        rec['torch'] = {
            'cuda_available': bool(torch.cuda.is_available()),
            'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
            'num_threads': int(torch.get_num_threads()),
        }
    except Exception:
        rec['torch'] = None
    return rec
