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

Protokol (hakem-2 #1, 2026-11-06): her kayit, getirilerinin hangi BILINEN
ACIK protokolle uretildigini tasir (`protocol`). Acik bir madde varsa kayit
'protocol_open' etiketi alir ve kanit olarak kullanilmaz (`load_trials`
varsayilan olarak dislar). Maddeler:
  price_stats_source — degerlendirme (val/test) ortaminin fiyat olcegi.
      Temiz: 'train_window' (B9, 2026-11-06'dan beri rota ve HPO). Acik:
      'own_window' (degerlendirme penceresinin kendi istatistigi, Faz 10 Y13;
      B9 oncesi rota kayitlari ayni seyi 'test_window' diye yazdi).
      `price_stats_source()` bunu ortamin GERCEKTEN kullandigi olcekten olcer.
  indicator_bfill=True — gosterge isinma boslugu bfill ile dolu (Y15 / B7), acik
  adx_constant=True    — ADX tum panelde 0 (K18), acik
  indicators_present=False — gozlemin gosterge sutunlari panelde yok, env sabit
      varsayilan goruyor (HPO ham OHLCV okuyor; Faz 9 §9.5 Y21), acik

DSR'nin N'si (hakem-2 #2): Bailey & LdP'nin N'si YAPILAN TUM denemelerdir.
Budanan (pruned) ve basarisiz denemeler de `status` ile kaydedilir ve N'ye
sayilir; V yalnizca getirisi olan denemelerden gelir. Farkli `kind`/`split`/
pencere karistirilmaz (`sharpe_trial_stats` hata verir).
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

STATUSES = ('completed', 'pruned', 'failed')
PROTOCOL_OPEN_TAG = 'protocol_open'
# protokol anahtari -> "temiz" deger. Farkli deger = bilinen acik.
CLEAN_PROTOCOL = {
    'price_stats_source': 'train_window',
    'indicator_bfill': False,
    'adx_constant': False,
    'indicators_present': True,
}


def open_protocol_items(protocol: Optional[dict]) -> List[str]:
    """Protokolde temiz olmayan (ya da hic beyan edilmemis) maddeler."""
    protocol = protocol or {}
    return [k for k, clean in CLEAN_PROTOCOL.items() if protocol.get(k, None) != clean]


def price_stats_source(eval_env, train_price_stats: Optional[dict]) -> Optional[str]:
    """Degerlendirme ortaminin fiyat olcegi nereden geliyor — OLCULUR (B9).

    'train_window'        : ortam olcegi disaridan almis VE her sembolu icin
                            egitim olcegiyle birebir ayni (temiz)
    'own_window'          : ortam olcegi kendi penceresinden hesaplamis (Y13)
    'supplied_other'      : disaridan verilmis ama egitim olcegiyle uyusmuyor
    'supplied_unverified' : disaridan verilmis, karsilastiracak egitim olcegi yok
    None                  : ortam yok (deneme ortam kurulmadan bitti)

    Etiket cagiranin beyanina degil ortamin durumuna dayanir: rota ya da HPO
    olcegi gecirmeyi unutursa kayit kendiliginden 'protocol_open' olur.
    Gymnasium sarmalayicilari (Monitor) `unwrapped` ile acilir.
    """
    if eval_env is None:
        return None
    base = getattr(eval_env, 'unwrapped', eval_env)
    if getattr(base, 'price_stats_source', 'own_window') != 'supplied':
        return 'own_window'
    if train_price_stats is None:
        return 'supplied_unverified'
    used = base.get_price_stats()
    for sym in base.symbols:
        ref = train_price_stats.get(sym)
        if ref is None or used.get(sym) != {'mean': float(ref['mean']), 'std': float(ref['std'])}:
            return 'supplied_other'
    return 'train_window'


def indicators_present(source) -> bool:
    """Gozlemin gosterge sutunlari (macd, rsi, cci, adx, turbulence) veride mi.

    `source` bir panel (DataFrame) ya da TradingEnv olabilir. Eksik sutunu env
    sabit varsayilanla doldurur (rsi=50, adx=0, ...) — o ozellik bilgi tasimaz.
    """
    from env.trading_env import OBS_INDICATOR_COLUMNS
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        frame = getattr(source, 'unwrapped', source).df
    return all(c in frame.columns for c in OBS_INDICATOR_COLUMNS)


def rl_eval_protocol(eval_env, train_price_stats: Optional[dict],
                     indicator_bfill: Optional[bool] = True) -> dict:
    """RL degerlendirme kaydinin protokolu; rota ve HPO ayni hesabi kullanir.

    Uc madde degerlendirme ortamindan OLCULUR (price_stats_source,
    adx_constant, indicators_present); `indicator_bfill` cagiranin beyanidir
    (B7 bitene kadar True). Gosterge sutunlari yoksa bfill sorusu anlamsizdir
    (None) ve ADX sabit varsayilan 0'dir (True).
    """
    if eval_env is None:
        return {k: None for k in CLEAN_PROTOCOL}
    base = getattr(eval_env, 'unwrapped', eval_env)
    present = indicators_present(base)
    cols = base.df.columns
    adx_const = ('adx' not in cols) or bool(base.df['adx'].nunique(dropna=True) <= 1)
    return {
        'price_stats_source': price_stats_source(base, train_price_stats),
        'indicator_bfill': indicator_bfill if present else None,
        'adx_constant': adx_const,
        'indicators_present': present,
    }


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
    status: str = 'completed',
    protocol: Optional[dict] = None,
    data_sha256: Optional[str] = None,
) -> str:
    """Bir denemeyi kaydet, trial_id dondur.

    Args:
        kind: deneme turu ('rl_train', 'rl_hpo', 'spike', ...).
        config: denemeyi tanimlayan yapilandirma (tohum ve VERI HASH'i HARIC —
            ayni yapilandirmanin farkli tohumlari ve veri tazelemeleri ayni
            `config_hash`'i paylasir; veri `data_sha256` alaninda).
        returns/dates: degerlendirme bolumunun donem getirileri (basit getiri).
        split: getirilerin geldigi bolum ('val', 'test', ...).
        status: 'completed' | 'pruned' | 'failed' — budanan/basarisiz denemeler
            da N'ye sayilir (getirisiz kaydedilir).
        protocol: bilinen acik protokol maddeleri (modul docstring'i).
    """
    if status not in STATUSES:
        raise ValueError(f"status {STATUSES} icinden olmali: {status}")
    directory = directory or _results_dir()
    os.makedirs(directory, exist_ok=True)
    trial_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    tag_set = set(tags or [])
    open_items = open_protocol_items(protocol)
    if open_items:
        tag_set.add(PROTOCOL_OPEN_TAG)
    rec = {
        'trial_id': trial_id,
        'kind': kind,
        'status': status,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'config_hash': config_hash(config),
        'config': config,
        'seed': seed,
        'split': split,
        'data_sha256': data_sha256,
        'protocol': protocol or {},
        'protocol_open': open_items,
        'metrics': metrics or {},
        'tags': sorted(tag_set),
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
    split: Optional[str] = None,
    exclude_tags: Iterable[str] = ('spike', PROTOCOL_OPEN_TAG),
    directory: Optional[str] = None,
) -> List[dict]:
    """Kayitlari oku. Varsayilan: spike ve acik-protokol kayitlari DISLANIR.

    Tum durumlar (completed/pruned/failed) doner — N sayimi icin gerekli.
    Acik-protokol kayitlarini gormek icin `exclude_tags=('spike',)`.
    """
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
            if split and rec.get('split') != split:
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


def sharpe_trial_stats(
    trials: List[dict],
    by: str = 'trial',
    risk_free=None,
    allow_mixed_windows: bool = False,
) -> dict:
    """DSR icin N ve V (DONEM BASI Sharpe).

    by='trial' (varsayilan): her kayit (yapilandirma x tohum) ayri deneme.
        Tohumlar arasinda secim de bir secimdir; N ve V buna gore sayilir.
    by='config': ayni yapilandirmanin tohumlari tek deneme sayilir ve SR'leri
        ORTALANIR. Uyari (hakem-2 #2b): ortalama V'yi ~1/k kucultur, SR0 duser,
        DSR sisar — ancak sinanan SR de ayni tohum ortalamasiysa tutarlidir.
        Hangi kuralin kullanilacagi M8 on kaydinda sabitlenir.

    N = ayni gruptaki TUM denemeler (budanan/basarisiz dahil). V = getirisi
    olan denemelerin SR varyansi. `trials` tek bir kind + split'ten ve ayni
    degerlendirme penceresinden gelmeli; karisiksa ValueError.

    risk_free: verilirse (tarih indeksli Series) her denemenin getirisinden
        dusulur — sinanan stratejinin DSR'sinde de AYNI seri kullanilmali.
    """
    if not trials:
        return {'n_trials': 0, 'n_with_returns': 0, 'var_trials': None, 'srs': {}, 'by': by}
    kinds = {t.get('kind') for t in trials}
    splits = {t.get('split') for t in trials}
    if len(kinds) > 1 or len(splits) > 1:
        raise ValueError(f"Denemeler tek bir kind/split'ten olmali: kind={kinds}, split={splits}")

    keys_all = {t['config_hash'] if by == 'config' else t['trial_id'] for t in trials}
    rows, windows = [], set()
    for t in trials:
        if t.get('status', 'completed') != 'completed' or not t.get('returns_path'):
            continue
        s = load_returns(t)
        if risk_free is not None:
            s = s - pd.Series(risk_free).reindex(s.index)
        r = s.to_numpy(dtype=float)
        ok = np.isfinite(r)
        if ok.sum() < 3 or r[ok].std(ddof=1) == 0:
            continue
        idx = s.index[ok]
        windows.add((str(idx.min())[:10], str(idx.max())[:10]))
        rows.append((t['config_hash'] if by == 'config' else t['trial_id'],
                     r[ok].mean() / r[ok].std(ddof=1)))
    if len(windows) > 1 and not allow_mixed_windows:
        raise ValueError(f"Denemeler farkli degerlendirme pencerelerinden: {sorted(windows)[:4]}")
    if not rows:
        return {'n_trials': len(keys_all), 'n_with_returns': 0, 'var_trials': None,
                'srs': {}, 'by': by}
    df = pd.DataFrame(rows, columns=['key', 'sr']).groupby('key')['sr'].mean()
    return {
        'n_trials': int(len(keys_all)),            # N: TUM denemeler
        'n_with_returns': int(len(df)),
        'var_trials': float(df.var(ddof=1)) if len(df) > 1 else 0.0,
        'srs': df.to_dict(),
        'by': by,
        'kind': next(iter(kinds)),
        'split': next(iter(splits)),
        'window': next(iter(windows)) if windows else None,
        'excess_of_risk_free': risk_free is not None,
    }
