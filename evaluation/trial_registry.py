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
      B9 oncesi rota kayitlari ayni seyi 'test_window' diye yazdi),
      'overlapping_window', 'supplied_other', 'supplied_unverified',
      'train_window_fallback'. `price_stats_source()` olcegin KOKENINI olcer
      (hakem-2 B9 #3): egitim ortaminin kendi penceresinden mi, o pencere
      degerlendirmeden once mi bitiyor, degerler birebir ayni mi.
  indicator_bfill=True — egitim ya da degerlendirme verisinde sembol basinda bfill
      imzali gosterge blogu kaldi (Y15 / B7), acik. OLCULUR (rl_eval_protocol);
      rota/HPO blogu veri kapisinda atiyor (prepare_rl_panel drop_indicator_warmup).
  adx_constant=True    — ADX egitim ya da degerlendirme verisinde sabit (legacy ADX = 0;
      K18), acik. Rota K18'den beri ADX'i 'wilder' ile yeniden hesaplar.
  indicators_present=False — gozlemin gosterge sutunlari panelde yok, env sabit
      varsayilan goruyor (HPO ham OHLCV okuyor; Faz 9 §9.5 Y21), acik
  price_in_support=False — degerlendirme fiyat ozelliklerinin > %1'i egitim
      penceresinin [min, max] destegi disinda (B9'un eski ortamdaki bedeli;
      hakem-2 B9 #1). B7/K18 kapansa da eski ortam kayitlarini acik tutar.
      Olcum: evaluation.observation_health.
  hyperparams_clean=False/None — hiperparametreler, kayitlari acik ya da
      bulunamayan bir HPO calismasindan (hakem-2 B9 #8). Varsayilan
      hiperparametre ve HPO'nun kendi denemesi icin True.
  obs_health=False/None — gozlem sagligi (K18): egitim ya da degerlendirme
      penceresinde fiyat/hacim/gosterge bloklarindan birinde kirpilan oran > %1
      ya da pencere boyunca sabit bir gozlem boyutu var. OLCULUR
      (evaluation.observation_health.obs_health_ok). Bu maddeyi tasimayan
      (K18 oncesi) kayit acik sayilir.
  features_in_support=False/None — fiyat DISI bir piyasa ozelliginin (hacim,
      gostergeler) degerlendirme degerlerinin > %1'i egitim destegi disinda (K18;
      ozellik basina). Eski olcekte MACD TL cinsinden: test degerlerinin %28,7'si
      destek disi (Faz 9 §9.9). OLCULUR.
  dims_in_support=False/None — kapili bloklarin bir GOZLEM BOYUTUNDA (sembol x ozellik)
      degerlendirme degerlerinin > %1'i egitim destegi disinda (hakem-2 K18 #1: havuzlu
      oran tek sembolu gizler; gercek panelde SASA hacmi %23). OLCULUR.
  price_stats_source='causal' de TEMIZDIR (K25 b): fiyat z'si yalniz <= t
      kapanislariyla kayan pencereden; egitim ve degerlendirme ortami ayni
      tanimla (mod + pencere) kurulmali, aksi 'mixed_norm' (acik).
  split_from_spec=False/None — pencereler tek bolme tanimindan (evaluation.splits)
      gelmiyor (R1; Faz 9 §9.2 #3). R1 oncesi kayitlar oranli `split_data` ile
      bolundu ve bu maddeyi tasimaz: acik sayilir.
  selection_metric — model/hiperparametre neyle secildi (R1; Faz 10 K20). Temiz:
      'val_excess_sharpe' (dogrulama blogunda risksiz faiz ustu Sharpe). Acik:
      'val_sharpe_rf0' (risksiz faiz serisi yoktu), None (dogrulamada secim yok).
  hyperparams_clean R1 ile soyu da olcer (`hpo_lineage`): HPO'nun gordugu veri
      katin TEST baslangicindan once bitmeli, HPO tohumu egitim tohumundan farkli
      olmali (Eimer vd. 2023), evren ve gozlem tanimi ayni olmali.

Kat (R1): her kayit `fold` ve `split` (bolme adi + ozeti) alanlarini tasir. Ayni
(yapilandirma, tohum, kat, bolum) ikinci kez kaydedilirse kayit 'test_reread'
etiketi alir — "test yapilandirma basina bir kez" (Faz 9 §9.2 #3) olculur.

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
    'price_in_support': True,
    'hyperparams_clean': True,
    'obs_health': True,
    'features_in_support': True,
    'dims_in_support': True,
    # R1 (Faz 9 §9.11): pencereler tek bolme tanimindan mi (evaluation/splits.py) ve
    # model/hiperparametre DOGRULAMADA, risksiz faiz ustu Sharpe ile mi secildi.
    'split_from_spec': True,
    'selection_metric': 'val_excess_sharpe',
    # R1 (hakem-2 R1 #1, kullanici karari 2026-11-14): degerlendirme penceresi R1'den once
    # basarim metrikleriyle okunmus bir pencereyle kesismiyor mu (EXPOSED_EVAL_WINDOWS).
    'eval_window_unexposed': True,
    # R1 (hakem-2 R1 #5; B4'un RL tarafi): faz 2 temel oranlari tek anlik goruntu (S4),
    # nokta-zamanli degil. Faz 1 (temel oran yok) temiz.
    'fundamentals_pit': True,
}
# R1'den ONCE basarim metrikleriyle okunmus degerlendirme pencereleri (kapsayici tarih
# araliklari). Kesisen kayit acik kalir: o donemde verilen tasarim kararlari o donemin
# sonucunu gormustu. R1 oncesi oranli bolmeyle yapilan egitimler test araligi KAYDETMIYOR
# (23 metrik dosyasi); olculemedigi icin listede degil, Faz 9 §9.11'de beyan edilir.
EXPOSED_EVAL_WINDOWS = (
    ('2024-08-06', '2026-07-27',
     "Faz 9 §9.5 B9 etki olcumu: A2C/PPO x 3 tohum, fazla getiri Sharpe'i, al-tut"),
)
# Ayni maddenin ikinci temiz degeri (K25 b: nedensel fiyat olcegi).
CLEAN_ALTERNATIVES = {
    'price_stats_source': ('causal',),
}
TEST_REREAD_TAG = 'test_reread'


def open_protocol_items(protocol: Optional[dict]) -> List[str]:
    """Protokolde temiz olmayan (ya da hic beyan edilmemis) maddeler."""
    protocol = protocol or {}
    return [k for k, clean in CLEAN_PROTOCOL.items()
            if protocol.get(k, None) != clean
            and protocol.get(k, None) not in CLEAN_ALTERNATIVES.get(k, ())]


def price_stats_source(eval_env, train_env) -> Optional[str]:
    """Degerlendirme ortaminin fiyat olcegi nereden geliyor — KOKENI olculur (B9).

    `train_env` olcegin alindigi EGITIM ortamidir (sozluk degil; hakem-2 B9 #3:
    sozlukle yalnizca tutarlilik olculebiliyordu — test penceresinin kendi
    olcegi "egitim olcegi" diye verilince etiket temiz cikiyordu).

    'train_window'          : egitim ortami olcegi KENDI penceresinden hesaplamis,
                              o pencere degerlendirme penceresinden ONCE bitiyor
                              ve degerlendirme olcegi her sembolde birebir ayni (temiz)
    'own_window'            : degerlendirme ortami olcegi kendi penceresinden (Y13)
    'overlapping_window'    : egitim penceresi degerlendirme baslamadan bitmiyor
    'supplied_other'        : verilen olcek egitim ortaminin olcegiyle uyusmuyor
    'supplied_unverified'   : egitim ortami yok ya da olcegini kendisi hesaplamamis
    'train_window_fallback' : egitim olcegi bir sembolde yedege dusmus (std tabani /
                              50-50; TradingEnv.price_stats_fallbacks)
    None                    : degerlendirme ortami yok (deneme ortam kurulmadan bitti)

    K25 (b) — ortamlardan biri price_norm='causal' ise olcek kullanilmaz:
    'causal'                : iki ortam da nedensel, ayni pencere ve MACD tanimi, egitim penceresi
                              degerlendirmeden ONCE bitiyor (temiz; baglamin ilk tarihten
                              once bittigini env kurulumda zorlar)
    'mixed_norm'            : biri nedensel digeri olcekli ya da pencereler farkli
    'causal_short_context'  : olcek pencerenin gercekte dolu olmadigi satirlarla kuruldu —
                              degerlendirmede HERHANGI bir satir, egitimde seri basi
                              (baglamin isaretledigi, yeni listelenmis) olmayan bir sembolde
                              (hakem-2 K18 #3; baglamsiz kurulan ortam boyle cikar)
    'causal_unverified'     : egitim ortami yok

    Gymnasium sarmalayicilari (Monitor) `unwrapped` ile acilir.
    """
    if eval_env is None:
        return None
    ev = getattr(eval_env, 'unwrapped', eval_env)
    tr0 = getattr(train_env, 'unwrapped', train_env) if train_env is not None else None
    ev_norm = getattr(ev, 'price_norm', 'stats')
    tr_norm = getattr(tr0, 'price_norm', 'stats') if tr0 is not None else None
    if 'causal' in (ev_norm, tr_norm):
        if tr0 is None:
            return 'causal_unverified'
        if ev_norm != tr_norm or any(
                getattr(ev, a, None) != getattr(tr0, a, None)
                for a in ('price_norm_window', 'price_norm_macd')):
            return 'mixed_norm'
        if not (len(tr0.dates) and len(ev.dates) and tr0.dates[-1] < ev.dates[0]):
            return 'overlapping_window'
        ev_short = any(getattr(ev, 'price_norm_warmup_rows', {}).values())
        tr_series = getattr(tr0, 'price_norm_series_start', set())
        tr_short = any(k for s_, k in getattr(tr0, 'price_norm_warmup_rows', {}).items()
                       if s_ not in tr_series)
        if ev_short or tr_short:
            return 'causal_short_context'
        return 'causal'
    if getattr(ev, 'price_stats_source', 'own_window') != 'supplied':
        return 'own_window'
    if train_env is None:
        return 'supplied_unverified'
    tr = getattr(train_env, 'unwrapped', train_env)
    if getattr(tr, 'price_stats_source', None) != 'own_window':
        return 'supplied_unverified'
    if not (len(tr.dates) and len(ev.dates) and tr.dates[-1] < ev.dates[0]):
        return 'overlapping_window'
    ref, used = tr.get_price_stats(), ev.get_price_stats()
    for sym in ev.symbols:
        if sym not in ref or used.get(sym) != {'mean': float(ref[sym]['mean']),
                                               'std': float(ref[sym]['std'])}:
            return 'supplied_other'
    fallbacks = getattr(tr, 'price_stats_fallbacks', {}) or {}
    if any(sym in fallbacks for sym in ev.symbols):
        return 'train_window_fallback'
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


def rl_eval_protocol(eval_env, train_env, indicator_bfill: Optional[bool] = None,
                     health: Optional[dict] = None,
                     hyperparams_clean: Optional[bool] = True,
                     split_from_spec: Optional[bool] = None,
                     selection_metric: Optional[str] = None) -> dict:
    """RL degerlendirme kaydinin protokolu; rota ve HPO ayni hesabi kullanir.

    Ortamdan OLCULENLER: price_stats_source (koken), adx_constant (egitim ya da
    degerlendirme verisinde), indicators_present, price_in_support ve obs_health
    (K18; ikisi de `health`'ten — verilmezse burada olculur) ve
    indicator_bfill (B7: egitim YA DA degerlendirme verisinde sembol basinda
    bfill imzali gosterge blogu kaldi mi; `indicator_bfill` verilirse beyan onu
    ezer). Cagiranin BEYANI: hyperparams_clean (rota HPO calismasinin kaydindan
    cozer; HPO denemesinin kendisi icin True). Gosterge sutunlari yoksa bfill
    sorusu anlamsizdir (None) ve ADX sabit varsayilan 0'dir (True).
    """
    if eval_env is None:
        return {k: None for k in CLEAN_PROTOCOL}
    from evaluation.observation_health import (
        dims_in_support, features_in_support, obs_health_ok, observation_health,
        price_in_support,
    )

    base = getattr(eval_env, 'unwrapped', eval_env)
    if health is None and train_env is not None:
        health = observation_health(train_env, base)
    present = indicators_present(base)

    def _adx_const(frame):
        return ('adx' not in frame.columns) or bool(frame['adx'].nunique(dropna=True) <= 1)

    # K18: ADX egitim YA DA degerlendirme verisinde sabitse acik (eskiden yalniz degerlendirme)
    adx_const = _adx_const(base.df) or (
        train_env is not None and _adx_const(getattr(train_env, 'unwrapped', train_env).df))
    if indicator_bfill is None and present:
        from data.integrity import BFILL_WARMUP_COLS, WARMUP_MAX_ROWS, indicator_warmup_blocks
        train_df = getattr(train_env, 'unwrapped', train_env).df if train_env is not None else None
        # K18: bfill blogu yalniz sembolun SERI BASINDA olusur ve en fazla WARMUP_MAX_ROWS
        # satir surer. Egitim penceresinde en az o kadar satiri olan sembolun degerlendirme
        # penceresi basi seri basi degildir; oradaki tekrar dogaldir (kapanis degismeyince
        # RSI birebir ayni kalir: gercek panelde VAKBN, test penceresinin ilk iki gunu —
        # B7 olcutu bunu bfill sayip her rota test kaydini acik yaziyordu). Egitimde daha
        # kisa gecmisi olan sembolun blogu degerlendirmeye tasabilir: o sembol sayilir
        # (hakem-rl K18 #3).
        if train_df is not None:
            counts = train_df.groupby(level='symbol').size()
            with_history = set(counts.index[counts >= WARMUP_MAX_ROWS])
        else:
            with_history = set()
        frames = [(base.df, with_history)] + ([(train_df, set())] if train_df is not None else [])

        def contaminated(f, skip):
            # bfill imzali bas blok YA DA ('nan' modu) isinma sutununda NaN satir
            live = [c for c in BFILL_WARMUP_COLS if c in f.columns and f[c].notna().any()]
            blocks = {s for s in indicator_warmup_blocks(f) if s not in skip}
            return bool(blocks) or bool(live and f[live].isna().any().any())

        indicator_bfill = any(contaminated(f, skip) for f, skip in frames)
    return {
        'price_stats_source': price_stats_source(base, train_env),
        'indicator_bfill': indicator_bfill if present else None,
        'adx_constant': adx_const,
        'indicators_present': present,
        'price_in_support': price_in_support(health),
        'hyperparams_clean': hyperparams_clean,
        'obs_health': obs_health_ok(health),
        'features_in_support': features_in_support(health),
        'dims_in_support': dims_in_support(health),
        # R1: cagiranin BEYANI degil, cagiranin OLCTUGU deger — rota/HPO pencereyi
        # `evaluation.splits`'ten aldiysa True; verilmezse (None) acik.
        'split_from_spec': split_from_spec,
        'selection_metric': selection_metric,
        'eval_window_unexposed': eval_window_unexposed(base),
        'fundamentals_pit': fundamentals_pit(base),
    }


def eval_window_unexposed(eval_env) -> Optional[bool]:
    """Degerlendirme ortaminin tarih araligi EXPOSED_EVAL_WINDOWS'tan biriyle kesismiyorsa True."""
    base = getattr(eval_env, 'unwrapped', eval_env)
    df = getattr(base, 'df', None)
    if df is None or not len(df):
        return None
    d = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values('date')))
    d = d.tz_localize(None) if d.tz is not None else d
    lo, hi = d.min(), d.max()
    return not any(lo <= pd.Timestamp(b) and hi >= pd.Timestamp(a)
                   for a, b, _ in EXPOSED_EVAL_WINDOWS)


def fundamentals_pit(env) -> Optional[bool]:
    """Faz 2 ortami tek anlik temel oran tablosuyla calisiyorsa False (S4); faz 1 True."""
    base = getattr(env, 'unwrapped', env)
    if getattr(base, 'phase', 1) != 2:
        return True
    return False if getattr(base, 'fundamental_df', None) is not None else True


def hpo_lineage(study_name: Optional[str], *, fold_test_start: Optional[str] = None,
                train_seed: Optional[int] = None, symbols: Optional[list] = None,
                split_digest: Optional[str] = None, obs_config: Optional[dict] = None,
                directory: Optional[str] = None):
    """Hiperparametre SOYU (R1; Faz 9 §9.6 #14'un kayit tarafi).

    `hpo_study_protocol`'un bulduklarina ek olarak, HPO calismasinin kayitlarindan
    OLCULUR (beyan degil):
      hpo_window_after_test_start — bir HPO denemesinin dogrulama penceresi bu katin
          test baslangicindan once bitmiyor: ayar, sinav donemini gorerek secildi
      hpo_seed_reused — egitim tohumu HPO'nun kullandigi tohumlardan biri (ayar
          tohumlari ile sinama tohumlari ayri olmali; Eimer vd. 2023, arXiv 2306.01324)
      hpo_seed_unknown — HPO denemesi tohum kaydetmemis: ayrim dogrulanamaz
      hpo_universe_mismatch — HPO baska bir sembol evreninde (ya da sirasinda) aradi
      hpo_split_mismatch — HPO baska bir bolme tanimi kullandi
      hpo_obs_mismatch — HPO baska bir gozlem tanimiyla aradi (price_norm, gostergeler)

    Returns: (temiz_mi, acik_maddeler). Kayit bulunamazsa (None, [neden]).
    """
    clean, items = hpo_study_protocol(study_name, directory=directory)
    if clean is None:
        return None, items
    recs = [t for t in load_trials(kind='rl_hpo', exclude_tags=(), directory=directory)
            if (t.get('extra') or {}).get('study') == study_name]
    found = set(items)
    for t in recs:
        cfg = t.get('config') or {}
        val = (t.get('extra') or {}).get('val_range') or cfg.get('val')
        if fold_test_start is not None:
            if not val or val[1] is None or pd.Timestamp(str(val[1])[:10]) >= pd.Timestamp(fold_test_start):
                found.add('hpo_window_after_test_start')
        if train_seed is not None:
            if t.get('seed') is None:
                found.add('hpo_seed_unknown')
            elif int(t['seed']) == int(train_seed):
                found.add('hpo_seed_reused')
        if symbols is not None and list(cfg.get('symbols') or []) != list(symbols):
            found.add('hpo_universe_mismatch')
        if split_digest is not None and (t.get('split_spec') or {}).get('digest') != split_digest:
            found.add('hpo_split_mismatch')
        if obs_config is not None:
            for key, want in obs_config.items():
                if cfg.get(key) != want:
                    found.add('hpo_obs_mismatch')
    out = sorted(found)
    return (not out), out


def hpo_study_protocol(study_name: Optional[str], directory: Optional[str] = None):
    """Bir HPO calismasinin deneme kaydindaki protokolu (hakem-2 B9 #8).

    Returns:
        (temiz_mi, acik_maddeler): kayitlari bulunamazsa (None, [neden]).
        Rota bu bilgiyi kendi kaydina yazar: hiperparametreleri acik bir
        HPO'dan gelen rota kaydi da acik kalir.
    """
    if not study_name:
        return None, ['study_name_missing']
    recs = [t for t in load_trials(kind='rl_hpo', exclude_tags=(), directory=directory)
            if (t.get('extra') or {}).get('study') == study_name]
    if not recs:
        return None, ['hpo_trials_not_in_registry']
    items = sorted({k for t in recs for k in t.get('protocol_open', [])})
    return (not items), items


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
    fold: Optional[str] = None,
    split_spec: Optional[dict] = None,
) -> str:
    """Bir denemeyi kaydet, trial_id dondur.

    R1: `fold` katin adi, `split_spec` {'name', 'digest'} (evaluation.splits).
    Ayni (kind, config_hash, tohum, kat, bolme ozeti, bolum) icin getirili bir
    kayit zaten varsa yeni kayit 'test_reread' etiketi alir: ayni sinavin ikinci
    okumasi kanit sayilmaz ama N'ye sayilir.

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
    cfg_hash = config_hash(config)
    digest = (split_spec or {}).get('digest')
    if fold is not None and returns is not None and status == 'completed' and seed is not None:
        for prev in load_trials(kind=kind, split=split, exclude_tags=(), directory=directory):
            if (prev.get('config_hash') == cfg_hash and prev.get('seed') == seed
                    and prev.get('fold') == str(fold) and prev.get('returns_path')
                    and (prev.get('split_spec') or {}).get('digest') == digest
                    and prev.get('data_sha256') == data_sha256):
                tag_set.add(TEST_REREAD_TAG)
                break
    rec = {
        'trial_id': trial_id,
        'kind': kind,
        'status': status,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'config_hash': cfg_hash,
        'config': config,
        'seed': seed,
        'split': split,
        'fold': None if fold is None else str(fold),
        'split_spec': split_spec,
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
    exclude_tags: Iterable[str] = ('spike', PROTOCOL_OPEN_TAG, TEST_REREAD_TAG),
    directory: Optional[str] = None,
) -> List[dict]:
    """Kayitlari oku. Varsayilan: spike, acik-protokol ve ayni sinavin ikinci
    okumasi ('test_reread', R1) kayitlari DISLANIR.

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
    # R1: farkli katlar (ya da farkli bolme tanimlari) tek N/V'de karistirilmaz —
    # katlar arasi birlestirme kurali (cikarim birimi) M8 on kaydinda sabitlenir.
    folds = {(t.get('fold'), (t.get('split_spec') or {}).get('digest')) for t in trials}
    if len(folds) > 1 and not allow_mixed_windows:
        raise ValueError(f"Denemeler farkli katlardan/bolme tanimlarindan: {sorted(map(str, folds))[:4]}")

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
