"""Faz 10 / V — Veri butunlugu kapisi (RL paneli).

Neden (Faz 10 §8 #1, hakem-2 RED): RL egitim penceresi 2000-05-10..2018-07-20
idi ve icinde (olculdu, 2026-11-06, stock_data_with_indicators.csv):
  * fiyat seviyesi artefaktlari — TUPRS 762, FROTO 1.051 satir > 10.000 TL
    (2000-2004); 2005 YTL gecisinde e^6.9 kat sicrama (THYAO 2005-01-03);
  * 185 "hayalet gun" (piyasa kapali; volume = 0, OHLC = onceki kapanis);
  * eksik (sembol, tarih) hucresinde env ONCEKI SEMBOLUN ozelliklerini
    kopyaliyordu (env/trading_env.py, legacy mod).

Karar K17 (kullanici, 2026-11-06): RL penceresi 2013-06'dan baslar — panelde
en gec baslayan sembol ODAS (2013-05-21), bu tarihten sonra 30 sembolun hepsi
panelde; 2000-2005 artefaktlari ve 2009 sicramasi pencere disinda kalir.
Tarayici yine de her kosumda calisir: pencere icinde artefakt cikarsa egitim
DURUR (sessizce temizlenmez).

Kavramlar:
  hayalet gun : aktif sembollerin >= `ghost_threshold`'u "duz" (volume == 0 ve
                O = H = L = C) olan gun -> panelden atilir.
  bayat hucre : hayalet olmayan bir gunde tek sembolun duz satiri (islem
                gormemis) -> satir atilir, hucre EKSIK sayilir. Env'in
                `missing_data_mode='own_last'` modu bu hucrede islem yapmaz ve
                sembolu kendi son gozlemiyle tasir.
  aktiflik    : (tarih, sembol) satiri var mi — `activity_mask`.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# K17 — RL veri penceresinin baslangici (tek kaynak; app/core/config.py
# RL_DATA_START ile override edilir).
RL_DATA_START = '2013-06-01'

PRICE_COLS = ('open', 'high', 'low', 'close')
MAX_ABS_RETURN = 0.5          # BIST gunluk fiyat limiti +-%10/%20; %50 cok comert esik
LEVEL_JUMP_FACTOR = 10.0      # kapanis / onceki 21 seans medyani > 10 ya da < 1/10
REDENOMINATION_WINDOW = ('2004-12-01', '2005-02-28')   # 2005 YTL gecisi
REDENOMINATION_FACTOR = 100.0


class DataIntegrityError(ValueError):
    """Pencere icinde artefakt var; egitim baslatilmaz."""


def _long(df: pd.DataFrame) -> pd.DataFrame:
    """(symbol, date) MultiIndex -> sirali uzun tablo."""
    x = df.reset_index()
    if 'symbol' not in x.columns or 'date' not in x.columns:
        raise ValueError("df (symbol, date) MultiIndex tasimali")
    x['date'] = pd.to_datetime(x['date'])
    if getattr(x['date'].dt, 'tz', None) is not None:
        x['date'] = x['date'].dt.tz_localize(None)
    return x.sort_values(['symbol', 'date']).reset_index(drop=True)


def scan_price_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Supheli satirlari nedenleriyle listele (satir basina bir neden).

    Nedenler: nonpositive_price, ohlc_inconsistent, extreme_return,
    level_jump, redenomination.
    """
    x = _long(df)
    out = []

    price = x[list(PRICE_COLS)]
    bad = (~np.isfinite(price)).any(axis=1) | (price <= 0).any(axis=1)
    out.append(x.loc[bad, ['symbol', 'date', 'close']].assign(reason='nonpositive_price'))

    ok = x[~bad]
    tol = 1e-6
    hl = ((ok['high'] < ok['low'] * (1 - tol))
          | (ok['close'] > ok['high'] * (1 + tol)) | (ok['close'] < ok['low'] * (1 - tol))
          | (ok['open'] > ok['high'] * (1 + tol)) | (ok['open'] < ok['low'] * (1 - tol)))
    out.append(ok.loc[hl, ['symbol', 'date', 'close']].assign(reason='ohlc_inconsistent'))

    g = ok.groupby('symbol', sort=False)['close']
    ret = g.pct_change()
    ext = ret.abs() > MAX_ABS_RETURN
    out.append(ok.loc[ext, ['symbol', 'date', 'close']].assign(reason='extreme_return', value=ret[ext]))

    med = g.transform(lambda s: s.shift(1).rolling(21, min_periods=5).median())
    ratio = ok['close'] / med
    jump = (ratio > LEVEL_JUMP_FACTOR) | (ratio < 1.0 / LEVEL_JUMP_FACTOR)
    out.append(ok.loc[jump, ['symbol', 'date', 'close']].assign(reason='level_jump', value=ratio[jump]))

    lo, hi = (pd.Timestamp(d) for d in REDENOMINATION_WINDOW)
    gross = ok['close'] / g.shift(1)
    red = (ok['date'].between(lo, hi)
           & ((gross > REDENOMINATION_FACTOR) | (gross < 1.0 / REDENOMINATION_FACTOR)))
    out.append(ok.loc[red, ['symbol', 'date', 'close']].assign(reason='redenomination', value=gross[red]))

    res = pd.concat(out, ignore_index=True)
    if 'value' not in res.columns:
        res['value'] = np.nan
    return res.sort_values(['date', 'symbol']).reset_index(drop=True)


def _flat_rows(x: pd.DataFrame) -> pd.Series:
    """volume == 0 ve O = H = L = C (islem gormemis satir).

    "= onceki kapanis" sarti KASITLI olarak yok: yfinance kapali gunlerde
    duzeltilmis fiyati onceki kapanistan biraz farkli yazabiliyor. Ilk tanim bu
    sarti iceriyordu ve 2013-10-15'i (Kurban Bayrami, 30 sembolun hepsinde
    volume = 0) kacirdi — hakem-2 incelemesi, 2026-11-06.
    """
    c = x['close']
    same = lambda a: np.isclose(a, c, rtol=1e-9, atol=0.0)  # noqa: E731
    return (x['volume'] == 0) & same(x['open']) & same(x['high']) & same(x['low'])


def find_ghost_days(df: pd.DataFrame, threshold: float = 0.9) -> pd.DatetimeIndex:
    """Aktif sembollerin >= threshold'u duz olan gunler."""
    x = _long(df)
    frac = _flat_rows(x).groupby(x['date']).mean()
    return pd.DatetimeIndex(frac.index[frac >= threshold])


def find_stale_cells(df: pd.DataFrame, ghost_days: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """Hayalet olmayan gunlerde tek tek duz satirlar (sembol islem gormemis)."""
    x = _long(df)
    flat = _flat_rows(x)
    if ghost_days is not None and len(ghost_days):
        flat &= ~x['date'].isin(ghost_days)
    return x.loc[flat, ['symbol', 'date']].reset_index(drop=True)


def activity_mask(df: pd.DataFrame) -> pd.DataFrame:
    """tarih x sembol bool tablo: hucrede satir var mi (sutun sirasi paneldeki gibi)."""
    x = _long(df)
    order = list(dict.fromkeys(df.index.get_level_values('symbol')))
    m = pd.crosstab(x['date'], x['symbol']).astype(bool)
    return m.reindex(columns=order, fill_value=False)


INDICATOR_COLS = ('macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'adx', 'turbulence')
# B7: bfill ile doldurulan isinma sutunlari. turbulence DISARIDA: Mahalanobis'in ilk
# 252 tarihi nedensel bir 0 yer tutucudur, bfill degil (hakem-2 B7 #1 — sayilinca 2018
# baslangicli panelde her sembolden ~1 yil atiliyordu). adx ICERIDE (K18): 'wilder' ADX'in
# ilk 2n-1 satiri NaN ('bfill' modunda geriye doldurulur); legacy ADX tum panelde sabit 0
# (ya da tumuyle NaN) oldugu icin dedektor onu zaten atlar (nunique <= 1, notna yok).
BFILL_WARMUP_COLS = ('macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'adx')
WARMUP_MIN_LEN = 2          # kapi, protokol ve rapor ayni esigi kullanir (hakem-2 B7 #4)
# Bir gosterge isinma blogunun en fazla uzunlugu: 'wilder' ADX'in 2n-1 = 27 NaN satiri +
# geriye doldurulan ilk gecerli deger (n = 14). CCI 20, RSI daha kisa. Protokol, egitimde
# bu kadar satiri olan sembolun degerlendirme basini seri basi saymaz (K18).
WARMUP_MAX_ROWS = 28
WARMUP_WARN_ROWS = 30       # sembol basina bundan fazla atim beklenmez -> uyari
LARGE_MOVE = 0.20   # rapor esigi (artefakt DEGIL): BIST fiyat limitini asan kapanis hareketi


def indicator_warmup_blocks(df: pd.DataFrame, min_len: int = WARMUP_MIN_LEN,
                            cols=BFILL_WARMUP_COLS) -> dict:
    """Sembol basinda ilk degerin birebir tekrarlandigi gosterge bloklari (bfill imzasi).

    `technical_indicators.py` varsayilan ('bfill') modda isinma boslugunu bfill ile
    dolduruyor (Faz 10 Y15): ilk gecerli (GELECEKTEKI) deger geriye yayiliyor. Ornek:
    ODAS CCI 2013-05-21..2013-06-17 sabit. B7 (Faz 9 §9.7) RL kapisinda bu blogu atar
    (`prepare_rl_panel(drop_indicator_warmup=True)`). Tum panelde sabit sutunlar (legacy ADX)
    ve turbulence'in nedensel 0 basi sayilmaz (BFILL_WARMUP_COLS).
    Donus: {sembol: {sutun: blok_uzunlugu}}.
    """
    x = _long(df)
    cols = [c for c in cols if c in x.columns and x[c].nunique(dropna=True) > 1]
    out = {}
    for sym, g in x.groupby('symbol', sort=False):
        for c in cols:
            v = g[c].to_numpy()
            if len(v) == 0:
                continue
            n = int(np.argmax(v != v[0])) if np.any(v != v[0]) else len(v)
            if n >= min_len:
                out.setdefault(sym, {})[c] = n
    return out


def large_moves(df: pd.DataFrame, threshold: float = LARGE_MOVE) -> pd.DataFrame:
    """|kapanis getirisi| > threshold olan hucreler + oncesindeki takvim boslugu.

    Artefakt sayilmaz (2023-02-15 deprem sonrasi yeniden acilis gibi gercek
    hareketler var); siniflandirma icin listelenir (hakem-2 #10).
    """
    x = _long(df)
    g = x.groupby('symbol', sort=False)
    r = g['close'].pct_change()
    gap = g['date'].diff().dt.days
    m = r.abs() > threshold
    return x.loc[m, ['symbol', 'date', 'close']].assign(ret=r[m], gap_days=gap[m]).reset_index(drop=True)


def health_report(df: pd.DataFrame, ghost_threshold: float = 0.9) -> dict:
    """Paneli degistirmeden saglik ozeti (B0 veri sagligi testi bunu kullanir)."""
    x = _long(df)
    art = scan_price_artifacts(df)
    ghosts = find_ghost_days(df, ghost_threshold)
    stale = find_stale_cells(df, ghosts)
    n_dates = x['date'].nunique()
    n_sym = x['symbol'].nunique()
    numeric = [c for c in x.columns if c not in ('symbol', 'date')
               and pd.api.types.is_numeric_dtype(x[c])]
    constant = sorted(c for c in numeric if x[c].nunique(dropna=True) <= 1)
    return {
        'rows': int(len(x)),
        'n_symbols': int(n_sym),
        'start': str(x['date'].min().date()) if len(x) else None,
        'end': str(x['date'].max().date()) if len(x) else None,
        'artifacts': {k: int(v) for k, v in art['reason'].value_counts().items()},
        'n_artifacts': int(len(art)),
        'n_ghost_days': int(len(ghosts)),
        'n_stale_cells': int(len(stale)),
        'missing_cell_frac': float(1.0 - len(x) / max(n_dates * n_sym, 1)),
        'constant_columns': constant,
        'indicator_warmup_blocks': indicator_warmup_blocks(df),
        'n_large_moves': int(len(large_moves(df))),
    }


# Kapanisi etkilemeyen artefakt nedeni: baglam yalniz 'close' kullanir (K25 b).
_CLOSE_SAFE_ARTIFACTS = frozenset({'ohlc_inconsistent'})


def rl_context_before(df: pd.DataFrame, start: Optional[str] = None, n_sessions: int = 252,
                      ghost_threshold: float = 0.9) -> Tuple[pd.DataFrame, dict]:
    """RL penceresinden ONCEKI son `n_sessions` panel tarihi (K25 b, hakem-rl K18 #2).

    Nedensel fiyat olceginin EGITIM ortamina baglami: kayan pencere egitimin ilk
    gunlerinde de dolu olsun, test ortamiyla ayni tanimi gorsun. Yalniz 'close'
    kullanilir ve gozleme girmez. Hayalet gunler atilir. Kapanisi etkileyen artefakt
    (negatif fiyat, sicrama, asiri getiri, sadelestirme) varsa DataIntegrityError;
    'ohlc_inconsistent' satirlari kalir ama SAYILIR (rapor meta'ya yazilir).
    """
    start = start or RL_DATA_START
    dates = pd.to_datetime(df.index.get_level_values('date'))
    if getattr(dates, 'tz', None) is not None:
        dates = dates.tz_localize(None)
    before = dates < pd.Timestamp(start)
    uniq = np.unique(dates[before])
    keep = uniq[-n_sessions:] if len(uniq) else uniq
    sub = df[before & np.isin(dates, keep)]
    report = {'n_sessions_requested': int(n_sessions), 'n_dates': int(len(keep)),
              'start': str(pd.Timestamp(keep[0]).date()) if len(keep) else None,
              'end': str(pd.Timestamp(keep[-1]).date()) if len(keep) else None,
              'n_ghost_days': 0, 'n_close_safe_artifacts': 0}
    if sub.empty:
        return sub, report
    ghosts = find_ghost_days(sub, ghost_threshold)
    sdates = pd.to_datetime(sub.index.get_level_values('date'))
    if getattr(sdates, 'tz', None) is not None:
        sdates = sdates.tz_localize(None)
    sub = sub[~sdates.isin(ghosts)]
    art = scan_price_artifacts(sub)
    bad = art[~art['reason'].isin(_CLOSE_SAFE_ARTIFACTS)] if len(art) else art
    if len(bad):
        raise DataIntegrityError(
            f"RL baglaminda ({report['start']}..{report['end']}) kapanisi etkileyen "
            f"{len(bad)} artefakt: {bad.groupby(['reason', 'symbol']).size().head(10).to_dict()}")
    report.update(n_ghost_days=int(len(ghosts)), n_close_safe_artifacts=int(len(art)),
                  rows_per_symbol_min=int(sub.groupby(level='symbol').size().min()))
    return sub, report


def prepare_rl_panel(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ghost_threshold: float = 0.9,
    drop_stale_cells: bool = True,
    on_artifact: str = 'raise',
    drop_indicator_warmup: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """RL egitimi/HPO icin paneli pencerele, tara ve temizle.

    Sira: pencere -> artefakt taramasi -> hayalet gun -> bayat hucre.

    Args:
        start: pencere baslangici; None -> RL_DATA_START (K17).
        on_artifact: 'raise' (varsayilan; DataIntegrityError) | 'warn'.
            Artefakt satirlari hicbir modda sessizce silinmez.
        drop_indicator_warmup: Faz 9 / B7 (Faz 10 Y15). Gosterge isinma boslugu
            `technical_indicators` tarafindan bfill ile dolduruldugu icin sembolun
            ilk satirlari ilk gecerli (GELECEKTEKI) degeri tasir. True ise bu blok
            (sembolun TAM gecmisinde ilk degerin birebir tekrari, >= 2 satir) pencere
            icinde atilir; hucre EKSIK sayilir (own_last: islem yok). Varsayilan False
            (eski davranis); rota ve HPO True verir. Ayni degerin dogal tekrari da
            blok sayilabilir: fazladan birkac gercek satir atilir, sizinti olmaz.

    Returns:
        (temiz panel, rapor). Rapor model meta dosyasina ve manifeste yazilir.
    """
    start = start or RL_DATA_START
    dates = pd.to_datetime(df.index.get_level_values('date'))
    if getattr(dates, 'tz', None) is not None:
        dates = dates.tz_localize(None)
    mask = dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    win = df[mask]
    if win.empty:
        raise DataIntegrityError(f"{start}..{end or 'son'} penceresinde veri yok")

    art = scan_price_artifacts(win)
    if len(art):
        summary = art.groupby(['reason', 'symbol']).size().head(20).to_dict()
        msg = (f"RL penceresinde ({start}..{end or 'son'}) {len(art)} supheli fiyat satiri: "
               f"{summary}. Egitim bu veriyle yapilmaz (bkz. data/integrity.py).")
        if on_artifact == 'raise':
            raise DataIntegrityError(msg)
        logger.warning(msg)

    ghosts = find_ghost_days(win, ghost_threshold)
    win_dates = pd.to_datetime(win.index.get_level_values('date'))
    if getattr(win_dates, 'tz', None) is not None:
        win_dates = win_dates.tz_localize(None)
    clean = win[~win_dates.isin(ghosts)]

    n_stale = 0
    if drop_stale_cells:
        stale = find_stale_cells(clean)
        if len(stale):
            n_stale = len(stale)
            key = pd.MultiIndex.from_frame(stale[['symbol', 'date']])
            idx = clean.index
            cdates = pd.to_datetime(idx.get_level_values('date'))
            if getattr(cdates, 'tz', None) is not None:
                cdates = cdates.tz_localize(None)
            own = pd.MultiIndex.from_arrays([idx.get_level_values('symbol'), cdates])
            clean = clean[~own.isin(key)]

    warmup_dropped = {}
    if drop_indicator_warmup:
        # Blok sembolun TAM gecmisinde olculur (pencere oncesinde baslayabilir)
        blocks = indicator_warmup_blocks(df)
        idx = clean.index
        cdates = pd.to_datetime(idx.get_level_values('date'))
        if getattr(cdates, 'tz', None) is not None:
            cdates = cdates.tz_localize(None)
        drop = np.zeros(len(clean), dtype=bool)
        full_dates = _long(df).groupby('symbol', sort=False)['date']
        for sym, cols in blocks.items():
            n = max(cols.values())
            block_dates = full_dates.get_group(sym).iloc[:n]
            m = (idx.get_level_values('symbol') == sym) & cdates.isin(block_dates)
            if m.any():
                warmup_dropped[sym] = int(m.sum())
                drop |= m
        # 'nan' modunda uretilmis panel: isinma satirlari NaN (bfill imzasi yok).
        # Tumuyle NaN sutun (legacy ADX, 'nan' modu) haric, isinma sutununda NaN olan satir
        # atilir; 'wilder' ADX'in ilk 2n-1 NaN satiri da boylece atilir (K18).
        live = [c for c in BFILL_WARMUP_COLS if c in clean.columns and clean[c].notna().any()]
        if live:
            nan_rows = clean[live].isna().any(axis=1).to_numpy() & ~drop
            if nan_rows.any():
                for sym, k in pd.Series(idx.get_level_values('symbol')[nan_rows]).value_counts().items():
                    warmup_dropped[sym] = warmup_dropped.get(sym, 0) + int(k)
                drop |= nan_rows
        clean = clean[~drop]
        big = {s: k for s, k in warmup_dropped.items() if k > WARMUP_WARN_ROWS}
        if big:
            logger.warning(f"Isinma atimi beklenenden buyuk (> {WARMUP_WARN_ROWS} satir): {big}. "
                           f"Gosterge sutunlarini kontrol edin (bkz. BFILL_WARMUP_COLS).")

    # Kullanilmayan dizin seviyelerini at: atilan tarihler seviyede kalirsa
    # unstack/kesitsel hesaplar sirasiz tarih uretebiliyor (hakem-2 #11).
    clean = clean.copy()
    clean.index = clean.index.remove_unused_levels()

    act = activity_mask(clean)
    first_active = {s: str(act.index[act[s].to_numpy()].min().date())
                    for s in act.columns if act[s].any()}
    report = {
        'window_start': str(start),
        'window_end': str(end) if end else None,
        'rows_in_window': int(len(win)),
        'rows_after': int(len(clean)),
        'n_artifacts': int(len(art)),
        'ghost_threshold': ghost_threshold,
        'n_ghost_days': int(len(ghosts)),
        'ghost_days': [str(d.date()) for d in ghosts],
        'n_stale_cells_dropped': int(n_stale),
        'n_symbols': int(act.shape[1]),
        'n_dates': int(act.shape[0]),
        'missing_cell_frac': float(1.0 - act.to_numpy().mean()) if act.size else None,
        'first_active': first_active,
        # B7: atilan isinma satirlari; atilmadiysa kalan bloklar RAPORLANIR
        'indicator_warmup_dropped': warmup_dropped,
        'indicator_warmup_blocks': indicator_warmup_blocks(clean),
        'n_large_moves': int(len(large_moves(clean))),
    }
    logger.info(
        f"RL paneli: {report['window_start']}.. | {report['rows_in_window']:,} -> "
        f"{report['rows_after']:,} satir | hayalet gun {report['n_ghost_days']} | "
        f"bayat hucre {n_stale} | eksik hucre %{(report['missing_cell_frac'] or 0) * 100:.2f}"
    )
    return clean, report
