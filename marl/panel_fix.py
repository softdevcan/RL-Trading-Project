"""Faz 10 M8 paneli veri duzeltmesi `m8-v1` (M2 tasarimi §6.1; kullanici karari 2026-11-20). YALNIZ ana venv.

Yalniz M8 paneli icin ve yalniz acikca istenince uygulanir (`scripts/marl/prepare_panel_m2.py --data-fix m8-v1`).
Paylasilan veri kapisi (`data.integrity`), M2 paneli ve Milestone 0' verisi DEGISMEZ. Uc duzeltme, bu sirayla:
  1. DROP_SESSIONS: islemleri iptal edilen seansin satiri atilir (piyasa o gun kapali sayilir).
  2. RIGHTS_FIXES: yfinance bedelli sermaye artirimi duzeltmesini hak kullanim gunu E yerine daha erken bir B
     gununden uygulamis; B oncesi ham kalmis (sahte dusus). B'den onceki acilis/yuksek/dusuk/kapanis
     f = TERP / P ile carpilir; TERP = (P + r S) / (1 + r), P = E'den onceki ham kapanis, r = yeni pay orani
     (sermaye_sonra / sermaye_once - 1), S = pay basi fiyat.
  3. Kapanis yamasi: ikinci kaynak Is Yatirim HisseTekil (HGDG_KAPANIS duzeltilmis kapanis). Sembol basina
     sapma = log(bizim / IsY) - 11 seanslik merkezi medyan; |sapma| > CLOSE_TOL olan kapanis IsY x yerel olcek
     (medyanin ustu) ile degistirilir; yuksek/dusuk IsY gun ici en yuksek/en dusugunu (ayni olcekle) ve yeni
     kapanisi kapsayacak sekilde genisletilir. Duz satirlar (hacim 0, tek fiyat; `data.integrity._flat_rows`)
     ne aday ne medyan girdisidir: onlari kapi hayalet gun / bayat hucre olarak isler, takvim degismez. Yama,
     esigi asan hucre kalmayana kadar tekrarlanir (komsu yamalar medyani kaydirir); PATCH_PASSES geciste
     yakinsamazsa hata.
Acilis ve hacim degismez (ikinci kaynak acilis vermiyor; beyan). Gostergeler (MACD, RSI, CCI) duzeltilmis fiyattan
CSV'yi ureten ayni fonksiyonlarla yeniden hesaplanir (`recompute_indicators`); ADX panel hattinda `with_adx` ile.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

VERSION = 'm8-v1'
CLOSE_TOL = 0.01
MED_WIN = 11
PATCH_PASSES = 10     # olculen: 5 geciste yakinsar (964, 33, 4, 1, 0); yakinsamazsa hata
PRICE_COLS = ('open', 'high', 'low', 'close')
ISY_CACHE = os.path.join('data', 'marl', 'sources', 'isyatirim')   # git disi; SHA'lar panel meta'sina yazilir

DROP_SESSIONS = {
    '2023-02-08': 'Borsa Istanbul KAP aciklamasi 08/02/2023 (Haberturk aktarimi): 8 Subat 2023 islemlerinin tamami '
                  'iptal, pay piyasasi 14 Subat aksamina kadar kapali; Is Yatirim verisinde bu gun yok',
}
RIGHTS_FIXES = (
    {'symbol': 'SKBNK.IS', 'yf_boundary': '2024-06-04', 'ex_date': '2024-06-13', 'capital_from': 1.86e9,
     'capital_to': 2.5e9, 'price': 1.0, 'cum_close_raw': 4.49,
     'source': 'Is Yatirim HisseTekil: 13/06/2024 ham kapanis 4,49 -> 3,71, sermaye 1,86 -> 2,5 milyar TL'},
    {'symbol': 'SKBNK.IS', 'yf_boundary': '2026-10-19', 'ex_date': '2026-10-22', 'capital_from': 2.5e9,
     'capital_to': 3.5e9, 'price': 1.0, 'cum_close_raw': 4.48,
     'source': 'Is Yatirim HisseTekil: 22/10/2026 ham kapanis 4,48 -> 3,21, sermaye 2,5 -> 3,5 milyar TL'},
)


def rights_factor(fx: dict) -> float:
    r = fx['capital_to'] / fx['capital_from'] - 1.0
    p = fx['cum_close_raw']
    return (p + r * fx['price']) / ((1.0 + r) * p)


def load_isy(symbol: str, cache: str = ISY_CACHE) -> pd.DataFrame:
    """Is Yatirim HisseTekil onbellegi -> 'YYYY-MM-DD' indeksli HGDG_KAPANIS, HG_KAPANIS, HG_MIN, HG_MAX."""
    path = os.path.join(cache, f"{symbol.replace('.IS', '')}.json")
    with open(path, encoding='utf-8') as fh:
        d = pd.DataFrame(json.load(fh)['value'])
    d.index = pd.to_datetime(d['HGDG_TARIH'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
    return d[['HGDG_KAPANIS', 'HG_KAPANIS', 'HG_MIN', 'HG_MAX']].astype(float)


def isy_sha256(symbols, cache: str = ISY_CACHE) -> dict:
    out = {}
    for s in symbols:
        with open(os.path.join(cache, f"{s.replace('.IS', '')}.json"), 'rb') as fh:
            out[s] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _deviation(close: pd.Series, isy: pd.DataFrame):
    """Ortak gunlerde (log q, merkezi medyan, sapma); indeks 'YYYY-MM-DD'."""
    j = pd.concat([close.rename('c'), isy['HGDG_KAPANIS']], axis=1, join='inner').sort_index()
    j = j[(j['c'] > 0) & (j['HGDG_KAPANIS'] > 0)]
    lq = np.log(j['c'] / j['HGDG_KAPANIS'])
    med = lq.rolling(MED_WIN, center=True, min_periods=MED_WIN // 2 + 1).median()
    return lq, med, lq - med


def _band_violations(df: pd.DataFrame) -> dict:
    """Kapanis -> kapanis ve acilis boslugu marj bandi disi (marj takvimi `marl.accounting`; 2013-06+)."""
    from marl import accounting as A
    out = {}
    for name, field in (('close', 'close'), ('open_gap', 'open')):
        n = {'2013-06..2020-03-12': 0, '2020-03-13..2020-12-31': 0, '2021+': 0}
        cells = []
        for s, g in df.groupby(level='symbol', sort=False):
            g = g.droplevel('symbol').sort_index()
            dts = pd.DatetimeIndex(g.index).strftime('%Y-%m-%d')
            x = (g[field] / g['close'].shift(1)).to_numpy()
            m = np.full(len(dts), A.PRICE_MARGIN_PRE_BISTECH)
            for start, mg in A.PRICE_MARGIN_SCHEDULE:
                m[np.asarray(dts >= start)] = mg
            bad = np.isfinite(x) & (np.abs(x - 1.0) > m + A.LIMIT_TOLERANCE) & np.asarray(dts >= '2013-06-01')
            for i in np.flatnonzero(bad):
                d = dts[i]
                n['2021+' if d >= '2021-01-01' else ('2020-03-13..2020-12-31' if d >= '2020-03-13' else
                                                     '2013-06..2020-03-12')] += 1
                if d >= '2021-01-01':
                    cells.append({'date': d, 'symbol': s, 'ratio': round(float(x[i]), 4)})
        out[name] = {'n': n, 'cells_2021plus': cells}
    return out


def apply_m8_v1(df: pd.DataFrame, isy: dict) -> tuple[pd.DataFrame, dict]:
    """(symbol, date) cok indeksli fiyat paneli -> (duzeltilmis kopya, rapor). Gostergelere dokunmaz.

    isy: sembol -> `load_isy` DataFrame'i (paneldeki her sembol icin). Acilis ve hacim degismez.
    """
    for c in (*PRICE_COLS, 'volume'):
        if c not in df.columns:
            raise ValueError(f"panelde eksik sutun: {c}")
    missing = [s for s in pd.unique(df.index.get_level_values('symbol')) if s not in isy]
    if missing:
        raise ValueError(f"ikinci kaynak eksik: {missing}")
    before = _band_violations(df)
    dstr = pd.DatetimeIndex(df.index.get_level_values('date')).strftime('%Y-%m-%d')
    drop = np.asarray(dstr.isin(list(DROP_SESSIONS)))
    out = df.loc[~drop].copy()
    rep = {'version': VERSION, 'close_tol': CLOSE_TOL, 'median_window': MED_WIN,
           'drop_sessions': {d: {'rows': int(np.sum(np.asarray(dstr == d))), 'source': src}
                             for d, src in DROP_SESSIONS.items()}}
    sym = out.index.get_level_values('symbol').to_numpy()
    dstr = pd.DatetimeIndex(out.index.get_level_values('date')).strftime('%Y-%m-%d')
    px = out[list(PRICE_COLS)].to_numpy(dtype=np.float64, copy=True)
    rep['rights_fixes'] = []
    for fx in RIGHTS_FIXES:
        f = rights_factor(fx)
        rows = (sym == fx['symbol']) & np.asarray(dstr < fx['yf_boundary'])
        px[rows] *= f
        rep['rights_fixes'].append({**fx, 'factor': round(f, 6), 'rows_scaled': int(rows.sum())})
    out[list(PRICE_COLS)] = px

    from data.integrity import _flat_rows   # kapinin duz satir tanimi (tek kaynak)
    flat = _flat_rows(out).to_numpy()
    h_, l_, c_ = (out.columns.get_loc(k) for k in ('high', 'low', 'close'))
    pos_of = {s: np.flatnonzero((sym == s) & ~flat) for s in pd.unique(sym)}   # duz satirlar disarida

    def _devs():
        res = {}
        for s, pos in pos_of.items():
            _, med, dev = _deviation(pd.Series(out.iloc[pos, c_].to_numpy(), index=dstr[pos]), isy[s])
            res[s] = (med, dev, {d: p for d, p in zip(dstr[pos], pos)})
        return res

    patched, passes = {}, []                   # (sembol, gun) -> kayit; ilk eski kapanis, son yeni kapanis
    for _ in range(PATCH_PASSES):
        n_ops = 0
        for s, (med, dev, loc) in _devs().items():
            for d in dev.index[np.abs(dev.to_numpy()) > CLOSE_TOL]:
                p = loc[d]
                row = isy[s].loc[d]
                scale = float(np.exp(med.loc[d]))
                c_new = float(row['HGDG_KAPANIS'] * scale)
                c_old = float(out.iat[p, c_])
                hi, lo = [float(out.iat[p, h_]), c_new], [float(out.iat[p, l_]), c_new]
                if row['HG_KAPANIS'] > 0 and row['HG_MAX'] > 0 and row['HG_MIN'] > 0:   # IsY gun ici aralik
                    k = row['HGDG_KAPANIS'] / row['HG_KAPANIS'] * scale
                    hi.append(float(row['HG_MAX'] * k))
                    lo.append(float(row['HG_MIN'] * k))
                out.iat[p, h_], out.iat[p, l_], out.iat[p, c_] = max(hi), min(lo), c_new
                rec = patched.setdefault((s, d), {'date': d, 'symbol': s, 'dev': round(float(dev.loc[d]), 4),
                                                  'close_old': round(c_old, 6)})
                rec['close_new'] = round(c_new, 6)
                n_ops += 1
        passes.append(n_ops)
        if n_ops == 0:
            break
    if passes[-1] != 0:
        raise RuntimeError(f"m8-v1 kapanis yamasi {PATCH_PASSES} geciste yakinsamadi: {passes}")
    residual = max(float(np.nanmax(np.abs(dev.to_numpy()))) for _, dev, _ in _devs().values())
    by_year = {}
    for s, d in patched:
        by_year[d[:4]] = by_year.get(d[:4], 0) + 1
    rep['close_patch'] = {'n': len(patched), 'passes': passes, 'flat_rows_excluded': int(flat.sum()),
                          'by_year': dict(sorted(by_year.items())),
                          'cells': sorted(patched.values(), key=lambda r: (r['date'], r['symbol']))}
    rep['verify'] = {'residual_max_abs_dev_after': round(residual, 6),
                     'band_violations_before': before, 'band_violations_after': _band_violations(out)}
    return out, rep


def recompute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """MACD (3 sutun), RSI, CCI'yi `TechnicalIndicators` ile sembol basina yeniden hesaplar; isinma CSV'deki gibi
    ffill -> bfill -> 0 (`add_all_indicators` warmup='bfill'). Diger sutunlar (adx, turbulence) degismez."""
    from data.technical_indicators import TechnicalIndicators as TI
    out = df.copy()
    cols = ['macd', 'macd_signal', 'macd_hist', 'rsi', 'cci']
    sym = out.index.get_level_values('symbol')
    for s in pd.unique(sym):
        pos = np.flatnonzero(sym == s)
        x = out.iloc[pos].droplevel('symbol')
        order = np.argsort(x.index.values, kind='stable')   # zaman sirasi (with_adx ile ayni)
        x = x.iloc[order]
        m, ms, mh = TI.calculate_macd(x)
        y = pd.DataFrame({'macd': m, 'macd_signal': ms, 'macd_hist': mh, 'rsi': TI.calculate_rsi(x),
                          'cci': TI.calculate_cci(x)}).ffill().bfill().fillna(0)
        for k in cols:
            out.iloc[pos[order], out.columns.get_loc(k)] = y[k].to_numpy(dtype=np.float64)
    return out
