"""Faz 11 S1: M8 ileri paneli (ileri veri testi = sanal para simulasyonu). YALNIZ ana venv (data/ import eder).

Plan `docs/development/phase-11-integration-plan.md` §3 S1. Ilke: dondurulmus `panel_m8.npz` satirlari HIC degismez;
yeni seanslar yalniz SONA eklenir ve bir kez yazilan ileri satir da degismez (append-only). Panel her guncellemede
M8 hattinin AYNI fonksiyonlariyla bastan kurulur ve onceki panelin tum satirlariyla bit-es oldugu denetlenir:

    dondurulmus kaynak CSV  --apply_m8_v1 (dondurulmus Is Yatirim kopyasi)-->  + ileri satirlar (rows.csv)
      --recompute_indicators--> --with_adx('wilder')--> --prepare_rl_panel--> --build_arrays--> + rf  -> panel_m8_fwd.npz

Kaynaklar `data/marl/frozen/m8-source/` altina bir kez kopyalanir (SHA'lar panel_m8 meta'siyla ayni olmali): uygulamanin
veri sayfasi `data/bist/stock_data_with_indicators.csv`'yi artimli guncelleyebilir; ileri hat o dosyayi OKUMAZ.

Ileri satir kurali (`chain_day`; yalniz d ve sembolun son yazili gunu okunur, ileriye bakmaz):
  - Zincir: s = son yazili kapanis / taze yfinance (auto_adjust) kapanisi ayni gunde; d satiri = s x yfinance OHLC,
    hacim = yfinance hacmi / s (ciro korunur). Temettu ve bolunme o gunun getirisine yansir; gecmis satir degismez.
  - Capraz kaynak (m8-v1'in nedensel karsiligi): Is Yatirim duzeltilmis kapanis getirisiyle zincirlenmis aday
    c_isy = son kapanis x G_d / G_onceki. |log(aday / c_isy)| > CLOSE_TOL (m8-v1 ile ayni %1) ise kapanis = c_isy,
    yuksek/dusuk Is Yatirim gun ici uclariyla (ayni olcekle) genisletilir. Olculen (2021+, panel_m8 ve Is Yatirim
    onbellegi): gunluk getiri farki %1'i 11 hucrede asti; son gunde iki kaynagin ham kapanisi birebir ayni.
    m8-v1'in 11 seanslik MERKEZI medyani 5 seans ileriye bakar; ileri veride kullanilamaz (plan §3.3).
Gun kapisi (`day_gate`; ilk engelde durur, sonraki gunler yazilmaz, sebep kayda girer):
  seans_degil (yfinance satiri sembollerin yarisindan az: tatil -> atlanir, yazilmaz), kesim_saati_oncesi,
  isy_bekleniyor, rf_bayat, marj_bandi_disi (kapanis ya da acilis boslugu +-(marj + 0,005) disinda: veri hatasi ya da
  marj degisikligi adayi; elle inceleme), yf_capa_yok.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from marl import panel as PN
from marl import panel_fix as PF

PANEL_M8 = os.path.join('data', 'marl', 'panel_m8.npz')
SOURCE = os.path.join('data', 'marl', 'frozen', 'm8-source')
SOURCE_CSV = 'stock_data_with_indicators.csv'
FWD_ROOT = os.path.join('data', 'marl', 'forward', 'm8')
ROWS = 'rows.csv'
PANEL_FWD = 'panel_m8_fwd.npz'
RF_FILE = 'risk_free.csv'
LOG = 'log.jsonl'

CLOSE_TOL = PF.CLOSE_TOL           # %1, m8-v1 ile ayni
SESSION_MIN_FRAC = 0.5             # yfinance satiri olan sembol orani; altinda gun seans sayilmaz (tatil)
RF_MAX_AGE_DAYS = 5                # rf_t icin d_{t-1}'de bilinen son fonlama gozleminin en fazla yasi (takvim gunu)
CUTOFF_HOUR_TRT = 19               # ayni gunun satiri ancak 19:00 TRT'den sonra (kapanis 18:00 + veri gecikmesi)
ROW_COLS = ('open', 'high', 'low', 'close', 'volume')


def _sha(path: str) -> str:
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Dondurulmus kaynak
# ---------------------------------------------------------------------------------------------------------------------
def freeze_source(panel_npz: str = PANEL_M8, dest: str = SOURCE, csv_src: str | None = None,
                  isy_src: str = PF.ISY_CACHE) -> dict:
    """panel_m8'in kaynak CSV'sini ve m8-v1 Is Yatirim dosyalarini kopyalar; SHA'lar panel meta'siyla ayni olmali."""
    if os.path.exists(dest) and os.listdir(dest):
        raise ValueError(f'hedef bos degil: {dest} (kaynak ustune yazilmaz)')
    meta = PN.load_npz(panel_npz)['meta']
    csv_src = csv_src or meta['source_csv']
    want_isy = meta['data_fix_summary']['isy_sha256']
    if _sha(csv_src) != meta['source_sha256']:
        raise ValueError('kaynak CSV panel_m8 meta SHA\'siyla ayni degil')
    os.makedirs(os.path.join(dest, 'isyatirim'), exist_ok=True)
    shutil.copy2(csv_src, os.path.join(dest, SOURCE_CSV))
    for s, sha in want_isy.items():
        src = os.path.join(isy_src, f"{s.replace('.IS', '')}.json")
        if _sha(src) != sha:
            raise ValueError(f'Is Yatirim dosyasi panel_m8 meta SHA\'siyla ayni degil: {s}')
        shutil.copy2(src, os.path.join(dest, 'isyatirim', os.path.basename(src)))
    man = {'panel': panel_npz.replace(os.sep, '/'), 'panel_sha256': _sha(panel_npz),
           'csv_sha256': meta['source_sha256'], 'isy_sha256': want_isy,
           'frozen': datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(dest, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1, ensure_ascii=False)
    verify_source(dest)
    return man


def verify_source(dest: str = SOURCE) -> dict:
    with open(os.path.join(dest, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    bad = [] if _sha(os.path.join(dest, SOURCE_CSV)) == man['csv_sha256'] else [SOURCE_CSV]
    bad += [s for s, sha in man['isy_sha256'].items()
            if _sha(os.path.join(dest, 'isyatirim', f"{s.replace('.IS', '')}.json")) != sha]
    if bad:
        raise ValueError(f'dondurulmus kaynak dogrulanamadi: {bad}')
    return man


def load_source(dest: str = SOURCE):
    """(m8-v1 duzeltilmis kaynak df, rapor): `prepare_panel_m2 --data-fix m8-v1` ile ayni adimlar, dondurulmus kopyadan."""
    from data.data_fetcher import DataFetcher
    verify_source(dest)
    f = DataFetcher()
    f.data_dir = dest
    df = f.load_data(SOURCE_CSV)
    syms = list(pd.unique(df.index.get_level_values('symbol')))
    isy = {s: PF.load_isy(s, cache=os.path.join(dest, 'isyatirim')) for s in syms}
    return PF.apply_m8_v1(df, isy)


# ---------------------------------------------------------------------------------------------------------------------
# Panel kurulumu
# ---------------------------------------------------------------------------------------------------------------------
def rows_frame(rows: pd.DataFrame, columns) -> pd.DataFrame:
    """rows.csv (date, symbol, OHLCV, ...) -> kaynak df bicimi ((symbol, date) indeks, kaynak sutunlari; gostergeler NaN)."""
    if rows is None or not len(rows):
        return None
    x = pd.DataFrame(index=pd.MultiIndex.from_arrays(
        [rows['symbol'].to_numpy(), pd.to_datetime(rows['date']).to_numpy()], names=['symbol', 'date']),
        columns=list(columns), dtype=np.float64)
    for c in ROW_COLS:
        x[c] = rows[c].to_numpy(dtype=np.float64)
    return x


def build_arrays(src_df: pd.DataFrame, rows: pd.DataFrame | None = None, rf_csv: str | None = None) -> dict:
    """Dondurulmus kaynak (m8-v1 sonrasi) + ileri satirlar -> panel dizileri (+ rf). M8 hattinin ayni fonksiyonlari."""
    from data.integrity import prepare_rl_panel
    from data.technical_indicators import with_adx
    df = src_df
    fr = rows_frame(rows, src_df.columns)
    if fr is not None:
        last = pd.to_datetime(src_df.index.get_level_values('date')).max()
        if pd.to_datetime(fr.index.get_level_values('date')).min() <= last:
            raise ValueError('ileri satir dondurulmus kaynagin son gunune ya da oncesine dusuyor')
        df = pd.concat([src_df, fr])
    df = with_adx(PF.recompute_indicators(df), 'wilder')
    panel, _ = prepare_rl_panel(df, start=None, end=None, drop_indicator_warmup=True)
    arrays = PN.build_arrays(panel)
    if rf_csv is not None:
        from evaluation.risk_free import risk_free_series
        arrays['rf'] = risk_free_series(list(arrays['dates']), csv_path=rf_csv).to_numpy(dtype=np.float64)
    return arrays


PANEL_KEYS = ('open', 'high', 'low', 'close', 'volume', 'rsi', 'cci', 'adx', 'macd', 'rf')


def check_prefix(new: dict, ref: dict, keys=PANEL_KEYS) -> dict:
    """ref'in tum satirlari new'de bit-es mi (ayni takvim basi, ayni semboller; NaN konumlari dahil)."""
    T = len(ref['dates'])
    errs = []
    if list(new['symbols']) != list(ref['symbols']):
        errs.append('sembol listesi farkli')
    if len(new['dates']) < T or list(new['dates'][:T]) != list(ref['dates']):
        errs.append('takvim basi farkli')
    for k in keys:
        if k not in ref:
            continue
        if k not in new or not np.array_equal(new[k][:T], ref[k], equal_nan=True):
            errs.append(f'{k} farkli')
    if errs:
        raise ValueError('ileri panel onceki satirlari degistirdi: ' + ', '.join(errs))
    return {'prefix_rows': int(T), 'new_rows': int(len(new['dates']) - T), 'keys': [k for k in keys if k in ref]}


# ---------------------------------------------------------------------------------------------------------------------
# Ileri satir kurali ve gun kapisi
# ---------------------------------------------------------------------------------------------------------------------
def anchors(arrays: dict) -> dict:
    """Sembol basina son gecerli kapanis ve tarihi (zincir capasi)."""
    out = {}
    for i, s in enumerate(arrays['symbols']):
        ok = np.flatnonzero(np.isfinite(arrays['close'][:, i]) & (arrays['close'][:, i] > 0))
        if ok.size:
            out[str(s)] = (str(arrays['dates'][ok[-1]]), float(arrays['close'][ok[-1], i]))
    return out


def chain_day(d: str, anchor: dict, yf: dict, isy: dict) -> tuple[list, dict]:
    """Bir gunun ileri satirlari. anchor {sym: (tarih, kapanis)}; yf {sym: df ('YYYY-MM-DD' indeks, OHLCV auto_adjust)};
    isy {sym: `load_isy` bicimi}. Donus (satirlar, sembol raporu). Eksik yfinance satiri -> satir yok (hucre eksik)."""
    rows, rep = [], {}
    for s, (ad, ac) in anchor.items():
        y = yf.get(s)
        if y is None or d not in y.index:
            rep[s] = {'status': 'yf_yok'}
            continue
        if ad not in y.index or not y.loc[ad, 'close'] > 0:
            rep[s] = {'status': 'yf_capa_yok', 'anchor': ad}
            continue
        sc = ac / float(y.loc[ad, 'close'])
        o, h, lo, c = (sc * float(y.loc[d, k]) for k in ('open', 'high', 'low', 'close'))
        vol = float(y.loc[d, 'volume']) / sc
        g = isy.get(s)
        if g is None or d not in g.index or ad not in g.index:
            rep[s] = {'status': 'isy_yok'}
            continue
        c_isy = ac * float(g.loc[d, 'HGDG_KAPANIS']) / float(g.loc[ad, 'HGDG_KAPANIS'])
        dev = float(np.log(c / c_isy))
        patched = abs(dev) > CLOSE_TOL
        if patched:
            k = c_isy / float(g.loc[d, 'HG_KAPANIS'])
            c = c_isy
            h = max(h, k * float(g.loc[d, 'HG_MAX']), c)
            lo = min(lo, k * float(g.loc[d, 'HG_MIN']), c)
        rows.append({'date': d, 'symbol': s, 'open': o, 'high': h, 'low': lo, 'close': c, 'volume': vol,
                     'anchor_date': ad, 'close_anchor': ac, 'chain_scale': sc, 'isy_dev': dev, 'patched': bool(patched)})
        rep[s] = {'status': 'ok', 'isy_dev': dev, 'patched': bool(patched)}
    return rows, rep


def rf_age(d_prev: str, rate: pd.Series) -> int | None:
    """d_{t-1}'de bilinen son oran gozleminin yasi (gun); gozlem yoksa None."""
    known = rate.index[rate.index <= pd.Timestamp(d_prev)]
    return None if not len(known) else int((pd.Timestamp(d_prev) - known.max()).days)


def day_gate(d: str, d_prev: str, rows: list, rep: dict, n_symbols: int, rate: pd.Series,
             as_of: datetime, margin: float = 0.10, tol: float = 0.005) -> dict:
    """Gun kabul karari. {'accept': bool, 'skip': bool, 'reason': str|None, ...}. skip = seans degil (tatil)."""
    out = {'date': d, 'accept': False, 'skip': False, 'reason': None}
    have = sum(v['status'] != 'yf_yok' for v in rep.values())
    if have < SESSION_MIN_FRAC * n_symbols:
        return {**out, 'skip': True, 'reason': 'seans_degil', 'yf_rows': have}
    if d == as_of.strftime('%Y-%m-%d') and as_of.hour < CUTOFF_HOUR_TRT:
        return {**out, 'reason': 'kesim_saati_oncesi'}
    if d > as_of.strftime('%Y-%m-%d'):
        return {**out, 'reason': 'gelecek_gun'}
    for status, reason in (('isy_yok', 'isy_bekleniyor'), ('yf_capa_yok', 'yf_capa_yok')):
        miss = sorted(s for s, v in rep.items() if v['status'] == status)
        if miss:
            return {**out, 'reason': reason, 'symbols': miss}
    age = rf_age(d_prev, rate)
    if age is None or age > RF_MAX_AGE_DAYS:
        return {**out, 'reason': 'rf_bayat', 'rf_age_days': age}
    lim = margin + tol
    band = []
    for r in rows:
        if r['anchor_date'] != d_prev:      # cok seanslik bosluk: tek gunluk marj uygulanmaz
            continue
        ac = r['close_anchor']
        for name, x in (('close', r['close']), ('open', r['open'])):
            if abs(x / ac - 1.0) > lim:
                band.append({'symbol': r['symbol'], 'field': name, 'ratio': float(x / ac)})
    if band:
        return {**out, 'reason': 'marj_bandi_disi', 'band': band}
    return {**out, 'accept': True, 'rf_age_days': age, 'n_rows': len(rows),
            'n_patched': int(sum(r['patched'] for r in rows))}
