"""
Faz 9 / P (Faz 10 §3.1.3 P1-P3; K11 v1, K19, Y26) — nokta-zamanli orneklem disi tahmin tablosu.

Her (sembol, islem gunu) icin YALNIZ o gunden once gerceklesmis etiketlerle egitilmis modellerin
tahmini. Ajanlarin gozlemine (S-obs) ve mesaj dikkatine (S-msg) giden tek tahmin kaynagi.

Kurgu (tasarim varsayimlari isaretli; M8 on kaydinda sabitlenir):
  * Model kapsami K11 v1: uc GBM (XGBoost, LightGBM, CatBoost) + lojistik meta (kullanici teyidi
    2026-11-15; P3 olcumu `measurements/phase9-p/p3_cost.json`: tam ensemble ~66 sa, 3 GBM ~0,9 sa).
  * Yalniz fiyat/hacim ozellikleri: makro 2018'de basliyor, temel oranlar tek anlik goruntu
    (nokta-zamanli degil, R1 `fundamentals_pit`). F olcumuyle ayni ozellik kumesi.
  * Capa (yeniden egitim) tarihleri: temiz panelin ORTAK takviminde `FIRST_ANCHOR`'dan itibaren
    her `RETRAIN_EVERY` (63) seansta bir (tasarim varsayimi). Blok k = [capa_k, capa_{k+1}).
  * Taban modeller (blok k): etiketi capa_k'dan ONCE gerceklesmis satirlar (label_date < capa_k;
    purge h — Lopez de Prado 2018), genisleyen pencere, en az `MIN_TRAIN_ROWS`. Erken durdurma
    icin son %15 dogrulama, arada h satir arindirma.
  * pred_logret = uc taban tahmininin esit agirlikli ortalamasi (ek uydurulan katman yok;
    tasarim varsayimi). model_disagreement = taban tahminlerinin std'si (ddof=0).
  * Meta (K19, Y26): p(UP) lojistigi, taban modellerin GECMIS bloklardaki ORNEKLEM DISI
    tahminlerinde uydurulur (etiketi capa_k'dan once gerceklesmis son `META_WINDOW` satir).
    Pencere kronolojik ikiye bolunur: ilk yari (son h satir arindirilir) lojistik, ikinci yari
    sicaklik. Boylece %80 yeniden uyum + orneklem ici meta yeniden uyumu (ensemble.py) bu yolda
    YOK: her girdi orneklem disi, her satirin son egitim tarihi kayitli.
  * Sicaklik T capa basina 30 sembolde HAVUZLU (Faz 10 §3.1.2 tasarim karari); sembol basina T
    duyarlilik icin rapora yazilir. p_up_cal = sigmoid(logit / T_havuz), p_up_raw = sigmoid(logit).
  * informative_asof: capa aninda kalibrasyon yarisinda bilgi kapisi (T sinirda degil, BSS > 0,
    AUC blok bootstrap GA'si 0,5'i dislar) — nedensel; ajan gozlemi bunu kullanabilir.
  * Eksik hucre ACIK maskeyle: `mask_reason` (ok / no_data / no_features / insufficient_history
    / model_failed / meta_insufficient). Sessiz 0,5 varsayilani YOK (p NaN kalir).

Kabul (P1): B0 — t sonrasi veri degistirilince t'ye kadarki satirlar bit-es; mutasyonla
(arindirmasiz egitim ya da meta) B0 kirilir (`tests/test_pit_table.py`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TABLE_VERSION = 'pit-v1'
V1_MODELS = ('xgboost', 'lightgbm', 'catboost')
PIT_DATA_START = '2009-09-01'   # 2009-08-04 MGROS sicramasindan sonra (data/integrity taramasi)
FIRST_ANCHOR = '2012-01-01'     # ilk blok; meta ~1 yil sonra (RL penceresi 2013-06 oncesi) dolar
RETRAIN_EVERY = 63
MIN_TRAIN_ROWS = 504
VAL_FRAC = 0.15
META_WINDOW = 504
MIN_HALF = 100
# Bilgi kapisinin degerlendirme penceresi: wf-v1'in ilk dogrulama yili (2020) ONCESI —
# RL secim/test pencerelerinin hicbiri okunmaz (evaluation/splits.py WF_V1).
DEV_WINDOW = ('2013-06-01', '2020-01-01')
T_BOUNDS = (0.05, 20.0)

# Yalniz test mutasyonu (tests/test_pit_table.py): False -> egitim/meta satiri "tarih < capa"
# (etiketi capadan SONRA gerceklesen satirlar da girer; B0 kirilmali).
PURGE_BASE = True
PURGE_META = True

MASK_REASONS = ('ok', 'no_data', 'no_features', 'insufficient_history', 'model_failed',
                'meta_insufficient')


def horizon_steps(horizon: str) -> int:
    if horizon not in ('daily', 'weekly'):
        raise ValueError("horizon 'daily' veya 'weekly'")
    return 1 if horizon == 'daily' else 5


def make_anchors(calendar, first_anchor: str = FIRST_ANCHOR,
                 every: int = RETRAIN_EVERY) -> pd.DatetimeIndex:
    cal = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(calendar))))
    i0 = int(cal.searchsorted(pd.Timestamp(first_anchor), side='left'))
    if i0 >= len(cal):
        raise ValueError(f"takvimde {first_anchor} sonrasi gun yok")
    return cal[i0::every]


def _block_of(dates: pd.DatetimeIndex, anchors: pd.DatetimeIndex) -> np.ndarray:
    """Her tarihin blok numarasi (capa_k <= t < capa_{k+1}); ilk capadan once -1."""
    return anchors.searchsorted(dates, side='right') - 1


# ----------------------------------------------------------------------------
# Sembol cercevesi
# ----------------------------------------------------------------------------

def symbol_frame(df_sym: pd.DataFrame, symbol: str, horizon: str):
    """Ozellik matrisi + ham kapanistan etiket ve etiket tarihi (sembolun KENDI seanslari).

    Ozellikler cikarim yolundan (`build_inference_features`: son h etiketsiz satir tutulur).
    Etiket: y_up = 1[kapanis_{i+h} > kapanis_i], y_ret = log(kapanis_{i+h} / kapanis_i);
    label_date = sembolun i+h. seansi (etiketin GERCEKLESTIGI gun).
    """
    from prediction.feature_engineer import PredictionFeatureEngineer

    h = horizon_steps(horizon)
    d = df_sym[['open', 'high', 'low', 'close', 'volume']].copy()
    d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(
        pd.to_datetime(d.index), 'tz', None) is not None else pd.to_datetime(d.index)
    d = d.sort_index()
    fe = PredictionFeatureEngineer(horizon)
    feat = fe.build_inference_features(d, symbol, target_type='log_return')
    cols = fe.get_feature_columns(feat)
    raw = d['close']
    fut = raw.shift(-h)
    ldate = pd.Series(raw.index, index=raw.index).shift(-h)
    fr = pd.DataFrame(index=feat.index)
    fr['label_date'] = ldate.reindex(feat.index)
    fr['y_ret'] = np.log(fut / raw).reindex(feat.index)
    fr['y_up'] = (fut > raw).astype(float).where(fut.notna()).reindex(feat.index)
    X = np.nan_to_num(feat[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return fr, X, cols


def _model(model_type: str, horizon: str, seed: Optional[int], params_override: Optional[dict]):
    from prediction.models.catboost_model import CatBoostModel
    from prediction.models.lightgbm_model import LightGBMModel
    from prediction.models.xgboost_model import XGBoostModel

    cls = {'xgboost': XGBoostModel, 'lightgbm': LightGBMModel, 'catboost': CatBoostModel}[model_type]
    m = cls(horizon=horizon)
    p = dict(m.params)
    if seed is not None:
        for k in ('random_state', 'random_seed'):
            if k in p:
                p[k] = int(seed)
    p.update((params_override or {}).get(model_type, {}))
    m.params = p
    m.target_type = 'log_return'
    return m


def base_predictions(df_sym: pd.DataFrame, symbol: str, anchors: pd.DatetimeIndex,
                     horizon: str = 'daily', models: Sequence[str] = V1_MODELS,
                     seed: Optional[int] = None, params_override: Optional[dict] = None
                     ) -> Tuple[pd.DataFrame, List[dict]]:
    """Asama 1: blok basina taban modeller, blok satirlarinda orneklem disi getiri tahmini."""
    h = horizon_steps(horizon)
    fr, X, cols = symbol_frame(df_sym, symbol, horizon)
    dates = fr.index
    y = fr['y_ret'].to_numpy(dtype=np.float64)
    out = fr[['label_date', 'y_up']].copy()
    out['block'] = _block_of(dates, anchors)
    for m in models:
        out[f'ret_{m}'] = np.nan
    out['train_end'] = pd.NaT
    out['label_end'] = pd.NaT
    out['model_version'] = None
    out['base_reason'] = None
    log: List[dict] = []
    for k, a in enumerate(anchors):
        pm = (out['block'] == k).to_numpy()
        if not pm.any():
            continue
        if PURGE_BASE:
            tm = (fr['label_date'].notna() & (fr['label_date'] < a)).to_numpy()
        else:
            tm = (dates < a) & fr['y_ret'].notna().to_numpy()
        tm &= np.isfinite(y)
        idx = np.flatnonzero(tm)
        entry = {'anchor': str(a.date()), 'n_train': int(len(idx)), 'n_pred': int(pm.sum())}
        if len(idx) < MIN_TRAIN_ROWS:
            out.loc[pm, 'base_reason'] = 'insufficient_history'
            log.append({**entry, 'status': 'insufficient_history'})
            continue
        n_val = int(len(idx) * VAL_FRAC)
        tr, va = idx[:len(idx) - n_val - h], idx[len(idx) - n_val:]
        failed = {}
        preds = {}
        for mt in models:
            try:
                mdl = _model(mt, horizon, seed, params_override)
                mdl.train(X[tr], y[tr].astype(np.float32), X[va], y[va].astype(np.float32),
                          feature_cols=cols)
                preds[mt] = np.asarray(mdl._predict_raw(X[pm]), dtype=np.float64)
            except Exception as exc:  # sessiz dusme yok: blok maskelenir
                failed[mt] = str(exc)[:200]
        if failed:
            out.loc[pm, 'base_reason'] = 'model_failed'
            log.append({**entry, 'status': 'model_failed', 'failed': failed})
            continue
        for mt, v in preds.items():
            out.loc[pm, f'ret_{mt}'] = v
        out.loc[pm, 'train_end'] = dates[idx[-1]]
        out.loc[pm, 'label_end'] = fr['label_date'].iloc[idx].max()
        out.loc[pm, 'model_version'] = f"{TABLE_VERSION}:{a.date()}"
        out.loc[pm, 'base_reason'] = 'ok'
        log.append({**entry, 'status': 'ok', 'train_end': str(dates[idx[-1]].date())})
    return out, log


# ----------------------------------------------------------------------------
# Asama 2: orneklem disi meta + havuzlu sicaklik
# ----------------------------------------------------------------------------

def _ret_cols(models: Sequence[str]) -> List[str]:
    return [f'ret_{m}' for m in models]


def meta_layer(base: Dict[str, pd.DataFrame], anchors: pd.DatetimeIndex, horizon: str = 'daily',
               models: Sequence[str] = V1_MODELS, ci_boot: int = 1000) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    """Capa basina: sembol basina lojistik (gecmis OOF ilk yari), havuzlu T (ikinci yari)."""
    from sklearn.linear_model import LogisticRegression
    from evaluation import calibration as cal

    h = horizon_steps(horizon)
    rc = _ret_cols(models)
    res = {}
    for s, b in base.items():
        r = pd.DataFrame(index=b.index)
        for c in ('logit', 'p_up_raw', 'p_up_cal', 'temperature', 'base_rate_asof'):
            r[c] = np.nan
        r['meta_label_end'] = pd.NaT
        r['informative_asof'] = False
        r['meta_reason'] = None
        res[s] = r
    alog: List[dict] = []
    for k, a in enumerate(anchors):
        fits = {}
        reasons = {}
        for s, b in base.items():
            blk = (b['block'] == k).to_numpy() & (b['base_reason'] == 'ok').to_numpy()
            if not blk.any():
                continue
            ok = (b['base_reason'] == 'ok') & b['y_up'].notna()
            if PURGE_META:
                hm = ok & b['label_date'].notna() & (b['label_date'] < a)
            else:
                hm = ok & (b.index < a)
            hist = b[hm.to_numpy()].iloc[-META_WINDOW:]
            half = len(hist) // 2
            fit, calh = hist.iloc[:max(half - h, 0)], hist.iloc[half:]
            yf, yc = fit['y_up'].to_numpy(), calh['y_up'].to_numpy()
            if (len(fit) < MIN_HALF or len(calh) < MIN_HALF or len(np.unique(yf)) < 2
                    or len(np.unique(yc)) < 2):
                reasons[s] = f'meta_insufficient (fit {len(fit)}, calib {len(calh)})'
                res[s].loc[blk, 'meta_reason'] = 'meta_insufficient'
                continue
            Xf = fit[rc].to_numpy(dtype=np.float64)
            mu, sd = Xf.mean(axis=0), Xf.std(axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit((Xf - mu) / sd, yf)
            zc = clf.decision_function((calh[rc].to_numpy(dtype=np.float64) - mu) / sd)
            fits[s] = {'clf': clf, 'mu': mu, 'sd': sd, 'zc': zc, 'yc': yc, 'blk': blk,
                       'base_rate': float(yf.mean()), 'label_end': hist['label_date'].max(),
                       'T_symbol': cal.fit_temperature(zc, yc, bounds=T_BOUNDS)}
        entry = {'anchor': str(a.date()), 'n_symbols': len(fits), 'skipped': reasons}
        if not fits:
            alog.append({**entry, 'T_pooled': None})
            continue
        T = cal.fit_temperature(np.concatenate([f['zc'] for f in fits.values()]),
                                np.concatenate([f['yc'] for f in fits.values()]), bounds=T_BOUNDS)
        at_bound = bool(T >= 0.99 * T_BOUNDS[1] or T <= T_BOUNDS[0] / 0.99)
        info_map = {}
        for s, f in fits.items():
            b, blk = base[s], f['blk']
            z = f['clf'].decision_function((b.loc[blk, rc].to_numpy(dtype=np.float64) - f['mu']) / f['sd'])
            pc = cal.apply_temperature(f['zc'], T)
            bss = cal.brier_skill(pc, f['yc'], f['base_rate'])
            ci = cal.block_bootstrap_ci(cal.auc, pc, f['yc'], n_boot=ci_boot)
            info = bool(not at_bound and np.isfinite(bss) and bss > 0 and ci is not None and ci[0] > 0.5)
            info_map[s] = info
            r = res[s]
            r.loc[blk, 'logit'] = z
            r.loc[blk, 'p_up_raw'] = cal.sigmoid(z)
            r.loc[blk, 'p_up_cal'] = cal.apply_temperature(z, T)
            r.loc[blk, 'temperature'] = T
            r.loc[blk, 'base_rate_asof'] = f['base_rate']
            r.loc[blk, 'meta_label_end'] = f['label_end']
            r.loc[blk, 'informative_asof'] = info
            r.loc[blk, 'meta_reason'] = 'ok'
        alog.append({**entry, 'T_pooled': T, 'T_at_bound': at_bound,
                     'T_symbol': {s: f['T_symbol'] for s, f in fits.items()},
                     'n_informative_asof': int(sum(info_map.values()))})
    return res, alog


# ----------------------------------------------------------------------------
# Tablo
# ----------------------------------------------------------------------------

def load_panel(df: Optional[pd.DataFrame] = None, start: str = PIT_DATA_START,
               end: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Gostergeli RL paneli -> veri kapisi (hayalet gun, bayat hucre). Artefaktlar UYARI ile kalir.

    Pencere RL'den (2013-06) erken basliyor (meta RL penceresinin ilk gununde dolsun diye).
    Pencere icinde tarayicinin buldugu supheli satirlar egitim girdisinde KALIR ve
    `report['artifacts_in_window']`'a yazilir (sessiz silme yok).
    """
    from data.integrity import prepare_rl_panel, scan_price_artifacts
    if df is None:
        from data.data_fetcher import DataFetcher
        df = DataFetcher().load_data('stock_data_with_indicators.csv')
    panel, report = prepare_rl_panel(df, start=start, end=end, on_artifact='warn')
    dates = pd.to_datetime(df.index.get_level_values('date'))
    if getattr(dates, 'tz', None) is not None:
        dates = dates.tz_localize(None)
    m = dates >= pd.Timestamp(start)
    if end is not None:
        m &= dates <= pd.Timestamp(end)
    art = scan_price_artifacts(df[m])
    report['artifacts_in_window'] = [
        {'symbol': r.symbol, 'date': str(pd.Timestamp(r.date).date()), 'reason': r.reason}
        for r in art.itertuples()]
    return panel, report


def _symbol_ohlcv(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sub = panel[panel.index.get_level_values('symbol') == symbol]
    d = sub.reset_index()
    d['date'] = pd.to_datetime(d['date'])
    if getattr(d['date'].dt, 'tz', None) is not None:
        d['date'] = d['date'].dt.tz_localize(None)
    return d.set_index('date').sort_index()[['open', 'high', 'low', 'close', 'volume']]


def build_pit_table(panel: pd.DataFrame, symbols: Optional[Sequence[str]] = None,
                    horizon: str = 'daily', first_anchor: str = FIRST_ANCHOR,
                    every: int = RETRAIN_EVERY, models: Sequence[str] = V1_MODELS,
                    seed: Optional[int] = None, params_override: Optional[dict] = None,
                    ci_boot: int = 1000, progress: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Temiz panelden (MultiIndex date, symbol) tam izgara PIT tablosu + rapor (P2 kapsami)."""
    t0 = time.time()
    syms = list(symbols) if symbols is not None else sorted(set(panel.index.get_level_values('symbol')))
    cal_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(panel.index.get_level_values('date')))))
    if getattr(cal_dates, 'tz', None) is not None:
        cal_dates = cal_dates.tz_localize(None)
    anchors = make_anchors(cal_dates, first_anchor, every)
    base, base_log, ohlcv_dates, secs = {}, {}, {}, {}
    for s in syms:
        ts = time.time()
        d = _symbol_ohlcv(panel, s)
        ohlcv_dates[s] = d.index
        base[s], base_log[s] = base_predictions(d, s, anchors, horizon, models, seed, params_override)
        secs[s] = round(time.time() - ts, 1)
        if progress:
            print(f"  [P] {s}: {secs[s]} sn, {sum(e['status'] == 'ok' for e in base_log[s])}"
                  f"/{len(base_log[s])} blok", flush=True)
    meta, meta_log = meta_layer(base, anchors, horizon, models, ci_boot=ci_boot)

    grid_dates = cal_dates[cal_dates >= anchors[0]]
    rc = _ret_cols(models)
    frames = []
    for s in syms:
        b = base[s].join(meta[s])
        g = pd.DataFrame(index=grid_dates)
        g.index.name = 'date'
        g = g.join(b.drop(columns=['y_up', 'block']), how='left')
        has_row = grid_dates.isin(ohlcv_dates[s])
        has_feat = grid_dates.isin(b.index)
        reason = np.where(~has_row, 'no_data', np.where(~has_feat, 'no_features', None)).astype(object)
        br = g['base_reason'].to_numpy(dtype=object)
        mr = g['meta_reason'].to_numpy(dtype=object)
        for i in range(len(g)):
            if reason[i] is None:
                reason[i] = br[i] if br[i] not in (None, 'ok') else (
                    'ok' if mr[i] == 'ok' else 'meta_insufficient')
        g['mask_reason'] = reason
        g['base_ok'] = g['base_reason'].eq('ok').to_numpy() & has_feat
        g['meta_ok'] = g['mask_reason'].eq('ok')
        r = g[rc].to_numpy(dtype=np.float64)
        g['pred_logret'] = np.where(g['base_ok'], r.mean(axis=1), np.nan)
        g['model_disagreement'] = np.where(g['base_ok'], r.std(axis=1), np.nan)
        g['informative_asof'] = g['informative_asof'].fillna(False).astype(bool) & g['meta_ok']
        g['symbol'] = s
        frames.append(g.drop(columns=['base_reason', 'meta_reason', 'label_date']).reset_index())
    cols = ['date', 'symbol', 'p_up_raw', 'p_up_cal', 'pred_logret', 'model_disagreement',
            'model_version', 'train_end', 'label_end', 'meta_label_end', *rc, 'logit',
            'temperature', 'base_rate_asof', 'informative_asof', 'base_ok', 'meta_ok', 'mask_reason']
    table = pd.concat(frames, ignore_index=True)[cols].sort_values(['date', 'symbol']).reset_index(drop=True)
    report = {
        'table_version': TABLE_VERSION, 'horizon': horizon, 'models': list(models),
        'seed': seed, 'params_override': params_override,
        'design': {'first_anchor': first_anchor, 'retrain_every': every,
                   'min_train_rows': MIN_TRAIN_ROWS, 'val_frac': VAL_FRAC,
                   'meta_window': META_WINDOW, 'min_half': MIN_HALF, 'T_bounds': list(T_BOUNDS),
                   'pred_logret': 'esit agirlikli taban ortalamasi',
                   'temperature': 'capa basina havuzlu (tum semboller)',
                   'purge': f'label_date < capa (h={horizon_steps(horizon)})'},
        'anchors': [str(a.date()) for a in anchors],
        'base_log': base_log, 'meta_log': meta_log, 'seconds_per_symbol': secs,
        'wall_seconds': round(time.time() - t0, 1),
    }
    return table, report


def coverage_report(table: pd.DataFrame, window_start: str = '2013-06-01',
                    window_end: Optional[str] = None) -> dict:
    """P2: sembol x gun doluluk ve maske orani (pencere icinde)."""
    t = table[table['date'] >= pd.Timestamp(window_start)]
    if window_end is not None:
        t = t[t['date'] < pd.Timestamp(window_end)]
    per = {}
    for s, g in t.groupby('symbol'):
        n = len(g)
        per[s] = {'n_days': int(n),
                  'has_data': round(float((g['mask_reason'] != 'no_data').mean()), 4),
                  'base_ok': round(float(g['base_ok'].mean()), 4),
                  'meta_ok': round(float(g['meta_ok'].mean()), 4),
                  'informative_asof': round(float(g['informative_asof'].mean()), 4),
                  'first_meta_ok': (str(g.loc[g['meta_ok'], 'date'].min().date())
                                    if g['meta_ok'].any() else None)}
    return {'window': [window_start, window_end], 'n_calendar_days': int(t['date'].nunique()),
            'meta_ok_rate': round(float(t['meta_ok'].mean()), 4),
            'base_ok_rate': round(float(t['base_ok'].mean()), 4),
            'mask_reasons': {k: int(v) for k, v in t['mask_reason'].value_counts().items()},
            'per_symbol': per}


def realized_labels(panel: pd.DataFrame, horizon: str = 'daily') -> pd.DataFrame:
    """(date, symbol) -> y_up (sembolun kendi h. sonraki seansi) + tie (esit kapanis). Yalniz degerlendirme icin.

    y_up = 1[kapanis_{t+h} > kapanis_t]: esit kapanis 0 sayilir (DOWN tarafina; hakem-finans I3 #2 ->
    `tie_rate` raporlanir).
    """
    h = horizon_steps(horizon)
    out = []
    for s in sorted(set(panel.index.get_level_values('symbol'))):
        c = _symbol_ohlcv(panel, s)['close']
        fut = c.shift(-h)
        out.append(pd.DataFrame({'date': c.index, 'symbol': s,
                                 'y_up': (fut > c).astype(float).where(fut.notna()).to_numpy(),
                                 'tie': (fut == c).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def _within_block_auc(p, y, groups) -> Optional[float]:
    """Blok (sembol x capa) icinde AUC'larin n agirlikli ortalamasi; tek sinifli blok atlanir.

    Capa basina p seviyesi (uyum yarisi taban orani, T) degisir; bloklar arasi havuzlanan AUC bu seviye
    farkini siralama sanabilir (hakem-finans I3 #2). Blok ici AUC yalniz ayni modelin siralamasini olcer.
    """
    from evaluation import calibration as cal
    num = den = 0.0
    for gid in np.unique(groups):
        m = groups == gid
        a = cal.auc(p[m], y[m])
        if a is not None:
            num += a * m.sum()
            den += m.sum()
    return float(num / den) if den else None


def _date_block_bootstrap(dates: np.ndarray, fn, n_boot: int = 500, block: int = 20, seed: int = 0):
    """Tarih bloklarini yeniden ornekleyen bootstrap: ayni gunun tum sembolleri birlikte (kesitsel
    bagimlilik korunur). fn(satir indeksleri) -> metrik. Doner: tekrar degerleri."""
    ud, inv = np.unique(dates, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(ud) + 1))
    rows_of = [order[bounds[i]:bounds[i + 1]] for i in range(len(ud))]
    n = len(ud)
    k = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=k)
        di = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        idx = np.concatenate([rows_of[i] for i in di])
        v = fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def _cluster_bootstrap(clusters: np.ndarray, fn, n_boot: int = 500, seed: int = 0):
    """Kume (capa = model_version) bootstrap: capalar yerine koyarak orneklenir, bir capanin tum
    sembol x gun satirlari birlikte gelir. Capa seviyesi 63 seans sabit oldugundan bagimsiz birim
    capadir; 20 seanslik tarih blogu ayni capanin parcalarini bagimsiz sayar (hakem-finans I3-2 #1)."""
    uc, inv = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(uc))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), size=len(uc))
        v = fn(np.concatenate([rows_of[i] for i in pick]))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def _p_one_two(vals: np.ndarray, null: float = 0.5):
    """(tek tarafli H0: metrik <= null, iki tarafli) yuzdelik bootstrap p-degerleri."""
    p_gt = float((1 + np.sum(vals <= null)) / (1 + len(vals)))
    p_lt = float((1 + np.sum(vals >= null)) / (1 + len(vals)))
    return p_gt, min(1.0, 2 * min(p_gt, p_lt))


def evaluate_gate(table: pd.DataFrame, labels: pd.DataFrame, window=DEV_WINDOW,
                  alpha: float = 0.05, n_boot: int = 1000, power: float = 0.8,
                  pooled_boot: int = 500, sensitivity: bool = False) -> dict:
    """Bilgi kapisi (sembol basina) + BH (30 sembol tek aile; hakem-finans I2 not 1) + guc (I3 #1-#2).

    BSS referansi satir basina `base_rate_asof` (capa anindaki uyum yarisi taban orani; test disi).
    p-degeri: tek tarafli H0 AUC <= 0,5, yuzdelik blok bootstrap; iki tarafli p ayrica.
    MDE (en kucuk tespit edilebilir AUC): 0,5 + (z_{1-a} + z_guc) * SE_boot; a = alpha (tek test) ve
    alpha/m (BH'de ilk reddin Bonferroni siniri). Havuzlu AUC icin tarih blogu bootstrap GA/p (ayni
    gunun sembolleri birlikte yeniden orneklenir). Blok ici AUC (sembol x capa) ayrica.
    """
    from scipy.stats import norm
    from evaluation import calibration as cal
    from evaluation.multiple_testing import benjamini_hochberg

    t = table[(table['date'] >= pd.Timestamp(window[0])) & (table['date'] < pd.Timestamp(window[1]))
              & table['meta_ok']].merge(labels, on=['date', 'symbol'], how='left')
    t = t[t['y_up'].notna()].reset_index(drop=True)
    n_sym = t['symbol'].nunique()
    z_pow = norm.ppf(power)
    z1, zm = norm.ppf(1 - alpha), norm.ppf(1 - alpha / max(n_sym, 1))
    per, pvals = {}, {}
    for s, g in t.groupby('symbol'):
        p, y, b = g['p_up_cal'].to_numpy(), g['y_up'].to_numpy(), g['base_rate_asof'].to_numpy()
        if len(g) < 2 * 20 or len(np.unique(y)) < 2:
            per[s] = {'n': int(len(g)), 'skipped': 'yetersiz satir'}
            continue
        vals = cal._block_bootstrap_vals(cal.auc, p, y, n_boot=n_boot)
        ci = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals is not None else None
        pv, pv2 = _p_one_two(vals) if vals is not None else (None, None)
        se = float(vals.std(ddof=1)) if vals is not None else None
        bss = cal.brier_skill(p, y, b)
        at_bound = bool(g['temperature'].ge(0.99 * T_BOUNDS[1]).any())
        per[s] = {'n': int(len(g)), 'auc': cal.auc(p, y), 'auc_ci': ci, 'p_auc_gt_half': pv,
                  'p_two_sided': pv2, 'auc_se_boot': se,
                  'mde_auc_single': None if se is None else float(0.5 + (z1 + z_pow) * se),
                  'mde_auc_family': None if se is None else float(0.5 + (zm + z_pow) * se),
                  'auc_within_block': _within_block_auc(p, y, g['model_version'].to_numpy()),
                  'tie_rate': float(g['tie'].mean()) if 'tie' in g else None,
                  'brier_skill': bss, 'ece': cal.ece(p, y),
                  'accuracy': float(np.mean((p >= 0.5) == (y == 1))),
                  'base_rate': float(y.mean()), 'T_at_bound_any': at_bound,
                  'informative': bool(not at_bound and np.isfinite(bss) and bss > 0
                                      and ci is not None and ci[0] > 0.5)}
        if pv is not None:
            pvals[s] = pv
    bh = benjamini_hochberg(pvals, alpha=alpha) if pvals else None
    pool_p, pool_y = t['p_up_cal'].to_numpy(), t['y_up'].to_numpy()
    grp = (t['symbol'] + '|' + t['model_version'].astype(str)).to_numpy()
    dates = t['date'].to_numpy()
    b_auc = _date_block_bootstrap(dates, lambda i: cal.auc(pool_p[i], pool_y[i]), n_boot=pooled_boot)
    b_wb = _date_block_bootstrap(dates, lambda i: _within_block_auc(pool_p[i], pool_y[i], grp[i]),
                                 n_boot=pooled_boot)

    def summ(point, vals):
        p1, p2 = _p_one_two(vals)
        return {'value': point, 'ci': [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                'se_boot': float(vals.std(ddof=1)), 'p_gt_half': p1, 'p_two_sided': p2,
                'n_boot': int(len(vals))}

    ses = [v['auc_se_boot'] for v in per.values() if v.get('auc_se_boot') is not None]
    sens = None
    if sensitivity:
        sens = _gate_sensitivity(t, per, grp, dates, pooled_boot, n_boot, z1, zm, z_pow, summ)
    return {'window': list(window), 'n_rows': int(len(t)), 'per_symbol': per, 'bh': bh,
            'sensitivity': sens,
            'n_informative': int(sum(v.get('informative', False) for v in per.values())),
            'n_bh_reject': int(sum(r['reject'] for r in bh['results'].values())) if bh else 0,
            'expected_false_pass_one_sided_2_5pct': round(0.025 * len(pvals), 2),
            'power': power,
            'mde_auc_family_median': float(np.median([0.5 + (zm + z_pow) * x for x in ses])) if ses else None,
            'mde_auc_single_median': float(np.median([0.5 + (z1 + z_pow) * x for x in ses])) if ses else None,
            'pooled': {'auc': summ(cal.auc(pool_p, pool_y), b_auc),
                       'auc_within_block': summ(_within_block_auc(pool_p, pool_y, grp), b_wb),
                       'ece': cal.ece(pool_p, pool_y),
                       'brier_skill': cal.brier_skill(pool_p, pool_y, t['base_rate_asof'].to_numpy()),
                       'tie_rate': float(t['tie'].mean()) if 'tie' in t else None},
            'note': ('havuzlu GA tarih blogu bootstrap (20 seans, ayni gunun sembolleri birlikte); '
                     'MDE = 0,5 + (z_{1-a} + z_guc) * SE_boot, a = alpha ya da alpha/m')}


def _gate_sensitivity(t, per, grp, dates, pooled_boot, n_boot, z1, zm, z_pow, summ) -> dict:
    """hakem-finans I3 ikinci okuma (#1-#5) duyarliliklari; ana kapi alanlarini degistirmez.

    - havuzlu ve blok ici AUC: tarih blogu 63 seans (capa araligi) ve capa kumesi bootstrap'i;
    - sembol basina SE ve MDE: 63 seanslik hareketli blokla; MDE ve blok ici AUC dagilimi (min/medyan/maks);
    - esit kapanislar (tie) disarida birakilarak havuzlu, blok ici ve sembol basina blok ici AUC;
    - Bonferroni (m = sembol sayisi) iki tarafli p.
    """
    from evaluation import calibration as cal
    pool_p, pool_y = t['p_up_cal'].to_numpy(), t['y_up'].to_numpy()
    clusters = t['model_version'].astype(str).to_numpy()
    f_auc = lambda i: cal.auc(pool_p[i], pool_y[i])
    f_wb = lambda i: _within_block_auc(pool_p[i], pool_y[i], grp[i])
    pt_auc, pt_wb = cal.auc(pool_p, pool_y), _within_block_auc(pool_p, pool_y, grp)
    out = {'pooled_block63': {'auc': summ(pt_auc, _date_block_bootstrap(dates, f_auc, pooled_boot, block=63)),
                              'auc_within_block': summ(pt_wb, _date_block_bootstrap(dates, f_wb, pooled_boot,
                                                                                    block=63))},
           'pooled_anchor_cluster': {'auc': summ(pt_auc, _cluster_bootstrap(clusters, f_auc, pooled_boot)),
                                     'auc_within_block': summ(pt_wb, _cluster_bootstrap(clusters, f_wb,
                                                                                        pooled_boot)),
                                     'n_clusters': int(len(np.unique(clusters)))}}

    def dist(xs):
        xs = [x for x in xs if x is not None]
        return {'min': float(min(xs)), 'median': float(np.median(xs)), 'max': float(max(xs)),
                'n': len(xs)} if xs else None

    ok = {s: v for s, v in per.items() if 'auc' in v}
    out['mde_family_block20'] = dist([v['mde_auc_family'] for v in ok.values()])
    out['mde_single_block20'] = dist([v['mde_auc_single'] for v in ok.values()])
    out['auc_within_block_per_symbol'] = dist([v['auc_within_block'] for v in ok.values()])
    se63, bonf = {}, {}
    m = len(ok)
    for s, g in t.groupby('symbol'):
        if s not in ok:
            continue
        vals = cal._block_bootstrap_vals(cal.auc, g['p_up_cal'].to_numpy(), g['y_up'].to_numpy(),
                                         block=63, n_boot=n_boot)
        se63[s] = None if vals is None else float(vals.std(ddof=1))
        bonf[s] = min(1.0, m * ok[s]['p_two_sided']) if ok[s].get('p_two_sided') is not None else None
    out['mde_family_block63'] = dist([0.5 + (zm + z_pow) * x for x in se63.values() if x is not None])
    out['mde_single_block63'] = dist([0.5 + (z1 + z_pow) * x for x in se63.values() if x is not None])
    out['p_two_sided_bonferroni'] = bonf
    out['n_two_sided_p_lt_005'] = int(sum(v['p_two_sided'] < 0.05 for v in ok.values()))
    if 'tie' in t:
        nt = ~t['tie'].astype(bool).to_numpy()
        tp, ty, tg = pool_p[nt], pool_y[nt], grp[nt]
        per_nt = {}
        for s in ok:
            ms = nt & (t['symbol'].to_numpy() == s)
            vals = cal._block_bootstrap_vals(cal.auc, pool_p[ms], pool_y[ms], n_boot=n_boot)
            p1, p2 = _p_one_two(vals) if vals is not None else (None, None)
            per_nt[s] = {'n': int(ms.sum()), 'auc': cal.auc(pool_p[ms], pool_y[ms]),
                         'auc_within_block': _within_block_auc(pool_p[ms], pool_y[ms], grp[ms]),
                         'p_two_sided': p2}
        td = dates[nt]
        out['ties_excluded'] = {
            'n_rows': int(nt.sum()),
            'pooled_auc': summ(cal.auc(tp, ty), _date_block_bootstrap(td, lambda i: cal.auc(tp[i], ty[i]),
                                                                      pooled_boot)),
            'pooled_auc_within_block': summ(_within_block_auc(tp, ty, tg),
                                            _date_block_bootstrap(td, lambda i: _within_block_auc(
                                                tp[i], ty[i], tg[i]), pooled_boot)),
            'per_symbol': per_nt}
    return out


# --- P4: P tablosu -> eski TradingEnv tahmin arayuzu (Milestone 0' DT/DTF kolu; kullanici karari 2026-11-15) ---
ENV_PRED_NEUTRAL = {'predicted_return': 0.0, 'predicted_direction': 0.5, 'confidence': 0.5,
                    'ensemble_agreement': 0.5}


def load_table(run_id: str, directory: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """`results_dir()/predictions_pit/<run_id>.parquet` + kimlik (yol, SHA-256, yan JSON'daki surum).
    run_id yol ayirici tasiyamaz (sema deseni de reddeder)."""
    if not run_id or os.path.basename(run_id) != run_id:
        raise ValueError(f"gecersiz P tablosu kimligi: {run_id!r}")
    if directory is None:
        from app.auth import workspace as ws
        directory = os.path.join(ws.results_dir(), 'predictions_pit')
    path = os.path.join(directory, f'{run_id}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"P tablosu yok: {path} (python scripts/build_pit_table.py)")
    with open(path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    side = {}
    sp = os.path.join(directory, f'{run_id}.json')
    if os.path.exists(sp):
        with open(sp, encoding='utf-8') as fh:
            side = json.load(fh)
    return pd.read_parquet(path), {'run_id': run_id, 'sha256': sha,
                                   'table_version': side.get('table_version', TABLE_VERSION),
                                   'seed': side.get('seed'), 'models': side.get('models')}


def env_prediction_features(table: pd.DataFrame, symbols: Sequence[str], dates: Sequence,
                            models: Sequence[str] = V1_MODELS) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], dict]:
    """P tablosu -> `TradingEnv(prediction_features=...)` sozlugu {gun: {sembol: 4 alan}}.

    Esleme (durum uzayi yapisi degismez; env donusumu `trading_env.py` tahmin blogunda):
      predicted_return    = pred_logret (uc taban tahmininin ortalamasi, log getiri)
      predicted_direction = 1[p_up_cal >= 0,5]
      confidence          = max(p_up_cal, 1 - p_up_cal)
      ensemble_agreement  = taban modellerinden isareti (ret > 0) bu yonle ayni olanlarin payi (olceksiz)
    `meta_ok` olmayan ya da tabloda bulunmayan HER (gun, sembol) hucresi ACIKCA notr doldurulur
    (env'de 4 alan da 0'a duser). Gerekce: env'in kendi varsayilani eksik hucrede yonu 0 -> -1 (DOWN)
    yapar; adaptor env'in varsayilanina hic dusmez. Satir t yalniz t'deki P satirindan (B0, P1).
    Doner: (sozluk, rapor: hucre sayilari, notr nedenleri, olculen dinamik aralik).
    """
    rc = _ret_cols(models)
    keys = [str(pd.Timestamp(d))[:10] for d in dates]
    t = table.copy()
    t['key'] = t['date'].astype(str).str[:10]
    t = t[t['key'].isin(set(keys)) & t['symbol'].isin(set(symbols))]
    ok = t[t['meta_ok']]
    up = (ok['p_up_cal'] >= 0.5).to_numpy()
    agree = np.mean((ok[rc].to_numpy() > 0) == up[:, None], axis=1)
    vals = pd.DataFrame({'key': ok['key'].to_numpy(), 'symbol': ok['symbol'].to_numpy(),
                         'predicted_return': ok['pred_logret'].to_numpy(dtype=float),
                         'predicted_direction': up.astype(float),
                         'confidence': np.maximum(ok['p_up_cal'], 1 - ok['p_up_cal']).to_numpy(dtype=float),
                         'ensemble_agreement': agree.astype(float)})
    lookup = {(k, s): r for k, s, *r in vals.itertuples(index=False, name=None)}
    reason = {(k, s): m for k, s, m in t[~t['meta_ok']][['key', 'symbol', 'mask_reason']].itertuples(
        index=False, name=None)}
    fields = ('predicted_return', 'predicted_direction', 'confidence', 'ensemble_agreement')
    out, counts = {}, {'ok': 0, 'not_in_table': 0}
    for k in keys:
        day = {}
        for s in symbols:
            r = lookup.get((k, s))
            if r is None:
                day[s] = dict(ENV_PRED_NEUTRAL)
                why = reason.get((k, s), 'not_in_table')
                counts[why] = counts.get(why, 0) + 1
            else:
                day[s] = dict(zip(fields, map(float, r)))
                counts['ok'] += 1
        out[k] = day
    env_ret = np.tanh(vals['predicted_return'].to_numpy() * 10)   # env'in uyguladigi donusum
    env_conf = vals['confidence'].to_numpy() * 2 - 1
    q = lambda x: {'p01': float(np.quantile(x, 0.01)), 'median': float(np.median(x)),
                   'p99': float(np.quantile(x, 0.99))} if len(x) else None
    report = {'n_cells': len(keys) * len(symbols), 'counts': counts,
              'ok_rate': counts['ok'] / max(len(keys) * len(symbols), 1),
              'env_obs_range': {'predicted_return_tanh10': q(env_ret), 'confidence_2c_minus_1': q(env_conf),
                                'direction_up_rate': float(up.mean()) if len(up) else None,
                                'agreement_levels': sorted(set(np.round(agree, 4).tolist()))},
              'mapping': 'P4 v1 (plan §9.32)'}
    return out, report


def write_table(table: pd.DataFrame, report: dict, run_id: Optional[str] = None,
                directory: Optional[str] = None, snapshot: Optional[dict] = None) -> Dict[str, str]:
    """`results_dir()/predictions_pit/<run_id>.parquet` + yan JSON + egitim manifesti."""
    from prediction.manifest import TrainingManifest
    if directory is None:
        from app.auth import workspace as ws
        directory = os.path.join(ws.results_dir(), 'predictions_pit')
    os.makedirs(directory, exist_ok=True)
    m = TrainingManifest(run_kind='pit_table', symbols=sorted(table['symbol'].unique()),
                         horizon=report.get('horizon', 'daily'),
                         expected_models=set(report.get('models', V1_MODELS)))
    run_id = run_id or m.run_id
    m.run_id = run_id
    path = os.path.join(directory, f'{run_id}.parquet')
    table.to_parquet(path, index=False)
    with open(path, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    side = {**report, 'run_id': run_id, 'table_path': path, 'table_sha256': sha,
            'n_rows': int(len(table)), **({'data_snapshot': snapshot} if snapshot else {})}
    with open(os.path.join(directory, f'{run_id}.json'), 'w', encoding='utf-8') as fh:
        json.dump(side, fh, indent=1, default=str)
    m.set_seed(report.get('seed'))
    if snapshot:
        m.add_data_snapshot(snapshot)
    for s in sorted(table['symbol'].unique()):
        ok = any(e['status'] == 'ok' for e in report['base_log'].get(s, []))
        m.record_symbol(s, {'models_trained': list(report.get('models', V1_MODELS)) if ok else [],
                            'n_models': len(report.get('models', V1_MODELS)) if ok else 0},
                        seconds=report['seconds_per_symbol'].get(s))
    m.extra = {'table_path': path, 'table_sha256': sha, 'table_version': TABLE_VERSION,
               'coverage': report.get('coverage')}
    mpath = m.finalize()
    return {'table': path, 'sidecar': os.path.join(directory, f'{run_id}.json'), 'manifest': mpath}
