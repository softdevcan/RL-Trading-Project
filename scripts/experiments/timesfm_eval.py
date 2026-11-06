"""
TimesFM-3 degerlendirme deneyi — tahmin kalitesi VE islem degeri.

Soru: sifir-atis TimesFM-3 (a) tahmini iyilestiriyor mu, (b) bu iyilesme
risk-ayarli getiriye ve al-sat dengesine yansiyor mu?

Protokol uretim ensemble'inin (prediction/models/ensemble.py) aynisi:
3'lu kronolojik bolme %60 base egitim / %20 meta (OOF) / %20 nihai test;
base bolumun %15'i validation. Ayni base modeller BIR kez egitilir ve
TimesFM'li / TimesFM'siz ensemble'lar ayni tahminlerden kurulur — fark
yalnizca TimesFM'den gelir. TATS ve %80 yeniden egitim yok (test temiz).

Uc katman:
  1. Nokta tahmini: yon isabeti, IC, R2_oos (sifira karsi), DM testi.
     Olasilik: Brier. Oynaklik: QLIKE (son 20 gun std / GARCH(1,1) /
     TimesFM dilim bandi). Dilimler: pinball + %80 bant kapsamasi
     (rakip: son 250 gunun ampirik dilimleri).
  2. Islem: test doneminde gunluk portfoyler — 1/N, sinyal (tahmin > 0),
     oynaklik hedefli 1/N ve sinyal + oynaklik hedefi. Maliyet 20 bps
     (donus basina), nakit politika faiziyle (TCMB AOFM) getiri kazanir.
     Sharpe/Sortino risksiz faiz USTU.
  3. RL: bu betigin kapsami DISI — 2. katman olumluysa ayri is.

Makro (faiz, enflasyon, USD/TRY): `data/macro/macro_data.csv` su an
sabit yedek degerler tasiyor (politika faizi 50,0, TUFE/UFE 0,0 — EVDS seri
kodu ve tarih ayristirma hatalari). Bu betik EVDS'ten DOGRU serileri kendisi
ceker ve paylasilan CSV'ye YAZMAZ:
  - faiz: TP.APIFON4 (TCMB agirlikli ortalama fonlama maliyeti, gunluk)
  - TUFE: TP.FE25.OKTG01 (2025=100), UFE: TP.TUFE1YI.T1 — yillik %, ay M'nin
    degeri M+1'in 5. gunune kadar BILINMEZ (aciklanma gecikmesi).
`--macro cached` ile uretimdeki (bozuk) makroyla da kosulabilir.

Kullanim:
    python scripts/experiments/timesfm_eval.py                   # PHASE1 5 sembol
    python scripts/experiments/timesfm_eval.py --skip-dl         # BiLSTM/TFT'siz, hizli
    python scripts/experiments/timesfm_eval.py --horizon weekly  # 5 gunluk hedef
    python scripts/experiments/timesfm_eval.py --symbols all --device cuda --start 2010-01-01

Cikti: <results>/experiments/timesfm_eval_<zaman>.{json,md} ve _predictions.csv
Lisans: TimesFM 3.0 agirliklari ticari olmayan kullanimla sinirli.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.auth import workspace as ws  # noqa: E402
from app.services.prediction_service import (  # noqa: E402
    _fetch_symbol_data, _fetch_fundamental_data, _fetch_cross_asset_data,
)
from data.bist30_symbols import BIST30_SYMBOLS, PHASE1_SYMBOLS  # noqa: E402
from prediction.feature_engineer import PredictionFeatureEngineer  # noqa: E402
from prediction.models.ensemble import StackingEnsemble  # noqa: E402
from prediction.models.timesfm_model import TimesFMModel, QUANTILE_LEVELS  # noqa: E402
from prediction.seeding import seed_everything  # noqa: E402

logger = logging.getLogger('timesfm_eval')

BASE_MODELS = ['xgboost', 'lightgbm', 'catboost', 'bilstm', 'tft']
DL_MODELS = {'bilstm', 'tft'}
TFM_VARIANTS = {
    'tfm_uni': {'past_covariates': [], 'calendar_covariates': False},
    'tfm_cov': {'past_covariates': ['macro_policy_rate', 'macro_cpi', 'macro_usdtry_return'],
                'calendar_covariates': True},
}
SEQ_PAD = 64          # DL lookback + TimesFM min_context icin onceki satirlar
TRADING_DAYS = 252
Q_SIGMAS = 2.0 * stats.norm.ppf(0.9)


# ----------------------------------------------------------------------
# Makro — EVDS'ten dogru seriler (paylasilan CSV'ye yazilmaz)
# ----------------------------------------------------------------------

def _evds_series(evds, code: str, start: str, end: str, chunk_years: int = 3) -> pd.Series:
    """EVDS istek basina ~1000 satir dondurur — gunluk seriyi yil dilimleriyle cek."""
    col = code.replace('.', '_')
    parts = []
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        stop = min(s + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), e)
        df = evds.get_data([code], startdate=s.strftime('%d-%m-%Y'),
                           enddate=stop.strftime('%d-%m-%Y'))
        if df is not None and not df.empty and col in df.columns:
            parts.append(df[['Tarih', col]])
        s = stop + pd.Timedelta(days=1)
    if not parts:
        raise ValueError(f"EVDS {code} bos dondu")
    df = pd.concat(parts).drop_duplicates('Tarih')
    # Gunluk 'dd-mm-YYYY', aylik 'YYYY-M' doner
    idx = pd.to_datetime(df['Tarih'], format='%d-%m-%Y', errors='coerce')
    monthly = idx.isna()
    idx[monthly] = pd.to_datetime(df.loc[monthly, 'Tarih'], format='%Y-%m', errors='coerce')
    return pd.Series(df[col].astype(float).values, index=idx).dropna().sort_index()


def _released_yoy(index: pd.Series, release_day: int = 5) -> pd.Series:
    """Aylik endeks -> yillik %, ay M'nin degeri M+1'in `release_day`'inde bilinir."""
    yoy = index.pct_change(12) * 100
    yoy.index = yoy.index + pd.offsets.MonthBegin(1) + pd.Timedelta(days=release_day - 1)
    return yoy.dropna()


def load_macro(start: str, source: str) -> pd.DataFrame:
    from data.macro_fetcher import MacroDataFetcher

    fetcher = MacroDataFetcher(start_date=start)
    if source == 'cached':
        return fetcher.load_data()

    # yfinance kolonlari taze (CSV Nisan 2026'da bitiyor); EVDS kolonlari asagida ezilir
    macro = fetcher.fetch_macro_data(save=False)
    end = datetime.now().strftime('%Y-%m-%d')
    evds = fetcher.evds
    policy = _evds_series(evds, 'TP.APIFON4', start, end)
    cpi = _released_yoy(_evds_series(evds, 'TP.FE25.OKTG01', '2004-01-01', end, chunk_years=30))
    ppi = _released_yoy(_evds_series(evds, 'TP.TUFE1YI.T1', '2004-01-01', end, chunk_years=30))

    daily = macro.index
    for name, s in (('policy_rate', policy), ('cpi_inflation', cpi), ('ppi_inflation', ppi)):
        # Yalnizca ileri doldurma: bfill gecmise gelecek degeri tasirdi
        macro[name] = s.reindex(daily.union(s.index)).ffill().reindex(daily)
    logger.info(
        f"Makro (EVDS): faiz {policy.index.min().date()}..{policy.index.max().date()} "
        f"son={policy.iloc[-1]:.2f} | TUFE son={cpi.iloc[-1]:.1f}% | UFE son={ppi.iloc[-1]:.1f}%"
    )
    return macro


# ----------------------------------------------------------------------
# Oynaklik rakipleri
# ----------------------------------------------------------------------

def garch11_sigma(v: np.ndarray, fit_end: int, H: int = 1) -> np.ndarray:
    """GARCH(1,1) — [0, fit_end) uzerinde MLE, tum seri uzerinde filtre.

    v[k] = r_{k-1} (log_return_1d); hedef r_{t+1} + .. + r_{t+H}, yani son
    gozlemden 2..H+1 adim ileri. k adim: s2(k) = u + p^(k-1) (s2(1) - u),
    p = alpha + beta, u = omega / (1 - p); getiriler iliskisiz varsayilir.
    """
    fit = v[:fit_end]
    fit = fit[np.abs(fit) > 0]
    mu = fit.mean()
    var0 = fit.var()

    def unpack(th):
        omega = np.exp(th[0])
        a, b = np.exp(th[1]), np.exp(th[2])
        s = 1 + a + b
        return omega, a / s, b / s  # alpha + beta < 1

    def nll(th):
        omega, a, b = unpack(th)
        e = fit - mu
        s2 = np.empty_like(e)
        s2[0] = var0
        for t in range(1, len(e)):
            s2[t] = omega + a * e[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(s2) + e ** 2 / s2)

    th0 = np.array([np.log(var0 * 0.05), np.log(0.1), np.log(0.85)])
    res = optimize.minimize(nll, th0, method='Nelder-Mead', options={'maxiter': 2000})
    omega, a, b = unpack(res.x)

    e = v - mu
    s2_next = np.empty_like(v)  # s2_next[t] = Var(v_{t+1} | v_0..v_t), v_{t+1} = r_t
    s2 = var0                   # Var(v_t | v_0..v_{t-1})
    for t in range(len(v)):
        s2_next[t] = omega + a * e[t] ** 2 + b * s2
        s2 = s2_next[t]
    p = a + b
    u = omega / (1 - p)
    s2_h = sum(u + p ** (k - 1) * (s2_next - u) for k in range(2, H + 2))
    logger.info(f"    GARCH: omega={omega:.2e} alpha={a:.3f} beta={b:.3f}")
    return np.sqrt(s2_h)


# ----------------------------------------------------------------------
# Sembol bazli tahminler
# ----------------------------------------------------------------------

def _segment(model, X: np.ndarray, a: int, b: int) -> np.ndarray:
    """[a, b) satirlari icin tahmin; sekans modelleri onceki SEQ_PAD satiri baglam alir."""
    p = np.asarray(model._predict_raw(X[max(0, a - SEQ_PAD):b]), dtype=float).ravel()
    n = b - a
    if len(p) >= n:
        return p[-n:]
    return np.concatenate([np.full(n - len(p), np.nan), p])


def run_symbol(symbol: str, args, macro, fund, cross) -> Optional[Dict]:
    t0 = time.perf_counter()
    H = args.H
    df = _fetch_symbol_data(symbol, args.start)
    fe = PredictionFeatureEngineer(horizon=args.horizon)
    feat = fe.build_features(df, symbol, macro_df=macro, fundamental_df=fund,
                             cross_asset_df=cross, target_type='log_return')
    cols = fe.get_feature_columns(feat)
    X = np.nan_to_num(feat[cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(feat['target_price'].values.astype(np.float64))
    # Islem katmani icin ertesi gunun getirisi (hedef haftalik olsa da para gunluk isler)
    close = df['close']
    r1 = np.log(close.shift(-1) / close).reindex(feat.index).values
    n = len(X)
    base_end, meta_end = int(n * 0.60), int(n * 0.80)
    val_split = int(base_end * 0.85)
    # Purge: haftalik hedef sonraki H satira tasar — egitim/val/meta bolumlerinin
    # son H satiri bir sonraki bolumun etiket donemine degmesin. Gunlukte 0
    # (uretim ensemble'iyla ayni protokol).
    P = H if H > 1 else 0
    logger.info(f"[{symbol}] {n} satir, {len(cols)} ozellik | ufuk {H}g, purge {P} | test "
                f"{feat.index[meta_end].date()}..{feat.index[-1].date()} ({n - meta_end} gun)")

    for c in ('macro_policy_rate', 'macro_cpi'):
        if c in cols:
            u = np.unique(X[:, cols.index(c)]).size
            if u <= 2:
                logger.warning(f"  [{symbol}] {c} sabit ({u} farkli deger) — makro bozuk")

    ens_helper = StackingEnsemble(horizon=args.horizon, meta_learner_type=args.meta)
    meta_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    failed: Dict[str, str] = {}

    for mt in args.models:
        try:
            ts = time.perf_counter()
            seed_everything()
            model = ens_helper._create_model(mt)
            model.train(X[:val_split - P], y[:val_split - P],
                        X[val_split:base_end - P], y[val_split:base_end - P],
                        feature_cols=cols)
            meta_preds[mt] = _segment(model, X, base_end, meta_end)
            test_preds[mt] = _segment(model, X, meta_end, n)
            logger.info(f"  [{mt}] {time.perf_counter() - ts:.0f}s")
        except Exception as exc:  # deney: bir model dusse de digerleri raporlanir
            failed[mt] = str(exc)
            logger.error(f"  [{mt}] HATA: {exc}")

    test_q: Dict[str, np.ndarray] = {}
    for name, over in TFM_VARIANTS.items():
        ts = time.perf_counter()
        m = TimesFMModel(horizon=args.horizon, params={
            **TimesFMModel().get_default_params(), 'device': args.device,
            'context_len': args.context_len, **over})
        m.feature_cols = cols
        m._build_model(m.params)
        m._fit(None, None, None, None)
        # Tum gecmis baglam olarak verilir; satir i yalnizca [.., i]'yi gorur.
        q_meta = m._forecast_rows(X[:meta_end], list(range(base_end, meta_end)))
        q_test = m._forecast_rows(X, list(range(meta_end, n)))
        meta_preds[name], test_preds[name] = q_meta[:, 4], q_test[:, 4]
        test_q[name] = q_test
        logger.info(f"  [{name}] {time.perf_counter() - ts:.0f}s")

    # Ensemble'lar: ayni base tahminlerinden, meta-learner meta bolumunde.
    # 'ens*' = --meta (uretim varsayilani xgboost); 'ensR*' = Ridge karsilastirmasi.
    y_meta = y[base_end:meta_end]
    meta_fit = np.arange(len(y_meta)) < len(y_meta) - P  # purge: test etiketine degen satirlar
    base = [m for m in args.models if m in meta_preds]
    helpers = {'ens': ens_helper,
               'ensR': StackingEnsemble(horizon=args.horizon, meta_learner_type='ridge')}
    for prefix, helper in helpers.items():
        if prefix == 'ensR' and args.meta == 'ridge':
            continue
        ensembles = {prefix: base, f'{prefix}_tfm_uni': base + ['tfm_uni'],
                     f'{prefix}_tfm_cov': base + ['tfm_cov']}
        for ename, members in ensembles.items():
            if len(members) < 2:
                continue
            keys = sorted(members)
            Xm = np.column_stack([meta_preds[k] for k in keys])
            Xt = np.column_stack([test_preds[k] for k in keys])
            ok = ~np.isnan(Xm).any(axis=1) & meta_fit
            learner = helper._build_meta_learner()
            learner.fit(Xm[ok], y_meta[ok])
            pred = np.full(len(Xt), np.nan)
            okt = ~np.isnan(Xt).any(axis=1)
            pred[okt] = learner.predict(Xt[okt])
            test_preds[ename] = pred

    v = X[:, cols.index('log_return_1d')].astype(np.float64)
    v_s = pd.Series(v)
    # Tarihsel ortalama: TL'de haftalik getirinin beklenen degeri sifirdan belirgin buyuk
    test_preds['hist250'] = (v_s.rolling(250, min_periods=250).mean() * H).values[meta_end:]
    sig = {
        'trail20': (v_s.rolling(20, min_periods=20).std() * np.sqrt(H)).values[meta_end:],
        'garch': garch11_sigma(v, base_end, H)[meta_end:],
    }
    for name, q in test_q.items():
        sig[name] = np.maximum((q[:, 8] - q[:, 0]) / Q_SIGMAS, 1e-6)
    v_h = v_s.rolling(H, min_periods=H).sum()  # H gunluk getiriler (ortusen)
    emp_q = np.column_stack([v_h.rolling(250, min_periods=250).quantile(lv).values[meta_end:]
                             for lv in QUANTILE_LEVELS])

    out = pd.DataFrame({'date': feat.index[meta_end:], 'symbol': symbol, 'y': y[meta_end:],
                        'r1': r1[meta_end:]})
    for k, p in test_preds.items():
        out[f'p_{k}'] = p
    for k, s in sig.items():
        out[f's_{k}'] = s
    for k, q in test_q.items():
        for j, lv in enumerate(QUANTILE_LEVELS):
            out[f'q_{k}_{int(lv * 100)}'] = q[:, j]
        out[f'pu_{k}'] = TimesFMModel.prob_up_from_quantiles(q)
    for j, lv in enumerate(QUANTILE_LEVELS):
        out[f'q_emp250_{int(lv * 100)}'] = emp_q[:, j]
    rate = macro['policy_rate'].reindex(out['date'], method='ffill').values
    out['rf'] = (1 + np.nan_to_num(rate) / 100) ** (1 / TRADING_DAYS) - 1
    logger.info(f"[{symbol}] bitti {time.perf_counter() - t0:.0f}s")
    return {'frame': out, 'failed': failed, 'n_rows': n, 'n_features': len(cols)}


# ----------------------------------------------------------------------
# Katman 1 — tahmin metrikleri
# ----------------------------------------------------------------------

def dm_test(loss_a: pd.Series, loss_b: pd.Series, h: int = 1) -> Dict[str, float]:
    """Diebold-Mariano, HLN duzeltmeli. Kayip farki tarih basina sembol ortalamasi.

    h > 1 (ortusen hedefler): varyans Newey-West (Bartlett, gecikme h-1).
    Negatif istatistik = A daha iyi (daha dusuk kayip).
    """
    d = (loss_a - loss_b).groupby(level=0).mean().dropna()
    T = len(d)
    if T < 30 or d.std(ddof=1) == 0:
        return {'stat': float('nan'), 'p': float('nan'), 'T': T}
    dc = (d - d.mean()).values
    var = d.var(ddof=1)
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.dot(dc[k:], dc[:-k]) / T
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = d.mean() / np.sqrt(max(var, 1e-30) / T) * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {'stat': round(float(stat), 3), 'p': round(float(p), 4), 'T': T}


def point_metrics(df: pd.DataFrame, preds: List[str], h: int = 1) -> Dict[str, Dict]:
    res = {}
    y = df['y']
    idx = pd.MultiIndex.from_arrays([df['date'], df['symbol']])
    for k in preds:
        p = df[f'p_{k}'] if k != 'zero' else pd.Series(0.0, index=df.index)
        m = p.notna() & y.notna()
        yy, pp = y[m].values, p[m].values
        nz = (yy != 0) & (pp != 0)
        r = {
            'n': int(m.sum()),
            'dir_acc': round(float(np.mean(np.sign(yy[nz]) == np.sign(pp[nz])) * 100), 2) if nz.any() else None,
            'rmse': float(np.sqrt(np.mean((yy - pp) ** 2))),
            'r2_oos_vs_zero_pct': round(float((1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2)) * 100), 3),
        }
        if k != 'zero' and np.std(pp) > 0:
            r['pearson'] = round(float(np.corrcoef(pp, yy)[0, 1]), 4)
            # IC: sembol icinde Spearman, sembollerin ortalamasi
            ics = [stats.spearmanr(g[f'p_{k}'], g['y'], nan_policy='omit')[0]
                   for _, g in df[m].groupby('symbol') if g[f'p_{k}'].std() > 0]
            r['ic_spearman'] = round(float(np.nanmean(ics)), 4) if ics else None
        se = pd.Series(((y - p) ** 2).values, index=idx)
        r['dm_vs_zero'] = dm_test(se, pd.Series((y ** 2).values, index=idx), h)
        # TimesFM'li ensemble kendi ailesinin TimesFM'siz haliyle karsilastirilir
        ref = 'ensR' if k.startswith('ensR') else 'ens'
        if k not in ('ens', 'ensR', 'zero') and f'p_{ref}' in df:
            se_ref = pd.Series(((y - df[f'p_{ref}']) ** 2).values, index=idx)
            r['dm_vs_ens'] = dm_test(se, se_ref, h)
            r['dm_ref'] = ref
        res[k] = r
    return res


def prob_metrics(df: pd.DataFrame, names: List[str]) -> Dict[str, Dict]:
    res = {}
    up = (df['y'] > 0).astype(float)
    for k in names:
        p = df[f'pu_{k}']
        res[k] = {
            'brier': round(float(np.mean((p - up) ** 2)), 5),
            'brier_base_rate': round(float(np.mean((up.mean() - up) ** 2)), 5),
            'mean_prob_up': round(float(p.mean()), 4),
            'realized_up_rate': round(float(up.mean()), 4),
        }
    return res


def vol_metrics(df: pd.DataFrame, names: List[str]) -> Dict[str, Dict]:
    res = {}
    for k in names:
        s = df[f's_{k}']
        m = s.notna() & (s > 0)
        y2, s2 = df['y'][m].values ** 2, s[m].values ** 2
        res[k] = {
            'n': int(m.sum()),
            'qlike': round(float(np.mean(np.log(s2) + y2 / s2)), 4),  # dusuk = iyi
            'mse_var_e8': round(float(np.mean((y2 - s2) ** 2) * 1e8), 4),
            'corr_sigma_absy': round(float(np.corrcoef(np.sqrt(s2), np.sqrt(y2))[0, 1]), 4),
        }
    return res


def quantile_metrics(df: pd.DataFrame, names: List[str]) -> Dict[str, Dict]:
    res = {}
    for k in names:
        Q = df[[f'q_{k}_{int(lv * 100)}' for lv in QUANTILE_LEVELS]].values
        m = ~np.isnan(Q).any(axis=1)
        y = df['y'].values[m][:, None]
        Q = Q[m]
        diff = y - Q
        pin = np.mean(np.maximum(QUANTILE_LEVELS * diff, (QUANTILE_LEVELS - 1) * diff))
        res[k] = {
            'n': int(m.sum()),
            'pinball_e4': round(float(pin * 1e4), 4),
            'coverage_80': round(float(np.mean((y[:, 0] >= Q[:, 0]) & (y[:, 0] <= Q[:, 8])) * 100), 2),
        }
    return res


# ----------------------------------------------------------------------
# Katman 2 — islem degeri
# ----------------------------------------------------------------------

def backtest(R: pd.DataFrame, W: pd.DataFrame, rf: pd.Series, cost_bps: float,
             rebalance: Optional[np.ndarray] = None) -> Dict:
    """Gunluk: t kapanisinda W_t hedeflenir, R_{t+1} kazanilir. Nakit rf kazanir.

    `rebalance[t]` False ise islem yapilmaz, kaymis agirlik tasinir (aylik
    dengeleme, haftalik sinyalin 5 gun tutulmasi).
    """
    w_prev = np.zeros(R.shape[1])
    rets, turns, expo = [], [], []
    for t in range(len(R)):
        if rebalance is None or rebalance[t]:
            w = np.nan_to_num(W.iloc[t].values)
        else:
            w = w_prev
        turn = float(np.abs(w - w_prev).sum())
        r = np.nan_to_num(R.iloc[t].values)
        gross = float(w @ r + (1 - w.sum()) * rf.iloc[t])
        rets.append(gross - cost_bps / 1e4 * turn)
        turns.append(turn)
        expo.append(w.sum())
        w_prev = w * (1 + r) / (1 + gross) if (1 + gross) != 0 else w
    ret = pd.Series(rets, index=R.index)
    ex = ret - rf
    eq = (1 + ret).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    T = len(ret)
    cagr = float(eq.iloc[-1] ** (TRADING_DAYS / T) - 1)
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else 0.0
    dn = ex[ex < 0]
    sortino = float(ex.mean() / np.sqrt((dn ** 2).mean()) * np.sqrt(TRADING_DAYS)) if len(dn) else 0.0
    sr_d = sharpe / np.sqrt(TRADING_DAYS)
    return {
        'cagr_pct': round(cagr * 100, 2),
        'vol_pct': round(float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100), 2),
        'sharpe_ex_rf': round(sharpe, 3),
        'sharpe_se': round(float(np.sqrt((1 + 0.5 * sr_d ** 2) / T) * np.sqrt(TRADING_DAYS)), 3),
        'sortino_ex_rf': round(sortino, 3),
        'max_dd_pct': round(mdd * 100, 2),
        'calmar': round(cagr / abs(mdd), 3) if mdd < 0 else None,
        'turnover_daily': round(float(np.mean(turns)), 4),
        'avg_exposure': round(float(np.mean(expo)), 3),
        'days': T,
    }


def trading_layer(df: pd.DataFrame, preds: List[str], vols: List[str], args) -> Dict[str, Dict]:
    piv = lambda col: df.pivot(index='date', columns='symbol', values=col)  # noqa: E731
    # Para her gun isler: hedef haftalik olsa da getiri ertesi gunun getirisi
    R = np.exp(piv('r1')) - 1
    common = R.dropna().index  # tum sembollerin birlikte oldugu gunler
    R = R.loc[common]
    rf = df.groupby('date')['rf'].first().loc[common]
    N = R.shape[1]
    H = args.H
    ew = pd.DataFrame(1.0 / N, index=common, columns=R.columns)
    month = pd.Series(common.to_period('M'), index=common)
    first_of_month = (month != month.shift(1)).values
    # Sinyal ufku kadar tutulur: haftalik tahmin 5 gunde bir yenilenir
    every_h = np.arange(len(common)) % H == 0

    def vt(volname):
        s_daily = piv(f's_{volname}').loc[common] / np.sqrt(H)
        target_d = args.target_vol / 100 / np.sqrt(TRADING_DAYS)
        return (ew * np.minimum(1.0, target_d / s_daily)).fillna(0.0)

    res = {}
    res['ew_daily'] = backtest(R, ew, rf, args.cost_bps)
    res['ew_monthly'] = backtest(R, ew, rf, args.cost_bps, rebalance=first_of_month)

    for k in preds:
        sig = (piv(f'p_{k}').loc[common] > 0).astype(float)
        res[f'sig_{k}'] = backtest(R, ew * sig, rf, args.cost_bps, rebalance=every_h)
    for vname in vols:
        res[f'vt_{vname}'] = backtest(R, vt(vname), rf, args.cost_bps, rebalance=every_h)
    for k in [p for p in ('ens', 'ens_tfm_uni', 'ens_tfm_cov', 'ensR', 'ensR_tfm_uni',
                          'ensR_tfm_cov') if p in preds]:
        for vname in vols:
            sig = (piv(f'p_{k}').loc[common] > 0).astype(float)
            res[f'sigvt_{k}__{vname}'] = backtest(R, vt(vname) * sig, rf, args.cost_bps,
                                                  rebalance=every_h)
    res['_period'] = {'start': str(common.min().date()), 'end': str(common.max().date()),
                      'days': len(common), 'symbols': list(R.columns)}
    return res


# ----------------------------------------------------------------------
# Rapor
# ----------------------------------------------------------------------

def to_markdown(report: Dict) -> str:
    a = report['args']
    L = [f"# TimesFM-3 degerlendirmesi ({a['horizon']}) — {report['run_at']}", "",
         f"Hedef: log(P_t+{a['H']} / P_t) | purge {a['H'] if a['H'] > 1 else 0} satir | "
         f"sinyal {a['H']} gunde bir yenilenir", "",
         f"Semboller: {', '.join(report['symbols'])} | baslangic {report['args']['start']} | "
         f"makro: {report['args']['macro']} | meta: {report['args']['meta']} | "
         f"maliyet {report['args']['cost_bps']} bps", ""]
    if report['failed']:
        L += [f"**Dusen modeller:** {report['failed']}", ""]

    L += ["## 1. Nokta tahmini (test, havuzlanmis)", "",
          "| Model | n | Yon % | R2_oos % | Pearson | IC | DM vs sifir (p) | DM vs ens (p) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in report['point'].items():
        dz, de = r.get('dm_vs_zero', {}), r.get('dm_vs_ens', {})
        L.append(f"| {k} | {r['n']} | {r['dir_acc']} | {r['r2_oos_vs_zero_pct']} | "
                 f"{r.get('pearson', '')} | {r.get('ic_spearman', '')} | "
                 f"{dz.get('stat', '')} ({dz.get('p', '')}) | "
                 f"{de.get('stat', '') if de else ''} ({de.get('p', '') if de else ''}) |")
    L += ["", "DM: negatif istatistik = satirdaki model daha dusuk kare hata. 'DM vs ens': "
          "ensR* satirlari ensR'ye (Ridge), digerleri ens'e (--meta) karsi. Ufuk > 1 gunde "
          "varyans Newey-West (ortusen hedefler). hist250 = son 250 gunun ortalama getirisi.", "",
          "## Yon olasiligi (TimesFM dilimlerinden)", "",
          "| Model | Brier | Taban oran Brier | Ort. P(yukari) | Gercek yukari orani |", "|---|---|---|---|---|"]
    for k, r in report['prob'].items():
        L.append(f"| {k} | {r['brier']} | {r['brier_base_rate']} | {r['mean_prob_up']} | {r['realized_up_rate']} |")
    L += ["", "## Oynaklik tahmini", "", "| Kaynak | n | QLIKE (dusuk iyi) | MSE var x1e8 | corr(sigma,|y|) |",
          "|---|---|---|---|---|"]
    for k, r in report['vol'].items():
        L.append(f"| {k} | {r['n']} | {r['qlike']} | {r['mse_var_e8']} | {r['corr_sigma_absy']} |")
    L += ["", "## Dilimler", "", "| Kaynak | n | Pinball x1e4 | %80 bant kapsamasi |", "|---|---|---|---|"]
    for k, r in report['quantile'].items():
        L.append(f"| {k} | {r['n']} | {r['pinball_e4']} | {r['coverage_80']} |")

    tr = report['trading']
    per = tr.get('_period', {})
    L += ["", f"## 2. Islem degeri ({per.get('start')}..{per.get('end')}, {per.get('days')} gun)", "",
          "| Strateji | CAGR % | Vol % | Sharpe (rf ustu) | ±SE | Sortino | MDD % | Calmar | Donus/gun | Ort. pozisyon |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in tr.items():
        if k.startswith('_'):
            continue
        L.append(f"| {k} | {r['cagr_pct']} | {r['vol_pct']} | {r['sharpe_ex_rf']} | {r['sharpe_se']} | "
                 f"{r['sortino_ex_rf']} | {r['max_dd_pct']} | {r['calmar']} | {r['turnover_daily']} | "
                 f"{r['avg_exposure']} |")
    L += ["", "SE: Sharpe'in yaklasik standart hatasi (Lo 2002, iid). Iki strateji arasindaki fark "
          "bunun ~2 katindan kucukse ayirt edilemez."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--symbols', default='phase1', help="'phase1', 'all' ya da virgullu liste")
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--horizon', default='daily', choices=['daily', 'weekly'])
    ap.add_argument('--skip-dl', action='store_true', help='BiLSTM ve TFT olmadan')
    ap.add_argument('--exclude', default='', help="Haric tutulacak base modeller, ornek: 'tft'")
    ap.add_argument('--device', default=None, help="TimesFM icin 'cuda' / 'cpu' (varsayilan otomatik)")
    ap.add_argument('--context-len', type=int, default=512)
    ap.add_argument('--meta', default='xgboost', choices=['xgboost', 'ridge'])
    ap.add_argument('--macro', default='evds', choices=['evds', 'cached'])
    ap.add_argument('--cost-bps', type=float, default=20.0)
    ap.add_argument('--target-vol', type=float, default=25.0, help='Sembol basina yillik hedef oynaklik %%')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    for noisy in ('prediction', 'data', 'app', 'httpx', 'httpx2', 'huggingface_hub'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    if args.symbols == 'phase1':
        symbols = PHASE1_SYMBOLS
    elif args.symbols == 'all':
        symbols = BIST30_SYMBOLS
    else:
        symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    excluded = {m.strip() for m in args.exclude.split(',') if m.strip()}
    args.models = [m for m in BASE_MODELS
                   if not (args.skip_dl and m in DL_MODELS) and m not in excluded]
    args.H = 1 if args.horizon == 'daily' else 5

    macro = load_macro(args.start, args.macro)
    fund = _fetch_fundamental_data(symbols)
    cross = _fetch_cross_asset_data(args.start)

    frames, failed, info = [], {}, {}
    for sym in symbols:
        try:
            r = run_symbol(sym, args, macro, fund, cross)
        except Exception as exc:
            logger.exception(f"[{sym}] atlandi: {exc}")
            failed[sym] = {'_symbol': str(exc)}
            continue
        frames.append(r['frame'])
        info[sym] = {'n_rows': r['n_rows'], 'n_features': r['n_features']}
        if r['failed']:
            failed[sym] = r['failed']
    if not frames:
        raise SystemExit("Hicbir sembol tamamlanamadi")
    df = pd.concat(frames, ignore_index=True)

    preds = [c[2:] for c in df.columns if c.startswith('p_')] + ['zero']
    tfm = list(TFM_VARIANTS)
    vols = [c[2:] for c in df.columns if c.startswith('s_')]
    report = {
        'run_at': datetime.now().isoformat(timespec='seconds'),
        'symbols': [s for s in symbols if s in info],
        'args': {k: v for k, v in vars(args).items()},
        'symbol_info': info,
        'failed': failed,
        'point': point_metrics(df, preds, args.H),
        'prob': prob_metrics(df, tfm),
        'vol': vol_metrics(df, vols),
        'quantile': quantile_metrics(df, tfm + ['emp250']),
        'trading': trading_layer(df, [p for p in preds if p != 'zero'], vols, args),
    }

    out_dir = args.out or os.path.join(ws.results_dir(), 'experiments')
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"timesfm_eval_{args.horizon}_{datetime.now():%Y%m%d_%H%M%S}")
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    md = to_markdown(report)
    with open(stem + '.md', 'w', encoding='utf-8') as f:
        f.write(md)
    df.to_csv(stem + '_predictions.csv', index=False)
    print(md)
    print(f"Yazildi: {stem}.{{json,md}} + _predictions.csv")


if __name__ == '__main__':
    main()
