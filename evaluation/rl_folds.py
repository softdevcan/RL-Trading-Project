"""Faz 9 / R1 — RL kat protokolu: egitim rotasi ve HPO'nun ORTAK yapi taslari.

Neden (Faz 9 §9.5 Y21, §9.6 #8; Faz 10 K20): HPO baska bir veride (ham OHLCV,
5 sembol, kendi pencereleri) ve baska bir hedefle (ortamin rf=0 Sharpe'i) ayar
ariyordu; rota ise dogrulama bolumunu hic kullanmiyor, testi son adimdaki
modelle okuyordu. Burada ikisinin de kullandigi tek yol var:

  load_rl_panel      panel: gostergeli CSV -> with_adx -> veri kapisi (+ nedensel baglam)
  build_fold         kat: egitim / dogrulama / test bloklari, egitim evrenine hizali
  make_fold_env      ortam: rol basina dogru olcek ve dogru nedensel baglam
  excess_sharpe      secim metrigi: risksiz faiz ustu getirinin yillik Sharpe'i
  run_episode        tek belirlenimci gecis -> gunluk getiri
  ValidationCheckpoints + select_checkpoint
                     kontrol noktasi kurali: sabit izgara, dogrulamada en iyi

Kontrol noktasi kurali (tasarim varsayimi, Faz 9 §9.11): egitim butcesi boyunca
esit aralikli `n_checkpoints` nokta (sonuncusu egitimin son modeli); her biri
DOGRULAMA blogunda belirlenimci tek gecisle olculur; en yuksek metrikli olan
secilir (esitlikte en erken). Erken durdurma yok (kollar esit ortam adimi alir),
yeniden uyum yok (secilen model sinanan modeldir). Test bu modulde okunmaz.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from evaluation.splits import SplitSpec, fold_frames

logger = logging.getLogger(__name__)

RL_PANEL_FILE = 'stock_data_with_indicators.csv'
RL_ADX_MODE = 'wilder'
# Kontrol noktasi sayisi: tasarim varsayimi. Bir yillik dogrulamada yillik
# Sharpe'in standart hatasi kabaca 1 (iid yaklasimi) — izgara bilerek kucuk.
N_CHECKPOINTS = 10
SELECTION_METRIC = 'val_excess_sharpe'
SELECTION_METRIC_RF0 = 'val_sharpe_rf0'   # risksiz faiz serisi yok: acik protokol
PERIODS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Panel ve kat
# ---------------------------------------------------------------------------
@dataclass
class RLPanel:
    panel: pd.DataFrame
    context: Optional[pd.DataFrame]          # pencere ONCESI satirlar (yalniz 'causal')
    integrity_report: dict
    context_report: Optional[dict]
    indicator_spec: dict
    start: str


def load_rl_panel(price_norm: str = 'stats', start: Optional[str] = None,
                  df: Optional[pd.DataFrame] = None, adx_mode: str = RL_ADX_MODE) -> RLPanel:
    """Rota ve HPO'nun ortak RL paneli.

    Sira: gostergeli CSV -> ADX 'wilder' (K18) -> [nedensel kolda pencere oncesi
    baglam, K25] -> veri kapisi (K17 penceresi, artefakt, hayalet gun, bayat
    hucre, gosterge isinma blogu; B7). `df` verilirse dosya okunmaz (test ve
    rotanin 'dosya yok' yedegi).
    """
    from data.integrity import RL_DATA_START, prepare_rl_panel, rl_context_before
    from data.technical_indicators import ADX_PERIOD, with_adx

    if start is None:
        try:
            from app.core.config import get_settings
            start = get_settings().RL_DATA_START
        except Exception:
            start = RL_DATA_START
    if df is None:
        from data.data_fetcher import DataFetcher
        df = DataFetcher().load_data(RL_PANEL_FILE)
    df = with_adx(df, mode=adx_mode)
    indicator_spec = {'adx': adx_mode, 'adx_period': ADX_PERIOD, 'source': RL_PANEL_FILE,
                      'warmup': 'bfill_csv_blocks_dropped_by_gate'}
    context, context_report = None, None
    if price_norm == 'causal':
        from env.trading_env import CAUSAL_WINDOW
        context, context_report = rl_context_before(df, start=start, n_sessions=CAUSAL_WINDOW)
    panel, report = prepare_rl_panel(df, start=start, drop_indicator_warmup=True)
    return RLPanel(panel=panel, context=context, integrity_report=report,
                   context_report=context_report, indicator_spec=indicator_spec, start=str(start))


@dataclass
class FoldData:
    spec: SplitSpec
    fold: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    report: dict
    symbols: List[str]
    dropped_symbols: List[str]
    contexts: Dict[str, Optional[pd.DataFrame]] = field(default_factory=dict)

    def frame(self, role: str) -> pd.DataFrame:
        return {'train': self.train, 'val': self.val, 'test': self.test}[role]


def build_fold(rl_panel: RLPanel, spec: SplitSpec, fold, symbols: Optional[List[str]] = None
               ) -> FoldData:
    """Kat verisi. Evren EGITIM blogunun sembolleridir; dogrulama ve test ona hizalanir
    (gozlem uzayi sembol sayisina bagli). Dusen sembol raporlanir, sessiz kalmaz.

    `symbols` verilirse panel once o alt kumeye indirilir (HPO'nun alt evren
    istegi; rotanin evreniyle uyusmazsa hiperparametre soyu acik cikar).
    """
    panel = rl_panel.panel
    if symbols is not None:
        want = list(symbols)
        have = set(panel.index.get_level_values('symbol'))
        missing = [s for s in want if s not in have]
        if missing:
            raise ValueError(f"Panelde olmayan sembol(ler): {missing}")
        panel = panel[panel.index.get_level_values('symbol').isin(want)]
    parts = fold_frames(panel, spec, fold)
    train_syms = list(pd.unique(parts['train'].index.get_level_values('symbol')))
    all_syms = list(pd.unique(panel.index.get_level_values('symbol')))
    dropped = sorted(set(all_syms) - set(train_syms))

    def align(frame):
        return frame[frame.index.get_level_values('symbol').isin(train_syms)] if dropped else frame

    train, val, test = parts['train'], align(parts['val']), align(parts['test'])
    contexts: Dict[str, Optional[pd.DataFrame]] = {'train': None, 'val': None, 'test': None}
    if rl_panel.context is not None:
        # Nedensel olcek yalniz <= t kapanislarla kurulur; held-out ortamin kayan
        # penceresi KENDI ilk tarihinden onceki satirlarla dolar (purge ile atilan
        # seanslar da gecmistir, baglama girer).
        ctx = rl_panel.context
        ctx = ctx[ctx.index.get_level_values('symbol').isin(train_syms)] if dropped else ctx
        dates = pd.DatetimeIndex(pd.to_datetime(panel.index.get_level_values('date')))
        dates = dates.tz_localize(None) if dates.tz is not None else dates
        in_universe = panel.index.get_level_values('symbol').isin(train_syms)

        def before(frame):
            fd = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values('date')))
            first = (fd.tz_localize(None) if fd.tz is not None else fd).min()
            return pd.concat([ctx, panel[(dates < first) & in_universe]])

        contexts = {'train': ctx, 'val': before(val), 'test': before(test)}
    report = dict(parts['report'])
    report['n_symbols'] = len(train_syms)
    report['dropped_symbols'] = dropped
    return FoldData(spec=spec, fold=report['fold'], train=train, val=val, test=test,
                    report=report, symbols=train_syms, dropped_symbols=dropped, contexts=contexts)


def make_fold_env(fold: FoldData, role: str, *, train_env=None, price_norm: str = 'stats',
                  price_norm_macd: bool = False, **env_kwargs):
    """Katin bir blogu icin TradingEnv (rota ve HPO ayni cagriyi yapar).

    Egitim ortami olcegi kendi penceresinden kurar (`price_stats=None`);
    dogrulama ve test ortami EGITIM ortaminin olcegini alir (B9). Nedensel
    kolda her ortam kendi ilk tarihinden ONCEKI satirlari baglam olarak alir (K25).
    """
    from env.trading_env import make_env
    if role not in ('train', 'val', 'test'):
        raise ValueError(f"role 'train' | 'val' | 'test' olmali: {role}")
    if role == 'train':
        price_stats = None
    else:
        if train_env is None:
            raise ValueError("Dogrulama/test ortami icin egitim ortami (olcegin kaynagi) gerekli")
        price_stats = getattr(train_env, 'unwrapped', train_env).get_price_stats()
    kwargs = dict(env_kwargs)
    kwargs.setdefault('missing_data_mode', 'own_last')
    kwargs.update(price_stats=price_stats, price_norm=price_norm, price_norm_macd=price_norm_macd)
    if price_norm == 'causal':
        kwargs['price_context'] = fold.contexts.get(role)
    return make_env(df=fold.frame(role), **kwargs)


# ---------------------------------------------------------------------------
# Secim metrigi
# ---------------------------------------------------------------------------
def load_risk_free(session_dates, csv_path: Optional[str] = None) -> Optional[pd.Series]:
    """Seans basi risksiz getiri; seri yoksa/yedek degerliyse None (cagiran ACIK yazar)."""
    from evaluation.risk_free import RiskFreeUnavailable, risk_free_series
    try:
        return risk_free_series(session_dates, csv_path=csv_path)
    except RiskFreeUnavailable as exc:
        logger.warning(f"Risksiz faiz serisi kullanilamiyor ({exc}); secim metrigi rf=0'a "
                       f"dusuyor ve kayit ACIK yazilir")
        return None


def excess_sharpe(returns, dates=None, risk_free: Optional[pd.Series] = None,
                  periods: int = PERIODS_PER_YEAR) -> Optional[float]:
    """Gunluk basit getiriden yillik fazla getiri Sharpe'i (M8 birincil metriginin
    dogrulamadaki karsiligi; islem maliyeti getirinin icinde).

    None: ucten az gecerli gozlem ya da BRUT getiri sabit (model hic islem
    yapmamis) — tanimsiz, secilemez. Risksiz faiz verilmezse rf=0 (cagiran
    metrigin adini `SELECTION_METRIC_RF0` yazar).
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return None
    gross_ok = np.isfinite(r)
    if gross_ok.sum() < 3 or np.nanstd(r[gross_ok], ddof=1) == 0:
        return None
    if risk_free is not None:
        if dates is None:
            raise ValueError("risk_free verildiyse dates de verilmeli")
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        rf = pd.Series(risk_free).reindex(idx).to_numpy(dtype=float)
        r = r - rf
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return None
    sd = r[ok].std(ddof=1)
    if sd == 0:
        return None
    return float(r[ok].mean() / sd * np.sqrt(periods))


def run_episode(model, env, collect_obs: bool = False) -> dict:
    """Ortamda belirlenimci tek gecis. Donus: gunluk getiri, tarihleri, portfoy serisi."""
    base = getattr(env, 'unwrapped', env)
    obs, _ = env.reset()
    seen = []
    steps = 0
    while True:
        if collect_obs:
            seen.append(obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    pv = np.asarray(base.portfolio_values, dtype=float)
    rets = pv[1:] / pv[:-1] - 1.0 if len(pv) > 1 else np.asarray([], dtype=float)
    s0 = base.frame_bound[0]
    dates = list(base._dates_list[s0 + 1: s0 + 1 + len(rets)])
    return {'returns': rets, 'dates': dates, 'portfolio_values': pv, 'steps': steps,
            'seen_obs': seen, 'n_trades': int(len(base.trades_history))}


# ---------------------------------------------------------------------------
# Kontrol noktasi kurali
# ---------------------------------------------------------------------------
def checkpoint_grid(total_timesteps: int, n_checkpoints: int = N_CHECKPOINTS) -> List[int]:
    """Esit aralikli adim izgarasi; son nokta egitim butcesinin sonu."""
    k = max(1, min(int(n_checkpoints), int(total_timesteps)))
    return sorted({int(round(total_timesteps * (i + 1) / k)) for i in range(k)})


def select_checkpoint(records: List[dict]) -> dict:
    """En yuksek dogrulama metrikli kontrol noktasi; esitlikte en erken adim.

    Hicbirinin metrigi tanimli degilse (hic islem yok) SON kontrol noktasi
    secilir ve `degenerate=True` yazilir — sessiz kalmaz.
    """
    if not records:
        raise ValueError("Kontrol noktasi kaydi yok")
    valid = [r for r in records if r.get('metric') is not None and np.isfinite(r['metric'])]
    if not valid:
        chosen = max(records, key=lambda r: r['step'])
        return {**chosen, 'degenerate': True, 'rank': None, 'n_valid': 0}
    chosen = sorted(valid, key=lambda r: (-r['metric'], r['step']))[0]
    return {**chosen, 'degenerate': False, 'rank': 1, 'n_valid': len(valid)}


def make_validation_callback(val_env_factory: Callable[[], object], total_timesteps: int,
                             checkpoint_dir: str, risk_free: Optional[pd.Series] = None,
                             n_checkpoints: int = N_CHECKPOINTS,
                             on_checkpoint: Optional[Callable[[int, dict], bool]] = None):
    """SB3 geri cagrisi: izgara noktalarinda modeli kaydet ve DOGRULAMADA olc.

    Args:
        val_env_factory: her olcumde TAZE bir dogrulama ortami donduren fonksiyon.
        checkpoint_dir: kontrol noktasi .zip'lerinin yazilacagi gecici dizin
            (model dizini DEGIL: pano her .zip'i model sayar).
        risk_free: seans basi risksiz getiri (yoksa metrik rf=0).
        on_checkpoint(idx, record) -> False donerse egitim durur (HPO budamasi).

    Olcum egitimin RNG'sine dokunmaz (belirlenimci `predict`, ayri ortam);
    `tests/test_rl_walk_forward.py` son agirliklarin geri cagri acik/kapali
    bit-es oldugunu olcer.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    grid = checkpoint_grid(total_timesteps, n_checkpoints)
    os.makedirs(checkpoint_dir, exist_ok=True)

    class ValidationCheckpoints(BaseCallback):
        def __init__(self):
            super().__init__()
            self.grid = grid
            self.records: List[dict] = []
            self.metric_name = SELECTION_METRIC if risk_free is not None else SELECTION_METRIC_RF0
            self.stopped_early = False
            self._next = 0

        def _measure(self, final: bool = False) -> bool:
            idx = len(self.records)
            path = os.path.join(checkpoint_dir, f'ckpt_{idx:02d}')
            self.model.save(path)
            env = val_env_factory()
            try:
                ep = run_episode(self.model, env)
            finally:
                close = getattr(env, 'close', None)
                if close:
                    close()
            rec = {
                'index': idx, 'step': int(self.num_timesteps),
                'grid_step': int(self.grid[-1] if final else self.grid[min(idx, len(self.grid) - 1)]),
                'final': bool(final), 'path': path + '.zip',
                'metric': excess_sharpe(ep['returns'], ep['dates'], risk_free),
                'sharpe_rf0': excess_sharpe(ep['returns']),
                'cumulative_return': float(ep['portfolio_values'][-1] / ep['portfolio_values'][0] - 1.0)
                if len(ep['portfolio_values']) > 1 else None,
                'n_trades': ep['n_trades'], 'n_returns': int(len(ep['returns'])),
            }
            self.records.append(rec)
            if on_checkpoint is not None and on_checkpoint(idx, rec) is False:
                self.stopped_early = True
                return False
            return True

        def _on_step(self) -> bool:
            # son izgara noktasi egitim bitince olculur (son guncellemeden SONRAKI model)
            while self._next < len(self.grid) - 1 and self.num_timesteps >= self.grid[self._next]:
                self._next += 1
                if not self._measure():
                    return False
            return True

        def _on_training_end(self) -> None:
            if not self.stopped_early and not any(r['final'] for r in self.records):
                self._measure(final=True)

    return ValidationCheckpoints()
