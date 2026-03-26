"""
Gelismis Tahmin Ozellik Muhendisligi

OHLCV, teknik indikator, makroekonomik ve fundamental verilerden
akademik literaturde kanitlanmis ozellikleri uretir.

Ozellik gruplari:
1. Getiri ozellikleri (log return, basit return, coklu gecikme)
2. Volatilite ozellikleri (Parkinson, Garman-Klass, realized vol)
3. Momentum ozellikleri (RSI turevleri, MACD histogram gradyani)
4. Hacim ozellikleri (hacim oranlari, hacim-fiyat korelasyonu)
5. Takvim ozellikleri (gun, ay, ceyrek)
6. Teknik indikator ozellikleri (gecikmeliler + turevler)
7. Capraz varlik ozellikleri (BIST-100, USD/TRY korelasyonu)
8. Makro ozellikleri (faiz degisimi, enflasyon trendi)
9. Fundamental ozellikleri (P/E, P/B degisim)
10. Piyasa rejimi ozellikleri (volatilite rejimi, trend rejimi)
"""

import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionFeatureEngineer:
    """Coklu veri kaynagindan gelismis ozellik uretici.

    Tum ozellikler en az 1 gun geciktirilir (shift) —
    data leakage onlenir.
    """

    HORIZONS = ('daily', 'weekly')

    def __init__(self, horizon: str = 'daily'):
        if horizon not in self.HORIZONS:
            raise ValueError(f"horizon {self.HORIZONS} icinden biri olmali")
        self.horizon = horizon

    def build_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Tek sembol icin ozellik matrisi olustur.

        Args:
            df: OHLCV + teknik indikatorlu tek-sembol DataFrame (tarih index)
            symbol: Sembol adi
            macro_df: Makroekonomik veriler (tarih index, kolonlar: policy_rate, cpi_inflation, vb.)
            fundamental_df: Fundamental oranlar (sembol index, kolonlar: pe_ratio, pb_ratio, vb.)
            cross_asset_df: Capraz varlik verileri (tarih index, kolonlar: bist100, usdtry)

        Returns:
            Ozellik kolonlari + hedef degisken iceren DataFrame
        """
        logger.info(f"[{symbol}] Gelismis ozellikler olusturuluyor (horizon={self.horizon})...")

        data = df.copy()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        self._add_return_features(data)
        self._add_volatility_features(data)
        self._add_momentum_features(data)
        self._add_volume_features(data)
        self._add_calendar_features(data)
        self._add_technical_features(data)

        if cross_asset_df is not None and not cross_asset_df.empty:
            self._add_cross_asset_features(data, cross_asset_df)

        if macro_df is not None and not macro_df.empty:
            self._add_macro_features(data, macro_df)

        if fundamental_df is not None and not fundamental_df.empty:
            self._add_fundamental_features(data, fundamental_df, symbol)

        self._add_market_regime_features(data)

        if self.horizon == 'weekly':
            self._add_weekly_features(data)

        self._build_targets(data)

        initial_len = len(data)
        data = data.dropna(subset=['target_price'])
        feature_cols = self.get_feature_columns(data)
        data = data.dropna(subset=feature_cols)
        logger.info(f"  {initial_len} -> {len(data)} satir ({len(feature_cols)} ozellik)")

        return data

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Tahmin icin kullanilacak kolon listesini dondur."""
        exclude = {
            'open', 'high', 'low', 'close', 'volume',
            'target_price', 'target_return', 'target_direction', 'target',
            'macd', 'macd_signal', 'macd_hist',
            'rsi', 'cci', 'adx', 'turbulence',
            'weekly_close',
        }
        return [c for c in df.columns if c not in exclude and not c.startswith('_')]

    # ------------------------------------------------------------------
    # 1. Getiri Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_return_features(data: pd.DataFrame):
        """Log return ve basit return ozellikleri."""
        close = data['close']
        safe_close = close.replace(0, np.nan)

        for lag in range(1, 11):
            data[f'log_return_{lag}d'] = np.log(safe_close / safe_close.shift(lag)).shift(1)

        for lag in [1, 2, 3, 5, 10, 20]:
            data[f'return_{lag}d'] = close.pct_change(lag).shift(1)

        for lag in range(1, 6):
            data[f'close_lag_{lag}'] = close.shift(lag)

        data['return_skew_20'] = data['log_return_1d'].rolling(20).skew().shift(1)
        data['return_kurt_20'] = data['log_return_1d'].rolling(20).kurt().shift(1)

    # ------------------------------------------------------------------
    # 2. Volatilite Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volatility_features(data: pd.DataFrame):
        """Parkinson, Garman-Klass ve realized volatility."""
        h = data['high']
        l = data['low']
        c = data['close']
        o = data['open']

        safe_hl = (h / l.replace(0, np.nan)).replace(0, np.nan)
        data['parkinson_vol_20'] = np.sqrt(
            (1.0 / (4.0 * np.log(2))) * (np.log(safe_hl) ** 2).rolling(20).mean()
        ).shift(1)

        safe_oc = (c / o.replace(0, np.nan)).replace(0, np.nan)
        gk_term = 0.5 * (np.log(safe_hl) ** 2) - (2 * np.log(2) - 1) * (np.log(safe_oc) ** 2)
        data['garman_klass_vol_20'] = np.sqrt(gk_term.rolling(20).mean()).shift(1)

        log_ret = np.log(c / c.shift(1))
        for w in [5, 10, 20]:
            data[f'realized_vol_{w}'] = log_ret.rolling(w).std().shift(1) * np.sqrt(252)

        data['vol_ratio_5_20'] = (
            data['realized_vol_5'] / data['realized_vol_20'].replace(0, np.nan)
        )

        for w in [5, 20]:
            data[f'std_{w}'] = c.rolling(w).std().shift(1)

    # ------------------------------------------------------------------
    # 3. Momentum Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_momentum_features(data: pd.DataFrame):
        """RSI turevleri, MACD histogram gradyani, ADX trend gucu."""
        close = data['close']

        for w in [5, 10, 20, 50]:
            ma = close.rolling(w).mean()
            data[f'price_to_ma_{w}'] = (close / ma.replace(0, np.nan) - 1).shift(1)

        data['ma_cross_5_20'] = (
            (close.rolling(5).mean() > close.rolling(20).mean()).astype(float).shift(1)
        )
        data['ma_cross_10_50'] = (
            (close.rolling(10).mean() > close.rolling(50).mean()).astype(float).shift(1)
        )

        for w in [5, 10, 20]:
            data[f'mean_{w}'] = close.rolling(w).mean().shift(1)

        data['momentum_5'] = (close / close.shift(5) - 1).shift(1)
        data['momentum_10'] = (close / close.shift(10) - 1).shift(1)
        data['momentum_20'] = (close / close.shift(20) - 1).shift(1)

        data['roc_5'] = close.pct_change(5).shift(1) * 100
        data['roc_10'] = close.pct_change(10).shift(1) * 100

    # ------------------------------------------------------------------
    # 4. Hacim Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_volume_features(data: pd.DataFrame):
        """Hacim bazli ozellikler."""
        vol = data['volume']
        close = data['close']

        data['volume_lag_1'] = vol.shift(1)
        for w in [5, 10, 20]:
            data[f'volume_ma_{w}'] = vol.rolling(w).mean().shift(1)

        vol_ma5 = vol.rolling(5).mean()
        data['volume_ratio_5'] = (vol / vol_ma5.replace(0, np.nan)).shift(1)

        data['volume_change_1d'] = vol.pct_change().shift(1)

        ret = close.pct_change()
        data['volume_price_corr_20'] = ret.rolling(20).corr(vol).shift(1)

        obv = (np.sign(close.diff()) * vol).cumsum()
        data['obv_slope_10'] = (obv - obv.shift(10)).shift(1)

    # ------------------------------------------------------------------
    # 5. Takvim Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_calendar_features(data: pd.DataFrame):
        """Takvim bazli ozellikler."""
        idx = data.index
        data['day_of_week'] = idx.dayofweek
        data['month'] = idx.month
        data['quarter'] = idx.quarter
        data['is_month_start'] = idx.is_month_start.astype(float)
        data['is_month_end'] = idx.is_month_end.astype(float)
        data['is_quarter_end'] = idx.is_quarter_end.astype(float)
        data['week_of_year'] = idx.isocalendar().week.astype(int)

        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 5)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 5)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    # ------------------------------------------------------------------
    # 6. Teknik Indikator Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_technical_features(data: pd.DataFrame):
        """Mevcut teknik indikatorlerden turetilmis ozellikler."""
        for col in ['macd', 'rsi', 'cci', 'adx', 'turbulence']:
            if col not in data.columns:
                continue
            data[f'{col}_lag_1'] = data[col].shift(1)
            data[f'{col}_lag_2'] = data[col].shift(2)
            data[f'{col}_change_1d'] = data[col].diff().shift(1)
            data[f'{col}_change_5d'] = data[col].diff(5).shift(1)

        if 'macd_signal' in data.columns:
            data['macd_signal_lag_1'] = data['macd_signal'].shift(1)
        if 'macd_hist' in data.columns:
            data['macd_hist_lag_1'] = data['macd_hist'].shift(1)
            data['macd_hist_gradient'] = data['macd_hist'].diff().shift(1)
            data['macd_hist_accel'] = data['macd_hist'].diff().diff().shift(1)

        if 'rsi' in data.columns:
            rsi = data['rsi']
            data['rsi_overbought'] = (rsi > 70).astype(float).shift(1)
            data['rsi_oversold'] = (rsi < 30).astype(float).shift(1)
            data['rsi_ma_5'] = rsi.rolling(5).mean().shift(1)

        if 'adx' in data.columns:
            data['adx_strong_trend'] = (data['adx'] > 25).astype(float).shift(1)

    # ------------------------------------------------------------------
    # 7. Capraz Varlik Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_cross_asset_features(data: pd.DataFrame, cross_asset_df: pd.DataFrame):
        """BIST-100, USD/TRY ile korelasyon ve etkilesim."""
        cross = cross_asset_df.copy()
        cross.index = pd.to_datetime(cross.index)

        if 'bist100' in cross.columns or 'bist100_index' in cross.columns:
            bist_col = 'bist100' if 'bist100' in cross.columns else 'bist100_index'
            bist = cross[bist_col].reindex(data.index, method='ffill')
            bist_ret = bist.pct_change()
            stock_ret = data['close'].pct_change()

            data['bist100_return_1d'] = bist_ret.shift(1)
            data['bist100_return_5d'] = bist.pct_change(5).shift(1)
            data['bist100_corr_20'] = stock_ret.rolling(20).corr(bist_ret).shift(1)
            data['bist100_beta_60'] = (
                stock_ret.rolling(60).cov(bist_ret) /
                bist_ret.rolling(60).var().replace(0, np.nan)
            ).shift(1)

        if 'usd_try' in cross.columns:
            usd = cross['usd_try'].reindex(data.index, method='ffill')
            usd_ret = usd.pct_change()

            data['usdtry_return_1d'] = usd_ret.shift(1)
            data['usdtry_return_5d'] = usd.pct_change(5).shift(1)
            data['usdtry_vol_20'] = usd_ret.rolling(20).std().shift(1)

        if 'eur_try' in cross.columns:
            eur = cross['eur_try'].reindex(data.index, method='ffill')
            data['eurtry_return_1d'] = eur.pct_change().shift(1)

    # ------------------------------------------------------------------
    # 8. Makro Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_macro_features(data: pd.DataFrame, macro_df: pd.DataFrame):
        """Makroekonomik gosterge ozellikleri."""
        macro = macro_df.copy()
        macro.index = pd.to_datetime(macro.index)

        macro_cols = {
            'policy_rate': 'macro_policy_rate',
            'cpi_inflation': 'macro_cpi',
            'ppi_inflation': 'macro_ppi',
            'usd_try': 'macro_usdtry',
            'eur_try': 'macro_eurtry',
            'bist100_index': 'macro_bist100',
        }

        for src_col, dst_col in macro_cols.items():
            if src_col not in macro.columns:
                continue
            aligned = macro[src_col].reindex(data.index, method='ffill')
            data[dst_col] = aligned.shift(1)

            if src_col in ('policy_rate', 'cpi_inflation', 'ppi_inflation'):
                data[f'{dst_col}_change'] = aligned.diff().shift(1)
            else:
                data[f'{dst_col}_return'] = aligned.pct_change().shift(1)

    # ------------------------------------------------------------------
    # 9. Fundamental Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_fundamental_features(
        data: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        symbol: str,
    ):
        """Fundamental oran ozellikleri (statik, sembol bazli)."""
        if symbol not in fundamental_df.index:
            return

        ratios = fundamental_df.loc[symbol]
        fund_cols = {
            'pe_ratio': 'fund_pe',
            'pb_ratio': 'fund_pb',
            'roe': 'fund_roe',
            'roa': 'fund_roa',
            'debt_to_equity': 'fund_leverage',
            'current_ratio': 'fund_current_ratio',
            'profit_margin': 'fund_margin',
        }

        for src, dst in fund_cols.items():
            val = ratios.get(src, np.nan)
            if pd.notna(val):
                data[dst] = float(val)

    # ------------------------------------------------------------------
    # 10. Piyasa Rejimi Ozellikleri
    # ------------------------------------------------------------------

    @staticmethod
    def _add_market_regime_features(data: pd.DataFrame):
        """Volatilite rejimi ve trend rejimi tespiti."""
        if 'realized_vol_20' not in data.columns:
            return

        vol = data['realized_vol_20']
        vol_median = vol.rolling(252, min_periods=60).median()

        data['vol_regime'] = 0.0
        if vol_median is not None:
            data.loc[vol > vol_median * 1.5, 'vol_regime'] = 1.0
            data.loc[vol < vol_median * 0.5, 'vol_regime'] = -1.0

        close = data['close']
        ma_50 = close.rolling(50, min_periods=20).mean()
        ma_200 = close.rolling(200, min_periods=60).mean()

        data['trend_regime'] = 0.0
        if ma_50 is not None and ma_200 is not None:
            data.loc[(close > ma_50) & (ma_50 > ma_200), 'trend_regime'] = 1.0
            data.loc[(close < ma_50) & (ma_50 < ma_200), 'trend_regime'] = -1.0

        data['drawdown'] = (
            close / close.rolling(252, min_periods=20).max() - 1
        ).shift(1)

    # ------------------------------------------------------------------
    # Haftalik Ek Ozellikler
    # ------------------------------------------------------------------

    @staticmethod
    def _add_weekly_features(data: pd.DataFrame):
        """Haftalik horizon icin ek ozellikler."""
        close = data['close']

        data['weekly_close'] = close.resample('W').last().reindex(
            data.index, method='ffill'
        )
        for lag in range(1, 5):
            data[f'weekly_return_lag_{lag}'] = (
                data['weekly_close'].pct_change(lag).shift(1)
            )
        data['weekly_high'] = data['high'].rolling(5).max().shift(1)
        data['weekly_low'] = data['low'].rolling(5).min().shift(1)
        data['weekly_range'] = (
            (data['weekly_high'] - data['weekly_low']) /
            data['weekly_low'].replace(0, np.nan)
        )

    # ------------------------------------------------------------------
    # Hedef Degiskenler
    # ------------------------------------------------------------------

    def _build_targets(self, data: pd.DataFrame):
        """Fiyat tahmini, getiri tahmini ve yon tahmini hedefleri."""
        close = data['close']

        if self.horizon == 'daily':
            shift_n = -1
        else:
            shift_n = -5

        data['target_price'] = close.shift(shift_n)
        data['target_return'] = (close.shift(shift_n) / close - 1)
        data['target_direction'] = (close.shift(shift_n) > close).astype(float)

        data['target'] = data['target_price']
