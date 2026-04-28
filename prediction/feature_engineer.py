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

from prediction.iceemdan_processor import ICEEMDANProcessor

logger = logging.getLogger(__name__)

_iceemdan_processor = ICEEMDANProcessor()


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
        self._active_groups: list = []  # build_features cagrisi sonrasi dolar

    def build_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        macro_df: Optional[pd.DataFrame] = None,
        fundamental_df: Optional[pd.DataFrame] = None,
        cross_asset_df: Optional[pd.DataFrame] = None,
        use_iceemdan: bool = False,
        feature_groups: Optional[List[str]] = None,
        target_type: str = 'log_return',
    ) -> pd.DataFrame:
        """Tek sembol icin ozellik matrisi olustur.

        Args:
            df: OHLCV + teknik indikatorlu tek-sembol DataFrame (tarih index)
            symbol: Sembol adi
            macro_df: Makroekonomik veriler
            fundamental_df: Fundamental oranlar
            cross_asset_df: Capraz varlik verileri
            use_iceemdan: ICEEMDAN gurultu filtresi uygula (yavas)
            feature_groups: Aktif grup id listesi (None = registry default'lari)
            target_type: 'log_return' (onerilen) veya 'abs_price'

        Returns:
            Ozellik kolonlari + hedef degisken iceren DataFrame
        """
        from prediction.feature_groups import resolve_groups
        active = resolve_groups(feature_groups)
        self._active_groups = active

        logger.info(f"[{symbol}] Gelismis ozellikler olusturuluyor "
                    f"(horizon={self.horizon}, target={target_type}, "
                    f"{len(active)} grup)...")

        data = df.copy()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        if (use_iceemdan or 'iceemdan' in active) and _iceemdan_processor.is_available():
            filtered_close = _iceemdan_processor.filter_noise(data['close'])
            data['close'] = filtered_close
            imf_feats = _iceemdan_processor.extract_imf_features(filtered_close)
            for col in imf_feats.columns:
                data[col] = imf_feats[col]
            logger.info(f"[{symbol}] ICEEMDAN filtreleme uygulandi")

        if 'returns'      in active: self._add_return_features(data)
        if 'volatility'   in active: self._add_volatility_features(data)
        if 'momentum'     in active: self._add_momentum_features(data)
        if 'volume'       in active: self._add_volume_features(data)
        if 'calendar'     in active: self._add_calendar_features(data)
        if 'technicals'   in active: self._add_technical_features(data)

        # Yeni teknik gruplar
        if 'fibonacci'      in active: self._add_fibonacci_features(data)
        if 'donchian'       in active: self._add_donchian_features(data)
        if 'rolling_zscore' in active: self._add_rolling_zscore_features(data)
        if 'obv'            in active: self._add_obv_features(data)
        if 'seasonality'    in active: self._add_seasonality_features(data, symbol)

        if 'cross_asset' in active and cross_asset_df is not None and not cross_asset_df.empty:
            self._add_cross_asset_features(data, cross_asset_df)

        if macro_df is not None and not macro_df.empty:
            if 'macro_tr'     in active: self._add_macro_features(data, macro_df)
            if 'macro_global' in active: self._add_global_macro_features(data, macro_df)

        if 'fundamental' in active and fundamental_df is not None and not fundamental_df.empty:
            self._add_fundamental_features(data, fundamental_df, symbol)

        if 'market_regime' in active: self._add_market_regime_features(data)

        if self.horizon == 'weekly':
            self._add_weekly_features(data)

        self._build_targets(data, target_type=target_type)

        initial_len = len(data)
        data = data.dropna(subset=['target_price'])
        feature_cols = self.get_feature_columns(data)

        # Tamamen NaN olan feature kolonlarini tani: dropna tum satirlari uçuruyorsa
        # kullaniciya hangi kolonun sorumlu oldugunu göster.
        all_nan_cols = [c for c in feature_cols if data[c].notna().sum() == 0]
        if all_nan_cols:
            logger.warning(
                f"[{symbol}] {len(all_nan_cols)} feature tamamen NaN, atlanıyor: "
                f"{all_nan_cols[:10]}{'...' if len(all_nan_cols) > 10 else ''}"
            )
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]
            data = data.drop(columns=all_nan_cols)

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
            # Faz 3.4.3: Global gostergeler
            'vix': 'macro_vix',
            'us10y': 'macro_us10y',
            'dxy': 'macro_dxy',
        }

        for src_col, dst_col in macro_cols.items():
            if src_col not in macro.columns:
                continue
            aligned = macro[src_col].reindex(data.index, method='ffill')
            # Tamamen NaN ise (kaynak tarih araligiyla cakismiyor) kolonu atla —
            # aksi halde dropna(subset=feature_cols) tum satirlari uçurur.
            if aligned.notna().sum() == 0:
                continue
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

    def _build_targets(self, data: pd.DataFrame, target_type: str = 'log_return'):
        """Fiyat tahmini, getiri tahmini ve yon tahmini hedefleri.

        Args:
            target_type: 'log_return' (onerilen, stationarity saglar) veya
                         'abs_price' (mutlak fiyat, eski davranis)
        """
        close = data['close']
        shift_n = -1 if self.horizon == 'daily' else -5

        if target_type == 'log_return':
            # log(P_{t+H} / P_t) — buyume olceginden bagimsiz, daha stasyoner
            data['target_price'] = np.log(close.shift(shift_n) / close)
        else:
            data['target_price'] = close.shift(shift_n)

        data['target_return'] = (close.shift(shift_n) / close - 1)
        data['target_direction'] = (close.shift(shift_n) > close).astype(float)
        data['target'] = data['target_price']
        data['_target_type'] = target_type   # predict asamasinda geri donusum icin

    # ------------------------------------------------------------------
    # Yeni Feature Metodlari (Faz 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_fibonacci_features(data: pd.DataFrame):
        """Fibonacci retracement seviyeleri.

        Son 50 / 100 / 200 gunluk swing high-low'dan %23.6, %38.2, %50, %61.8
        uzakligi — gunluk fiyatin bu seviyelere goreceli konumu.
        """
        close = data['close']
        for window in (50, 100, 200):
            rolling_high = close.rolling(window, min_periods=window // 2).max()
            rolling_low  = close.rolling(window, min_periods=window // 2).min()
            rng = (rolling_high - rolling_low).replace(0, np.nan)
            for level, pct in [('236', 0.236), ('382', 0.382),
                                ('500', 0.500), ('618', 0.618)]:
                fib_val = rolling_high - rng * pct
                col = f'fib_{window}_{level}'
                # Fiyatin fib seviyesinin kac % uzaginda
                data[col] = ((close - fib_val) / close).shift(1)
        # Fiyatin son 50g range icindeki normalized konumu (0-1)
        data['price_position_50'] = (
            (close - close.rolling(50, min_periods=20).min()) /
            (close.rolling(50, min_periods=20).max() -
             close.rolling(50, min_periods=20).min() + 1e-9)
        ).shift(1)

    @staticmethod
    def _add_donchian_features(data: pd.DataFrame):
        """Donchian kanalı ve Chandelier Exit."""
        close  = data['close']
        high   = data.get('high', close)
        low    = data.get('low', close)

        for w in (20, 55):
            dc_high = high.rolling(w, min_periods=w // 2).max()
            dc_low  = low.rolling(w,  min_periods=w // 2).min()
            dc_mid  = (dc_high + dc_low) / 2
            dc_width = ((dc_high - dc_low) / close.replace(0, np.nan))
            data[f'dc_{w}_width']    = dc_width.shift(1)
            data[f'dc_{w}_position'] = (
                (close - dc_low) / (dc_high - dc_low + 1e-9)
            ).shift(1)
            data[f'dc_{w}_mid_dist'] = ((close - dc_mid) / close.replace(0, np.nan)).shift(1)

        # Chandelier Exit (22-gun, 3xATR)
        atr_22 = (high - low).rolling(22, min_periods=10).mean()
        chandelier_long  = high.rolling(22, min_periods=10).max() - atr_22 * 3
        chandelier_short = low.rolling(22,  min_periods=10).min() + atr_22 * 3
        data['chandelier_long_dist']  = ((close - chandelier_long)  / close.replace(0, np.nan)).shift(1)
        data['chandelier_short_dist'] = ((close - chandelier_short) / close.replace(0, np.nan)).shift(1)

    @staticmethod
    def _add_rolling_zscore_features(data: pd.DataFrame):
        """Rolling Z-score — ortalamaya donus ve momentum sinyali."""
        close = data['close']
        log_ret = np.log(close / close.shift(1))

        for w in (20, 60, 252):
            mu  = log_ret.rolling(w, min_periods=w // 2).mean()
            std = log_ret.rolling(w, min_periods=w // 2).std()
            data[f'zscore_ret_{w}'] = ((log_ret - mu) / std.replace(0, np.nan)).shift(1)

        # Fiyatin kendi hareketli ortalamasina gore Z-score
        for w in (20, 50):
            mu_p  = close.rolling(w, min_periods=w // 2).mean()
            std_p = close.rolling(w, min_periods=w // 2).std()
            data[f'zscore_price_{w}'] = ((close - mu_p) / std_p.replace(0, np.nan)).shift(1)

    @staticmethod
    def _add_obv_features(data: pd.DataFrame):
        """On Balance Volume ve Chaikin Money Flow."""
        close  = data['close']
        volume = data.get('volume', pd.Series(1, index=data.index))

        # OBV
        direction = np.sign(close.diff())
        obv = (direction * volume).fillna(0).cumsum()
        obv_ma20 = obv.rolling(20, min_periods=5).mean()
        data['obv_ma_ratio'] = (obv / obv_ma20.replace(0, np.nan)).shift(1)
        data['obv_slope_5']  = obv.diff(5).shift(1)

        # CMF (Chaikin Money Flow)
        high = data.get('high', close)
        low  = data.get('low', close)
        mfm  = ((close - low) - (high - close)) / (high - low + 1e-9)
        mfv  = mfm * volume
        for w in (20, 40):
            cmf = mfv.rolling(w, min_periods=w // 2).sum() / \
                  volume.rolling(w, min_periods=w // 2).sum().replace(0, np.nan)
            data[f'cmf_{w}'] = cmf.shift(1)

    @staticmethod
    def _add_global_macro_features(data: pd.DataFrame, macro_df: pd.DataFrame):
        """Global makro feature'lari: petrol, GVZ, reel faiz, altin-gumus orani."""
        aligned = macro_df.reindex(data.index, method='ffill')

        global_cols = {
            'oil_wti':      'oil_wti',
            'gold_vix':     'gold_vix',
            'silver':       'silver',
            'us_real_rate': 'us_real_rate',
        }
        for src_col, feat_name in global_cols.items():
            if src_col in aligned.columns:
                series = aligned[src_col]
                # Seviye
                data[feat_name] = series.shift(1)
                # Normalize degisim (1g, 5g)
                data[f'{feat_name}_chg1'] = series.pct_change(1).shift(1)
                data[f'{feat_name}_chg5'] = series.pct_change(5).shift(1)

        # Altin/Gumus orani (silver varsa)
        if 'silver' in aligned.columns:
            silver_s = aligned['silver'].replace(0, np.nan)
            # close = gold price (bu fonksiyon gold dataframe'i uzerinde cagriliyor olabilir)
            gold_silver_ratio = data['close'] / silver_s
            data['gold_silver_ratio']      = gold_silver_ratio.shift(1)
            data['gold_silver_ratio_chg5'] = gold_silver_ratio.pct_change(5).shift(1)

    @staticmethod
    def _add_seasonality_features(data: pd.DataFrame, symbol: str = ''):
        """Mevsimsellik ve takvim etki ozellikleri.

        - Hindistan dugun sezonu (Ekim-Kasim) — altin icin pozitif sezonalite
        - Cin Yeni Yili (Ocak-Subat)
        - Merkez bankasi alim donemleri (1. ve 3. ceyrek)
        - Aylik ve ceyrek siklik encoding (sin/cos)
        """
        idx = data.index
        month   = idx.month
        quarter = idx.quarter

        # Siklik encoding
        data['month_sin'] = np.sin(2 * np.pi * month / 12)
        data['month_cos'] = np.cos(2 * np.pi * month / 12)
        data['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
        data['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)

        # Ozel donem bayraklari (gold-specific; diger varliklar icin nortral)
        data['indian_wedding_season'] = month.isin([10, 11]).astype(float)
        data['chinese_new_year']      = month.isin([1, 2]).astype(float)
        data['cb_buying_season']      = quarter.isin([1, 3]).astype(float)

        # Yil ici gun konumu (0-1)
        data['year_progress'] = (idx.dayofyear / 365.0)

        # Gecikme uygulama (data leakage onleme) — bu ozellikler gelecege bakmiyor
        # ama tutarlilik icin shift(1) uygulariz
        for col in ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                    'indian_wedding_season', 'chinese_new_year',
                    'cb_buying_season', 'year_progress']:
            if col in data.columns:
                data[col] = data[col].shift(1)


