"""
Tahmin Özellik Mühendisliği
Günlük ve haftalık fiyat tahmini için özellik üretir.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class PredictionFeatureEngineer:
    """XGBoost tabanlı fiyat tahmini için özellik üretici.

    OHLCV + teknik indikatör verisini alır, gecikmeli özellikler,
    rolling istatistikler ve takvim özellikleri ekler.
    """

    def __init__(self, horizon: str = 'daily'):
        """
        Args:
            horizon: 'daily' (yarın) veya 'weekly' (haftaya)
        """
        if horizon not in ('daily', 'weekly'):
            raise ValueError("horizon 'daily' veya 'weekly' olmalı")
        self.horizon = horizon

    # ------------------------------------------------------------------
    # Ana API
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Tek sembol için özellik matrisi oluştur.

        Args:
            df: OHLCV + teknik indikatörlü tek-sembol DataFrame (tarih index'li)
            symbol: Hisse/varlık sembolü (log amaçlı)

        Returns:
            Özellik kolonları + hedef değişken içeren DataFrame
        """
        logger.info(f"[{symbol}] Özellikler oluşturuluyor (horizon={self.horizon})...")

        data = df.copy()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()

        # --- Gecikmeli kapanış ve getiriler ---
        for lag in range(1, 6):
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'return_lag_{lag}'] = data['close'].pct_change(lag).shift(1)

        # --- Rolling istatistikler ---
        for window in (5, 10, 20):
            data[f'mean_{window}'] = data['close'].rolling(window).mean().shift(1)
        for window in (5, 20):
            data[f'std_{window}'] = data['close'].rolling(window).std().shift(1)

        # --- Hacim özellikleri ---
        data['volume_lag_1'] = data['volume'].shift(1)
        data['volume_ma_5'] = data['volume'].rolling(5).mean().shift(1)

        # --- Fiyat oranları ---
        data['close_open_ratio'] = (data['close'] / data['open'].replace(0, np.nan)).shift(1)
        data['high_low_ratio'] = (data['high'] / data['low'].replace(0, np.nan)).shift(1)

        # --- Takvim özellikleri ---
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month

        # --- Teknik indikatörler (varsa) ---
        for col in ['macd', 'rsi', 'cci', 'adx', 'turbulence']:
            if col in data.columns:
                data[f'{col}_lag_1'] = data[col].shift(1)

        # --- Haftalık ek özellikler ---
        if self.horizon == 'weekly':
            data['weekly_close'] = data['close'].resample('W').last().reindex(
                data.index, method='ffill'
            )
            for lag in range(1, 5):
                data[f'weekly_return_lag_{lag}'] = (
                    data['weekly_close'].pct_change(lag).shift(1)
                )
            data['weekly_high'] = data['high'].rolling(5).max().shift(1)
            data['weekly_low'] = data['low'].rolling(5).min().shift(1)

        # --- Hedef değişken ---
        if self.horizon == 'daily':
            data['target'] = data['close'].shift(-1)
        else:
            data['target'] = data['close'].shift(-5)

        # NaN satırlarını kaldır
        initial_len = len(data)
        data = data.dropna(subset=['target'])
        feature_cols = self._get_feature_cols(data)
        data = data.dropna(subset=feature_cols)
        logger.info(
            f"  {initial_len} → {len(data)} satır (NaN kaldırıldı)"
        )

        return data

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Tahmin için kullanılacak kolon listesini döndür."""
        return self._get_feature_cols(df)

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Özellik kolonlarını otomatik tespit et (target hariç)."""
        exclude = {'open', 'high', 'low', 'close', 'volume', 'target',
                   'macd', 'rsi', 'cci', 'adx', 'turbulence',
                   'weekly_close'}
        return [c for c in df.columns if c not in exclude]
