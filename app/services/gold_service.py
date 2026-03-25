"""
Altın Servisi
Güncel altın fiyatları ve tarihsel veri için servis katmanı
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

ASSET_NAMES = {
    'GC=F': {'name': 'Altin Ons (USD)', 'currency': 'USD'},
    'USDTRY=X': {'name': 'USD/TRY', 'currency': 'TRY'},
    'GOLD_GRAM_TRY': {'name': 'Gram Altin (TRY)', 'currency': 'TRY'},
}


class GoldService:
    """Altın ve döviz verisi için servis."""

    def get_current_prices(self) -> List[Dict[str, Any]]:
        """Güncel ons altın, gram altın ve USD/TRY fiyatlarını döndür."""
        from data.gold_fetcher import GoldFetcher

        try:
            fetcher = GoldFetcher(
                start_date=(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
            )
            raw = fetcher.get_latest_prices()
        except Exception as exc:
            logger.error(f"Altın fiyatları çekilemedi: {exc}")
            return []

        prices = []
        for symbol, info in raw.items():
            meta = ASSET_NAMES.get(symbol, {'name': symbol, 'currency': 'USD'})
            prices.append({
                'symbol': symbol,
                'name': meta['name'],
                'close': info['close'],
                'open': info['open'],
                'high': info['high'],
                'low': info['low'],
                'change_pct': info['change_pct'],
                'date': info['date'],
                'currency': meta['currency'],
            })

        return prices

    def get_history(self, symbol: str = 'GOLD_GRAM_TRY', days: int = 90) -> Dict[str, Any]:
        """Sembol için tarihsel OHLCV verisi döndür."""
        from data.gold_fetcher import GoldFetcher

        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        fetcher = GoldFetcher(start_date=start)

        try:
            df = fetcher.fetch_all_gold_data()
        except Exception as exc:
            logger.error(f"Altın tarih verisi çekilemedi: {exc}")
            return {'symbol': symbol, 'dates': [], 'open': [], 'high': [],
                    'low': [], 'close': [], 'volume': []}

        if symbol not in df.index.get_level_values('symbol').unique():
            logger.warning(f"{symbol} verisi yok")
            return {'symbol': symbol, 'dates': [], 'open': [], 'high': [],
                    'low': [], 'close': [], 'volume': []}

        sub = df.xs(symbol, level='symbol').sort_index()
        sub.index = pd.to_datetime(sub.index).tz_localize(None)

        return {
            'symbol': symbol,
            'dates': sub.index.strftime('%Y-%m-%d').tolist(),
            'open': sub['open'].round(4).tolist(),
            'high': sub['high'].round(4).tolist(),
            'low': sub['low'].round(4).tolist(),
            'close': sub['close'].round(4).tolist(),
            'volume': sub['volume'].fillna(0).tolist(),
        }
