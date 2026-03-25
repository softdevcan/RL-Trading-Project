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


def _source_for_symbol(symbol: str) -> tuple:
    """Sembol için kaynak ve metrik döndür."""
    from data.gold_fetcher import METRIC_INFO, OUTPUT_SYMBOL
    metric = next((m for m, out in OUTPUT_SYMBOL.items() if out == symbol), None)
    if metric is None:
        return "borsapy", None
    info = METRIC_INFO[metric]
    source = "yfinance" if not info["borsapy"] else "borsapy"
    return source, metric


class GoldService:
    """Altın ve döviz verisi için servis."""

    def get_current_prices(self) -> List[Dict[str, Any]]:
        """Güncel ons altın, gram altın ve USD/TRY fiyatlarını döndür."""
        from data.gold_fetcher import GoldFetcher

        # Her kaynak grubu için ayrı çek
        results: Dict[str, Any] = {}

        for source, metrics in [
            ("borsapy",  ["gram_altin_try", "usd_try"]),
            ("yfinance", ["ons_altin_usd",  "usd_try"]),
        ]:
            try:
                start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
                fetcher = GoldFetcher(source=source, metrics=metrics, start_date=start)
                df = fetcher.fetch_all()
                for sym in df.index.get_level_values('symbol').unique():
                    if sym in results:
                        continue
                    sub = df.xs(sym, level='symbol').sort_index()
                    if sub.empty:
                        continue
                    last = sub.iloc[-1]
                    prev = sub.iloc[-2] if len(sub) > 1 else last
                    change_pct = ((last['close'] - prev['close']) / prev['close'] * 100
                                  if prev['close'] else 0.0)
                    results[sym] = {
                        'open': float(last['open']),
                        'high': float(last['high']),
                        'low':  float(last['low']),
                        'close': float(last['close']),
                        'change_pct': round(change_pct, 2),
                        'date': str(sub.index[-1].date()),
                    }
            except Exception as exc:
                logger.error(f"Altın fiyatları çekilemedi ({source}): {exc}")

        prices = []
        for symbol, info in results.items():
            meta = ASSET_NAMES.get(symbol, {'name': symbol, 'currency': 'USD'})
            prices.append({'symbol': symbol, 'name': meta['name'],
                           'currency': meta['currency'], **info})
        return prices

    def get_history(self, symbol: str = 'GOLD_GRAM_TRY', days: int = 90) -> Dict[str, Any]:
        """Sembol için tarihsel OHLCV verisi döndür."""
        from data.gold_fetcher import GoldFetcher, OUTPUT_SYMBOL

        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        source, metric = _source_for_symbol(symbol)
        metrics = [metric] if metric else None

        try:
            fetcher = GoldFetcher(source=source, metrics=metrics, start_date=start)
            df = fetcher.fetch_all()
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
            'open':   sub['open'].round(4).tolist(),
            'high':   sub['high'].round(4).tolist(),
            'low':    sub['low'].round(4).tolist(),
            'close':  sub['close'].round(4).tolist(),
            'volume': sub['volume'].fillna(0).tolist(),
        }
