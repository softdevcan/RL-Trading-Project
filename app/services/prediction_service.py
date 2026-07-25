"""
Gelismis Tahmin Servisi

Ensemble tabanli model egitim, tahmin uretme ve performans takibi.
Mevcut PredictionService API'sini korurken yeni ensemble mimarisini kullanir.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)


def is_gold_symbol(symbol: str) -> bool:
    """Sembol gold/FX kategorisinde mi?"""
    from data.gold_fetcher import BORSAPY_SYMBOLS, YFINANCE_SYMBOLS, OUTPUT_SYMBOL
    _gold_output = set(OUTPUT_SYMBOL.values())
    _gold_raw = set(BORSAPY_SYMBOLS.values()) | set(YFINANCE_SYMBOLS.values())
    return symbol in _gold_output or symbol in _gold_raw


def get_supported_sources(symbol: str) -> List[str]:
    """Sembol icin desteklenen kaynaklar (yfinance, borsapy)."""
    from data.gold_fetcher import OUTPUT_SYMBOL, METRIC_INFO

    if not is_gold_symbol(symbol):
        return ['yfinance']

    metric = next((m for m, out in OUTPUT_SYMBOL.items() if out == symbol), None)
    if metric is None:
        return ['yfinance']

    info = METRIC_INFO.get(metric, {})
    sources = []
    if info.get('borsapy'):
        sources.append('borsapy')
    if info.get('yfinance') in ('direct', 'computed'):
        sources.append('yfinance')
    return sources or ['yfinance']


def _resolve_source(symbol: str, source: Optional[str]) -> Optional[str]:
    """Kullanicinin verdigi source'u dogrula; gold dis sembollerde None doner."""
    if not is_gold_symbol(symbol):
        return None  # hisse sembolleri icin source ayrimi yok
    supported = get_supported_sources(symbol)
    if source is None:
        return supported[0]  # default: ilk desteklenen
    if source not in supported:
        raise ValueError(
            f"'{symbol}' icin '{source}' kaynagi desteklenmiyor. "
            f"Desteklenen: {supported}"
        )
    return source


def _fetch_symbol_data(
    symbol: str,
    start_date: str,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Sembol icin OHLCV + teknik indikator verisi cek.

    Args:
        symbol: Sembol adi
        start_date: Baslangic tarihi
        source: Gold/FX icin 'borsapy' veya 'yfinance'. Hisse icin yoksayilir.
    """
    from data.technical_indicators import add_indicators_to_multi_symbol_df

    if is_gold_symbol(symbol):
        from data.gold_fetcher import GoldFetcher, OUTPUT_SYMBOL as _OUT

        resolved = _resolve_source(symbol, source)
        metric = next((m for m, out in _OUT.items() if out == symbol), None)
        metrics = [metric] if metric else None

        fetcher = GoldFetcher(source=resolved, metrics=metrics, start_date=start_date)
        df_multi = fetcher.fetch_all()
    else:
        from data.data_fetcher import DataFetcher
        fetcher = DataFetcher(start_date=start_date)
        df_multi = fetcher.fetch_stock_data([symbol], save=False)

    df_multi = add_indicators_to_multi_symbol_df(df_multi)

    available = df_multi.index.get_level_values('symbol').unique()
    if symbol in available:
        df = df_multi.xs(symbol, level='symbol').copy()
    else:
        raise ValueError(f"{symbol} verisi bulunamadi. Mevcut: {list(available)}")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = _clean_ohlcv(df)
    return df


# ── Predictor cache ──────────────────────────────────────────────────────
# Loading an ensemble means deserializing XGBoost + LightGBM + CatBoost +
# BiLSTM + TFT from disk, which used to happen on every single /predict
# call. Cache the loaded PricePredictor per (symbol, horizon, source) and
# invalidate on the ensemble meta file's mtime, so a fresh training run is
# picked up without a server restart.
_predictor_cache: Dict[tuple, tuple] = {}  # key -> (predictor, meta_mtime)


def _get_cached_predictor(horizon: str, symbol: str, source: Optional[str]):
    """PricePredictor'i cache'ten dondur; egitilmis meta dosyasi degismisse yeniden yukle."""
    from prediction.models import PricePredictor

    key = (symbol, horizon, source)
    predictor = PricePredictor(horizon, source=source)
    meta_path = predictor.ensemble._ensemble_meta_path(symbol)

    try:
        meta_mtime = os.path.getmtime(meta_path)
    except OSError:
        meta_mtime = None

    cached = _predictor_cache.get(key)
    if cached is not None and cached[1] == meta_mtime:
        return cached[0]

    if predictor.is_trained(symbol):
        predictor.load(symbol)
    _predictor_cache[key] = (predictor, meta_mtime)
    return predictor


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV verisi icin temel temizleme."""
    df = df[~df.index.duplicated(keep='last')].copy()
    df = df.sort_index()

    df = df[df['close'] > 0]

    flat_mask = (df['open'] == df['high']) & (df['high'] == df['low']) & (df['low'] == df['close'])
    if flat_mask.any():
        logger.debug(f"  {flat_mask.sum()} duz (flat) satir bulundu, forward fill uygulaniyor")
        df.loc[flat_mask, ['open', 'high', 'low', 'close']] = None

    rolling_med = df['close'].rolling(30, min_periods=5).median()
    lower = rolling_med * 0.5
    upper = rolling_med * 1.5
    outlier_mask = (df['close'] < lower) | (df['close'] > upper)
    if outlier_mask.any():
        logger.warning(f"  {outlier_mask.sum()} aykiri close degeri kirpiliyor")
        df.loc[outlier_mask, 'close'] = None

    df = df.ffill().bfill()

    return df


def _fetch_macro_data(start_date: str = '2018-01-01') -> Optional[pd.DataFrame]:
    """Makroekonomik verileri cek veya yukle (opsiyonel)."""
    try:
        from data.macro_fetcher import MacroDataFetcher
        fetcher = MacroDataFetcher(start_date=start_date)
        try:
            return fetcher.load_data()
        except FileNotFoundError:
            return fetcher.fetch_macro_data(save=True)
    except Exception as exc:
        logger.warning(f"Makro veri yuklenemedi: {exc}")
        return None


def _fetch_fundamental_data(symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Fundamental verileri cek veya yukle (opsiyonel)."""
    try:
        from data.fundamental_fetcher import FundamentalDataFetcher
        fetcher = FundamentalDataFetcher()
        try:
            return fetcher.load_data()
        except FileNotFoundError:
            if symbols:
                return fetcher.fetch_fundamental_data(symbols, save=True)
            return None
    except Exception as exc:
        logger.warning(f"Fundamental veri yuklenemedi: {exc}")
        return None


def _fetch_cross_asset_data(start_date: str = '2018-01-01') -> Optional[pd.DataFrame]:
    """BIST-100 ve USD/TRY capraz varlik verilerini cek."""
    try:
        import yfinance as yf

        cross_data = {}
        for name, symbol in [('bist100', 'XU100.IS'), ('usd_try', 'TRY=X'), ('eur_try', 'EURTRY=X')]:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date)
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                cross_data[name] = hist['Close']

        if cross_data:
            return pd.DataFrame(cross_data).ffill().bfill()
        return None
    except Exception as exc:
        logger.warning(f"Capraz varlik verisi yuklenemedi: {exc}")
        return None


# ------------------------------------------------------------------
# Faz 4: Veri tazelik orkestratoru
# ------------------------------------------------------------------
# Sabah eğit -> gün sonu tahmin akışı için: tahmin günü T olduğunda
# eğitim/tahmin verisi T-1 işgünü kapsamalı. Saat-aware (BIST 18:00,
# US 22:00) hedef tarih hesaplanır; her veri kaynağı incremental
# güncellenir. Sessiz fail kabul edilmez — durum response'a yansır.

_BIST_CLOSE_HOUR_LOCAL = 18  # BIST kapanış (Europe/Istanbul)
_US_CLOSE_HOUR_LOCAL = 22    # ABD kapanış (yaklaşık, yerel saat)


def _previous_business_day(dt: datetime) -> datetime:
    """Verilen tarihten onceki ilk isgunu (hafta sonu atla)."""
    d = dt - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Cmt, 6=Pzr
        d -= timedelta(days=1)
    return d


def _resolve_target_date(asset_type: str, now: Optional[datetime] = None) -> datetime:
    """Saat-aware T-1 isgunu hedefi.

    - stock_bist: yerel saat 18:00+ ise today-1 BDay; oncesinde today-2 BDay
                  (BIST henuz kapanmadan T-1 garantisi yok)
    - gold/fx:    24x5 piyasa, today-1 BDay (saatten bagimsiz)
    - macro:      US ufku icin 22:00+ ise today-1 BDay; oncesinde today-2 BDay

    Hafta sonu cagirildiginda en yakin onceki Cuma'ya duser.
    """
    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Hafta sonu cagirildiysa en yakin Cuma
    if today.weekday() >= 5:
        return _previous_business_day(today + timedelta(days=1))

    if asset_type == 'stock_bist':
        if now.hour >= _BIST_CLOSE_HOUR_LOCAL:
            return _previous_business_day(today + timedelta(days=1))  # = today (BDay)
        return _previous_business_day(today)  # T-1 BDay

    if asset_type == 'macro':
        if now.hour >= _US_CLOSE_HOUR_LOCAL:
            return _previous_business_day(today + timedelta(days=1))
        return _previous_business_day(today)

    # gold, fx: 24x5 piyasa, ama T-1 isgunu hedeflenir
    # (T gunu icinde piyasa hala acik, kapanis verisi yok)
    return _previous_business_day(today)


def _classify_asset(symbol: str) -> str:
    """Sembol -> asset_type. ASSET_PROFILES anahtari."""
    if not is_gold_symbol(symbol):
        return 'stock_bist'
    # Gold/FX ayrımı: USDTRY=X, EURTRY=X => fx; geri kalan => gold
    fx_symbols = {'USDTRY=X', 'EURTRY=X', 'TRY=X'}
    if symbol in fx_symbols:
        return 'fx'
    return 'gold'


def _gold_metrics_for(symbol: str) -> Optional[List[str]]:
    """Verilen output sembol icin GoldFetcher metric listesi."""
    from data.gold_fetcher import OUTPUT_SYMBOL
    metric = next((m for m, out in OUTPUT_SYMBOL.items() if out == symbol), None)
    return [metric] if metric else None


def _refresh_stock_bist(symbol: str, source: Optional[str], target: datetime) -> dict:
    """BIST hisse fiyat verisini incremental guncelle."""
    from data.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    status = fetcher.get_source_status([symbol])
    last = status.get('last_date')
    if last and pd.to_datetime(last).date() >= target.date():
        return {'mode': 'skip', 'last_date': last, 'target': str(target.date())}
    return fetcher.fetch_incremental([symbol])


def _refresh_gold(symbol: str, source: Optional[str], target: datetime) -> dict:
    """Gold (USD/TRY/ons/gram) fiyat verisini incremental guncelle."""
    from data.gold_fetcher import GoldFetcher
    resolved = _resolve_source(symbol, source) or 'yfinance'
    metrics = _gold_metrics_for(symbol)
    fetcher = GoldFetcher(source=resolved, metrics=metrics)
    status = fetcher.get_source_status()
    last = status.get('last_date')
    if last and pd.to_datetime(last).date() >= target.date():
        return {'mode': 'skip', 'last_date': last, 'target': str(target.date()),
                'source': resolved}
    return {**fetcher.fetch_incremental(), 'source': resolved}


def _refresh_fx(symbol: str, source: Optional[str], target: datetime) -> dict:
    """FX (USDTRY, EURTRY) — Gold pipeline'ini kullanir (ayni fetcher)."""
    return _refresh_gold(symbol, source, target)


def _refresh_macro(target: datetime) -> dict:
    """Makro veriyi incremental guncelle (yfinance + EVDS)."""
    try:
        from data.macro_fetcher import MacroDataFetcher
        fetcher = MacroDataFetcher()
        status = fetcher.get_source_status()
        last = status.get('last_date')
        if last and pd.to_datetime(last).date() >= target.date():
            return {'mode': 'skip', 'last_date': last, 'target': str(target.date())}
        return fetcher.fetch_incremental()
    except Exception as exc:
        logger.warning(f"Makro tazeleme basarisiz: {exc}")
        return {'mode': 'error', 'error': str(exc)}


def _refresh_fundamental(symbol: str, asset_type: str,
                         max_staleness_days: int = 30) -> dict:
    """Fundamental verisi (sirket bilanco) — sadece BIST hisse, 30g esigi."""
    if asset_type != 'stock_bist':
        return {'mode': 'n/a', 'reason': f'asset_type={asset_type} icin sirket fundamentali yok'}
    try:
        from data.fundamental_fetcher import FundamentalDataFetcher
        fetcher = FundamentalDataFetcher()
        # Basit yas kontrolu: load deneyip CSV mtime'a bak
        import os
        filepath = os.path.join('data', 'fundamental', 'fundamental_data.csv')
        if os.path.exists(filepath):
            age_days = (datetime.now()
                        - datetime.fromtimestamp(os.path.getmtime(filepath))).days
            if age_days < max_staleness_days:
                return {'mode': 'skip', 'age_days': age_days}
        df = fetcher.fetch_fundamental_data([symbol], save=True)
        return {'mode': 'refresh', 'rows': len(df) if df is not None else 0}
    except Exception as exc:
        logger.warning(f"Fundamental tazeleme basarisiz: {exc}")
        return {'mode': 'error', 'error': str(exc)}


# Asset tipi profilleri — yeni varlik tipi eklemek icin tek nokta
ASSET_PROFILES: Dict[str, Dict[str, Any]] = {
    'stock_bist': {
        'price_refresher': _refresh_stock_bist,
        'needs_company_fundamental': True,
        'needs_macro_tr': True,
        'needs_macro_global': True,
    },
    'gold': {
        'price_refresher': _refresh_gold,
        'needs_company_fundamental': False,  # emtia — sirket fundamentali yok
        'needs_macro_tr': True,   # GOLD_GRAM_TRY vb. icin TRY bagi
        'needs_macro_global': True,  # reel faiz, DXY, oil — altin icin kritik
    },
    'fx': {
        'price_refresher': _refresh_fx,
        'needs_company_fundamental': False,
        'needs_macro_tr': True,
        'needs_macro_global': True,
    },
}


def ensure_fresh_data(
    symbol: str,
    source: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Egitim/tahmin oncesi tum veri kaynaklarini T-1 isgunu hedefine guncelle.

    Bir sembol icin:
      1. Asset tipini belirle
      2. Hedef tarih = T-1 BDay (saat-aware)
      3. Fiyat verisini incremental cek (asset adapter)
      4. Makro veriyi incremental cek
      5. Fundamental (varsa) yas kontrolu yap

    Returns:
        {asset_type, target_date, price, macro, fundamental}
    """
    asset_type = _classify_asset(symbol)
    profile = ASSET_PROFILES.get(asset_type)
    if profile is None:
        return {'asset_type': asset_type, 'error': f'unknown asset_type'}

    target = _resolve_target_date(asset_type, now=now)
    macro_target = _resolve_target_date('macro', now=now)
    logger.info(f"ensure_fresh_data | symbol={symbol} | asset={asset_type} | target={target.date()}")

    report: Dict[str, Any] = {
        'symbol': symbol,
        'source': source,
        'asset_type': asset_type,
        'target_date': str(target.date()),
        'macro_target_date': str(macro_target.date()),
    }

    try:
        report['price'] = profile['price_refresher'](symbol, source, target)
    except Exception as exc:
        logger.error(f"Fiyat tazeleme basarisiz: {exc}")
        report['price'] = {'mode': 'error', 'error': str(exc)}

    if profile.get('needs_macro_tr') or profile.get('needs_macro_global'):
        report['macro'] = _refresh_macro(macro_target)
    else:
        report['macro'] = {'mode': 'n/a'}

    report['fundamental'] = _refresh_fundamental(
        symbol, asset_type,
    ) if profile.get('needs_company_fundamental') else {'mode': 'n/a'}

    return report


# ------------------------------------------------------------------
# Arka plan egitim durum takibi
# ------------------------------------------------------------------
# Basit in-memory store:
#   {(user_id, symbol, horizon, source): {state, started_at, finished_at, error, result}}
# Faz 7: anahtara kullanici eklendi — bir kullanicinin egitimi digerinin
# durum ekraninda gorunmez ve ayni sembolu es zamanli egitmelerini engellemez.
# Tek islemli uvicorn icin yeterli; cok-worker kurulumda Redis/DB'ye tasinmali.
_training_state: Dict[tuple, Dict[str, Any]] = {}


def _current_user_key(user_id: Optional[str] = None) -> str:
    if user_id:
        return user_id
    try:
        from app.auth import workspace as ws

        return ws.current_user_id() or 'local'
    except Exception:
        return 'local'


def get_training_state(
    symbol: Optional[str] = None,
    horizon: Optional[str] = None,
    source: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Egitim durumlarini raporla (yalnizca aktif kullaniciya ait kayitlar).

    symbol+horizon (+source) verilirse tek kayit, yoksa tumu.
    """
    uid = _current_user_key(user_id)
    own = {k: v for k, v in _training_state.items() if k[0] == uid}

    if symbol and horizon:
        # source verilmediyse en yenisi (varsa) doner
        if source is not None:
            return own.get((uid, symbol, horizon, source))
        # source belirtilmediyse: o symbol+horizon icin running olan varsa onu, yoksa en son baslayan
        candidates = [
            (k, v) for k, v in own.items()
            if k[1] == symbol and k[2] == horizon
        ]
        if not candidates:
            return None
        running = [(k, v) for k, v in candidates if v.get('state') == 'running']
        if running:
            return running[0][1]
        return max(candidates, key=lambda kv: kv[1].get('started_at', ''))[1]
    return [
        {'symbol': s, 'horizon': h, 'source': src, **info}
        for (_u, s, h, src), info in own.items()
    ]


class PredictionService:
    """Gelismis tahmin pipeline'i servis katmani.

    Ensemble (XGBoost + LightGBM + CatBoost + BiLSTM + TFT) tabanli tahmin.
    Eski API'yi koruyarak geriye uyumlu calisir.
    """

    def train_model_async(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01', test_ratio: float = 0.2,
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
        user_id: Optional[str] = None,
    ) -> None:
        """Arka plan egitimi: BackgroundTasks ile cagrilir, sonucu _training_state'e yazar.

        user_id: egitimi baslatan kullanici. Arka plan gorevi istek baglamini
        devralmadigi icin calisma alani burada acikca kurulur — model dosyalari
        dogru kullanicinin dizinine yazilir.
        """
        from app.auth import workspace as ws

        resolved = _resolve_source(symbol, source)
        uid = _current_user_key(user_id)
        key = (uid, symbol, horizon, resolved)
        _training_state[key] = {
            'state': 'running',
            'source': resolved,
            'started_at': datetime.now().isoformat(),
            'finished_at': None,
            'error': None,
            'result': None,
        }
        try:
            with ws.use_workspace(user_id):
                result = self.train_model(
                    symbol, horizon,
                    start_date=start_date, test_ratio=test_ratio,
                    source=resolved,
                    feature_groups=feature_groups,
                    target_type=target_type,
                )
            _training_state[key].update({
                'state': 'completed',
                'finished_at': datetime.now().isoformat(),
                'result': result,
            })
        except Exception as exc:
            logger.error(f"[{symbol} {horizon} {resolved}] Arka plan egitimi basarisiz: {exc}", exc_info=True)
            _training_state[key].update({
                'state': 'error',
                'finished_at': datetime.now().isoformat(),
                'error': str(exc),
            })

    def train_model(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01', test_ratio: float = 0.2,
        optimize: bool = False, n_hpo_trials: int = 30,
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
    ) -> Dict[str, Any]:
        """Sembol icin ensemble modeli egit."""
        from prediction.trainer import WalkForwardTrainer

        resolved = _resolve_source(symbol, source)
        logger.info(f"Ensemble egitimi basliyor: {symbol} {horizon} source={resolved}")

        df = _fetch_symbol_data(symbol, start_date, source=resolved)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        trainer = WalkForwardTrainer(
            horizon=horizon,
            source=resolved,
            feature_groups=feature_groups,
            target_type=target_type,
        )
        result = trainer.train_final_models(
            df, symbol,
            test_ratio=test_ratio,
            optimize_first=optimize,
            n_hpo_trials=n_hpo_trials,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )

        logger.info(f"  Egitim tamamlandi: {symbol} {horizon} source={resolved}")

        suffix = f'__{resolved}' if resolved else ''
        test_metrics = result.get('ensemble_test_metrics', {})
        return {
            'symbol': result.get('symbol', symbol),
            'source': resolved,
            'horizon': result.get('horizon', horizon),
            'n_train': result.get('n_train', 0),
            'n_test': result.get('n_test', 0),
            'n_features': result.get('n_features', 0),
            'train_metrics': test_metrics,
            'test_metrics': test_metrics,
            'model_path': f'models/prediction/{symbol}{suffix}_{horizon}_ensemble',
            'trained_at': result.get('trained_at', datetime.now().isoformat()),
            'n_models': result.get('n_models', 0),
            'models_trained': result.get('models_trained', []),
            'model_results': result.get('model_results', {}),
            'ensemble_test_metrics': test_metrics,
            'training_time_seconds': result.get('training_time_seconds'),
        }

    def train_batch(
        self, symbols: List[str], horizon: str = 'daily',
        start_date: str = '2018-01-01',
        optimize: bool = False, n_hpo_trials: int = 30,
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
        resume_from: Optional[str] = None,
        strict: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Faz 6 (G.3+G.4): Cok-sembol batch egitim + manifest + resume.

        Her sembol tek tek egitilir; hata bir sembolu dusurur ama batch'i
        durdurmaz (strict=False). Sonuc yapilandirilmis bir manifest'e yazilir
        (results/training_runs/<run_id>.json) — gece batch'i sabah tek bakista
        teshis edilir (G.3).

        Resume (G.4): resume_from bir onceki manifest yolu/run_id ise, orada
        'ok' biten semboller ATLANIR; sadece kalanlar egitilir. Uzun batch
        ortada kesilirse bastan baslamaz.

        Args:
            symbols: Egitilecek sembol listesi.
            resume_from: Onceki manifest dosya yolu VEYA run_id. Verilirse o
                kosumda basariyla tamamlanan semboller atlanir.
            strict: True ise ilk basarisiz sembolde batch durur (fail-fast).
                Varsayilan False — dayanikli batch (hatali sembol atlanir).
            user_id: Faz 7 — batch bir arka plan gorevinden calisiyorsa
                egitimi baslatan kullanici. Verilirse modeller ve manifest o
                kullanicinin calisma alanina yazilir. None = mevcut baglam
                korunur (CLI betiginde baglam yok -> ortak dizinler).

        Returns:
            {'manifest_path', 'run_id', 'summary', 'skipped', 'trained', 'failed'}
        """
        if user_id:
            from app.auth import workspace as ws
            with ws.use_workspace(user_id):
                return self._train_batch(
                    symbols, horizon=horizon, start_date=start_date,
                    optimize=optimize, n_hpo_trials=n_hpo_trials, source=source,
                    feature_groups=feature_groups, target_type=target_type,
                    resume_from=resume_from, strict=strict,
                )
        return self._train_batch(
            symbols, horizon=horizon, start_date=start_date,
            optimize=optimize, n_hpo_trials=n_hpo_trials, source=source,
            feature_groups=feature_groups, target_type=target_type,
            resume_from=resume_from, strict=strict,
        )

    def _train_batch(
        self, symbols: List[str], horizon: str = 'daily',
        start_date: str = '2018-01-01',
        optimize: bool = False, n_hpo_trials: int = 30,
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
        resume_from: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """train_batch govdesi — calisma alani baglami cagiran tarafta kurulur."""
        import time as _time
        from prediction.manifest import TrainingManifest

        # --- Resume: onceki manifest'ten biten sembolleri cikart ---
        already_done: set = set()
        if resume_from:
            prev = self._load_manifest(resume_from)
            if prev:
                already_done = {
                    s for s, d in prev.get('symbols', {}).items()
                    if d.get('status') == 'ok'
                }
                logger.info(
                    f"[batch] Resume: onceki kosumda {len(already_done)} sembol "
                    f"'ok' bitmis, atlanacak."
                )
            else:
                logger.warning(f"[batch] Resume kaynagi bulunamadi: {resume_from}")

        pending = [s for s in symbols if s not in already_done]
        skipped = [s for s in symbols if s in already_done]

        manifest = TrainingManifest(
            run_kind='batch_train', symbols=symbols, horizon=horizon,
            extra={
                'start_date': start_date, 'target_type': target_type,
                'optimize': optimize, 'resume_from': resume_from,
                'skipped_on_resume': skipped,
            },
        )

        # Veri kalite bayragi (G.2): makro frame'in attrs'inden al — batch'te
        # bir kez cekilir, tum sembollerde ayni makro kullanilir.
        try:
            macro_dq = _fetch_macro_data(start_date)
            if macro_dq is not None and hasattr(macro_dq, 'attrs'):
                manifest.set_data_quality(macro_dq.attrs.get('data_quality'))
        except Exception as exc:
            logger.debug(f"[batch] data_quality okunamadi: {exc}")

        trained, failed = [], []
        n_workers = self._parallel_workers(len(pending))

        def _train_one(sym: str) -> tuple:
            """Tek sembol egitimi — (sembol, sonuc, hata, sure)."""
            t0 = _time.perf_counter()
            try:
                res = self.train_model(
                    sym, horizon,
                    start_date=start_date, optimize=optimize,
                    n_hpo_trials=n_hpo_trials, source=source,
                    feature_groups=feature_groups, target_type=target_type,
                )
                return sym, res, None, _time.perf_counter() - t0
            except Exception as exc:
                logger.error(f"[batch] {sym} basarisiz: {exc}", exc_info=True)
                return sym, None, str(exc), _time.perf_counter() - t0

        def _record(sym: str, res, err, seconds) -> str:
            """Manifest'e yaz + checkpoint. Cagiran taraf sirali olmali."""
            if err is not None:
                manifest.record_symbol(sym, error=err, seconds=seconds)
                failed.append(sym)
            else:
                manifest.record_symbol(sym, result=res, seconds=seconds)
                status = manifest.symbols[sym]['status']
                (trained if status != 'failed' else failed).append(sym)
                if status == 'degraded':
                    logger.warning(f"[batch] {sym} DEGRADED: eksik modeller "
                                   f"{manifest.symbols[sym].get('missing_models')}")
            # Her sembolden sonra manifest'i diske yaz (checkpoint) — kesinti
            # olsa bile o ana kadarki ilerleme resume icin kalir.
            manifest.finalize()
            return manifest.symbols[sym]['status']

        if n_workers > 1:
            # Epic 2.2 (B3): semboller birbirinden bagimsiz -> is parcaciklari.
            # Process yerine thread: tek CUDA baglami (VRAM cogalmaz), veri
            # pickle'lanmaz; agac modelleri ve torch zaten GIL'i birakir. DL
            # adimi ayrica DL_GPU_SLOTS yuvasiyla sinirlanir.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            from app.auth import workspace as ws

            # ContextVar is parcacigi sinirini gecmez -> aktif kullaniciyi
            # acikca tasi, yoksa modeller yanlis calisma alanina yazilir.
            current = ws.get_current_user()

            def _worker(sym: str):
                if current:
                    with ws.use_workspace(current.get('id'), current.get('email', ''),
                                          current.get('role', 'user')):
                        return _train_one(sym)
                return _train_one(sym)

            logger.info(f"[batch] Paralel mod: {n_workers} is parcacigi, "
                        f"{len(pending)} sembol.")
            done = 0
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker, s): s for s in pending}
                for fut in as_completed(futures):
                    sym, res, err, seconds = fut.result()
                    done += 1
                    logger.info(f"[batch] ({done}/{len(pending)}) {sym} bitti "
                                f"({seconds:.1f}s)")
                    # Manifest tek is parcaciginda (ana) guncellenir -> kilit gerekmez.
                    status = _record(sym, res, err, seconds)
                    if strict and status == 'failed':
                        logger.error("[batch] strict mod: hata sonrasi kalan "
                                     "semboller iptal ediliyor.")
                        for f in futures:
                            f.cancel()
                        break
        else:
            for i, sym in enumerate(pending, 1):
                logger.info(f"[batch] ({i}/{len(pending)}) {sym} egitiliyor...")
                sym, res, err, seconds = _train_one(sym)
                status = _record(sym, res, err, seconds)
                if strict and status == 'failed':
                    logger.error("[batch] strict mod: ilk hatada durduruluyor.")
                    break

        path = manifest.finalize()
        summary = manifest._summary()
        logger.info(
            f"[batch] Bitti. ok={summary['ok']} degraded={summary['degraded']} "
            f"failed={summary['failed']} skipped(resume)={len(skipped)}. Manifest: {path}"
        )
        return {
            'manifest_path': path,
            'run_id': manifest.run_id,
            'summary': summary,
            'skipped': skipped,
            'trained': trained,
            'failed': failed,
        }

    @staticmethod
    def _parallel_workers(n_pending: int) -> int:
        """Epic 2.2: es zamanli sembol sayisi (1 = seri, varsayilan).

        Sembol sayisindan fazla is parcacigi acmanin faydasi yok; kucuk
        batch'lerde paralellik masrafi kazanci yiyebilir.
        """
        try:
            from app.core.config import get_settings

            configured = int(getattr(get_settings(), 'TRAIN_PARALLEL_SYMBOLS', 1) or 1)
        except Exception:
            configured = 1
        return max(1, min(configured, max(1, n_pending)))

    @staticmethod
    def _load_manifest(ref: str) -> Optional[Dict[str, Any]]:
        """Manifest'i dosya yolu veya run_id'den yukle. G.4 resume yardimcisi.

        Arama sirasi (Faz 7): once aktif kullanicinin calisma alani, sonra
        kullanici oncesi ortak dizin — eski kosumlar da surdurulebilir kalir.
        """
        import json
        from prediction.manifest import find_manifest
        path = find_manifest(ref)
        if not path:
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Manifest okunamadi ({path}): {exc}")
            return None

    def predict(
        self, symbols: List[str], horizon: str = 'daily',
        save: bool = True,
        source: Optional[str] = None,
        targets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Sembol listesi icin tahmin uret.

        Args:
            symbols: Tahmin yapilacak sembol listesi (basit kullanim)
            horizon: 'daily' veya 'weekly'
            save: Tracker'a yaz
            source: Tum sembollerde ayni source kullan (ops.); gold disinda yoksayilir
            targets: Detayli kullanim — her oge {symbol, source} icerir.
                     Verilirse symbols+source goz ardi edilir.
        """
        from prediction.tracker import PredictionTracker

        tracker = PredictionTracker()

        # Hedef listesini normalize et
        if targets:
            jobs = [{'symbol': t['symbol'], 'source': t.get('source')} for t in targets]
        else:
            jobs = [{'symbol': s, 'source': source} for s in symbols]

        predictions = []
        today = datetime.now()
        start = (today - timedelta(days=365)).strftime('%Y-%m-%d')

        macro_df = _fetch_macro_data(start)
        cross_asset_df = _fetch_cross_asset_data(start)
        fundamental_df = _fetch_fundamental_data([j['symbol'] for j in jobs])

        for job in jobs:
            symbol = job['symbol']
            try:
                resolved = _resolve_source(symbol, job.get('source'))
                predictor = _get_cached_predictor(horizon, symbol, resolved)

                if not predictor.is_trained(symbol):
                    logger.warning(
                        f"  {symbol} (source={resolved}) icin egitilmis model yok, atlaniyor"
                    )
                    continue

                df = _fetch_symbol_data(symbol, start, source=resolved)
                pred = predictor.predict_next(
                    df, symbol,
                    macro_df=macro_df,
                    fundamental_df=fundamental_df,
                    cross_asset_df=cross_asset_df,
                )
                pred['source'] = resolved
                predictions.append(pred)

                if save:
                    tracker.store_prediction(pred)

            except Exception as exc:
                logger.error(f"  {symbol} tahmini basarisiz: {exc}")

        return predictions

    def cross_validate(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01',
        model_types: Optional[List[str]] = None,
        n_splits: int = 5,
        source: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Walk-forward cross-validation ile model degerlendirme.

        strict=True (Faz 6 R2): bir fold patlarsa fail-fast. Varsayilan False —
        fold atlanir ama sonuc `status='degraded'` + `failed_folds` tasir.
        """
        from prediction.trainer import WalkForwardTrainer

        resolved = _resolve_source(symbol, source)
        df = _fetch_symbol_data(symbol, start_date, source=resolved)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        trainer = WalkForwardTrainer(
            horizon=horizon,
            n_splits=n_splits,
            source=resolved,
            strict=strict,
        )
        return trainer.cross_validate(
            df, symbol,
            model_types=model_types,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )

    def optimize_hyperparameters(
        self, symbol: str, horizon: str = 'daily',
        start_date: str = '2018-01-01',
        model_types: Optional[List[str]] = None,
        n_trials: int = 50,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tum modeller icin hiperparametre optimizasyonu."""
        from prediction.hyperopt import PredictionHyperOptimizer
        from prediction.feature_engineer import PredictionFeatureEngineer
        import numpy as np

        resolved = _resolve_source(symbol, source)
        df = _fetch_symbol_data(symbol, start_date, source=resolved)
        macro_df = _fetch_macro_data(start_date)
        fundamental_df = _fetch_fundamental_data([symbol])
        cross_asset_df = _fetch_cross_asset_data(start_date)

        fe = PredictionFeatureEngineer(horizon)
        feat_df = fe.build_features(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
        )
        feature_cols = fe.get_feature_columns(feat_df)
        X = feat_df[feature_cols].values.astype(np.float32)
        y = feat_df['target_price'].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer = PredictionHyperOptimizer(n_trials=n_trials)
        return optimizer.optimize_all_models(X, y, horizon, model_types)

    def evaluate_pending(self, symbols: Optional[List[str]] = None) -> int:
        """Bekleyen tahminleri degerlendir."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.evaluate_pending(symbols)

    def get_performance(
        self, symbol: str, horizon: str = 'daily', window: int = 30
    ) -> Dict[str, Any]:
        """Sembol icin performans metrikleri."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_rolling_metrics(symbol, horizon, window)

    def get_summary(self) -> List[Dict[str, Any]]:
        """Tum semboller icin ozet."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        return tracker.get_summary()

    def list_trained_models(self) -> List[Dict[str, Any]]:
        """Egitilmis modelleri listele.

        `prediction/models/` paketi eksikse (bilinen eksiklik, bkz.
        prediction/__init__.py) bos liste doner — tahmin altsistemi devre disi
        olsa da bu endpoint 500 yerine "egitilmis model yok" anlamiyla yanit
        verir.
        """
        try:
            from prediction.models import PricePredictor
        except ImportError:
            logger.warning(
                "prediction.models paketi yok — list_trained_models() bos liste donduruyor."
            )
            return []
        models = []
        for horizon in ('daily', 'weekly'):
            p = PricePredictor(horizon)
            models.extend(p.list_trained_models())
        return models

    def get_prediction_history(
        self, symbol: str, horizon: str = 'daily'
    ) -> Dict[str, Any]:
        """Sembol icin tahmin gecmisi."""
        from prediction.tracker import PredictionTracker
        tracker = PredictionTracker()
        raw = tracker.get_prediction_history(symbol)

        history = []
        for pred_date, horizons in sorted(raw.items()):
            entry = horizons.get(horizon)
            if entry:
                history.append({'prediction_date': pred_date, **entry})

        return {
            'symbol': symbol,
            'horizon': horizon,
            'history': history,
            'total': len(history),
        }

    def get_chart_data(
        self, symbol: str, horizon: str = 'daily', days: int = 30
    ) -> Dict[str, Any]:
        """Tahmin vs gercek chart verisi."""
        from prediction.tracker import PredictionTracker
        from prediction.evaluator import PredictionEvaluator
        tracker = PredictionTracker()
        history = tracker.get_performance_history(symbol, horizon, days)
        chart = PredictionEvaluator.build_prediction_chart_data(history)
        trend = PredictionEvaluator.build_accuracy_trend_data(history)
        return {'chart': chart, 'trend': trend}
