"""
Daily Trading Service
Helper functions for daily trading decisions
"""

import os
import json
import logging
import threading
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from filelock import FileLock

from data.data_fetcher import to_naive_market_dates
from data.technical_indicators import add_indicators_to_multi_symbol_df
from app.auth import workspace as ws

logger = logging.getLogger(__name__)


# ==================== RISK PARAMETERS ====================

def get_risk_parameters(risk_mode: str) -> dict:
    """
    Get risk parameters based on risk mode

    Args:
        risk_mode: 'conservative', 'moderate', or 'aggressive'

    Returns:
        Dictionary with risk parameters
    """
    params = {
        "conservative": {
            "min_signal_threshold": 0.5,
            "max_position_pct": 0.20,
            "max_daily_trades": 2,
            "description": "Konservatif - Yüksek güven eşiği, düşük pozisyon limiti"
        },
        "moderate": {
            "min_signal_threshold": 0.3,
            "max_position_pct": 0.30,
            "max_daily_trades": 3,
            "description": "Orta - Dengeli risk/getiri"
        },
        "aggressive": {
            "min_signal_threshold": 0.1,
            "max_position_pct": 0.40,
            "max_daily_trades": 5,
            "description": "Agresif - Düşük güven eşiği, yüksek pozisyon limiti"
        }
    }

    return params.get(risk_mode, params["moderate"])


# ==================== TRADE UNIVERSE RESOLUTION ====================

# Egitilmis modelin yanina yazilan tanim dosyasi. `.zip` ile ayni koke sahip.
MODEL_META_SUFFIX = ".meta.json"


def model_meta_path(model_path: str) -> str:
    """`<model>.zip` yolundan `<model>.meta.json` yolunu uret."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + MODEL_META_SUFFIX


def write_model_meta(model_path: str, meta: dict) -> Optional[str]:
    """Model tanimini yan dosyaya yaz. Basarisizlik egitimi dusurmez."""
    path = model_meta_path(model_path)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        logger.info(f"Model tanimi yazildi: {os.path.basename(path)} "
                    f"({meta.get('n_symbols')} sembol)")
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Model tanimi yazilamadi ({exc}) — egitim etkilenmedi")
        return None


def read_model_meta(model_path: str) -> Optional[dict]:
    """Yan dosyayi oku; yoksa veya bozuksa None."""
    path = model_meta_path(model_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Model tanimi okunamadi ({path}): {exc}")
        return None


def _panel_train_symbols() -> List[str]:
    """Egitim rotasinin kullandigi paneli birebir yeniden uret.

    Egitim `stock_data_with_indicators.csv`'yi yukler, `split_data()` ile
    KRONOLOJIK boler ve modelin evrenini EGITIM boluminin sembolleri belirler.
    Sembol sirasi CSV'deki gorunme sirasidir — durum vektorunun dizilimi de
    bu, dolayisiyla `sorted()` yapmak YANLIS olurdu.
    """
    from data.data_fetcher import DataFetcher

    fetcher = DataFetcher()
    df = fetcher.load_data('stock_data_with_indicators.csv')
    train_df, _, _ = fetcher.split_data(df)
    return train_df.index.get_level_values('symbol').unique().tolist()


def resolve_trade_universe(model, model_path: str, model_name: str) -> Tuple[List[str], str]:
    """Modelin EGITILDIGI sembol evrenini, egitimdeki SIRAYLA dondur.

    Neden model adindan turetilemiyor: egitim rotasi `get_symbols(phase)`
    listesini yalnizca veri cekerken kullanir, egitimi yuklenen panelin
    tamamiyla yapar. Yani "phase1" adli bir model pekala 30 sembolle
    egitilmis olabilir (gozlem uzayi 331). Ada guvenmek durum vektorunu
    56 uzunlugunda uretiyordu ve SB3 `predict` asamasinda
    "Unexpected observation shape (56,) ... please use (331,)" ile patliyordu.

    Tek gercek kaynak modelin kendisi: `action_space` hisse basina bir eylem
    tasir, yani uzunlugu sembol sayisidir. Adaylar bu sayiya gore elenir.

    Returns:
        (semboller, kaynak_etiketi)
    """
    action_shape = getattr(model.action_space, "shape", None)
    n_expected = int(action_shape[0]) if action_shape else 0
    if n_expected <= 0:
        raise ValueError("Modelin eylem uzayi okunamadi; sembol sayisi bilinmiyor")

    # 1) Yan dosya — modelle birlikte yazildigi icin en guvenilir kaynak.
    meta = read_model_meta(model_path)
    if meta:
        symbols = [str(x) for x in (meta.get("symbols") or [])]
        if len(symbols) == n_expected:
            return symbols, "model tanim dosyasi"
        if symbols:
            logger.warning(
                f"Model tanimindaki sembol sayisi ({len(symbols)}) modelin eylem "
                f"uzayiyla ({n_expected}) uyusmuyor; tanim dosyasi yok sayildi"
            )

    # 2) Egitim panelini yeniden uret (yan dosyasi olmayan eski modeller).
    try:
        panel = _panel_train_symbols()
        if len(panel) == n_expected:
            return panel, "egitim paneli (yeniden uretildi)"
        logger.warning(
            f"Egitim paneli {len(panel)} sembol veriyor ama model {n_expected} "
            f"bekliyor; panel yok sayildi"
        )
    except FileNotFoundError:
        logger.info("stock_data_with_indicators.csv yok; panel yolu atlandi")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Egitim paneli okunamadi ({exc}); panel yolu atlandi")

    # 3) Ad tabanli sabitler — yalnizca sayi tutuyorsa (eski davranis).
    from data.bist30_symbols import BIST30_SYMBOLS, PHASE1_SYMBOLS, PHASE3_SYMBOLS

    lower = model_name.lower()
    candidates = [
        (list(PHASE3_SYMBOLS), "PHASE3_SYMBOLS"),
        (list(PHASE1_SYMBOLS), "PHASE1_SYMBOLS"),
        (list(BIST30_SYMBOLS), "BIST30_SYMBOLS"),
    ]
    if "phase3" not in lower:
        candidates = candidates[1:] + candidates[:1]
    for symbols, label in candidates:
        if len(symbols) == n_expected:
            return symbols, f"model adi ({label})"

    raise ValueError(
        f"Model {n_expected} sembolle egitilmis ama bu evren cozumlenemedi. "
        f"Model tanim dosyasi ({os.path.basename(model_meta_path(model_path))}) yok ve "
        f"egitim paneli farkli sayida sembol iceriyor. Modeli yeniden egitin "
        f"(yeni modeller tanim dosyasini kendileri yazar) ya da Veri sayfasindan "
        f"paneli egitimdeki haline getirin."
    )


# ==================== PANEL TAZELIGI VE ONBELLEK ====================

# Ayni panel dakikalar icinde defalarca isteniyor: kullanici formu yeniden
# gonderiyor, pano callback'i tetikleniyor. TTL boyunca ayni sonucu don —
# 30 yfinance cagrisi da gosterge hesabi da tekrarlanmasin.
# Bir seansin "gecerli" sayilmasi icin gereken sembol kapsami. 1.0 fazla
# kati olurdu: borsaya sonradan giren sembolun eski gunlerde satiri hic
# yoktur ve panel o gunlerde asla tam dolmaz.
MIN_SESSION_COVERAGE = 0.9

_PANEL_CACHE_TTL_SEC = 900  # 15 dk
_panel_cache: Dict[tuple, Tuple[float, pd.DataFrame]] = {}
_panel_cache_lock = threading.Lock()

RAW_CSV_PATH = os.path.join("data", "bist", "raw_stock_data.csv")


def _raw_csv_mtime() -> float:
    """Onbellek anahtarina CSV surumunu koy.

    Veri sayfasindan yeni veri indirilince dosyanin mtime'i degisir, anahtar
    duser ve panel TTL beklemeden tazelenir.
    """
    try:
        return os.path.getmtime(RAW_CSV_PATH)
    except OSError:
        return 0.0


def last_expected_session(target_date):
    """Hedef tarihte veya oncesinde KAPANMIS olmasi beklenen son seans gunu.

    Tazelik olcutu takvim gunune baglanamaz. Hedef 30 Agustos 2026 (Pazar)
    iken CSV 28 Agustos Cuma'ya kadar DOLU olsa bile `cached_last < target`
    dogru cikiyordu; sonucta 30 sembolun HEPSI yfinance'ten yeniden
    indiriliyordu — hafta sonu ve tatil boyunca her karar isteginde, panel hic
    degismeden. Olculdu: tek karar istegi = 30 CSV okumasi + 30 indirme.

    Hafta sonunu geriye sararak son is gunune iniyoruz. Resmi BIST tatilleri
    (30 Agustos, bayramlar) burada bilinmiyor; o gunlerde tek bir bosuna tur
    atilir ve TTL onbellegi tekrarini keser.
    """
    d = target_date.date() if isinstance(target_date, datetime) else target_date
    while d.weekday() >= 5:  # 5=Cumartesi, 6=Pazar
        d -= timedelta(days=1)
    return d


def _panel_cache_get(key):
    with _panel_cache_lock:
        hit = _panel_cache.get(key)
        if not hit:
            return None
        ts, df = hit
        if time.time() - ts > _PANEL_CACHE_TTL_SEC:
            _panel_cache.pop(key, None)
            return None
        return df


def _panel_cache_put(key, df) -> None:
    with _panel_cache_lock:
        _panel_cache[key] = (time.time(), df)


def clear_panel_cache() -> None:
    """Testler ve veri yenileme sonrasi elle dusurme icin."""
    with _panel_cache_lock:
        _panel_cache.clear()


# ==================== MARKET DATA FETCHING ====================

async def fetch_latest_market_data(
    symbols: List[str],
    target_date: str,
    lookback_days: int = 30
) -> pd.DataFrame:
    """
    Fetch latest market data for given symbols
    Uses the same proven pattern as data_fetcher.py

    Args:
        symbols: List of stock symbols (e.g., ['ASELS.IS', 'THYAO.IS'])
        target_date: Target date in YYYY-MM-DD format
        lookback_days: Number of days to look back

    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    logger.info(f"Fetching market data for {len(symbols)} symbols, target: {target_date}")

    try:
        # Calculate date range - add extra buffer for technical indicators
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_days + 30)

        # Normalize tickers up front (BIST `.IS` suffix).
        norm_symbols = [s if "." in s else f"{s}.IS" for s in symbols]

        # Ayni panel + ayni hedef gun + ayni CSV surumu => ayni sonuc.
        cache_key = (tuple(norm_symbols), target_date, lookback_days,
                     _raw_csv_mtime())
        cached_panel = _panel_cache_get(cache_key)
        if cached_panel is not None:
            logger.info(
                f"Panel onbellekten geldi ({len(norm_symbols)} sembol, "
                f"hedef {target_date}) — indirme yapilmadi"
            )
            return cached_panel

        all_data: Dict[str, pd.DataFrame] = {}

        # 1) CSV-first: try the local cache; covers the common case where the
        # `Veri` page already pulled fresh data this morning. yfinance is only
        # used when the CSV is missing, lacks a symbol, or doesn't cover the
        # required date range.
        try:
            from data.data_fetcher import DataFetcher
            cached = DataFetcher().load_data('raw_stock_data.csv')
            idx_dates = cached.index.get_level_values('date')
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            if idx_dates.tz is not None:
                start_ts = start_ts.tz_localize(idx_dates.tz)
                end_ts = end_ts.tz_localize(idx_dates.tz)

            for symbol in norm_symbols:
                if symbol not in cached.index.get_level_values('symbol').unique():
                    continue
                sub = cached.xs(symbol, level='symbol')
                sub = sub[(sub.index >= start_ts) & (sub.index <= end_ts)]
                if sub.empty:
                    continue
                # Match the yfinance Ticker.history() shape: lowercase OHLCV cols
                sub = sub[['open', 'high', 'low', 'close', 'volume']]
                # yfinance seans kapanmadan once OHLC'si NaN, volume'u DOLU bir
                # taslak satir dondurur ve bu satir CSV'ye yazilabiliyor
                # (28.08.2026: 30 sembolun tamami boyle). yfinance yolunda ayni
                # temizlik asagida zaten var; CSV yolunda olmayinca panelde
                # tamamen bos bir gun kaliyordu. `actual_date` o gunu seciyor,
                # sonra her sembol tek tek bir onceki gune dusuyordu — yani
                # tazelenebilen semboller 28'inin, digerleri 27'nin fiyatiyla
                # AYNI durum vektorune giriyordu.
                before_rows = len(sub)
                sub = sub[sub['close'].notna() & (sub['close'] > 0)]
                dropped = before_rows - len(sub)
                if dropped:
                    logger.info(f"  {symbol}: CSV'de {dropped} gecersiz-close "
                                f"satiri atlandi")
                if sub.empty:
                    continue
                all_data[symbol] = sub
                logger.info(f"  📂 {symbol}: {len(sub)} rows from CSV "
                            f"({sub.index[0].date()} → {sub.index[-1].date()})")
        except FileNotFoundError:
            logger.info("Cached CSV not found; using yfinance only")
        except Exception as exc:
            logger.warning(f"CSV read failed ({exc}); falling back to yfinance")

        # 2) yfinance fallback for any symbol still missing or with stale data.
        # Olcut hedef TAKVIM gunu degil, kapanmis olmasi beklenen son SEANS:
        # aksi halde hafta sonu/tatil boyunca panelin tamami bosuna indiriliyor
        # (bkz. last_expected_session docstring'i).
        expected_session = last_expected_session(end_date)
        if expected_session != end_date.date():
            logger.info(
                f"{end_date.date()} seans gunu degil; tazelik olcutu "
                f"{expected_session} kabul edildi"
            )
        for symbol in norm_symbols:
            cached_last = all_data[symbol].index[-1].date() if symbol in all_data else None
            needs_refresh = (
                symbol not in all_data
                or (cached_last is not None and cached_last < expected_session)
            )
            if not needs_refresh:
                continue
            try:
                logger.info(f"Downloading {symbol} from yfinance...")
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                )
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                df.columns = df.columns.str.lower()
                df = df[['open', 'high', 'low', 'close', 'volume']]
                # yfinance returns a tz-aware index (Europe/Istanbul for .IS),
                # the CSV cache is tz-naive. Concatenating the two produces an
                # object index and the sort below raises "Cannot compare
                # tz-naive and tz-aware timestamps". Normalize to the same
                # naive market-local calendar day the CSV uses.
                df.index = to_naive_market_dates(df.index)
                # Drop stub rows yfinance emits for the current trading day
                # before market close (close=NaN). These poison both the state
                # vector and the JSON response.
                before = len(df)
                df = df.dropna(subset=['close'])
                df = df[df['close'] > 0]
                if len(df) < before:
                    logger.info(f"  {symbol}: dropped {before - len(df)} NaN/zero-close rows")
                # Merge with whatever we already have from CSV (yfinance wins
                # on overlapping dates because it's fresher).
                if symbol in all_data:
                    combined_sym = pd.concat([all_data[symbol], df])
                    combined_sym = combined_sym[~combined_sym.index.duplicated(keep='last')]
                    combined_sym = combined_sym.sort_index()
                    all_data[symbol] = combined_sym
                else:
                    all_data[symbol] = df
                logger.info(f"  ✓ {symbol}: {len(all_data[symbol])} rows total")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No market data could be fetched")

        # Combine using keys parameter - same as data_fetcher.py
        # This creates clean multi-index without duplicate symbol columns
        combined_df = pd.concat(all_data.values(), keys=all_data.keys())
        combined_df.index.names = ['symbol', 'date']

        logger.info(f"Combined data: {len(combined_df)} rows")
        logger.info(f"Date range: {combined_df.index.get_level_values('date').min().date()} to {combined_df.index.get_level_values('date').max().date()}")

        # Add technical indicators
        logger.info("Calculating technical indicators...")
        combined_df = add_indicators_to_multi_symbol_df(combined_df)

        # Simple date check - find actual available date closest to target
        available_dates = combined_df.index.get_level_values('date').unique().sort_values()
        target_dt = pd.to_datetime(target_date)

        # Karar TEK bir seansa ait olmali. Eskiden yalnizca "o tarihte satir
        # var mi" soruluyordu; panelin bir kismi tazelenip kalani tazelenmeyince
        # (28.08.2026: 30 sembolden 3'u yfinance'ten geldi, 27'si CSV'de kaldi)
        # en son tarihte YALNIZCA 3 sembolun gecerli kapanisi oluyordu. Ucun
        # geri kalani sembol basina bir onceki gune dusuyor, sonucta durum
        # vektoru 3 sembol icin 28'inin, 27 sembol icin 27'nin fiyatini
        # tasiyordu — tek bir gunu temsil etmeyen karma bir goruntu.
        # Artik gunun kendisi elenir: sembollerin cogunlugu o gun gecerli
        # kapanis tasimiyorsa bir onceki seansa gecilir.
        n_symbols_total = combined_df.index.get_level_values('symbol').nunique()
        coverage_floor = max(1, int(round(n_symbols_total * MIN_SESSION_COVERAGE)))

        def _valid_close_count(day) -> int:
            try:
                rows = combined_df.xs(day, level='date')
            except KeyError:
                return 0
            close = rows['close']
            return int((close.notna() & (close > 0)).sum())

        # Find closest available date (prefer exact match or most recent before target)
        actual_date = None
        fallback_date = None
        for date in reversed(available_dates):
            if date.date() > target_dt.date():
                continue
            if fallback_date is None:
                fallback_date = date
            covered = _valid_close_count(date)
            if covered >= coverage_floor:
                actual_date = date
                break
            logger.warning(
                f"{date.date()}: {covered}/{n_symbols_total} sembolde gecerli "
                f"kapanis var (esik {coverage_floor}); bu gun atlandi"
            )
        if actual_date is None and fallback_date is not None:
            # Hicbir gun esigi tutturamadi (ornegin panelin cogu yeni listelenmis
            # sembollerden olusuyor). Eski davranisa don, ama sessizce degil.
            logger.warning(
                "Hicbir seans sembol kapsami esigini gecemedi; en son tarihe "
                f"donuluyor: {fallback_date.date()}"
            )
            actual_date = fallback_date

        if actual_date is None:
            # No data before target, use earliest available
            actual_date = available_dates[0]
            logger.warning(f"No data on or before {target_date}. Using earliest: {actual_date.date()}")
        elif actual_date.date() != target_dt.date():
            logger.warning(f"Target date {target_date} not available. Using closest: {actual_date.date()}")
        else:
            logger.info(f"Using exact target date: {target_date}")

        # Store actual date used
        combined_df.attrs['actual_date'] = actual_date.strftime("%Y-%m-%d")
        combined_df.attrs['requested_date'] = target_date

        _panel_cache_put(cache_key, combined_df)
        return combined_df

    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}", exc_info=True)
        raise


# ==================== STATE BUILDING ====================

def build_live_state(
    balance: float,
    shares_owned: Dict[str, int],
    market_data: pd.DataFrame,
    target_date: str,
    initial_balance: float = 1_000_000,
    max_shares_per_trade: int = 100,
    price_stats: Optional[Dict[str, Dict[str, float]]] = None,
    prediction_data: Optional[Dict[str, Dict[str, float]]] = None,
) -> np.ndarray:
    """
    Build state vector for inference

    Args:
        balance: Current cash balance
        shares_owned: Dict of {symbol: shares}
        market_data: Market data with indicators
        target_date: Target date (may be adjusted to latest available)
        initial_balance: Initial balance for normalization
        max_shares_per_trade: Max shares for normalization
        price_stats: Per-symbol price normalization stats
        prediction_data: {symbol: {predicted_return, predicted_direction, confidence,
            ensemble_agreement}} — ensemble tahmin verileri (opsiyonel)

    Returns:
        State vector (numpy array)
    """
    # Use actual_date if target_date was not available
    actual_date = market_data.attrs.get('actual_date', target_date)
    logger.info(f"Building state for date: {actual_date}")

    symbols = list(shares_owned.keys())
    n_stocks = len(symbols)

    # 1. Balance (normalized with log scale)
    balance_norm = np.log(balance / initial_balance + 1)

    # 2. Shares owned (normalized)
    shares_array = np.array([shares_owned.get(symbol, 0) for symbol in symbols])
    shares_norm = shares_array / max_shares_per_trade

    # 3. Market features
    features = []

    # Get the actual datetime from the index (with timezone if present)
    available_dates = market_data.index.get_level_values('date').unique()
    target_dt = pd.to_datetime(actual_date)

    # Find matching date in index (handles timezone differences)
    matching_date = None
    for date in available_dates:
        if date.date() == target_dt.date():
            matching_date = date
            break

    if matching_date is None:
        # Fallback to closest date
        matching_date = available_dates[-1]
        logger.warning(f"Exact date {actual_date} not found in index, using {matching_date}")

    for symbol in symbols:
        try:
            # Walk back to the most recent date with a non-NaN close — yfinance
            # sometimes hands back a stub row for the live trading day.
            symbol_df = market_data.xs(symbol, level='symbol')
            row = None
            for d in [matching_date] + [x for x in available_dates[::-1] if x < matching_date]:
                try:
                    candidate = symbol_df.loc[d]
                except KeyError:
                    continue
                if pd.notna(candidate.get('close')) and float(candidate.get('close', 0)) > 0:
                    row = candidate
                    break
            if row is None:
                raise KeyError(f"no valid row for {symbol}")

            # OHLCV - Convert Series to dict if needed
            if isinstance(row, pd.Series):
                row_dict = row.to_dict()
            else:
                row_dict = row  # type: ignore

            open_price = row_dict.get('open', 0)  # type: ignore
            high_price = row_dict.get('high', 0)  # type: ignore
            low_price = row_dict.get('low', 0)  # type: ignore
            close_price = row_dict.get('close', 0)  # type: ignore
            volume = row_dict.get('volume', 0)  # type: ignore

            # Technical indicators
            macd = row_dict.get('macd', 0)  # type: ignore
            rsi = row_dict.get('rsi', 50)  # type: ignore
            cci = row_dict.get('cci', 0)  # type: ignore
            adx = row_dict.get('adx', 0)  # type: ignore
            turbulence = row_dict.get('turbulence', 0)  # type: ignore 
            # Dynamic z-score normalization — same as training env (#30)
            if price_stats and symbol in price_stats:
                p_mean = price_stats[symbol]['mean']
                p_std = price_stats[symbol]['std']
            else:
                p_mean, p_std = 50.0, 50.0  # fallback

            volume_norm = np.log(volume / 1e6 + 1) / 3 if volume > 0 else 0

            features.extend([
                (open_price - p_mean) / p_std,
                (high_price - p_mean) / p_std,
                (low_price - p_mean) / p_std,
                (close_price - p_mean) / p_std,
                volume_norm,
                np.tanh(macd / 0.1),
                (rsi - 50) / 50,
                np.tanh(cci / 100),
                (adx - 25) / 25,
                np.tanh(turbulence / 2)
            ])

        except KeyError:
            logger.warning(f"No data for {symbol} on {target_date}, using zeros")
            features.extend([0] * 10)

    # Prediction features (4 per stock)
    pred_features = []
    if prediction_data:
        for symbol in symbols:
            sym_pred = prediction_data.get(symbol, {})
            pred_features.extend([
                np.tanh(sym_pred.get('predicted_return', 0.0) * 10),
                sym_pred.get('predicted_direction', 0.0) * 2 - 1,
                sym_pred.get('confidence', 0.5) * 2 - 1,
                sym_pred.get('ensemble_agreement', 0.5) * 2 - 1,
            ])

    # Combine all parts
    components = [[balance_norm], shares_norm, features]
    if pred_features:
        components.append(pred_features)

    state = np.concatenate(components).astype(np.float32)

    # Guard: SB3 policies emit NaN actions when the obs contains NaN/Inf.
    # Lookback windows (ADX 14, CCI 20, Mahalanobis 252) can leave NaN on
    # short live windows; coerce to safe scalars before inference.
    nan_count = int(np.isnan(state).sum())
    inf_count = int(np.isinf(state).sum())
    if nan_count or inf_count:
        logger.warning(
            f"State has {nan_count} NaN and {inf_count} Inf values — replacing with 0"
        )
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    logger.info(f"State built: shape={state.shape}")

    return state


# ==================== PRICE HELPERS ====================

def get_current_prices(market_data: pd.DataFrame, target_date: str) -> Dict[str, float]:
    """
    Get current prices for all symbols

    Args:
        market_data: Market data
        target_date: Target date (may be adjusted to latest available)

    Returns:
        Dict of {symbol: price}
    """
    # Use actual_date if target_date was not available
    actual_date = market_data.attrs.get('actual_date', target_date)
    prices = {}

    # Get the actual datetime from the index (with timezone if present)
    available_dates = market_data.index.get_level_values('date').unique()
    target_dt = pd.to_datetime(actual_date)

    # Find matching date in index (handles timezone differences)
    matching_date = None
    for date in available_dates:
        if date.date() == target_dt.date():
            matching_date = date
            break

    if matching_date is None:
        matching_date = available_dates[-1]
        logger.warning(f"Date {actual_date} not found in index, using {matching_date}")

    symbols = market_data.index.get_level_values('symbol').unique()

    for symbol in symbols:
        # yfinance occasionally returns a row for the current day with NaN
        # close (market still open). Walk back through available dates until
        # we find a valid close — downstream JSON serialisation rejects NaN.
        close_price = float('nan')
        try:
            symbol_df = market_data.xs(symbol, level='symbol')
            valid_dates = [d for d in available_dates[::-1] if d <= matching_date]
            for d in valid_dates:
                try:
                    candidate = symbol_df.loc[d, 'close']
                except KeyError:
                    continue
                if pd.notna(candidate) and float(candidate) > 0:
                    close_price = float(candidate)
                    if d != matching_date:
                        logger.warning(
                            f"{symbol}: close NaN on {matching_date.date()}, "
                            f"falling back to {d.date()}"
                        )
                    break
        except KeyError:
            pass

        if pd.isna(close_price) or close_price <= 0:
            logger.warning(f"No valid price for {symbol} on or before {actual_date}")
            prices[symbol] = 0.0
        else:
            prices[symbol] = close_price

    return prices


# ==================== ACTION INTERPRETATION ====================

def interpret_actions_with_risk(
    action: np.ndarray,
    symbols: List[str],
    current_prices: Dict[str, float],
    balance: float,
    shares_owned: Dict[str, int],
    risk_params: dict,
    max_shares_per_trade: int,
    commission_rate: float = 0.001
) -> List[dict]:
    """
    Interpret model actions with risk filtering

    Args:
        action: Raw action from model (shape: n_stocks)
        symbols: List of symbols
        current_prices: Current prices
        balance: Current balance
        shares_owned: Current shares owned
        risk_params: Risk parameters
        max_shares_per_trade: Maximum shares per trade
        commission_rate: Commission rate

    Returns:
        List of trade decisions
    """
    logger.info(f"Interpreting actions with risk mode: {risk_params.get('description', 'N/A')}")

    # Flatten action if needed
    action = np.array(action).flatten()

    if len(action) != len(symbols):
        raise ValueError(f"Action length mismatch: {len(action)} vs {len(symbols)}")

    decisions = []
    trades_executed = 0
    max_trades = risk_params['max_daily_trades']
    min_threshold = risk_params['min_signal_threshold']
    max_position_pct = risk_params['max_position_pct']

    # Calculate current portfolio value
    portfolio_value = balance + sum(
        shares_owned.get(symbol, 0) * current_prices.get(symbol, 0)
        for symbol in symbols
    )

    for i, symbol in enumerate(symbols):
        raw_signal = float(action[i])
        price = current_prices.get(symbol, 0)
        current_shares = shares_owned.get(symbol, 0)

        # Default: HOLD
        decision = {
            "symbol": symbol,
            "action": "HOLD",
            "raw_signal": raw_signal,
            "shares": 0,
            "price": price,
            "cost": 0.0,
            "revenue": 0.0,
            "commission": 0.0,
            "reason": "",
            "executed": False
        }

        # Check if price is available
        if price == 0:
            decision["reason"] = "No price data available"
            decisions.append(decision)
            continue

        # Check signal threshold
        if abs(raw_signal) < min_threshold:
            decision["reason"] = f"Weak signal ({raw_signal:.2f}), below threshold ({min_threshold})"
            decisions.append(decision)
            continue

        # Check max daily trades
        if trades_executed >= max_trades:
            decision["reason"] = f"Daily trade limit reached ({max_trades})"
            decisions.append(decision)
            continue

        # Scale action to shares
        shares_to_trade = int(round(raw_signal * max_shares_per_trade))

        # BUY SIGNAL
        if shares_to_trade > 0:
            # Check position limit
            position_value = (current_shares + shares_to_trade) * price
            if position_value > portfolio_value * max_position_pct:
                # Adjust shares to fit position limit
                max_position_value = portfolio_value * max_position_pct
                max_shares = int((max_position_value - current_shares * price) / price)
                shares_to_trade = max(0, min(shares_to_trade, max_shares))

                if shares_to_trade == 0:
                    decision["reason"] = f"Position limit reached ({max_position_pct*100:.0f}% of portfolio)"
                    decisions.append(decision)
                    continue

            # Check if we have enough balance
            cost = shares_to_trade * price * (1 + commission_rate)

            if cost > balance:
                # Adjust shares to fit balance
                max_affordable = int(balance / (price * (1 + commission_rate)))
                shares_to_trade = min(shares_to_trade, max_affordable)

                if shares_to_trade == 0:
                    decision["reason"] = f"Insufficient balance (need ₺{cost:,.2f}, have ₺{balance:,.2f})"
                    decisions.append(decision)
                    continue

                cost = shares_to_trade * price * (1 + commission_rate)

            commission = shares_to_trade * price * commission_rate

            decision.update({
                "action": "BUY",
                "shares": shares_to_trade,
                "cost": cost,
                "commission": commission,
                "reason": f"Strong buy signal ({raw_signal:.2f})",
                "executed": True
            })

            trades_executed += 1

        # SELL SIGNAL
        elif shares_to_trade < 0:
            shares_to_sell = min(abs(shares_to_trade), current_shares)

            if shares_to_sell == 0:
                decision["reason"] = "No shares to sell"
                decisions.append(decision)
                continue

            revenue = shares_to_sell * price * (1 - commission_rate)
            commission = shares_to_sell * price * commission_rate

            decision.update({
                "action": "SELL",
                "shares": shares_to_sell,
                "revenue": revenue,
                "commission": commission,
                "reason": f"Sell signal ({raw_signal:.2f})",
                "executed": True
            })

            trades_executed += 1

        decisions.append(decision)

    logger.info(f"Generated {trades_executed} executable trades out of {len(symbols)} signals")

    return decisions


# ==================== PORTFOLIO CALCULATION ====================

def calculate_portfolio_value(
    balance: float,
    shares: Dict[str, int],
    prices: Dict[str, float]
) -> dict:
    """
    Calculate portfolio value

    Args:
        balance: Cash balance
        shares: Shares owned
        prices: Current prices

    Returns:
        Portfolio snapshot dict
    """
    portfolio_value = balance + sum(
        shares.get(symbol, 0) * prices.get(symbol, 0)
        for symbol in shares.keys()
    )

    return {
        "balance": balance,
        "shares": shares.copy(),
        "portfolio_value": portfolio_value
    }


def simulate_portfolio_after_trades(
    balance: float,
    shares: Dict[str, int],
    decisions: List[dict]
) -> dict:
    """
    Simulate portfolio after executing decisions

    Args:
        balance: Current balance
        shares: Current shares
        decisions: Trade decisions

    Returns:
        Portfolio snapshot after trades
    """
    new_balance = balance
    new_shares = shares.copy()

    for decision in decisions:
        if not decision["executed"]:
            continue

        symbol = decision["symbol"]

        if decision["action"] == "BUY":
            new_balance -= decision["cost"]
            new_shares[symbol] = new_shares.get(symbol, 0) + decision["shares"]

        elif decision["action"] == "SELL":
            new_balance += decision["revenue"]
            new_shares[symbol] = new_shares.get(symbol, 0) - decision["shares"]

    # Calculate new portfolio value
    portfolio_value = new_balance + sum(
        new_shares.get(decision["symbol"], 0) * decision["price"]
        for decision in decisions
        if decision["symbol"] in new_shares
    )

    return {
        "balance": new_balance,
        "shares": new_shares,
        "portfolio_value": portfolio_value
    }


# ==================== DATA PERSISTENCE ====================

def save_daily_decision(
    date: str,
    decisions: List[dict],
    portfolio_before: dict,
    portfolio_after: dict,
    risk_mode: str,
    max_shares_per_trade: int
):
    """
    Save daily decision to JSON file

    Args:
        date: Decision date
        decisions: List of trade decisions
        portfolio_before: Portfolio before trades
        portfolio_after: Portfolio after trades
        risk_mode: Risk mode used
        max_shares_per_trade: Max shares per trade
    """
    logger.info(f"Saving daily decision for {date}")

    # Kararlar kullanici bazlidir: her kullanicinin kendi portfoyu ve
    # gunluk karar gecmisi kendi calisma alaninda tutulur.
    decision_file = os.path.join(ws.live_trading_dir(), 'trade_decisions.json')
    lock_file = decision_file + '.lock'

    # File-level lock for concurrent access safety (#31)
    with FileLock(lock_file):
        if os.path.exists(decision_file):
            with open(decision_file, 'r') as f:
                all_decisions = json.load(f)
        else:
            all_decisions = {}

        all_decisions[date] = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_before": portfolio_before,
            "decisions": decisions,
            "portfolio_after": portfolio_after,
            "summary": {
                "total_trades": len([d for d in decisions if d["executed"]]),
                "total_commission": sum(d.get("commission", 0) for d in decisions),
                "daily_return_pct": (
                    (portfolio_after["portfolio_value"] - portfolio_before["portfolio_value"])
                    / portfolio_before["portfolio_value"] * 100
                    if portfolio_before["portfolio_value"] > 0 else 0
                ),
                "risk_mode": risk_mode,
                "max_shares_per_trade": max_shares_per_trade
            }
        }

        with open(decision_file, 'w') as f:
            json.dump(all_decisions, f, indent=2)

    logger.info(f"Decision saved to {decision_file}")


def append_to_portfolio_history(
    date: str,
    portfolio_after: dict,
    daily_return_pct: float,
    realized_pnl: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    total_pnl: Optional[float] = None,
):
    """
    Append to portfolio history CSV

    Args:
        date: Date
        portfolio_after: Portfolio snapshot
        daily_return_pct: Gunluk getiri — bir ONCEKI kaydin toplam degerine
            gore. Kararin `summary.daily_return_pct` alani bu DEGILDIR: orada
            alim-satim ayni gunun ayni fiyatlariyla simule edildigi icin geriye
            yalnizca komisyon kalir ve deger tanim geregi ~0 cikar.
        realized_pnl / unrealized_pnl / total_pnl: kagit portfoyun o gunku
            mark-to-market kirilimi (opsiyonel — eski satirlarda bos kalir)
    """
    logger.info(f"Appending to portfolio history: {date}")

    history_file = os.path.join(ws.live_trading_dir(), 'portfolio_history.csv')

    # Create new row
    new_row = {
        'date': date,
        'balance': portfolio_after['balance'],
        'portfolio_value': portfolio_after['portfolio_value'],
        'daily_return_pct': daily_return_pct
    }
    for key, val in (('realized_pnl', realized_pnl),
                     ('unrealized_pnl', unrealized_pnl),
                     ('total_pnl', total_pnl)):
        if val is not None:
            new_row[key] = val

    # Add shares columns
    for symbol, shares in portfolio_after['shares'].items():
        # Clean symbol name for column
        col_name = symbol.replace('.IS', '_shares')
        new_row[col_name] = shares

    # Append or create
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        # Remove existing entry for this date if exists
        df = df[df['date'] != date]
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        new_df = pd.DataFrame([new_row])

    new_df.to_csv(history_file, index=False)
    logger.info(f"Portfolio history updated: {history_file}")


def load_portfolio_history(days: int = 30) -> dict:
    """
    Load portfolio history

    Args:
        days: Number of days to load

    Returns:
        Dict with dates, values, returns, balances
    """
    history_file = os.path.join(ws.live_trading_dir(), 'portfolio_history.csv')

    if not os.path.exists(history_file):
        return {
            "dates": [],
            "portfolio_values": [],
            "daily_returns": [],
            "balances": []
        }

    df = pd.read_csv(history_file)
    df = df.tail(days)

    out = {
        "dates": df['date'].tolist(),
        "portfolio_values": df['portfolio_value'].tolist(),
        "daily_returns": df['daily_return_pct'].tolist(),
        "balances": df['balance'].tolist()
    }
    # Kar/zarar kirilimi yalnizca yeni satirlarda var; eski dosyalarda kolon
    # hic bulunmayabilir. Bos liste dondurmek cagirani dallanmaya zorlardi.
    for col in ('realized_pnl', 'unrealized_pnl', 'total_pnl'):
        out[col + 's'] = (
            [None if pd.isna(v) else float(v) for v in df[col]]
            if col in df.columns else [None] * len(df)
        )
    return out
