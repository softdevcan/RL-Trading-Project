"""
Feature Group Registry

Her feature grubu bir kimlik (id), etiket, kategori, varsayilan durum
ve gerekli veri kaynagi bilgisini icerir. Bu kayit; egitim isteginde
hangi gruplarin kullanilacagini belirler, API ve dashboard arasinida
sozlesme saglar.

Kullanim:
    from prediction.feature_groups import REGISTRY, default_groups, resolve_groups

    active = resolve_groups(request.feature_groups)   # ['returns', 'macro_tr', ...]
"""

from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Grup tanimlamalari
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Teknik / OHLCV ──────────────────────────────────────────────────────
    "returns": {
        "label": "Getiri ozellikleri",
        "description": "Log getiri, basit getiri, coklu gecikme",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "volatility": {
        "label": "Volatilite ozellikleri",
        "description": "Parkinson, Garman-Klass, realized vol, ATR",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "momentum": {
        "label": "Momentum ozellikleri",
        "description": "RSI turevleri, MACD histogram gradyani",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "volume": {
        "label": "Hacim ozellikleri",
        "description": "Hacim oranlari, hacim-fiyat korelasyonu",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "calendar": {
        "label": "Takvim ozellikleri",
        "description": "Gun, ay, ceyrek, hafta ici",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "technicals": {
        "label": "Teknik indikatorler",
        "description": "EMA, BB, Stoch, CCI, Williams %R ve gecikmeleri",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "market_regime": {
        "label": "Piyasa rejimi",
        "description": "Volatilite rejimi, trend rejimi bayraklari",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "iceemdan": {
        "label": "ICEEMDAN gurultu filtresi",
        "description": "EMD tabanli sinyal ayrimi (yavas, GPU/CPU yogun)",
        "category": "technical",
        "default": False,
        "requires": [],
    },

    # ── Yeni teknik ─────────────────────────────────────────────────────────
    "fibonacci": {
        "label": "Fibonacci seviyeleri",
        "description": "Son 50/100/200 g swing H/L'den %23.6, %38.2, %50, %61.8",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "donchian": {
        "label": "Donchian / Breakout",
        "description": "20/55 gunluk kanal genisligi, fiyat konumu, Chandelier Exit",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "rolling_zscore": {
        "label": "Rolling Z-score",
        "description": "20/60/252 gunluk Z-score (ortalamaya donus sinyali)",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "obv": {
        "label": "OBV / Hacim analizi",
        "description": "On Balance Volume trendi, OBV/MA orani, CMF",
        "category": "technical",
        "default": True,
        "requires": [],
    },
    "seasonality": {
        "label": "Mevsimsellik",
        "description": "Hindistan dugun sezonu, Cin Yeni Yili, merkez bankasi alim periyotlari",
        "category": "alternative",
        "default": True,
        "requires": [],
    },

    # ── Makro ────────────────────────────────────────────────────────────────
    "macro_tr": {
        "label": "Turk makro (EVDS)",
        "description": "TCMB politika faizi, TUFE/UFE enflasyon",
        "category": "macro",
        "default": True,
        "requires": ["macro_df"],
    },
    "cross_asset": {
        "label": "Capraz varlik (BIST/TRY)",
        "description": "BIST-100, USD/TRY, EUR/TRY korelasyonu",
        "category": "macro",
        "default": True,
        "requires": ["cross_asset_df"],
    },
    "macro_global": {
        "label": "Global makro",
        "description": "Reel faiz (US10Y-CPI), Petrol (WTI), GVZ, Altin/Gumus orani",
        "category": "macro",
        "default": True,
        "requires": ["macro_df"],
    },
    "fundamental": {
        "label": "Fundamental oranlar",
        "description": "P/E, P/B, ROE, ROA, borclanma",
        "category": "macro",
        "default": False,
        "requires": ["fundamental_df"],
    },
}


# ---------------------------------------------------------------------------
# Hedef tip tanimlamalari
# ---------------------------------------------------------------------------

TARGET_TYPES: Dict[str, Dict[str, str]] = {
    "log_return": {
        "label": "Log getiri (onerilen)",
        "description": (
            "target = log(close_t+H / close_t). "
            "Stationarity saglar; fiyat seviyesinden bagimsiz; "
            "non-stationary varliklar (altin, doviz) icin zorunlu."
        ),
    },
    "abs_price": {
        "label": "Mutlak fiyat (eski)",
        "description": (
            "target = close_t+H. "
            "Geleneksel regression hedefi; "
            "uzun vadeli/yuksek buyumeli serilerde distribution shift riski."
        ),
    },
}

DEFAULT_TARGET_TYPE = "log_return"


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------

def default_groups() -> List[str]:
    """Varsayilan olarak acik gruplarin id listesini dondur."""
    return [gid for gid, meta in REGISTRY.items() if meta["default"]]


def resolve_groups(enabled: Optional[List[str]]) -> List[str]:
    """
    Kullanicinin istedigi grup listesini dogrula ve dondur.

    Args:
        enabled: None ise default gruplar; aksi halde filtrele
                 (bilinmeyen id'ler sessizce atlenir).

    Returns:
        Gecerli ve aktif grup id listesi
    """
    if enabled is None:
        return default_groups()
    valid = set(REGISTRY.keys())
    resolved = [g for g in enabled if g in valid]
    if not resolved:
        return default_groups()
    return resolved


def groups_by_category() -> Dict[str, List[Dict[str, Any]]]:
    """Gruplari kategoriye gore sirala (dashboard icin)."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for gid, meta in REGISTRY.items():
        cat = meta["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append({"id": gid, **meta})
    return result


def group_requires_data(group_id: str) -> List[str]:
    """Grup hangi veri kaynagi parametrelerini gerektiriyor?"""
    return REGISTRY.get(group_id, {}).get("requires", [])
