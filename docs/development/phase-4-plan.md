Faz 4 — Modüler Feature Sistemi
Adım 1 — prediction/feature_groups.py (yeni dosya)
Feature'ların merkezi kaydı. Her grup:

id: makine kodu (API'de kullanılır)
label: arayüzde görünen ad
category: teknik / makro / alternatif / hedef
requires: hangi veri kaynağı gerektirir
default: varsayılan açık/kapalı
GROUP_REGISTRY = {
  # Mevcut (her zaman açık)
  "core_ohlcv":       { category: "technical", default: True, requires: [] }
  "returns":          { category: "technical", default: True, requires: [] }
  "volatility":       { category: "technical", default: True, requires: [] }
  "momentum":         { category: "technical", default: True, requires: [] }
  "volume":           { category: "technical", default: True, requires: [] }
  "calendar":         { category: "technical", default: True, requires: [] }
  "technicals":       { category: "technical", default: True, requires: [] }
  "market_regime":    { category: "technical", default: True, requires: [] }
  "cross_asset":      { category: "macro",     default: True, requires: ["cross_asset_df"] }
  "macro_tr":         { category: "macro",     default: True, requires: ["macro_df"] }
  "fundamental":      { category: "macro",     default: True, requires: ["fundamental_df"] }
  "iceemdan":         { category: "technical", default: False, requires: [] }
  # Yeni
  "fibonacci":        { category: "technical", default: True,  requires: [] }
  "donchian":         { category: "technical", default: True,  requires: [] }
  "rolling_zscore":   { category: "technical", default: True,  requires: [] }
  "obv_features":     { category: "technical", default: True,  requires: [] }
  "macro_global":     { category: "macro",     default: True,  requires: ["macro_df"] }
    # → reel faiz (US10Y - CPI), petrol (CL=F), GVZ, altın-gümüş oranı
  "seasonality":      { category: "alternative", default: True, requires: [] }
    # → Hindistan düğün sezonu, Çin YY, merkez bankası alım dönemleri
}
TARGET_TYPES = {
  "log_return":  "Log getiri tahmini (önerilen — stationarity)",
  "abs_price":   "Mutlak fiyat tahmini (eski davranış)"
}
Adım 2 — Yeni veri kaynakları (data/macro_fetcher.py)
Eklenenler:

# Global makro (yfinance)
'oil_wti':       'CL=F'      # ham petrol
'gold_vix':      '^GVZ'      # altın volatilite endeksi
'silver':        'SI=F'      # altın/gümüş oranı için
'us_real_rate':  hesaplama   # US10Y - CPI (FRED veya varolan verilerden)
Adım 3 — feature_engineer.py güncellemeleri
build_features imzası:

def build_features(
    df, symbol,
    macro_df=None, fundamental_df=None, cross_asset_df=None,
    feature_groups: Optional[List[str]] = None,  # YENİ — None = hepsi
    target_type: str = 'log_return',              # YENİ — 'log_return' | 'abs_price'
    use_iceemdan: bool = False,
) -> pd.DataFrame
Her grup metod çağrısı if group_id in active_groups: ile korunur.

Yeni metodlar:

_add_fibonacci_features(data) — son 50/100/200 günlük swing H/L'den %23.6, %38.2, %50, %61.8 fark
_add_donchian_features(data) — 20/55 günlük kanal genişliği, fiyat konumu
_add_rolling_zscore_features(data) — 20/60/252 günlük Z-score
_add_obv_features(data) — OBV trend, OBV/MA oranı
_add_global_macro_features(data, macro_df) — reel faiz, petrol, GVZ, altın-gümüş
_add_seasonality_features(data, symbol) — aylık siklik encoding, özel dönemler
_build_targets güncellemesi:

if target_type == 'log_return':
    data['target_price'] = np.log(data['close'].shift(-horizon) / data['close'])
else:
    data['target_price'] = data['close'].shift(-horizon)  # mevcut
Adım 4 — Schema güncellemeleri
class FeatureGroupsConfig(BaseModel):
    enabled_groups: Optional[List[str]] = None   # None = tüm default'lar
    target_type: str = 'log_return'
    use_iceemdan: bool = False
class PredictionTrainRequest(BaseModel):
    symbol: str
    horizon: str = 'daily'
    start_date: str = '2018-01-01'
    test_ratio: float = 0.2
    source: Optional[str] = None
    feature_config: FeatureGroupsConfig = FeatureGroupsConfig()  # YENİ
Adım 5 — Ensemble metadata'ya kayıt
Eğitilen model hangi grup konfigürasyonuyla eğitildi bilgisi ensemble_meta.json'a eklenir. Predict zamanında aynı konfigürasyon kullanılır → tutarlılık.

Adım 6 — Dashboard UI (prediction.py "Model Egit" tab)
Feature Grup Paneli:

╔══ Teknik Göstergeler ══════════════════════╗
│ ☑ Getiri özellikleri    ☑ Volatilite       │
│ ☑ Momentum              ☑ Hacim            │
│ ☑ Fibonacci seviyeleri  ☑ Donchian         │
│ ☑ Rolling Z-score       ☑ OBV              │
│ ☐ ICEEMDAN (yavaş)                         │
╚════════════════════════════════════════════╝
╔══ Makro / Alternatif ══════════════════════╗
│ ☑ Türk makro (EVDS)     ☑ Çapraz varlık   │
│ ☑ Global makro (petrol, reel faiz, GVZ)    │
│ ☑ Mevsimsellik          ☐ Fundamental      │
╚════════════════════════════════════════════╝
╔══ Hedef Tipi ══════════════════════════════╗
│ ◉ Log getiri (önerilen)                    │
│ ○ Mutlak fiyat (eski)                      │
╚════════════════════════════════════════════╝
Uygulama Sırası
#	Dosya	İş
1	prediction/feature_groups.py	Grup registry (yeni dosya)
2	data/macro_fetcher.py	Oil, GVZ, silver, real_rate ekle
3	prediction/feature_engineer.py	6 yeni metod + feature_groups + target_type param
4	prediction/models/ensemble.py	feature_config kaydet/yükle
5	prediction/trainer.py	feature_config pass-through
6	app/services/prediction_service.py	feature_config pass-through
7	app/schemas/prediction.py	FeatureGroupsConfig + TrainRequest güncelle
8	app/api/routes/prediction.py	/symbols → available groups döndür, /train feature_config al
9	dashboard/pages/prediction.py	Feature seçim paneli + hedef tipi toggleFaz 4 — Modüler Feature Sistemi
Adım 1 — prediction/feature_groups.py (yeni dosya)
Feature'ların merkezi kaydı. Her grup:

id: makine kodu (API'de kullanılır)
label: arayüzde görünen ad
category: teknik / makro / alternatif / hedef
requires: hangi veri kaynağı gerektirir
default: varsayılan açık/kapalı
GROUP_REGISTRY = {
  # Mevcut (her zaman açık)
  "core_ohlcv":       { category: "technical", default: True, requires: [] }
  "returns":          { category: "technical", default: True, requires: [] }
  "volatility":       { category: "technical", default: True, requires: [] }
  "momentum":         { category: "technical", default: True, requires: [] }
  "volume":           { category: "technical", default: True, requires: [] }
  "calendar":         { category: "technical", default: True, requires: [] }
  "technicals":       { category: "technical", default: True, requires: [] }
  "market_regime":    { category: "technical", default: True, requires: [] }
  "cross_asset":      { category: "macro",     default: True, requires: ["cross_asset_df"] }
  "macro_tr":         { category: "macro",     default: True, requires: ["macro_df"] }
  "fundamental":      { category: "macro",     default: True, requires: ["fundamental_df"] }
  "iceemdan":         { category: "technical", default: False, requires: [] }
  # Yeni
  "fibonacci":        { category: "technical", default: True,  requires: [] }
  "donchian":         { category: "technical", default: True,  requires: [] }
  "rolling_zscore":   { category: "technical", default: True,  requires: [] }
  "obv_features":     { category: "technical", default: True,  requires: [] }
  "macro_global":     { category: "macro",     default: True,  requires: ["macro_df"] }
    # → reel faiz (US10Y - CPI), petrol (CL=F), GVZ, altın-gümüş oranı
  "seasonality":      { category: "alternative", default: True, requires: [] }
    # → Hindistan düğün sezonu, Çin YY, merkez bankası alım dönemleri
}
TARGET_TYPES = {
  "log_return":  "Log getiri tahmini (önerilen — stationarity)",
  "abs_price":   "Mutlak fiyat tahmini (eski davranış)"
}
Adım 2 — Yeni veri kaynakları (data/macro_fetcher.py)
Eklenenler:

# Global makro (yfinance)
'oil_wti':       'CL=F'      # ham petrol
'gold_vix':      '^GVZ'      # altın volatilite endeksi
'silver':        'SI=F'      # altın/gümüş oranı için
'us_real_rate':  hesaplama   # US10Y - CPI (FRED veya varolan verilerden)
Adım 3 — feature_engineer.py güncellemeleri
build_features imzası:

def build_features(
    df, symbol,
    macro_df=None, fundamental_df=None, cross_asset_df=None,
    feature_groups: Optional[List[str]] = None,  # YENİ — None = hepsi
    target_type: str = 'log_return',              # YENİ — 'log_return' | 'abs_price'
    use_iceemdan: bool = False,
) -> pd.DataFrame
Her grup metod çağrısı if group_id in active_groups: ile korunur.

Yeni metodlar:

_add_fibonacci_features(data) — son 50/100/200 günlük swing H/L'den %23.6, %38.2, %50, %61.8 fark
_add_donchian_features(data) — 20/55 günlük kanal genişliği, fiyat konumu
_add_rolling_zscore_features(data) — 20/60/252 günlük Z-score
_add_obv_features(data) — OBV trend, OBV/MA oranı
_add_global_macro_features(data, macro_df) — reel faiz, petrol, GVZ, altın-gümüş
_add_seasonality_features(data, symbol) — aylık siklik encoding, özel dönemler
_build_targets güncellemesi:

if target_type == 'log_return':
    data['target_price'] = np.log(data['close'].shift(-horizon) / data['close'])
else:
    data['target_price'] = data['close'].shift(-horizon)  # mevcut
Adım 4 — Schema güncellemeleri
class FeatureGroupsConfig(BaseModel):
    enabled_groups: Optional[List[str]] = None   # None = tüm default'lar
    target_type: str = 'log_return'
    use_iceemdan: bool = False
class PredictionTrainRequest(BaseModel):
    symbol: str
    horizon: str = 'daily'
    start_date: str = '2018-01-01'
    test_ratio: float = 0.2
    source: Optional[str] = None
    feature_config: FeatureGroupsConfig = FeatureGroupsConfig()  # YENİ
Adım 5 — Ensemble metadata'ya kayıt
Eğitilen model hangi grup konfigürasyonuyla eğitildi bilgisi ensemble_meta.json'a eklenir. Predict zamanında aynı konfigürasyon kullanılır → tutarlılık.

Adım 6 — Dashboard UI (prediction.py "Model Egit" tab)
Feature Grup Paneli:

╔══ Teknik Göstergeler ══════════════════════╗
│ ☑ Getiri özellikleri    ☑ Volatilite       │
│ ☑ Momentum              ☑ Hacim            │
│ ☑ Fibonacci seviyeleri  ☑ Donchian         │
│ ☑ Rolling Z-score       ☑ OBV              │
│ ☐ ICEEMDAN (yavaş)                         │
╚════════════════════════════════════════════╝
╔══ Makro / Alternatif ══════════════════════╗
│ ☑ Türk makro (EVDS)     ☑ Çapraz varlık   │
│ ☑ Global makro (petrol, reel faiz, GVZ)    │
│ ☑ Mevsimsellik          ☐ Fundamental      │
╚════════════════════════════════════════════╝
╔══ Hedef Tipi ══════════════════════════════╗
│ ◉ Log getiri (önerilen)                    │
│ ○ Mutlak fiyat (eski)                      │
╚════════════════════════════════════════════╝
Uygulama Sırası
#	Dosya	İş
1	prediction/feature_groups.py	Grup registry (yeni dosya)
2	data/macro_fetcher.py	Oil, GVZ, silver, real_rate ekle
3	prediction/feature_engineer.py	6 yeni metod + feature_groups + target_type param
4	prediction/models/ensemble.py	feature_config kaydet/yükle
5	prediction/trainer.py	feature_config pass-through
6	app/services/prediction_service.py	feature_config pass-through
7	app/schemas/prediction.py	FeatureGroupsConfig + TrainRequest güncelle
8	app/api/routes/prediction.py	/symbols → available groups döndür, /train feature_config al
9	dashboard/pages/prediction.py	Feature seçim paneli + hedef tipi toggle



prediction_service.py
+4
        source: Optional[str] = None,
        feature_groups: Optional[list] = None,
        target_type: str = 'log_return',
    ) -> None:
                source=resolved,
                feature_groups=feature_groups,
Adım 7 — schemas/prediction.py: FeatureGroupsConfig ekle, PredictionTrainRequest'e entegre et:

prediction.py
+32
-1
class FeatureGroupsConfig(BaseModel):
    """Egitim sirasinda kullanilacak feature grup konfigurasyonu."""
    enabled_groups: Optional[List[str]] = Field(
        default=None,
        description=(
            "Aktif feature grup id listesi. None = registry default gruplari. "
SymbolEntry'e available_feature_groups ekle:

prediction.py
+1
    sources: List[str]  # ['yfinance'] veya ['borsapy', 'yfinance']
    available_feature_groups: Optional[List[str]] = None  # Registry default gruplari
TrainedModelEntry'e feature_groups ve target_type ekle:

prediction.py
+2
    models_trained: List[str] = []
    feature_groups: Optional[List[str]] = None
    target_type: Optional[str] = None
Adım 8 — routes/prediction.py: /train'e feature_config, /symbols'e gruplar + /price-history endpoint'i:

prediction.py
+4
      source=source,
      feature_groups=(request.feature_config.enabled_groups
                      if request.feature_config else None),
      target_type=(request.feature_config.target_type
                   if request.feature_config else 'log_return'),
  )
prediction.py
+4
        source=source,
        feature_groups=(request.feature_config.enabled_groups
                        if request.feature_config else None),
        target_type=(request.feature_config.target_type
                     if request.feature_config else 'log_return'),
    )
Şimdi /symbols endpoint'ine available_feature_groups ekle ve /price-history yeni endpoint'i ekle:

prediction.py
+71
    from app.services.prediction_service import get_supported_sources
    from prediction.feature_groups import default_groups
    info_map = {**STOCK_INFO, **ASSET_INFO}
    default_grp = default_groups()
            sources=get_supported_sources(sym),
            available_feature_groups=default_grp,
pd import'u routes dosyasında var mı diye kontrol edelim:

prediction.py
+1
from typing import Any, Optional
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response, status
Şimdi dashboard/api_client.py'a /price-history endpoint'i ekleyelim:

api_client.py
+8
def get_price_history(symbol: str, source: Optional[str] = None, days: int = 90) -> Dict:
    """Sembol icin tarihsel fiyat verisi (OHLCV + MA + BB)."""
    params: Dict = {"symbol": symbol, "days": days}
    if source:
        params["source"] = source
    return _get("/prediction/price-history", params=params) or {}
Optional kullanıldı, import'u kontrol edelim:

Adım 9 — Dashboard: İnteraktif fiyat grafiği + feature seçim paneli. En kapsamlı değişiklik:

Şimdi prediction sekmesini ve callback'leri güncelleyelim — interaktif fiyat grafiği + feature seçim paneli ekleyelim: