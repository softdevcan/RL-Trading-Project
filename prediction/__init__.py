"""
Gelismis Fiyat Tahmin Modulu

Ensemble tabanli (XGBoost + LightGBM + CatBoost + BiLSTM + TFT)
gunluk ve haftalik kapanis fiyati ve yon tahmini.

NOT (bilinen eksiklik): `prediction/models/` paketi (PricePredictor,
StackingEnsemble, MODEL_REGISTRY ve model siniflari) bu repo kopyasinda
mevcut degil — commit 25518b6 eski `prediction/models.py` dosyasini sildi ama
yerine gelen `prediction/models/` paketi hic commit edilmemis. Bu paket
yeniden olusturulana kadar tahmin altsistemi (dashboard tahmin sayfasi,
/api/prediction/* uctan-uca egitim/tahmin) calismaz. RL egitim hatti
(`env/`, `/api/trading/*`, dashboard data/training/models/daily-decision
sayfalari) bundan etkilenmez. Ayrintilar: docs/development/roadmap.md.
"""

from prediction.feature_engineer import PredictionFeatureEngineer
from prediction.feature_selector import FeatureSelector
from prediction.tracker import PredictionTracker

# `prediction.models` paketi eksik olabilir (yukaridaki nota bakin). Eksikse
# import zincirini (app.main -> config -> prediction) cokertmeden devam et —
# bu sinif/registry'lere fiilen erisen tahmin ozellikleri kullanildiklarinda
# net bir hatayla basarisiz olur.
try:
    from prediction.models import PricePredictor, StackingEnsemble, MODEL_REGISTRY
    _MODELS_AVAILABLE = True
except ImportError as _exc:  # prediction/models paketi yok
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "prediction.models paketi yuklenemedi (%s) — tahmin altsistemi devre disi. "
        "RL egitim hatti etkilenmez.", _exc,
    )
    PricePredictor = None
    StackingEnsemble = None
    MODEL_REGISTRY = {}
    _MODELS_AVAILABLE = False

__all__ = [
    'PredictionFeatureEngineer',
    'FeatureSelector',
    'PricePredictor',
    'StackingEnsemble',
    'MODEL_REGISTRY',
    'PredictionTracker',
]
