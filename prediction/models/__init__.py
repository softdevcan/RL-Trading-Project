"""
Coklu Tahmin Modeli Paketi

Base model, gradient boosting (XGBoost, LightGBM, CatBoost),
deep learning (BiLSTM, TFT) ve stacking ensemble modelleri icerir.
"""

from prediction.models.base import BasePredictionModel
from prediction.models.xgboost_model import XGBoostModel
from prediction.models.lightgbm_model import LightGBMModel
from prediction.models.catboost_model import CatBoostModel
from prediction.models.lstm_model import BiLSTMModel
from prediction.models.tft_model import TFTModel
from prediction.models.ensemble import StackingEnsemble

MODEL_REGISTRY = {
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'bilstm': BiLSTMModel,
    'tft': TFTModel,
}


class PricePredictor:
    """Geriye uyumlu wrapper — ensemble tabanli tahmin.

    Eski PricePredictor API'sini korurken yeni ensemble
    mimarisini kullanir. prediction_service.py ve diger
    mevcut kodlarla uyumlu calisir.
    """

    def __init__(self, horizon: str = 'daily', source=None):
        self.horizon = horizon
        self.source = source
        self.ensemble = StackingEnsemble(horizon=horizon, source=source)

    def train(self, df, symbol, test_ratio=0.2, **kwargs):
        return self.ensemble.train(df, symbol, test_ratio=test_ratio, **kwargs)

    def predict_next(self, df, symbol, macro_df=None, fundamental_df=None,
                     cross_asset_df=None, **kwargs):
        return self.ensemble.predict_next(
            df, symbol,
            macro_df=macro_df,
            fundamental_df=fundamental_df,
            cross_asset_df=cross_asset_df,
            **kwargs,
        )

    def save(self, symbol, **kwargs):
        self.ensemble.save(symbol, **kwargs)

    def load(self, symbol):
        self.ensemble.load(symbol)

    def is_trained(self, symbol):
        return self.ensemble.is_trained(symbol)

    def list_trained_models(self):
        return self.ensemble.list_trained_models()


__all__ = [
    'BasePredictionModel',
    'XGBoostModel',
    'LightGBMModel',
    'CatBoostModel',
    'BiLSTMModel',
    'TFTModel',
    'StackingEnsemble',
    'PricePredictor',
    'MODEL_REGISTRY',
]
