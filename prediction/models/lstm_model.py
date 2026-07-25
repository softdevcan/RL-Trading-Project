"""
BiLSTM Tahmin Modeli

PyTorch tabanli cift yonlu LSTM ile fiyat tahmini.
RTX 4060 (8GB VRAM) icin optimize edilmistir.
"""

import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prediction.models.base import BasePredictionModel
from prediction.models import torch_perf as tp

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class _BiLSTMNet(nn.Module):
    """Cift yonlu LSTM agi."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_price = nn.Linear(hidden_size * 2, 1)
        self.fc_direction = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        price = self.fc_price(last_hidden).squeeze(-1)
        direction = torch.sigmoid(self.fc_direction(last_hidden)).squeeze(-1)
        return price, direction


class BiLSTMModel(BasePredictionModel):
    """PyTorch BiLSTM fiyat tahmin modeli.

    Girdi: (batch, lookback_window, n_features) sekans verisi
    Cikti: fiyat tahmini + yon olasiligi
    """

    MODEL_TYPE = 'bilstm'

    def __init__(self, horizon: str = 'daily', params: Optional[Dict] = None,
                 source: Optional[str] = None):
        super().__init__(horizon, params, source)
        self.device = _get_device()
        self.net: Optional[_BiLSTMNet] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.2,
            'lookback': 30,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 100,
            'patience': 15,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        return {
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lookback': trial.suggest_categorical('lookback', [10, 20, 30, 60]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'epochs': 100,
            'patience': 15,
        }

    def _build_model(self, params: Dict[str, Any]):
        pass

    def _supports_warm_start(self) -> bool:
        return True

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray, lookback: int,
    ):
        """Zaman serisi verisini LSTM sekansarina donustur."""
        sequences_X, sequences_y, sequences_y_dir = [], [], []
        for i in range(lookback, len(X)):
            sequences_X.append(X[i - lookback:i])
            sequences_y.append(y[i])
            if i > 0:
                sequences_y_dir.append(float(y[i] > y[i - 1]))
            else:
                sequences_y_dir.append(0.5)

        return (
            np.array(sequences_X),
            np.array(sequences_y),
            np.array(sequences_y_dir),
        )

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        # Epic 2.2: semboller paralel egitilirken GPU'ya ayni anda kac DL
        # egitimi girecegini sinirla (seri kosumda etkisiz).
        with tp.gpu_slot():
            return self._fit_impl(X_train, y_train, X_val, y_val)

    def _fit_impl(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        p = self.params
        lookback = p['lookback']

        self.scaler_mean = X_train.mean(axis=0)
        self.scaler_std = X_train.std(axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        self.target_mean = float(y_train.mean())
        self.target_std = float(y_train.std())
        if self.target_std == 0:
            self.target_std = 1.0

        X_tr_norm = (X_train - self.scaler_mean) / self.scaler_std
        X_va_norm = (X_val - self.scaler_mean) / self.scaler_std
        y_tr_norm = (y_train - self.target_mean) / self.target_std
        y_va_norm = (y_val - self.target_mean) / self.target_std

        X_tr_seq, y_tr_seq, y_tr_dir = self._create_sequences(X_tr_norm, y_tr_norm, lookback)
        X_va_seq, y_va_seq, y_va_dir = self._create_sequences(X_va_norm, y_va_norm, lookback)

        if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
            logger.warning("Yeterli veri yok (lookback cok buyuk)")
            return {'mape': 999, 'rmse': 999, 'mae': 999, 'direction_accuracy': 0}

        n_features = X_tr_seq.shape[2]
        self.net = _BiLSTMNet(
            input_size=n_features,
            hidden_size=p['hidden_size'],
            num_layers=p['num_layers'],
            dropout=p['dropout'],
        ).to(self.device)

        # Faz 6 (2.1): warm-start — onceki egitimin agirliklarini yukle (ayni
        # mimari sart). Sadece ensemble warm_start=True verdiginde dolu; aksi
        # halde None = sifirdan (mevcut davranis, bit-es). Uyumsuz mimaride
        # sessizce sifirdan devam eder.
        ws = getattr(self, '_warm_start_from', None)
        if ws is not None and getattr(ws, 'net', None) is not None:
            try:
                self.net.load_state_dict(ws.net.state_dict())
                logger.info("  [bilstm] warm-start: onceki agirliklar yuklendi")
            except Exception as exc:
                logger.info(f"  [bilstm] warm-start atlandi (mimari uyumsuz?): {exc}")

        # Faz 6 (3.1/B7): veri kucuk — tum sekanslari bir kez cihaza tasi ve
        # dilimleyerek ilerle. shuffle=False oldugu icin batch sinirlari
        # DataLoader ile birebir ayni, sayisal sonuc degismez.
        tp.configure_threads()
        batch_size = p['batch_size']
        if tp.gpu_preload_enabled():
            tr_tensors = tp.to_device_tensors([X_tr_seq, y_tr_seq, y_tr_dir], self.device)
            va_tensors = tp.to_device_tensors([X_va_seq, y_va_seq, y_va_dir], self.device)
            train_batches = lambda: tp.iter_batches(tr_tensors, batch_size)
            val_batches = lambda: tp.iter_batches(va_tensors, batch_size)
            n_val_batches = tp.n_batches(len(X_va_seq), batch_size)
        else:
            train_ds = TensorDataset(
                torch.FloatTensor(X_tr_seq),
                torch.FloatTensor(y_tr_seq),
                torch.FloatTensor(y_tr_dir),
            )
            val_ds = TensorDataset(
                torch.FloatTensor(X_va_seq),
                torch.FloatTensor(y_va_seq),
                torch.FloatTensor(y_va_dir),
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            _dev = self.device
            train_batches = lambda: (
                tuple(t.to(_dev) for t in batch) for batch in train_loader
            )
            val_batches = lambda: (
                tuple(t.to(_dev) for t in batch) for batch in val_loader
            )
            n_val_batches = len(val_loader)

        autocast, scaler = tp.amp_components(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=p['learning_rate'],
            weight_decay=p['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )
        price_loss_fn = nn.MSELoss()
        dir_loss_fn = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(p['epochs']):
            self.net.train()
            for batch_x, batch_y, batch_dir in train_batches():
                # AMP acikken forward fp16; loss FP32'de hesaplanir (BCELoss
                # autocast altinda guvenli degil).
                with autocast():
                    pred_price, pred_dir = self.net(batch_x)
                loss = (price_loss_fn(pred_price.float(), batch_y)
                        + 0.3 * dir_loss_fn(pred_dir.float(), batch_dir))

                tp.backward_step(loss, optimizer, scaler, self.net.parameters())

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, batch_dir in val_batches():
                    with autocast():
                        pred_price, pred_dir = self.net(batch_x)
                    val_loss += price_loss_fn(pred_price.float(), batch_y).item()

            val_loss /= max(n_val_batches, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= p['patience']:
                    logger.info(f"  Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)

        val_pred_norm = self._predict_sequences(X_va_seq)
        val_pred = val_pred_norm * self.target_std + self.target_mean
        y_val_actual = y_va_seq * self.target_std + self.target_mean

        return self.compute_metrics(y_val_actual, val_pred)

    def _predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Sekans verisinden fiyat tahmini uret."""
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            pred_price, _ = self.net(X_tensor)
            return pred_price.cpu().numpy()

    def _predict_sequences_with_direction(self, X_seq: np.ndarray):
        """Sekans verisinden hem fiyat hem yon olasiligi uret."""
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            pred_price, pred_dir = self.net(X_tensor)
            return pred_price.cpu().numpy(), pred_dir.cpu().numpy()

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Ham tahmin — duz veriyi sekansa cevirip tahmin eder."""
        lookback = self.params.get('lookback', 30)

        if X.ndim == 2 and X.shape[0] >= lookback:
            X_norm = (X - self.scaler_mean) / self.scaler_std
            sequences = []
            for i in range(lookback, len(X_norm) + 1):
                sequences.append(X_norm[i - lookback:i])
            X_seq = np.array(sequences)
            pred_norm = self._predict_sequences(X_seq)
            return pred_norm * self.target_std + self.target_mean
        elif X.ndim == 3:
            pred_norm = self._predict_sequences(X)
            return pred_norm * self.target_std + self.target_mean
        else:
            X_norm = (X - self.scaler_mean) / self.scaler_std
            X_seq = X_norm.reshape(1, -1, X.shape[-1]) if X.ndim == 2 else X_norm.reshape(1, 1, -1)
            pred_norm = self._predict_sequences(X_seq)
            return pred_norm * self.target_std + self.target_mean

    def _predict_direction_raw(self, X: np.ndarray):
        """Direction head'den yon olasiligi uret (sigmoid, 0-1)."""
        if self.net is None:
            return None
        lookback = self.params.get('lookback', 30)
        if X.ndim == 2 and X.shape[0] >= lookback:
            X_norm = (X - self.scaler_mean) / self.scaler_std
            sequences = [X_norm[i - lookback:i] for i in range(lookback, len(X_norm) + 1)]
            X_seq = np.array(sequences)
        elif X.ndim == 3:
            X_seq = X
        else:
            return None
        _, dir_probs = self._predict_sequences_with_direction(X_seq)
        return dir_probs

    def _save_model(self, path: str):
        state = {
            'net_state': self.net.state_dict() if self.net else None,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'params': self.params,
        }
        torch.save(state, path + '.pt')

    def _load_model(self, path: str):
        state = torch.load(path + '.pt', map_location=self.device, weights_only=False)
        self.scaler_mean = state['scaler_mean']
        self.scaler_std = state['scaler_std']
        self.target_mean = state['target_mean']
        self.target_std = state['target_std']
        self.params = state.get('params', self.params)

        if state.get('net_state'):
            n_features = self.scaler_mean.shape[0]
            self.net = _BiLSTMNet(
                input_size=n_features,
                hidden_size=self.params['hidden_size'],
                num_layers=self.params['num_layers'],
                dropout=self.params['dropout'],
            ).to(self.device)
            self.net.load_state_dict(state['net_state'])
            self.net.eval()
