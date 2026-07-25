"""
Temporal Fusion Transformer (TFT) Tahmin Modeli

Google Research TFT makalesinden esinlenmistir (Lim et al., 2021).
Basitlestirilmis ama etkili bir implementasyon:
- Gated Residual Network (GRN)
- Variable Selection Network
- LSTM Encoder
- Multi-head Attention
- Fiyat + yon tahmini

RTX 4060 (8GB VRAM) icin optimize edilmistir.
"""

import logging
import math
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


class _GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))


class _GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = _GatedLinearUnit(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        h = self.glu(h)
        return self.layer_norm(h + residual)


class _VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — her ozelligin katkisini agirliklandirir."""

    def __init__(self, input_size: int, n_vars: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.grn_var = nn.ModuleList([
            _GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(n_vars)
        ])
        self.grn_weights = _GatedResidualNetwork(input_size, hidden_size, n_vars, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.grn_weights(x))

        var_outputs = []
        for i in range(min(self.n_vars, x.shape[-1])):
            var_input = x[..., i:i + 1]
            var_outputs.append(self.grn_var[i](var_input))

        var_outputs = torch.stack(var_outputs, dim=-2)
        weights = weights[..., :var_outputs.shape[-2]].unsqueeze(-1)
        return (var_outputs * weights).sum(dim=-2)


class _GroupedGRN(nn.Module):
    """n_vars adet ozdes yapili GRN'i (girdi boyutu 1) tek batched islemde hesaplar.

    Faz 6 (3.1) — Olcum (2026-07-24, RTX 4060): TFT egitimin ~%90'iydi ve
    masrafin kaynagi VSN'in `for i in range(n_vars)` dongusuydu: 131 ozellik x
    ~6 katman = adim basina ~800 kucuk kernel. Ag kucuk oldugu icin GPU bos
    bekliyor, zaman kernel firlatma masrafinda gidiyordu.

    Buradaki matematik `_GatedResidualNetwork`(input_size=1) ile birebir ayni;
    fark yalnizca degiskenlerin ayri ayri degil yiginlanmis parametrelerle
    hesaplanmasi. Init de ayni dagilimdan gelir (asagida gercek nn.Linear
    modullerinden yiginlanir).

    NOT (egitim determinizmi): dropout maskesi tek seferde (B,T,V,H) icin
    cekilir, dongudeki V ayri cekilise gore farkli bir RNG tuketimidir —
    egitim yorungesi degisir. `eval()` modunda dropout kapali oldugundan
    cikarim (inference) birebir aynidir.
    """

    def __init__(self, n_vars: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-5

        def _stack(make):
            """V adet gercek nn.Linear kurup agirliklarini yiginla — boylece
            baslangic dagilimi dongulu surumle ayni kalir."""
            mods = [make() for _ in range(n_vars)]
            w = torch.stack([m.weight.detach().clone() for m in mods])
            b = torch.stack([m.bias.detach().clone() for m in mods])
            return nn.Parameter(w), nn.Parameter(b)

        self.w1, self.b1 = _stack(lambda: nn.Linear(1, hidden_size))
        self.w2, self.b2 = _stack(lambda: nn.Linear(hidden_size, hidden_size))
        self.w_fc, self.b_fc = _stack(lambda: nn.Linear(hidden_size, output_size))
        self.w_gate, self.b_gate = _stack(lambda: nn.Linear(hidden_size, output_size))
        self.w_skip, self.b_skip = _stack(lambda: nn.Linear(1, output_size))

        self.ln_weight = nn.Parameter(torch.ones(n_vars, output_size))
        self.ln_bias = nn.Parameter(torch.zeros(n_vars, output_size))

    def forward(self, x):
        """x: (B, T, V) — her degisken skaler girdi. Cikti: (B, T, V, output).

        V, kurulumdaki n_vars'tan kucuk gelebilir (dongulu surumdeki
        `min(n_vars, x.shape[-1])` kapaginin karsiligi) — parametreler dilimlenir.
        """
        v = x.shape[-1]
        xu = x.unsqueeze(-1)                                        # (B,T,V,1)

        residual = xu * self.w_skip[:v].squeeze(-1) + self.b_skip[:v]
        h = torch.nn.functional.elu(xu * self.w1[:v].squeeze(-1) + self.b1[:v])
        h = torch.einsum('btvh,vkh->btvk', h, self.w2[:v]) + self.b2[:v]
        h = self.dropout(h)

        gate_in = torch.einsum('btvh,vkh->btvk', h, self.w_gate[:v]) + self.b_gate[:v]
        h = ((torch.einsum('btvh,vkh->btvk', h, self.w_fc[:v]) + self.b_fc[:v])
             * torch.sigmoid(gate_in))

        z = h + residual
        mean = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, unbiased=False, keepdim=True)
        z = (z - mean) / torch.sqrt(var + self.eps)
        return z * self.ln_weight[:v] + self.ln_bias[:v]


class _FastVariableSelectionNetwork(nn.Module):
    """VSN — degisken bazli GRN'ler gruplanmis (bkz. _GroupedGRN)."""

    def __init__(self, input_size: int, n_vars: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.grn_vars = _GroupedGRN(n_vars, hidden_size, hidden_size, dropout)
        self.grn_weights = _GatedResidualNetwork(input_size, hidden_size, n_vars, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.grn_weights(x))
        n = min(self.n_vars, x.shape[-1])
        var_outputs = self.grn_vars(x[..., :n])                 # (B,T,n,H)
        weights = weights[..., :n].unsqueeze(-1)
        return (var_outputs * weights).sum(dim=-2)


class _TFTNet(nn.Module):
    """Basitlestirilmis Temporal Fusion Transformer agi."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_lstm_layers: int = 1,
        dropout: float = 0.1,
        fast_vsn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fast_vsn = fast_vsn

        self.input_proj = nn.Linear(input_size, hidden_size)

        # fast_vsn: degisken bazli GRN'ler gruplanir (ayni matematik, tek
        # batched islem). Parametre duzeni farkli oldugu icin checkpoint'e
        # `params['fast_vsn']` ile yazilir ve yuklerken ayni varyant kurulur.
        vsn_cls = _FastVariableSelectionNetwork if fast_vsn else _VariableSelectionNetwork
        self.vsn = vsn_cls(
            input_size=input_size,
            n_vars=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_glu = _GatedLinearUnit(hidden_size, hidden_size)

        self.grn_final = _GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        self.fc_price = nn.Linear(hidden_size, 1)
        self.fc_direction = nn.Linear(hidden_size, 1)

    def forward(self, x):
        vsn_out = self.vsn(x)
        projected = self.input_proj(x) + vsn_out

        lstm_out, _ = self.lstm(projected)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(self.attn_glu(attn_out) + lstm_out)

        final = self.grn_final(attn_out[:, -1, :])

        price = self.fc_price(final).squeeze(-1)
        direction = torch.sigmoid(self.fc_direction(final)).squeeze(-1)
        return price, direction


class TFTModel(BasePredictionModel):
    """Temporal Fusion Transformer fiyat tahmin modeli."""

    MODEL_TYPE = 'tft'

    def __init__(self, horizon: str = 'daily', params: Optional[Dict] = None,
                 source: Optional[str] = None):
        super().__init__(horizon, params, source)
        self.device = _get_device()
        self.net: Optional[_TFTNet] = None
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'hidden_size': 64,
            'num_heads': 4,
            'num_lstm_layers': 1,
            'dropout': 0.1,
            'lookback': 30,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'epochs': 100,
            'patience': 15,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        return {
            'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
            'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
            'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 2),
            'dropout': trial.suggest_float('dropout', 0.05, 0.4),
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

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, lookback: int):
        sequences_X, sequences_y, sequences_y_dir = [], [], []
        for i in range(lookback, len(X)):
            sequences_X.append(X[i - lookback:i])
            sequences_y.append(y[i])
            sequences_y_dir.append(float(y[i] > y[i - 1]) if i > 0 else 0.5)
        return np.array(sequences_X), np.array(sequences_y), np.array(sequences_y_dir)

    def _fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        # Epic 2.2: paralel sembol egitiminde GPU yuvasi (seri kosumda etkisiz).
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
        # Faz 6 (3.1): hangi VSN varyantiyla egitildigi parametrelere yazilir —
        # checkpoint bu bilgiyle yuklenir, config sonradan degisse bile model
        # kendi mimarisiyle acilir.
        p['fast_vsn'] = tp.fast_vsn_enabled()
        self.params = p

        self.net = _TFTNet(
            input_size=n_features,
            hidden_size=p['hidden_size'],
            num_heads=p['num_heads'],
            num_lstm_layers=p['num_lstm_layers'],
            dropout=p['dropout'],
            fast_vsn=p['fast_vsn'],
        ).to(self.device)

        # Faz 6 (2.1): warm-start — onceki agirliklardan basla (ayni mimari sart).
        # Sadece ensemble warm_start=True verdiginde dolu; aksi halde None =
        # sifirdan (mevcut davranis, bit-es). Uyumsuzsa sessizce sifirdan.
        ws = getattr(self, '_warm_start_from', None)
        if ws is not None and getattr(ws, 'net', None) is not None:
            try:
                self.net.load_state_dict(ws.net.state_dict())
                logger.info("  [tft] warm-start: onceki agirliklar yuklendi")
            except Exception as exc:
                logger.info(f"  [tft] warm-start atlandi (mimari uyumsuz?): {exc}")

        # Faz 6 (3.1/B7): TFT olcumde egitimin ~%90'i (sembol basina ~135s,
        # epoch ~3s) — bu buyuklukteki bir ag icin masraf batch basina Python
        # + transferde. Sekanslar bir kez cihaza tasinir; batch sinirlari
        # DataLoader ile ayni (shuffle=False), sayisal sonuc degismez.
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
            self.net.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p['epochs'])
        price_loss_fn = nn.HuberLoss(delta=1.0)
        dir_loss_fn = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(p['epochs']):
            self.net.train()
            for batch_x, batch_y, batch_dir in train_batches():
                # AMP acikken forward fp16; loss FP32 (BCELoss autocast altinda
                # guvenli degil).
                with autocast():
                    pred_price, pred_dir = self.net(batch_x)
                loss = (price_loss_fn(pred_price.float(), batch_y)
                        + 0.3 * dir_loss_fn(pred_dir.float(), batch_dir))

                tp.backward_step(loss, optimizer, scaler, self.net.parameters())

            scheduler.step()

            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y, batch_dir in val_batches():
                    with autocast():
                        pred_price, _ = self.net(batch_x)
                    val_loss += price_loss_fn(pred_price.float(), batch_y).item()
            val_loss /= max(n_val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= p['patience']:
                    logger.info(f"  TFT early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)

        val_pred_norm = self._predict_sequences(X_va_seq)
        val_pred = val_pred_norm * self.target_std + self.target_mean
        y_val_actual = y_va_seq * self.target_std + self.target_mean
        return self.compute_metrics(y_val_actual, val_pred)

    def _predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            pred_price, _ = self.net(X_tensor)
            return pred_price.cpu().numpy()

    def _predict_sequences_with_direction(self, X_seq: np.ndarray):
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            pred_price, pred_dir = self.net(X_tensor)
            return pred_price.cpu().numpy(), pred_dir.cpu().numpy()

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
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
            self.net = _TFTNet(
                input_size=n_features,
                hidden_size=self.params['hidden_size'],
                num_heads=self.params['num_heads'],
                num_lstm_layers=self.params['num_lstm_layers'],
                dropout=self.params['dropout'],
                # Eski checkpoint'lerde bu anahtar yok -> dongulu (eski) VSN.
                fast_vsn=bool(self.params.get('fast_vsn', False)),
            ).to(self.device)
            self.net.load_state_dict(state['net_state'])
            self.net.eval()
