"""
TimesFM-3 Tahmin Modeli (sifir-atis, egitimsiz)

Google Research'un onceden egitilmis zaman serisi temel modeli. Diger base
modeller gibi satir bazli X alir ama X'i ozellik matrisi olarak degil, getiri
serisinin ve ek degiskenlerin kaynagi olarak kullanir:

- Hedef baglam: X'in `log_return_1d` kolonu. Bu kolon t satirinda
  log(P_{t-1}/P_{t-2})'dir (feature_engineer iki kez kaydiriyor), hedef ise
  log(P_{t+1}/P_t). Yani baglam r_{t-1}'de biter, hedef ufkun 2. adimidir —
  diger modellerle AYNI bilgi kumesi.
- Gecmis-yalnizca ek degiskenler: faiz, enflasyon, USD/TRY (X'teki makro
  kolonlari; makro hatti ne tasiyorsa — sizinti dahil — onu tasir).
- Gelecegi bilinen ek degiskenler: haftanin gunu ve ay (sin/cos). Gelecek
  adimlar is gunu varsayimiyla uretilir; tatiller ve ay donumu yok sayilir.

Baglam X'in kendi satirlarindan kurulur (BiLSTM'in lookback penceresi gibi):
ilk `min_context - 1` satir icin tahmin uretilmez, dizi sondan hizalanir.

Lisans: TimesFM 3.0 agirliklari `timesfm-non-commercial-license-v1.0` —
ticari / uretim kullanimi YASAK. Arastirma ve kagit portfoy icin uygun.
"""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from prediction.models.base import BasePredictionModel

logger = logging.getLogger(__name__)

QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# q90 - q10 = 2 * 1.2816 sigma (normal varsayimi)
_Q10_Q90_SIGMAS = 2.0 * norm.ppf(0.9)

# 330M parametre — model ornegi basina degil, surec basina BIR kez yuklenir.
_FORECASTERS: Dict[Tuple[str, str], Any] = {}
_FORECASTER_LOCK = threading.Lock()

# Ayni baglam (egitim/val/meta/test ve %80 yeniden egitim boyunca tekrar
# tekrar gelir) bir kez hesaplanir. Anahtar: parametreler + baglam baytlari.
_CACHE_MAX = 200_000
_CACHE: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _get_forecaster(checkpoint: str, device: Optional[str]):
    key = (checkpoint, device or 'auto')
    with _FORECASTER_LOCK:
        if key not in _FORECASTERS:
            from timesfm3 import TimesFM3Forecaster
            logger.info(f"  [timesfm] {checkpoint} yukleniyor (device={key[1]})...")
            _FORECASTERS[key] = TimesFM3Forecaster.from_pretrained(checkpoint, device=device)
        return _FORECASTERS[key]


def clear_cache():
    """Tahmin onbellegini bosalt (testler ve deneyler icin)."""
    with _CACHE_LOCK:
        _CACHE.clear()


class TimesFMModel(BasePredictionModel):
    """TimesFM-3 sifir-atis tahmin modeli (egitim yok, yalnizca cikarim)."""

    MODEL_TYPE = 'timesfm'

    CALENDAR_COLS = ('day_of_week', 'month')

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'checkpoint': 'google/timesfm-3.0-pytorch',
            'device': None,             # None = cuda varsa cuda
            'context_len': 512,
            'min_context': 64,
            'batch_size': 32,
            'target_col': 'log_return_1d',
            # Faiz, enflasyon, dolar kuru — yoksa sessizce atlanir (log'a yazilir)
            'past_covariates': ['macro_policy_rate', 'macro_cpi', 'macro_usdtry_return'],
            'calendar_covariates': True,
        }

    def get_optuna_search_space(self, trial) -> Dict[str, Any]:
        params = self.get_default_params()
        params['context_len'] = trial.suggest_categorical('context_len', [256, 512, 1024])
        params['calendar_covariates'] = trial.suggest_categorical(
            'calendar_covariates', [True, False])
        return params

    def _build_model(self, params: Dict[str, Any]):
        self.model = _get_forecaster(params['checkpoint'], params.get('device'))
        return self.model

    # ------------------------------------------------------------------
    # Kolon cozumleme
    # ------------------------------------------------------------------

    def _resolve_columns(self):
        cols = self.feature_cols
        target = self.params['target_col']
        if target not in cols:
            raise ValueError(
                f"timesfm: hedef baglam kolonu '{target}' ozelliklerde yok "
                f"(returns grubu kapali mi?)"
            )
        self._target_idx = cols.index(target)

        wanted = list(self.params.get('past_covariates') or [])
        self._past_idx = [cols.index(c) for c in wanted if c in cols]
        missing = [c for c in wanted if c not in cols]
        if missing:
            logger.warning(f"  [timesfm] ek degisken bulunamadi, atlaniyor: {missing}")

        self._cal_idx = None
        if self.params.get('calendar_covariates'):
            if all(c in cols for c in self.CALENDAR_COLS):
                self._cal_idx = [cols.index(c) for c in self.CALENDAR_COLS]
            else:
                logger.warning("  [timesfm] takvim kolonlari yok, takvim ek degiskeni kapali")

    def _fit(self, X_train, y_train, X_val, y_val) -> Dict[str, float]:
        # Sifir-atis: agirlik guncellenmez. Yalnizca kolonlar cozulur.
        self._resolve_columns()
        return {}

    # ------------------------------------------------------------------
    # Cikarim
    # ------------------------------------------------------------------

    def _steps(self) -> int:
        """Hedef ufku. Baglam r_{t-1}'de bittigi icin +1 adim (r_t) atlanir."""
        return 1 if self.horizon == 'daily' else 5

    def _calendar(self, X: np.ndarray, rows: slice, n_future: int) -> np.ndarray:
        """(4, len(rows) + n_future): dow sin/cos, ay sin/cos."""
        dow = X[rows, self._cal_idx[0]].astype(float)
        month = X[rows, self._cal_idx[1]].astype(float)
        last_dow, last_month = dow[-1], month[-1]
        fut_dow = (last_dow + np.arange(1, n_future + 1)) % 5
        dow = np.concatenate([dow, fut_dow])
        month = np.concatenate([month, np.full(n_future, last_month)])
        return np.stack([
            np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        ]).astype(np.float32)

    def _query(self, X: np.ndarray, i: int, n_future: int):
        start = max(0, i + 1 - self.params['context_len'])
        rows = slice(start, i + 1)
        target = X[rows, self._target_idx].astype(np.float32)[None, :]
        past = X[rows][:, self._past_idx].T.astype(np.float32) if self._past_idx else None
        cal = self._calendar(X, rows, n_future) if self._cal_idx is not None else None
        return target, past, cal

    @staticmethod
    def _key(params_sig: bytes, *arrays) -> bytes:
        h = hashlib.blake2b(params_sig, digest_size=20)
        for a in arrays:
            if a is not None:
                h.update(np.ascontiguousarray(a).tobytes())
                h.update(str(a.shape).encode())
        return h.digest()

    def _forecast_rows(self, X: np.ndarray, rows: List[int]) -> np.ndarray:
        """Verilen satirlar icin (len(rows), 9) hedef-ufku dilimleri."""
        if not hasattr(self, '_target_idx'):
            self._resolve_columns()
        if self.model is None:
            self._build_model(self.params)

        n_future = self._steps() + 1
        sig = json.dumps([self.params.get(k) for k in (
            'checkpoint', 'context_len', 'target_col', 'past_covariates',
            'calendar_covariates')], default=str).encode() + self.horizon.encode()

        out = np.empty((len(rows), len(QUANTILE_LEVELS)), dtype=np.float64)
        pending = []  # (pozisyon, anahtar, sorgu)
        with _CACHE_LOCK:
            for pos, i in enumerate(rows):
                q = self._query(X, i, n_future)
                key = self._key(sig, *q)
                hit = _CACHE.get(key)
                if hit is not None:
                    _CACHE.move_to_end(key)
                    out[pos] = hit
                else:
                    pending.append((pos, key, q))

        bs = self.params['batch_size']
        for b in range(0, len(pending), bs):
            chunk = pending[b:b + bs]
            res = list(self.model.predict_batch(
                contexts=[q[0] for _, _, q in chunk],
                horizon=n_future,
                past_only_covariates=[q[1] for _, _, q in chunk],
                past_future_covariates=[q[2] for _, _, q in chunk],
                return_quantiles=True,
            ))
            with _CACHE_LOCK:
                for (pos, key, _), r in zip(chunk, res):
                    qs = self._aggregate(np.asarray(r.quantiles)[0])
                    out[pos] = qs
                    _CACHE[key] = qs
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
        return out

    def _aggregate(self, step_q: np.ndarray) -> np.ndarray:
        """(n_future, 9) adim dilimlerinden hedef ufkunun 9 dilimi.

        Gunluk: dogrudan 2. adim (r_{t+1}). Haftalik: adim 2..6'nin toplami;
        dilimler toplanamaz, normal yaklasimiyla birlestirilir.
        """
        steps = step_q[1:1 + self._steps()]
        if len(steps) == 1:
            return steps[0]
        mu = steps[:, 4].sum()
        sigma = np.sqrt(np.sum(((steps[:, 8] - steps[:, 0]) / _Q10_Q90_SIGMAS) ** 2))
        return mu + norm.ppf(QUANTILE_LEVELS) * sigma

    def _rows_for(self, X: np.ndarray) -> List[int]:
        first = self.params['min_context'] - 1
        return list(range(first, len(X)))

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """(n - min_context + 1, 9) hedef dilimleri — sondan hizali."""
        return self._forecast_rows(X, self._rows_for(X))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(X)[:, 4]

    @staticmethod
    def prob_up_from_quantiles(qs: np.ndarray) -> np.ndarray:
        """P(hedef > 0): medyan ve q10-q90 bandindan normal yaklasimi."""
        qs = np.atleast_2d(qs)
        sigma = np.maximum((qs[:, 8] - qs[:, 0]) / _Q10_Q90_SIGMAS, 1e-12)
        return np.clip(norm.cdf(qs[:, 4] / sigma), 0.01, 0.99)

    def _predict_direction_raw(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self.prob_up_from_quantiles(self.predict_quantiles(X))

    def _predict_last(self, X: np.ndarray) -> Tuple[float, Optional[float]]:
        # predict_next tum gecmisi verir; binlerce satiri hesaplamak yerine son satir.
        if len(X) < self.params['min_context']:
            raise ValueError(f"timesfm: en az {self.params['min_context']} satir gerekli")
        qs = self._forecast_rows(X, [len(X) - 1])
        return float(qs[0, 4]), float(self.prob_up_from_quantiles(qs)[0])

    # ------------------------------------------------------------------
    # Kaydetme / Yukleme — agirlik yok, yalnizca parametreler
    # ------------------------------------------------------------------

    def _save_model(self, path: str):
        with open(path + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=2)

    def _load_model(self, path: str):
        with open(path + '.json', 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self.model = None  # ilk tahminde paylasilan ornekten alinir
