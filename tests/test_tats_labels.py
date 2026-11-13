"""Faz 9 B10 (Y20) — TATS uyum girdileri: meta bolumunun kendi kapanisi + gercek getiri etiketi.

    python tests/test_tats_labels.py

Kabul (Faz 10 §3.1.1): bilinen trendli sentetik veride etiket isabeti ~ 1; uyum girdileri
meta bolumu tarihleriyle birebir hizali. Eski girdiler (y_meta, y_base) ayni veride etiketi
bozar. Mutasyon uretim yolundan (abs_price; kapanis bir gun kayik) isabeti dusurur;
log_return'de kapanis sadelesir (gelecek = c * exp(y)), etiket yalniz y'ye baglidir.
Gercek StackingEnsemble.train (yalniz XGBoost + LightGBM), diske yazmadan, CPU.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prediction.models.base import BasePredictionModel  # noqa: E402
from prediction.models import ensemble as E  # noqa: E402
from prediction.tats import TATSCorrector  # noqa: E402

FAILED: list[str] = []
N_CHECKS = 9


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    if not ok:
        FAILED.append(name)


def make_df(n=700, seed=3):
    # Bilinen trend: 30 gunluk rejimler (yukari / duz / asagi) + kucuk gurultu
    rng = np.random.RandomState(seed)
    drift = np.tile(np.repeat([0.012, 0.0, -0.012], 30), n // 90 + 1)[:n]
    close = 100 * np.cumprod(1 + drift + rng.randn(n) * 0.001)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998,
                         "close": close, "volume": rng.randint(1e5, 1e6, n).astype(float)}, index=idx)


def labels(future, current, thr=0.005):
    r = (future - current) / np.where(current == 0, 1, current)
    return np.where(r > thr, 2, np.where(r < -thr, 0, 1))


def main() -> int:
    captured = {}
    orig_fit = TATSCorrector.fit

    def spy(self, X, y_prices, current_prices):
        captured.update(y=np.asarray(y_prices), cur=np.asarray(current_prices))
        return orig_fit(self, X, y_prices, current_prices)

    TATSCorrector.fit = spy
    BasePredictionModel.save = lambda self, *a, **k: None  # diske yazma
    E.StackingEnsemble._save_ensemble_meta = lambda self, *a, **k: None

    df = make_df()
    cfg = {m: dict(c, enabled=(m in ("xgboost", "lightgbm"))) for m, c in E.BASE_MODEL_CONFIGS.items()}

    def train_capture(target_type, shift=False):
        captured.clear()
        orig_close = E.StackingEnsemble.__dict__["_tats_inputs_close"]  # staticmethod nesnesi
        if shift:  # uretim yolundan mutasyon: meta kapanisi bir gun geriden okunur
            E.StackingEnsemble._tats_inputs_close = staticmethod(
                lambda f, b, m: f["close"].to_numpy(dtype=np.float64)[b - 1:m - 1])
        try:
            ens = E.StackingEnsemble(horizon="daily", use_tats=True, model_configs=cfg,
                                     target_type=target_type)
            ens.train(df, "SYN")
        finally:
            E.StackingEnsemble._tats_inputs_close = orig_close
        return ens, dict(captured)

    def truth(ens, target_type, k):
        feat = ens.feature_engineer.build_features(df, "SYN", target_type=target_type)
        n = len(feat)
        be, me = int(n * 0.60), int(n * 0.80)
        close_meta = feat["close"].to_numpy(float)[be:me][-k:]
        nxt = df["close"].shift(-1).reindex(feat.index[be:me][-k:]).to_numpy(float)
        y = np.nan_to_num(feat["target_price"].to_numpy(np.float32))
        return close_meta, labels(nxt, close_meta), y, be, me

    for tt in ("log_return", "abs_price"):
        print(f"\nB10 — TATS uyum girdileri (target_type={tt})")
        ens, cap = train_capture(tt)
        check(f"[{tt}] TATS uyumu cagrildi", bool(cap))
        if not cap:
            continue
        k = len(cap["cur"])
        close_meta, true_lab, y, be, me = truth(ens, tt, k)
        check(f"[{tt}] mevcut fiyat = meta satirlarinin kendi kapanisi (birebir)",
              np.array_equal(cap["cur"], close_meta), f"(k={k})")
        acc = float(np.mean(true_lab == labels(cap["y"], cap["cur"])))
        check(f"[{tt}] etiket isabeti ~ 1 (olculen {acc:.4f}, {len(np.unique(true_lab))} sinif)",
              acc > 0.999 and len(np.unique(true_lab)) == 3)
        # Eski girdiler (Y20): gelecek = y_meta, mevcut = y_base'in son k'si
        old_acc = float(np.mean(true_lab == labels(y[be:me][-k:], y[:be][-k:])))
        check(f"[{tt}] eski girdilerle (Y20) isabet dusuk (olculen {old_acc:.4f})", old_acc < 0.8)
        if tt == "abs_price":
            _, cs = train_capture(tt, shift=True)
            sh = float(np.mean(true_lab == labels(cs["y"], cs["cur"])))
            check(f"[{tt}] mutasyon (uretim yolu, kapanis 1 gun kayik) isabeti dusurur "
                  f"(olculen {sh:.4f})", sh < 0.99)
    TATSCorrector.fit = orig_fit

    print("\n" + "=" * 60)
    print(f"Sonuc: {N_CHECKS - len(FAILED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
