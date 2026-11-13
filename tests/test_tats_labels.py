"""Faz 9 B10 (Y20) — TATS uyum girdileri: meta satirlarinin HAM kapanisi + gerceklesen getiri.

    python tests/test_tats_labels.py

Kabul (Faz 10 §3.1.1): bilinen trendli sentetik veride etiket isabeti ~ 1; uyum girdileri
meta bolumu tarihleriyle birebir hizali. Kapsam (hakem-2 B10): gunluk + haftalik ufuk,
ICEEMDAN kolu (feat_df['close'] filtreli seriyle uzerine yazilir — taklit), kisa OOF
donduren taban model (DL gibi; sondan kirpma sinanir). Eski girdiler (y_meta, y_base)
etiketi bozar; mutasyonlar uretim yolundan: kapanis bir gun kayik, kirpma bastan.
Gercek StackingEnsemble.train (XGBoost + LightGBM + kisa-OOF sahte model), diske yazmadan, CPU.
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
PASSED: list[str] = []
SHORT = 29  # kisa OOF modeli bastan bu kadar satir eksik dondurur (lookback gibi)


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    (PASSED if ok else FAILED).append(name)


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
        captured.update(y=np.asarray(y_prices), cur=np.asarray(current_prices), n_x=len(X))
        return orig_fit(self, X, y_prices, current_prices)

    TATSCorrector.fit = spy
    BasePredictionModel.save = lambda self, *a, **k: None  # diske yazma
    E.StackingEnsemble._save_ensemble_meta = lambda self, *a, **k: None

    # Kisa OOF donduren taban model: LightGBM'in tahminini bastan SHORT satir kirpar
    from prediction.models.lightgbm_model import LightGBMModel

    class ShortOOF(LightGBMModel):
        def _predict_raw(self, X):
            return super()._predict_raw(X)[SHORT:]

    orig_get = E.StackingEnsemble._get_model_class
    E.StackingEnsemble._get_model_class = lambda self, mt: ShortOOF if mt == "short" else orig_get(self, mt)
    cfg = {m: dict(c, enabled=(m in ("xgboost", "lightgbm"))) for m, c in E.BASE_MODEL_CONFIGS.items()}
    cfg["short"] = dict(cfg["lightgbm"], enabled=True)

    df = make_df()
    orig_inputs = E.StackingEnsemble.__dict__["_tats_inputs"]  # staticmethod nesnesi

    def train_capture(horizon, target_type="log_return", iceemdan=False, mutate=None):
        captured.clear()
        if mutate == "shift":  # kapanis bir gun geriden
            def bad(d, f, b, m, hz):
                c, fu = orig_inputs.__func__(d, f, b, m, hz)
                return np.r_[c[:1], c[:-1]], fu
            E.StackingEnsemble._tats_inputs = staticmethod(bad)
        elif mutate == "head":  # kirpma bastan: dogru uzunluk, yanlis satirlar
            def bad(d, f, b, m, hz):
                c, fu = orig_inputs.__func__(d, f, b, m, hz)
                k = len(c) - SHORT
                return np.r_[c[:k], np.full(SHORT, np.nan)], np.r_[fu[:k], np.full(SHORT, np.nan)]
            E.StackingEnsemble._tats_inputs = staticmethod(bad)
        try:
            ens = E.StackingEnsemble(horizon=horizon, use_tats=True, model_configs=cfg,
                                     target_type=target_type)
            if iceemdan:  # ICEEMDAN kolunu taklit: build_features close'u merkezli (iki yonlu) filtreyle ezer
                fe_build = ens.feature_engineer.build_features

                def filtered(*a, **k):
                    f = fe_build(*a, **k)
                    f["close"] = f["close"].rolling(9, center=True, min_periods=1).mean()
                    return f
                ens.feature_engineer.build_features = filtered
            ens.train(df, "SYN")
        finally:
            E.StackingEnsemble._tats_inputs = orig_inputs
        return ens, dict(captured)

    def truth(ens, horizon, target_type):
        h = 1 if horizon == "daily" else 5
        feat = ens.feature_engineer.build_features(df, "SYN", target_type=target_type)
        n = len(feat)
        be, me = int(n * 0.60), int(n * 0.80)
        y = np.nan_to_num(feat["target_price"].to_numpy(np.float32))
        return feat.index[be:me], h, y, be, me

    runs = [("daily", "log_return", False), ("weekly", "log_return", False),
            ("daily", "abs_price", False), ("daily", "log_return", True)]
    for horizon, tt, ice in runs:
        tag = f"{horizon}/{tt}" + ("/iceemdan" if ice else "")
        print(f"\nB10 — {tag}")
        ens, cap = train_capture(horizon, tt, ice)
        if not cap:
            check(f"[{tag}] TATS uyumu cagrildi", False)
            continue
        k = len(cap["cur"])
        meta_dates, h, y, be, me = truth(ens, horizon, tt)
        check(f"[{tag}] kisa OOF sondan kirpildi (k = meta - {SHORT})", k == len(meta_dates) - SHORT,
              f"(k={k}, meta={len(meta_dates)})")
        dates = meta_dates[-k:]
        cur = df["close"].reindex(dates).to_numpy(float)
        true_lab = labels(df["close"].shift(-h).reindex(dates).to_numpy(float), cur)
        check(f"[{tag}] mevcut fiyat = meta tarihlerindeki HAM kapanis (birebir)",
              np.array_equal(cap["cur"], cur), f"(k={k})")
        acc = float(np.mean(true_lab == labels(cap["y"], cap["cur"])))
        check(f"[{tag}] etiket isabeti (gerceklesen getiri) ~ 1 (olculen {acc:.4f}, "
              f"{len(np.unique(true_lab))} sinif)", acc > 0.999 and len(np.unique(true_lab)) == 3)
        if not ice and horizon == "daily":
            old_acc = float(np.mean(true_lab == labels(y[be:me][-k:], y[:be][-k:])))
            check(f"[{tag}] eski girdilerle (Y20) isabet dusuk (olculen {old_acc:.4f})", old_acc < 0.8)

    print("\nMutasyonlar (uretim yolu, daily/log_return)")
    _, base = train_capture("daily")
    _, cs = train_capture("daily", mutate="shift")
    check("kapanis bir gun kayik -> etiket degisir",
          not np.array_equal(labels(cs["y"], cs["cur"]), labels(base["y"], base["cur"])))
    _, ch = train_capture("daily", mutate="head")
    check("kirpma bastan -> girdiler degisir (NaN kuyruk ya da farkli satirlar)",
          not np.array_equal(ch["cur"], base["cur"], equal_nan=True))
    TATSCorrector.fit = orig_fit
    E.StackingEnsemble._get_model_class = orig_get

    print("\n" + "=" * 60)
    print(f"Sonuc: {len(PASSED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
