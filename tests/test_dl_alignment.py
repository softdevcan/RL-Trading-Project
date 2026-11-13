"""Faz 9 B11 (Y22) — BiLSTM/TFT egitim <-> cikarim hizalamasi ve yon basi etiketi.

    python tests/test_dl_alignment.py

Kabul (plan §9.17 B11 tasarimi):
  - Egitim penceresi satir i'yi icerir (X[i-L+1:i+1] -> y[i]); _predict_raw'in cikarim
    penceresiyle ayni satira hizali. Davranis testi: y[i] YALNIZ satir i'nin ozelligine bagli
    (bagimsiz gurultu satirlari). Dogru hizalamada model y'yi ogrenir (korelasyon yuksek); eski
    hizalamada (X[i-L:i]) satir i pencerede yoktur, y ogrenilemez.
  - Yon etiketi ham hedeften: log_return -> y > 0; abs_price -> y[i] > y[i-1].
  - Mutasyon uretim yolundan: _create_sequences eski tanima cevrilince korelasyon duser.
CPU, kucuk model, < 2 dk.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from prediction.models.lstm_model import BiLSTMModel  # noqa: E402
from prediction.models.tft_model import TFTModel  # noqa: E402

FAILED: list[str] = []
PASSED: list[str] = []
L = 5


def check(name, ok, detail=""):
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    (PASSED if ok else FAILED).append(name)


def data(n=900, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (0.02 * X[:, 0]).astype(np.float32)  # yalniz AYNI satirin ozelligi; isareti karisik
    return X, y


def old_sequences(self, X, y, lookback, y_raw=None):  # B11 oncesi tanim (mutasyon)
    sx, sy, sd = [], [], []
    for i in range(lookback, len(X)):
        sx.append(X[i - lookback:i]); sy.append(y[i])
        sd.append(float(y[i] > y[i - 1]) if i > 0 else 0.5)
    return np.array(sx), np.array(sy), np.array(sd)


def fit_corr(cls, params, mutate=False):
    import torch
    torch.manual_seed(0); np.random.seed(0)  # hakem-2 B11 #2: tohumlu, tekrar uretilebilir
    X, y = data()
    tr, va = 600, 750
    m = cls(horizon="daily", params=dict(cls(horizon="daily").get_default_params(), **params))
    m.target_type = "log_return"
    orig = cls._create_sequences
    if mutate:
        cls._create_sequences = old_sequences
    try:
        m.train(X[:tr], y[:tr], X[tr:va], y[tr:va])
    finally:
        cls._create_sequences = orig
    pred = m._predict_raw(X[va:])
    t = y[va:][-len(pred):]
    return float(np.corrcoef(pred, t)[0, 1])


def main() -> int:
    print("\nB11 — sekans hizasi ve yon etiketi (yapisal)")
    X, y = data(n=40)
    for cls in (BiLSTMModel, TFTModel):
        m = cls(horizon="daily")
        m.target_type = "log_return"
        sx, sy, sd = m._create_sequences(X, y, L, y)
        name = cls.__name__
        check(f"{name}: sekans sayisi n - L + 1 (cikarimla ayni)", len(sx) == len(X) - L + 1, str(len(sx)))
        check(f"{name}: penceresinin son satiri hedefin satiri (X[i] -> y[i])",
              all(np.array_equal(sx[j][-1], X[j + L - 1]) and sy[j] == y[j + L - 1] for j in range(len(sx))))
        check(f"{name}: log_return yon etiketi = (y > 0)", np.array_equal(sd, (y[L - 1:] > 0).astype(float)))
        m.target_type = "abs_price"
        price = 100 + np.cumsum(y)
        _, _, sd2 = m._create_sequences(X, price, L, price)
        check(f"{name}: abs_price gunluk yon etiketi = (y[i] > y[i-1])",
              np.array_equal(sd2, (price[L - 1:] > price[L - 2:-1]).astype(float)))
        mw = cls(horizon="weekly"); mw.target_type = "abs_price"
        _, _, sdw = mw._create_sequences(X, price, L, price)
        check(f"{name}: abs_price haftalik yon etiketi = (y[i] > y[i-5]) (hakem-2 #4)",
              np.array_equal(sdw, np.array([float(price[i] > price[i - 5]) if i >= 5 else 0.5
                                        for i in range(L - 1, len(price))])))
        seen = {}
        orig = cls._create_sequences

        def spy(self, Xs, ys, lb, y_raw=None):
            seen.setdefault("raw", y_raw)
            return orig(self, Xs, ys, lb, y_raw)
        cls._create_sequences = spy
        try:
            mf = cls(horizon="daily", params=dict(cls(horizon="daily").get_default_params(),
                                                  lookback=L, epochs=1, patience=1, hidden_size=8))
            mf.target_type = "log_return"
            yb = (y + 0.05).astype(np.float32)  # ortalamasi sifirdan uzak: normalize isaret farkli
            mf.train(X, yb, X, yb)
        finally:
            cls._create_sequences = orig
        check(f"{name}: _fit yon etiketine ham y'yi iletiyor (y_raw = y_train; hakem-2 #3)",
              seen.get("raw") is not None and np.array_equal(seen["raw"], yb))

    print("\nB11 — davranis: y yalniz ayni satira bagli")
    small = {"lookback": L, "epochs": 40, "patience": 40, "batch_size": 64}
    for cls, extra in ((BiLSTMModel, {"hidden_size": 16, "num_layers": 1}),
                       (TFTModel, {"hidden_size": 16, "num_heads": 2})):
        c_new = fit_corr(cls, dict(small, **extra))
        c_old = fit_corr(cls, dict(small, **extra), mutate=True)
        check(f"{cls.__name__}: dogru hizada tahmin-hedef korelasyonu yuksek (olculen {c_new:.3f})", c_new > 0.8)
        # Eski hizada y ogrenilemez; kalan korelasyon gurultuye asiri uyum (isaret rastgele)
        check(f"{cls.__name__}: mutasyon (eski X[i-L:i]) |korelasyon| < 0,5 (olculen {c_old:.3f})",
              abs(c_old) < 0.5)

    print("\n" + "=" * 60)
    from prediction.models.ensemble import StackingEnsemble
    for tt in ("abs_price", "log_return"):
        e = StackingEnsemble(horizon="daily", target_type=tt)
        check(f"ensemble({tt}) -> BiLSTM/TFT target_type == {tt} (hakem-2 #3)",
              e._create_model("bilstm").target_type == tt and e._create_model("tft").target_type == tt)
    print(f"Sonuc: {len(PASSED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
