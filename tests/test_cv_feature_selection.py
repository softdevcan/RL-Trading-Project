"""Faz 9 B8 (Y16) — walk-forward CV'de ozellik secimi yalniz katin egitim bolumunde.

    python tests/test_cv_feature_selection.py

Kabul (Faz 10 §3.1.1): test katinin y'si karistirildiginda secilen kume DEGISMEZ.
Her kat kendi test bolumuyle sinanir (hakem-finans B8 #1). Mutasyon egitim hattindan:
trainer'in secicisi tum seride secime cevrilince ayni karistirma kumeyi degistirir (hakem-2 #1).
Sentetik veri + sahte model; gercek FeatureSelector. Ag ve GPU yok, < 1 dk.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prediction.models as pm  # noqa: E402
from prediction.feature_selector import FeatureSelector  # noqa: E402
from prediction.trainer import WalkForwardTrainer  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}]   {name} {'' if ok else detail}")
    if not ok:
        FAILED.append(name)


class _Mean:
    """Ucuz sahte model: egitim ortalamasini tahmin eder."""
    def __init__(self, horizon="daily"):
        self.m = 0.0

    def train(self, X_tr, y_tr, X_val, y_val, feature_cols=None):
        self.m = float(np.mean(y_tr))

    def _predict_raw(self, X):
        return np.full(len(X), self.m, dtype=np.float32)


N, N_SPLITS, MAX_F = 900, 4, 6
F = 6 * (N_SPLITS + 1)
COLS = [f"f{i}" for i in range(F)]
FS = N // (N_SPLITS + 1)


def make_frame(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, F))
    # Blok j'de y yalniz f[6j..6j+5]'e bagli: kat k'nin test bolgesi (blok k+1) egitimin
    # gormedigi bir rejim tasir; secim onu gorurse kume degisir.
    y = np.zeros(N)
    for j in range(N_SPLITS + 1):
        a, b = j * FS, (N if j == N_SPLITS else (j + 1) * FS)
        w = 1.0 + j  # sonraki rejim daha guclu: egitim+test'te secimi ele gecirir
        y[a:b] = w * X[a:b, 6 * j:6 * j + 6].sum(axis=1)
    df = pd.DataFrame(X, columns=COLS)
    df["target_price"] = y + 100.0
    return df


def run_cv(feat_df: pd.DataFrame, leaky: pd.DataFrame | None = None) -> list:
    """leaky verilirse trainer'in FeatureSelector'u eski davranisa (tum seride secim) cevrilir."""
    import prediction.trainer as T

    class _Leaky(FeatureSelector):
        def fit_select(self, X, y, cols, **kw):
            return super().fit_select(leaky[COLS], leaky["target_price"], cols, **kw)

    orig = T.FeatureSelector
    T.FeatureSelector = _Leaky if leaky is not None else orig
    try:
        tr = WalkForwardTrainer(n_splits=N_SPLITS, purge_days=5, embargo_days=3,
                                select_features=True, max_features=MAX_F)
        tr.feature_engineer.build_features = lambda df, symbol, **kw: feat_df
        tr.feature_engineer.get_feature_columns = lambda df: list(COLS)
        tr._save_experiment = lambda *a, **k: None
        summary = tr.cross_validate(feat_df, "SYN", model_types=["_mean"])
    finally:
        T.FeatureSelector = orig
    assert summary.get("feature_selection") == "per_fold_train", summary.get("feature_selection")
    return summary.get("selected_features_per_fold", [])


def test_region(k, purge=5):
    start = FS * (k + 1) + purge
    return start, min(start + FS, N)


def shuffled_at(df, a, b, seed=7):
    out = df.copy()
    out.loc[a:b - 1, "target_price"] = np.random.default_rng(seed).permutation(
        out.loc[a:b - 1, "target_price"].to_numpy())
    return out


def main() -> int:
    pm.MODEL_REGISTRY["_mean"] = _Mean
    base = make_frame()
    s0 = run_cv(base)
    print("\nB8 — kat ici secim (her kat KENDI test bolumuyle)")
    check("her kat icin secim raporlandi", len(s0) == N_SPLITS, f"({len(s0)})")
    for k in range(N_SPLITS):
        a, b = test_region(k)
        r = s0[k]
        check(f"kat {k + 1}: train_end ({r['train_end']}) = test baslangici ({a}) - purge",
              r["train_end"] == a - 5)
        sh = shuffled_at(base, a, b)
        sk = run_cv(sh)
        check(f"kat {k + 1}: kendi test y'si karisinca kumesi bit-es",
              sk[k]["features"] == r["features"], f"({r['features']} vs {sk[k]['features']})")
        # Mutasyon EGITIM HATTINDAN (hakem-2 B8 #1): trainer'in secicisi eski davranisa
        # (tum seride secim) dondurulunce ayni karistirma bu katin kumesini degistirir.
        m0, m1 = run_cv(base, leaky=base), run_cv(sh, leaky=sh)
        check(f"kat {k + 1}: mutasyon (trainer tum seride seciyor) kumeyi DEGISTIRIR",
              m0[k]["features"] != m1[k]["features"], f"({m0[k]['features']} vs {m1[k]['features']})")

    n = 1 + 3 * N_SPLITS
    print("\n" + "=" * 60)
    print(f"Sonuc: {n - len(FAILED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
