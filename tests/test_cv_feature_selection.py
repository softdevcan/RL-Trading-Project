"""Faz 9 B8 (Y16) — walk-forward CV'de ozellik secimi yalniz katin egitim bolumunde.

    python tests/test_cv_feature_selection.py

Kabul (Faz 10 §3.1.1): test katinin y'si karistirildiginda secilen kume DEGISMEZ.
Mutasyon: eski davranis (secim tum seride) ayni karistirmada kumeyi degistirir.
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


N, F, N_SPLITS, MAX_F = 900, 24, 4, 6
COLS = [f"f{i}" for i in range(F)]


def make_frame(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, F))
    # Egitim bolgesinde y, f0..f5'e bagli; son blokta (son katin testi) f12..f17'ye.
    y = X[:, :6].sum(axis=1)
    tail = N - N // (N_SPLITS + 1)
    y[tail:] = 3 * X[tail:, 12:18].sum(axis=1)
    df = pd.DataFrame(X, columns=COLS)
    df["target_price"] = y + 100.0
    return df


def run_cv(feat_df: pd.DataFrame) -> list:
    tr = WalkForwardTrainer(n_splits=N_SPLITS, purge_days=5, embargo_days=3,
                            select_features=True, max_features=MAX_F)
    tr.feature_engineer.build_features = lambda df, symbol, **kw: feat_df
    tr.feature_engineer.get_feature_columns = lambda df: list(COLS)
    tr._save_experiment = lambda *a, **k: None
    summary = tr.cross_validate(feat_df, "SYN", model_types=["_mean"])
    return summary.get("selected_features_per_fold", [])


def last_test_region(n=N, k=N_SPLITS, purge=5):
    fs = n // (k + 1)
    start = fs * k + purge
    return start, min(start + fs, n)


def main() -> int:
    pm.MODEL_REGISTRY["_mean"] = _Mean
    base = make_frame()
    a, b = last_test_region()
    shuffled = base.copy()
    rng = np.random.default_rng(7)
    shuffled.loc[a:b - 1, "target_price"] = rng.permutation(shuffled.loc[a:b - 1, "target_price"].to_numpy())

    print("\nB8 — kat ici secim")
    s0, s1 = run_cv(base), run_cv(shuffled)
    check("her kat icin secim raporlandi", len(s0) == N_SPLITS and len(s1) == N_SPLITS, f"({len(s0)}, {len(s1)})")
    check("secim yalniz egitim bolumunde (train_end <= son test baslangici)",
          all(r["train_end"] <= a for r in s0), str([r["train_end"] for r in s0]))
    check("son katin test y'si karisinca HER katin kumesi bit-es",
          [r["features"] for r in s0] == [r["features"] for r in s1],
          f"\n{[r['features'] for r in s0]}\n{[r['features'] for r in s1]}")

    print("\nMutasyon — eski davranis (secim tum seride)")
    sel = FeatureSelector(max_features=MAX_F)
    full0 = sel.fit_select(base[COLS], base["target_price"], COLS)["selected_features"]
    full1 = sel.fit_select(shuffled[COLS], shuffled["target_price"], COLS)["selected_features"]
    check("tum seride secim ayni karistirmada kumeyi DEGISTIRIR (test kirilabilir)",
          sorted(full0) != sorted(full1), f"({full0} vs {full1})")

    print("\n" + "=" * 60)
    print(f"Sonuc: {4 - len(FAILED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
