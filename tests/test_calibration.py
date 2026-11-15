"""
Faz 9 / F1-F4 — yon olasiligi, kalibrasyon (ECE 15 kutu, Brier, sicaklik), ham p saklama.

Kaynak: Guo vd. (2017, ICML) — ECE esit genislikli M=15 kutu; sicaklik olcekleme NLL ile.

  [A] evaluation/calibration.py (saf):
      1. ECE elle hesapla ayni; Brier elle hesapla ayni.
      2. Kalibre ornek (y ~ Bernoulli(p)) -> ECE kucuk; asiri guvenli (logit x 3) -> ECE buyuk,
         fit_temperature T ~ 3 bulur ve ECE'yi dusurur.
  [B] ensemble (uretim yolu, agac modelleri + yon basli sahte model):
      3. 'legacy' (varsayilan): test kalibrasyonu olculur (F3, Y4 catisma orani); predict_next'te
         kalibre alanlar yok; fiyat tahmini iki modda AYNI.
      4. 'calibrated': p_up_raw + p_up_cal (F4); yon = 1[p_cal >= 0,5], guven = max(p, 1-p) (F1);
         fiyat yonu ayri alan + catisma bayragi.
      5. Kaydet/yukle: yuklenen modelin p_up_cal'i bellektekiyle ayni; mod modelden okunur.
      6. Sizinti bekcisi (B0 benzeri): test bolumunun kapanislari bozulunca yon modeli
         (katsayilar, T) BIT-ES. Mutasyon: arindirma 0 -> bozulma meta etiketlerine sizar, degisir.

Calistirma:
    python tests/test_calibration.py        (~1-2 dk, CPU)
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluation import calibration as cal  # noqa: E402

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


print("\n[A1] ECE ve Brier elle hesapla")
p = np.array([0.95, 0.9, 0.2, 0.3, 0.55, 0.6, 0.85, 0.1])
y = np.array([1, 0, 0, 1, 1, 0, 1, 0])
conf = np.maximum(p, 1 - p)
corr = ((p >= 0.5) == (y == 1)).astype(float)
edges = np.linspace(0, 1, 16)
manual = 0.0
for i in range(15):
    m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
    if m.any():
        manual += m.sum() / len(p) * abs(corr[m].mean() - conf[m].mean())
check("ECE (15 kutu, [0,1]) elle hesapla ayni", abs(cal.ece(p, y) - manual) < 1e-12,
      f"{cal.ece(p, y)} vs {manual}")
check("Brier elle hesapla ayni", abs(cal.brier(p, y) - np.mean((p - y) ** 2)) < 1e-12)

print("\n[A2] Kalibre / asiri guvenli / sicaklik")
rng = np.random.default_rng(0)
z = rng.normal(0, 1.2, 20000)
y_true = (rng.uniform(size=z.size) < cal.sigmoid(z)).astype(float)
e_ok = cal.ece(cal.sigmoid(z), y_true)
e_over = cal.ece(cal.sigmoid(3 * z), y_true)
T = cal.fit_temperature(3 * z, y_true)
e_fix = cal.ece(cal.apply_temperature(3 * z, T), y_true)
print(f"    ECE kalibre {e_ok:.4f}, asiri guvenli {e_over:.4f}, T={T:.3f} sonrasi {e_fix:.4f}")
check("kalibre ornekte ECE < 0,02; asiri guvenlide > 0,08", e_ok < 0.02 and e_over > 0.08)
check("fit_temperature asiri guveni bulur (|T - 3| < 0,3) ve ECE'yi kalibre duzeye indirir",
      abs(T - 3) < 0.3 and e_fix < 0.02, f"T={T:.3f}, ECE {e_fix:.4f}")


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


check("gecersiz girdi -> ValueError (y 0/1 degil, T <= 0, tek sinif)",
      raises(lambda: cal.ece([0.5], [2])) and raises(lambda: cal.apply_temperature([0.0], 0.0))
      and raises(lambda: cal.fit_temperature([0.1, 0.2], [1, 1])))

# ---------------------------------------------------------------------------
print("\n[B] Ensemble uretim yolu")
from app.auth import workspace as ws  # noqa: E402
import prediction.models.ensemble as E  # noqa: E402
from prediction.models.lightgbm_model import LightGBMModel  # noqa: E402

SHORT = 13


class HeadModel(LightGBMModel):
    """Yon basli sahte model: DL gibi kisa dondurur ve p(UP) verir (dir_ yolunu sinar)."""

    def _predict_raw(self, X):
        return super()._predict_raw(X)[SHORT:]

    def _predict_direction_raw(self, X):
        return 1 / (1 + np.exp(-200 * super()._predict_raw(X)[SHORT:]))


def make_df(n=1100, seed=3):
    r = np.random.RandomState(seed)
    drift = np.tile(np.repeat([0.006, 0.0, -0.006], 30), n // 90 + 1)[:n]
    close = 100 * np.cumprod(1 + drift + r.randn(n) * 0.006)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997,
                         "close": close, "volume": r.randint(1e5, 1e6, n).astype(float)}, index=idx)


tmp = tempfile.mkdtemp(prefix="rlt_f_")
orig_pmd, orig_rd = ws.prediction_models_dir, ws.read_dirs
ws.prediction_models_dir = lambda *a, **k: tmp
ws.read_dirs = lambda kind, *a, **k: [tmp]
orig_get = E.StackingEnsemble._get_model_class
E.StackingEnsemble._get_model_class = lambda self, mt: HeadModel if mt == "head" else orig_get(self, mt)
cfg = {m: dict(c, enabled=(m in ("xgboost", "lightgbm"))) for m, c in E.BASE_MODEL_CONFIGS.items()}
cfg["head"] = dict(cfg["lightgbm"], enabled=True)
df = make_df()


def train(mode, data=df, sym="SYN", purge=None):
    E.StackingEnsemble.CALIB_PURGE = purge
    try:
        ens = E.StackingEnsemble(horizon="daily", use_tats=False, model_configs=cfg,
                                 confidence_mode=mode)
        res = ens.train(data, sym)
    finally:
        E.StackingEnsemble.CALIB_PURGE = None
    return ens, res["ensemble_test_metrics"]


try:
    leg, tm_leg = train("legacy", sym="LEG")
    c_leg = tm_leg.get("calibration", {})
    print(f"    legacy test: ECE {c_leg.get('legacy', {}).get('ece')}, bas-fiyat catisma "
          f"{c_leg.get('legacy', {}).get('head_vs_price_conflict_rate')}")
    check("[3] legacy: test kalibrasyonu (ECE, Brier, 15 kutu, Y4 catisma orani) olculdu",
          {"ece", "brier", "reliability", "head_vs_price_conflict_rate"} <= set(c_leg.get("legacy", {}))
          and len(c_leg["legacy"]["reliability"]) == 15 and c_leg["legacy"]["has_direction_heads"],
          str(c_leg)[:300])
    check("[3] legacy: yon modeli kurulmaz", leg.direction_model is None)
    p_leg = leg.predict_next(df, "LEG")
    check("[3] legacy predict_next: kalibre alan yok, mod 'legacy'",
          p_leg["calibration"] == {"confidence_mode": "legacy"}, str(p_leg["calibration"]))

    cl, tm_cl = train("calibrated", sym="CAL")
    cc = tm_cl.get("calibration", {}).get("calibrated") or {}
    print(f"    calibrated test: T={cc.get('temperature')}, ECE ham "
          f"{cc.get('raw', {}).get('ece')} -> sicaklik {cc.get('temperature_scaled', {}).get('ece')}, "
          f"izotonik {cc.get('isotonic_compare', {}).get('ece')}; n_fit "
          f"{tm_cl['calibration'].get('n_fit')}, n_calib {tm_cl['calibration'].get('n_calib')}")
    check("[4] calibrated: yon modeli (ret_ + dir_ ozellikleri, T > 0), test ECE ham/sicaklik/izotonik",
          cl.direction_model is not None and cc.get("temperature", 0) > 0
          and any(k.startswith("dir_") for k in cl.direction_model["features"])
          and any(k.startswith("ret_") for k in cl.direction_model["features"])
          and {"raw", "temperature_scaled", "isotonic_compare", "price_vs_p_conflict_rate"} <= set(cc),
          str(tm_cl.get("calibration"))[:300])
    n_meta = int(len(leg.feature_engineer.build_features(df, "SYN")) * 0.8) - \
        int(len(leg.feature_engineer.build_features(df, "SYN")) * 0.6)
    check("[4] uyum + kalibrasyon satirlari ayrik ve arindirmali (n_fit + n_calib <= meta - 2h)",
          tm_cl["calibration"]["n_fit"] + tm_cl["calibration"]["n_calib"] <= n_meta - 2,
          f"{tm_cl['calibration']['n_fit']} + {tm_cl['calibration']['n_calib']} vs {n_meta}")
    pc = cl.predict_next(df, "CAL")
    c = pc["calibration"]
    ok = {"p_up_raw", "p_up_cal", "direction_price", "direction_conflict"} <= set(c)
    check("[4] predict_next: p_up_raw ve p_up_cal birlikte (F4); yon = 1[p_cal >= 0,5], "
          "guven = max(p, 1-p) (F1); fiyat yonu ayri + catisma bayragi",
          ok and pc["predicted_direction"] == ("UP" if c["p_up_cal"] >= 0.5 else "DOWN")
          and abs(pc["confidence"] - round(max(c["p_up_cal"], 1 - c["p_up_cal"]), 4)) < 1e-12
          and c["direction_conflict"] == (c["direction_price"] != pc["predicted_direction"]), str(c))
    check("[3] fiyat tahmini iki modda AYNI (yalniz yon/guven degisir)",
          pc["predicted_close"] == p_leg["predicted_close"]
          and pc["model_predictions"] == p_leg["model_predictions"],
          f"{pc['predicted_close']} vs {p_leg['predicted_close']}")

    loaded = E.StackingEnsemble(horizon="daily", model_configs=cfg)  # kurucu 'legacy'
    loaded.load("CAL")
    pl = loaded.predict_next(df, "CAL")
    check("[5] yuklenen model: mod modelden ('calibrated'), p_up_cal bellektekiyle ayni",
          loaded.confidence_mode == "calibrated"
          and pl["calibration"].get("p_up_cal") == c["p_up_cal"], str(pl["calibration"]))

    print("\n[6] Sizinti bekcisi: test bolumu bozulunca yon modeli bit-es")
    feat = leg.feature_engineer.build_features(df, "SYN")
    me = int(len(feat) * 0.80)
    cut = feat.index[me]  # ilk test satiri: meta'nin son h etiketi buna bakar (arindirilmali)
    bad = df.copy()
    r = np.random.RandomState(9)
    mask = bad.index >= cut
    f = 1 + r.randn(mask.sum()) * 0.05
    # ilk test kapanisi son meta kapanisina gore yonu KESIN ters cevirsin (etiket degissin)
    c0, c1 = df["close"].loc[feat.index[me - 1]], df["close"].loc[cut]
    f[0] = (c0 * (0.9 if c1 > c0 else 1.1)) / c1
    for col in ("open", "high", "low", "close"):
        bad.loc[mask, col] = bad.loc[mask, col].to_numpy() * f
    keys = ("features", "mean", "scale", "coef", "intercept", "temperature")
    cl2, _ = train("calibrated", data=bad, sym="CAL2")
    check("test kapanislari bozuldu -> yon modeli (katsayilar, T) bit-es",
          all(cl.direction_model[k] == cl2.direction_model[k] for k in keys))
    m0, _ = train("calibrated", sym="M0", purge=0)
    m1, _ = train("calibrated", data=bad, sym="M1", purge=0)
    check("mutasyon (arindirma 0): bozulma meta etiketine sizar -> yon modeli DEGISIR",
          any(m0.direction_model[k] != m1.direction_model[k] for k in keys))
finally:
    E.StackingEnsemble._get_model_class = orig_get
    ws.prediction_models_dir, ws.read_dirs = orig_pmd, orig_rd
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
