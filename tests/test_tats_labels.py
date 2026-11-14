"""Faz 9 B10 (Y20) — TATS uyum girdileri: meta satirlarinin HAM kapanisi + gerceklesen getiri.

    python tests/test_tats_labels.py

Kabul (Faz 10 §3.1.1): bilinen trendli sentetik veride etiket isabeti ~ 1; uyum girdileri
meta bolumu tarihleriyle birebir hizali. Kapsam (hakem-2 B10): gunluk + haftalik ufuk,
ICEEMDAN kolu (feat_df['close'] filtreli seriyle uzerine yazilir — taklit), kisa OOF
donduren taban model (DL gibi; sondan kirpma sinanir). Eski girdiler (y_meta, y_base)
etiketi bozar; mutasyonlar uretim yolundan: kapanis bir gun kayik, kirpma bastan.
K21: test bolumunde TATS'li/TATS'siz metrik; gecici dizine kaydet/yukle, yuklenen modelde TATS.
Gercek StackingEnsemble.train (XGBoost + LightGBM + kisa-OOF sahte model), CPU.
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
    orig_save, orig_meta = BasePredictionModel.save, E.StackingEnsemble._save_ensemble_meta
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
        # K21: uyum meta bolumunun son h satirini atar (etiketi test donemine bakar)
        check(f"[{tag}] kisa OOF sondan kirpildi, son h={h} satir atildi (k = meta - {SHORT} - h)",
              k == len(meta_dates) - SHORT - h, f"(k={k}, meta={len(meta_dates)})")
        dates = meta_dates[-(k + h):-h]
        cur = df["close"].reindex(dates).to_numpy(float)
        true_lab = labels(df["close"].shift(-h).reindex(dates).to_numpy(float), cur)
        check(f"[{tag}] mevcut fiyat = meta tarihlerindeki HAM kapanis (birebir)",
              np.array_equal(cap["cur"], cur), f"(k={k})")
        acc = float(np.mean(true_lab == labels(cap["y"], cap["cur"])))
        check(f"[{tag}] etiket isabeti (gerceklesen getiri) ~ 1 (olculen {acc:.4f}, "
              f"{len(np.unique(true_lab))} sinif)", acc > 0.999 and len(np.unique(true_lab)) == 3)
        if not ice and horizon == "daily":
            old_acc = float(np.mean(true_lab == labels(y[be:me][-(k + h):-h], y[:be][-k:])))
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
    BasePredictionModel.save, E.StackingEnsemble._save_ensemble_meta = orig_save, orig_meta

    # ---- K21 (Faz 10): test bolumunde TATS'li/TATS'siz + diske yaz/yukle ---------------
    import shutil
    import tempfile
    from app.auth import workspace as ws
    tmp = tempfile.mkdtemp(prefix="rlt_k21_")
    orig_pmd, orig_rd = ws.prediction_models_dir, ws.read_dirs
    ws.prediction_models_dir = lambda *a, **k: tmp           # gercek models/ dizinine dokunma
    ws.read_dirs = lambda kind, *a, **k: [tmp]
    try:
        print("\nK21 — TATS test degerlendirmesi ve kalicilik (daily/log_return)")
        cfg2 = {m: dict(c, enabled=(m in ("xgboost", "lightgbm"))) for m, c in E.BASE_MODEL_CONFIGS.items()}
        ens = E.StackingEnsemble(horizon="daily", use_tats=True, model_configs=cfg2)
        res = ens.train(df, "SYN")
        tm = res.get("ensemble_test_metrics", {})
        t = tm.get("tats", {})
        check("test metriginde TATS'li ve TATS'siz birlikte raporlandi",
              {"with_tats", "without_tats", "n_applied", "n_rows"} <= set(t)
              and "tda" in t.get("with_tats", {}), str(list(t)))
        check("TATS'siz kol = ana test metrigi (ayni satirlar, ayni tahmin)",
              bool(t) and abs(t["without_tats"]["mape"] - tm["mape"]) < 1e-9
              and t["n_rows"] > 0, f"({t.get('without_tats')}, mape={tm.get('mape')})")
        print(f"         n_applied={t.get('n_applied')}/{t.get('n_rows')}, "
              f"dir_acc {t.get('without_tats', {}).get('direction_accuracy')} -> "
              f"{t.get('with_tats', {}).get('direction_accuracy')}")
        # Duzeltme gercekten metrige giriyor mu: correct() zorla uygulanir (yalniz bu egitimde)
        orig_correct = TATSCorrector.correct
        TATSCorrector.correct = lambda self, pc, cur, X: (pc * 1.01, {"applied": True})
        try:
            tf = E.StackingEnsemble(horizon="daily", use_tats=True, model_configs=cfg2)
            tf_t = tf.train(df, "SYNF").get("ensemble_test_metrics", {}).get("tats", {})
        finally:
            TATSCorrector.correct = orig_correct
        check("zorla uygulanan duzeltme TATS'li metrige giriyor (n_applied = n_rows, metrik farkli)",
              bool(tf_t) and tf_t["n_applied"] == tf_t["n_rows"]
              and tf_t["with_tats"]["mae"] != tf_t["without_tats"]["mae"], str(tf_t)[:200])
        p_mem = ens.predict_next(df, "SYN")
        loaded = E.StackingEnsemble(horizon="daily", model_configs=cfg2)
        loaded.load("SYN")
        check("yuklenen modelde TATS var", loaded.tats_corrector is not None
              and loaded.tats_corrector.is_fitted())
        p_load = loaded.predict_next(df, "SYN")
        same = {k: p_mem["tats"].get(k) for k in ("applied", "trend", "trend_confidence")} ==                {k: p_load["tats"].get(k) for k in ("applied", "trend", "trend_confidence")}
        check("yuklenen modelin TATS ciktisi bellektekiyle ayni", same,
              f"({p_mem['tats']} vs {p_load['tats']})")
        # Mutasyon: TATS dosyasi yoksa (eski model) yuklenen modelde TATS yok
        os.remove(os.path.join(tmp, f"{ens._model_id('SYN')}_daily_tats.json"))
        old_model = E.StackingEnsemble(horizon="daily", model_configs=cfg2)
        old_model.load("SYN")
        check("mutasyon: TATS dosyasi silinince yuklenen modelde TATS yok (eski davranis)",
              old_model.tats_corrector is None)
        # Y25: use_tats=False kurulan ablasyon kolu TATS'li modeli yuklese de TATS ALMAZ
        abl = E.StackingEnsemble(horizon="daily", use_tats=False, model_configs=cfg2)
        abl.load("SYNF")
        check("Y25: kurucudaki use_tats=False yuklemede ezilmez", abl.use_tats is False)
        check("Y25: predict_next 'tats.available' (TATS yok != uygulanmadi)",
              p_load["tats"].get("available") is True
              and old_model.predict_next(df, "SYN")["tats"].get("available") is False)

        # ---- Y28: canli predict_next en SON satiri kullanir (gunluk + haftalik) ----------
        from prediction.feature_engineer import PredictionFeatureEngineer as FE
        T = df.index[-1]
        ret1_T = float(df["close"].iloc[-2] / df["close"].iloc[-3] - 1)   # return_1d[T] (shift 1)
        ret1_prev = float(df["close"].iloc[-3] / df["close"].iloc[-4] - 1)

        def probe(e):
            """predict_next'in gordugu ozellik cercevesinin son tarihi + taban modele giden son satir."""
            seen = {}
            fe = e.feature_engineer
            for name in ("build_features", "build_inference_features"):
                f = getattr(fe, name)

                def wrap(*a, _f=f, **k):
                    out = _f(*a, **k)
                    seen["last"] = out.index[-1]
                    return out
                setattr(fe, name, wrap)
            m = next(iter(e.base_models.values()))
            raw = m._predict_raw
            m._predict_raw = lambda X: (seen.__setitem__("X", X), raw(X))[1]
            try:
                e.predict_next(df, "SYN")
            finally:
                for name in ("build_features", "build_inference_features"):
                    fe.__dict__.pop(name, None)
                m.__dict__.pop("_predict_raw", None)
            j = e.feature_cols.index("return_1d")
            return seen.get("last"), float(seen["X"][-1, j])

        for hz, h in (("daily", 1), ("weekly", 5)):
            print(f"\nY28 — canli tahmin son satir ({hz})")
            fe = FE(hz)
            tr, inf = fe.build_features(df, "SYN"), fe.build_inference_features(df, "SYN")
            cols = fe.get_feature_columns(tr)
            check(f"[{hz}] cikarim matrisi son tarih = df.index[-1]; egitim matrisinden tam h={h} satir fazla",
                  inf.index[-1] == T and len(inf) - len(tr) == h and tr.index[-1] == df.index[-1 - h],
                  f"(inf {inf.index[-1]}, fark {len(inf) - len(tr)})")
            check(f"[{hz}] egitim satirlarinda iki matris birebir (egitim yolu degismedi); fazla satirlarin hedefi NaN",
                  inf.loc[tr.index, cols].equals(tr[cols])
                  and inf["target_price"].iloc[-h:].isna().all())
            e = E.StackingEnsemble(horizon=hz, use_tats=False, model_configs=cfg2)
            e.train(df, "Y28" + hz)
            last, r1 = probe(e)
            check(f"[{hz}] predict_next son ozellik satiri tarihi = df.index[-1]", last == T, f"({last})")
            check(f"[{hz}] taban modele giden son satir return_1d = close[T-1]/close[T-2]-1 (dfden bagimsiz)",
                  abs(r1 - ret1_T) < 1e-5 and abs(ret1_T - ret1_prev) > 1e-4, f"({r1} vs {ret1_T})")
            # Mutasyon (uretim yolu): cikarim yolu eski dropna'li build_features'a doner
            orig_inf = FE.build_inference_features
            FE.build_inference_features = lambda self, *a, **k: self.build_features(*a, **k)
            try:
                m_last, m_r1 = probe(e)
            finally:
                FE.build_inference_features = orig_inf
            check(f"[{hz}] mutasyon: dropna geri gelince son satir df.index[-{h + 1}] ve return_1d bayat",
                  m_last == df.index[-1 - h] and abs(m_r1 - ret1_T) > 1e-5, f"({m_last}, {m_r1})")
    finally:
        ws.prediction_models_dir, ws.read_dirs = orig_pmd, orig_rd
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- Y25: gercek correct() kurali DEGERLE (siniflandirici sabit DOWN, guven 0,8) ---------
    print("\nY25 — TATS degerlendirme yolu degerle")

    class _Down:
        def predict_proba(self, X):
            return np.array([[0.8, 0.1, 0.1]])

    idx = pd.bdate_range("2024-01-01", periods=12)
    raw_close = pd.Series(np.linspace(100, 111, 12), index=idx)
    for tt in ("log_return", "abs_price"):
        e2 = E.StackingEnsemble(horizon="daily", model_configs=cfg2, target_type=tt)
        tc = TATSCorrector()
        tc.classifier, tc._is_fitted, tc._semantic_classes = _Down(), True, np.array([0, 1, 2])
        e2.tats_corrector = tc
        cur = raw_close.to_numpy(float)
        r = np.array([0.02, -0.01, 0.03, 0.001, -0.02, 0.015, 0.0, 0.04, -0.03, 0.01, 0.002, 0.025])
        base = r if tt == "log_return" else cur * np.exp(r)
        y_t = r[::-1].copy() if tt == "log_return" else cur * np.exp(r[::-1])
        pc = cur * np.exp(base) if tt == "log_return" else base
        conflict = (pc - cur) / cur > tc.flat_threshold          # tahmin UP, trend DOWN
        new_pc = np.where(conflict, pc - np.abs(pc - cur) * 0.8 * 0.5, pc)
        exp_corr = np.log(new_pc / cur) if tt == "log_return" else new_pc
        df_t = pd.DataFrame({"close": raw_close})
        out = e2._evaluate_tats_on_test(df_t, pd.DataFrame(index=idx), 0, np.zeros((12, 2)), base, y_t)
        exp_m = BasePredictionModel.compute_metrics(y_t, exp_corr)
        check(f"[{tt}] n_applied = catisan satir sayisi ({int(conflict.sum())})",
              out["n_applied"] == int(conflict.sum()) > 0, str(out["n_applied"]))
        check(f"[{tt}] TATS'li MAE = elle hesaplanan duzeltmenin MAE'si (geri ceviris dogru)",
              abs(out["with_tats"]["mae"] - exp_m["mae"]) < 1e-9, f"({out['with_tats']['mae']} vs {exp_m['mae']})")
        dsign = np.sign(new_pc - cur) == np.sign(pc - cur)
        check(f"[{tt}] isaret korunuyor ve TDA TATS'li = TATS'siz",
              bool(dsign.all()) and out["with_tats"]["tda"] == out["without_tats"]["tda"])
        if tt == "abs_price":
            # Mutasyon: cari fiyat bir gun kayik -> duzeltme degisir (log_return'de cari sadelesir)
            df_s = pd.DataFrame({"close": raw_close.shift(1).bfill()})
            out_s = e2._evaluate_tats_on_test(df_s, pd.DataFrame(index=idx), 0, np.zeros((12, 2)), base, y_t)
            check("[abs_price] mutasyon: cari fiyat kayik -> TATS'li MAE elle hesaplanandan sapar",
                  abs(out_s["with_tats"]["mae"] - exp_m["mae"]) > 1e-6)

    print("\n" + "=" * 60)
    print(f"Sonuc: {len(PASSED)} gecti, {len(FAILED)} kaldi")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
