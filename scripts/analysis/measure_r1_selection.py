"""Faz 9 / R1 hakem incelemesi olcumu — dogrulama secim metriginin isareti ve kat sinirinda gercek panel B0.

1) Secim metrigi (hakem-finans R1 #1-#2): ortamda nakit faiz kazanmiyor, secim metrigi ise
   risksiz faizi dusuyor (`rl_folds.excess_sharpe`). Ortalama fazla getiri negatifken Sharpe
   ~ -rf/sigma olur ve argmax en OYNAK noktayi secer. Bu betik gercek panelde rotanin A2C
   varsayilanlariyla (CPU, tek tohum) egitir, her kontrol noktasinin DOGRULAMA getirisinden
   ortalama brut / risksiz / fazla getiriyi, oynakligi ve dort kuralin secimini olcer:
     val_excess_sharpe (mevcut kural), sharpe_rf0, mean_excess, sign_aware_sharpe
     (ortalama < 0 iken mean * sigma, aksi halde mean / sigma; yalnizca tani).
   Deneme kaydina YAZILMAZ (N'yi kirletmez); test blogu KURULMAZ.
2) Gercek panel B0 (hakem-finans R1 soru 1): katin dogrulama/test sinirindan sonraki ham
   fiyatlar degistirilir, gosterge hatti (add_indicators_to_multi_symbol_df) yeniden kosulur;
   sinirdan onceki ortamin gozlem + odul + portfoy izi bit-es olmali (stats, causal).

Bu bir OLCUMDUR, test degil. Kullanim (depo kokunden):
    python scripts/analysis/measure_r1_selection.py --folds 2021,2025 --timesteps 50000 \
        --b0-fold 2026 --out docs/development/measurements/phase9-r1/selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU (Windows'ta bos dize GPU'yu acik birakir)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

KW = dict(initial_balance=1_000_000, commission_rate=0.001, max_shares_per_trade=100,
          phase=1, reward_type="simple")
# Rotanin A2C varsayilanlari (app/api/routes/trading.py, hiperparametre calismasi yokken)
A2C_DEFAULTS = dict(learning_rate=0.0007, n_steps=256, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    vf_coef=0.25, max_grad_norm=0.5, normalize_advantage=True, use_rms_prop=True,
                    rms_prop_eps=1e-5)
ANN = np.sqrt(252)


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


def params_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def criteria(r, rf):
    """Dort kural icin deger (None: tanimsiz). r, rf seans basi; ayni uzunlukta."""
    ok = np.isfinite(r) & np.isfinite(rf)
    r, rf = r[ok], rf[ok]
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return {"val_excess_sharpe": None, "sharpe_rf0": None, "mean_excess": None,
                "sign_aware_sharpe": None}
    ex = r - rf
    m, s = ex.mean(), ex.std(ddof=1)
    return {"val_excess_sharpe": float(m / s * ANN),
            "sharpe_rf0": float(r.mean() / r.std(ddof=1) * ANN),
            "mean_excess": float(m * 252),
            "sign_aware_sharpe": float((m / s if m >= 0 else m * s) * ANN)}


def pick(rows, key):
    valid = [x for x in rows if x[key] is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda x: (-x[key], x["step"]))[0]["index"]


def selection_diag(rp, spec, fold, timesteps, seed):
    from stable_baselines3 import A2C
    from evaluation import rl_folds

    fd = rl_folds.build_fold(rp, spec, fold)
    env = rl_folds.make_fold_env(fd, "train", **KW)
    stats_env = rl_folds.make_fold_env(fd, "train", **KW)
    val_dates = sorted(pd.unique(pd.to_datetime(fd.val.index.get_level_values("date"))))
    rf = rl_folds.load_risk_free(val_dates)
    if rf is None:
        raise SystemExit("risk_free.csv yok: secim metrigi tanisi rf olmadan anlamsiz")
    ckpt = tempfile.mkdtemp(prefix=f"r1_sel_{fold}_")
    make_val = lambda: rl_folds.make_fold_env(fd, "val", train_env=stats_env, **KW)  # noqa: E731
    cb = rl_folds.make_validation_callback(make_val, total_timesteps=timesteps,
                                           checkpoint_dir=ckpt, risk_free=rf)
    model = A2C("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **A2C_DEFAULTS)
    t0 = time.time()
    model.learn(timesteps, callback=cb)
    learn_s = time.time() - t0
    rows = []
    for rec in cb.records:
        m = A2C.load(rec["path"], device="cpu")
        ep = rl_folds.run_episode(m, make_val())
        r = np.asarray(ep["returns"], dtype=float)
        rfv = pd.Series(rf).reindex(pd.DatetimeIndex(ep["dates"])).to_numpy(dtype=float)
        c = criteria(r, rfv)
        rows.append({"index": rec["index"], "step": rec["step"], "params_sha": params_sha(m),
                     "n_trades": ep["n_trades"],
                     "mean_gross_ann": float(np.nanmean(r) * 252),
                     "mean_rf_ann": float(np.nanmean(rfv) * 252),
                     "vol_ann": float(np.nanstd(r, ddof=1) * ANN),
                     "callback_metric": rec["metric"], **c})
    shutil.rmtree(ckpt, ignore_errors=True)
    picks = {k: pick(rows, k) for k in ("val_excess_sharpe", "sharpe_rf0", "mean_excess",
                                         "sign_aware_sharpe")}
    vals = [x for x in rows if x["mean_excess"] is not None]
    by_vol = sorted(vals, key=lambda x: -x["vol_ann"])
    return {"fold": str(fold), "ranges": fd.report["ranges"], "timesteps": timesteps, "seed": seed,
            "learn_seconds": round(learn_s, 1), "rf_val_mean_ann": float(np.nanmean(
                pd.Series(rf).reindex(pd.DatetimeIndex(val_dates)).to_numpy(dtype=float)) * 252),
            "checkpoints": rows, "picks": picks,
            "n_negative_mean_excess": int(sum(1 for x in vals if x["mean_excess"] < 0)),
            "n_valid": len(vals),
            "n_distinct_weights": len({x["params_sha"] for x in rows}),
            "max_vol_index": by_vol[0]["index"] if by_vol else None,
            "callback_metric_matches": all(
                (x["callback_metric"] is None and x["val_excess_sharpe"] is None)
                or (x["callback_metric"] is not None and x["val_excess_sharpe"] is not None
                    and abs(x["callback_metric"] - x["val_excess_sharpe"]) < 1e-9) for x in rows)}


def env_digest(env, seed=42):
    base = getattr(env, "unwrapped", env)
    obs, _ = env.reset()
    n = base.frame_bound[1] - base.frame_bound[0] + 1
    acts = np.random.default_rng(seed).uniform(-1, 1, size=(n, base.n_stocks)).astype(np.float32)
    h = hashlib.sha256()
    h.update(obs.tobytes())
    for a in acts:
        obs, r, term, trunc, info = env.step(a)
        h.update(obs.tobytes())
        h.update(np.float64(r).tobytes())
        h.update(np.float64(info.get("portfolio_value", np.nan)).tobytes())
        if term or trunc:
            break
    return h.hexdigest()[:16]


def real_b0(raw, spec, fold):
    """Ham panelde test baslangicindan SONRAKI fiyatlari degistir, gosterge hattini yeniden kos."""
    from data.data_fetcher import DataFetcher
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    from evaluation import rl_folds

    f = spec.fold(fold)
    clean = DataFetcher().clean_data(raw.copy())
    d = pd.to_datetime(clean.index.get_level_values("date"))

    sym = clean.index.get_level_values("symbol")

    def perturbed_from(t, seed=13):
        # hakem-2 R1 #4: sembol basina FARKLI surukleme (seans basina -%1..+%1; kesit
        # siralamasi degisir) ve hacim x U(0,3; 3) — duzgun ortak surukleme yalniz fiyat
        # kanalini sinardi
        rng = np.random.default_rng(seed)
        pert = clean.copy()
        later = pd.DatetimeIndex(sorted(d[d >= t].unique()))
        rank = pd.Series(np.arange(1, len(later) + 1), index=later)
        mask = d >= t
        rates = {s: 1 + rng.uniform(-0.01, 0.01) for s in pd.unique(sym)}
        factor = np.array([rates[s] for s in sym[mask]]) ** rank.reindex(d[mask]).to_numpy()
        for c in ("open", "high", "low", "close"):
            pert.loc[mask, c] = pert.loc[mask, c].to_numpy() * factor
        pert.loc[mask, "volume"] = np.round(pert.loc[mask, "volume"].to_numpy()
                                            * rng.uniform(0.3, 3.0, int(mask.sum())))
        return add_indicators_to_multi_symbol_df(pert)

    tv, tt = pd.Timestamp(f.val_start), pd.Timestamp(f.test_start)
    panels = {"base": add_indicators_to_multi_symbol_df(clean),
              "from_val_start": perturbed_from(tv), "from_test_start": perturbed_from(tt)}
    out = {"fold": str(fold), "val_start": str(tv.date()), "test_start": str(tt.date()),
           "perturbation": "sembol basina 1+U(-0,01; 0,01) seans basina bilesik, hacim x U(0,3; 3)", "pipeline": "clean_data -> add_indicators_to_multi_symbol_df "
           "(CSV ureten hat) -> load_rl_panel -> build_fold -> make_fold_env"}
    for pn in ("stats", "causal"):
        dig = {}
        for k, panel in panels.items():
            rp = rl_folds.load_rl_panel(price_norm=pn, df=panel)
            fd = rl_folds.build_fold(rp, spec, fold)
            tr = rl_folds.make_fold_env(fd, "train", price_norm=pn, **KW)
            dig[k] = {role: env_digest(rl_folds.make_fold_env(
                fd, role, train_env=None if role == "train" else tr, price_norm=pn, **KW))
                for role in ("train", "val", "test")}
        b, v, t = dig["base"], dig["from_val_start"], dig["from_test_start"]
        out[pn] = {"digests": dig,
                   "val_boundary": {"train_bit_equal": v["train"] == b["train"],
                                    "val_changed": v["val"] != b["val"]},
                   "test_boundary": {"train_bit_equal": t["train"] == b["train"],
                                     "val_bit_equal": t["val"] == b["val"],
                                     "test_changed": t["test"] != b["test"]}}
        print(f"B0 {pn} {fold}: {out[pn]['val_boundary']} {out[pn]['test_boundary']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", default="2021,2025")
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--b0-fold", default="2026")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from data.data_fetcher import DataFetcher
    from evaluation import rl_folds
    from evaluation.reproducibility import frame_sha256
    from evaluation.splits import get_spec

    spec = get_spec()
    fetcher = DataFetcher()
    csv = fetcher.load_data("stock_data_with_indicators.csv")
    commit, dirty = git_state()
    result = {"created_at": datetime.now().isoformat(timespec="seconds"),
              "command": "python scripts/analysis/measure_r1_selection.py " + " ".join(sys.argv[1:]),
              "git_commit": commit, "git_dirty": dirty,
              "indicator_csv_frame_sha256": frame_sha256(csv),
              "split_spec": {"name": spec.name, "digest": spec.digest()},
              "env_kwargs": KW, "a2c": A2C_DEFAULTS, "device": "cpu"}
    rp = rl_folds.load_rl_panel(price_norm="stats", df=csv)
    result["selection"] = {}
    for fold in [x for x in args.folds.split(",") if x]:
        res = selection_diag(rp, spec, fold, args.timesteps, args.seed)
        result["selection"][fold] = res
        print(f"secim {fold}: rf {res['rf_val_mean_ann']:.3f}, negatif ort. fazla getiri "
              f"{res['n_negative_mean_excess']}/{res['n_valid']}, farkli agirlik "
              f"{res['n_distinct_weights']}, secimler {res['picks']}, en oynak {res['max_vol_index']}, "
              f"{res['learn_seconds']} sn", flush=True)
    if args.b0_fold:
        raw = fetcher.load_data("raw_stock_data.csv")
        result["real_b0"] = real_b0(raw, spec, args.b0_fold)
    out = args.out or os.path.join("results", f"r1_selection_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("yazildi:", out)


if __name__ == "__main__":
    main()
