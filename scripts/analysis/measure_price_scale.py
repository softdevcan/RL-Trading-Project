"""Faz 9 / B9 olcumu — RL val/test ortaminin fiyat olcegi (Faz 10 Y13).

Iki kip:

  saturation (hizli, egitim yok)
      RL penceresinde (prepare_rl_panel, RL_DATA_START) split_data bolmeleri;
      val ve test pencerelerinde O/H/L/C fiyat z'lerinin kirpilan (|z| >= 10)
      orani — egitim penceresi olcegiyle (B9) ve pencerenin kendi olcegiyle
      (eski davranis).

  effect (egitimli)
      Egitim rotasinin varsayilanlariyla (app/api/routes/trading.py
      `_run_training_inner`: A2C/PPO hiperparametreleri, 1M bakiye, %0,1
      komisyon, 100 lot, own_last) tohum basina bir model egitir ve AYNI
      modeli test penceresinde iki olcekle degerlendirir. Fark YALNIZCA
      olcekten gelir. Tohum basina eslestirilmis farklar ve tek orneklem
      t-testi raporlanir (kucuk n; yon iddiasi icin degil, buyukluk icin).

Ciktilar JSON (varsayilan: results_dir()/price_scale_<kip>_<tarih>.json).
Bu bir OLCUMDUR, test degil; sayilari Faz 9 §9.5'e komutuyla yazilir.

Kullanim (depo kokunden):
    python scripts/analysis/measure_price_scale.py --mode saturation
    python scripts/analysis/measure_price_scale.py --mode effect --algos A2C,PPO --seeds 1,2,3 --steps 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

KW = dict(initial_balance=1_000_000, commission_rate=0.001, max_shares_per_trade=100,
          phase=1, reward_type="simple", missing_data_mode="own_last")


def load_windows():
    from app.core.config import get_settings
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel

    fetcher = DataFetcher()
    df = fetcher.load_data("stock_data_with_indicators.csv")
    df, _ = prepare_rl_panel(df, start=get_settings().RL_DATA_START)
    train_df, val_df, test_df = fetcher.split_data(df)
    # egitim rotasi gibi: val/test egitim bolumunun sembol evrenine hizali
    syms = set(train_df.index.get_level_values("symbol").unique())

    def align(f):
        return f[f.index.get_level_values("symbol").isin(syms)]

    return df, train_df, align(val_df), align(test_df)


def span(frame):
    d = frame.index.get_level_values("date")
    return [str(pd.Timestamp(d.min()).date()), str(pd.Timestamp(d.max()).date()), int(d.nunique())]


def symbol_stats(frame):
    """TradingEnv._compute_price_stats ile ayni tanim."""
    out = {}
    for s, sub in frame.groupby(level="symbol"):
        c = sub["close"].astype(float)
        sd = float(c.std())
        out[s] = (float(c.mean()), sd if sd > 1e-8 else 1.0)
    return out


def saturation(train_df, frame):
    st_train, st_own = symbol_stats(train_df), symbol_stats(frame)
    z_tr, z_own, per = [], [], {}
    for s, sub in frame.groupby(level="symbol"):
        px = sub[["open", "high", "low", "close"]].to_numpy(dtype=float).ravel()
        z_tr.append((px - st_train[s][0]) / st_train[s][1])
        z_own.append((px - st_own[s][0]) / st_own[s][1])
        c = sub["close"].to_numpy(dtype=float)
        per[s] = float(np.mean(np.abs((c - st_train[s][0]) / st_train[s][1]) >= 10))
    zt, zo = np.concatenate(z_tr), np.concatenate(z_own)
    ser = pd.Series(per)

    def summ(z):
        return {"clip_frac": float(np.mean(np.abs(z) >= 10)), "median_z": float(np.median(z)),
                "p05": float(np.percentile(z, 5)), "p95": float(np.percentile(z, 95))}

    return {"n_values": int(len(zt)), "train_window_scale": summ(zt), "own_window_scale": summ(zo),
            "close_clip_by_symbol": {"min": float(ser.min()), "median": float(ser.median()),
                                     "max": float(ser.max()),
                                     "n_fully_clipped": int((ser >= 0.999).sum()),
                                     "n_symbols": int(len(ser))}}


def model_for(algo, env, seed):
    """Rotanin varsayilan hiperparametreleri (trading.py, PPO/A2C dallari)."""
    from stable_baselines3 import A2C, PPO
    if algo == "PPO":
        return PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                   gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.15, vf_coef=0.5,
                   max_grad_norm=0.5, seed=seed, verbose=0, device="cpu")
    if algo == "A2C":
        return A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=256, gamma=0.99, gae_lambda=0.95,
                   ent_coef=0.01, vf_coef=0.25, max_grad_norm=0.5, normalize_advantage=True,
                   use_rms_prop=True, rms_prop_eps=1e-5, seed=seed, verbose=0, device="cpu")
    raise ValueError(f"desteklenmeyen algoritma: {algo}")


def evaluate(model, env):
    obs, _ = env.reset()
    n = env.n_stocks
    clipped = total = p_clip = p_total = 0
    done = False
    while True:
        clipped += int(np.sum(np.abs(obs) >= 10.0))
        total += obs.size
        blk = obs[1 + n:1 + n + n * env.features_per_stock].reshape(n, env.features_per_stock)[:, :4]
        p_clip += int(np.sum(np.abs(blk) >= 10.0))
        p_total += blk.size
        if done:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    m = env.get_metrics()
    return {"cumulative_return": float(m["cumulative_return"]), "sharpe_ratio": float(m["sharpe_ratio"]),
            "max_drawdown": float(m["max_drawdown"]), "total_trades": int(m["total_trades"]),
            "obs_clip_frac": clipped / total, "price_clip_frac": p_clip / p_total}


def effect(train_df, test_df, algos, seeds, steps, log):
    from scipy import stats
    from stable_baselines3.common.vec_env import DummyVecEnv

    from env.trading_env import make_env

    rows = []
    for algo in algos:
        for seed in seeds:
            t0 = time.time()
            venv = DummyVecEnv([lambda: make_env(df=train_df, **KW)])
            model = model_for(algo, venv, seed)
            model.learn(total_timesteps=steps)
            train_stats = venv.envs[0].get_price_stats()
            own = evaluate(model, make_env(df=test_df, **KW))  # B9-legacy: olcum, eski davranis
            b9 = evaluate(model, make_env(df=test_df, price_stats=train_stats, **KW))
            row = {"algo": algo, "seed": seed, "steps": steps, "own_window": own, "train_window": b9,
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            log(json.dumps(row))
    summary = {}
    keys = ("cumulative_return", "sharpe_ratio", "max_drawdown", "total_trades",
            "price_clip_frac", "obs_clip_frac")
    for algo in list(algos) + ["pooled"]:
        sel = rows if algo == "pooled" else [r for r in rows if r["algo"] == algo]
        if not sel:
            continue
        out = {"n": len(sel)}
        for k in keys:
            o = np.array([r["own_window"][k] for r in sel], dtype=float)
            t = np.array([r["train_window"][k] for r in sel], dtype=float)
            dlt = t - o
            ent = {"own_mean": float(o.mean()), "train_mean": float(t.mean()),
                   "diff_mean": float(dlt.mean()), "diffs": [float(x) for x in dlt]}
            if len(sel) > 1:
                ent.update(own_sd=float(o.std(ddof=1)), train_sd=float(t.std(ddof=1)),
                           diff_sd=float(dlt.std(ddof=1)))
                if dlt.std(ddof=1) > 0:
                    ent["paired_t_p"] = float(stats.ttest_1samp(dlt, 0.0).pvalue)
            out[k] = ent
        summary[algo] = out
    close = test_df["close"].unstack("symbol").sort_index()
    ew = float((close.iloc[-1] / close.iloc[0]).mean() - 1.0)
    return {"rows": rows, "summary": summary,
            "context_equal_weight_buy_hold_return_no_cost": ew}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("saturation", "effect"), default="saturation")
    ap.add_argument("--algos", default="A2C,PPO")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel, train_df, val_df, test_df = load_windows()
    result = {"created_at": datetime.now().isoformat(timespec="seconds"), "mode": args.mode,
              "panel": span(panel), "train": span(train_df), "val": span(val_df), "test": span(test_df),
              "n_symbols": int(train_df.index.get_level_values("symbol").nunique())}
    print(json.dumps({k: result[k] for k in ("panel", "train", "val", "test", "n_symbols")}))

    if args.mode == "saturation":
        result["val_saturation"] = saturation(train_df, val_df)
        result["test_saturation"] = saturation(train_df, test_df)
        for w in ("val", "test"):
            print(w, json.dumps(result[f"{w}_saturation"]))
    else:
        result["effect"] = effect(train_df, test_df, args.algos.split(","),
                                  [int(s) for s in args.seeds.split(",")], args.steps,
                                  lambda line: print(line, flush=True))
        print(json.dumps(result["effect"]["summary"], indent=1))
        print("baglam (komisyonsuz esit agirlikli al-tut):",
              result["effect"]["context_equal_weight_buy_hold_return_no_cost"])

    out = args.out
    if out is None:
        try:
            from app.auth import workspace as ws
            rdir = ws.results_dir()
        except Exception:
            rdir = "results"
        out = os.path.join(rdir, f"price_scale_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("yazildi:", out)


if __name__ == "__main__":
    main()
