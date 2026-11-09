"""Faz 9 / K18 (+ K25) olcumu — RL gozlem sagligi, gercek panelde.

Egitim rotasinin veri yolunu (app/api/routes/trading.py `_run_training_inner`)
egitim YAPMADAN izler: `stock_data_with_indicators.csv` -> (K18) ADX yeniden
hesabi -> `prepare_rl_panel(drop_indicator_warmup=True)` -> `split_data` ->
val/test egitim evrenine hizali -> egitim ve test ortami. Her yapilandirmada
`evaluation.observation_health` (env'in kendi gozlem fonksiyonu, politikadan
bagimsiz) ve deneme kaydi protokolu (`rl_eval_protocol`) olculur.

Yapilandirmalar:
  pre_k18             CSV oldugu gibi (ADX legacy = 0), fiyat olcegi B9 (egitim penceresi)
  k18_b9              ADX 'wilder', fiyat olcegi B9           -> K18 sonrasi rota varsayilani
  k18_causal_252      ADX 'wilder', price_norm='causal', W=252 -> K25 (b) kolu
  k18_causal_63       ayni, W=63 (pencere duyarliligi)
  k18_causal_expanding ayni, genisleyen pencere (neden kayan secildi)

Ayrica: gercek panelde ADX dagilimi ve veri kapisinin ADX isinmasi yuzunden
fazladan attigi satirlar.

Bu bir OLCUMDUR, test degil; sayilar Faz 9 §9.9'a komutuyla yazilir.

Kullanim (depo kokunden):
    python scripts/analysis/measure_obs_health.py --out docs/development/measurements/phase9-k18/obs_health.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
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

CONFIGS = {
    "pre_k18": dict(adx=None, price_norm="stats"),
    "k18_b9": dict(adx="wilder", price_norm="stats"),
    "k18_causal_252": dict(adx="wilder", price_norm="causal", window=252),
    "k18_causal_63": dict(adx="wilder", price_norm="causal", window=63),
    "k18_causal_expanding": dict(adx="wilder", price_norm="causal", window=None),
}


def span(frame):
    d = frame.index.get_level_values("date")
    return [str(pd.Timestamp(d.min()).date()), str(pd.Timestamp(d.max()).date()), int(d.nunique())]


def windows(csv, adx_mode):
    from app.core.config import get_settings
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from data.technical_indicators import with_adx

    df = csv if adx_mode is None else with_adx(csv, mode=adx_mode)
    panel, report = prepare_rl_panel(df, start=get_settings().RL_DATA_START,
                                     drop_indicator_warmup=True)
    train_df, val_df, test_df = DataFetcher().split_data(panel)
    syms = set(train_df.index.get_level_values("symbol").unique())

    def align(f):
        return f[f.index.get_level_values("symbol").isin(syms)]

    return panel, report, train_df, align(val_df), align(test_df)


def summarize(health):
    """Uzun gozlem sagligi sozlugunden rapora girecek ozet."""
    out = {}
    for split in ("train", "eval"):
        out[split] = {b: {k: (None if v is None else round(v, 4)) for k, v in st.items()}
                      for b, st in health[split].items()}
        out[f"{split}_features"] = {
            f: {k: (None if v is None else round(v, 4)) for k, v in st.items()}
            for f, st in health[f"{split}_features"].items()}
    out["eval_out_of_train_support"] = {
        b: round(v, 4) for b, v in (health["eval_out_of_train_support"] or {}).items()}
    out["eval_out_of_train_support_features"] = {
        f: round(v, 4) for f, v in (health.get("eval_out_of_train_support_features") or {}).items()}
    return out


def adx_profile(panel):
    a = panel["adx"].astype(float)
    g = a.groupby(level="symbol")
    q = a.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(a.notna().sum()), "n_nan": int(a.isna().sum()),
        "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
        "quantiles": {str(k): round(float(v), 3) for k, v in q.items()},
        "frac_gt_25": round(float((a > 25).mean()), 4),
        "per_symbol_nunique_min": int(g.nunique().min()),
        "per_symbol_mean_range": [round(float(g.mean().min()), 2), round(float(g.mean().max()), 2)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    args = ap.parse_args()

    from data.data_fetcher import DataFetcher
    from env.trading_env import make_env
    from evaluation.observation_health import observation_health
    from evaluation.reproducibility import frame_sha256
    from evaluation.trial_registry import open_protocol_items, rl_eval_protocol

    csv = DataFetcher().load_data("stock_data_with_indicators.csv")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=ROOT, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
    except Exception:
        commit, dirty = None, None
    result = {"created_at": datetime.now().isoformat(timespec="seconds"),
              "command": "python scripts/analysis/measure_obs_health.py " + " ".join(sys.argv[1:]),
              "git_commit": commit, "git_dirty": dirty,
              "csv_sha256": frame_sha256(csv), "csv_span": span(csv),
              "env_kwargs": KW, "configs": {}}

    cache = {}
    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        if cfg["adx"] not in cache:
            cache[cfg["adx"]] = windows(csv, cfg["adx"])
        panel, report, train_df, val_df, test_df = cache[cfg["adx"]]
        norm = {}
        if cfg["price_norm"] == "causal":
            norm = dict(price_norm="causal", price_norm_window=cfg["window"])
        tr_env = make_env(df=train_df, price_stats=None, **norm, **KW)
        st = tr_env.get_price_stats()
        entry = {"cfg": cfg, "panel_sha256": frame_sha256(panel), "panel": span(panel),
                 "train": span(train_df), "val": span(val_df), "test": span(test_df),
                 "n_symbols": int(tr_env.n_stocks),
                 "gate_warmup_dropped": report.get("indicator_warmup_dropped"),
                 "gate_rows_after": report.get("rows_after"), "splits": {}}
        for split, frame, ctx in (("val", val_df, train_df),
                                  ("test", test_df, pd.concat([train_df, val_df]))):
            extra = dict(norm)
            if norm:
                extra["price_context"] = ctx
            ev_env = make_env(df=frame, price_stats=st, **extra, **KW)
            health = observation_health(tr_env, ev_env)
            protocol = rl_eval_protocol(ev_env, tr_env, health=health, hyperparams_clean=True)
            entry["splits"][split] = {
                "protocol": protocol, "open_items": open_protocol_items(protocol),
                "health": summarize(health),
                "price_norm_context_rows_min": (min(ev_env.price_norm_context_rows.values())
                                                if norm else None),
            }
        result["configs"][name] = entry
        t = entry["splits"]["test"]
        print(f"{name:22s} open={t['open_items']} "
              f"price_oos={t['health']['eval_out_of_train_support'].get('price')} "
              f"price_clip={t['health']['eval']['price']['clip_frac']}", flush=True)

    for mode, (panel, report, *_rest) in cache.items():
        result[f"adx_profile_{mode or 'legacy'}"] = adx_profile(panel)

    out = args.out or os.path.join("results", f"obs_health_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("yazildi:", out)


if __name__ == "__main__":
    main()
