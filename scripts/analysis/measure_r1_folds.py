"""Faz 9 / R1 olcumu — wf-v1 katlari gercek panelde, EGITIMSIZ.

Rotanin ve HPO'nun ORTAK veri yolu (`evaluation.rl_folds`): gostergeli CSV -> ADX
'wilder' (K18) -> veri kapisi (K17, B7; nedensel kolda pencere oncesi baglam, K25)
-> kat -> rol basina ortam (dogrulama/test egitim olcegini alir, B9). Her kat x kol
('stats', 'causal') icin olculenler:
  * seans sayilari, araliklar, evren (dusen sembol)
  * dogrulama ve test ortaminin gozlem sagligi (`evaluation.observation_health`,
    politikadan bagimsiz piyasa bloklari) ve deneme kaydi protokolu
    (`rl_eval_protocol`). Model egitilmedigi icin `selection_metric` olculemez;
    `open_items_env` onu disarida birakir (yalniz ortamdan olculen maddeler).

Ayrica:
  gate           tam panel ve RL penceresi kapi sayilari (hayalet gun, bayat hucre,
                 eksik hucre, artefakt, buyuk hareket)
  panel_refresh  (--old-panel) eski indirmeyle karsilastirma: sembol basina kapanis
                 orani ve basamak tarihleri, gunluk getiri farki, yalniz eskide olan
                 satirlar, hayalet gun farki. --dividends: yfinance temettu tarihleri
                 (ag gerekir; cekim tarihi kayda girer)

Bu bir OLCUMDUR, test degil; sayilar Faz 9 §9.11'e komutuyla yazilir.

Kullanim (depo kokunden):
    python scripts/analysis/measure_r1_folds.py \
        --old-panel data/bist/backup_2026-07-27/raw_stock_data.csv --dividends \
        --out docs/development/measurements/phase9-r1/folds.json
"""

from __future__ import annotations

import argparse
import hashlib
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

# Egitim istegi (TrainingRequest) varsayilanlari; faz 1, basit odul
KW = dict(initial_balance=1_000_000, commission_rate=0.001, max_shares_per_trade=100,
          phase=1, reward_type="simple")
NORMS = ("stats", "causal")
NOT_MEASURABLE_WITHOUT_TRAINING = ("selection_metric",)


def r4(x):
    return None if x is None else round(float(x), 4)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def health_summary(health):
    from evaluation.observation_health import (GATED_BLOCKS, dims_failures,
                                               obs_health_failures, support_failures)
    out = {"n_train_days": health["n_train_days"], "n_eval_days": health["n_eval_days"]}
    for split in ("train", "eval"):
        out[split] = {b: {k: r4(v) for k, v in st.items()}
                      for b, st in (health.get(split) or {}).items() if b in GATED_BLOCKS}
        out[f"{split}_saturated"] = {f: r4(st["sat_frac"]) for f, st in
                                     (health.get(f"{split}_features") or {}).items()
                                     if st.get("sat_frac") and st["sat_frac"] > 0.5}
    out["eval_out_of_train_support"] = {
        b: r4(v) for b, v in (health.get("eval_out_of_train_support") or {}).items()}
    out["eval_out_of_train_support_features"] = {
        f: r4(v) for f, v in (health.get("eval_out_of_train_support_features") or {}).items()}
    out["dims_over_eps"] = {
        f: {"n_symbols": len(v), "worst": [max(v, key=v.get), r4(max(v.values()))]}
        for f, v in dims_failures(health).items()}
    out["obs_health_failures"] = obs_health_failures(health)
    out["support_failures"] = support_failures(health)
    out["price_norm_warmup"] = health.get("price_norm_warmup")
    return out


def measure_folds(csv, spec):
    from evaluation import rl_folds
    from evaluation.observation_health import observation_health
    from evaluation.reproducibility import frame_sha256
    from evaluation.splits import is_canonical
    from evaluation.trial_registry import open_protocol_items, rl_eval_protocol

    result, gate = {}, None
    for pn in NORMS:
        rp = rl_folds.load_rl_panel(price_norm=pn, df=csv)
        if gate is None:
            gate = {"window": rp.integrity_report, "panel_sha256": frame_sha256(rp.panel),
                    "indicator_spec": rp.indicator_spec}
        result[pn] = {"train_context": rp.context_report, "folds": {}}
        for f in spec.folds:
            fd = rl_folds.build_fold(rp, spec, f)
            tr = rl_folds.make_fold_env(fd, "train", price_norm=pn, **KW)
            entry = {"report": fd.report, "n_symbols": int(tr.n_stocks),
                     "price_stats_fallbacks": dict(getattr(tr, "price_stats_fallbacks", {}) or {}),
                     "roles": {}}
            for role in ("val", "test"):
                ev = rl_folds.make_fold_env(fd, role, train_env=tr, price_norm=pn, **KW)
                health = observation_health(tr, ev)
                prot = rl_eval_protocol(ev, tr, health=health, hyperparams_clean=True,
                                        split_from_spec=is_canonical(spec), selection_metric=None)
                items = open_protocol_items(prot)
                entry["roles"][role] = {
                    "protocol": prot, "open_items": items,
                    "open_items_env": [i for i in items if i not in NOT_MEASURABLE_WITHOUT_TRAINING],
                    "health": health_summary(health)}
                ev.close()
            tr.close()
            result[pn]["folds"][f.name] = entry
            t = entry["roles"]["test"]
            print(f"{pn:6s} {f.name}: seans {fd.report['n_sessions']} sembol {entry['n_symbols']} "
                  f"test acik={t['open_items_env']} fiyat_dd="
                  f"{t['health']['eval_out_of_train_support'].get('price')}", flush=True)
    return result, gate


def panel_refresh(old_path, new_path, start, dividends=False):
    from data.integrity import find_ghost_days
    W = pd.Timestamp(start)

    def load(p):
        d = pd.read_csv(p, parse_dates=["date"])
        if d["date"].dt.tz is not None:
            d["date"] = d["date"].dt.tz_localize(None)
        return d

    old, new = load(old_path), load(new_path)
    old_end = old["date"].max()
    out = {"old": {"path": os.path.relpath(old_path, ROOT).replace("\\", "/"),
                   "file_sha256": file_sha256(old_path), "rows": int(len(old)),
                   "span": [str(old["date"].min().date()), str(old_end.date())]},
           "new": {"path": os.path.relpath(new_path, ROOT).replace("\\", "/"),
                   "file_sha256": file_sha256(new_path), "rows": int(len(new)),
                   "span": [str(new["date"].min().date()), str(new["date"].max().date())]},
           "window": [str(W.date()), str(old_end.date())]}
    out["new"]["nonpositive_price_rows"] = int(
        (new[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    out["old"]["nonpositive_price_rows"] = int(
        (old[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())

    m = old.merge(new, on=["symbol", "date"], suffixes=("_o", "_n"))
    m = m[(m["date"] >= W) & (m["close_o"] > 0) & (m["close_n"] > 0)].sort_values(["symbol", "date"])
    out["common_rows_in_window"] = int(len(m))
    per = {}
    for s, g in m.groupby("symbol"):
        r = (g["close_n"] / g["close_o"]).to_numpy()
        lr_o, lr_n = np.diff(np.log(g["close_o"].to_numpy())), np.diff(np.log(g["close_n"].to_numpy()))
        dr = np.abs(lr_n - lr_o)
        steps = np.flatnonzero(np.abs(np.diff(r)) > 1e-6) + 1
        per[s] = {"ratio_min": round(float(r.min()), 6), "ratio_max": round(float(r.max()), 6),
                  "spread_pct": round(float((r.max() / r.min() - 1) * 100), 5),
                  "step_dates": [str(pd.Timestamp(g["date"].iloc[i]).date()) for i in steps],
                  "max_abs_logret_diff": float(dr.max()) if dr.size else None,
                  "n_logret_diff_gt_1e-5": int((dr > 1e-5).sum())}
    out["per_symbol"] = per
    out["symbols_ratio_not_one"] = sorted(s for s, v in per.items()
                                          if abs(v["ratio_min"] - 1) > 1e-4 or abs(v["ratio_max"] - 1) > 1e-4)
    out["symbols_ratio_not_constant"] = sorted(s for s, v in per.items() if v["step_dates"])

    ko = old[(old["date"] >= W)].set_index(["symbol", "date"])
    kn = new.set_index(["symbol", "date"])
    only_old = ko.loc[~ko.index.isin(kn.index)].reset_index()
    flat = ((only_old["volume"] == 0) & (only_old["open"] == only_old["high"])
            & (only_old["high"] == only_old["low"]) & (only_old["low"] == only_old["close"]))
    out["rows_only_old_in_window"] = {"n": int(len(only_old)), "n_flat": int(flat.sum()),
                                      "by_date": {str(k.date()): int(v) for k, v in
                                                  only_old.groupby("date").size().items()}}
    kn_win = new[(new["date"] >= W) & (new["date"] <= old_end)].set_index(["symbol", "date"])
    out["rows_only_new_in_window"] = int((~kn_win.index.isin(ko.index)).sum())

    def ghosts(d):
        x = d[(d["date"] >= W) & (d["date"] <= old_end)].set_index(["symbol", "date"]).sort_index()
        return set(find_ghost_days(x))

    go, gn = ghosts(old), ghosts(new)
    out["ghost_days"] = {"old": len(go), "new_same_span": len(gn),
                         "only_old": sorted(str(d.date()) for d in go - gn),
                         "only_new": sorted(str(d.date()) for d in gn - go)}
    if dividends:
        import yfinance as yf
        syms = out["symbols_ratio_not_one"]
        divs = {}
        for s in syms:
            d = yf.Ticker(s).dividends
            if d.index.tz is not None:
                d.index = d.index.tz_localize(None)
            divs[s] = [[str(i.date()), round(float(v), 6)] for i, v in d[d.index >= W].items()]
        out["dividends"] = {"source": "yfinance Ticker.dividends",
                            "fetched_at": datetime.now().isoformat(timespec="seconds"),
                            "by_symbol": divs}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--old-panel", default=None, help="eski ham CSV (panel tazeleme karsilastirmasi)")
    ap.add_argument("--dividends", action="store_true", help="yfinance temettu tarihleri (ag)")
    args = ap.parse_args()

    from app.core.config import get_settings
    from data.data_fetcher import DataFetcher
    from data.integrity import health_report
    from evaluation.reproducibility import frame_sha256
    from evaluation.splits import fold_coverage, get_spec

    settings = get_settings()
    data_dir = os.path.join(ROOT, settings.DATA_DIR) if not os.path.isabs(settings.DATA_DIR) \
        else settings.DATA_DIR
    ind_path = os.path.join(data_dir, "stock_data_with_indicators.csv")
    raw_path = os.path.join(data_dir, "raw_stock_data.csv")
    csv = DataFetcher().load_data("stock_data_with_indicators.csv")
    spec = get_spec()
    commit, dirty = git_state()
    d = csv.index.get_level_values("date")
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": "python scripts/analysis/measure_r1_folds.py " + " ".join(sys.argv[1:]),
        "git_commit": commit, "git_dirty": dirty,
        "indicator_csv": {"file_sha256": file_sha256(ind_path), "frame_sha256": frame_sha256(csv),
                          "span": [str(pd.Timestamp(d.min()).date()), str(pd.Timestamp(d.max()).date())],
                          "rows": int(len(csv))},
        "raw_csv": {"file_sha256": file_sha256(raw_path)},
        "split_spec": {"name": spec.name, "digest": spec.digest()},
        "rl_data_start": settings.RL_DATA_START,
        "env_kwargs": KW,
        "fold_coverage": fold_coverage(csv, spec),
    }
    full = health_report(csv)
    full.pop("indicator_warmup_blocks", None)
    folds, gate = measure_folds(csv, spec)
    result["gate"] = {"full_panel": full, **gate}
    result["norms"] = folds
    if args.old_panel:
        result["panel_refresh"] = panel_refresh(os.path.join(ROOT, args.old_panel)
                                                if not os.path.isabs(args.old_panel) else args.old_panel,
                                                raw_path, settings.RL_DATA_START, args.dividends)

    out = args.out or os.path.join("results", f"r1_folds_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print("yazildi:", out)


if __name__ == "__main__":
    main()
