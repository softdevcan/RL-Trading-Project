"""Faz 9 / R1 olcumu — R1 oncesi basarimla okunmus degerlendirme pencereleri (hakem-2 R1 #1, ikinci okuma).

Kaynaklar (salt okuma):
  1) Calisma alanlarindaki `*_metrics.json` dosyalari: eski rota test bolumunun islemlerini
     (`trades[*].date`), portfoy serisini (`portfolio_history`) ve basarim metriklerini
     yaziyordu; `test_range` alani R1 oncesi yok. Pencere islem tarihlerinden ve portfoy
     serisinin uzunlugundan, egitim anindaki panelin tarihleriyle eslenerek cikarilir.
  2) R1 oncesi oranli bolmenin (`DataFetcher.split_data`, 70/15/15) eski panelde (yedek
     `data/bist/backup_2026-07-27/`) urettigi test araligi: kaynak 1'den BAGIMSIZ kontrol.
  3) Faz 9 B9 etki olcumu (`docs/development/measurements/phase9-b9/price_scale_effect.json`).

Cikti `EXPOSED_EVAL_WINDOWS`'un (evaluation/trial_registry.py) dayanagidir. Bu bir OLCUMDUR.

Kullanim (depo kokunden):
    python scripts/analysis/measure_r1_exposure.py --out docs/development/measurements/phase9-r1/exposure.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

PERF_KEYS = ("sharpe_ratio", "cumulative_return", "max_drawdown", "total_trades")


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


def metrics_files():
    pats = [os.path.join(ROOT, "results", "**", "*_metrics.json"),
            os.path.join(ROOT, "workspaces", "*", "results", "**", "*_metrics.json")]
    out = []
    for p in pats:
        out += glob.glob(p, recursive=True)
    return sorted(set(out))


def scan(path):
    j = json.load(open(path, encoding="utf-8"))
    dates = sorted(str(t.get("date"))[:10] for t in (j.get("trades") or []) if t.get("date"))
    first_idx = min((t.get("index") for t in (j.get("trades") or []) if t.get("index") is not None),
                    default=None)
    return {"file": os.path.relpath(path, ROOT).replace("\\", "/"),
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
            "trained_at": (j.get("trained_at") or "")[:19] or None,
            "algorithm": j.get("algorithm"), "phase": j.get("phase"),
            "test_range_field": j.get("test_range"),
            "n_trades": len(dates), "first_trade": dates[0] if dates else None,
            "last_trade": dates[-1] if dates else None, "first_trade_index": first_idx,
            "n_portfolio_values": len(j.get("portfolio_history") or []),
            "performance_recorded": {k: j.get(k) for k in PERF_KEYS if k in j}}


def ratio_split_test(csv_path):
    """R1 oncesi rotanin test bolumu: split_data (70/15/15) panelin TARIHLERI uzerinde."""
    from data.data_fetcher import DataFetcher
    f = DataFetcher()
    df = f.load_data(os.path.relpath(csv_path, os.path.join(ROOT, f.data_dir)))
    tr, va, te = f.split_data(df)
    rng = lambda x: [str(pd.Timestamp(x.index.get_level_values("date").min()).date()),
                     str(pd.Timestamp(x.index.get_level_values("date").max()).date())]
    return {"panel": os.path.relpath(csv_path, ROOT).replace("\\", "/"), "train": rng(tr),
            "val": rng(va), "test": rng(te),
            "test_sessions": int(te.index.get_level_values("date").nunique())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-panel", default="data/bist/backup_2026-07-27/stock_data_with_indicators.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    commit, dirty = git_state()
    files = [scan(p) for p in metrics_files()]
    with_dates = [x for x in files if x["first_trade"]]
    b9 = json.load(open(os.path.join(ROOT, "docs/development/measurements/phase9-b9/price_scale_effect.json"),
                        encoding="utf-8"))
    groups = {}
    for x in with_dates:
        key = (x["trained_at"] or "")[:10]
        g = groups.setdefault(key, {"n": 0, "first_trade_min": x["first_trade"],
                                    "last_trade_max": x["last_trade"], "files": []})
        g["n"] += 1
        g["first_trade_min"] = min(g["first_trade_min"], x["first_trade"])
        g["last_trade_max"] = max(g["last_trade_max"], x["last_trade"])
        g["files"].append(x["file"])
    old_panel = os.path.join(ROOT, args.old_panel)
    result = {"created_at": datetime.now().isoformat(timespec="seconds"),
              "command": "python scripts/analysis/measure_r1_exposure.py " + " ".join(sys.argv[1:]),
              "git_commit": commit, "git_dirty": dirty,
              "n_metrics_files": len(files), "n_with_trade_dates": len(with_dates),
              "n_with_performance": sum(1 for x in files if x["performance_recorded"]),
              "by_training_day": groups, "files": files,
              "ratio_split_on_old_panel": ratio_split_test(old_panel) if os.path.exists(old_panel) else None,
              "b9_effect": {"test": b9.get("test"), "val": b9.get("val"), "command": b9.get("command")}}
    out = args.out or os.path.join("results", f"r1_exposure_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(json.dumps({k: result[k] for k in ("n_metrics_files", "n_with_trade_dates", "n_with_performance",
                                              "by_training_day", "ratio_split_on_old_panel", "b9_effect")},
                     indent=1, default=str)[:3000])
    print("yazildi:", out)


if __name__ == "__main__":
    main()
