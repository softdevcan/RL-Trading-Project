"""Faz 9 B5 — nedensel ICEEMDAN maliyet olcumu (plan §4 B5 "once maliyet olcumu"; K5).

Nedensel tasarimda her (sembol, gun t) icin son W ham kapanisin penceresi BIR kez
ayristirilir; sonuc (sembol, t, W, parametreler) anahtariyla onbellege girer, kollar,
katlar, tohumlar ve HPO denemeleri yeniden hesaplamaz. Toplam maliyet:
    sum_sembol n_gun(sembol)  x  ayristirma suresi  /  paralel verim.

Olcum: W in {128, 256} x trials in {50, 100}; her yapilandirmada her sembolden
--per-symbol pencere (2018 sonrasina esit yayilmis) bir surec havuzunda ayristirilir
(her surec tek is parcacigi, PyEMD parallel=False — varsayilan parallel=True her cagrida
ayri havuz aciyor). Verim duvar saatinden olculur, toplam saat ondan tahmin edilir.
Oz-denetim: ayni pencere iki kez ayristirilir, bit-es olmali.

    python scripts/analysis/measure_b5_iceemdan_cost.py \
        --out docs/development/measurements/phase9-b5/cost.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPSILON = 0.005  # prediction/iceemdan_processor.py varsayilani
SEED = 42        # ayni dosyadaki noise_seed


def _decompose(task):
    from PyEMD import CEEMDAN
    window, trials = task
    c = CEEMDAN(trials=trials, epsilon=EPSILON, parallel=False)
    c.noise_seed(SEED)
    t0 = time.perf_counter()
    imfs = c.ceemdan(np.asarray(window, dtype=float))
    return time.perf_counter() - t0, int(len(imfs))


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ROOT, "data", "bist", "raw_stock_data.csv"))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--windows", default="128,256")
    ap.add_argument("--trials", default="50,100")
    ap.add_argument("--per-symbol", type=int, default=4)
    ap.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.panel, "rb") as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    df = pd.read_csv(a.panel, usecols=["date", "symbol", "close"], parse_dates=["date"])
    df = df[df["close"] > 0].dropna(subset=["close"]).sort_values(["symbol", "date"])
    series = {s: g.set_index("date")["close"].to_numpy(float) for s, g in df.groupby("symbol")}
    dates = {s: g["date"].to_numpy() for s, g in df.groupby("symbol")}
    start = np.datetime64(a.start)
    Ws = [int(x) for x in a.windows.split(",")]
    Ts = [int(x) for x in a.trials.split(",")]

    # n_gun: start'tan sonraki seanslardan onunde W-1 gecmisi olanlar (panel 2005'ten)
    n_days = {}
    for W in Ws:
        n_days[W] = {s: int(sum(1 for i, d in enumerate(dates[s]) if d >= start and i >= W - 1))
                     for s in series}

    # Oz-denetim: belirlenimcilik
    s0 = sorted(series)[0]
    w = series[s0][-max(Ws):]
    from PyEMD import CEEMDAN
    A = CEEMDAN(trials=min(Ts), epsilon=EPSILON, parallel=False); A.noise_seed(SEED)
    B = CEEMDAN(trials=min(Ts), epsilon=EPSILON, parallel=False); B.noise_seed(SEED)
    ia, ib = A.ceemdan(w), B.ceemdan(w)
    deterministic = bool(ia.shape == ib.shape and np.array_equal(ia, ib))
    print(f"oz-denetim belirlenimcilik: {deterministic}", flush=True)

    rng = np.random.default_rng(0)
    configs = []
    ctx = get_context("spawn")
    with ctx.Pool(a.processes) as pool:
        for W in Ws:
            for T in Ts:
                tasks = []
                for s in sorted(series):
                    idx = [i for i, d in enumerate(dates[s]) if d >= start and i >= W - 1]
                    pick = np.linspace(0, len(idx) - 1, a.per_symbol).round().astype(int)
                    pick = np.unique(np.clip(pick + rng.integers(-5, 6, len(pick)), 0, len(idx) - 1))
                    tasks += [(series[s][idx[p] - W + 1: idx[p] + 1], T) for p in pick]
                t0 = time.perf_counter()
                res = pool.map(_decompose, tasks, chunksize=1)
                wall = time.perf_counter() - t0
                secs = np.array([r[0] for r in res]); nimf = np.array([r[1] for r in res])
                total = int(sum(n_days[W].values()))
                thr = len(tasks) / wall
                cfg = {
                    "W": W, "trials": T, "n_tasks": len(tasks), "wall_s": round(wall, 2),
                    "task_s_mean": round(float(secs.mean()), 4),
                    "task_s_p90": round(float(np.quantile(secs, 0.9)), 4),
                    "task_s_max": round(float(secs.max()), 4),
                    "throughput_per_s": round(thr, 4),
                    "n_imfs": {str(k): int(v) for k, v in zip(*np.unique(nimf, return_counts=True))},
                    "total_decompositions": total,
                    "est_hours_measured_pool": round(total / thr / 3600, 2),
                    "est_hours_single_process": round(total * float(secs.mean()) / 3600, 2),
                }
                configs.append(cfg)
                print(json.dumps(cfg), flush=True)

    out = {
        "what": "B5 nedensel ICEEMDAN maliyet olcumu (plan §4 B5, K5)",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([os.path.relpath(sys.argv[0], ROOT)] + sys.argv[1:]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "panel": os.path.relpath(a.panel, ROOT), "panel_sha256": panel_sha,
        "start": a.start, "n_symbols": len(series),
        "n_days_per_symbol": {str(W): {"min": min(v.values()), "max": max(v.values()),
                                       "sum": sum(v.values())} for W, v in n_days.items()},
        "params": {"epsilon": EPSILON, "noise_seed": SEED, "pyemd_parallel": False,
                   "per_symbol": a.per_symbol, "processes": a.processes},
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "python": sys.version.split()[0]},
        "self_check_deterministic": deterministic,
        "configs": configs,
        "note": "Tahmin = toplam ayristirma / olculen havuz verimi. Kenar etkisi (W duyarliligi) "
                "ve ozellik kalitesi bu olcumun kapsami DISINDA (B5 uygulamasinda).",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("yazildi:", a.out, flush=True)
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
