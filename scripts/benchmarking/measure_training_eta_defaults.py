"""
ETA varsayilan katsayilarini olc

`app/services/training_eta.py` icindeki `_DEFAULT_STEP_COST` tablosu, gecmis
kosum kaydi olmayan bir makinede ilk tahmini uretmek icin kullanilir. Bu script
o tabloyu **gercek egitim rotasiyla** uretir.

Neden rota uzerinden: izole bir `TradingEnv` + `model.learn` olcumu yaniltici
cikti. Rota (`app/api/routes/trading.py::_run_training_inner`) SB3'e cihaz
sectirir ve MlpPolicy CUDA'da CPU'dan yavas kosar; ayrica egitim, faz secimine
bakmaksizin yuklenen panelin tamamiyla (30 sembol) yapilir. Ilk olcum bu iki
farki kacirinca tahmin ~5x saptigi icin script rotanin kendisini cagirir.

Her algoritma iki farkli timestep sayisiyla kosturulur; dogru maliyeti egimden
okumak sabit maliyeti (veri yukleme, degerlendirme, model kaydi) disarida
birakir:

    sure(N) = sabit + N x adim_maliyeti
    adim_maliyeti = (sure(N2) - sure(N1)) / (N2 - N1)

Kullanim:
    python scripts/benchmarking/measure_training_eta_defaults.py
    python scripts/benchmarking/measure_training_eta_defaults.py --algos ppo,a2c
    python scripts/benchmarking/measure_training_eta_defaults.py --steps 4000,8000

Cikti dogrudan `_DEFAULT_STEP_COST` sozlugune yapistirilabilir.
"""

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.CRITICAL)

from app.api.routes import trading as tr  # noqa: E402
from app.schemas.trading import TrainingRequest  # noqa: E402

# Off-policy algoritmalar adim basina gradient adimi attigi icin cok daha yavas;
# ayni wall-clock butcesi altinda kalmak icin daha az timestep kosturulur.
_OFF_POLICY_SCALE = {"td3": 0.25, "sac": 0.25}


def run_once(algo, timesteps):
    state = tr._empty_training_state()
    state.update({"total_steps": timesteps, "run_start_ts": time.time()})
    request = TrainingRequest(
        algorithm=algo, phase=1, total_timesteps=timesteps, initial_balance=100_000,
    )
    t0 = time.perf_counter()
    asyncio.run(tr._run_training_inner(request, state))
    return time.perf_counter() - t0, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="ppo,a2c,td3,sac")
    ap.add_argument("--steps", default="4000,8000",
                    help="Iki timestep degeri (egim icin), virgulle")
    args = ap.parse_args()

    n1, n2 = [int(s) for s in args.steps.split(",")[:2]]

    print(f"Gercek egitim rotasi uzerinden olcum (timestep: {n1:,} ve {n2:,})\n")
    print(f"{'algo':6} {'N1 sure':>9} {'N2 sure':>9} {'sn/timestep':>13} "
          f"{'sabit':>8} {'sembol':>7} {'50k tahmini':>13}")
    print("-" * 74)

    results = {}
    n_symbols = None
    for algo in [a.strip().lower() for a in args.algos.split(",") if a.strip()]:
        scale = _OFF_POLICY_SCALE.get(algo, 1.0)
        s1, s2 = max(int(n1 * scale), 500), max(int(n2 * scale), 1000)
        try:
            t1, _ = run_once(algo, s1)
            t2, state = run_once(algo, s2)
        except Exception as exc:
            print(f"{algo:6} {'HATA':>9}  {exc}")
            continue

        cost = (t2 - t1) / (s2 - s1)
        fixed = t1 - s1 * cost
        if cost <= 0:
            print(f"{algo:6} {t1:>8.1f}s {t2:>8.1f}s  olcum gurultulu (egim <= 0), atlandi")
            continue

        hist = tr.training_eta.load_history()
        if hist:
            n_symbols = hist[-1].get("n_symbols")
        results[algo] = cost
        print(f"{algo:6} {t1:>8.1f}s {t2:>8.1f}s {cost:>13.6f} {fixed:>7.1f}s "
              f"{str(n_symbols or '?'):>7} {cost * 50_000 / 60:>11.1f} dk")

    if results:
        print("\n_DEFAULT_STEP_COST = {")
        for algo, cost in results.items():
            print(f'    "{algo}": {cost:.6f},')
        print("}")
        print(f"_DEFAULT_STEP_COST_SYMBOLS = {n_symbols or '?'}")


if __name__ == "__main__":
    main()
