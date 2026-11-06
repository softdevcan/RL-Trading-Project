"""
Faz 9 / C — tekrar uretilebilirlik (evaluation/reproducibility.py,
evaluation/trial_registry.py, prediction/manifest.py).

  C1 ayni tohum + CPU -> iki PPO/A2C kosumu ayni odul egrisi ve ayni agirliklar;
     farkli tohum -> farkli. GPU varsa sapma OLCULUR ve basilir (assert yok —
     cuDNN/atomik islemler bit-eslik garanti etmez; Henderson vd. 2018).
  C2 deneme kaydi: yaz/oku, ayni yapilandirmanin tohumlari ayni config_hash,
     DSR'nin N ve V'si, PBO icin hizali getiri matrisi.
  C3 veri anlik goruntusu: hash kararli, satir sirasina duyarsiz, veri
     degisince verify uyarir.
  C4 ortam kaydi + manifest alanlari (seed, environment, data_snapshots).

Calistirma:
    python tests/test_rl_reproducibility.py
"""

import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

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


def panel(n_days=80, n_sym=3, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 50.0 * (s + 1)
        for d in dates:
            p *= 1 + rng.normal(0, 0.015)
            rows.append({"symbol": f"R{s}.IS", "date": d, "open": p, "high": p * 1.01,
                         "low": p * 0.99, "close": p, "volume": 1e6, "macd": 0.0,
                         "rsi": 50.0, "cci": 0.0, "adx": 20.0, "turbulence": 0.0})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


# ---------------------------------------------------------------------------
print("\nC1 — RL tohumu")
from stable_baselines3 import A2C, PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from env.trading_env import make_env  # noqa: E402
from evaluation.reproducibility import resolve_seed  # noqa: E402

df = panel()


def run(algo, seed, device="cpu", steps=256):
    env = DummyVecEnv([lambda: make_env(df=df, initial_balance=100_000,
                                        missing_data_mode="own_last")])
    kw = dict(n_steps=64, batch_size=32, n_epochs=2) if algo is PPO else dict(n_steps=16)
    model = algo("MlpPolicy", env, seed=seed, device=device, verbose=0, **kw)
    rewards = []

    from stable_baselines3.common.callbacks import BaseCallback

    class Rec(BaseCallback):
        def _on_step(self):
            rewards.append(float(self.locals["rewards"][0]))
            return True

    model.learn(total_timesteps=steps, callback=Rec())
    params = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.policy.parameters()])
    return np.array(rewards), params


for algo in (PPO, A2C):
    r1, p1 = run(algo, 123)
    r2, p2 = run(algo, 123)
    r3, p3 = run(algo, 124)
    check(f"{algo.__name__}: ayni tohum (CPU) -> ayni odul egrisi", np.array_equal(r1, r2))
    check(f"{algo.__name__}: ayni tohum (CPU) -> ayni agirliklar", np.array_equal(p1, p2))
    check(f"{algo.__name__}: farkli tohum -> farkli agirliklar", not np.array_equal(p1, p3))

try:
    import torch
    if torch.cuda.is_available():
        g1 = run(PPO, 123, device="cuda")
        g2 = run(PPO, 123, device="cuda")
        print(f"    olcum (GPU {torch.cuda.get_device_name(0)}): odul egrisi esit={np.array_equal(g1[0], g2[0])}, "
              f"max |agirlik farki|={np.max(np.abs(g1[1] - g2[1])):.3e}")
    else:
        print("    (GPU yok — GPU sapma olcumu atlandi)")
except Exception as exc:  # olcum, test degil
    print(f"    (GPU olcumu yapilamadi: {exc})")

s = resolve_seed(None)
check("resolve_seed(None) gecerli bir tohum uretir", isinstance(s, int) and 0 <= s < 2 ** 31)
check("resolve_seed(7) -> 7", resolve_seed(7) == 7)


# ---------------------------------------------------------------------------
print("\nC2 — Deneme kaydi")
from evaluation.sharpe import deflated_sharpe_ratio  # noqa: E402
from evaluation.trial_registry import (  # noqa: E402
    config_hash, load_trials, register_trial, returns_matrix, sharpe_trial_stats,
)

with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    clean = {"price_stats_source": "train_window", "indicator_bfill": False, "adx_constant": False}
    cfgs = [{"algorithm": "PPO", "lr": lr} for lr in (1e-4, 3e-4, 1e-3)]
    for i, cfg in enumerate(cfgs):
        for seed in (1, 2):
            register_trial("rl_hpo", cfg, seed, metrics={"sharpe": 0.1},
                           returns=rng.normal(0.0005, 0.01, 300), dates=dates,
                           split="val", directory=tmp, protocol=clean,
                           data_sha256=f"veri{i}")          # veri tazelense de ayni config
    register_trial("rl_hpo", {"algorithm": "PPO", "lr": 5e-3}, 1, split="val",
                   status="pruned", directory=tmp, protocol=clean)
    register_trial("spike", cfgs[0], 9, returns=rng.normal(0, 0.01, 300), dates=dates,
                   tags=["spike"], directory=tmp, protocol=clean)
    register_trial("rl_train", cfgs[0], 3, returns=rng.normal(0, 0.01, 300), dates=dates,
                   split="test", directory=tmp,
                   protocol={"price_stats_source": "test_window", "indicator_bfill": True,
                             "adx_constant": True})
    trials = load_trials(directory=tmp)
    check("7 deneme okundu (spike ve acik-protokol varsayilan olarak dislandi)",
          len(trials) == 7, str(len(trials)))
    open_ = [t for t in load_trials(directory=tmp, exclude_tags=("spike",)) if t["kind"] == "rl_train"]
    check("acik-protokol kaydi etiketli ve maddeleri listeli",
          len(open_) == 1 and "protocol_open" in open_[0]["tags"]
          and set(open_[0]["protocol_open"]) == {"price_stats_source", "indicator_bfill", "adx_constant"})
    check("ayni yapilandirmanin tohumlari ayni config_hash (veri hash'i config disinda)",
          len({t["config_hash"] for t in trials if t["status"] == "completed"}) == 3)
    check("tohum kayitta", sorted({t["seed"] for t in trials}) == [1, 2])
    st = sharpe_trial_stats(trials)
    check("varsayilan by='trial': N = 6 tamamlanan + 1 budanan = 7", st["n_trials"] == 7, str(st["n_trials"]))
    check("V yalnizca getirisi olanlardan (6)", st["n_with_returns"] == 6)
    check("V > 0 ve sonlu", st["var_trials"] is not None and 0 < st["var_trials"] < 1)
    st_c = sharpe_trial_stats(trials, by="config")
    check("by='config': N = 4 yapilandirma (budanan dahil)", st_c["n_trials"] == 4)
    check("by='config' tohum ortalamasi V'yi kucultur (hakem-2 #2b, belgelendi)",
          st_c["var_trials"] < st["var_trials"])
    mixed = trials + open_
    try:
        sharpe_trial_stats(mixed)
        check("farkli kind/split karistirilamaz", False)
    except ValueError:
        check("farkli kind/split karistirilamaz", True)
    register_trial("rl_hpo", cfgs[0], 7, returns=rng.normal(0, 0.01, 200),
                   dates=pd.bdate_range("2025-01-01", periods=200), split="val",
                   directory=tmp, protocol=clean)
    try:
        sharpe_trial_stats(load_trials(directory=tmp))
        check("farkli degerlendirme pencereleri karistirilamaz", False)
    except ValueError:
        check("farkli degerlendirme pencereleri karistirilamaz", True)
    M = returns_matrix(trials)
    check("PBO matrisi tarih x deneme, hizali (budanan haric)", M.shape == (300, 6) and M.index.equals(dates))
    d = deflated_sharpe_ratio(n_trials=st["n_trials"], var_trials=st["var_trials"], returns=M.iloc[:, 0])
    check("kayittan DSR hesaplanabiliyor", 0.0 <= d["dsr"] <= 1.0)
    try:
        register_trial("rl_hpo", {}, 1, status="bilinmeyen", directory=tmp)
        check("gecersiz status reddediliyor", False)
    except ValueError:
        check("gecersiz status reddediliyor", True)
    check("config_hash anahtar sirasina duyarsiz", config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1}))
    try:
        register_trial("rl_train", {}, 1, returns=[0.1, 0.2], dates=dates[:3], directory=tmp)
        check("returns/dates uzunluk uyusmazligi reddediliyor", False)
    except ValueError:
        check("returns/dates uzunluk uyusmazligi reddediliyor", True)


# ---------------------------------------------------------------------------
print("\nC3 — Veri anlik goruntusu")
from evaluation.reproducibility import frame_sha256, snapshot_frame, verify_snapshot  # noqa: E402

h1 = frame_sha256(df)
check("hash kararli", h1 == frame_sha256(df.copy()))
check("satir sirasina duyarsiz (kanonik siralama)", h1 == frame_sha256(df.sample(frac=1, random_state=0)))
mod = df.copy()
mod.iloc[5, mod.columns.get_loc("close")] *= 1.0001
check("tek degerin degismesi hash'i degistiriyor", frame_sha256(mod) != h1)
with tempfile.TemporaryDirectory() as tmp:
    entry = snapshot_frame(df, "rl_panel", directory=tmp)
    check("parquet yazildi", entry["path"] and os.path.exists(entry["path"]))
    check("ozet alanlari", entry["rows"] == len(df) and entry["n_symbols"] == 3 and entry["start"] == "2021-01-01")
    check("kayitli dosyadan dogrulama ok", verify_snapshot(entry)["ok"])
    check("bugunku veri degisince verify uyarir (ok=False)", not verify_snapshot(entry, df=mod)["ok"])
    again = snapshot_frame(df, "rl_panel", directory=tmp)
    check("ayni veri ikinci kez yazilmiyor (ayni yol)", again["path"] == entry["path"]
          and len(os.listdir(tmp)) == 1)


# ---------------------------------------------------------------------------
print("\nC4 — Ortam kaydi ve manifest")
from evaluation.reproducibility import environment_record  # noqa: E402
from prediction.manifest import TrainingManifest, verify_manifest_snapshots  # noqa: E402

rec = environment_record()
check("paket surumleri kayitli", rec["packages"].get("numpy") and rec["packages"].get("stable_baselines3"))
check("torch belirlenimcilik bayraklari kayitli",
      rec["torch"] is not None and "cudnn_deterministic" in rec["torch"])
with tempfile.TemporaryDirectory() as tmp:
    m = TrainingManifest(run_kind="rl_train", symbols=["R0.IS"], expected_models=set())
    m.runs_dir = tmp
    m.set_seed(42)
    entry = snapshot_frame(df, "rl_panel", directory=tmp)
    m.add_data_snapshot(entry)
    path = m.finalize()
    data = json.load(open(path, encoding="utf-8"))
    check("manifestte seed", data.get("seed") == 42)
    check("manifestte environment", "packages" in data.get("environment", {}))
    check("manifestte data_snapshots", data.get("data_snapshots", [{}])[0].get("sha256") == entry["sha256"])
    check("verify_manifest_snapshots ok", all(r["ok"] for r in verify_manifest_snapshots(path)))
    m2 = TrainingManifest(run_kind="batch_train", symbols=[])
    m2.runs_dir = tmp
    d2 = json.load(open(m2.finalize(), encoding="utf-8"))
    check("tohum verilmeyen manifest 'seed' alani tasimiyor (uydurulmuyor)", "seed" not in d2)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
