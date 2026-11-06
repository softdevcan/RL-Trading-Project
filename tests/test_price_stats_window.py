"""
Faz 9 / B9 (Faz 10 Y13) — val/test ortami EGITIM penceresinin fiyat olcegiyle.

Kabul olcutu (Faz 10 §3.1.1 B9): "Test penceresinde t sonrasi fiyatlar
degistirildiginde t gozlemi bit-es kalir." Uc duzeyde sinanir:

  E  env      : TradingEnv(price_stats=...) — t'ye kadar (dahil) TUM gozlem
                vektoru bit-es. Eski davranis (kendi penceresi) ayni olcutten
                KALIR: sizinti sayisal olarak gosterilir.
  R  rota     : _run_training_inner uctan uca (sentetik panel, CPU, sabit
                tohum). Rotanin test ortaminin GORDUGU gozlem dizisi t'ye kadar
                bit-es; deneme kaydi/manifest/meta 'train_window' tasir.
  H  HPO      : objective() — dogrulama ortami egitim ortaminin olcegini alir;
                kayit 'train_window', ham OHLCV yuzunden yine 'protocol_open'.
  S  betikler : sarmalayici fonksiyonla env kuran betikler (AST bekcisinin
                goremedigi yol) davranisla: train_a2c_phase1, ab_test_rewards,
                optimize_reward_weights, generate_academic_report.
  G  bekci    : held-out (test/val/eval) adli df ile kurulan her make_env /
                TradingEnv cagrisi price_stats tasimali (AST taramasi). Yalnizca
                DOGRUDAN cagrilari gorur; sarmalayicilar S ve H'de.

Her olcut icin MUTASYON oz-testi: olcek parametresini kabul edip kullanmayan
env, kopyalamayan env ve olcegi gecirmeyen rota olcutu KIRMALI. Kirmazsa
test ici bos demektir.

Varsayilan davranis (price_stats=None) main'deki koddan uretilmis dondurulmus
yorunge ozetiyle (SHA-256) bit-es denetlenir (D).

Calistirma:
    python tests/test_price_stats_window.py
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Rota/HPO uctan uca kosumu GERCEK dizinlere yazmasin; model CPU'da egitilsin
# (tohumla bit-eslik CPU'da olculdu, C1). "-1": Windows'ta bos dize degiskeni
# SILER ve GPU gorunur kalir (ilk kosumda boyle oldu, rota CUDA'da egitildi).
_TMP = tempfile.mkdtemp(prefix="b9_price_stats_")
for _k, _sub in (("MODELS_DIR", "models"), ("RESULTS_DIR", "results"),
                 ("LOGS_DIR", "logs"), ("HYPEROPT_DIR", "hyperopt")):
    os.environ[_k] = os.path.join(_TMP, _sub)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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


# ---------------------------------------------------------------------------
# Sentetik veri
# ---------------------------------------------------------------------------

def synthetic_panel(n_days=160, n_sym=3, seed=5, indicators=True, start="2021-01-04",
                    drift=0.002, drift_after=None):
    """Yukari egilimli rastgele yuruyus: test penceresi egitimden pahali (TL gibi).

    drift_after=(gun, drift): o gunden sonra egilim degisir (rejim kirilmasi;
    egitim penceresi yatay, sonrasi sert yukselis -> egitim olceginde kirpma).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 40.0 * (s + 1)
        for i, d in enumerate(dates):
            mu = drift if drift_after is None or i < drift_after[0] else drift_after[1]
            p *= 1 + mu + rng.normal(0, 0.012)
            o = p * (1 + rng.normal(0, 0.003))
            h = max(o, p) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, p) * (1 - abs(rng.normal(0, 0.004)))
            row = {"symbol": f"S{s}.IS", "date": d, "open": o, "high": h, "low": lo,
                   "close": p, "volume": float(rng.integers(500_000, 2_000_000))}
            if indicators:
                row.update(macd=float(rng.normal(0, 0.05)), rsi=float(rng.uniform(20, 80)),
                           cci=float(rng.normal(0, 80)), adx=float(rng.uniform(10, 40)),
                           turbulence=float(abs(rng.normal(0, 1))))
            rows.append(row)
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def dates_of(frame):
    return pd.DatetimeIndex(sorted(frame.index.get_level_values("date").unique()))


def perturb_after(frame, t):
    """t'den SONRAKI fiyatlari yumusak bir surukleyle degistir (gunluk %2;
    veri kapisinin sicrama/asiri getiri esiklerine takilmaz)."""
    out = frame.copy()
    d = pd.to_datetime(out.index.get_level_values("date"))
    later = dates_of(out)[dates_of(out) > t]
    rank = pd.Series(np.arange(1, len(later) + 1), index=later)
    mask = d > t
    factor = 1.02 ** rank.reindex(d[mask]).to_numpy()
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * factor
    return out


def own_stats(frame):
    """_compute_price_stats ile ayni tanim, bagimsiz hesap."""
    out = {}
    for s in frame.index.get_level_values("symbol").unique():
        c = frame.xs(s, level="symbol")["close"].astype(float)
        sd = float(c.std())
        out[s] = {"mean": float(c.mean()), "std": sd if sd > 1e-8 else 1.0}
    return out


def obs_trajectory(env, actions):
    obs, _ = env.reset()
    out = [obs.copy()]
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        out.append(obs.copy())
        if term or trunc:
            break
    return out


def trajectory_digest(env_cls, panel, mode, n_steps=None):
    """Varsayilan yolun parmak izi: gozlem + odul + portfoy degeri (D)."""
    env = env_cls(panel, initial_balance=100_000, missing_data_mode=mode)
    rng = np.random.default_rng(42)
    n = n_steps or (env.frame_bound[1] - env.frame_bound[0])
    actions = rng.uniform(-1, 1, size=(n, env.n_stocks)).astype(np.float32)
    h = hashlib.sha256()
    obs, _ = env.reset()
    h.update(obs.tobytes())
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        h.update(obs.tobytes())
        h.update(np.float64(r).tobytes())
        h.update(np.float64(info["portfolio_value"]).tobytes())
        if term or trunc:
            break
    return h.hexdigest()


def holed(panel):
    """Bir sembolun ortadaki 5 gunu eksik (legacy kopya ve own_last yolu calissin)."""
    d = dates_of(panel)
    drop = d[60:65]
    idx = panel.index
    mask = (idx.get_level_values("symbol") == "S1.IS") & idx.get_level_values("date").isin(drop)
    return panel[~mask]


# main @ 78edf06'daki env/trading_env.py ile uretildi (B9 oncesi):
#   git show 78edf06:env/trading_env.py > <tmp>/trading_env_main.py
#   trajectory_digest(<o modulun TradingEnv'i>, ...) — ayni 4 senaryo.
# Varsayilan yol (price_stats=None) degisirse bu ozetler degisir.
FROZEN_DIGESTS = {
    ("full", "legacy"): "333a13a18e96b7fb07015e470c9a069d1b3f60f2e712984916c6b58e963eaf7f",
    ("full", "own_last"): "333a13a18e96b7fb07015e470c9a069d1b3f60f2e712984916c6b58e963eaf7f",
    ("holed", "legacy"): "55e94c65a87f188c9785f8c823ffc89fe3766d44afe7ebfd1dfeca26fb83714a",
    ("holed", "own_last"): "d738a67bc5c06d45d8717b25ce8c67aa17eb2f9660e72af2cde3fddd5aa00ff7",
}


# ---------------------------------------------------------------------------
# AST bekcisi
# ---------------------------------------------------------------------------

_HELD_OUT = re.compile(r"(^|_)(test|val|valid|validation|eval)(_|$)", re.IGNORECASE)
_LEGACY_MARK = "B9-legacy"
_SKIP_DIRS = {"venv", ".git", "docs", "workspaces", "node_modules", "__pycache__",
              "results", "models", "logs", ".claude"}


def _held_out_name(node):
    """df argumaninin held-out adini dondur (test_df, test_df.copy(), data['test'])."""
    while isinstance(node, (ast.Call, ast.Attribute)):
        node = node.func if isinstance(node, ast.Call) else node.value
    if isinstance(node, ast.Name) and _HELD_OUT.search(node.id):
        return node.id
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and _HELD_OUT.search(sl.value):
            return f"[{sl.value!r}]"
    return None


def held_out_env_calls(source, filename="<src>"):
    """(dosya, satir, df adi, uyumlu_mu) — uyumlu: price_stats var ya da isaretli."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name not in ("make_env", "TradingEnv"):
            continue
        df_arg = next((k.value for k in node.keywords if k.arg == "df"),
                      node.args[0] if node.args else None)
        held = _held_out_name(df_arg) if df_arg is not None else None
        if not held:
            continue
        ok = any(k.arg == "price_stats" for k in node.keywords) or _LEGACY_MARK in lines[node.lineno - 1]
        out.append((filename, node.lineno, held, ok))
    return out


def scan_repo():
    calls = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                src = open(path, encoding="utf-8").read()
                calls.extend(held_out_env_calls(src, os.path.relpath(path, ROOT)))
            except (SyntaxError, UnicodeDecodeError):
                continue
    return calls


# ---------------------------------------------------------------------------
# Rota ve HPO uctan uca
# ---------------------------------------------------------------------------

def run_route(panel, strip_price_stats=False, seed=11):
    """_run_training_inner'i sentetik panelle kostur; olusan env'leri ve
    her env'in GORDUGU gozlem dizisini yakala.

    strip_price_stats=True MUTASYONDUR: rotanin B9 oncesi hali (test ortamina
    olcek gecirilmez).
    """
    import asyncio

    import data.data_fetcher as dfm
    import env.trading_env as tem
    from app.api.routes import trading as tr
    from app.schemas.trading import TrainingRequest

    created = []
    orig_make_env = tem.make_env

    def rec_make_env(df, **kw):
        if strip_price_stats:
            kw.pop("price_stats", None)
        env = orig_make_env(df, **kw)
        log = []
        inner = env._get_observation

        def logged():
            o = inner()
            log.append(o.copy())
            return o

        env._get_observation = logged
        created.append({"kw": kw, "env": env, "obs": log})
        return env

    orig_load = dfm.DataFetcher.load_data
    dfm.DataFetcher.load_data = lambda self, filename="raw_stock_data.csv": panel.copy()
    tem.make_env = rec_make_env
    try:
        state = tr._empty_training_state()
        req = TrainingRequest(algorithm="A2C", phase=1, total_timesteps=256,
                              initial_balance=100_000, seed=seed)
        asyncio.run(tr._run_training_inner(req, state))
    finally:
        tem.make_env = orig_make_env
        dfm.DataFetcher.load_data = orig_load
    return state, created


def env_for_window(created, start):
    hits = [c for c in created if dates_of(c["env"].df)[0] == start]
    return hits[-1] if hits else None


def route_trial(state):
    from evaluation.trial_registry import load_trials
    tid = state["metrics"]["trial_id"]
    recs = [t for t in load_trials(directory=os.environ["RESULTS_DIR"], exclude_tags=())
            if t["trial_id"] == tid]
    return recs[0] if recs else None


def main():
    global passed, failed
    from app.core.config import get_settings
    get_settings.cache_clear()

    import evaluation.reproducibility as rep
    rep.SNAPSHOT_DIR = os.path.join(_TMP, "snapshots")

    from env.trading_env import TradingEnv
    from evaluation.trial_registry import (
        CLEAN_PROTOCOL, indicators_present, open_protocol_items, price_stats_source,
        rl_eval_protocol,
    )

    panel = synthetic_panel()
    d = dates_of(panel)
    train_df = panel[panel.index.get_level_values("date") <= d[109]]
    test_df = panel[panel.index.get_level_values("date") > d[109]]
    tdates = dates_of(test_df)
    t_idx = 15
    t = tdates[t_idx]
    train_env = TradingEnv(train_df, initial_balance=100_000, missing_data_mode="own_last")
    train_stats = train_env.get_price_stats()
    actions = np.random.default_rng(1).uniform(-1, 1, size=(len(tdates), 3)).astype(np.float32)

    # -----------------------------------------------------------------------
    print("\nD — varsayilan yol (price_stats=None) degismedi")
    e0 = TradingEnv(test_df, initial_balance=100_000)  # B9-legacy
    check("kaynak etiketi 'own_window'", e0.price_stats_source == "own_window")
    check("olcek bagimsiz hesapla bit-es (kendi penceresi)", e0.get_price_stats() == own_stats(test_df))
    frozen_ready = all(v is not None for v in FROZEN_DIGESTS.values())
    for (pname, mode) in FROZEN_DIGESTS:
        p = panel if pname == "full" else holed(panel)
        dig = trajectory_digest(TradingEnv, p, mode)
        want = FROZEN_DIGESTS[(pname, mode)]
        if want is None:
            print(f"    ozet ({pname}, {mode}) = {dig}")
        else:
            check(f"yorunge ozeti main ile bit-es ({pname}, {mode})", dig == want, dig)
    if not frozen_ready:
        check("dondurulmus ozetler doldurulmus", False, "FROZEN_DIGESTS bos")

    # -----------------------------------------------------------------------
    print("\nE — env: verilen olcek, kabul olcutu, sizinti")
    e1 = TradingEnv(test_df, initial_balance=100_000, price_stats=train_stats)
    check("kaynak etiketi 'supplied'", e1.price_stats_source == "supplied")
    check("olcek egitim penceresinin olcegi (aynen)", e1.get_price_stats() == own_stats(train_df))
    o = e1.reset()[0]
    blk = o[1 + 3:].reshape(3, 10)
    first = test_df.xs(tdates[0], level="date")
    exp = [(first.loc[s, "close"] - train_stats[s]["mean"]) / train_stats[s]["std"] for s in e1.symbols]
    check("gozlemdeki kapanis z'si (p - egitim ort.) / egitim std",
          np.allclose(blk[:, 3], np.clip(np.array(exp, dtype=np.float32), -10, 10), rtol=0, atol=1e-5))

    def acceptance(make):
        """t'ye kadar (dahil) gozlem bit-es mi + t sonrasi gercekten degisiyor mu."""
        a = obs_trajectory(make(test_df), actions)
        b = obs_trajectory(make(perturb_after(test_df, t)), actions)
        prefix = all(np.array_equal(a[i], b[i]) for i in range(t_idx + 1))
        later = any(not np.array_equal(a[i], b[i]) for i in range(t_idx + 1, len(a)))
        diff0 = float(np.max(np.abs(a[0] - b[0])))
        return prefix, later, diff0

    pre, later, _ = acceptance(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats))
    check("KABUL: t sonrasi fiyatlar degisince t'ye kadar gozlem bit-es (B9)", pre)
    check("  t sonrasi gozlem gercekten degisiyor (olcut bos degil)", later)
    pre_l, _, diff0 = acceptance(lambda f: TradingEnv(f, initial_balance=100_000))  # B9-legacy
    check("SIZINTI: eski davranista t=0 gozlemi bile degisiyor (olcut kaliyor)",
          (not pre_l) and diff0 > 0, f"max|fark| t=0: {diff0:.4f}")
    print(f"    eski davranis t=0 gozleminde max|fark| = {diff0:.4f} (gelecek fiyatlardan)")
    pre_o, _, _ = acceptance(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats,
                                                  missing_data_mode="own_last"))
    check("KABUL own_last modunda da gecerli", pre_o)

    print("\nM — mutasyon oz-testleri (olcut KIRILMALI)")
    orig_validate = TradingEnv._validated_price_stats
    try:
        def ignore_supplied(self, supplied):  # parametreyi kabul et, KULLANMA
            self._compute_price_stats()
            return self.price_stats
        TradingEnv._validated_price_stats = ignore_supplied
        pre_m, _, _ = acceptance(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats))
        check("mutant 'olcegi yok say' kabul olcutunu kiriyor", not pre_m)

        TradingEnv._validated_price_stats = lambda self, supplied: supplied  # kopyalamadan
        st = {s: dict(v) for s, v in train_stats.items()}
        em = TradingEnv(test_df, initial_balance=100_000, price_stats=st)
        before = em.reset()[0]
        st["S0.IS"]["mean"] += 1000.0
        after = em.reset()[0]
        check("mutant 'kopyalama' kopya olcutunu kiriyor", not np.array_equal(before, after))
    finally:
        TradingEnv._validated_price_stats = orig_validate

    st = {s: dict(v) for s, v in train_stats.items()}
    ec = TradingEnv(test_df, initial_balance=100_000, price_stats=st)
    before = ec.reset()[0]
    st["S0.IS"]["mean"] += 1000.0
    check("kopya: cagiranin sozlugu sonradan degisse de gozlem degismez",
          np.array_equal(before, ec.reset()[0]))

    print("\nV — denetim (sessiz yedege dusmez)")

    def raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    part = {s: v for s, v in train_stats.items() if s != "S2.IS"}
    check("eksik sembol -> ValueError", raises(lambda: TradingEnv(test_df, price_stats=part)))
    bad = {s: dict(v) for s, v in train_stats.items()}
    bad["S1.IS"]["mean"] = float("nan")
    check("NaN ortalama -> ValueError", raises(lambda: TradingEnv(test_df, price_stats=bad)))
    bad = {s: dict(v) for s, v in train_stats.items()}
    bad["S1.IS"]["std"] = 0.0
    check("std = 0 -> ValueError", raises(lambda: TradingEnv(test_df, price_stats=bad)))
    bad = {s: {"mean": 1.0} for s in train_stats}
    check("'std' anahtari yok -> ValueError", raises(lambda: TradingEnv(test_df, price_stats=bad)))
    extra = dict(train_stats, **{"XX.IS": {"mean": 1.0, "std": 1.0}})
    ee = TradingEnv(test_df, price_stats=extra)
    check("fazla sembol yok sayilir", set(ee.get_price_stats()) == set(ee.symbols))
    e2 = TradingEnv(test_df, phase=2, price_stats=train_stats)
    check("faz 2 ortami da olcegi kabul ediyor", e2.get_price_stats() == own_stats(train_df))

    # -----------------------------------------------------------------------
    print("\nP — protokol yardimcilari (etiket OLCULUR)")
    from stable_baselines3.common.monitor import Monitor
    check("kendi penceresi -> 'own_window'", price_stats_source(e0, train_stats) == "own_window")
    check("egitim olcegi -> 'train_window'", price_stats_source(e1, train_stats) == "train_window")
    check("Monitor sarmalayicisi aciliyor",
          price_stats_source(Monitor(TradingEnv(test_df, price_stats=train_stats)), train_stats)
          == "train_window")
    other = {s: {"mean": v["mean"] * 1.1, "std": v["std"]} for s, v in train_stats.items()}
    check("baska olcek -> 'supplied_other'",
          price_stats_source(TradingEnv(test_df, price_stats=other), train_stats) == "supplied_other")
    check("referanssiz -> 'supplied_unverified'", price_stats_source(e1, None) == "supplied_unverified")
    check("ortam yok -> None", price_stats_source(None, train_stats) is None)
    raw = panel[["open", "high", "low", "close", "volume"]]
    check("gostergeli panel -> indicators_present True", indicators_present(panel))
    check("ham OHLCV -> indicators_present False", not indicators_present(raw))
    pr = rl_eval_protocol(e1, train_stats)
    check("gostergeli + egitim olcegi: acik madde yalnizca indicator_bfill",
          open_protocol_items(pr) == ["indicator_bfill"], str(open_protocol_items(pr)))
    pr_raw = rl_eval_protocol(TradingEnv(raw.loc[test_df.index], price_stats=train_stats), train_stats)
    check("ham OHLCV: adx sabit, bfill anlamsiz (None), gosterge yok",
          pr_raw["adx_constant"] is True and pr_raw["indicator_bfill"] is None
          and pr_raw["indicators_present"] is False)
    check("tum maddeler temizse acik madde yok",
          open_protocol_items(dict(CLEAN_PROTOCOL)) == [])

    # -----------------------------------------------------------------------
    print("\nR — rota uctan uca (_run_training_inner)")
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel

    rl_panel, _ = prepare_rl_panel(panel, start=get_settings().RL_DATA_START)
    r_train, _, r_test = DataFetcher().split_data(rl_panel)
    r_tdates = dates_of(r_test)
    k = 6
    r_t = r_tdates[k]

    state_a, created_a = run_route(panel)
    check("rota kosumu tamamlandi", state_a.get("state") == "completed", str(state_a.get("error")))
    test_a = env_for_window(created_a, r_tdates[0])
    check("rotanin test ortami bulundu", test_a is not None)
    if state_a.get("state") == "completed" and test_a is not None:
        exp_train = own_stats(r_train)
        check("test ortami 'supplied' ve olcegi EGITIM bolumunun olcegi",
              test_a["env"].price_stats_source == "supplied"
              and test_a["env"].get_price_stats() == exp_train)
        check("  (ayirt edici: test penceresinin kendi olcegiyle ayni DEGIL)",
              test_a["env"].get_price_stats() != own_stats(r_test))
        rec = route_trial(state_a)
        check("deneme kaydi price_stats_source='train_window'",
              rec is not None and rec["protocol"].get("price_stats_source") == "train_window",
              str(rec and rec["protocol"]))
        check("  price_stats_source acik maddelerde DEGIL",
              rec is not None and "price_stats_source" not in rec["protocol_open"])
        check("  kayit yine protocol_open (B7/K18 acik)",
              rec is not None and "protocol_open" in rec["tags"])
        import json
        man = json.load(open(os.path.join(os.environ["RESULTS_DIR"], "training_runs",
                                          f"{state_a['metrics']['run_id']}.json"), encoding="utf-8"))
        check("manifest protokolu ayni", man.get("extra", {}).get("protocol") == rec["protocol"])
        meta_path = os.path.join(os.environ["MODELS_DIR"], f"{rec['extra']['model_name']}.meta.json")
        meta = json.load(open(meta_path, encoding="utf-8"))
        check("meta price_stats = egitim olcegi = test ortaminin olcegi",
              meta["price_stats"] == exp_train == test_a["env"].get_price_stats())

        def expected_health(entry):
            seen = np.asarray(entry["obs"][:-1], dtype=np.float32)  # son gozlemle karar verilmez
            n_s, fps = entry["env"].n_stocks, entry["env"].features_per_stock
            px = seen[:, 1 + n_s: 1 + n_s + n_s * fps].reshape(len(seen), n_s, fps)[:, :, :4]
            return {"test_obs_clip_frac": float(np.mean(np.abs(seen) >= 10.0)),
                    "test_price_clip_frac": float(np.mean(np.abs(px) >= 10.0)),
                    "n_obs": int(len(seen))}

        oh = rec.get("extra", {}).get("observation_health")
        check("gozlem sagligi kayitta + manifestte, bagimsiz hesapla ayni",
              oh == expected_health(test_a) == man.get("extra", {}).get("observation_health"),
              f"{oh} / {expected_health(test_a)}")

        hot = synthetic_panel(drift=0.0, drift_after=(100, 0.03))
        state_h, created_h = run_route(hot)
        hot_rl, _ = prepare_rl_panel(hot, start=get_settings().RL_DATA_START)
        test_h = env_for_window(created_h, dates_of(DataFetcher().split_data(hot_rl)[2])[0])
        oh_h = route_trial(state_h)["extra"]["observation_health"]
        check("guclu yukselis: test fiyat ozellikleri kirpiliyor ve olcum dogru",
              oh_h["test_price_clip_frac"] > 0.5 and oh_h == expected_health(test_h), str(oh_h))
        check("  kirpma panoya uyari olarak tasiniyor",
              any("kirpildi" in w for w in state_h.get("warnings", [])))
        check("  dusuk kirpmada uyari yok (esik %10)",
              not any("kirpildi" in w for w in state_a.get("warnings", [])))

        state_b, created_b = run_route(perturb_after(panel, r_t))
        test_b = env_for_window(created_b, r_tdates[0])
        oa, ob = test_a["obs"], test_b["obs"]
        pre = len(oa) > k + 1 and all(np.array_equal(oa[i], ob[i]) for i in range(k + 1))
        later = any(not np.array_equal(oa[i], ob[i]) for i in range(k + 1, min(len(oa), len(ob))))
        check("KABUL (rota): t sonrasi fiyatlar degisince test gozlemleri t'ye kadar bit-es", pre)
        check("  t sonrasi gozlem degisiyor (olcut bos degil)", later)
        from evaluation.trial_registry import load_returns
        ra, rb = load_returns(route_trial(state_a)), load_returns(route_trial(state_b))
        check("  kayittaki test getirileri t'ye kadar bit-es",
              np.array_equal(ra.iloc[:k].to_numpy(), rb.iloc[:k].to_numpy()))

        print("  mutasyon: rota olcegi gecirmiyor (B9 oncesi)")
        state_c, created_c = run_route(panel, strip_price_stats=True)
        state_d, created_d = run_route(perturb_after(panel, r_t), strip_price_stats=True)
        oc = env_for_window(created_c, r_tdates[0])["obs"]
        od = env_for_window(created_d, r_tdates[0])["obs"]
        check("mutant rota kabul olcutunu kiriyor (t=0 gozlemi degisiyor)",
              not np.array_equal(oc[0], od[0]))
        rc = route_trial(state_c)
        check("mutant rotanin kaydi kendiliginden 'own_window' + acik (etiket olculuyor)",
              rc is not None and rc["protocol"].get("price_stats_source") == "own_window"
              and "price_stats_source" in rc["protocol_open"])

    # -----------------------------------------------------------------------
    print("\nH — HPO objective (dogrulama ortami)")
    import optuna

    import data.data_fetcher as dfm
    from hyperparameter_optimization.base_optimizer import BaseHyperparameterOptimizer

    class TinyOptimizer(BaseHyperparameterOptimizer):
        def create_model(self, trial, env, hyperparams):
            from stable_baselines3 import A2C
            return A2C("MlpPolicy", env, n_steps=8, seed=5, device="cpu", verbose=0)

    raw_panel = synthetic_panel(indicators=False)
    rd = dates_of(raw_panel)
    tr_s, tr_e, va_s, va_e = (str(rd[0].date()), str(rd[109].date()),
                              str(rd[110].date()), str(rd[-1].date()))
    cwd = os.getcwd()
    hpo_dir = os.path.join(_TMP, "hpo_cwd")
    os.makedirs(hpo_dir, exist_ok=True)
    orig_load = dfm.DataFetcher.load_data
    dfm.DataFetcher.load_data = lambda self, filename="raw_stock_data.csv": raw_panel.copy()
    os.chdir(hpo_dir)  # Monitor goreli 'logs/hyperopt_*' dizinine yaziyor
    try:
        db = os.path.join(_TMP, "hpo.db").replace("\\", "/")
        opt = TinyOptimizer("a2c", study_name="b9_hpo", storage=f"sqlite:///{db}", n_trials=1,
                            seed=5, phase=1, reward_type="simple")
        made = []
        orig_create = opt.create_env

        def rec_create(*a, **kw):
            env = orig_create(*a, **kw)
            made.append(env)
            return env

        opt.create_env = rec_create
        trial = opt.create_study().ask()
        opt.objective(trial, ["S0.IS", "S1.IS", "S2.IS"], tr_s, tr_e, va_s, va_e,
                      total_timesteps=16, eval_freq=10 ** 9, n_eval_episodes=1)
    finally:
        os.chdir(cwd)
        dfm.DataFetcher.load_data = orig_load
    tr_env, va_env = made[0].unwrapped, made[1].unwrapped
    check("dogrulama ortami 'supplied' ve olcegi egitim ortaminin olcegi",
          va_env.price_stats_source == "supplied"
          and va_env.get_price_stats() == {s: tr_env.get_price_stats()[s] for s in va_env.symbols})
    check("  (ayirt edici: dogrulama penceresinin kendi olcegi DEGIL)",
          va_env.get_price_stats() != own_stats(va_env.df))
    from evaluation.trial_registry import load_trials
    hrec = [t_ for t_ in load_trials(directory=os.environ["RESULTS_DIR"], exclude_tags=())
            if t_["kind"] == "rl_hpo"]
    check("HPO deneme kaydi yazildi", len(hrec) == 1, str(len(hrec)))
    if hrec:
        hp = hrec[0]
        check("HPO kaydi price_stats_source='train_window', acik maddelerde degil",
              hp["protocol"].get("price_stats_source") == "train_window"
              and "price_stats_source" not in hp["protocol_open"], str(hp["protocol"]))
        check("HPO kaydi ham OHLCV yuzunden acik: indicators_present=False",
              hp["protocol"].get("indicators_present") is False
              and "indicators_present" in hp["protocol_open"])

    # -----------------------------------------------------------------------
    print("\nS — sarmalayiciyla env kuran betikler (davranis)")
    from stable_baselines3 import A2C

    import scripts.optimization.ab_test_rewards as ab
    import scripts.optimization.optimize_reward_weights as orw
    import scripts.training.train_a2c_phase1 as a2cs

    def spy_make_env(module):
        made_envs, orig_fn = [], module.make_env

        def spy(df, **kw):
            env = orig_fn(df, **kw)
            made_envs.append(env)
            return env

        module.make_env = spy
        return made_envs, (lambda: setattr(module, "make_env", orig_fn))

    def supplied_matches_train(envs):
        sup = [e for e in envs if e.price_stats_source == "supplied"]
        own = [e for e in envs if e.price_stats_source == "own_window"]
        return (len(sup) == 1 and bool(own)
                and sup[0].get_price_stats() == {s: own[0].get_price_stats()[s] for s in sup[0].symbols}
                and dates_of(sup[0].df)[0] > dates_of(own[0].df)[-1])

    made_a2c, undo = spy_make_env(a2cs)
    try:
        venv = a2cs.create_env(train_df)
        model = A2C("MlpPolicy", venv, n_steps=8, seed=1, device="cpu", verbose=0)
        model.learn(16)
        a2cs.evaluate_model(model, test_df, n_episodes=1)
    finally:
        undo()
    check("train_a2c_phase1.evaluate_model: test ortami modelin egitim olceginde",
          supplied_matches_train(made_a2c))

    made_ab, undo = spy_make_env(ab)
    try:
        ab.train_and_evaluate(train_df, test_df, None, None, phase=1, reward_type="simple", timesteps=64)
    finally:
        undo()
    check("ab_test_rewards.train_and_evaluate: test ortami egitim olceginde",
          supplied_matches_train(made_ab))

    # Faz 2 + PSR: temel ve makro tablo zorunlu (yoksa faz 2 env'in boyutu tutmuyor —
    # B9'dan bagimsiz, Faz 9 §9.5 gozlemler)
    syms = list(panel.index.get_level_values("symbol").unique())
    fund = pd.DataFrame({c: [1.0] * len(syms) for c in (
        "roe", "roa", "debt_to_equity", "current_ratio", "pe_ratio", "pb_ratio", "profit_margin")},
        index=syms)
    macro = pd.DataFrame({"policy_rate": 30.0, "cpi_inflation": 40.0, "ppi_inflation": 40.0,
                          "usd_try": 30.0, "eur_try": 33.0, "bist100_index": 9000.0}, index=d)
    made_orw, undo = spy_make_env(orw)
    try:
        trial = optuna.trial.FixedTrial({"w1": 0.5, "w2": 0.3, "w3": 0.1, "w4": 0.05, "w5": 0.05,
                                         "rolling_window": 30, "target_trades_per_100": 50})
        score = orw.objective(trial, train_df, test_df, fund, macro)
    finally:
        undo()
    check("optimize_reward_weights.objective: dogrulama ortami egitim olceginde",
          score != -999999 and supplied_matches_train(made_orw), f"score={score}")

    import scripts.analysis.generate_academic_report as gar
    from app.services.daily_trading import write_model_meta
    rdir = os.path.join(_TMP, "report_models")
    os.makedirs(rdir, exist_ok=True)
    new_model, old_model = os.path.join(rdir, "a2c_new"), os.path.join(rdir, "a2c_old")
    model.save(new_model)
    write_model_meta(new_model, {"price_stats": train_stats})
    check("rapor: meta'daki egitim olcegi okunuyor", gar.resolve_price_stats(new_model) == train_stats)
    check("rapor: meta'sinda olcek olmayan eski model icin olcek uydurulmuyor (None -> atlanir)",
          gar.resolve_price_stats(old_model) is None)
    made_rep = []
    orig_te = gar.TradingEnv

    def spy_te(*a, **kw):
        env = orig_te(*a, **kw)
        made_rep.append(env)
        return env

    gar.TradingEnv = spy_te
    try:
        gar.evaluate_model_on_test_set(new_model, test_df, initial_balance=100_000,
                                       price_stats=gar.resolve_price_stats(new_model))
    except KeyError:
        pass  # SELL'de trade['total'] — onceden var olan hata (Faz 9 §9.5, D5)
    finally:
        gar.TradingEnv = orig_te
    check("rapor: test ortami modelin meta olceginde",
          len(made_rep) == 1 and made_rep[0].price_stats_source == "supplied"
          and made_rep[0].get_price_stats() == train_stats)

    # -----------------------------------------------------------------------
    print("\nG — held-out cagiran bekcisi (AST)")
    calls = scan_repo()
    bad_calls = [c for c in calls if not c[3]]
    check("held-out df ile kurulan her env price_stats tasiyor", not bad_calls,
          "; ".join(f"{f}:{ln} ({n})" for f, ln, n, _ in bad_calls))
    check(f"bekci bos degil: {len(calls)} held-out cagrisi bulundu (>= 8)", len(calls) >= 8,
          "; ".join(f"{f}:{ln}" for f, ln, _, _ in calls))
    sample = (
        "make_env(df=test_df, initial_balance=1)\n"
        "TradingEnv(val_df)\n"
        "make_env(data['test'])\n"
        "make_env(test_df.copy(), price_stats=s)\n"
        "make_env(df=latest_df)\n"
        "TradingEnv(eval_df)  # B9-legacy\n"
    )
    flags = [(ln, ok) for _, ln, _, ok in held_out_env_calls(sample)]
    check("mutasyon: bekci olceksiz held-out cagrilari yakaliyor",
          sorted(flags) == [(1, False), (2, False), (3, False), (4, True), (6, True)],
          str(sorted(flags)))

    print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
