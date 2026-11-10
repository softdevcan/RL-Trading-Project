"""
Faz 9 / B9 (Faz 10 Y13) — val/test ortami EGITIM penceresinin fiyat olcegiyle.

Kabul olcutu (Faz 10 §3.1.1 B9): "Test penceresinde t sonrasi fiyatlar
degistirildiginde t gozlemi bit-es kalir." Olcut HER t icin gecerlidir; burada
t in {0, orta, T-2} sinanir (hakem-2 B9 #4: tek t, pencerenin basindan kurulan
olcek gibi sizintilari kaciriyordu — en guclusu t=0).

  D  varsayilan : price_stats=None yolu main @ 78edf06 ile bit-es. Birincil
                  kanit `git show` ile yuklenen main env'iyle 11 yapilandirmada
                  ayni kutuphanelerle karsilastirma; git yoksa dondurulmus ozetler.
  E  env        : t'ye kadar (dahil) TUM gozlem vektoru bit-es. Eski davranis
                  (kendi penceresi) ayni olcutten KALIR: sizinti sayisal.
  M  mutasyon   : olcegi yok sayan, kopyalamayan ve olcegi degerlendirme
                  penceresinin ilk 16 gununden kuran env olcutu KIRMALI.
  V  denetim    : eksik/bozuk olcek ValueError; uretici env yedege duseni isaretler.
  P  protokol   : etiket olcegin KOKENINI olcer (hakem-2 B9 #3): egitim env'i
                  kendi penceresinden mi, o pencere degerlendirmeden once mi
                  bitiyor, degerler birebir ayni mi; desteği olcer (#1).
  O  saglik     : piyasa bloklari politikadan bagimsiz; destek disi oran.
  R  rota       : _run_training_inner uctan uca (sentetik panel, CPU, sabit
                  tohum). Kabul PIYASA BLOGUNDA (politikadan bagimsiz, GPU
                  belirlenimsizliginden etkilenmez; hakem-2 B9 #5) ve tum
                  gozlemde; ilk sapmanin adimi ve blogu raporlanir. Kayit,
                  manifest, meta ve metrik dosyasi; hiperparametre kokeni (#8).
  H  HPO        : objective() — dogrulama ortami egitim olcegini alir.
  S  betikler   : sarmalayiciyla env kuran betikler davranisla; yardimcilarin
                  price_stats'i ZORUNLU anahtar (hakem-2 B9 #6).
  G  bekci      : held-out adli df ile kurulan env cagrisi (make_env, TradingEnv
                  ve adinda 'env' gecen sarmalayicilar) price_stats tasimali ve
                  bu deger None sabiti olmamali (AST taramasi).

Calistirma:
    python tests/test_price_stats_window.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
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

MAIN_REF = "78edf06"  # B9 oncesi main


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


def in_support_panel(n_days=160, copy_from=20, train_days=112):
    """Degerlendirme gunleri egitim gunlerinin KOPYASI: fiyat egitim desteginde."""
    base = synthetic_panel(n_days=n_days, drift=0.0)
    dates = dates_of(base)
    frames = []
    for s, sub in base.groupby(level="symbol"):
        sub = sub.droplevel("symbol")
        tail = sub.iloc[copy_from: copy_from + (n_days - train_days)].copy()
        tail.index = dates[train_days:]
        both = pd.concat([sub.iloc[:train_days], tail])
        both.index.name = "date"
        both = both.reset_index()
        both.insert(0, "symbol", s)
        frames.append(both)
    return pd.concat(frames).set_index(["symbol", "date"]).sort_index()


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


def fund_macro(panel):
    syms = list(panel.index.get_level_values("symbol").unique())
    fund = pd.DataFrame({c: [1.0] * len(syms) for c in (
        "roe", "roa", "debt_to_equity", "current_ratio", "pe_ratio", "pb_ratio", "profit_margin")},
        index=syms)
    macro = pd.DataFrame({"policy_rate": 30.0, "cpi_inflation": 40.0, "ppi_inflation": 40.0,
                          "usd_try": 30.0, "eur_try": 33.0, "bist100_index": 9000.0},
                         index=dates_of(panel))
    return fund, macro


def trajectory_digest(env_cls, panel, mode="legacy", n_steps=None, no_cache=False, **env_kw):
    """Varsayilan yolun parmak izi: gozlem + odul + portfoy degeri (D)."""
    kw = dict(initial_balance=100_000, missing_data_mode=mode)
    kw.update(env_kw)
    env = env_cls(panel, **kw)
    if no_cache:
        env._lookup_cache = None  # .loc yolu
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


def load_main_env():
    """main @ MAIN_REF'teki TradingEnv (git yoksa None)."""
    try:
        src = subprocess.run(["git", "show", f"{MAIN_REF}:env/trading_env.py"], cwd=ROOT,
                             capture_output=True, timeout=30)
        if src.returncode != 0 or not src.stdout:
            return None
        path = os.path.join(_TMP, f"trading_env_{MAIN_REF}.py")
        with open(path, "wb") as fh:
            fh.write(src.stdout)
        spec = importlib.util.spec_from_file_location(f"trading_env_{MAIN_REF}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TradingEnv
    except Exception:
        return None


# main @ 78edf06'daki env/trading_env.py ile uretildi (B9 oncesi). Git
# varsa birincil kanit main env'iyle dogrudan karsilastirmadir; bu ozetler
# git'siz ortam icin yedektir ve numpy/pandas surumune baglidir.
FROZEN_DIGESTS = {
    ("full", "legacy"): "333a13a18e96b7fb07015e470c9a069d1b3f60f2e712984916c6b58e963eaf7f",
    ("full", "own_last"): "333a13a18e96b7fb07015e470c9a069d1b3f60f2e712984916c6b58e963eaf7f",
    ("holed", "legacy"): "55e94c65a87f188c9785f8c823ffc89fe3766d44afe7ebfd1dfeca26fb83714a",
    ("holed", "own_last"): "d738a67bc5c06d45d8717b25ce8c67aa17eb2f9660e72af2cde3fddd5aa00ff7",
}


# ---------------------------------------------------------------------------
# AST bekcisi
# ---------------------------------------------------------------------------

_HELD_OUT = re.compile(r"(^|_)(test|val|valid|validation|eval|holdout|oos)(_|$)", re.IGNORECASE)
_ENV_CALLEE = re.compile(r"env", re.IGNORECASE)
_LEGACY_MARK = "B9-legacy"
_SKIP_DIRS = {"venv", ".git", "docs", "workspaces", "node_modules", "__pycache__",
              "results", "models", "logs", ".claude"}


def _held_out_name(node):
    """df argumaninin held-out adini dondur: test_df, test_df.copy(), splits.test,
    data['test'], df.loc[test_mask]. Bulamazsa None."""
    while True:
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute):
            if _HELD_OUT.search(node.attr) and node.attr not in ("copy",):
                return node.attr
            node = node.value
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and _HELD_OUT.search(sl.value):
                return f"[{sl.value!r}]"
            if isinstance(sl, ast.Name) and _HELD_OUT.search(sl.id):
                return f"[{sl.id}]"
            node = node.value
        else:
            break
    if isinstance(node, ast.Name) and _HELD_OUT.search(node.id):
        return node.id
    return None


def held_out_env_calls(source, filename="<src>"):
    """(dosya, satir, df adi, uyumlu_mu). Uyumlu: price_stats anahtari VAR ve
    degeri None sabiti DEGIL — ya da satir `# B9-legacy` ile isaretli."""
    tree = ast.parse(source)
    lines = source.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if not name or not (name in ("make_env", "TradingEnv") or _ENV_CALLEE.search(name)):
            continue
        df_arg = next((k.value for k in node.keywords if k.arg == "df"),
                      node.args[0] if node.args else None)
        held = _held_out_name(df_arg) if df_arg is not None else None
        if not held:
            continue
        ps = next((k.value for k in node.keywords if k.arg == "price_stats"), None)
        ok_kw = ps is not None and not (isinstance(ps, ast.Constant) and ps.value is None)
        ok = ok_kw or _LEGACY_MARK in lines[node.lineno - 1]
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

def run_route(panel, strip_price_stats=False, seed=11, **req_kw):
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
    # R1: rota pencereyi bolme tanimindan alir. Sentetik panel icin eski oranli
    # bolmeyle (split_data) AYNI sinirlari veren gecici, kanonik OLMAYAN tanim.
    from evaluation import rl_folds, splits
    if 'split_spec' not in req_kw:
        gated = rl_folds.load_rl_panel(price_norm=req_kw.get('price_norm', 'stats'),
                                       df=panel.copy(), adx_mode=tr.RL_ADX_MODE).panel
        req_kw['split_spec'] = splits.register_ratio_spec(
            gated.index.get_level_values('date')).name
    try:
        state = tr._empty_training_state()
        req = TrainingRequest(algorithm="A2C", phase=1, total_timesteps=256,
                              initial_balance=100_000, seed=seed, **req_kw)
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


def market_blocks(obs_list, n, fps):
    return [o[1 + n: 1 + n + n * fps] for o in obs_list]


def first_divergence(a, b, n, fps, upto):
    """Ilk farkli adim ve blok ('balance' / 'positions' / 'market'); yoksa None."""
    for i in range(min(upto + 1, len(a), len(b))):
        if not np.array_equal(a[i], b[i]):
            if a[i][0] != b[i][0]:
                return i, "balance"
            if not np.array_equal(a[i][1:1 + n], b[i][1:1 + n]):
                return i, "positions"
            return i, "market"
    return None


def main():
    global passed, failed
    from app.core.config import get_settings
    get_settings.cache_clear()

    import evaluation.reproducibility as rep
    rep.SNAPSHOT_DIR = os.path.join(_TMP, "snapshots")

    from env.trading_env import TradingEnv
    from evaluation.observation_health import (
        SUPPORT_EPS, block_stats, market_matrix, observation_health, price_in_support,
    )
    from evaluation.trial_registry import (
        CLEAN_PROTOCOL, indicators_present, open_protocol_items, price_stats_source,
        register_trial, rl_eval_protocol,
    )

    panel = synthetic_panel()
    d = dates_of(panel)
    train_df = panel[panel.index.get_level_values("date") <= d[109]]
    test_df = panel[panel.index.get_level_values("date") > d[109]]
    tdates = dates_of(test_df)
    train_env = TradingEnv(train_df, initial_balance=100_000, missing_data_mode="own_last")
    train_stats = train_env.get_price_stats()
    actions = np.random.default_rng(1).uniform(-1, 1, size=(len(tdates), 3)).astype(np.float32)
    t_values = (0, 15, len(tdates) - 2)

    # -----------------------------------------------------------------------
    print("\nD — varsayilan yol (price_stats=None) degismedi")
    e0 = TradingEnv(test_df, initial_balance=100_000)  # B9-legacy
    check("kaynak etiketi 'own_window'", e0.price_stats_source == "own_window")
    check("olcek bagimsiz hesapla bit-es (kendi penceresi)", e0.get_price_stats() == own_stats(test_df))
    fund, macro = fund_macro(panel)
    configs = {
        ("full", "legacy"): (panel, dict(mode="legacy")),
        ("full", "own_last"): (panel, dict(mode="own_last")),
        ("holed", "legacy"): (holed(panel), dict(mode="legacy")),
        ("holed", "own_last"): (holed(panel), dict(mode="own_last")),
        ("faz2_psr", "legacy"): (panel, dict(phase=2, reward_type="psr", fundamental_df=fund,
                                             macro_df=macro)),
        ("faz3", "legacy"): (panel, dict(phase=3)),
        ("atr", "legacy"): (panel, dict(use_atr_sizing=True)),
        ("atr_kelly", "legacy"): (panel, dict(use_atr_sizing=True, use_kelly=True)),
        ("loc_yolu", "own_last"): (holed(panel), dict(mode="own_last", no_cache=True)),
        ("ham_ohlcv", "legacy"): (panel[["open", "high", "low", "close", "volume"]], dict()),
        ("komisyon_lot", "legacy"): (panel, dict(commission_rate=0.003, max_shares_per_trade=37)),
    }
    MainEnv = load_main_env()
    if MainEnv is not None:
        print(f"    main env'i git'ten yuklendi ({MAIN_REF}); ayni kutuphanelerle karsilastiriliyor")
        for key, (p, kw) in configs.items():
            dig = trajectory_digest(TradingEnv, p, **kw)
            ref = trajectory_digest(MainEnv, p, **kw)
            check(f"main ile bit-es: {key[0]} / {key[1]}", dig == ref)
            if key in FROZEN_DIGESTS and dig != FROZEN_DIGESTS[key]:
                print(f"    not: {key} dondurulmus ozetten farkli (kutuphane surumu degismis "
                      f"olabilir; birincil kanit main karsilastirmasi)")
    else:
        print("    git yok: dondurulmus ozetler kullaniliyor")
        for key, want in FROZEN_DIGESTS.items():
            p, kw = configs[key]
            check(f"dondurulmus ozetle bit-es: {key}", trajectory_digest(TradingEnv, p, **kw) == want)

    # -----------------------------------------------------------------------
    print("\nE — env: verilen olcek, kabul olcutu (t in {0, orta, T-2}), sizinti")
    e1 = TradingEnv(test_df, initial_balance=100_000, price_stats=train_stats)
    check("kaynak etiketi 'supplied'", e1.price_stats_source == "supplied")
    check("olcek egitim penceresinin olcegi (aynen)", e1.get_price_stats() == own_stats(train_df))
    o = e1.reset()[0]
    blk = o[1 + 3:].reshape(3, 10)
    first = test_df.xs(tdates[0], level="date")
    exp = [(first.loc[s, "close"] - train_stats[s]["mean"]) / train_stats[s]["std"] for s in e1.symbols]
    check("gozlemdeki kapanis z'si (p - egitim ort.) / egitim std",
          np.allclose(blk[:, 3], np.clip(np.array(exp, dtype=np.float32), -10, 10), rtol=0, atol=1e-5))

    def acceptance(make, t_idx):
        """t'ye kadar (dahil) gozlem bit-es mi + t sonrasi gercekten degisiyor mu."""
        t = tdates[t_idx]
        a = obs_trajectory(make(test_df), actions)
        b = obs_trajectory(make(perturb_after(test_df, t)), actions)
        prefix = all(np.array_equal(a[i], b[i]) for i in range(t_idx + 1))
        later = any(not np.array_equal(a[i], b[i]) for i in range(t_idx + 1, len(a)))
        diff0 = float(np.max(np.abs(a[0] - b[0])))
        return prefix, later, diff0

    def accept_all(make):
        return [acceptance(make, t)[0] for t in t_values]

    for t_idx in t_values:
        pre, later, _ = acceptance(lambda f: TradingEnv(f, initial_balance=100_000,
                                                        price_stats=train_stats), t_idx)
        check(f"KABUL t={t_idx}: t sonrasi fiyat degisince t'ye kadar gozlem bit-es", pre)
        check(f"  t={t_idx}: t sonrasi gozlem degisiyor (olcut bos degil)", later)
    pre_l, _, diff0 = acceptance(lambda f: TradingEnv(f, initial_balance=100_000), 0)  # B9-legacy
    check("SIZINTI: eski davranista t=0 gozlemi bile degisiyor (olcut kaliyor)",
          (not pre_l) and diff0 > 0, f"max|fark| t=0: {diff0:.4f}")
    print(f"    eski davranis t=0 gozleminde max|fark| = {diff0:.4f} (gelecek fiyatlardan)")
    check("KABUL own_last modunda da (tum t)",
          all(accept_all(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats,
                                              missing_data_mode="own_last"))))

    print("\nM — mutasyon oz-testleri (olcut KIRILMALI)")
    orig_validate = TradingEnv._validated_price_stats
    try:
        def ignore_supplied(self, supplied):  # parametreyi kabul et, KULLANMA
            self._compute_price_stats()
            return self.price_stats
        TradingEnv._validated_price_stats = ignore_supplied
        res = accept_all(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats))
        check("mutant 'olcegi yok say' kabulu her t'de kiriyor", not any(res), str(res))

        def head16(self, supplied):  # hakem-2 B9 K5: olcek pencerenin ilk 16 gununden
            first_days = self.dates[:16]
            sub = self.df[self.df.index.get_level_values("date").isin(first_days)]
            return own_stats(sub)
        TradingEnv._validated_price_stats = head16
        res = accept_all(lambda f: TradingEnv(f, initial_balance=100_000, price_stats=train_stats))
        check("mutant 'ilk 16 gunden olcek' t=0'da yakalaniyor (t=15 tek basina kacirirdi)",
              res[0] is False and res[1] is True, str(res))
        em = TradingEnv(test_df, initial_balance=100_000, price_stats=train_stats)
        check("  ayni mutant etikette de acik ('supplied_other')",
              price_stats_source(em, train_env) == "supplied_other")

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

    print("\nV — denetim (sessiz yedek yok)")

    def raises(fn, exc=ValueError):
        try:
            fn()
            return False
        except exc:
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
    flat = train_df.copy()
    flat.loc[flat.index.get_level_values("symbol") == "S2.IS", ["open", "high", "low", "close"]] = 50.0
    ef = TradingEnv(flat, initial_balance=100_000)
    check("uretici env yedege duseni isaretliyor (sabit fiyat -> std tabani)",
          ef.price_stats_fallbacks == {"S2.IS": "std_floor_1.0"}
          and ef.get_price_stats()["S2.IS"] == {"mean": 50.0, "std": 1.0})
    check("  normal sembolde isaret yok", train_env.price_stats_fallbacks == {})

    # -----------------------------------------------------------------------
    print("\nP — protokol: olcegin KOKENI ve destek olculur")
    from stable_baselines3.common.monitor import Monitor
    check("kendi penceresi -> 'own_window'", price_stats_source(e0, train_env) == "own_window")
    check("egitim env'inden -> 'train_window'", price_stats_source(e1, train_env) == "train_window")
    check("Monitor sarmalayicisi aciliyor",
          price_stats_source(Monitor(TradingEnv(test_df, price_stats=train_stats)),
                             Monitor(train_env)) == "train_window")
    eK7 = TradingEnv(test_df, price_stats=own_stats(test_df))
    check("hakem-2 K7: test penceresinin kendi olcegi 'egitim olcegi' diye verilince temiz CIKMAZ",
          price_stats_source(eK7, train_env) == "supplied_other")
    over_env = TradingEnv(panel[panel.index.get_level_values("date") <= d[130]], initial_balance=100_000)
    e_over = TradingEnv(test_df, price_stats=over_env.get_price_stats())
    check("egitim penceresi degerlendirmeyle kesisiyor -> 'overlapping_window'",
          price_stats_source(e_over, over_env) == "overlapping_window")
    sup_train = TradingEnv(train_df, price_stats=train_stats)
    check("egitim env'i olcegini kendisi hesaplamamis -> 'supplied_unverified'",
          price_stats_source(e1, sup_train) == "supplied_unverified")
    check("egitim env'i yok -> 'supplied_unverified'", price_stats_source(e1, None) == "supplied_unverified")
    check("degerlendirme env'i yok -> None", price_stats_source(None, train_env) is None)
    flat_test = test_df.copy()
    e_fb = TradingEnv(flat_test, price_stats=ef.get_price_stats())
    check("egitim olcegi yedege dusmus -> 'train_window_fallback'",
          price_stats_source(e_fb, ef) == "train_window_fallback")
    raw = panel[["open", "high", "low", "close", "volume"]]
    check("gostergeli panel -> indicators_present True", indicators_present(panel))
    check("ham OHLCV -> indicators_present False", not indicators_present(raw))

    sp = in_support_panel()
    sd_ = dates_of(sp)
    sp_train = sp[sp.index.get_level_values("date") < sd_[112]]
    sp_eval = sp[sp.index.get_level_values("date") >= sd_[112]]
    sp_train_env = TradingEnv(sp_train, initial_balance=100_000, missing_data_mode="own_last")
    sp_eval_env = TradingEnv(sp_eval, price_stats=sp_train_env.get_price_stats())
    pr_clean = rl_eval_protocol(sp_eval_env, sp_train_env, indicator_bfill=False,
                                split_from_spec=True, selection_metric='val_excess_sharpe')
    check("destekte + egitim olcegi + bfill yok: protokol tamamen temiz",
          open_protocol_items(pr_clean) == [], str(open_protocol_items(pr_clean)))
    pr = rl_eval_protocol(e1, train_env)
    check("yukselen panel: fiyat destek disi -> 'price_in_support' acik (olculuyor)",
          pr["price_in_support"] is False and "price_in_support" in open_protocol_items(pr),
          str(open_protocol_items(pr)))
    pr_raw = rl_eval_protocol(TradingEnv(raw.loc[test_df.index], price_stats=train_stats),
                              TradingEnv(raw.loc[train_df.index]))
    check("ham OHLCV: adx sabit, bfill anlamsiz (None), gosterge yok",
          pr_raw["adx_constant"] is True and pr_raw["indicator_bfill"] is None
          and pr_raw["indicators_present"] is False)
    check("tum maddeler temizse acik madde yok", open_protocol_items(dict(CLEAN_PROTOCOL)) == [])

    # -----------------------------------------------------------------------
    print("\nO — gozlem sagligi (piyasa bloklari politikadan bagimsiz)")
    ep = TradingEnv(test_df, initial_balance=100_000, price_stats=train_stats, missing_data_mode="own_last")
    traj = obs_trajectory(ep, actions)
    mm = market_matrix(ep)
    check("market_matrix = bolum boyunca gozlemin piyasa blogu (rastgele eylemle)",
          len(traj) == len(mm) and all(np.array_equal(np.asarray(market_blocks([o_], 3, 10)[0]).reshape(3, 10), m)
                                       for o_, m in zip(traj, mm)))
    calls = []
    inner = ep._get_observation
    ep._get_observation = lambda: (calls.append(1), inner())[1]
    market_matrix(ep)
    check("ornek sarmalayicisi tetiklenmiyor (sinif metodu)", calls == [])
    h_clean = observation_health(sp_train_env, sp_eval_env)
    h_up = observation_health(train_env, e1)
    check("kopya degerlendirme gunleri: fiyat destek disi oran 0",
          h_clean["eval_out_of_train_support"]["price"] == 0.0 and price_in_support(h_clean) is True)
    check(f"yukselen panel: destek disi oran > esik ({SUPPORT_EPS})",
          h_up["eval_out_of_train_support"]["price"] > SUPPORT_EPS and price_in_support(h_up) is False)
    br = block_stats(market_matrix(TradingEnv(raw.loc[test_df.index], price_stats=train_stats)))
    check("ham OHLCV: gosterge boyutlarinin tamami sabit", br["indicators"]["constant_dims_frac"] == 1.0)
    check("ayni sembol evreni yoksa destek olculmez (None, uydurulmaz)",
          observation_health(TradingEnv(train_df.loc[["S0.IS"]], initial_balance=100_000),
                             e1)["eval_out_of_train_support"] is None)

    # -----------------------------------------------------------------------
    print("\nR — rota uctan uca (_run_training_inner)")
    from data.data_fetcher import DataFetcher
    from data.integrity import prepare_rl_panel
    from evaluation.trial_registry import load_returns

    from data.technical_indicators import with_adx
    # rotanin veri yolu: K18 ADX yeniden hesabi -> veri kapisi (B7 isinma atimi dahil)
    rl_panel, _ = prepare_rl_panel(with_adx(panel), start=get_settings().RL_DATA_START,
                                   drop_indicator_warmup=True)
    r_train, _, r_test = DataFetcher().split_data(rl_panel)
    r_tdates = dates_of(r_test)

    state_a, created_a = run_route(panel)
    check("rota kosumu tamamlandi", state_a.get("state") == "completed", str(state_a.get("error")))
    test_a = env_for_window(created_a, r_tdates[0])
    check("rotanin test ortami bulundu", test_a is not None)
    if state_a.get("state") == "completed" and test_a is not None:
        n_s, fps = test_a["env"].n_stocks, test_a["env"].features_per_stock
        exp_train = own_stats(r_train)
        check("test ortami 'supplied' ve olcegi EGITIM bolumunun olcegi",
              test_a["env"].price_stats_source == "supplied"
              and test_a["env"].get_price_stats() == exp_train)
        check("  (ayirt edici: test penceresinin kendi olcegiyle ayni DEGIL)",
              test_a["env"].get_price_stats() != own_stats(r_test))
        rec = route_trial(state_a)
        check("deneme kaydi price_stats_source='train_window' (koken olculdu)",
              rec is not None and rec["protocol"].get("price_stats_source") == "train_window",
              str(rec and rec["protocol"]))
        oh = rec["extra"]["observation_health"]
        check("price_in_support kayittaki saglik olcumuyle tutarli",
              rec["protocol"]["price_in_support"] == (oh["eval_out_of_train_support"]["price"] <= SUPPORT_EPS))
        check("  yukselen panelde acik: price_in_support (B7/K18 kapansa da kayit temiz sayilmaz)",
              "price_in_support" in rec["protocol_open"] and "protocol_open" in rec["tags"])
        check("hiperparametre kokeni: varsayilan, temiz",
              rec["protocol"].get("hyperparams_source") == "default"
              and rec["protocol"].get("hyperparams_clean") is True)
        man = json.load(open(os.path.join(os.environ["RESULTS_DIR"], "training_runs",
                                          f"{state_a['metrics']['run_id']}.json"), encoding="utf-8"))
        check("manifest protokolu ve sagligi ayni",
              man.get("extra", {}).get("protocol") == rec["protocol"]
              and man.get("extra", {}).get("observation_health") == oh)
        mpath = os.path.join(os.environ["RESULTS_DIR"], f"{rec['extra']['model_name']}_metrics.json")
        mj = json.load(open(mpath, encoding="utf-8"))
        check("metrik dosyasi protokol + acik maddeler + saglik tasiyor (hakem-2 #9)",
              mj.get("protocol") == rec["protocol"] and mj.get("protocol_open") == rec["protocol_open"]
              and mj.get("observation_health") == oh)
        meta = json.load(open(os.path.join(os.environ["MODELS_DIR"],
                                           f"{rec['extra']['model_name']}.meta.json"), encoding="utf-8"))
        check("meta: price_stats = egitim olcegi; yedek isareti ve hiperparametre kokeni",
              meta["price_stats"] == exp_train == test_a["env"].get_price_stats()
              and meta.get("price_stats_fallbacks") == {} and meta.get("hyperparams_source") == "default")
        seen = np.asarray(test_a["obs"][:-1], dtype=np.float32)
        check("pozisyon kirpmasi modelin gordugu gozlemden (bagimsiz hesapla ayni)",
              oh["eval_seen"]["position_clip_frac"] == float(np.mean(np.abs(seen[:, 1:1 + n_s]) >= 10.0)))
        check("destek disi uyarisi panoda", any("destegi disinda" in w for w in state_a.get("warnings", [])))

        for k in (0, 6):
            state_b, created_b = run_route(perturb_after(panel, r_tdates[k]))
            ob = env_for_window(created_b, r_tdates[0])["obs"]
            oa = test_a["obs"]
            mk_pre = all(np.array_equal(x, y) for x, y in
                         zip(market_blocks(oa, n_s, fps)[:k + 1], market_blocks(ob, n_s, fps)[:k + 1]))
            full_pre = len(oa) > k + 1 and all(np.array_equal(oa[i], ob[i]) for i in range(k + 1))
            later = any(not np.array_equal(oa[i], ob[i]) for i in range(k + 1, min(len(oa), len(ob))))
            div = first_divergence(oa, ob, n_s, fps, k)
            check(f"KABUL (rota) t={k}: piyasa blogu t'ye kadar bit-es (politikadan bagimsiz)", mk_pre)
            check(f"  t={k}: tum gozlem t'ye kadar bit-es (CPU)", full_pre, f"ilk sapma: {div}")
            check(f"  t={k}: t sonrasi gozlem degisiyor (olcut bos degil)", later)
            if k == 6:
                ra, rb = load_returns(route_trial(state_a)), load_returns(route_trial(state_b))
                check("  kayittaki test getirileri t'ye kadar bit-es",
                      np.array_equal(ra.iloc[:k].to_numpy(), rb.iloc[:k].to_numpy()))

        print("  mutasyon: rota olcegi gecirmiyor (B9 oncesi)")
        state_c, created_c = run_route(panel, strip_price_stats=True)
        state_d, created_d = run_route(perturb_after(panel, r_tdates[0]), strip_price_stats=True)
        oc = env_for_window(created_c, r_tdates[0])["obs"]
        od = env_for_window(created_d, r_tdates[0])["obs"]
        check("mutant rota kabul olcutunu kiriyor (t=0 piyasa blogu degisiyor)",
              not np.array_equal(market_blocks(oc, n_s, fps)[0], market_blocks(od, n_s, fps)[0]))
        rc = route_trial(state_c)
        check("mutant rotanin kaydi kendiliginden 'own_window' + acik (etiket olculuyor)",
              rc is not None and rc["protocol"].get("price_stats_source") == "own_window"
              and "price_stats_source" in rc["protocol_open"])

        state_s, _ = run_route(in_support_panel())
        rs = route_trial(state_s)
        check("destekte panel: price_in_support True, uyari yok",
              rs["protocol"]["price_in_support"] is True
              and not any("destegi disinda" in w for w in state_s.get("warnings", [])))

        hot = synthetic_panel(drift=0.0, drift_after=(100, 0.03))
        state_h, created_h = run_route(hot)
        oh_h = route_trial(state_h)["extra"]["observation_health"]
        check("guclu yukselis: test fiyatlari kirpiliyor ve destek disi",
              oh_h["eval"]["price"]["clip_frac"] > 0.5
              and oh_h["eval_out_of_train_support"]["price"] > 0.5, str(oh_h["eval"]["price"]))

        print("  hiperparametre kokeni (hakem-2 B9 #8)")
        hdir = os.environ["HYPEROPT_DIR"]
        os.makedirs(hdir, exist_ok=True)
        for study, registered in (("b9_study_acik", True), ("b9_study_kayitsiz", False)):
            with open(os.path.join(hdir, f"{study}.json"), "w", encoding="utf-8") as fh:
                json.dump({"study_name": study, "best_params": {"learning_rate": 0.0007,
                                                                 "n_steps": 256}}, fh)
            if registered:
                register_trial("rl_hpo", {"algorithm": "a2c"}, 1, split="val",
                               directory=os.environ["RESULTS_DIR"], extra={"study": study},
                               protocol={"price_stats_source": "train_window",
                                         "indicators_present": False})
        st_k, _ = run_route(panel, hyperparameter_study="b9_study_acik.json")
        pk = route_trial(st_k)["protocol"]
        check("acik HPO calismasindan gelen hiperparametre: kayit acik, kaynak ve maddeler yazili",
              pk["hyperparams_clean"] is False and pk["hyperparams_source"] == "hpo:b9_study_acik"
              and "indicators_present" in pk["hyperparams_open_items"]
              and "hyperparams_clean" in open_protocol_items(pk))
        st_u, _ = run_route(panel, hyperparameter_study="b9_study_kayitsiz.json")
        pu = route_trial(st_u)["protocol"]
        check("kaydi bulunamayan HPO calismasi: hyperparams_clean None (acik), nedeni yazili",
              pu["hyperparams_clean"] is None and pu["hyperparams_open_items"] == ["hpo_trials_not_in_registry"])

    # -----------------------------------------------------------------------
    print("\nH — HPO objective (dogrulama ortami; R1: rotayla ortak panel ve kat)")
    from evaluation import rl_folds, splits
    from hyperparameter_optimization.base_optimizer import BaseHyperparameterOptimizer

    class TinyOptimizer(BaseHyperparameterOptimizer):
        def create_model(self, trial, env, hyperparams):
            from stable_baselines3 import A2C
            return A2C("MlpPolicy", env, n_steps=8, seed=self.seed, device="cpu", verbose=0)

    # R1 (Y21): HPO artik ham OHLCV'yi degil rotanin GOSTERGELI panelini okuyor
    hpo_panel = synthetic_panel()
    gated_h = rl_folds.load_rl_panel(df=hpo_panel.copy()).panel
    hspec = splits.register_ratio_spec(gated_h.index.get_level_values("date"))
    cwd = os.getcwd()
    hpo_dir = os.path.join(_TMP, "hpo_cwd")
    os.makedirs(hpo_dir, exist_ok=True)
    os.chdir(hpo_dir)  # Monitor goreli 'logs/hyperopt_*' dizinine yaziyor
    made = []
    try:
        db = os.path.join(_TMP, "hpo.db").replace("\\", "/")
        opt = TinyOptimizer("a2c", study_name="b9_hpo", storage=f"sqlite:///{db}", n_trials=1,
                            seed=5, phase=1, reward_type="simple")
        orig_make = opt.make_env

        def rec_make(fold_data, role, **kw):
            env = orig_make(fold_data, role, **kw)
            made.append((role, env, fold_data))
            return env

        opt.make_env = rec_make
        opt.optimize(fold="ratio", total_timesteps=64, split_spec=hspec.name,
                     show_progress_bar=False, panel_df=hpo_panel.copy())
    finally:
        os.chdir(cwd)
    tr_env = next(e for r, e, _ in made if r == "train").unwrapped
    va_env = next(e for r, e, _ in made if r == "val").unwrapped
    check("dogrulama ortami egitim ortami verilmeden kurulamaz (hakem-2 #6: sizintili varsayilan yok)",
          raises(lambda: orig_make(made[0][2], "val"), ValueError))
    check("HPO test blogunu HIC kurmuyor", all(r in ("train", "val") for r, _, _ in made),
          str(sorted({r for r, _, _ in made})))
    check("dogrulama ortami 'supplied' ve olcegi egitim ortaminin olcegi",
          va_env.price_stats_source == "supplied"
          and va_env.get_price_stats() == {s: tr_env.get_price_stats()[s] for s in va_env.symbols})
    check("  (ayirt edici: dogrulama penceresinin kendi olcegi DEGIL)",
          va_env.get_price_stats() != own_stats(va_env.df))
    from evaluation.trial_registry import load_trials
    hrec = [t_ for t_ in load_trials(directory=os.environ["RESULTS_DIR"], exclude_tags=())
            if t_["kind"] == "rl_hpo" and (t_.get("extra") or {}).get("study") == "b9_hpo"]
    check("HPO deneme kaydi yazildi", len(hrec) == 1, str(len(hrec)))
    if hrec:
        hp = hrec[0]
        check("HPO kaydi price_stats_source='train_window', acik maddelerde degil",
              hp["protocol"].get("price_stats_source") == "train_window"
              and "price_stats_source" not in hp["protocol_open"], str(hp["protocol"]))
        check("HPO kaydi gostergeli panelden: indicators_present=True (Y21 kapandi)",
              hp["protocol"].get("indicators_present") is True
              and "indicators_present" not in hp["protocol_open"])
        check("HPO kaydi destek olcumunu ve sagligi tasiyor",
              hp["protocol"].get("price_in_support") is not None
              and (hp.get("extra") or {}).get("observation_health") is not None)
        check("HPO kaydi kati, bolme ozetini ve ayar tohumunu tasiyor",
              hp.get("fold") == "ratio" and (hp.get("split_spec") or {}).get("digest") == hspec.digest()
              and hp.get("seed") == 5, str((hp.get("fold"), hp.get("split_spec"), hp.get("seed"))))
        check("gecici (kanonik olmayan) bolme tanimi: split_from_spec acik",
              hp["protocol"].get("split_from_spec") is False
              and "split_from_spec" in hp["protocol_open"])

    # -----------------------------------------------------------------------
    print("\nS — sarmalayiciyla env kuran betikler (davranis)")
    import optuna
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
                and price_stats_source(sup[0], own[0]) == "train_window")

    check("train_a2c_phase1.create_env: price_stats ZORUNLU anahtar",
          raises(lambda: a2cs.create_env(train_df), TypeError))
    check("optimize_reward_weights.create_env_with_weights: price_stats ZORUNLU anahtar",
          raises(lambda: orw.create_env_with_weights(train_df, None, None, {}), TypeError))

    made_a2c, undo = spy_make_env(a2cs)
    try:
        venv = a2cs.create_env(train_df, price_stats=None)
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
    check("ab_test_rewards.train_and_evaluate: test ortami egitim olceginde", supplied_matches_train(made_ab))

    # Faz 2 + PSR: temel ve makro tablo zorunlu (yoksa faz 2 env'in boyutu
    # tutmuyor — B9'dan bagimsiz, Faz 9 §9.5 gozlemler)
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
    check("rapor: D5'e kadar varsayilan olarak DURUYOR (cikis kodu 2)",
          raises(lambda: gar.main([]), SystemExit))
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
    me = os.path.relpath(os.path.abspath(__file__), ROOT)
    outside = [c for c in calls if c[0] != me]
    check("held-out df ile kurulan her env price_stats tasiyor (None sabiti degil)", not bad_calls,
          "; ".join(f"{f}:{ln} ({n})" for f, ln, n, _ in bad_calls))
    check(f"bekci bos degil: test dosyasi disinda {len(outside)} held-out cagrisi (>= 10)",
          len(outside) >= 10, "; ".join(f"{f}:{ln}" for f, ln, _, _ in outside))
    print(f"    toplam {len(calls)} cagri; {len(calls) - len(outside)} tanesi bu test dosyasinda")
    sample = (
        "make_env(df=test_df, initial_balance=1)\n"       # 1 olceksiz
        "TradingEnv(val_df)\n"                            # 2 olceksiz
        "make_env(data['test'])\n"                        # 3 olceksiz
        "make_env(test_df.copy(), price_stats=s)\n"       # 4 uyumlu
        "make_env(df=latest_df)\n"                        # 5 held-out degil
        "TradingEnv(eval_df)  # B9-legacy\n"              # 6 isaretli
        "make_env(df=test_df, price_stats=None)\n"        # 7 None sabiti
        "create_env(splits.test)\n"                       # 8 sarmalayici + nitelik
        "make_env(df.loc[test_mask], price_stats=st)\n"   # 9 maske dilimi, uyumlu
        "make_env(df.loc[test_mask])\n"                   # 10 maske dilimi, olceksiz
    )
    flags = sorted((ln, ok) for _, ln, _, ok in held_out_env_calls(sample))
    want = [(1, False), (2, False), (3, False), (4, True), (6, True), (7, False),
            (8, False), (9, True), (10, False)]
    check("mutasyon: bekci olceksiz / None / sarmalayici / nitelik / maske cagrilarini yakaliyor",
          flags == want, str(flags))

    print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
