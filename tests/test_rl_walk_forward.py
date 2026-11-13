"""
Faz 9 / R1 — RL walk-forward bolmesi, kontrol noktasi secimi, rota <-> HPO ortak yolu.

Kabul olcutleri (Faz 10 §6 K20, §8 #8; Faz 9 §9.11):
  A  bolme   : wf-v1 sabit (6 kat, test 2021..2026, dogrulama onceki yil, egitim
               2013-06'dan genisleyen, purge 0; ozet sabit). Bloklar ortusmez ve
               bosluk birakmaz: [egitim baslangici, test bitisi) icindeki HER seans tam
               bir bloktadir. Tamamlanmamis test / eksik egitim baslangici hata verir.
               `fold_coverage` egitim baslangicini da denetler (2018 paneli tamam DEGIL).
  B  B0      : kat sinirinda kesme degismezligi — purge 0'in KANITI. Sentetik OHLCV ->
               GERCEK gosterge hatti (add_indicators_to_multi_symbol_df) -> load_rl_panel
               -> build_fold -> make_fold_env. Sinirdan sonraki fiyatlar degisince sinirdan
               onceki ortamin gozlem + odul + portfoy izi bit-es ('stats' ve 'causal');
               sonraki ortamin izi DEGISIR (olcut bos degil). Mutasyonlar: (i) egitim
               bloguna dogrulamanin ilk seansi eklenir, (ii) ADX bir gun ileriden okunur.
  C  secim   : test blogu degisince secilen adim, kontrol noktasi metrikleri ve secilen
               agirliklar bit-es; test metrigi degisir. Ayni kosumun tekrari bit-es ve
               kaydi 'test_reread' alir. Mutasyon: dogrulama ortami test blogundan.
  D  geri cagri: dogrulama geri cagrisi acik/kapali son agirliklar bit-es (A2C, PPO;
               PPO'da olcum rollout ortasina duser). Mutasyon: olcum stokastik predict.
  E  ortak yol: ayni kat + ayni kolda HPO'nun egitim ve dogrulama ortami rotanin
               ortamlariyla bit-es iz uretir ('stats', 'causal'). Mutasyon: HPO ADX'i
               yeniden hesaplamazsa iz ayrisir.
  F  AST     : rota egitim fonksiyonu, HPO optimizer'i ve HPO rotasinda `split_data`
               cagrisi ve 'raw_stock_data' dizesi yok (docstring haric); trading.py'nin
               tamaminda split_data yok (veri uclari kat kapsami doner). Bekci oz-testi.
  G  kayit   : farkli katlar tek N/V'de karismaz; ayni sinavin ikinci okumasi
               'test_reread'; hpo_lineage maddeleri; wf-v1 rota kaydinda
               split_from_spec=True; risksiz faiz yoksa selection_metric acik.
  H  uclar   : HPO eski tarih alani / bilinmeyen kat -> 400; /train bilinmeyen kat -> 400.
  I  birimler: kontrol noktasi izgarasi, argmax + esitlikte en erken, degenerate,
               excess_sharpe elle hesapla esit.

Risksiz faiz `rl_folds.load_risk_free` yamalanarak verilir (yerel risk_free.csv'ye
bagimli degil). Model CPU'da egitilir, butun yazimlar gecici dizine gider.

Calistirma:
    python tests/test_rl_walk_forward.py
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import glob
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Rota/HPO kosumlari GERCEK dizinlere yazmasin; model CPU'da egitilsin ("-1": Windows'ta
# bos dize degiskeni SILER ve GPU gorunur kalir; bkz. test_price_stats_window).
_TMP = tempfile.mkdtemp(prefix="rl_walk_forward_")
for _k, _sub in (("MODELS_DIR", "models"), ("RESULTS_DIR", "results"),
                 ("LOGS_DIR", "logs"), ("HYPEROPT_DIR", "hyperopt")):
    os.environ[_k] = os.path.join(_TMP, _sub)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

WF_V1_DIGEST = "33a196c602976de1"   # tanim degisirse bu test KIRILIR (yeni surum = wf-v2)
RF_DAILY = 0.0002                   # testte sabit seans basi risksiz getiri
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    passed, failed = (passed + 1, failed) if cond else (passed, failed + 1)
    print(f"  [{'OK' if cond else 'FAIL'}]   {name}" + ("" if cond else f"  {detail}"))


# ---------------------------------------------------------------------------
# Fiksturler
# ---------------------------------------------------------------------------

def raw_ohlcv(start, end, n_sym=3, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for s in range(n_sym):
        p = 30.0 * (s + 1)
        for d in dates:
            p *= 1 + rng.normal(0.0004, 0.015)
            o = p * (1 + rng.normal(0, 0.003))
            h = max(o, p) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, p) * (1 - abs(rng.normal(0, 0.004)))
            rows.append({"symbol": f"W{s}.IS", "date": d, "open": o, "high": h, "low": lo,
                         "close": p, "volume": float(rng.integers(400_000, 900_000))})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def indicator_panel(raw):
    """CSV'yi ureten GERCEK hat (varsayilanlar: warmup 'bfill', ADX 'legacy')."""
    from data.technical_indicators import add_indicators_to_multi_symbol_df
    return add_indicators_to_multi_symbol_df(raw.copy())


def random_indicator_panel(raw, seed=7):
    """Uzun panel icin hizli yol (yalniz kayit/kat alanlari olculur)."""
    rng = np.random.default_rng(seed)
    out = raw.copy()
    n = len(out)
    out["macd"] = rng.normal(0, 0.05, n)
    out["rsi"] = rng.uniform(20, 80, n)
    out["cci"] = rng.normal(0, 80, n)
    out["adx"] = rng.uniform(10, 40, n)
    out["turbulence"] = np.abs(rng.normal(0, 1, n))
    return out


def session_dates(frame):
    d = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("date")))
    return pd.DatetimeIndex(sorted((d.tz_localize(None) if d.tz is not None else d).unique()))


def perturb_after(raw, t, rate=1.005):
    """t'den SONRAKI fiyatlari yumusak bir surukleyle degistir (seans basina %0,5)."""
    out = raw.copy()
    d = pd.to_datetime(out.index.get_level_values("date"))
    later = session_dates(out)[session_dates(out) > pd.Timestamp(t)]
    rank = pd.Series(np.arange(1, len(later) + 1), index=later)
    mask = d > pd.Timestamp(t)
    factor = rate ** rank.reindex(d[mask]).to_numpy()
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * factor
    return out


def perturb_after_rich(raw, t, seed=13):
    """hakem-2 R1 #4: sembol basina FARKLI surukleme (kesit siralamasi degisir) + hacim bozulur."""
    rng = np.random.default_rng(seed)
    out = raw.copy()
    d = pd.to_datetime(out.index.get_level_values("date"))
    sym = out.index.get_level_values("symbol")
    later = session_dates(out)[session_dates(out) > pd.Timestamp(t)]
    rank = pd.Series(np.arange(1, len(later) + 1), index=later)
    mask = d > pd.Timestamp(t)
    rates = {s: 1 + rng.uniform(-0.01, 0.01) for s in pd.unique(sym)}
    factor = np.array([rates[s] for s in sym[mask]]) ** rank.reindex(d[mask]).to_numpy()
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * factor
    out.loc[mask, "volume"] = np.round(out.loc[mask, "volume"].to_numpy()
                                       * rng.uniform(0.3, 3.0, int(mask.sum())))
    return out


def env_digest(env, seed=42):
    """Gozlem + odul + portfoy degeri parmak izi (sabit eylem dizisi, tum bolum)."""
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
    return h.hexdigest()


def params_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.policy.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def const_risk_free(dates, csv_path=None):
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    return pd.Series(RF_DAILY, index=idx)


@contextlib.contextmanager
def patched(obj, name, value):
    orig = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield orig
    finally:
        setattr(obj, name, orig)


def http_status(coro):
    from fastapi import HTTPException
    try:
        asyncio.run(coro)
    except HTTPException as exc:
        return exc.status_code
    return None


# Test bolmesi (kanonik DEGIL): sentetik panel 2015..2019, egitim 2016-2017.
def test_spec():
    from evaluation import splits
    return splits.register_spec(splits.SplitSpec(
        name="r1-test",
        folds=(splits.Fold("t", "2016-01-04", "2018-01-01", "2019-01-01", "2019-07-01"),),
        purge_sessions=0))


def fold_digests(raw_or_panel, price_norm, *, is_panel=False, start="2016-01-04"):
    from evaluation import rl_folds
    panel = raw_or_panel if is_panel else indicator_panel(raw_or_panel)
    rp = rl_folds.load_rl_panel(price_norm=price_norm, start=start, df=panel)
    fd = rl_folds.build_fold(rp, test_spec(), "t")
    kw = dict(initial_balance=100_000, price_norm=price_norm)
    tr = rl_folds.make_fold_env(fd, "train", **kw)
    va = rl_folds.make_fold_env(fd, "val", train_env=tr, **kw)
    te = rl_folds.make_fold_env(fd, "test", train_env=tr, **kw)
    return {"train": env_digest(tr), "val": env_digest(va), "test": env_digest(te)}, fd


def run_route(panel, *, seed=11, total=1024, spec="r1-test", fold="t", price_norm="stats",
              risk_free=const_risk_free):
    """_run_training_inner'i sentetik panelle kostur (SB3 ciktisi yutulur)."""
    import data.data_fetcher as dfm
    from app.api.routes import trading as tr
    from app.schemas.trading import TrainingRequest
    from evaluation import rl_folds

    state = tr._empty_training_state()
    req = TrainingRequest(algorithm="A2C", phase=1, total_timesteps=total,
                          initial_balance=100_000, seed=seed, split_spec=spec, fold=fold,
                          price_norm=price_norm)
    with patched(dfm.DataFetcher, "load_data",
                 lambda self, filename="raw_stock_data.csv": panel.copy()), \
            patched(rl_folds, "load_risk_free", risk_free), \
            contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(tr._run_training_inner(req, state))
    trial = route_trial(state)
    weights = None
    if trial is not None:
        name = trial["extra"]["model_name"]
        paths = glob.glob(os.path.join(_TMP, "**", f"{name}.zip"), recursive=True)
        if paths:
            from stable_baselines3 import A2C
            weights = params_digest(A2C.load(paths[0], device="cpu"))
    return state, trial, weights


def route_trial(state):
    from evaluation.trial_registry import load_trials
    tid = (state.get("metrics") or {}).get("trial_id")
    recs = [t for t in load_trials(directory=os.environ["RESULTS_DIR"], exclude_tags=())
            if t["trial_id"] == tid]
    return recs[0] if recs else None


@contextlib.contextmanager
def recording_envs():
    """make_fold_env'in rolunu ve make_env'e verilen (df, kwargs)'i kaydet."""
    import env.trading_env as tem
    from evaluation import rl_folds
    rec, roles = [], []
    orig_mfe, orig_me = rl_folds.make_fold_env, tem.make_env

    def mfe(fold, role, **kw):
        roles.append(role)
        try:
            return orig_mfe(fold, role, **kw)
        finally:
            roles.pop()

    def me(df, **kw):
        rec.append((roles[-1] if roles else None, df, kw))
        return orig_me(df, **kw)

    rl_folds.make_fold_env, tem.make_env = mfe, me
    try:
        yield rec
    finally:
        rl_folds.make_fold_env, tem.make_env = orig_mfe, orig_me


def first_digest(rec, role):
    from env.trading_env import TradingEnv
    for r, df, kw in rec:
        if r == role:
            return env_digest(TradingEnv(df, **kw))
    return None


def run_hpo(panel, price_norm, study, seed=5):
    from app.core.config import get_settings
    from evaluation import rl_folds
    from hyperparameter_optimization.optimizers.a2c_optimizer import A2COptimizer
    get_settings()  # lru_cache: chdir'den once
    db = Path(_TMP, "hpo", f"{study}.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    opt = A2COptimizer(study_name=study, storage=f"sqlite:///{db.as_posix()}", n_trials=1,
                       seed=seed, phase=1, reward_type="simple")
    orig_suggest = opt.suggest_hyperparameters
    opt.suggest_hyperparameters = lambda trial: {**orig_suggest(trial), "n_steps": 16}
    cwd = os.getcwd()
    os.chdir(_TMP)   # tensorboard / Monitor goreli 'logs/' yaziyor
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        with patched(rl_folds, "load_risk_free", const_risk_free), \
                contextlib.redirect_stdout(io.StringIO()):
            opt.optimize(fold="t", split_spec="r1-test", total_timesteps=64,
                         price_norm=price_norm, env_settings={"initial_balance": 100_000},
                         show_progress_bar=False, panel_df=panel.copy())
    finally:
        os.chdir(cwd)
    return opt


def ast_hits(src, func=None):
    """`split_data` cagrisi ve 'raw_stock_data' dizesi (docstring haric)."""
    tree = ast.parse(src)
    node = tree
    if func is not None:
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func),
                    None)
        if node is None:
            return None
    docs = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) \
                and isinstance(n.body[0].value.value, str):
            docs.add(id(n.body[0].value))
    hits, calls = [], set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            calls.add(name)
            if name == "split_data":
                hits.append(("split_data", n.lineno))
        if isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and "raw_stock_data" in n.value and id(n) not in docs:
            hits.append(("raw_stock_data", n.lineno))
    return hits, calls


# ---------------------------------------------------------------------------

def main():
    from evaluation import rl_folds, splits
    from evaluation import trial_registry as reg

    # ------------------------------------------------------------------ A
    print("\nA — bolme tanimi (wf-v1)")
    wf = splits.WF_V1
    names = [f.name for f in wf.folds]
    check("wf-v1: 6 kat, test yillari 2021..2026", names == [str(y) for y in range(2021, 2027)],
          str(names))
    shape_ok = all(
        f.train_start == "2013-06-01" and f.val_start == f"{int(f.name) - 1}-01-01"
        and f.test_start == f"{f.name}-01-01"
        and f.test_end == ("2026-11-01" if f.name == "2026" else f"{int(f.name) + 1}-01-01")
        for f in wf.folds)
    check("her kat: egitim 2013-06-01'den, dogrulama onceki yil, test yil (2026 -> 11-01)",
          shape_ok)
    check("purge 0, tanim kanonik, ozet sabit", wf.purge_sessions == 0 and splits.is_canonical(wf)
          and wf.digest() == WF_V1_DIGEST, wf.digest())
    shifted = splits.SplitSpec(name="wf-v1-mut", folds=(splits.Fold(
        "2021", "2013-06-01", "2020-01-02", "2021-01-01", "2022-01-01"),) + wf.folds[1:])
    check("mutasyon: bir sinir bir gun kayinca ozet degisir ve kanonik sayilmaz",
          shifted.digest() != wf.digest() and not splits.is_canonical(shifted))
    try:
        splits.register_spec(splits.SplitSpec(name="wf-v1", folds=shifted.folds))
        renamed_ok = False
    except ValueError:
        renamed_ok = True
    check("kayitli bir adin tanimi degistirilemez (register_spec ValueError)", renamed_ok)

    raw_long = raw_ohlcv("2012-06-01", "2026-11-06", n_sym=2, seed=5)
    long_panel = random_indicator_panel(raw_long)
    gated_long = rl_folds.load_rl_panel(price_norm="stats", df=long_panel).panel
    cover = []
    for f in wf.folds:
        parts = splits.fold_frames(gated_long, wf, f)
        blocks = {r: set(session_dates(parts[r])) for r in ("train", "val", "test")}
        span = session_dates(gated_long)
        span = set(span[(span >= pd.Timestamp(f.train_start)) & (span < pd.Timestamp(f.test_end))])
        disjoint = not (blocks["train"] & blocks["val"] or blocks["val"] & blocks["test"]
                        or blocks["train"] & blocks["test"])
        union = blocks["train"] | blocks["val"] | blocks["test"]
        cover.append(disjoint and union == span)
    check("her katta bloklar ortusmez ve [egitim, test bitisi) icindeki HER seans bir blokta "
          "(bosluk yok)", all(cover) and len(cover) == 6, str(cover))
    short = gated_long[pd.to_datetime(gated_long.index.get_level_values("date"))
                       < pd.Timestamp("2026-10-15")]
    try:
        splits.fold_frames(short, wf, "2026")
        cov_err = False
    except splits.SplitCoverageError:
        cov_err = True
    check("tamamlanmamis test blogu SplitCoverageError (kisa test sessizce kabul edilmez)",
          cov_err)
    check("latest_complete_fold: panel 2026-10-15'te bitince 2025, tam panelde 2026",
          splits.latest_complete_fold(short, wf).name == "2025"
          and splits.latest_complete_fold(gated_long, wf).name == "2026")
    late = gated_long[pd.to_datetime(gated_long.index.get_level_values("date"))
                      >= pd.Timestamp("2015-01-01")]
    try:
        splits.fold_frames(late, wf, "2021")
        start_err = False
    except splits.SplitCoverageError:
        start_err = True
    check("panel egitim baslangicini kapsamiyorsa SplitCoverageError", start_err)
    cov_full = splits.fold_coverage(gated_long, wf)
    late18 = gated_long[pd.to_datetime(gated_long.index.get_level_values("date"))
                        >= pd.Timestamp("2018-01-02")]
    cov_late = splits.fold_coverage(late18, wf)
    check("fold_coverage: tam panelde 6 kat tamam",
          [c["complete"] for c in cov_full] == [True] * 6, str(cov_full))
    check("fold_coverage: 2018'de baslayan panel test bitisini kapsasa da HICBIR kat tamam degil "
          "(eskiden yalniz bitise bakiliyordu)",
          all(c["ends_ok"] and not c["starts_ok"] and not c["complete"] for c in cov_late))
    check("fold_coverage: 2026-10-15'te biten panelde yalniz 2026 kati eksik",
          [c["complete"] for c in splits.fold_coverage(short, wf)] == [True] * 5 + [False])

    # ------------------------------------------------------------------ B
    print("\nB — kat sinirinda kesme degismezligi (purge 0'in kaniti; gercek gosterge hatti)")
    raw = raw_ohlcv("2015-01-01", "2019-12-31", n_sym=3, seed=3)
    panel = indicator_panel(raw)
    base_s, fd = fold_digests(panel, "stats", is_panel=True)
    t1, t2 = fd.report["ranges"]["train"][1], fd.report["ranges"]["val"][1]
    print(f"    sinirlar: egitim son {t1}, dogrulama son {t2}; seans {fd.report['n_sessions']}")
    panel_p1 = indicator_panel(perturb_after(raw, t1))
    panel_p2 = indicator_panel(perturb_after(raw, t2))
    for pn in ("stats", "causal"):
        base = base_s if pn == "stats" else fold_digests(panel, pn, is_panel=True)[0]
        p1 = fold_digests(panel_p1, pn, is_panel=True)[0]
        p2 = fold_digests(panel_p2, pn, is_panel=True)[0]
        check(f"[{pn}] dogrulama sonrasi fiyat degisince EGITIM ortaminin izi bit-es",
              p1["train"] == base["train"])
        check(f"[{pn}] ... ve dogrulama ortaminin izi degisir (olcut bos degil)",
              p1["val"] != base["val"])
        check(f"[{pn}] test sonrasi fiyat degisince egitim ve DOGRULAMA izi bit-es",
              p2["train"] == base["train"] and p2["val"] == base["val"])
        check(f"[{pn}] ... ve test ortaminin izi degisir (olcut bos degil)",
              p2["test"] != base["test"])

    # hakem-2 R1 #4: daha genis bozma ailesi — sembol basina farkli surukleme ve hacim
    rich1 = indicator_panel(perturb_after_rich(raw, t1))
    rich2 = indicator_panel(perturb_after_rich(raw, t2))
    for pn in ("stats", "causal"):
        base = base_s if pn == "stats" else fold_digests(panel, pn, is_panel=True)[0]
        q1 = fold_digests(rich1, pn, is_panel=True)[0]
        q2 = fold_digests(rich2, pn, is_panel=True)[0]
        check(f"[{pn}] sembol basina surukleme + hacim bozulunca: dogrulama sonrasi -> egitim "
              f"bit-es, dogrulama degisir; test sonrasi -> egitim + dogrulama bit-es, test degisir",
              q1["train"] == base["train"] and q1["val"] != base["val"]
              and q2["train"] == base["train"] and q2["val"] == base["val"]
              and q2["test"] != base["test"])

    orig_ff = rl_folds.fold_frames
    calls = []

    def leaky_fold_frames(df, spec, fold):
        calls.append(1)
        out = orig_ff(df, spec, fold)
        vd = pd.to_datetime(out["val"].index.get_level_values("date"))
        out["train"] = pd.concat([out["train"], out["val"][vd == vd.min()]]).sort_index()
        return out

    with patched(rl_folds, "fold_frames", leaky_fold_frames):
        m0 = fold_digests(panel, "stats", is_panel=True)[0]
        m1 = fold_digests(panel_p1, "stats", is_panel=True)[0]
    check("mutasyon (i): egitim bloguna dogrulamanin ilk seansi eklenince B0 KALIR "
          "(sinir hatasi yakalanir)", bool(calls) and m1["train"] != m0["train"])

    import data.technical_indicators as ti
    orig_with_adx = ti.with_adx

    def future_adx(df, mode="wilder", period=ti.ADX_PERIOD):
        out = orig_with_adx(df, mode=mode, period=period)
        out["adx"] = out.groupby(level="symbol")["adx"].shift(-1)
        return out

    with patched(ti, "with_adx", future_adx):
        m0 = fold_digests(panel, "stats", is_panel=True)[0]
        m1 = fold_digests(panel_p1, "stats", is_panel=True)[0]
    check("mutasyon (ii): ADX bir gun ileriden okununca B0 KALIR (nedensel olmayan "
          "gosterge yakalanir)", m1["train"] != m0["train"])

    # ------------------------------------------------------------------ C
    print("\nC — test blogu secime sizmiyor (rota uctan uca, A2C 1024 adim, CPU)")
    panel_test_changed = panel_p2
    s_base, t_base, w_base = run_route(panel)
    s_pert, t_pert, w_pert = run_route(panel_test_changed)
    s_again, t_again, w_again = run_route(panel)
    ok_runs = all(s.get("state") == "completed" for s in (s_base, s_pert, s_again)) \
        and None not in (w_base, w_pert, w_again)
    check("uc rota kosumu tamamlandi, modeller okundu", ok_runs,
          str([s.get("error") for s in (s_base, s_pert, s_again)]))
    sel = lambda s: json.dumps((s.get("metrics") or {}).get("selection"), sort_keys=True)
    n_ck = len(((s_base.get("metrics") or {}).get("selection") or {}).get("checkpoints") or [])
    print(f"    kontrol noktasi: {n_ck}, secilen adim "
          f"{((s_base.get('metrics') or {}).get('selection') or {}).get('selected_step')}")
    check("belirlenimcilik: ayni kosumun tekrari secim + agirlik bit-es",
          sel(s_again) == sel(s_base) and w_again == w_base)
    check("test blogu degisince secim (adim + 10 kontrol noktasinin metrikleri) bit-es",
          sel(s_pert) == sel(s_base) and n_ck == rl_folds.N_CHECKPOINTS)
    check("test blogu degisince secilen agirliklar bit-es", w_pert == w_base)
    sel_b = (s_base.get("metrics") or {}).get("selection") or {}
    shas = [c.get("params_sha") for c in sel_b.get("checkpoints") or []]
    check("secim raporu farkli agirlik sayisini tasir (A2C n_steps=256, 1024 adim: guncelleme "
          "arasina dusen noktalar ayni model; hakem-rl R1 #4)",
          None not in shas and sel_b.get("n_distinct") == len(set(shas))
          and 1 < sel_b["n_distinct"] < rl_folds.N_CHECKPOINTS, f"{sel_b.get('n_distinct')} / {shas}")
    cks = sel_b.get("checkpoints") or []
    last = max(cks, key=lambda c: c["step"]) if cks else {}
    check("secim raporu tani tasir: her noktada ortalama fazla getiri, negatif nokta sayisi, "
          "son noktanin dogrulama metrigi (hakem-finans R1 #1; karar 2026-11-14)",
          bool(cks) and all(c.get("mean_excess") is not None for c in cks)
          and sel_b.get("n_negative_mean_excess") == sum(c["mean_excess"] < 0 for c in cks)
          and sel_b.get("last_metric") == last.get("metric"), str({k: sel_b.get(k) for k in
                                                                    ("n_negative_mean_excess", "last_metric")}))
    best_mx = sorted(cks, key=lambda c: (-c["mean_excess"], c["step"]))[0]["step"] if cks else None
    check("secim raporu ortalama fazla getiriyle secilecek noktayi ve uyusmayi tasir (hakem-2 R1 #2)",
          sel_b.get("pick_mean_excess_step") == best_mx
          and sel_b.get("picks_agree") == (best_mx == sel_b.get("selected_step")))
    cr = lambda s: (s.get("metrics") or {}).get("cumulative_return")
    check("... ve test metrigi degisir (olcut bos degil)", cr(s_pert) != cr(s_base),
          f"{cr(s_base)} / {cr(s_pert)}")
    tags = lambda t: set((t or {}).get("tags") or [])
    check("ayni sinavin ikinci okumasi 'test_reread' alir; ilki ve farkli veriyle olan almaz",
          reg.TEST_REREAD_TAG in tags(t_again) and reg.TEST_REREAD_TAG not in tags(t_base)
          and reg.TEST_REREAD_TAG not in tags(t_pert), str([tags(t_base), tags(t_again)]))
    # Bu kayitlar protocol_open da tasiyor (test bolmesi kanonik degil); dislamanin
    # test_reread'den geldigini gormek icin YALNIZ o etiketi dislayan okuma kullanilir.
    only_reread = {t["trial_id"] for t in reg.load_trials(
        directory=os.environ["RESULTS_DIR"], exclude_tags=(reg.TEST_REREAD_TAG,))}
    check("test_reread etiketiyle okuma ikinci okumayi dislar, ilkini tutar",
          t_again is not None and t_base is not None
          and t_again["trial_id"] not in only_reread and t_base["trial_id"] in only_reread)
    check("varsayilan okumanin dislama listesinde test_reread var",
          reg.TEST_REREAD_TAG in reg.load_trials.__defaults__[2])

    # Secim son nokta degilse rota kontrol noktasini diskten yukler; yukaridaki kosumlarda
    # secim son noktaya dustu, bu dal calismadi. Secimi 4. noktaya zorla.
    from stable_baselines3 import A2C as _A2C
    captured = {}
    real_select = rl_folds.select_checkpoint

    def forced_pick(records):
        chosen = records[3]
        final = [r for r in records if r["final"]][0]
        captured["chosen"] = params_digest(_A2C.load(chosen["path"], device="cpu"))
        captured["final"] = params_digest(_A2C.load(final["path"], device="cpu"))
        captured["step"] = chosen["step"]
        return {**real_select(records), **chosen}   # tani alanlari gercek kuraldan

    with patched(rl_folds, "select_checkpoint", forced_pick):
        s_f, t_f, w_f = run_route(panel)
    check("secim son nokta degilse kaydedilen model SECILEN noktanin agirliklari "
          "(son noktanin degil); meta adimi ayni",
          bool(captured) and w_f == captured["chosen"] and captured["chosen"] != captured["final"]
          and ((s_f.get("metrics") or {}).get("selection") or {}).get("selected_step")
          == captured["step"], f"{captured.get('step')}")

    orig_mfe = rl_folds.make_fold_env

    def val_from_test(fold, role, **kw):
        return orig_mfe(fold, "test" if role == "val" else role, **kw)

    with patched(rl_folds, "make_fold_env", val_from_test):
        s_m0, _, _ = run_route(panel)
        s_m1, _, _ = run_route(panel_test_changed)
    check("mutasyon: dogrulama ortami test blogundan kurulunca secim degisir (yakalanir)",
          s_m0.get("state") == "completed" and sel(s_m1) != sel(s_m0))

    # ------------------------------------------------------------------ D
    print("\nD — dogrulama geri cagrisi egitimi bozmuyor")
    from stable_baselines3 import A2C, PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise

    def learn(algo, with_cb, total=320):
        env = rl_folds.make_fold_env(fd, "train", initial_balance=100_000)
        stats_env = rl_folds.make_fold_env(fd, "train", initial_balance=100_000)
        if algo == "A2C":
            model = A2C("MlpPolicy", env, n_steps=32, seed=5, device="cpu", verbose=0)
        elif algo == "PPO":
            model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2, seed=5,
                        device="cpu", verbose=0)
        else:
            # off-policy: olcum collect_rollouts icinde; ilk noktalar learning_starts'tan once
            # (rotadaki gibi TD3'te global numpy RNG'li eylem gurultusu var)
            n = env.action_space.shape[0]
            extra = ({"action_noise": NormalActionNoise(np.zeros(n), 0.2 * np.ones(n))}
                     if algo == "TD3" else {})
            cls = TD3 if algo == "TD3" else SAC
            model = cls("MlpPolicy", env, learning_starts=64, batch_size=32, buffer_size=1000,
                        seed=5, device="cpu", verbose=0, **extra)
        cb = rl_folds.make_validation_callback(
            lambda: rl_folds.make_fold_env(fd, "val", train_env=stats_env, initial_balance=100_000),
            total_timesteps=total, checkpoint_dir=tempfile.mkdtemp(dir=_TMP),
            risk_free=None)
        model.learn(total, callback=cb if with_cb else None)
        return params_digest(model), (len(cb.records) if with_cb else 0)

    for algo in ("A2C", "PPO", "TD3", "SAC"):
        w_off, _ = learn(algo, False)
        w_on, n_rec = learn(algo, True)
        check(f"[{algo}] geri cagri acik/kapali son agirliklar bit-es (olcum {n_rec} nokta)",
              w_on == w_off and n_rec == rl_folds.N_CHECKPOINTS, f"kayit {n_rec}")

    orig_run = rl_folds.run_episode

    def stochastic_episode(model, env, collect_obs=False):
        orig_predict = model.predict
        model.predict = lambda obs, deterministic=True: orig_predict(obs, deterministic=False)
        try:
            return orig_run(model, env, collect_obs)
        finally:
            model.predict = orig_predict

    with patched(rl_folds, "run_episode", stochastic_episode):
        w_mut, _ = learn("A2C", True)
    w_ref, _ = learn("A2C", False)
    check("mutasyon: olcum stokastik predict ile yapilinca agirliklar ayrisir (yakalanir)",
          w_mut != w_ref)

    # ------------------------------------------------------------------ E
    print("\nE — rota <-> HPO ayni gozlem (ayni kat, ayni kol)")
    import data.technical_indicators as ti2
    for pn in ("stats", "causal"):
        with recording_envs() as r_rec:
            s_r, _, _ = run_route(panel, total=256, price_norm=pn)
        with recording_envs() as h_rec:
            opt = run_hpo(panel, pn, study=f"r1_e_{pn}")
        done = s_r.get("state") == "completed" and opt.best_params is not None
        dr = {r: first_digest(r_rec, r) for r in ("train", "val")}
        dh = {r: first_digest(h_rec, r) for r in ("train", "val")}
        check(f"[{pn}] rota ve HPO kosumu tamamlandi, ortamlar yakalandi",
              done and None not in dr.values() and None not in dh.values(),
              str(s_r.get("error")))
        check(f"[{pn}] egitim ortami bit-es iz", dr["train"] == dh["train"])
        check(f"[{pn}] dogrulama ortami bit-es iz", dr["val"] == dh["val"])
    with patched(ti2, "with_adx", lambda df, mode="wilder", period=14: df.copy()), \
            recording_envs() as m_rec:
        run_hpo(panel, "stats", study="r1_e_mut")
    with recording_envs() as r_rec:
        run_route(panel, total=256, price_norm="stats")
    check("mutasyon: HPO ADX'i yeniden hesaplamazsa egitim izi ayrisir (yakalanir)",
          first_digest(m_rec, "train") != first_digest(r_rec, "train"))

    # ------------------------------------------------------------------ F
    print("\nF — AST bekcisi")
    targets = [("app/api/routes/trading.py", "_run_training_inner", "build_fold"),
               ("hyperparameter_optimization/base_optimizer.py", None, "build_fold"),
               ("app/api/routes/hyperopt.py", None, "get_spec")]
    for path, func, positive in targets:
        res = ast_hits(Path(ROOT, path).read_text(encoding="utf-8"), func)
        label = f"{path}" + (f"::{func}" if func else "")
        check(f"{label}: split_data / raw_stock_data yok (bekci '{positive}' cagrisini goruyor)",
              res is not None and not res[0] and positive in res[1],
              str(res[0] if res else "fonksiyon bulunamadi"))
    # Veri bilgi ucu da (data/info; data/generate §9.13'te kaldirildi) oranli bolme raporlamiyor:
    # trading.py'nin TAMAMINDA split_data cagrisi yok ('raw_stock_data' dizesi veri
    # durumu uclarinda mesru oldugu icin burada yalniz cagri aranir).
    res = ast_hits(Path(ROOT, "app/api/routes/trading.py").read_text(encoding="utf-8"))
    sd = [h for h in res[0] if h[0] == "split_data"]
    check("app/api/routes/trading.py (tum modul): split_data cagrisi yok", not sd, str(sd))
    probe = ('async def _run_training_inner(r, s):\n'
             '    """raw_stock_data docstring"""\n'
             '    x = fetcher.split_data(df)\n'
             '    y = "data/bist/raw_stock_data.csv"\n')
    hits = ast_hits(probe, "_run_training_inner")[0]
    check("bekci oz-testi: enjekte split_data ve raw_stock_data yakalanir, docstring sayilmaz",
          sorted(h[0] for h in hits) == ["raw_stock_data", "split_data"], str(hits))

    # ------------------------------------------------------------------ G
    print("\nG — deneme kaydi")
    REG = os.path.join(_TMP, "reg")
    os.makedirs(REG, exist_ok=True)
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2021-01-04", periods=120)
    spec_a = {"name": "wf-v1", "digest": WF_V1_DIGEST}

    def put(fold, seed, cfg, sha="s1", kind="rl_train", **kw):
        return reg.register_trial(kind=kind, config=cfg, seed=seed,
                                  returns=rng.normal(0.0005, 0.01, len(days)), dates=days,
                                  split="test", fold=fold, split_spec=spec_a, data_sha256=sha,
                                  protocol=dict(reg.CLEAN_PROTOCOL), directory=REG, **kw)

    for i in range(2):
        put("2021", 100 + i, {"a": i})
        put("2022", 200 + i, {"a": i})
    recs = reg.load_trials(kind="rl_train", directory=REG)
    try:
        reg.sharpe_trial_stats(recs)
        mixed_err = False
    except ValueError:
        mixed_err = True
    check("farkli katlarin denemeleri tek N/V'de karisinca ValueError", mixed_err)
    one = [t for t in recs if t["fold"] == "2021"]
    try:
        reg.sharpe_trial_stats(one)
        one_ok = True
    except ValueError as exc:
        one_ok = False
        print(f"    {exc}")
    check("tek katin denemeleri hesaplanir", one_ok)

    first = put("2023", 7, {"b": 1})
    second = put("2023", 7, {"b": 1})
    other_seed = put("2023", 8, {"b": 1})
    other_data = put("2023", 7, {"b": 1}, sha="s2")
    all_recs = {t["trial_id"]: t for t in reg.load_trials(directory=REG, exclude_tags=())}
    rr = lambda tid: reg.TEST_REREAD_TAG in (all_recs[tid].get("tags") or [])
    check("ayni (yapilandirma, tohum, kat, bolme, veri) ikinci kez -> test_reread; "
          "farkli tohum / farkli veri -> yok",
          rr(second) and not rr(first) and not rr(other_seed) and not rr(other_data))

    ind = {"adx": "wilder"}
    hpo_cfg = {"symbols": ["A", "B"], "indicator_spec": ind, "price_norm": "stats"}

    def put_hpo(study, seed=7, val_end="2020-12-31", cfg=None, digest=WF_V1_DIGEST):
        return reg.register_trial(
            kind="rl_hpo", config=cfg or hpo_cfg, seed=seed, split="val", fold="2021",
            split_spec={"name": "wf-v1", "digest": digest},
            protocol=dict(reg.CLEAN_PROTOCOL), directory=REG,
            extra={"study": study, "val_range": ["2020-01-02", val_end]})

    put_hpo("s_clean")
    want = dict(fold_test_start="2021-01-01", train_seed=8, symbols=["A", "B"],
                split_digest=WF_V1_DIGEST, obs_config={"price_norm": "stats", "indicator_spec": ind},
                directory=REG)
    lin = lambda study, **over: reg.hpo_lineage(study, **{**want, **over})
    check("hpo_lineage: temiz soy -> (True, [])", lin("s_clean") == (True, []), str(lin("s_clean")))
    cases = {
        "hpo_window_after_test_start": lin("s_clean", fold_test_start="2020-06-01"),
        "hpo_seed_reused": lin("s_clean", train_seed=7),
        "hpo_universe_mismatch": lin("s_clean", symbols=["B", "A"]),
        "hpo_split_mismatch": lin("s_clean", split_digest="0" * 16),
        "hpo_obs_mismatch": lin("s_clean", obs_config={"price_norm": "causal",
                                                       "indicator_spec": ind}),
    }
    put_hpo("s_noseed", seed=None)
    cases["hpo_seed_unknown"] = lin("s_noseed")
    for item, res in cases.items():
        check(f"hpo_lineage: tek degisiklik -> yalniz '{item}'", res == (False, [item]), str(res))
    put_hpo("s_failed")
    put_hpo("s_failed", cfg={**hpo_cfg, "price_norm": None})   # ortam kurulmadan dusen deneme
    check("hpo_lineage: ortamsiz dusen denemenin price_norm=None'i soyu acar (guvenli yon, beyanli)",
          lin("s_failed") == (False, ["hpo_obs_mismatch"]), str(lin("s_failed")))
    check("hpo_lineage: kaydi olmayan calisma -> (None, [hpo_trials_not_in_registry])",
          lin("s_yok") == (None, ["hpo_trials_not_in_registry"]))

    s_wf, t_wf, _ = run_route(long_panel, total=256, spec=None, fold=None)
    prot = (t_wf or {}).get("protocol") or {}
    ranges = (t_wf or {}).get("extra", {}).get("ranges") or {}
    check("wf-v1 rota kosumu: varsayilan kat panelin tamamladigi son kat (2026)",
          s_wf.get("state") == "completed" and s_wf.get("fold") == "2026", str(s_wf.get("error")))
    check("wf-v1 rota kaydi: split_from_spec=True, selection_metric='val_excess_sharpe', ozet",
          prot.get("split_from_spec") is True and prot.get("selection_metric") == "val_excess_sharpe"
          and (t_wf.get("split_spec") or {}).get("digest") == WF_V1_DIGEST
          and not {"split_from_spec", "selection_metric"} & set(t_wf.get("protocol_open") or []),
          str(prot))
    check("wf-v1 kat 2026 araliklari: egitim < 2025, dogrulama 2025, test 2026 (< 11-01)",
          bool(ranges) and ranges["train"][1] < "2025-01-01"
          and ranges["val"][0] >= "2025-01-01" and ranges["val"][1] < "2026-01-01"
          and ranges["test"][0] >= "2026-01-01" and ranges["test"][1] < "2026-11-01", str(ranges))
    s_rf, t_rf, _ = run_route(panel, total=256, risk_free=lambda dates, csv_path=None: None)
    check("risksiz faiz yoksa secim rf=0 Sharpe ile, kayitta 'selection_metric' ACIK + uyari",
          (t_rf or {}).get("protocol", {}).get("selection_metric") == rl_folds.SELECTION_METRIC_RF0
          and "selection_metric" in ((t_rf or {}).get("protocol_open") or [])
          and any("Risksiz faiz" in w for w in s_rf.get("warnings") or []),
          str((t_rf or {}).get("protocol_open")))
    check("wf-v1 2026 kati test penceresi R1 oncesi okunmus pencereyle kesisiyor -> "
          "eval_window_unexposed=False, kayit acik (hakem-2 R1 #1; karar 2026-11-14)",
          prot.get("eval_window_unexposed") is False
          and "eval_window_unexposed" in (t_wf.get("protocol_open") or [])
          and any("B9" in w for w in s_wf.get("warnings") or []), str(prot.get("eval_window_unexposed")))
    check("2019 testli sentetik kat okunmamis pencerede: eval_window_unexposed=True; faz 1 "
          "fundamentals_pit=True",
          (t_base or {}).get("protocol", {}).get("eval_window_unexposed") is True
          and (t_base or {}).get("protocol", {}).get("fundamentals_pit") is True)
    check("ratio/test tanimiyla kayit split_from_spec=False (kanonik degil, acik)",
          (t_base or {}).get("protocol", {}).get("split_from_spec") is False
          and "split_from_spec" in ((t_base or {}).get("protocol_open") or []))

    # ------------------------------------------------------------------ H
    print("\nH — uclar")
    from fastapi import BackgroundTasks
    from app.api.routes import hyperopt as hp
    from app.api.routes import trading as tr
    from app.schemas.hyperopt import OptimizationRequest
    from app.schemas.trading import TrainingRequest
    check("HPO: eski tarih alani (train_start) -> 400",
          http_status(hp.start_optimization(
              OptimizationRequest(algorithm="a2c", train_start="2018-01-01"),
              BackgroundTasks(), None)) == 400)
    check("HPO: bilinmeyen kat -> 400",
          http_status(hp.start_optimization(OptimizationRequest(algorithm="a2c", fold="1999"),
                                            BackgroundTasks(), None)) == 400)
    spawned = []
    key = tr._training_key(None)
    with patched(tr, "_check_training_data_freshness", lambda phase, **kw: None), \
            patched(tr.jobs, "spawn", lambda fn, name=None: spawned.append(name)):
        bad = http_status(tr.start_training(TrainingRequest(fold="1999"), BackgroundTasks(), None))
        tr._training_states.pop(key, None)
        n_bad = len(spawned)
        good = http_status(tr.start_training(TrainingRequest(fold="2021"), BackgroundTasks(), None))
        tr._training_states.pop(key, None)
    check("/train: bilinmeyen kat -> 400 ve arka plan gorevi BASLAMAZ",
          bad == 400 and n_bad == 0, f"durum {bad}, baslatilan {n_bad}")
    check("/train: gecerli kat kabul edilir", good is None and len(spawned) == n_bad + 1,
          f"durum {good}")

    # ------------------------------------------------------------------ I
    print("\nI — secim kurali birimleri")
    check("izgara: 10.000 adim -> 1.000'er, son nokta butce sonu",
          rl_folds.checkpoint_grid(10_000) == list(range(1000, 10_001, 1000)))
    check("izgara: butce nokta sayisindan kucukse butce kadar nokta",
          rl_folds.checkpoint_grid(5) == [1, 2, 3, 4, 5])
    recs_ck = [{"index": i, "step": s, "metric": m}
               for i, (s, m) in enumerate([(10, 0.1), (20, 0.5), (30, 0.5), (40, None)])]
    pick = rl_folds.select_checkpoint(recs_ck)
    check("secim: argmax, esitlikte en erken adim, tanimsiz metrik sayilmaz",
          pick["step"] == 20 and pick["n_valid"] == 3 and not pick["degenerate"])
    dup = rl_folds.select_checkpoint([{"index": i, "step": s, "metric": m, "params_sha": h}
                                      for i, (s, m, h) in enumerate([(10, 0.1, "a"), (20, 0.1, "a"),
                                                                     (30, 0.3, "b")])])
    check("secim: ayni agirlikli noktalar tek aday sayilir (n_distinct)",
          dup["n_distinct"] == 2 and dup["step"] == 30)
    neg = rl_folds.select_checkpoint([{"index": i, "step": s, "metric": m, "mean_excess": e}
                                      for i, (s, m, e) in enumerate([(10, -0.5, -0.2), (20, 0.4, 0.1),
                                                                     (30, -0.1, -0.05)])])
    check("tani: negatif ortalamali nokta sayisi ve son noktanin metrigi; secim kurali degismez",
          neg["n_negative_mean_excess"] == 2 and neg["last_metric"] == -0.1 and neg["step"] == 20)
    vs = rl_folds.select_checkpoint([{"index": i, "step": s, "metric": m, "mean_excess": e}
                                     for i, (s, m, e) in enumerate([(10, -0.2, -0.30), (20, -0.1, -0.40),
                                                                    (30, -0.3, -0.10)])])
    check("tani: oynaklik ortalamadan farkli noktayi sectirince picks_agree=False "
          "(Sharpe 20'yi, ortalama fazla getiri 30'u secer)",
          vs["step"] == 20 and vs["pick_mean_excess_step"] == 30 and vs["picks_agree"] is False)
    import types
    import evaluation.trial_registry as treg
    win = lambda a, b: types.SimpleNamespace(df=pd.DataFrame(
        {"close": [1.0, 1.0]}, index=pd.MultiIndex.from_tuples(
            [("A", pd.Timestamp(a)), ("A", pd.Timestamp(b))], names=["symbol", "date"])))
    check("eval_window_unexposed: okunmus pencereyle kesisen -> False, son okunmus gunden "
          "sonrasi ve 2021 -> True (olculen pencereler: exposure.json)",
          treg.eval_window_unexposed(win("2026-08-28", "2026-10-30")) is False
          and treg.eval_window_unexposed(win("2026-08-31", "2026-10-30")) is True
          and treg.eval_window_unexposed(win("2022-07-18", "2022-12-30")) is False
          and treg.eval_window_unexposed(win("2021-01-04", "2022-07-15")) is True)
    check("fundamentals_pit: faz 2 + temel oran tablosu -> False; faz 1 -> True",
          treg.fundamentals_pit(types.SimpleNamespace(phase=2, fundamental_df=pd.DataFrame({"x": [1]})))
          is False and treg.fundamentals_pit(types.SimpleNamespace(phase=1, fundamental_df=None)) is True)
    deg = rl_folds.select_checkpoint([{"index": i, "step": s, "metric": None}
                                      for i, s in enumerate([10, 20])])
    check("hicbir metrik tanimli degilse son nokta + degenerate=True",
          deg["step"] == 20 and deg["degenerate"] is True)
    r = np.random.default_rng(9).normal(0.001, 0.01, 60)
    d = pd.bdate_range("2024-01-02", periods=60)
    rf = pd.Series(np.linspace(0.0001, 0.0003, 60), index=d)
    x = r - rf.to_numpy()
    hand = x.mean() / x.std(ddof=1) * np.sqrt(252)
    check("excess_sharpe: elle hesapla esit (risksiz faiz seans basina dusulur)",
          abs(rl_folds.excess_sharpe(r, d, rf) - hand) < 1e-12)
    check("mean_excess: elle hesapla esit (yillik), risksiz faiz yoksa None",
          abs(rl_folds.mean_excess(r, d, rf) - x.mean() * 252) < 1e-12
          and rl_folds.mean_excess(r, d, None) is None)
    check("excess_sharpe: sabit brut getiri (islem yok) ve < 3 gozlem -> None",
          rl_folds.excess_sharpe(np.zeros(30)) is None
          and rl_folds.excess_sharpe([0.01, 0.02]) is None)
    try:
        rl_folds.excess_sharpe(r, None, rf)
        rf_err = False
    except ValueError:
        rf_err = True
    check("excess_sharpe: risksiz faiz verilip tarih verilmezse ValueError", rf_err)

    print("\n" + "=" * 60)
    print(f"Sonuc: {passed} gecti, {failed} kaldi")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
