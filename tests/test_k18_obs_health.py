"""
Faz 9 / K18 (+ K25 b) — ADX duzeltmesi, nedensel fiyat olcegi, gozlem sagligi.

Kabul olcutleri (Faz 10 §6 K18, K25; Faz 9 §9.9):
  A  ADX    : legacy hesabin kok nedeni kayit olarak sabit (tarih indeksli df'te
              tumuyle NaN / 0). 'wilder' indeks hizali, [0, 100], ilk 2n-1 satir NaN;
              iki BAGIMSIZ referansa yakinsar (SMA tohumlu klasik Wilder dongusu,
              pandas_ta). B0: 'nan' modunda her t'de kesme degismez; 'bfill' modu
              isinma icindeki t'de KALIR (olcut bos degil).
  D  dondurma: varsayilanlar main @ MAIN_REF ile bit-es (git show): gostergeler
              ('bfill' / 'nan'), veri kapisi (legacy panel) ve env (8 yapilandirma).
  G  kapi   : 'wilder' ADX'in NaN isinmasi atilir; atilan satirlar bagimsiz 'nan'
              referansinin gecerli satirlarini kapsar. Mutasyon: dedektor adx'i
              bilmezse geriye doldurulmus ADX satirlari kalir -> yakalanir.
  N  K25    : price_norm='causal' — z tanimi bagimsiz hesapla ayni; B0 (t sonrasi
              fiyat degisince t'ye kadar TUM gozlem bit-es; t in {0, orta, T-2}).
              Mutasyon: merkezli pencere (gelecek) t=0'da yakalanir. Baglam ilk
              tarihten once bitmeli. Yukselen panelde destek ICINDE, B9 olcegi DISINDA.
  P  protokol: 'causal' temiz, 'mixed_norm' acik; obs_health kirpma ve sabit boyutu
              olcer; adx_constant egitim ortamini da olcer.
  R  rota   : _run_training_inner (sentetik panel, CPU): ADX 'wilder', meta/manifest/
              kayit gozlem tanimini tasir; causal kolda test ortaminin piyasa blogu
              t sonrasi fiyat degisince t'ye kadar bit-es, etiket 'causal'.
  X  gercek panel (CSV varsa): ADX invaryantlari, kapinin attigi satirlar.

Calistirma:
    python tests/test_k18_obs_health.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Rota kosumu GERCEK dizinlere yazmasin; model CPU'da egitilsin ("-1": Windows'ta
# bos dize degiskeni SILER ve GPU gorunur kalir; bkz. test_price_stats_window).
_TMP = tempfile.mkdtemp(prefix="k18_obs_health_")
for _k, _sub in (("MODELS_DIR", "models"), ("RESULTS_DIR", "results"),
                 ("LOGS_DIR", "logs"), ("HYPEROPT_DIR", "hyperopt")):
    os.environ[_k] = os.path.join(_TMP, _sub)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

MAIN_REF = "01aabdb"  # K18 oncesi main
# Gercek panelde kapinin pencere icinde attigi isinma satirlari (B7'de ODAS 10, PGSUS 17;
# 'wilder' ADX'in 27 satirlik isinmasi CCI'nin 20 satirlik blogundan uzun). Olcum:
# scripts/analysis/measure_obs_health.py (Faz 9 §9.9). Panel tazelenince degisebilir.
EXPECTED_REAL_DROPS = {"ODAS.IS": 17, "PGSUS.IS": 24}
N = 14
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    passed, failed = (passed + 1, failed) if cond else (passed, failed + 1)
    print(f"  [{'OK' if cond else 'FAIL'}]   {name}" + ("" if cond else f"  {detail}"))


# ---------------------------------------------------------------------------
# Fiksturler
# ---------------------------------------------------------------------------

def ohlcv(n_days=500, n_sym=3, seed=3, late=None, drift=0.0005, start="2021-01-04"):
    """late=(sembol_indeksi, ilk_gun): o sembol sonradan listelenir."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 30.0 * (s + 1)
        for i, d in enumerate(dates):
            p *= 1 + rng.normal(drift, 0.015)
            if late and s == late[0] and i < late[1]:
                continue
            o = p * (1 + rng.normal(0, 0.003))
            h = max(o, p) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, p) * (1 - abs(rng.normal(0, 0.004)))
            rows.append({"symbol": f"W{s}.IS", "date": d, "open": o, "high": h, "low": lo,
                         "close": p, "volume": float(rng.integers(400_000, 900_000))})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def with_random_indicators(frame, seed=7):
    """Rota testindeki CSV benzeri: gosterge sutunlari dolu (adx rastgele)."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    n = len(out)
    out["macd"] = rng.normal(0, 0.05, n)
    out["rsi"] = rng.uniform(20, 80, n)
    out["cci"] = rng.normal(0, 80, n)
    out["adx"] = rng.uniform(10, 40, n)
    out["turbulence"] = np.abs(rng.normal(0, 1, n))
    return out


def dates_of(frame):
    return pd.DatetimeIndex(sorted(frame.index.get_level_values("date").unique()))


def before(frame, ts):
    return frame[frame.index.get_level_values("date") < ts]


def from_(frame, ts):
    return frame[frame.index.get_level_values("date") >= ts]


def perturb_after(frame, t):
    """t'den SONRAKI fiyatlari yumusak bir surukleyle degistir (gunluk %2)."""
    out = frame.copy()
    d = pd.to_datetime(out.index.get_level_values("date"))
    later = dates_of(out)[dates_of(out) > t]
    rank = pd.Series(np.arange(1, len(later) + 1), index=later)
    mask = d > t
    factor = 1.02 ** rank.reindex(d[mask]).to_numpy()
    for c in ("open", "high", "low", "close"):
        out.loc[mask, c] = out.loc[mask, c].to_numpy() * factor
    return out


def wilder_reference(h, l, c, n=N):
    """Klasik Wilder (1978): SMA tohumu + (A*(n-1) + x)/n. Koddan BAGIMSIZ."""
    T = len(c)
    tr, pdm, mdm = (np.full(T, np.nan) for _ in range(3))
    for i in range(1, T):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    atr, sp, sm = (np.full(T, np.nan) for _ in range(3))
    atr[n], sp[n], sm[n] = tr[1:n + 1].mean(), pdm[1:n + 1].mean(), mdm[1:n + 1].mean()
    for i in range(n + 1, T):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
        sp[i] = (sp[i - 1] * (n - 1) + pdm[i]) / n
        sm[i] = (sm[i - 1] * (n - 1) + mdm[i]) / n
    pdi, mdi = 100 * sp / atr, 100 * sm / atr
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = np.full(T, np.nan)
    adx[2 * n - 1] = dx[n:2 * n].mean()
    for i in range(2 * n, T):
        adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx


def load_main(module_path, name):
    """main @ MAIN_REF'teki modul (git yoksa None)."""
    try:
        src = subprocess.run(["git", "show", f"{MAIN_REF}:{module_path}"], cwd=ROOT,
                             capture_output=True, timeout=30)
        if src.returncode != 0 or not src.stdout:
            return None
        path = os.path.join(_TMP, f"{name}_{MAIN_REF}.py")
        with open(path, "wb") as fh:
            fh.write(src.stdout)
        spec = importlib.util.spec_from_file_location(f"{name}_{MAIN_REF}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def frames_equal(a, b):
    a, b = a.sort_index(), b.sort_index()
    if not (a.index.equals(b.index) and list(a.columns) == list(b.columns)):
        return False
    return all(np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                              equal_nan=True) for c in a.columns)


def digest(env_cls, panel, n_steps=None, **kw):
    """Gozlem + odul + portfoy degeri parmak izi (sabit eylem dizisi)."""
    env = env_cls(panel, initial_balance=100_000, **kw)
    rng = np.random.default_rng(42)
    n = n_steps or (env.frame_bound[1] - env.frame_bound[0])
    acts = rng.uniform(-1, 1, size=(n, env.n_stocks)).astype(np.float32)
    h = hashlib.sha256()
    obs, _ = env.reset()
    h.update(obs.tobytes())
    for a in acts:
        obs, r, term, trunc, info = env.step(a)
        h.update(obs.tobytes())
        h.update(np.float64(r).tobytes())
        h.update(np.float64(info["portfolio_value"]).tobytes())
        if term or trunc:
            break
    return h.hexdigest()


def trajectory(env, actions):
    obs, _ = env.reset()
    out = [obs.copy()]
    for a in actions:
        obs, _, term, trunc, _ = env.step(a)
        out.append(obs.copy())
        if term or trunc:
            break
    return out


def holed(panel, sym="W1.IS", lo=60, hi=65):
    d = dates_of(panel)[lo:hi]
    idx = panel.index
    m = (idx.get_level_values("symbol") == sym) & idx.get_level_values("date").isin(d)
    return panel[~m]


def run_route(panel, seed=11, perturb=None, **req_kw):
    """_run_training_inner'i sentetik panelle kostur; env'leri ve gozlemlerini yakala."""
    import asyncio

    import data.data_fetcher as dfm
    import env.trading_env as tem
    from app.api.routes import trading as tr
    from app.schemas.trading import TrainingRequest

    created = []
    orig_make_env = tem.make_env

    def rec_make_env(df, **kw):
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
                              initial_balance=100_000, seed=seed, **req_kw)
        asyncio.run(tr._run_training_inner(req, state))
    finally:
        tem.make_env = orig_make_env
        dfm.DataFetcher.load_data = orig_load
    return state, created


def route_trial(state):
    from evaluation.trial_registry import load_trials
    tid = (state.get("metrics") or {}).get("trial_id")
    recs = [t for t in load_trials(directory=os.environ["RESULTS_DIR"], exclude_tags=())
            if t["trial_id"] == tid]
    return recs[0] if recs else None


# ---------------------------------------------------------------------------

def main():
    from app.core.config import get_settings
    get_settings.cache_clear()
    import evaluation.reproducibility as rep
    rep.SNAPSHOT_DIR = os.path.join(_TMP, "snapshots")

    import data.integrity as integ
    from data.technical_indicators import (
        ADX_MODES, TechnicalIndicators as TI, add_indicators_to_multi_symbol_df as add_ind,
        with_adx,
    )
    from env.trading_env import CAUSAL_MIN_PERIODS, CAUSAL_WINDOW, TradingEnv
    from evaluation.observation_health import (
        feature_stats, features_in_support, market_matrix, obs_health_failures, obs_health_ok,
        observation_health, price_in_support, support_failures,
    )
    from evaluation.trial_registry import (
        CLEAN_PROTOCOL, open_protocol_items, price_stats_source, rl_eval_protocol,
    )

    raw = ohlcv()
    d = dates_of(raw)
    w0 = raw.xs("W0.IS", level="symbol")

    # -----------------------------------------------------------------------
    print("\nA — ADX: kok neden (legacy) ve duzeltme (wilder)")
    leg = TI.calculate_adx(w0)
    check("legacy: tarih indeksli df'te RangeIndex'li sifir serisi (DI'lar hizalamadan NaN)",
          isinstance(leg.index, pd.RangeIndex) and bool((leg == 0).all()))
    check("legacy: 'nan' modunda TUMUYLE NaN, 'bfill' modunda sabit 0 (kayit)",
          bool(add_ind(raw, warmup="nan")["adx"].isna().all())
          and set(add_ind(raw, warmup="bfill")["adx"].unique()) == {0.0})
    check("varsayilan mod legacy (davranis dondurma)", ADX_MODES == ("legacy", "wilder"))
    wil = TI.calculate_adx(w0, mode="wilder")
    check("wilder: df'in indeksini tasiyor", wil.index.equals(w0.index))
    check(f"wilder: ilk 2n-1 = {2 * N - 1} satir NaN, sonrasi tanimli",
          bool(wil.iloc[:2 * N - 1].isna().all()) and not wil.iloc[2 * N - 1:].isna().any(),
          str(int(wil.isna().sum())))
    check("wilder: deger araligi [0, 100] ve sabit degil",
          float(wil.min()) >= 0 and float(wil.max()) <= 100 and wil.nunique() > 100)
    ref = wilder_reference(w0["high"].to_numpy(), w0["low"].to_numpy(), w0["close"].to_numpy())
    diff_ref = np.abs(wil.to_numpy() - ref)
    check("bagimsiz referans 1 (SMA tohumlu klasik Wilder): satir >= 400'de |fark| < 1e-6",
          float(np.nanmax(diff_ref[400:])) < 1e-6, f"{np.nanmax(diff_ref[400:]):.2e}")
    check("  referansla ilk tanimli satir ayni (2n-1)",
          int(np.argmax(~np.isnan(ref))) == 2 * N - 1)
    try:
        import pandas_ta as ta
        pt = ta.adx(w0["high"], w0["low"], w0["close"], length=N)
        col = [c for c in pt.columns if c.startswith("ADX_")][0]
        dpt = np.abs(wil.to_numpy() - pt[col].to_numpy())
        check("bagimsiz referans 2 (pandas_ta.adx): satir >= 400'de |fark| < 1e-6",
              float(np.nanmax(dpt[400:])) < 1e-6, f"{np.nanmax(dpt[400:]):.2e}")
    except ImportError:
        print("    pandas_ta yok: referans 2 atlandi")
    # Yon: yukselen tepeler/dipler -> +DI baskin, ADX yuksek
    up = pd.DataFrame({"high": np.arange(100, 160, 1.0) + 1, "low": np.arange(100, 160, 1.0) - 1,
                       "close": np.arange(100, 160, 1.0)},
                      index=pd.bdate_range("2021-01-04", periods=60))
    check("duz trend (her gun +1): ADX > 90", float(TI.calculate_adx(up, mode="wilder").iloc[-1]) > 90)
    flat = up.assign(high=101.0, low=99.0, close=100.0)
    check("yatay seri (yon hareketi yok): ADX = 0 (tanimsiz bolme 0)",
          float(TI.calculate_adx(flat, mode="wilder").iloc[-1]) == 0.0)
    try:
        TI.calculate_adx(w0, mode="yanlis")
        check("gecersiz mod reddediliyor", False)
    except ValueError:
        check("gecersiz mod reddediliyor", True)

    cut_ts = (3, 2 * N - 2, 2 * N - 1, 60, 300)

    def cut_ok(warmup, t):
        full = add_ind(raw, warmup=warmup, adx="wilder")
        cut = add_ind(raw[raw.index.get_level_values("date") <= d[t]], warmup=warmup, adx="wilder")
        sub = full[full.index.get_level_values("date") <= d[t]]
        return frames_equal(sub[["adx"]], cut[["adx"]])

    res = [cut_ok("nan", t) for t in cut_ts]
    check(f"B0: 'nan' + wilder ADX her t'de t'ye kadar bit-es {cut_ts}", all(res), str(res))
    res_bf = [cut_ok("bfill", t) for t in cut_ts]
    check("SIZINTI: 'bfill' + wilder ADX isinma icindeki t'de kaliyor (olcut bos degil)",
          not res_bf[0] and not res_bf[1], str(res_bf))
    pw = with_adx(raw)
    sym_ok = all(np.array_equal(pw.xs(s, level="symbol")["adx"].to_numpy(),
                                TI.calculate_adx(raw.xs(s, level="symbol"), mode="wilder").to_numpy(),
                                equal_nan=True) for s in ("W0.IS", "W1.IS", "W2.IS"))
    check("with_adx: sembol basina calculate_adx(wilder) ile ayni; diger sutunlar bit-es",
          sym_ok and all(pw[c].equals(raw[c]) for c in raw.columns) and pw.index.equals(raw.index))
    shuffled = raw.sample(frac=1.0, random_state=0)
    check("with_adx: satir sirasindan bagimsiz (sirasiz panelde de ayni deger)",
          frames_equal(with_adx(shuffled), pw))
    try:
        with_adx(raw, mode="legacy")
        check("with_adx yalnizca 'wilder'", False)
    except ValueError:
        check("with_adx yalnizca 'wilder'", True)

    # -----------------------------------------------------------------------
    print(f"\nD — varsayilan yol main @ {MAIN_REF} ile bit-es (git show)")
    small = ohlcv(n_days=330, late=(2, 40))
    main_ti = load_main("data/technical_indicators.py", "technical_indicators")
    main_int = load_main("data/integrity.py", "integrity")
    MainEnv = getattr(load_main("env/trading_env.py", "trading_env"), "TradingEnv", None)
    if main_ti is None or main_int is None or MainEnv is None:
        print("    git yok: D atlandi (birincil kanit main karsilastirmasi)")
    else:
        for wm in ("bfill", "nan"):
            check(f"gostergeler warmup='{wm}' (adx varsayilan) main ile bit-es",
                  frames_equal(add_ind(small, warmup=wm),
                               main_ti.add_indicators_to_multi_symbol_df(small, warmup=wm)))
        legacy_panel = add_ind(small)
        for drop in (False, True):
            a, ra = integ.prepare_rl_panel(legacy_panel, start="2021-01-01",
                                           drop_indicator_warmup=drop)
            b, rb = main_int.prepare_rl_panel(legacy_panel, start="2021-01-01",
                                              drop_indicator_warmup=drop)
            check(f"veri kapisi (legacy ADX panel, drop={drop}) main ile ayni panel + rapor",
                  frames_equal(a, b) and ra == rb)
        ind_panel = with_random_indicators(small)
        sd = dates_of(ind_panel)
        tr_part = before(ind_panel, sd[220])
        sup = TradingEnv(tr_part).get_price_stats()
        configs = {
            "tam / legacy": (ind_panel, {}),
            "tam / own_last": (ind_panel, dict(missing_data_mode="own_last")),
            "delikli / legacy": (holed(ind_panel), {}),
            "delikli / own_last": (holed(ind_panel), dict(missing_data_mode="own_last")),
            "faz 3": (ind_panel, dict(phase=3)),
            "ATR + Kelly": (ind_panel, dict(use_atr_sizing=True, use_kelly=True)),
            "ham OHLCV": (small, {}),
            "verilen olcek (B9)": (from_(ind_panel, sd[220]), dict(price_stats=sup)),
        }
        for name, (p, kw) in configs.items():
            check(f"env main ile bit-es: {name}", digest(TradingEnv, p, **kw) == digest(MainEnv, p, **kw))

    # -----------------------------------------------------------------------
    print("\nG — veri kapisi: wilder ADX isinmasi")
    late = ohlcv(n_days=200, late=(2, 60))
    csv_like = add_ind(late)                         # CSV gibi: bfill + legacy ADX
    gp = with_adx(csv_like)                          # rota: ADX yeniden hesap
    gated, rep_g = integ.prepare_rl_panel(gp, start="2021-01-01", drop_indicator_warmup=True)
    check("kapi: her sembolden ilk 2n-1 = 27 satir atildi (cci bfill blogu 20 bunun icinde)",
          rep_g["indicator_warmup_dropped"] == {"W0.IS": 27, "W1.IS": 27, "W2.IS": 27},
          str(rep_g["indicator_warmup_dropped"]))
    check("kapi sonrasi gosterge sutunlarinda NaN yok, ADX sabit degil",
          not gated[list(integ.BFILL_WARMUP_COLS)].isna().any().any() and gated["adx"].nunique() > 1)
    ref_nan = add_ind(late, warmup="nan", adx="wilder")
    valid = ref_nan[~ref_nan[["rsi", "cci", "adx"]].isna().any(axis=1)].index

    def subset(frame):
        return bool(frame.index.isin(valid).all())

    check("bagimsiz 'nan' referansi: kalan her satirda gostergeler nedensel olarak tanimli",
          subset(gated))
    old_cols = tuple(c for c in integ.BFILL_WARMUP_COLS if c != "adx")
    bf_wilder = add_ind(late, warmup="bfill", adx="wilder")     # ADX isinmasi GERIYE dolu
    g_ok, _ = integ.prepare_rl_panel(bf_wilder, start="2021-01-01", drop_indicator_warmup=True)
    check("'bfill' + wilder panel: dedektor ADX blogunu buluyor, kalan satirlar nedensel",
          subset(g_ok) and "adx" in integ.indicator_warmup_blocks(bf_wilder).get("W0.IS", {}))
    # MUTASYON: K18 oncesi sutun listesi. Hem dedektor (varsayilan argumani tanim aninda
    # baglanir — modul ozniteligini degistirmek yetmez) hem kapinin NaN taramasi.
    saved_cols, saved_fn = integ.BFILL_WARMUP_COLS, integ.indicator_warmup_blocks
    try:
        integ.BFILL_WARMUP_COLS = old_cols
        integ.indicator_warmup_blocks = lambda df, min_len=integ.WARMUP_MIN_LEN, cols=old_cols: \
            saved_fn(df, min_len=min_len, cols=cols)
        g_mut, _ = integ.prepare_rl_panel(bf_wilder, start="2021-01-01", drop_indicator_warmup=True)
    finally:
        integ.BFILL_WARMUP_COLS, integ.indicator_warmup_blocks = saved_cols, saved_fn
    check("MUTASYON: dedektor adx'i bilmezse geriye doldurulmus ADX satirlari kaliyor -> yakalandi",
          not subset(g_mut))
    proto_g = rl_eval_protocol(TradingEnv(from_(gated, dates_of(gated)[120])),
                               TradingEnv(before(gated, dates_of(gated)[120])))
    check("protokol: kapi sonrasi indicator_bfill False, adx_constant False",
          proto_g["indicator_bfill"] is False and proto_g["adx_constant"] is False, str(proto_g))

    # -----------------------------------------------------------------------
    print("\nN — K25 (b): nedensel fiyat olcegi")
    npan = with_random_indicators(ohlcv(n_days=420, drift=0.003, seed=5))   # TL gibi yukselen
    nd = dates_of(npan)
    n_tr, n_ev = before(npan, nd[300]), from_(npan, nd[300])
    ev_dates = dates_of(n_ev)
    e_c = TradingEnv(n_ev, price_norm="causal", price_context=n_tr, missing_data_mode="own_last")
    check("varsayilan price_norm 'stats'; get_price_norm kaydi",
          TradingEnv(n_tr).get_price_norm() == {"mode": "stats", "window": None, "min_periods": None}
          and e_c.get_price_norm() == {"mode": "causal", "window": CAUSAL_WINDOW,
                                       "min_periods": CAUSAL_MIN_PERIODS})
    check("baglam satirlari sembol basina sayiliyor", e_c.price_norm_context_rows ==
          {s: 300 for s in ("W0.IS", "W1.IS", "W2.IS")})
    mm = market_matrix(e_c)

    def ref_z(sym, k, window=CAUSAL_WINDOW, ctx=n_tr, ev=n_ev):
        c_all = np.concatenate([ctx.xs(sym, level="symbol")["close"].to_numpy(),
                                ev.xs(sym, level="symbol")["close"].to_numpy()])
        j = len(ctx.xs(sym, level="symbol")) + k
        w = c_all[max(0, j - window + 1): j + 1] if window else c_all[: j + 1]
        row = ev.xs(sym, level="symbol").iloc[k]
        m, s = w.mean(), w.std(ddof=1)
        return np.clip([(row[c] - m) / s for c in ("open", "high", "low", "close")], -10, 10)

    ok = all(np.allclose(mm[k, si, :4], ref_z(sym, k), rtol=0, atol=1e-4)
             for k in (0, 1, 50, len(ev_dates) - 1) for si, sym in enumerate(e_c.symbols))
    check("z = (p - ort(son 252 kapanis <= t)) / std(ddof=1), O/H/L/C ayni istatistik", ok)

    def ref_sigma(sym, k):
        c_all = np.concatenate([n_tr.xs(sym, level="symbol")["close"].to_numpy(),
                                n_ev.xs(sym, level="symbol")["close"].to_numpy()])
        j = len(n_tr.xs(sym, level="symbol")) + k
        return c_all[max(0, j - CAUSAL_WINDOW + 1): j + 1].std(ddof=1)

    ok_m = all(np.isclose(mm[k, si, 5], np.tanh(n_ev.xs(sym, level="symbol")["macd"].iloc[k]
                                                / ref_sigma(sym, k)), rtol=0, atol=1e-5)
               for k in (0, 50, len(ev_dates) - 1) for si, sym in enumerate(e_c.symbols))
    check("MACD (fiyat cinsinden) ayni nedensel olcekle: tanh(macd / std_t)", ok_m)
    e_s = TradingEnv(n_ev, price_stats=TradingEnv(n_tr).get_price_stats())
    check("  'stats' modunda MACD eski tanim tanh(macd / 0.1)",
          np.isclose(market_matrix(e_s)[3, 0, 5],
                     np.tanh(n_ev.xs("W0.IS", level="symbol")["macd"].iloc[3] / 0.1), atol=1e-6))
    e_x = TradingEnv(n_ev, price_norm="causal", price_norm_window=None, price_context=n_tr)
    mx = market_matrix(e_x)
    check("genisleyen pencere (window=None) bagimsiz hesapla ayni",
          all(np.allclose(mx[k, si, :4], ref_z(sym, k, window=None), rtol=0, atol=1e-4)
              for k in (0, 60) for si, sym in enumerate(e_x.symbols)))
    e_cold = TradingEnv(n_ev, price_norm="causal")
    mc = market_matrix(e_cold)
    check(f"baglamsiz: ilk {CAUSAL_MIN_PERIODS - 1} gun fiyat z'si notr 0, {CAUSAL_MIN_PERIODS}. gunden itibaren degil",
          bool((mc[:CAUSAL_MIN_PERIODS - 1, :, :4] == 0).all())
          and bool((mc[CAUSAL_MIN_PERIODS - 1, :, :4] != 0).all()))

    acts = np.random.default_rng(1).uniform(-1, 1, size=(len(ev_dates), 3)).astype(np.float32)
    t_vals = (0, 60, len(ev_dates) - 2)

    def b0(make, t_idx):
        t = ev_dates[t_idx]
        a = trajectory(make(n_ev), acts)
        b = trajectory(make(perturb_after(n_ev, t)), acts)
        upto = all(np.array_equal(a[i], b[i]) for i in range(t_idx + 1))
        after = not np.array_equal(a[t_idx + 1], b[t_idx + 1])
        return upto, after

    def causal_env(f):
        return TradingEnv(f, price_norm="causal", price_context=n_tr, missing_data_mode="own_last")

    res = [b0(causal_env, k) for k in t_vals]
    check(f"B0: t sonrasi fiyat degisince t'ye kadar TUM gozlem bit-es, t in {t_vals}",
          all(r[0] for r in res), str(res))
    check("  olcut bos degil: t+1 gozlemi degisiyor", all(r[1] for r in res), str(res))
    import env.trading_env as tem
    orig_build = tem.TradingEnv._build_causal_stats

    def centered(self, context):  # MUTASYON: merkezli pencere gelecek kapanislari kullanir
        out = orig_build(self, context)
        for sym in self.symbols:
            own = self._lookup_cache["values"][sym][:, 3]
            s = pd.Series(own).rolling(CAUSAL_WINDOW, min_periods=CAUSAL_MIN_PERIODS, center=True)
            out[sym] = np.column_stack([s.mean().to_numpy(), s.std().to_numpy()])
        return out

    tem.TradingEnv._build_causal_stats = centered
    try:
        res_m = b0(causal_env, 0)
    finally:
        tem.TradingEnv._build_causal_stats = orig_build
    check("MUTASYON: merkezli (gelecekli) pencere t=0'da yakalaniyor", not res_m[0])
    res_own = b0(lambda f: TradingEnv(f), 0)  # B9-legacy: olcum, eski davranis
    check("eski olcek (kendi penceresi) t=0'da kaliyor (Y13 hatirlatma)", not res_own[0])

    def raises(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    check("baglam degerlendirme penceresine tasarsa ValueError (ileriye bakis)",
          raises(lambda: TradingEnv(n_ev, price_norm="causal", price_context=npan)))
    check("baglam 'stats' modunda ValueError",
          raises(lambda: TradingEnv(n_ev, price_context=n_tr)))
    check("gecersiz price_norm / pencere < min_periods ValueError",
          raises(lambda: TradingEnv(n_ev, price_norm="kayan"))
          and raises(lambda: TradingEnv(n_ev, price_norm="causal", price_norm_window=5)))
    check("'close'suz baglam ve lookup cache'siz panel ValueError",
          raises(lambda: TradingEnv(n_ev, price_norm="causal", price_context=n_tr[["open"]]))
          and raises(lambda: TradingEnv(n_ev.drop(columns=["volume"]), price_norm="causal")))
    hol = holed(n_ev, sym="W1.IS", lo=40, hi=45)
    eh = TradingEnv(hol, price_norm="causal", price_context=n_tr, missing_data_mode="own_last")
    mh = market_matrix(eh)
    check("own_last eksik gun: sembol son satirinin degeri ve O SATIRIN istatistigi",
          all(np.array_equal(mh[k, 1, :4], mh[39, 1, :4]) for k in range(40, 45))
          and not np.array_equal(mh[45, 1, :4], mh[39, 1, :4]))

    tr_stats_env = TradingEnv(n_tr, missing_data_mode="own_last")
    e_b9 = TradingEnv(n_ev, price_stats=tr_stats_env.get_price_stats(), missing_data_mode="own_last")
    h_b9 = observation_health(tr_stats_env, e_b9)
    tr_causal_env = TradingEnv(n_tr, price_norm="causal", missing_data_mode="own_last")
    h_c = observation_health(tr_causal_env, e_c)
    check("yukselen panel: B9 olcegi destek DISI (price_in_support False)",
          price_in_support(h_b9) is False, str(h_b9["eval_out_of_train_support"]))
    check("yukselen panel: nedensel olcek destek ICI (price_in_support True)",
          price_in_support(h_c) is True, str(h_c["eval_out_of_train_support"]))

    # -----------------------------------------------------------------------
    print("\nP — protokol ve gozlem sagligi")
    w_tr = TradingEnv(n_tr, price_norm="causal", price_norm_window=63)
    over = TradingEnv(npan, price_norm="causal")
    check("price_stats_source: nedensel + nedensel -> 'causal'",
          price_stats_source(e_c, tr_causal_env) == "causal")
    check("  nedensel test + olcekli egitim -> 'mixed_norm'",
          price_stats_source(e_c, tr_stats_env) == "mixed_norm")
    check("  olcekli test + nedensel egitim -> 'mixed_norm'",
          price_stats_source(e_b9, tr_causal_env) == "mixed_norm")
    check("  pencereler farkli (252 / 63) -> 'mixed_norm'", price_stats_source(e_c, w_tr) == "mixed_norm")
    check("  egitim penceresi degerlendirmeyle kesisiyor -> 'overlapping_window'",
          price_stats_source(e_c, over) == "overlapping_window")
    check("  egitim ortami yok -> 'causal_unverified'", price_stats_source(e_c, None) == "causal_unverified")
    check("  olcekli yol degismedi (B9 -> 'train_window')",
          price_stats_source(e_b9, tr_stats_env) == "train_window")
    base_p = dict(CLEAN_PROTOCOL)
    check("open_protocol_items: 'causal' temiz, 'mixed_norm' acik",
          open_protocol_items({**base_p, "price_stats_source": "causal"}) == []
          and open_protocol_items({**base_p, "price_stats_source": "mixed_norm"}) == ["price_stats_source"])
    old_rec = {k: v for k, v in base_p.items() if k != "obs_health"}
    check("K18 oncesi kayit (obs_health yok) acik sayilir", open_protocol_items(old_rec) == ["obs_health"])

    check("obs_health: nedensel + rastgele gosterge -> True", obs_health_ok(h_c) is True,
          str(obs_health_failures(h_c)))
    check("ayirt edici: bu fikstürde B9 destek disi ama kirpma yok -> obs_health True "
          "(dagilim disiligi price_in_support olcer, obs_health kirpma/sabit boyut)",
          obs_health_ok(h_b9) is True and price_in_support(h_b9) is False,
          str(obs_health_failures(h_b9)))
    jump = n_ev.copy()
    for c in ("open", "high", "low", "close"):
        jump[c] = jump[c] * 6.0                      # rejim kirilmasi: fiyatlar egitimin 6 kati
    h_jump = observation_health(tr_stats_env, TradingEnv(
        jump, price_stats=tr_stats_env.get_price_stats(), missing_data_mode="own_last"))
    check("obs_health: B9 olcegi fiyat sicramasinda -> False (eval.price.clip)",
          obs_health_ok(h_jump) is False and "eval.price.clip" in obs_health_failures(h_jump),
          str(obs_health_failures(h_jump)))
    leg_full = add_ind(npan[list(raw.columns)])
    leg_tr, leg_ev = before(leg_full, nd[300]), from_(leg_full, nd[300])
    lt = TradingEnv(leg_tr, price_norm="causal")
    le = TradingEnv(leg_ev, price_norm="causal", price_context=leg_tr)
    h_leg = observation_health(lt, le)
    check("obs_health: legacy ADX (sabit) -> False (train/eval.indicators.constant)",
          obs_health_ok(h_leg) is False
          and {"train.indicators.constant", "eval.indicators.constant"} <= set(obs_health_failures(h_leg)),
          str(obs_health_failures(h_leg)))
    check("  ozellik basina: legacy adx boyutlarinin tamami sabit, digerleri degil",
          h_leg["train_features"]["adx"]["constant_dims_frac"] == 1.0
          and h_leg["train_features"]["rsi"]["constant_dims_frac"] == 0.0)
    pr_leg = rl_eval_protocol(TradingEnv(with_adx(leg_ev)), lt)
    check("adx_constant egitim ortamini da olcuyor (egitim legacy, test wilder -> True)",
          pr_leg["adx_constant"] is True)
    tm = market_matrix(TradingEnv(npan.assign(turbulence=50.0 + np.arange(len(npan)) % 7)))
    fs = feature_stats(tm)
    check("doygunluk raporlaniyor: tanh(turb/2), turb >= 50 -> sat_frac 1, kapi degil",
          fs["turbulence"]["sat_frac"] == 1.0 and fs["turbulence"]["clip_frac"] == 0.0)
    check("obs_health tek gunluk pencerede olculemez (None)",
          obs_health_ok({"train": {"price": {"clip_frac": 0.0, "constant_dims_frac": None}},
                         "eval": {}}) is None)
    # Bagimsiz rastgele gostergede test degerinin egitim [min, max] disinda kalma beklentisi
    # ~2/301 = %0,7 (gurultuyle %1'i asabilir). Temiz kurgu: test gostergeleri egitim
    # satirlarinin kopyasi (destek icinde kurgu geregi); fiyatlar yine yukselen.
    ev_sup = n_ev.copy()
    for sym in ev_sup.index.get_level_values("symbol").unique():
        m = ev_sup.index.get_level_values("symbol") == sym
        src = n_tr.xs(sym, level="symbol").iloc[: int(m.sum())]
        for col in ("macd", "rsi", "cci", "adx", "turbulence", "volume"):
            ev_sup.loc[m, col] = src[col].to_numpy()
    e_sup = TradingEnv(ev_sup, price_norm="causal", price_context=n_tr, missing_data_mode="own_last")
    pr_c = rl_eval_protocol(e_sup, tr_causal_env)
    check("rl_eval_protocol: nedensel kol, yukselen fiyat + destekte gostergeler -> tamamen temiz",
          open_protocol_items(pr_c) == [], str(open_protocol_items(pr_c)))
    pr_r = rl_eval_protocol(e_c, tr_causal_env, health=h_c)
    check("  bagimsiz rastgele gostergede features_in_support olculuyor (gurultu acabilir)",
          pr_r["features_in_support"] in (True, False)
          and open_protocol_items(pr_r) in ([], ["features_in_support"]), str(open_protocol_items(pr_r)))

    def fake_health(**oos):
        base = {f: 0.0 for f in ("open", "high", "low", "close", "volume", "macd", "rsi",
                                 "cci", "adx", "turbulence")}
        base.update(oos)
        return {"eval_out_of_train_support_features": base}

    check("features_in_support: fiyat disi her ozellik <= %1 -> True; fiyat sayilmaz",
          features_in_support(fake_health(close=0.9)) is True)
    check("  tek ozellik (macd %28,7) blok ortalamasina gizlenmez -> False",
          features_in_support(fake_health(macd=0.287)) is False
          and support_failures(fake_health(macd=0.287)) == ["macd"])
    check("  olcum yoksa None", features_in_support({}) is None)

    # B7 olcutunun yanlis pozitifi (K18'de gercek panelde bulundu: VAKBN RSI)
    gd = dates_of(gated)
    tr_fp, ev_fp = before(gated, gd[120]), from_(gated, gd[120]).copy()
    e0_, e1_ = dates_of(ev_fp)[:2]
    ev_fp.loc[("W0.IS", e1_), "rsi"] = ev_fp.loc[("W0.IS", e0_), "rsi"]   # dogal tekrar
    check("ayirt edici: dedektor pencere basinda W0 'blogu' goruyor (eski olcut bunu bfill sayardi)",
          "W0.IS" in integ.indicator_warmup_blocks(ev_fp))
    pr_fp = rl_eval_protocol(TradingEnv(ev_fp), TradingEnv(tr_fp))
    check("egitimde gecmisi olan sembolun pencere basi tekrari bfill sayilmiyor -> indicator_bfill False",
          pr_fp["indicator_bfill"] is False, str(pr_fp))
    late2 = add_ind(ohlcv(n_days=200, late=(2, 150)))        # W2 degerlendirmede listeleniyor
    l2d = dates_of(late2)
    tr_l2 = late2[(late2.index.get_level_values("date") >= l2d[30])
                  & (late2.index.get_level_values("date") < l2d[120])]
    ev_l2 = from_(late2, l2d[120])
    pr_l2 = rl_eval_protocol(TradingEnv(ev_l2), TradingEnv(tr_l2))
    pr_l2c = rl_eval_protocol(TradingEnv(ev_l2.drop(index="W2.IS", level="symbol")), TradingEnv(tr_l2))
    check("degerlendirmede listelenen sembolun bfill blogu hala yakalaniyor (True); o sembol "
          "cikinca False", pr_l2["indicator_bfill"] is True and pr_l2c["indicator_bfill"] is False,
          f"{pr_l2['indicator_bfill']} / {pr_l2c['indicator_bfill']}")

    # -----------------------------------------------------------------------
    print("\nR — rota uctan uca (_run_training_inner, sentetik panel, CPU)")
    from data.data_fetcher import DataFetcher

    rpan = with_random_indicators(ohlcv(n_days=180, drift=0.002, seed=9, start="2021-01-04"))
    state, created = run_route(rpan)
    check("varsayilan rota tamamlandi", state.get("state") == "completed", str(state.get("error")))
    if state.get("state") == "completed":
        gated_r, _ = integ.prepare_rl_panel(with_adx(rpan), start=get_settings().RL_DATA_START,
                                            drop_indicator_warmup=True)
        r_train, r_val, r_test = DataFetcher().split_data(gated_r)
        tr_env_r = created[0]["env"]
        check("egitim ortaminin ADX'i rotanin wilder hesabi (rastgele CSV ADX'i degil)",
              np.allclose(tr_env_r.df["adx"].to_numpy(), r_train["adx"].to_numpy(), equal_nan=False)
              and not np.allclose(tr_env_r.df["adx"].to_numpy(), rpan.loc[r_train.index, "adx"].to_numpy()))
        rec = route_trial(state)
        model = rec["extra"]["model_name"] if rec else None
        meta = json.load(open(os.path.join(os.environ["MODELS_DIR"], f"{model}.meta.json"),
                              encoding="utf-8")) if model else {}
        check("meta: indicator_spec adx='wilder', price_norm 'stats'",
              (meta.get("indicator_spec") or {}).get("adx") == "wilder"
              and (meta.get("price_norm") or {}).get("mode") == "stats", str(meta.get("indicator_spec")))
        check("kayit: adx_constant False, obs_health olculmus, config gozlem tanimini tasiyor",
              rec["protocol"]["adx_constant"] is False and rec["protocol"]["obs_health"] in (True, False)
              and rec["config"].get("indicator_spec", {}).get("adx") == "wilder"
              and rec["config"].get("price_norm", {}).get("mode") == "stats", str(rec and rec["protocol"]))

    state_c, created_c = run_route(rpan, price_norm="causal")
    check("causal rota tamamlandi", state_c.get("state") == "completed", str(state_c.get("error")))
    if state_c.get("state") == "completed":
        test_c = created_c[-1]
        r_tdates = dates_of(test_c["env"].df)
        check("test ortami causal; baglam = egitim + dogrulama satirlari (testten once)",
              test_c["env"].price_norm == "causal"
              and dates_of(test_c["kw"]["price_context"]).max() < r_tdates[0]
              and min(test_c["env"].price_norm_context_rows.values())
              == len(dates_of(test_c["kw"]["price_context"])))
        rec_c = route_trial(state_c)
        check("kayit: price_stats_source='causal'; meta ve config price_norm causal",
              rec_c["protocol"]["price_stats_source"] == "causal"
              and rec_c["config"]["price_norm"]["mode"] == "causal", str(rec_c["protocol"]))
        n_s, fps = test_c["env"].n_stocks, test_c["env"].features_per_stock
        for k in (0, 6):
            t = r_tdates[k]
            st_p, cr_p = run_route(perturb_after(rpan, t), price_norm="causal")
            tp = cr_p[-1] if st_p.get("state") == "completed" else None
            if tp is None:
                check(f"rota B0 t={k}: kosum tamamlandi", False, str(st_p.get("error")))
                continue
            mk = [o[1 + n_s: 1 + n_s + n_s * fps] for o in test_c["obs"]]
            mp = [o[1 + n_s: 1 + n_s + n_s * fps] for o in tp["obs"]]
            upto = all(np.array_equal(mk[i], mp[i]) for i in range(k + 1))
            after = not np.array_equal(mk[k + 1], mp[k + 1])
            check(f"rota B0 t={k}: test piyasa blogu t'ye kadar bit-es, t+1'de degisiyor",
                  upto and after, f"upto={upto} after={after}")

    # -----------------------------------------------------------------------
    print("\nX — gercek panel (CSV varsa)")
    try:
        real = DataFetcher().load_data("stock_data_with_indicators.csv")
    except FileNotFoundError:
        real = None
        print("    stock_data_with_indicators.csv yok: X atlandi")
    if real is not None:
        rw = with_adx(real)
        nan_per = rw["adx"].isna().groupby(level="symbol").sum()
        check("gercek panel: her sembolde yalniz ilk 2n-1 = 27 satir NaN",
              bool((nan_per == 2 * N - 1).all()), str(nan_per[nan_per != 2 * N - 1].to_dict()))
        a = rw["adx"].dropna()
        check("gercek panel: ADX [0, 100] ve sembol basina sabit degil",
              float(a.min()) >= 0 and float(a.max()) <= 100
              and int(a.groupby(level="symbol").nunique().min()) > 1000)
        rg, rrep = integ.prepare_rl_panel(rw, start=get_settings().RL_DATA_START,
                                          drop_indicator_warmup=True)
        drops = rrep["indicator_warmup_dropped"]
        print(f"    kapi atimi (RL penceresi): {drops}")
        check("gercek panel: kapi yalniz pencerede isinmasi olan sembollerden atiyor "
              f"{EXPECTED_REAL_DROPS}", drops == EXPECTED_REAL_DROPS, str(drops))
        calc = TI()
        sub_ok = []
        for sym in drops:
            ref_s = calc.add_all_indicators(real.xs(sym, level="symbol")[list(raw.columns)],
                                            warmup="nan", adx="wilder")
            valid_d = ref_s.index[~ref_s[["rsi", "cci", "adx"]].isna().any(axis=1)]
            sub_ok.append(bool(rg.xs(sym, level="symbol").index.isin(valid_d).all()))
        check("gercek panel: o sembollerde kalan her satirda gostergeler nedensel tanimli "
              "(sembol basina bagimsiz 'nan' + wilder referansi)", all(sub_ok) and bool(sub_ok),
              str(sub_ok))

    print("\n" + "=" * 60)
    print(f"Sonuc: {passed} gecti, {failed} kaldi")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
