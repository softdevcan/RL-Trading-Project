"""
Faz 9 / B7 (Faz 10 Y15) — gosterge isinma boslugu bfill ile doldurulmaz.

Kabul olcutu (Faz 10 §3.1.1 B7): "B0 gostergeler ve OHLCV icin gecer" — ozellikler
tam seride ve t'de kesilmis seride hesaplaninca t'ye kadarki degerler bit-es.

  B0  add_indicators_to_multi_symbol_df(warmup='nan'): her t'de bit-es (NaN = NaN).
      Eski davranis (warmup='bfill') isinma icindeki t'de KALIR: sizinti sayisal.
  D   varsayilan yol (warmup='bfill', clean_data leading='bfill') main @ MAIN_REF ile
      bit-es (git show ile yuklenen eski modul, ayni kutuphaneler).
  C   clean_data(leading='drop'): bas NaN satirlara gelecek yazilmaz.
  G   prepare_rl_panel(drop_indicator_warmup=True): bfill imzali bas blok atilir;
      atilan satirlar 'nan' modunun NaN isinma satirlarini KAPSAR (en fazla 1 fazla).
  P   protokol `indicator_bfill` OLCULUR: kapi blogu atinca False, atmayinca True.
      Mutasyon: kapinin dedektoru bos donerse olcut yakalar.
  R   gercek panel (varsa): RL penceresinde atilan satirlar ve kalan blok yok.

Calistirma:
    python tests/test_indicator_warmup.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

MAIN_REF = "90d50f3"  # B7 oncesi main
IND = ("macd", "macd_signal", "macd_hist", "rsi", "cci", "adx", "turbulence")
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    passed, failed = (passed + 1, failed) if cond else (passed, failed + 1)
    print(f"  [{'OK' if cond else 'FAIL'}]   {name}" + ("" if cond else f"  {detail}"))


def ohlcv(n_days=140, n_sym=3, seed=3, late=None):
    """late=(sembol_indeksi, ilk_gun): o sembol sonradan listelenir."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 30.0 * (s + 1)
        for i, d in enumerate(dates):
            p *= 1 + rng.normal(0.0005, 0.015)
            if late and s == late[0] and i < late[1]:
                continue
            o = p * (1 + rng.normal(0, 0.003))
            rows.append({"symbol": f"W{s}.IS", "date": d, "open": o,
                         "high": max(o, p) * 1.004, "low": min(o, p) * 0.996, "close": p,
                         "volume": float(rng.integers(400_000, 900_000))})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def dates_of(frame):
    return pd.DatetimeIndex(sorted(frame.index.get_level_values("date").unique()))


def load_main(module_path, name):
    try:
        src = subprocess.run(["git", "show", f"{MAIN_REF}:{module_path}"], cwd=ROOT,
                             capture_output=True, timeout=30)
        if src.returncode != 0:
            return None
        path = os.path.join(tempfile.mkdtemp(prefix="b7_main_"), f"{name}.py")
        open(path, "wb").write(src.stdout)
        spec = importlib.util.spec_from_file_location(f"{name}_{MAIN_REF}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def frames_equal(a, b, cols):
    a, b = a.sort_index(), b.sort_index()
    return a.index.equals(b.index) and all(
        np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True)
        for c in cols)


def main():
    from data.data_fetcher import DataFetcher
    import data.integrity as integ
    from data.technical_indicators import add_indicators_to_multi_symbol_df as add_ind

    raw = ohlcv()
    d = dates_of(raw)
    cut_ts = (3, 10, 18, 25, 60, 120)  # isinma ici (cci 20, macd 26) ve sonrasi

    def cut_invariant(warmup, t):
        full = add_ind(raw, warmup=warmup)
        cut = add_ind(raw[raw.index.get_level_values("date") <= d[t]], warmup=warmup)
        sub = full[full.index.get_level_values("date") <= d[t]]
        return frames_equal(sub, cut, IND)

    print("\nB0 — gosterge kesme degismezligi")
    res_nan = [cut_invariant("nan", t) for t in cut_ts]
    check(f"warmup='nan': her t'de t'ye kadar bit-es {cut_ts}", all(res_nan), str(res_nan))
    res_bf = [cut_invariant("bfill", t) for t in cut_ts]
    check("SIZINTI: warmup='bfill' isinma icindeki t'de kaliyor (olcut bos degil)",
          not all(res_bf[:3]), str(res_bf))
    nanp = add_ind(raw, warmup="nan")
    w0 = nanp.xs("W0.IS", level="symbol")
    check("warmup='nan': cci'nin ilk 19 satiri NaN, sonra degil",
          bool(w0["cci"].iloc[:19].isna().all()) and not w0["cci"].iloc[19:].isna().any())
    try:
        add_ind(raw, warmup="yanlis")
        check("gecersiz warmup reddediliyor", False)
    except ValueError:
        check("gecersiz warmup reddediliyor", True)

    print("\nD — varsayilan yol main ile bit-es (git show)")
    ti_main = load_main("data/technical_indicators.py", "technical_indicators")
    df_main = load_main("data/data_fetcher.py", "data_fetcher")
    holey = raw.copy()
    holey.loc[("W1.IS", d[:3]), :] = np.nan  # bas NaN satirlar (clean_data icin)
    if ti_main is not None and df_main is not None:
        check("add_indicators varsayilani main ile bit-es",
              frames_equal(add_ind(raw), ti_main.add_indicators_to_multi_symbol_df(raw), IND))
        check("clean_data varsayilani main ile bit-es",
              frames_equal(DataFetcher().clean_data(holey), df_main.DataFetcher().clean_data(holey),
                           ["open", "high", "low", "close", "volume"]))
    else:
        print("    git yok: main karsilastirmasi atlandi")

    print("\nC — clean_data bas satirlari")
    f = DataFetcher()
    leg = f.clean_data(holey)
    new = f.clean_data(holey, leading="drop")
    w1_leg, w1_new = leg.xs("W1.IS", level="symbol"), new.xs("W1.IS", level="symbol")
    check("leading='bfill' (eski): bas satirlara ilk gecerli (gelecek) kapanis yaziliyor",
          bool((w1_leg["close"].iloc[:3] == raw.xs("W1.IS", level="symbol")["close"].iloc[3]).all()))
    check("leading='drop': bas NaN satirlar atiliyor, gerisi ayni",
          len(w1_new) == len(w1_leg) - 3 and w1_new.equals(w1_leg.iloc[3:]))

    print("\nG — RL kapisi isinma blogunu atiyor")
    late_raw = ohlcv(late=(2, 40))
    panel = add_ind(late_raw, warmup="bfill")  # eski CSV gibi
    nan_panel = add_ind(late_raw, warmup="nan")
    start = str(dates_of(panel)[30].date())  # W2 listelenmeden once baslayan pencere
    kept, rep_keep = integ.prepare_rl_panel(panel, start=start)
    dropped, rep_drop = integ.prepare_rl_panel(panel, start=start, drop_indicator_warmup=True)
    w2_nan = nan_panel.xs("W2.IS", level="symbol")
    # ADX 'nan' modunda TUMUYLE NaN (K18'in koku: hesap NaN uretiyor, eski yol 0'a
    # ceviriyordu) — isinma tanimina yalnizca deger ureten gostergeler girer
    adx_all_nan = bool(nan_panel["adx"].isna().all())
    check("K18 kaydi: warmup='nan' modunda ADX tumuyle NaN (eski yolda fillna(0) -> sabit 0)",
          adx_all_nan and bool((panel["adx"] == 0).all()))
    live = [c for c in IND if w2_nan[c].notna().any()]
    nan_rows = set(w2_nan.index[w2_nan[live].isna().any(axis=1)])
    gone = set(kept.xs("W2.IS", level="symbol").index) - set(dropped.xs("W2.IS", level="symbol").index)
    check("atilan W2 satirlari 'nan' modunun isinma satirlarini kapsiyor (en fazla 1 fazla)",
          nan_rows <= gone and len(gone - nan_rows) <= 1, f"nan={len(nan_rows)} atilan={len(gone)}")
    check("rapor atilani sayiyor ve yalnizca gec listelenen sembol etkileniyor",
          rep_drop["indicator_warmup_dropped"] == {"W2.IS": len(gone)}, str(rep_drop["indicator_warmup_dropped"]))
    check("kapi sonrasi pencerede bfill blogu kalmiyor",
          integ.indicator_warmup_blocks(dropped, min_len=2) == {})
    check("varsayilan (drop yok) eski davranis: hicbir satir atilmiyor",
          len(kept) == rep_keep["rows_after"] and rep_keep["indicator_warmup_dropped"] == {})
    late_start = str(dates_of(panel)[80].date())  # blok bittikten sonra baslayan pencere
    _, rep_after = integ.prepare_rl_panel(panel, start=late_start, drop_indicator_warmup=True)
    check("blok pencereden once bitiyorsa hicbir sey atilmiyor", rep_after["indicator_warmup_dropped"] == {})

    print("\nP — protokol indicator_bfill olculuyor")
    from env.trading_env import TradingEnv
    from evaluation.trial_registry import rl_eval_protocol

    def protocol_for(frame):
        dd = dates_of(frame)
        tr = frame[frame.index.get_level_values("date") <= dd[len(dd) * 2 // 3]]
        ev = frame[frame.index.get_level_values("date") > dd[len(dd) * 2 // 3]]
        tr_env = TradingEnv(tr, initial_balance=100_000, missing_data_mode="own_last")
        ev_env = TradingEnv(ev, initial_balance=100_000, missing_data_mode="own_last",
                            price_stats=tr_env.get_price_stats())
        return rl_eval_protocol(ev_env, tr_env)

    check("kapi blogu atti -> indicator_bfill False", protocol_for(dropped)["indicator_bfill"] is False)
    check("kapi blogu atmadi -> indicator_bfill True (egitim verisinde blok)",
          protocol_for(kept)["indicator_bfill"] is True)
    orig = integ.indicator_warmup_blocks
    try:
        integ.indicator_warmup_blocks = lambda *a, **k: {}  # MUTANT: dedektor kor
        mutant, _ = integ.prepare_rl_panel(panel, start=start, drop_indicator_warmup=True)
    finally:
        integ.indicator_warmup_blocks = orig
    check("mutasyon: dedektoru kor kapi olcutu kiriyor (protokol True)",
          protocol_for(mutant)["indicator_bfill"] is True)

    print("\nR — gercek panel")
    try:
        real = DataFetcher().load_data("stock_data_with_indicators.csv")
    except FileNotFoundError:
        real = None
    if real is None:
        print("    panel yok: atlandi")
    else:
        from app.core.config import get_settings
        rp, rrep = integ.prepare_rl_panel(real, start=get_settings().RL_DATA_START,
                                          drop_indicator_warmup=True)
        print(f"    atilan isinma satiri: {rrep['indicator_warmup_dropped']}")
        check("gercek panel: yalnizca gec listelenen semboller etkileniyor (ODAS, PGSUS)",
              set(rrep["indicator_warmup_dropped"]) <= {"ODAS.IS", "PGSUS.IS"}
              and len(rrep["indicator_warmup_dropped"]) > 0)
        check("gercek panel: kapi sonrasi RL penceresinde bfill blogu yok",
              integ.indicator_warmup_blocks(rp, min_len=2) == {},
              str(integ.indicator_warmup_blocks(rp, min_len=2)))

    print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
