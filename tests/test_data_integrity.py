"""
Faz 10 / V — veri butunlugu kapisi (data/integrity.py) ve B0 veri sagligi testi.

  1. Sentetik panelde her artefakt turu, hayalet gun ve bayat hucre YAKALANIYOR.
  2. prepare_rl_panel: pencere icinde artefakt -> DataIntegrityError (sessiz
     temizlik yok); pencere disindaki artefakt egitimi durdurmaz.
  3. Gercek panel (data/bist/stock_data_with_indicators.csv varsa):
     K17 penceresi (2013-06) artefaktsiz, hayalet gun kalmiyor, 30 sembol.
     Sabit sutunlar BILINEN acik olarak sabitlenir: ADX tum panelde 0
     (Faz 10 K18, bu kapida duzeltilmedi). ADX duzeltilince bu kontrol kalir
     ve guncellenmesi gerekir — bu kasitli.

Calistirma:
    python tests/test_data_integrity.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.integrity import (  # noqa: E402
    RL_DATA_START, DataIntegrityError, activity_mask, find_ghost_days,
    find_stale_cells, health_report, prepare_rl_panel, scan_price_artifacts,
)

logging.disable(logging.CRITICAL)

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


def make_panel(n_days=120, n_sym=4, start="2014-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 50.0 * (s + 1)
        for d in dates:
            p *= 1 + rng.normal(0, 0.01)
            o = p * (1 + rng.normal(0, 0.002))
            rows.append({"symbol": f"S{s}.IS", "date": d, "open": o,
                         "high": max(o, p) * 1.005, "low": min(o, p) * 0.995,
                         "close": p, "volume": 1e6})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def set_row(df, sym, date, **vals):
    for k, v in vals.items():
        df.loc[(sym, pd.Timestamp(date)), k] = v


print("\n1) Sentetik panel — tarayici")
df = make_panel()
clean_art = scan_price_artifacts(df)
check("temiz panelde artefakt yok", len(clean_art) == 0, str(clean_art.head()))

dates = df.index.get_level_values("date").unique()
bad = df.copy()
set_row(bad, "S0.IS", dates[30], close=-1.0)                                     # nonpositive
set_row(bad, "S1.IS", dates[40], high=1.0)                                       # ohlc
c = bad.loc[("S2.IS", dates[59]), "close"]
set_row(bad, "S2.IS", dates[60], close=c * 3, open=c * 3, high=c * 3.1, low=c * 2.9)  # extreme_return
for d in dates[80:]:                                                             # seviye sicramasi
    for col in ("open", "high", "low", "close"):
        bad.loc[("S3.IS", d), col] *= 1000
reasons = set(scan_price_artifacts(bad)["reason"])
for r in ("nonpositive_price", "ohlc_inconsistent", "extreme_return", "level_jump"):
    check(f"'{r}' yakalaniyor", r in reasons, str(reasons))

red = make_panel(n_days=60, start="2004-12-01")
d_red = red.index.get_level_values("date").unique()
for d in d_red[25:]:
    for col in ("open", "high", "low", "close"):
        red.loc[("S0.IS", d), col] /= 1e6 / 1000     # 1000 kat dusus, YTL penceresinde
check("2005 YTL gecisi 'redenomination' olarak isaretleniyor",
      "redenomination" in set(scan_price_artifacts(red)["reason"]))

print("\n2) Hayalet gun ve bayat hucre")
g = make_panel()
gd = g.index.get_level_values("date").unique()
ghost = gd[50]
for s in g.index.get_level_values("symbol").unique():
    prev = g.loc[(s, gd[49]), "close"]
    set_row(g, s, ghost, open=prev, high=prev, low=prev, close=prev, volume=0)
stale_day = gd[70]
prev = g.loc[("S1.IS", gd[69]), "close"]
set_row(g, "S1.IS", stale_day, open=prev, high=prev, low=prev, close=prev, volume=0)

ghosts = find_ghost_days(g)
check("hayalet gun bulundu (tum semboller duz)", list(ghosts) == [ghost], str(list(ghosts)))
st = find_stale_cells(g, ghosts)
check("tek sembolun duz satiri bayat hucre",
      len(st) == 1 and st.iloc[0]["symbol"] == "S1.IS" and st.iloc[0]["date"] == stale_day, str(st))

out, rep = prepare_rl_panel(g, start="2014-01-01")
od = out.index.get_level_values("date")
check("prepare: hayalet gun atildi", ghost not in set(od) and rep["n_ghost_days"] == 1)
check("prepare: bayat hucre atildi, diger semboller o gun duruyor",
      ("S1.IS", stale_day) not in out.index and ("S0.IS", stale_day) in out.index)
check("rapor sayilari", rep["n_stale_cells_dropped"] == 1 and rep["n_symbols"] == 4)
mask = activity_mask(out)
check("aktiflik maskesi bayat hucrede False", not mask.loc[stale_day, "S1.IS"] and mask.loc[stale_day, "S0.IS"])
check("maske sutun sirasi paneldeki gibi (env sembol sirasi)",
      list(mask.columns) == list(dict.fromkeys(out.index.get_level_values("symbol"))))

print("\n3) Pencere ve sessiz temizlik yok")
w = make_panel(start="2013-01-01", n_days=200)
wd = w.index.get_level_values("date").unique()
set_row(w, "S0.IS", wd[5], close=-5.0)                 # pencere DISINDA (2013-01)
_, rep_w = prepare_rl_panel(w, start="2013-06-01")
check("pencere disindaki artefakt egitimi durdurmuyor", rep_w["n_artifacts"] == 0)
set_row(w, "S0.IS", wd[150], close=-5.0)               # pencere ICINDE
try:
    prepare_rl_panel(w, start="2013-06-01")
    check("pencere icindeki artefakt -> DataIntegrityError", False)
except DataIntegrityError as exc:
    check("pencere icindeki artefakt -> DataIntegrityError", "nonpositive_price" in str(exc))
check("varsayilan pencere K17 = 2013-06-01", RL_DATA_START == "2013-06-01")
from app.core.config import get_settings  # noqa: E402
check("config RL_DATA_START tek kaynakla ayni", get_settings().RL_DATA_START == RL_DATA_START)

print("\n4) Gercek panel (B0 veri sagligi)")
path = os.path.join("data", "bist", "stock_data_with_indicators.csv")
if not os.path.exists(path):
    print("    (panel yok — atlandi)")
else:
    from data.data_fetcher import DataFetcher
    panel = DataFetcher().load_data("stock_data_with_indicators.csv")
    full = health_report(panel)
    print(f"    tam panel: {full['start']}..{full['end']}, artefakt {full['artifacts']}, "
          f"hayalet gun {full['n_ghost_days']}, bayat hucre {full['n_stale_cells']}, "
          f"eksik hucre %{full['missing_cell_frac'] * 100:.2f}")
    check("tam panelde artefakt var (2000-2005) — tarayici calisiyor", full["n_artifacts"] > 0)
    rl, rep = prepare_rl_panel(panel)
    h = health_report(rl)
    print(f"    RL penceresi: {h['start']}..{h['end']}, {h['rows']:,} satir, hayalet gun atilan "
          f"{rep['n_ghost_days']}, bayat hucre atilan {rep['n_stale_cells_dropped']}, "
          f"eksik hucre %{h['missing_cell_frac'] * 100:.3f}")
    check("RL penceresinde artefakt yok", h["n_artifacts"] == 0, str(h["artifacts"]))
    check("RL penceresinde hayalet gun kalmadi", h["n_ghost_days"] == 0)
    check("RL penceresinde 30 sembol", h["n_symbols"] == 30, str(h["n_symbols"]))
    check("RL penceresi 2013-06'dan once baslamiyor", h["start"] >= RL_DATA_START)
    check("sabit sutunlar = {'adx'} (BILINEN acik, K18 — duzeltilince guncelle)",
          h["constant_columns"] == ["adx"], str(h["constant_columns"]))

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
