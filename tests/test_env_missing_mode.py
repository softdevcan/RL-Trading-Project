"""
Faz 10 / V — TradingEnv eksik hucre modu (`missing_data_mode`).

legacy   : eski davranis BIREBIR korunur (egitilmis modeller). Kanit olarak
           hatanin kendisi de sabitlenir: eksik hucrede ONCEKI SEMBOLUN blogu
           kopyalaniyor, degerlemede fiyat 0.
own_last : sembol kendi son gozlemiyle tasinir; hic gozlemi yoksa sifir blok;
           degerleme kendi son kapanisiyla; o hucrede islem YOK.
Panel tam oldugunda iki mod bit-es gozlem uretir (fark yalnizca eksik hucrede).
Iki yol (lookup cache ve .loc) ayni sonucu vermeli.

Calistirma:
    python tests/test_env_missing_mode.py
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.trading_env import TradingEnv  # noqa: E402

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


def panel(n_days=30, n_sym=3, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 100.0 * (s + 1)
        for d in dates:
            p *= 1 + rng.normal(0, 0.01)
            rows.append({"symbol": f"T{s}.IS", "date": d, "open": p, "high": p * 1.01,
                         "low": p * 0.99, "close": p, "volume": 1e6 * (s + 1),
                         "macd": 0.01 * s, "rsi": 40 + 10 * s, "cci": 5.0 * s,
                         "adx": 20.0, "turbulence": 0.1})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


def env_at(df, step, mode, no_cache=False):
    e = TradingEnv(df, initial_balance=100_000, missing_data_mode=mode)
    if no_cache:
        e._lookup_cache = None
    e.reset()
    e.current_step = step
    return e


def block(obs, i, n):
    start = 1 + n + i * 10
    return obs[start:start + 10]


full = panel()
dates = full.index.get_level_values("date").unique()
n = 3

print("\n1) Tam panelde iki mod bit-es")
same = True
for step in range(len(dates)):
    a = env_at(full, step, "legacy")._get_observation()
    b = env_at(full, step, "own_last")._get_observation()
    same &= np.array_equal(a, b)
check("tum adimlarda gozlem bit-es", same)

print("\n2) Eksik hucre: legacy kopyalar, own_last kendi gecmisini tasir")
gap_day = dates[10]
holed = full.drop(index=("T1.IS", gap_day))
for no_cache in (False, True):
    tag = ".loc yolu" if no_cache else "cache yolu"
    leg = env_at(holed, 10, "legacy", no_cache)._get_observation()
    own = env_at(holed, 10, "own_last", no_cache)._get_observation()
    ref_prev = env_at(full, 9, "legacy", no_cache)._get_observation()
    check(f"[{tag}] legacy: T1 blogu = T0 blogu (bilinen hata, dondurulmus)",
          np.array_equal(block(leg, 1, n), block(leg, 0, n)))
    # own_last'ta T1 blogu, T1'in 9. gundeki satirinin bu env'in istatistikleriyle olceklenmis hali
    e_full_stats = env_at(holed, 9, "own_last", no_cache)._get_observation()
    check(f"[{tag}] own_last: T1 blogu = T1'in onceki gunku blogu",
          np.array_equal(block(own, 1, n), block(e_full_stats, 1, n)))
    check(f"[{tag}] own_last: diger semboller etkilenmiyor",
          np.array_equal(block(own, 0, n), block(leg, 0, n)) and np.array_equal(block(own, 2, n), block(leg, 2, n)))

print("\n3) Henuz listelenmemis sembol: sifir blok, komsu kopyasi yok")
late = full.drop(index=[("T2.IS", d) for d in dates[:5]])
own = env_at(late, 2, "own_last")._get_observation()
leg = env_at(late, 2, "legacy")._get_observation()
check("own_last: listelenmeden once sifir blok", np.all(block(own, 2, n) == 0.0))
check("legacy: listelenmeden once T1 blogu kopyalaniyor (dondurulmus)",
      np.array_equal(block(leg, 2, n), block(leg, 1, n)))

print("\n4) Degerleme ve islem kapisi")
for mode in ("legacy", "own_last"):
    e = TradingEnv(holed, initial_balance=100_000, missing_data_mode=mode)
    e.reset()
    e.current_step = 9
    e.shares_owned[1] = 10
    p9 = e._get_current_price("T1.IS")
    e.current_step = 10
    e._price_memo_step = -1
    p10 = e._get_current_price("T1.IS")
    ok, _ = e._execute_trade(1, 5)
    if mode == "legacy":
        check("legacy: eksik gunde fiyat 0 (dondurulmus)", p10 == 0.0)
    else:
        check("own_last: eksik gunde fiyat = kendi son kapanisi", p10 == p9 and p10 > 0)
        check("own_last: eksik gunde islem yapilmiyor", not ok and e.shares_owned[1] == 10)
        pv = e._get_portfolio_value()
        check("own_last: portfoy degeri pozisyonu kapsiyor", np.isclose(pv, e.balance + 10 * p9))

e = TradingEnv(holed, initial_balance=100_000, missing_data_mode="own_last")
e.reset()
e.current_step = 11
ok, _ = e._execute_trade(1, 5)
check("own_last: satir olan gunde islem normal", ok and e.shares_owned[1] == 5)

print("\n5) Bolum sonuna kadar kosum")
for mode in ("legacy", "own_last"):
    e = TradingEnv(holed, initial_balance=100_000, missing_data_mode=mode)
    obs, _ = e.reset()
    rng = np.random.default_rng(0)
    done = False
    finite = True
    while not done:
        obs, r, done, _, _ = e.step(rng.uniform(-1, 1, n))
        finite &= bool(np.all(np.isfinite(obs)) and np.isfinite(r))
    check(f"{mode}: bolum tamamlandi, gozlem/odul sonlu", finite)

try:
    TradingEnv(full, missing_data_mode="bilinmeyen")
    check("gecersiz mod reddediliyor", False)
except ValueError:
    check("gecersiz mod reddediliyor", True)

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
