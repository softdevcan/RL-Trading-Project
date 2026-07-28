"""
TradingEnv fiyat butunlugu korumalari

Bu testler HPO'yu cokerten gercek bir hatanin regresyonudur:

  raw_stock_data.csv icinde 8.155 negatif fiyat satiri vardi (yfinance'in 2005
  TL sadelestirmesi oncesine ait duzeltme artefaktlari). HPO bu dosyayi
  clean_data() cagirmadan yukluyordu. Negatif fiyatta:
    BUY  -> cost < 0  -> `cost <= balance` her zaman gecer -> balance ARTAR (yoktan para)
    SELL -> revenue < 0 -> balance sinirsiz DUSER
  Bakiye -initial_balance'in altina inince _get_observation'daki np.log()
  NaN uretiyor, NaN politika agina yayilip SB3'u
  "Expected parameter loc ... Normal(loc: nan)" ile cokertiyordu.

Ayrica fiyati 0 olan (veri yok) satirlarda eski env BEDAVA HISSE veriyordu:
normal egitim panelinde islemlerin ~%32'si bu sekildeydi.

Calistirma:
    python tests/test_env_price_guards.py
"""

import logging
import os
import sys
import warnings

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


def make_panel(n_days=400, n_symbols=3, seed=42, corrupt=None):
    """corrupt: {(sembol_idx, gun_idx): fiyat} ile belirli satirlari bozar."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for s in range(n_symbols):
        sym = f"T{s}.IS"
        price = 100.0 * (s + 1)
        for di, d in enumerate(dates):
            price = max(1.0, price * (1 + rng.normal(0, 0.02)))
            p = price
            if corrupt and (s, di) in corrupt:
                p = corrupt[(s, di)]
            rows.append({
                'symbol': sym, 'date': d,
                'open': p, 'high': p, 'low': p, 'close': p,
                'volume': float(rng.integers(1e5, 1e7)),
                'macd': rng.normal(0, 0.5), 'rsi': rng.uniform(10, 90),
                'cci': rng.normal(0, 100), 'adx': rng.uniform(5, 60),
                'turbulence': abs(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows).set_index(['symbol', 'date']).sort_index()


def env_for(df, **kw):
    return TradingEnv(df=df, initial_balance=1_000_000, commission_rate=0.001,
                      max_shares_per_trade=100, phase=1, **kw)


def test_negative_price_no_trade():
    print("\n[1] Negatif fiyatta islem yapilmaz")
    df = make_panel(corrupt={(0, 5): -50000.0})
    env = env_for(df)
    env.reset()
    env.current_step = 5
    env._price_memo_step = -1

    bal0 = env.balance
    ok, _ = env._execute_trade(0, 100)          # AL
    check("negatif fiyatta BUY reddedildi", ok is False)
    check("BUY bakiyeyi degistirmedi", env.balance == bal0,
          f"{bal0} -> {env.balance}")
    check("BUY pozisyon acmadi", env.shares_owned[0] == 0, str(env.shares_owned[0]))

    env.shares_owned[0] = 100                   # elde hisse varmis gibi
    bal1 = env.balance
    ok, _ = env._execute_trade(0, -100)         # SAT
    check("negatif fiyatta SELL reddedildi", ok is False)
    check("SELL bakiyeyi dusurmedi", env.balance == bal1,
          f"{bal1} -> {env.balance}")
    check("SELL pozisyonu bozmadi", env.shares_owned[0] == 100)


def test_zero_price_no_free_shares():
    print("\n[2] Fiyat 0 iken bedava hisse verilmez")
    df = make_panel()
    env = env_for(df)
    env.reset()
    # Veri olmayan sembol/tarih -> _get_current_price 0.0 doner
    env.symbols.append("YOK.IS")
    idx = len(env.symbols) - 1
    env.shares_owned = np.zeros(len(env.symbols))
    bal0 = env.balance
    ok, _ = env._execute_trade(idx, 100)
    check("fiyat 0 iken BUY reddedildi", ok is False)
    check("bedava hisse verilmedi", env.shares_owned[idx] == 0,
          str(env.shares_owned[idx]))
    check("bakiye degismedi", env.balance == bal0)


def test_no_nan_observation():
    print("\n[3] Bozuk fiyatla bile gozlem sonlu kalir")
    # Bircok negatif fiyat: eski kodda bakiye -initial'in altina inip NaN uretirdi
    corrupt = {(0, d): -80000.0 for d in range(10, 60)}
    corrupt.update({(1, d): -120000.0 for d in range(10, 60)})
    df = make_panel(corrupt=corrupt)
    env = env_for(df)
    obs, _ = env.reset()
    rng = np.random.default_rng(3)

    nan_adim = None
    log_uyari = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for i in range(300):
            obs, r, t1, t2, _ = env.step(rng.uniform(-1, 1, env.n_stocks))
            if nan_adim is None and not np.all(np.isfinite(obs)):
                nan_adim = i
            if not np.isfinite(r):
                check("odul sonlu", False, f"adim {i}: {r}")
            if t1 or t2:
                break
        log_uyari = sum(1 for x in w if "log" in str(x.message))

    check("hicbir adimda NaN/Inf gozlem yok", nan_adim is None,
          f"ilk NaN adim {nan_adim}")
    check("np.log 'invalid value' uyarisi yok", log_uyari == 0, str(log_uyari))
    check("bakiye -initial_balance'in altina inmedi",
          env.balance > -env.initial_balance, f"{env.balance}")
    check("gozlem [-10,10] araliginda",
          bool(np.all(obs >= -10) and np.all(obs <= 10)))


def test_balance_norm_floor():
    print("\n[4] balance_norm tabani (dogrudan)")
    df = make_panel()
    env = env_for(df)
    env.reset()
    for bal, lbl in [(-2_000_000.0, "bakiye < -initial"),
                     (-1_000_000.0, "bakiye == -initial (log argumani 0)"),
                     (float('nan'), "bakiye NaN"),
                     (float('-inf'), "bakiye -inf")]:
        env.balance = bal
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obs = env._get_observation()
            uy = [str(x.message) for x in w if "log" in str(x.message)]
        check(f"{lbl} -> sonlu gozlem", bool(np.all(np.isfinite(obs))), str(obs[0]))
        check(f"{lbl} -> log uyarisi yok", not uy, str(uy))


def test_clean_data_path_is_bit_exact():
    print("\n[5] Saglam veride davranis degismez")
    # Deliksiz + tum fiyatlar > 0: koruma hic devreye girmemeli.
    df = make_panel(n_days=500, n_symbols=4)
    env = env_for(df)
    env.reset()
    rng = np.random.default_rng(11)
    red = 0
    orig = TradingEnv._execute_trade

    def sayan(self, idx, sh):
        p = self._get_current_price(self.symbols[idx])
        if (not np.isfinite(p)) or p <= 0:
            nonlocal red
            red += 1
        return orig(self, idx, sh)

    TradingEnv._execute_trade = sayan
    try:
        for _ in range(400):
            obs, r, t1, t2, _ = env.step(rng.uniform(-1, 1, env.n_stocks))
            if t1 or t2:
                break
    finally:
        TradingEnv._execute_trade = orig

    check("saglam veride koruma hic tetiklenmedi", red == 0, f"{red} kez tetiklendi")
    check("islemler gerceklesti", len(env.trades_history) > 0)


def test_warning_throttle():
    print("\n[6] Uyari seli sembol basina tek uyariya iniyor")
    df = make_panel(n_days=300, n_symbols=3)
    # T0'in ilk 200 gununu sil -> her adimda 'missing' olacak
    mask = ~((df.index.get_level_values('symbol') == 'T0.IS') &
             (df.index.get_level_values('date') <= df.index.get_level_values('date')[200]))
    holed = df[mask]

    kayitlar = []
    handler = logging.Handler()
    handler.emit = lambda rec: kayitlar.append(rec.getMessage())
    import env.trading_env as te
    logging.disable(logging.NOTSET)
    eski_h, eski_p = te.logger.handlers, te.logger.propagate
    te.logger.handlers = [handler]
    te.logger.propagate = False
    te.logger.setLevel(logging.WARNING)
    try:
        env = env_for(holed)
        env.reset()
        rng = np.random.default_rng(2)
        for _ in range(200):
            obs, r, t1, t2, _ = env.step(rng.uniform(-1, 1, env.n_stocks))
            if t1 or t2:
                break
    finally:
        te.logger.handlers, te.logger.propagate = eski_h, eski_p
        logging.disable(logging.CRITICAL)

    eksik = [k for k in kayitlar if "Missing data" in k or "Price not found" in k]
    check(f"200 adimda eksik-veri uyarisi <= sembol sayisi x 2 ({len(eksik)} adet)",
          len(eksik) <= env.n_stocks * 2, f"{len(eksik)} uyari")
    check("uyari tamamen susturulmadi (en az bir kez bildirildi)", len(eksik) >= 1)
    check("tekrar uyarilmayacagi mesajda belirtiliyor",
          any("tekrar uyarilmayacak" in k for k in eksik) if eksik else False)


if __name__ == "__main__":
    print("=" * 70)
    print("TradingEnv fiyat butunlugu korumalari")
    print("=" * 70)
    test_negative_price_no_trade()
    test_zero_price_no_free_shares()
    test_no_nan_observation()
    test_balance_norm_floor()
    test_clean_data_path_is_bit_exact()
    test_warning_throttle()
    print("\n" + "=" * 70)
    print(f"SONUC: {passed} gecti, {failed} kaldi")
    print("=" * 70)
    sys.exit(1 if failed else 0)
