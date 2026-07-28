"""
TradingEnv lookup cache eşdeğerlik testi

`_get_observation` / `_get_current_price` / `_get_atr` artık sembol×tarih satırlarını
önceden numpy'a çıkarılmış bir cache'ten okuyor (eski yol: adım başına ~17 MultiIndex
`.loc`). Bu testin işi tek: **cache'li ve cache'siz yol bit-eş sonuç üretsin.**
Gözlem vektörü değişirse mevcut eğitilmiş modeller sessizce bozulur — bu yüzden
karşılaştırma `array_equal` ile yapılır, tolerans yok.

Çalıştırma:
    python tests/test_env_lookup_equivalence.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.trading_env import TradingEnv  # noqa: E402

logging.disable(logging.CRITICAL)  # eksik-veri uyarıları çıktıyı boğmasın

N_STEPS = 1200
SEED = 7

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


def make_df(n_days=600, n_symbols=4, seed=42):
    """Sentetik ama gerçekçi OHLCV+indikatör paneli (gerçek CSV'ye bağımlı olmasın)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    records = []
    for s in range(n_symbols):
        symbol = f"TEST{s}.IS"
        price = 100.0 * (1 + s)
        for d in dates:
            ret = rng.normal(0, 0.02)
            price = max(1.0, price * (1 + ret))
            high = price * (1 + abs(rng.normal(0, 0.01)))
            low = price * (1 - abs(rng.normal(0, 0.01)))
            records.append({
                'symbol': symbol,
                'date': d,
                'open': price * (1 + rng.normal(0, 0.005)),
                'high': high,
                'low': low,
                'close': price,
                'volume': float(rng.integers(1e5, 5e7)),
                'macd': rng.normal(0, 0.5),
                'rsi': rng.uniform(10, 90),
                'cci': rng.normal(0, 120),
                'adx': rng.uniform(5, 60),
                'turbulence': abs(rng.normal(0, 1.5)),
            })
    return pd.DataFrame(records).set_index(['symbol', 'date']).sort_index()


def make_env(df, disable_cache, **kwargs):
    env = TradingEnv(df=df, initial_balance=1_000_000, phase=1, **kwargs)
    if disable_cache:
        env._lookup_cache = None  # eski `.loc` yoluna zorla
    return env


def compare_episode(df, label, **env_kwargs):
    """Cache'li ve cache'siz env'i aynı aksiyon dizisiyle sürüp her şeyi karşılaştır."""
    fast = make_env(df, disable_cache=False, **env_kwargs)
    slow = make_env(df, disable_cache=True, **env_kwargs)

    check(f"{label}: cache kuruldu", fast._lookup_cache is not None)
    check(f"{label}: karşılaştırma env'i cache'siz", slow._lookup_cache is None)

    o_fast, i_fast = fast.reset()
    o_slow, i_slow = slow.reset()
    check(f"{label}: reset() gözlemi bit-eş", np.array_equal(o_fast, o_slow))
    check(f"{label}: reset() portföy değeri eş",
          i_fast['portfolio_value'] == i_slow['portfolio_value'])

    rng = np.random.default_rng(SEED)
    obs_diff = rew_diff = 0
    steps = 0
    for _ in range(N_STEPS):
        action = rng.uniform(-1, 1, fast.n_stocks)
        of, rf, tf, uf, _ = fast.step(action)
        os_, rs, ts, us, _ = slow.step(action)
        steps += 1
        if not np.array_equal(of, os_):
            obs_diff += 1
        if rf != rs:
            rew_diff += 1
        if tf or uf or ts or us:
            break

    check(f"{label}: {steps} adım gözlem bit-eş", obs_diff == 0, f"{obs_diff} adımda fark")
    check(f"{label}: {steps} adım ödül bit-eş", rew_diff == 0, f"{rew_diff} adımda fark")
    check(f"{label}: bakiye eş", fast.balance == slow.balance,
          f"{fast.balance!r} != {slow.balance!r}")
    check(f"{label}: işlem sayısı eş", len(fast.trades_history) == len(slow.trades_history),
          f"{len(fast.trades_history)} != {len(slow.trades_history)}")
    check(f"{label}: portföy değeri eş",
          fast._get_portfolio_value() == slow._get_portfolio_value())


def test_price_lookup(df):
    """Fiyat okuması tüm tarih aralığında bit-eş mi?"""
    fast = make_env(df, disable_cache=False)
    slow = make_env(df, disable_cache=True)
    diff = 0
    n = 0
    for step_i in range(0, len(fast.dates) - 1, 3):
        fast.current_step = slow.current_step = step_i
        fast._price_memo_step = slow._price_memo_step = -1
        for sym in fast.symbols:
            a = fast._get_current_price(sym)
            b = slow._get_current_price(sym)
            n += 1
            if a != b:
                diff += 1
    check(f"fiyat lookup bit-eş ({n} karşılaştırma)", diff == 0, f"{diff} fark")


def test_atr(df):
    """ATR penceresi (searchsorted) pandas mask+tail ile aynı sonucu vermeli."""
    fast = make_env(df, disable_cache=False, use_atr_sizing=True)
    slow = make_env(df, disable_cache=True, use_atr_sizing=True)
    diff = 0
    n = 0
    for step_i in range(0, len(fast.dates) - 1, 5):
        fast.current_step = slow.current_step = step_i
        for sym in fast.symbols:
            a = fast._get_atr(sym)
            b = slow._get_atr(sym)
            n += 1
            if a != b:
                diff += 1
    check(f"ATR bit-eş ({n} karşılaştırma)", diff == 0, f"{diff} fark")


def test_missing_rows(df):
    """Eksik satır olduğunda da eşdeğer davranmalı (KeyError -> forward-fill yolu)."""
    holed = df.drop(df.index[[10, 11, 250, 251, 252, 900]])
    compare_episode(holed, "eksik satırlı panel")


def test_cache_skipped_when_column_missing(df):
    """Zorunlu kolon yoksa cache kurulmamalı, env yine çalışmalı."""
    trimmed = df.drop(columns=['high'])
    env = TradingEnv(df=trimmed, initial_balance=1_000_000, phase=1)
    check("zorunlu kolon eksikse cache kurulmuyor", env._lookup_cache is None)
    obs, _ = env.reset()
    check("cache'siz env yine de gözlem üretiyor",
          obs.shape == env.observation_space.shape and np.all(np.isfinite(obs)))


def test_optional_column_missing(df):
    """Opsiyonel indikatör kolonu yoksa default'lar eski `.get(col, default)` ile aynı olmalı."""
    trimmed = df.drop(columns=['turbulence', 'cci'])
    compare_episode(trimmed, "opsiyonel kolon eksik")


def test_price_memo_consistency(df):
    """Adım-içi fiyat memo'su, adım ilerleyince eskimemeli."""
    env = make_env(df, disable_cache=False)
    env.reset()
    ok = True
    for _ in range(50):
        expected = {s: float(env.df.loc[(s, env._get_current_date()), 'close'])
                    for s in env.symbols}
        got = {s: env._get_current_price(s) for s in env.symbols}
        got_again = {s: env._get_current_price(s) for s in env.symbols}
        if got != expected or got_again != expected:
            ok = False
            break
        env.step(np.zeros(env.n_stocks))
    check("fiyat memo'su adım ilerleyince tazeleniyor", ok)


if __name__ == '__main__':
    print("=" * 70)
    print("TradingEnv lookup cache eşdeğerlik testi")
    print("=" * 70)

    df = make_df()

    print("\n[1] Fiyat lookup")
    test_price_lookup(df)

    print("\n[2] Tam episode (temel panel)")
    compare_episode(df, "temel panel")

    print("\n[3] ATR penceresi")
    test_atr(df)

    print("\n[4] ATR sizing açıkken tam episode")
    compare_episode(df, "ATR sizing", use_atr_sizing=True)

    print("\n[5] Eksik satırlar")
    test_missing_rows(df)

    print("\n[6] Opsiyonel kolon eksik")
    test_optional_column_missing(df)

    print("\n[7] Zorunlu kolon eksik -> cache devre dışı")
    test_cache_skipped_when_column_missing(df)

    print("\n[8] Adım-içi fiyat memo'su")
    test_price_memo_consistency(df)

    print("\n" + "=" * 70)
    print(f"SONUC: {passed} gecti, {failed} kaldi")
    print("=" * 70)
    sys.exit(1 if failed else 0)
