"""
Faz 9 / G1 — PSR odulunde Moody & Saffell (2001) diferansiyel Sharpe (`sharpe_term`).

Kaynak: Moody & Saffell, "Learning to trade via direct reinforcement", IEEE TNN 12(4),
2001, s.4 (yazar surumu, Zotero JV7QAJNU):
  (13) S_t ~= S_{t-1} + eta * D_t + O(eta^2)
  (14) D_t = (B_{t-1} dA_t - 1/2 A_{t-1} dB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
  (15) A_t = A_{t-1} + eta (R_t - A_{t-1}),  B_t = B_{t-1} + eta (R_t^2 - B_{t-1})
  (16) dD_t/dR_t = (B_{t-1} - A_{t-1} R_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
R_t fazla getiri (R_t - r_f; s.4).

Kontroller:
  1. Denklem 13: EMA Sharpe'in gercek artisi ile eta*D_t farki O(eta^2); goreli hata
     eta ile dogrusal kuculur (eta 1/30 -> 1/300). Eski 'ema_level' terimi (Y2) ve
     isaret mutasyonu bu kontrolu KIRAR.
  2. Denklem 16: D_t'nin R_t'ye gore sonlu farki makaledeki turevle ayni (14 <-> 16).
  3. Fazla getiri: risk_free_rate=r ile D_t, r=0'da R - r/252 serisinin D_t'si.
  4. Isinma: ilk 1/eta adimda D_t = 0 (tasarim varsayimi), sonra degil.
  5. Ortam: varsayilan 'ema_level', 'differential' ham D_t'yi bilesende tasiyor,
     gecersiz deger / 'simple' odulle 'differential' ValueError.
  6. Soy: HPO baska Sharpe terimiyle aradiysa `hpo_reward_mismatch`; G1 oncesi PSR
     kaydi (anahtarsiz) 'ema_level' sayilir.
Varsayilanin bit-esligi `tests/test_price_stats_window.py` (faz2_psr, main ile) bekcisinde.

Calistirma:
    python tests/test_reward_differential.py
"""

import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.reward_functions import RewardCalculator  # noqa: E402
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


def returns(n=2000, seed=3, mu=0.0004, sd=0.012):
    rng = np.random.default_rng(seed)
    return mu + sd * rng.standard_t(5, size=n) / np.sqrt(5 / 3)


# 1/2*A*dB terimi ancak ortalama getiri buyukken agirlik tasir; isaret mutasyonu bu seride
HIGH_MEAN = dict(mu=0.006, sd=0.01)


def sharpe(a, b):
    return a / np.sqrt(b - a * a)


def eq13_error(window, dt_fn=None, r=None):
    """Isinma sonrasi max |dS - eta*D| / |eta*D| (D'si 0'dan uzak adimlarda)."""
    r = returns() if r is None else r
    c = RewardCalculator(rolling_window=window, risk_free_rate=0.0, sharpe_term='differential')
    eta = c.dsr_eta
    errs = []
    for x in r:
        a0, b0 = c.dsr_a, c.dsr_b
        if dt_fn is None:
            _, d = c._calculate_moody_saffell_dt(x)
        else:
            d = dt_fn(c, x)
        if d == 0.0 or c.dsr_n <= 3 * window:
            continue
        ds = sharpe(c.dsr_a, c.dsr_b) - sharpe(a0, b0)
        if abs(eta * d) > 1e-3 * abs(sharpe(a0, b0)) + 1e-6:
            errs.append(abs(ds - eta * d) / abs(eta * d))
    return float(np.median(errs)), eta


def mutant_level(c, x):
    """Y2: eski terim — EMA Sharpe DUZEYI (turev degil)."""
    c.dsr_a += c.dsr_eta * (x - c.dsr_a)
    c.dsr_b += c.dsr_eta * (x * x - c.dsr_b)
    c.dsr_n += 1
    return sharpe(c.dsr_a, c.dsr_b) if c.dsr_n > 30 else 0.0


def mutant_sign(c, x):
    """Paydaki -1/2 yerine +1/2."""
    a0, b0 = c.dsr_a, c.dsr_b
    c.dsr_a += c.dsr_eta * (x - a0)
    c.dsr_b += c.dsr_eta * (x * x - b0)
    c.dsr_n += 1
    if c.dsr_n <= round(1 / c.dsr_eta):
        return 0.0
    return (b0 * (x - a0) + 0.5 * a0 * (x * x - b0)) / (b0 - a0 * a0) ** 1.5


print("\n[0] Varsayilan ('ema_level') main'deki odul moduluyle bit-es")
# Not: test_price_stats_window'un "main ile bit-es" bekcisi trading_env.py'yi git'ten
# yukler ama o modul env.reward_functions'i CALISMA KOPYASINDAN alir — odul modulunu
# sinamaz. Burada main'deki reward_functions.py dogrudan yuklenir.
import importlib.util  # noqa: E402
import subprocess  # noqa: E402

MAIN_REF = os.environ.get("G1_MAIN_REF", "main")
src = subprocess.run(["git", "show", f"{MAIN_REF}:env/reward_functions.py"],
                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     capture_output=True, timeout=30)
if src.returncode == 0 and src.stdout:
    path = os.path.join(tempfile.mkdtemp(prefix="g1_main_"), "reward_functions_main.py")
    with open(path, "wb") as fh:
        fh.write(src.stdout)
    spec = importlib.util.spec_from_file_location("reward_functions_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def run(calc_cls, **kw):
        c = calc_cls(**kw)
        rng = np.random.default_rng(11)
        pv, rets, out = [1_000_000.0], [], []
        for step in range(400):
            ch = rng.normal(0.0005, 0.015)
            pv.append(pv[-1] * (1 + ch))
            rets.append(ch * 100)
            out.append(c.calculate_psr_reward(pv, rets, step + 1, int(rng.integers(0, 90)),
                                              float(rng.uniform(0, 300)), 1_000_000))
            if step == 199:
                c.reset()
        return out

    for kw in ({}, dict(w1=0.4, w2=0.4, w3=0.1, w4=0.05, w5=0.05, rolling_window=20,
                        risk_free_rate=0.45)):
        new, old = run(RewardCalculator, **kw), run(mod.RewardCalculator, **kw)
        same = all(a[0] == b[0] and a[1] == b[1] for a, b in zip(new, old))
        check(f"main ile bit-es (odul + bilesenler, reset dahil): {kw or 'varsayilan'}", same)
else:
    print(f"    [ATLANDI] git show {MAIN_REF}:env/reward_functions.py okunamadi")

print("\n[1] Denklem 13: dS = eta*D + O(eta^2)")
e30, eta30 = eq13_error(30)
e300, eta300 = eq13_error(300)
h30, _ = eq13_error(30, r=returns(**HIGH_MEAN))
h300, _ = eq13_error(300, r=returns(**HIGH_MEAN))
print(f"    medyan goreli hata: eta={eta30:.4f} -> {e30:.4f};  eta={eta300:.5f} -> {e300:.5f}"
      f"  (yuksek ortalama: {h30:.4f}, {h300:.5f})")
check("goreli hata kucuk (eta=1/30: < 0.1; iki seri)", e30 < 0.1 and h30 < 0.1, f"{e30:.4f} {h30:.4f}")
check("goreli hata eta ile dogrusal kuculur (oran 5..20; eta 10 kat kuculdu; iki seri)",
      5 < e30 / e300 < 20 and 5 < h30 / h300 < 20, f"oran {e30 / e300:.2f} {h30 / h300:.2f}")
em_level, _ = eq13_error(30, mutant_level)
em_sign, _ = eq13_error(30, mutant_sign, r=returns(**HIGH_MEAN))
print(f"    mutasyon: duzey (Y2) {em_level:.3f}, isaret {em_sign:.3f}")
check("mutasyon (Y2, EMA Sharpe duzeyi) denklem 13'u KIRAR (hata > 1)", em_level > 1, f"{em_level:.3f}")
check("mutasyon (isaret +1/2, yuksek ortalamali seri) denklem 13'u KIRAR (hata > 10 x dogru)",
      em_sign > 10 * h30, f"{em_sign:.3f} / {h30:.4f}")

print("\n[2] Denklem 16: dD/dR sonlu farkla")
c = RewardCalculator(rolling_window=30, risk_free_rate=0.0, sharpe_term='differential')
for x in returns()[:200]:
    c._calculate_moody_saffell_dt(x)
a0, b0, n0 = c.dsr_a, c.dsr_b, c.dsr_n
worst = 0.0
for x in (-0.03, -0.005, 0.0, 0.004, 0.02):
    h = 1e-6
    vals = []
    for xx in (x - h, x + h):
        c.dsr_a, c.dsr_b, c.dsr_n = a0, b0, n0
        vals.append(c._calculate_moody_saffell_dt(xx)[1])
    fd = (vals[1] - vals[0]) / (2 * h)
    an = (b0 - a0 * x) / (b0 - a0 * a0) ** 1.5
    worst = max(worst, abs(fd - an) / abs(an))
check("sonlu fark = (B - A R)/(B - A^2)^{3/2}  (goreli < 1e-5)", worst < 1e-5, f"{worst:.2e}")

print("\n[3] Fazla getiri")
r = returns()[:300]
rf = 0.45
c1 = RewardCalculator(risk_free_rate=rf, sharpe_term='differential')
c0 = RewardCalculator(risk_free_rate=0.0, sharpe_term='differential')
d1 = [c1._calculate_moody_saffell_dt(x)[1] for x in r]
d0 = [c0._calculate_moody_saffell_dt(x - rf / 252)[1] for x in r]
check("D_t(R; r_f) == D_t(R - r_f/252; 0)", np.allclose(d1, d0, rtol=1e-12, atol=0), "")

print("\n[4] Isinma")
c = RewardCalculator(rolling_window=30, sharpe_term='differential')
d = [c._calculate_moody_saffell_dt(x)[1] for x in returns()[:60]]
check("ilk 30 adimda D_t = 0, sonraki 30 adimin hepsinde != 0",
      all(v == 0.0 for v in d[:30]) and all(v != 0.0 for v in d[30:]), str(d[28:33]))
c.reset()
check("reset isinmayi da sifirlar", (c.dsr_a, c.dsr_b, c.dsr_n) == (0.0, 0.0, 0))

print("\n[5] Ortam")


def panel(n_days=60, n_sym=2, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for s in range(n_sym):
        p = 100.0 * (s + 1)
        for dd in dates:
            p *= 1 + rng.normal(0, 0.01)
            rows.append({"symbol": f"T{s}.IS", "date": dd, "open": p, "high": p * 1.01,
                         "low": p * 0.99, "close": p, "volume": 1e6, "macd": 0.0,
                         "rsi": 50.0, "cci": 0.0, "adx": 20.0, "turbulence": 0.1})
    return pd.DataFrame(rows).set_index(["symbol", "date"]).sort_index()


df = panel()
e_def = TradingEnv(df, reward_type='psr')
check("varsayilan sharpe_term 'ema_level'",
      e_def.sharpe_term == 'ema_level' and e_def.reward_calculator.sharpe_term == 'ema_level')
check("'simple' odulde sharpe_term None", TradingEnv(df).sharpe_term is None)
e = TradingEnv(df, reward_type='psr', sharpe_term='differential')
e.reset(seed=0)
rng = np.random.default_rng(0)
comps = []
for _ in range(45):
    _, _, term, trunc, info = e.step(rng.uniform(-1, 1, size=e.action_space.shape).astype(np.float32))
    comps.append(e.reward_calculator.sharpe_term)
    if term or trunc:
        break
check("'differential' ortamda adim atiyor; hesaplayici 'differential'",
      e.reward_calculator.sharpe_term == 'differential' and e.reward_calculator.dsr_n > 30,
      str(e.reward_calculator.dsr_n))
_, comp = e.reward_calculator.calculate_psr_reward(
    portfolio_values=[1.0, 1.01], returns=[1.0], current_step=1, trades_executed=0,
    commission_cost=0.0, initial_balance=1.0)
check("bilesende ham D_t ('sharpe_raw') ve tanh(D/2) ('sharpe_ratio')",
      'sharpe_raw' in comp and abs(comp['sharpe_ratio'] - np.tanh(comp['sharpe_raw'] / 2)) < 1e-12)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


check("bilinmeyen sharpe_term -> ValueError (env ve hesaplayici)",
      raises(lambda: TradingEnv(df, reward_type='psr', sharpe_term='dsr'))
      and raises(lambda: RewardCalculator(sharpe_term='dsr')))
check("'simple' odulle 'differential' -> ValueError (sessizce yok sayilmaz)",
      raises(lambda: TradingEnv(df, sharpe_term='differential')))

print("\n[6] Soy (hpo_lineage)")
import evaluation.trial_registry as reg  # noqa: E402

REG = tempfile.mkdtemp(prefix="g1_reg_")
base = {"symbols": ["A"], "reward_type": "psr"}


def put(study, cfg):
    reg.register_trial(kind="rl_hpo", config=cfg, seed=7, split="val", fold="2021",
                       split_spec={"name": "wf-v1", "digest": "d"},
                       protocol=dict(reg.CLEAN_PROTOCOL), directory=REG,
                       extra={"study": study, "val_range": ["2020-01-02", "2020-12-31"]})


put("s_ema", {**base, "sharpe_term": "ema_level"})
put("s_diff", {**base, "sharpe_term": "differential"})
put("s_old", dict(base))  # G1 oncesi: anahtar yok
lin = lambda study, term: reg.hpo_lineage(
    study, reward_config={"reward_type": "psr", "sharpe_term": term}, directory=REG)
check("ayni terim -> temiz", lin("s_ema", "ema_level") == (True, []) and
      lin("s_diff", "differential") == (True, []))
check("farkli terim -> yalniz 'hpo_reward_mismatch'",
      lin("s_ema", "differential") == (False, ["hpo_reward_mismatch"]) and
      lin("s_diff", "ema_level") == (False, ["hpo_reward_mismatch"]))
check("G1 oncesi PSR kaydi 'ema_level' sayilir",
      lin("s_old", "ema_level") == (True, []) and
      lin("s_old", "differential") == (False, ["hpo_reward_mismatch"]))

print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
