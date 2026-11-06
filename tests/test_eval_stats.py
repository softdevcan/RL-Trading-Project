"""
Faz 9 / A — degerlendirme cekirdegi testleri (evaluation/).

Kabul olcutleri docs/development/phase-9-research-validity-plan.md §4 A'dan:
  A1 makalenin s.10 ornegi: SR0 = 0,1132, DSR = 0,9004 (N=100); N=46 -> 0,9505;
     normal getiri, N=88 -> 0,9505 (+-1e-4)
  A2 H0 altinda DM boyutu %5 +- 1 (h-adim / MA(h-1) hatalar); eski fonksiyon
     ayni testte belirgin fazla reddediyor
  A3 Ledoit-Wolf, korelasyonlu GARCH esit Sharpe: boyut ~%5
  A4 acikca kotu model MCS'den duser
  A5 rliable ciktisiyla birebir
  A6 saf gurultude PBO ~0,5; gercek sinyalde belirgin dusus
  A7 BH duzeltmesi el hesabiyla ayni
  A8 yedek degerli seride acik hata; bilesik donusum
  A9 eslestirilmis test; model_analysis artik ttest_ind kullanmiyor

Simulasyonlar sabit tohumla kosar (deterministik). Olculen oranlar ciktida
basilir; esikler ve olcumler docs/development/phase-9-research-validity-plan.md
"Uygulama durumu" bolumunde.

Calistirma:
    python tests/test_eval_stats.py
"""

import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print("\nA1 — PSR / DSR (Bailey & Lopez de Prado 2014, s.10)")
from evaluation.sharpe import (  # noqa: E402
    ReturnMoments, deflated_sharpe_ratio, expected_max_sharpe, return_moments,
    probabilistic_sharpe_ratio,
)

q = 250                        # makalenin yillik gozlem sayisi
sr_d = 2.5 / np.sqrt(q)        # yillik 2,5 -> gunluk
V_d = 0.5 / q                  # yillik V = 1/2 -> gunluk
for N, g3, g4, sr0_exp, dsr_exp in [(100, -3, 10, 0.1132, 0.9004),
                                    (46, -3, 10, None, 0.9505),
                                    (88, 0, 3, None, 0.9505)]:
    d = deflated_sharpe_ratio(n_trials=N, var_trials=V_d, moments=ReturnMoments(sr_d, 1250, g3, g4))
    if sr0_exp is not None:
        check(f"N={N}: SR0 = {sr0_exp}", abs(d['sr_benchmark'] - sr0_exp) < 1e-4, f"{d['sr_benchmark']:.5f}")
    check(f"N={N}, g3={g3}, g4={g4}: DSR = {dsr_exp}", abs(d['dsr'] - dsr_exp) < 1e-4, f"{d['dsr']:.5f}")

check("N=1 -> SR0 = 0 (secim duzeltmesi yok)", expected_max_sharpe(1, 0.0) == 0.0)
try:
    expected_max_sharpe(10, None)
    check("N>1 iken V zorunlu", False)
except ValueError:
    check("N>1 iken V zorunlu", True)

rng = np.random.default_rng(1)
r = rng.standard_t(5, size=2000) * 0.01 + 0.0005
m = return_moments(r)
check("return_moments: ham basiklik (t5 > 3)", m.kurt > 3.5, f"{m.kurt:.2f}")
psr = probabilistic_sharpe_ratio(m.sr, 0.0, m.n_obs, m.skew, m.kurt)
check("PSR olasilik araliginda", 0.0 < psr < 1.0)

from prediction.evaluator import PredictionEvaluator as PE  # noqa: E402
old = PE.compute_deflated_sharpe_ratio(r, n_trials=1)
check("evaluator.DSR n_trials=1 -> dsr None (eskiden PSR'yi DSR diye veriyordu)",
      old['dsr'] is None and old['psr'] is not None)
new = PE.compute_deflated_sharpe_ratio(r, n_trials=20, var_trials=1e-4)
ref = deflated_sharpe_ratio(n_trials=20, var_trials=1e-4, returns=r)
check("evaluator.DSR yeni pakete yonlendi", abs(new['dsr'] - round(ref['dsr'], 4)) < 1e-9)
check("evaluator.DSR n_trials>1 ve V yok -> dsr None",
      PE.compute_deflated_sharpe_ratio(r, n_trials=20)['dsr'] is None)


# ---------------------------------------------------------------------------
print("\nA2 — Diebold-Mariano (HAC + HLN, t dagilimi)")
from evaluation.forecast_tests import diebold_mariano  # noqa: E402

rng = np.random.default_rng(2026)


def ma_err(T, h):
    eps = rng.standard_normal(T + h - 1)
    return np.convolve(eps, np.ones(h), "valid")


def legacy_dm_p(e1, e2):
    """Eski evaluator.diebold_mariano_test'in formulu (iid varyans + normal)."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    return 2 * (1 - stats.norm.cdf(abs(d.mean() / np.sqrt(d.var(ddof=1) / len(d)))))


S, T, h = 2000, 250, 4
rej_new = rej_old = 0
for _ in range(S):
    e1, e2 = ma_err(T, h), ma_err(T, h)
    rej_new += diebold_mariano(e1, e2, h=h)["p_value"] < 0.05
    rej_old += legacy_dm_p(e1, e2) < 0.05
size_new, rej_old = rej_new / S, rej_old / S
print(f"    olcum: h={h}, MA({h-1}) hata, T={T}, {S} sim -> yeni {size_new:.4f}, eski formul {rej_old:.4f}")
check("H0 boyutu %5 +- 1 (h-adim, varsayilan lag='h-1')", 0.04 <= size_new <= 0.06, f"{size_new:.4f}")
check("eski formul belirgin fazla reddediyor (> %10)", rej_old > 0.10, f"{rej_old:.4f}")


def ar1(T, phi):
    x = np.zeros(T)
    e = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


S2 = 1000
rej_auto = rej_old2 = 0
for _ in range(S2):
    e1, e2 = ar1(500, 0.5), ar1(500, 0.5)
    rej_auto += diebold_mariano(e1, e2, lag="auto")["p_value"] < 0.05
    rej_old2 += legacy_dm_p(e1, e2) < 0.05
print(f"    olcum: AR(1) phi=0.5 hata, T=500, {S2} sim -> lag='auto' {rej_auto/S2:.4f}, eski {rej_old2/S2:.4f}")
print("    NOT: planin '%5 +- 1 (AR(1))' olcutu KARSILANMIYOR (%7,6, MC SE ~%0,7) — Faz 9 §9 A2 'Kismen'.")
check("AR(1) kayip farkinda lag='auto' eskiden daha az fazla reddediyor (ilerleme kontrolu, olcut degil)",
      rej_auto / S2 < rej_old2 / S2, f"{rej_auto/S2:.4f}")
check("evaluator sarmalayicisi ufku tasiyor (h=5)",
      "h=5" in PE.diebold_mariano_test(np.zeros(300), -ma_err(300, 5), -ma_err(300, 5), h=5)["method"])

e1 = rng.standard_normal(300)
e2 = 1.5 * rng.standard_normal(300)
res = diebold_mariano(e1, e2, alternative="less")
check("guc: belirgin daha iyi model -> p < 0,01", res["p_value"] < 0.01, f"{res['p_value']:.4g}")
check("dagilim t(T-1)", res["distribution"] == "t(299)")
dm_old_api = PE.diebold_mariano_test(np.zeros(300), -e1, -e2)
check("evaluator.diebold_mariano_test yeni pakete yonlendi", "DM-HLN" in dm_old_api.get("method", ""))


# ---------------------------------------------------------------------------
print("\nA3 — Ledoit-Wolf Sharpe farki (HAC + studentize dairesel blok bootstrap)")
from evaluation.sharpe_tests import sharpe_diff_bootstrap, sharpe_diff_hac  # noqa: E402

rng = np.random.default_rng(7)


def garch_pair(T, rho=0.5, mu=(0.0005, 0.0005)):
    w, a, b = 1e-6, 0.08, 0.90
    hv = np.full(2, w / (1 - a - b))
    out = np.empty((T, 2))
    L = np.linalg.cholesky([[1, rho], [rho, 1]])
    for t in range(T):
        e = np.sqrt(hv) * (L @ rng.standard_normal(2))
        out[t] = np.asarray(mu) + e
        hv = w + a * e ** 2 + b * hv
    return out


x = garch_pair(1000)
res = sharpe_diff_hac(x[:, 0], x[:, 1])
direct = x[:, 0].mean() / x[:, 0].std() - x[:, 1].mean() / x[:, 1].std()
check("delta = dogrudan Sharpe farki (ddof=0, makaledeki gibi)", abs(res["delta"] - direct) < 1e-12)

S3 = 400
rej_hac = sum(sharpe_diff_hac(*garch_pair(1000).T)["p_value"] < 0.05 for _ in range(S3)) / S3
S4 = 600
rej_bs = sum(sharpe_diff_bootstrap(*garch_pair(1000).T, n_boot=299, seed=i)["p_value"] < 0.05
             for i in range(S4)) / S4
print(f"    olcum: GARCH(1,1), rho=0.5, esit Sharpe, T=1000 -> HAC {rej_hac:.4f} ({S3} sim), "
      f"bootstrap {rej_bs:.4f} ({S4} sim, M=299, b=5)")
check("HAC boyutu ~%5 (aralik %3,5-%7,5; 400 sim SE ~%1,1)",
      0.035 <= rej_hac <= 0.075, f"{rej_hac:.4f}")
check("bootstrap boyutu ~%5 (aralik %2,5-%7,5; 600 sim SE ~%0,9)",
      0.025 <= rej_bs <= 0.075, f"{rej_bs:.4f}")

# Gercek kullanim kosulu (hakem-2 #8): tek test penceresi T~500, stratejiler
# arasi yuksek korelasyon (rho=0,9), blok boyu duyarliligi.
S5 = 300
sims = [garch_pair(500, rho=0.9) for _ in range(S5)]
for b in (2, 5, 10):
    rb = sum(sharpe_diff_bootstrap(s[:, 0], s[:, 1], block_size=b, n_boot=299, seed=i)["p_value"] < 0.05
             for i, s in enumerate(sims)) / S5
    print(f"    olcum: T=500, rho=0.9, b={b} -> bootstrap boyutu {rb:.4f} ({S5} sim)")
    check(f"bootstrap boyutu b={b}, rho=0,9, T=500 (aralik %2-%9)", 0.02 <= rb <= 0.09, f"{rb:.4f}")
rh9 = sum(sharpe_diff_hac(s[:, 0], s[:, 1])["p_value"] < 0.05 for s in sims) / S5
print(f"    olcum: T=500, rho=0.9 -> HAC boyutu {rh9:.4f} ({S5} sim)")
check("HAC boyutu rho=0,9, T=500 (aralik %2-%9)", 0.02 <= rh9 <= 0.09, f"{rh9:.4f}")

x = garch_pair(1500, mu=(0.0015, 0.0))
p_h = sharpe_diff_hac(x[:, 0], x[:, 1])["p_value"]
p_b = sharpe_diff_bootstrap(x[:, 0], x[:, 1], n_boot=999, seed=0)["p_value"]
check("guc: farkli Sharpe -> iki test de reddediyor", p_h < 0.05 and p_b < 0.05, f"{p_h:.4g} {p_b:.4g}")
try:
    sharpe_diff_hac(np.zeros(10), np.zeros(11))
    check("farkli uzunluk reddedilir", False)
except ValueError:
    check("farkli uzunluk reddedilir", True)


# ---------------------------------------------------------------------------
print("\nA4 — Model Confidence Set (arch)")
from evaluation.model_sets import model_confidence_set  # noqa: E402

rng = np.random.default_rng(3)
base = rng.standard_normal((800, 1))
losses = pd.DataFrame({
    "m1": (base[:, 0] + 0.3 * rng.standard_normal(800)) ** 2,
    "m2": (base[:, 0] + 0.3 * rng.standard_normal(800)) ** 2,
    "m3": (base[:, 0] + 0.3 * rng.standard_normal(800)) ** 2,
    "kotu": (base[:, 0] + 1.5 * rng.standard_normal(800)) ** 2,
})
mcs = model_confidence_set(losses, size=0.10, reps=1000, seed=0)
check("acikca kotu model kumeden dusuyor", "kotu" in mcs["excluded"], str(mcs))
check("esdeger modeller kumede", {"m1", "m2", "m3"} <= set(mcs["included"]), str(mcs["included"]))


# ---------------------------------------------------------------------------
print("\nA5 — Tohumlar arasi raporlama (rliable)")
from rliable import library as rly, metrics as rmetrics  # noqa: E402
from scipy import stats as sst  # noqa: E402
from evaluation.seed_stats import aggregate_with_ci, probability_of_improvement  # noqa: E402

rng = np.random.default_rng(5)
scores = {"A": rng.normal(1.0, 0.3, (10, 3)), "B": rng.normal(0.8, 0.3, (10, 3))}
np.random.seed(123)
probe = np.random.get_state()[1][:5].copy()
np.random.seed(123)
ours = aggregate_with_ci(scores, reps=500, seed=11)
check("kuresel numpy durumu geri yuklendi", np.array_equal(np.random.get_state()[1][:5], probe))
again = aggregate_with_ci(scores, reps=500, seed=11)
check("ayni tohum -> ayni GA (rliable random_state'i yok saydigi icin sarmalayici tohumluyor)",
      again["A"]["iqm"] == ours["A"]["iqm"])
np.random.seed(11)   # rliable kuresel np.random kullaniyor (seed_stats._seeded_global_rng)
pt, ci = rly.get_interval_estimates(
    scores,
    lambda x: np.array([rmetrics.aggregate_median(x), rmetrics.aggregate_iqm(x),
                        rmetrics.aggregate_mean(x), rmetrics.aggregate_optimality_gap(x, 1.0)]),
    reps=500)
check("nokta ve GA rliable ile birebir",
      all(np.isclose(ours[k]["iqm"][0], pt[k][1]) and np.isclose(ours[k]["iqm"][1], ci[k][0, 1])
          and np.isclose(ours[k]["iqm"][2], ci[k][1, 1]) for k in scores))
check("IQM = %25 kirpilmis ortalama", np.isclose(ours["A"]["iqm"][0], sst.trim_mean(scores["A"].ravel(), 0.25)))
poi = probability_of_improvement({"A,B": (scores["A"], scores["B"])}, reps=500, seed=0)
check("P(A > B) > 0,5 ve GA icinde", poi["A,B"][1] <= poi["A,B"][0] <= poi["A,B"][2] and poi["A,B"][0] > 0.5)


# ---------------------------------------------------------------------------
print("\nA6 — PBO / CSCV (Bailey vd. 2017)")
from evaluation.overfitting import probability_of_backtest_overfitting  # noqa: E402

rng = np.random.default_rng(9)
noise = rng.normal(0, 0.01, (1600, 40))
pbo_noise = probability_of_backtest_overfitting(noise, n_blocks=12)["pbo"]
sig = noise.copy()
sig[:, 0] += 0.003                       # tek gercek sinyal
pbo_sig = probability_of_backtest_overfitting(sig, n_blocks=12)["pbo"]
mod = noise.copy()
mod[:, 0] += 0.01 / np.sqrt(252)        # orta sinyal: yillik SR ~1 (hakem-2 #14)
pbo_mod = probability_of_backtest_overfitting(mod, n_blocks=12)["pbo"]
print(f"    olcum: saf gurultu PBO {pbo_noise:.3f}, orta sinyal (yillik SR~1) {pbo_mod:.3f}, "
      f"guclu sinyal (yillik SR~4,8) {pbo_sig:.3f}")
check("saf gurultude PBO ~0,5 (0,35-0,65)", 0.35 <= pbo_noise <= 0.65, f"{pbo_noise:.3f}")
check("guclu sinyalde PBO belirgin dusuk (< 0,1)", pbo_sig < 0.1, f"{pbo_sig:.3f}")
check("orta sinyalde PBO gurultuden dusuk", pbo_mod < pbo_noise, f"{pbo_mod:.3f}")


# ---------------------------------------------------------------------------
print("\nA7 — Benjamini-Hochberg")
from evaluation.multiple_testing import benjamini_hochberg  # noqa: E402

bh = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}, alpha=0.05)["results"]
# el hesabi: sirali p (0.005, 0.01, 0.03, 0.04) * 4/i -> (0.02, 0.02, 0.04, 0.04)
check("p_adj el hesabiyla ayni", np.allclose([bh[k]["p_adj"] for k in "dacb"], [0.02, 0.02, 0.04, 0.04]))
check("hepsi reddediliyor (alpha=0,05)", all(v["reject"] for v in bh.values()))


# ---------------------------------------------------------------------------
print("\nA8 — Tek kaynak risksiz faiz")
from evaluation.risk_free import RiskFreeUnavailable, per_period_rate, risk_free_series  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    # Yedek degerle yazilmis makro dosyasi (B2 oncesi gercek durum) -> acik hata
    p = os.path.join(tmp, "macro_data.csv")
    pd.DataFrame({"policy_rate": [50.0] * 3},
                 index=pd.bdate_range("2020-01-01", periods=3)).to_csv(p)
    json.dump({"data_quality": {"policy_rate": "fallback_constant_50.0"}},
              open(os.path.join(tmp, "macro_data_quality.json"), "w"))
    try:
        risk_free_series(pd.bdate_range("2020-01-02", periods=2), csv_path=p, column="policy_rate")
        check("yedek degerli seri -> acik hata (sabite dusulmez)", False, "hata yukselmedi")
    except RiskFreeUnavailable:
        check("yedek degerli seri -> acik hata (sabite dusulmez)", True)

real_rf = os.path.join("data", "macro", "risk_free.csv")
if os.path.exists(real_rf):
    rs = risk_free_series(pd.bdate_range("2013-06-03", "2026-07-27"))
    ann = (1 + rs.dropna()) ** 365 - 1     # kaba yilliklandirma, yalnizca makul aralik kontrolu
    print(f"    gercek seri (TP.APIFON4): {rs.notna().sum()} seans, donem basi ort. {rs.mean():.2e}")
    check("gercek risksiz faiz serisi RL penceresini kapsiyor ve sonlu",
          rs.iloc[1:].notna().all() and np.isfinite(rs.iloc[1:]).all())
else:
    print("    (data/macro/risk_free.csv yok — gercek seri kontrolu atlandi)")

rate = pd.Series([50.0, 50.0, 40.0], index=pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-08"]))
sess = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
rf = per_period_rate(sess, rate)
check("ilk seans NaN (onceki oran bilinmiyor)", np.isnan(rf.iloc[0]))
check("Cuma->Pazartesi 3 takvim gunu, t-1 orani (%50)",
      np.isclose(rf.loc["2024-01-08"], 1.5 ** (3 / 365) - 1))
check("ileriye bakma yok: 01-08'de degisen oran 01-09'da uygulanir",
      np.isclose(rf.loc["2024-01-09"], 1.4 ** (1 / 365) - 1))
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "m.csv")
    pd.DataFrame({"policy_rate": [45.0, 46.0]}, index=pd.to_datetime(["2024-02-01", "2024-02-02"])).to_csv(p)
    try:
        risk_free_series(pd.to_datetime(["2024-01-15", "2024-02-02"]), csv_path=p)
        check("serinin basindan once istenen donem -> hata (bfill yok)", False)
    except RiskFreeUnavailable:
        check("serinin basindan once istenen donem -> hata (bfill yok)", True)


# ---------------------------------------------------------------------------
print("\nA9 — Eslestirilmis testler")
from scipy import stats  # noqa: E402
from evaluation.paired_tests import mean_diff_bootstrap, mean_diff_hac, paired_t  # noqa: E402

rng = np.random.default_rng(12)
mkt = rng.normal(0, 0.015, 750)
r1 = mkt + rng.normal(0.0004, 0.002, 750)
r2 = mkt + rng.normal(0.0, 0.002, 750)
p_ind = stats.ttest_ind(r1, r2).pvalue
p_pair = paired_t(r1, r2)["p_value"]
print(f"    olcum: ortak piyasa bileseni, kucuk fark -> bagimsiz p={p_ind:.3f}, eslestirilmis p={p_pair:.2e}")
check("eslestirilmis test farki yakaliyor, bagimsiz kaciriyor", p_pair < 0.01 and p_ind > 0.2)
check("HAC ve bootstrap da yakaliyor",
      mean_diff_hac(r1, r2)["p_value"] < 0.01 and mean_diff_bootstrap(r1, r2, n_boot=999)["p_value"] < 0.01)

from app.services.model_analysis import ModelAnalyzer  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    ma = ModelAnalyzer(results_dir=tmp)
    out = ma.statistical_significance_test(r1, r2)
    check("model_analysis varsayilani artik eslestirilmis", out["test"] == "Paired t-test", out["test"])
    out2 = ma.statistical_significance_test(r1, r2[:-50])
    check("farkli uzunluk sessiz degil: not tasiyor", "note" in out2)
    out3 = ma.statistical_significance_test(r1, r2, test_type="sharpe")
    check("Sharpe farki secenegi Ledoit-Wolf", out3["test"].startswith("Ledoit-Wolf"))

src = open(os.path.join("app", "services", "model_analysis.py"), encoding="utf-8").read()
check("model_analysis'te ttest_ind cagrisi kalmadi", "ttest_ind(" not in src)


print(f"\n{'=' * 60}\nSonuc: {passed} gecti, {failed} kaldi")
sys.exit(1 if failed else 0)
