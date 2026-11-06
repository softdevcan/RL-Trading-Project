"""Faz 9 / A — Degerlendirme cekirdegi (tahmin ve RL'e ortak istatistik).

Moduller:
    sharpe            A1  PSR, beklenen en buyuk Sharpe, Deflated Sharpe
    forecast_tests    A2  Diebold-Mariano (HAC + Harvey-Leybourne-Newbold)
    sharpe_tests      A3  Ledoit-Wolf (2008) Sharpe farki testi (HAC + bootstrap)
    model_sets        A4  Model Confidence Set (arch)
    seed_stats        A5  Tohumlar arasi raporlama (rliable)
    overfitting       A6  PBO / CSCV
    multiple_testing  A7  Benjamini-Hochberg FDR (statsmodels)
    risk_free         A8  Tek kaynak risksiz faiz
    paired_tests      A9  Eslestirilmis getiri farki testleri
    trial_registry    C2  Deneme kaydi (DSR'nin N ve V'si, PBO matrisi)
    reproducibility   C1/C3/C4 RL tohumu, veri anlik goruntusu, ortam kaydi

Kaynaklar ve kabul olcutleri: docs/development/phase-9-research-validity-plan.md §4 A, C.
Testler: tests/test_eval_stats.py, tests/test_rl_reproducibility.py.
"""
