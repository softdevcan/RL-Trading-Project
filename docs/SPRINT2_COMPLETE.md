# SPRINT 2 TAMAMLANDI - PSR REWARD ENGINEERING

**Tarih:** 2025-12-14
**Proje:** RL Trading System - BIST-30 Algoritmik Ticaret
**Referans:** Ansari et al. (2024) - "A Multifaceted Approach to Stock Market Trading"

---

## SPRINT 2 OZET

Sprint 2'de PSR (Portfolio-Sharpe-Returns) reward function'i basariyla implement edildi ve TradingEnv'e entegre edildi. Ayrica Optuna ile hyperparameter optimization ve A/B testing scriptleri olusturuldu.

---

## TAMAMLANAN GOREVLER

### 1. TradingEnv PSR Entegrasyonu ✓

**Dosya:** [env/trading_env.py](../env/trading_env.py)

**Degisiklikler:**
- `reward_type` parametresi eklendi ('simple' veya 'psr')
- `reward_weights` parametresi eklendi (PSR agirliklari icin)
- `RewardCalculator` instance olusturuldu
- `reset()` metodunda reward calculator state sifirlama
- `step()` metodunda reward type'a gore reward hesaplama
- `returns_history` listesi eklendi (PSR icin gerekli)
- `reward_components` bilgisi info dict'e eklendi

**Kullanim:**
```python
# Simple reward (Faz 1 baseline)
env = make_env(df, phase=1, reward_type='simple')

# PSR reward (Faz 2) - default weights
env = make_env(df, phase=2, reward_type='psr',
               fundamental_df=fund_df, macro_df=macro_df)

# PSR reward - custom weights
custom_weights = {
    'w1': 0.50,  # Portfolio return
    'w2': 0.30,  # Sharpe ratio
    'w3': 0.10,  # MDD penalty
    'w4': 0.05,  # Volatility penalty
    'w5': 0.05   # Trade frequency
}
env = make_env(df, phase=2, reward_type='psr',
               reward_weights=custom_weights,
               fundamental_df=fund_df, macro_df=macro_df)
```

---

### 2. Optuna Hyperparameter Optimization ✓

**Dosya:** [scripts/optimization/optimize_reward_weights.py](../scripts/optimization/optimize_reward_weights.py)

**Ozellikler:**
- Bayesian optimization ile w1-w5 agirliklari optimize edilir
- 50-100 trial ile en iyi weight kombinasyonu bulunur
- Objective: Sharpe Ratio maksimize, MDD minimize
- Quick training (10k steps per trial) ile hizli iterasyon
- Sonuclar JSON, CSV ve HTML grafikleri olarak kaydedilir

**Kullanim:**
```bash
# Default (50 trials)
venv\Scripts\python.exe scripts\optimization\optimize_reward_weights.py

# Custom trial count
venv\Scripts\python.exe scripts\optimization\optimize_reward_weights.py --trials 100
```

**Ciktilar:**
- `results/optimization/best_psr_weights.json` - En iyi parametreler
- `results/optimization/optimization_trials.csv` - Tum trial sonuclari
- `results/optimization/optimization_history.html` - Optimizasyon grafigi
- `results/optimization/param_importances.html` - Parametre onemleri
- `results/optimization/contour_w1_w2.html` - Contour plot

---

### 3. A/B Testing Script ✓

**Dosya:** [scripts/optimization/ab_test_rewards.py](../scripts/optimization/ab_test_rewards.py)

**Test Senaryolari:**
1. **Phase1-Simple**: Faz 1 (56 features) + Simple Reward (Baseline)
2. **Phase2-Simple**: Faz 2 (97 features) + Simple Reward
3. **Phase1-PSR**: Faz 1 (56 features) + PSR Reward
4. **Phase2-PSR**: Faz 2 (97 features) + PSR Reward (Best)

**Metrikler:**
- Sharpe Ratio
- Max Drawdown (MDD)
- Total Return
- Final Portfolio Value
- Total Trades

**Kullanim:**
```bash
# Default (50k timesteps per experiment)
venv\Scripts\python.exe scripts\optimization\ab_test_rewards.py

# Custom timesteps
venv\Scripts\python.exe scripts\optimization\ab_test_rewards.py --timesteps 100000
```

**Ciktilar:**
- `results/ab_test/portfolio_comparison.png` - Portfolio degeri grafigi
- `results/ab_test/metrics_comparison.png` - Metrikler bar chart
- `results/ab_test/ab_test_results.csv` - Sonuc tablosu

---

### 4. PSR Integration Test ✓

**Dosya:** [tests/test_psr_integration.py](../tests/test_psr_integration.py)

**Test Senaryolari:**
1. Simple reward ile environment olusturma
2. PSR reward ile environment olusturma
3. Custom PSR weights ile environment
4. Full episode completion test

**Kullanim:**
```bash
venv\Scripts\python.exe tests\test_psr_integration.py
```

---

## PSR REWARD COMPONENTS

PSR reward asagidaki 5 bilesenden olusur:

### 1. Portfolio Return (w1 = 0.50)
```python
portfolio_return = ((current_value - prev_value) / prev_value) * 100
```

### 2. Differential Sharpe Ratio - DSR (w2 = 0.30)
```python
A_t = A_{t-1} + eta * (R_t - A_{t-1})  # EMA of returns
B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})  # EMA of squared returns
DSR = (A_t - r_f) / sqrt(B_t - A_t^2)
```

### 3. MDD Penalty (w3 = 0.10)
```python
mdd = min((values - cummax) / cummax)
penalty = mdd * abs(mdd) / 10  # Quadratic penalty
```

### 4. Volatility Penalty (w4 = 0.05)
```python
annualized_vol = std(returns) * sqrt(252)
excess_vol = max(0, annualized_vol - 30%)
penalty = -(excess_vol^2) / 100
```

### 5. Trade Frequency (w5 = 0.05)
```python
trades_per_100 = (total_trades / current_step) * 100
# Optimal range: 30-70 trades per 100 steps
# Bonus in range, penalty outside
```

### Total Reward
```python
total_reward = (
    w1 * portfolio_return +
    w2 * dsr +
    w3 * mdd_penalty +
    w4 * volatility_penalty +
    w5 * trade_frequency -
    commission_penalty
)
```

---

## BEKLENEN SONUCLAR

### Hedef Metrikler (Sprint 2 Sonunda)
- **MDD:** <= -12.5% (Faz 1: -13.81%)
- **Sharpe Ratio:** >= 1.26 (koruma veya iyilestirme)
- **Total Return:** >= 35%
- **Trade Quality:** Daha az commission loss, daha optimize timing

### A/B Test Beklentileri
1. **Phase2-PSR > Phase1-Simple** (Baseline)
2. **PSR Reward > Simple Reward** (Ayni phase'de)
3. **Phase 2 > Phase 1** (Fundamental + Macro data ekstra deger katiyor)

---

## SONRAKI ADIMLAR (SPRINT 3)

### 1. Hyperparameter Tuning
- [ ] Optuna script'ini calistir (50-100 trials)
- [ ] En iyi PSR weights'i bul
- [ ] Results/optimization/ klasorunu incele

### 2. A/B Testing
- [ ] A/B test script'ini calistir
- [ ] 4 experiment karsilastir
- [ ] Results/ab_test/ klasorunu incele

### 3. Model Training
- [ ] En iyi weights ile Phase 2 modelini train et
- [ ] 100k-200k timesteps (full training)
- [ ] Validation metrics ile compare et

### 4. Academic Reporting
- [ ] Ansari et al. (2024) ile karsilastirma
- [ ] Tables ve plots olustur
- [ ] Academic paper format

---

## KRITIK DOSYALAR

### Yeni Olusturulan
- [env/trading_env.py](../env/trading_env.py) - PSR entegrasyonu
- [scripts/optimization/optimize_reward_weights.py](../scripts/optimization/optimize_reward_weights.py)
- [scripts/optimization/ab_test_rewards.py](../scripts/optimization/ab_test_rewards.py)
- [tests/test_psr_integration.py](../tests/test_psr_integration.py)

### Mevcut (Referans)
- [env/reward_functions.py](../env/reward_functions.py) - PSR calculator
- [data/fundamental_fetcher.py](../data/fundamental_fetcher.py)
- [data/macro_fetcher.py](../data/macro_fetcher.py)

---

## NOTLAR

1. **Windows Console:** ASCII karakterler kullanildi (Unicode sorunlari icin)
2. **Macro Data Timezone:** Timezone-aware comparison implement edildi
3. **EVDS API:** Bazi seriler 403 veriyor (alternatif seriler var)
4. **Test Encoding:** UTF-8 encoding fix eklendi
5. **Virtual Environment:** `venv\Scripts\python.exe` kullanilmali

---

## TEST KOMUTLARI

```bash
# PSR integration test
venv\Scripts\python.exe tests\test_psr_integration.py

# Optuna optimization (hizli test)
venv\Scripts\python.exe scripts\optimization\optimize_reward_weights.py --trials 10

# A/B testing (hizli test)
venv\Scripts\python.exe scripts\optimization\ab_test_rewards.py --timesteps 10000
```

---

## REFERANSLAR

- Ansari et al. (2024). "A Multifaceted Approach to Stock Market Trading Using Machine Learning and Sentiment Analysis"
- Moody & Saffell (2001). "Learning to Trade via Direct Reinforcement"
- TCMB EVDS API Documentation

---

**Sprint 2 Durumu:** ✅ TAMAMLANDI
**Sonraki Chat:** Sprint 3 - Hyperparameter Tuning & Model Training
