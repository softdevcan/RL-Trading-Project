# Ödül tanımı (PSR) — kodla aynı terimler

**Durum:** Faz 9 G3 (2026-11-15). Tezde ve makalelerde tek-ajan ortamının ödülü **bu metinden** yazılır. Kaynak kod: [env/reward_functions.py](../../env/reward_functions.py) (`RewardCalculator.calculate_psr_reward`) ve [env/trading_env.py](../../env/trading_env.py) (`step`). Kod değişirse bu metin aynı commit'te güncellenir.

**Seminer belgeleri (Mayıs 2026) bu konuda yanlıştır:** ödüldeki Sharpe terimini "Differential Sharpe (DSR)" diye adlandırıyorlar. Kodun varsayılan terimi bir diferansiyel Sharpe değil, EMA Sharpe'ın **düzeyi** (Faz 9 Y2).

## 1. Adım ödülü

Her adımda (bir seans) ödül şudur:

$$R_t = \operatorname{clip}_{[-100,\,100]}\big(w_1\, r_t + w_2\, s_t + w_3\, m_t + w_4\, v_t + w_5\, f_t - c_t\big)$$

Toplam sonlu değilse 0 alınır. Varsayılan ağırlıklar: w₁ = 0,50, w₂ = 0,30, w₃ = 0,10, w₄ = 0,05, w₅ = 0,05 (`RewardCalculator.__init__`). Eğitim rotası faz 2'de bu ödülü (`reward_type='psr'`), faz 1'de basit ödülü (§3) kullanır.

| Terim | Kod adı | Tanım |
|---|---|---|
| rₜ | `portfolio_return` | Portföy getirisi, **yüzde**: 100·(Vₜ − Vₜ₋₁)/Vₜ₋₁. Vₜ, işlemlerden ve bir adım ilerlemeden sonraki portföy değeri (nakit + Σ adet × kapanış) |
| sₜ | `sharpe_ratio` | Sharpe terimi; `sharpe_term` ile seçilir (§2) |
| mₜ | `mdd_penalty` | M = bölüm başından bu yana en büyük düşüş, yüzde (≤ 0); mₜ = M·\|M\|/10 |
| vₜ | `volatility_penalty` | σ = son 30 adımın rₜ'lerinin standart sapması (yüzde, ddof = 0), yıllık σ√252; vₜ = −max(0, σ√252 − 30)²/100. İkiden az getiri varsa 0 |
| fₜ | `trade_frequency` | q = 100·(bölümdeki toplam başarılı işlem)/(bölümdeki adım). T = 50 için: q < 0,6T ise −(0,6T − q)/10; q > 1,4T ise −(q − 1,4T)/10; arada ise 5·max(0, (20 − \|q − T\|)/20) |
| cₜ | `commission` | 100·(bu adımın komisyonu)/(başlangıç bakiyesi) |

## 2. Sharpe terimi sₜ (`sharpe_term`, Faz 9 G1)

Ortak tanımlar: xₜ = rₜ/100 (oran), η = 1/30 (`rolling_window` = 30), r_f = 0,15/252 (sabit yıllık %15). A₀ = B₀ = 0 ve her bölüm başında sıfırlanır.

**`'ema_level'`** (varsayılan; mevcut eğitilmiş modellerin hepsi bununla eğitildi):

$$A_t = A_{t-1} + \eta\,(x_t - A_{t-1}),\quad B_t = B_{t-1} + \eta\,(x_t^2 - B_{t-1}),\quad s_t = \tanh\!\Big(\tfrac{1}{2}\,\tfrac{A_t - r_f}{\sqrt{\max(B_t - A_t^2,\;10^{-9})}}\Big)$$

Bu, EMA ile tahmin edilen fazla getiri Sharpe'ının **düzeyidir**. Moody & Saffell'in diferansiyel Sharpe'ı değildir. Tezde "diferansiyel Sharpe" denmez; "EMA Sharpe düzeyi" denir.

**`'differential'`** (opt-in): Moody & Saffell (2001, IEEE TNN 12(4), s.4, denklem 14–15) diferansiyel Sharpe'ı. Makaledeki gibi fazla getiri x̃ₜ = xₜ − r_f üzerinde tanımlıdır. Aₜ, Bₜ x̃'nin EMA'larıdır:

$$D_t = \frac{B_{t-1}\,\Delta A_t - \tfrac{1}{2} A_{t-1}\,\Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}},\quad \Delta A_t = \tilde x_t - A_{t-1},\;\; \Delta B_t = \tilde x_t^2 - B_{t-1},\qquad s_t = \tanh(D_t/2)$$

Tezdeki adı: **"tanh ile sıkıştırılmış, ısınmalı Moody-Saffell Dₜ"**.

Tasarım varsayımları (makalede yok):
- İlk 1/η = 30 adımda Dₜ = 0 (EMA ısınması); A ve B yine güncellenir.
- Paydada varyans tabanı: max(Bₜ₋₁ − Aₜ₋₁², 10⁻⁹), `'ema_level'` ile aynı taban. Sonlu olmayan Dₜ durumunda 0 alınır. Tabansız sürümde nakitte Dₜ patlıyordu (tam nakitte en büyük |Dₜ| 293; hakem-rl İ1 #2). Tabanla durağan nakitte Dₜ → 0 olur: yeni bilgi yok.
- tanh(·/2) sıkıştırması `'ema_level'` ile ortaktır.
- Denklem 13'ün toplanabilirliği (Σ η·Dₜ ≈ ΔS) **ham** Dₜ için geçerli. Ödüle giren sıkıştırılmış terimde yalnız yaklaşık. Gerçek panelde |sₜ| > 0,9 oranı 0,02, yani tanh çoğunlukla doğrusala yakın bölgede ([sharpe_term_scale.json](../development/measurements/phase9-g1/sharpe_term_scale.json)).

Ham Dₜ `sharpe_raw` bileşenindedir. Kabul testi `python tests/test_reward_differential.py`:
- denklem 13'e göre Sₜ − Sₜ₋₁ = η·Dₜ + O(η²);
- denklem 16'daki türev;
- iki mutasyonun yakalanması.

**K4:** Milestone 0' ön elemesinde iki terim de denenir; seçim deneme kaydına girer (`config.sharpe_term`). HPO başka bir terimle arandıysa rota kaydı `hpo_reward_mismatch` ile açık kalır.

## 3. Basit ödül (faz 1)

$$R_t = \operatorname{clip}_{[-100,\,100]}\big(r_t - c_t\big)$$

rₜ ve cₜ §1'deki gibidir (`SimpleRewardCalculator`).

## 4. Tezde beyan edilecekler

- **Komisyon iki kez sayılıyor.** Vₜ komisyonu zaten içeriyor: alımda nakitten adet·fiyat·(1 + k), satımda adet·fiyat·(1 − k) ([trading_env.py](../../env/trading_env.py) `_execute_trade`). Dolayısıyla komisyon rₜ'ye (ağırlık w₁) bir kez girer, cₜ ile (ağırlık 1) bir kez daha düşülür. cₜ'nin paydası da Vₜ₋₁ değil, başlangıç bakiyesi. Davranış dondurma gereği değiştirilmedi. Tanım budur ve böyle yazılır.
- **Risksiz faiz sabit %15'tir.** Tek kaynak seriye (A8, `evaluation/risk_free.py`) bağlanması Faz 9 §9.22 madde 6'da. Ortamda nakit faiz kazanmaz (§9.2 #7).
- **fₜ bir hedef işlem sıklığına ödül verir** (T = 50 işlem / 100 adım). Bu sayı Ansari vd. (2024) uyarlamasının kodundaki bir tasarım sabitidir. Kaynakla gerekçelendirilmesi doğrulanmadı.
- **Varsayılan Sharpe terimi düşük maruziyette −1'de doyar.** Eski ortamda nakit 0 kazanır, terim ise sabit r_f'yi düşer. Fazla getiri negatif olur, varyans ise küçüktür. Eşit ağırlıklı portföy getirisinin %1'i kadar maruziyette `'ema_level'` terimi zamanın %84'ünde |sₜ| > 0,9 durumunda (ortalama −0,94); tam nakitte her zaman −1. Tam yatırımda doyma yok. `'differential'` için aynı oranlar %14 ve %3. Kaynak: [sharpe_term_scale.json](../development/measurements/phase9-g1/sharpe_term_scale.json) → `series` (r_f = 0,15). Mevcut eğitilmiş modeller bu terimle eğitildi.
- Ödül ±100'e kırpılır. Bileşen ölçekleri farklıdır: rₜ yüzde, sₜ ∈ (−1, 1), fₜ ∈ [−∞, 5].
