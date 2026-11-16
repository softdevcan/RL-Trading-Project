# Faz 10 M2 — Ortam ve karar katmanı çekirdeği: tasarım

**Sürüm:** v0.5 · 2026-11-16 · dal `feat/phase10-m2-core`
**Üst belge:** [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.2 (RL katmanı), §3.3 (karar mekanizması), §8 #13 (muhasebe denklemi).
**Durum:** saf çekirdek ve PettingZoo ortamı yazıldı ve testli; hakem-rl + hakem-finans Kabul (§10). Gerçek 30 sembollü panelde ölçüm yapıldı (§9 #1, `measurements/phase10-m2/obs_health.json`) ve fiyat limiti ön kayıtlı kuralı uygulandı: taban açılışta satış engeli (§9 #9). Performans sayısı **yok** (S1). Kalan: S-obs ufuk hizası + PC060 yeni ortamda (§9 #2–#3), sonra hakem-2.

Bu belge planın M2 tasarım maddelerini koda bağlar. Bir madde burada ve kodda farklıysa kod hatalıdır. Bu belge değişmeden kod değiştirilmez.

---

## 0. Kapsam

| Plan maddesi | Burada | Kod |
|---|---|---|
| §8 #13 zaman indeksli muhasebe denklemi | §2 | `marl/accounting.py` |
| §3.2.1 nakit faizi + G2 (A8 serisi, t−1) | §2.2 | `accounting.step_fractional`, `panel.risk_free_for` |
| §3.2.2 eylem uzayı, nakit logiti (M1 girdisi) | §3.1 | `weights.logit_bound`, `target_weights` |
| §3.2.2 + K6 risk projeksiyonu | §3.2 | `weights.project` |
| §9 #9 ön kaydı: taban açılışta satış engeli | §2.5, §3.2 e | `accounting.fill_prices(limit_down_gap)`, `project(no_sell)`, `rebalance_orders(no_sell)` |
| §3.2.2 kesirli/tam sayı, K5 emir üretimi | §4 | `marl/orders.py` |
| §3.2.3 gözlem, K18/K26 ölçeksiz özellikler, S-obs | §5 | `marl/obs_spec.py` |
| K2 ortak gözlem kurucu (bit-eş) | §5.4 | `obs_spec.build_obs_at` |
| M1 girdisi: eksik hücre maskesi | §2.3, §6 | `accounting.fill_prices`, `panel.build_arrays` |
| K1 model meta alanları (ortam kısmı) | §7 | `MarlPortfolioEnv.model_meta` |
| PettingZoo Parallel ortamı | §7 | `marl/env.py` |

**Kapsam dışı (sonraki adımlar):**
- BenchMARL görev sarmalayıcısı ve eğitim betiği (M3)
- canlı uç `POST /api/marl/daily-decision` (karar katmanı K3–K10)
- iletişim kolları (M4)
- PC060'ın yeni ortamda kurulması (§9)

---

## 1. Paket yerleşimi (plandan sapma)

Plan §3.2 yeni ortamı `env/marl_trading_env.py` olarak adlandırıyor. Kod **`marl/`** üst düzey paketine kondu:

- `env/__init__.py`, `trading_env`'i hevesle import ediyor. `env/` altındaki her modül eski ortamı ve onun bağımlılıklarını yanında getirirdi.
- K9 uyarınca üretim kodu PettingZoo/TorchRL'e bağlanmamalı; ama K2 ve K5 için canlı karar ortamla **aynı** fonksiyonu çağırmalı.

Bu iki gereklilik paketin iki katmana ayrılmasını gerektiriyor:

| Katman | Modüller | Bağımlılık | Çalıştığı ortam |
|---|---|---|---|
| Saf çekirdek | `weights`, `accounting`, `orders`, `obs_spec`, `panel` | numpy (panel: pandas) | `venv/` (canlı karar) ve `venv-marl/` (eğitim) |
| Ortam | `env` | + gymnasium, pettingzoo | yalnız `venv-marl/` |

`marl/__init__.py` hiçbir alt modülü import etmez. `env/trading_env.py` ve eski state space değişmez.

---

## 2. Muhasebe denklemi (hakem-2 #13)

### 2.1 Gösterim

- Seanslar t = 0..T−1; bunlar veri kapısından geçmiş takvimin satırlarıdır (§6).
- N sembol, sırası panelin görünme sırası (meta `symbols`).
- Ağırlık vektörü w ∈ Δᴺ: indeks 0 **nakit**, 1..N hisseler.
- Oᵗᵢ, Cᵗᵢ ham açılış ve kapanış. `fill_prices` bunlardan taşınmış seriler üretir (§2.3):
  - Cf: son geçerli kapanış
  - Of: geçerli günde O, değilse Cfₜ₋₁
- rfₜ₊₁: A8 dönem başı risksiz getiri, `evaluation.risk_free.per_period_rate`. t'de bilinen yıllık oran, t ile t+1 arasındaki takvim günü kadar bileşik uygulanır.
- c: **tek yön** maliyet = gidiş-dönüş bps / 2 / 1e4. D4 ızgarasıyla aynı (`evaluation/baselines.py`); ana senaryo 20 bps → c = 0,001.

### 2.2 Bir adım: karar t kapanışında, yürütme t+1 açılışında (K12)

| Parça | Aralık | Denklem |
|---|---|---|
| 1. Gece | kapanışₜ → açılışₜ₊₁ | g_onᵢ = Ofₜ₊₁,ᵢ / Cfₜ,ᵢ; nakit (1 + rfₜ₊₁). g₁ = w₀(1+rf) + Σᵢ wᵢ g_onᵢ; kaymış ağırlık w̃ = drift / g₁ |
| 2. Yürütme | açılışₜ₊₁ | Hedef w* = project(softmax([0, z]), w̃, …) (§3). Devir τ = Σᵢ \|w*ᵢ − w̃ᵢ\|, **yalnız hisseler**; nakit devre girmez. Maliyet payı c·τ |
| 3. Seans içi | açılışₜ₊₁ → kapanışₜ₊₁ | g_dayᵢ = Cfₜ₊₁,ᵢ / Ofₜ₊₁,ᵢ; g₂ = w*₀ + Σᵢ w*ᵢ g_dayᵢ. Nakit seans içinde faiz **kazanmaz** |

- **Portföy getirisi:** R = g₁ (1 − cτ) g₂ − 1
- **Fazla getiri:** X = R − rfₜ₊₁
- **Ajan katkısı:** kᵢ = wᵢ (g_onᵢ − 1 − rf) + w*ᵢ (g_dayᵢ − 1) − c \|w*ᵢ − w̃ᵢ\|
- **Ödül:** rᵢ = ölçek · (α kᵢ + (1 − α) G), G = X (`'excess'`) ya da X üzerinde Moody-Saffell Dₜ (`'differential'`). α ∈ [0, 1); α = 1 reddedilir (vision §3.3 β).
- **Birincil ödül `'excess'` — KARAR (2026-11-16, kullanıcı; hakem-rl M2 #1–#2).** α ön elemesi {0; 0,25; 0,5} yalnız `'excess'` ile yapılır. `'differential'` yalnız α = 0 ile kurulabilir (ablasyon); başka α `ValueError`. Gerekçe:
  - Ölçek: tanh(Dₜ/2) ∈ (−1, 1), kᵢ ise günlük getiri mertebesinde. α = 0,25'te portföy teriminin std'si bireysel terimin 242 katı (`'excess'`te 6,3; sentetik N = 4, `measurements/phase10-m2/synthetic_checks.json` → `reward_term_scale`). DSR'de α fiilen 0 olur.
  - Markov: Dₜ DSR'nin (a, b) durumuna bağlı, bu durum gözlemde yok; her bölüm başında sıfırlanır.

**Tasarım varsayımları:**
1. **Faiz yalnız gece kazanılır.** Fonlama faizi gecelik bir orandır; t kapanışında tutulan nakit t → t+1 gecesinde rfₜ₊₁ kazanır. Hedefin nakit kısmı ilk kez bir sonraki gece faiz alır.
2. **Maliyet portföyden orantılı düşer.** Jiang vd. (2017) kesin işlem kalanı μ'su yerine birinci derece c·τ kullanılır. Bunun bedeli §2.4'te ölçüldü.
3. **Katkı ayrışımı yalnız birinci derecedir** (plan §8 #13). Σᵢ kᵢ = X + ikinci derece çapraz terimler; kalan `contrib_residual` her adımda `last_info`'da raporlanır. Çapraz terim (g_on − 1)(g_day − 1) günlük ~1e-4, yani maliyet mertebesinde. **Gece terimi wᵢ(g_onᵢ − 1 − rf) bu adımdaki eylemden bağımsızdır;** bir önceki kararın sonucudur. kᵢ "ajanın bu adımdaki kararının katkısı" diye okunmaz (hakem-finans M2 #6). Nakdin fazla getiri katkısı tanım gereği 0'dır: Σᵢ wᵢ rf + w₀ rf − rf = 0. "Katkılar toplamı = portföy fazla getirisi" iddiası **yalnız bu basit getiri terimi** için yapılır; diferansiyel Sharpe için yapılmaz.
4. **Eski ortamla fark:** komisyon eski ortamda ödülde iki kez sayılıyor (G3). Burada yalnız bir kez sayılır: `test_marl_core` [3], "maliyet bir kez".

### 2.3 Eksik hücre (M1 girdisi)

- **Kural:** hücre atılmaz. Geçersiz gün (O ya da C NaN veya ≤ 0) için:
  - g = 1, yani getiri 0
  - hisse o gün `tradable = False` olur ve projeksiyonda dondurulur (§3.2)
  - değerleme sembolün kendi son kapanışıyla yapılır
- **Hücre yeniden geçerli olunca:** g_on = O / son geçerli kapanış. Aradaki hareket o gün yazılır.
- **Listelenmemiş sembol:** Cf = NaN, ağırlığı 0 kalır.
- **Listeleme kilidi — KARAR (2026-11-16, kullanıcı; hakem-finans M2 #1).** Panelde ilk satırdan sonra listelenen sembol ilk geçerli gününden itibaren **20 seans** işlem göremez (`LISTING_LOCKOUT`, ortamda `listing_lockout`). Gerekçe: BIST'te halka arz sonrası ilk seanslar çoğunlukla tavan serisi; açılıştan tam dolum gerçekçi değil. Kilit nedenseldir (ilk geçerli gün o gün bilinir; `test_marl_core` [3] B0). Panel başında zaten listeli sembolün halka arz tarihi panelden görünmez; bu beyan edilir. Etkilenen hücre sayısı gerçek panel ölçümünde (`market.listing_lockout_cells`); kilit sonrası ilk 10 seanstaki ±%9,5 açılış boşluğu ayrıca (`market.post_lockout_first10_gaps`; tavan serisi 20 seanstan uzunsa kilit uzatılır ya da §9 #9 yön kısıtı bu hücreleri kapsar).
  - **Panel kesimi:** kilit "panelin ilk satırından sonra listelenen" sembole uygulanır. Panel her zaman `RL_DATA_START`'tan (K17) bütün olarak kurulur, eğitim/doğrulama/test pencereleri bunun içinden `start_row`/`end_row` ile kesilir. Bu yüzden kilit kümesi pencere seçimine bağlı değildir. Panel başka bir başlangıçla kurulursa kilit kümesi değişir (hakem-rl M2 ikinci okuma #9).
  - **Gerçek panel ölçümü (`measurements/phase10-m2/obs_health.json`, `market.listing_lockout_*`):** 3 sembol, 60 hücre. Üçü de **gerçek halka arz değil, veri artefaktı**; pencerede (2013-06..2020-12) 30 sembolden hiçbiri halka arz olmadı:
    - EKGYO: 2010'dan beri işlem görüyor; ilk 5 satır bayat hücre olarak atıldı, ilk geçerli gün 2013-06-10.
    - ODAS (halka arz 2013-05-21): gösterge ısınması 17 satır atıldı, ilk geçerli gün 2013-06-27.
    - PGSUS (halka arz 2013-04-26): yfinance'te 2013-05..07 boşluğu ve 24 satır ısınma atıldı, ilk geçerli gün 2013-08-28.
    EKGYO ve ODAS kilidi 59 satırlık ısınmanın (`WARMUP_ROWS`) içinde kalır, ortamı etkilemez. PGSUS kilidi 2013-08-28..2013-09-25 ortam penceresinin ilk ~20 seansındadır: bu sürede PGSUS alınamaz. Yön muhafazakâr (işlem yok); kod değişmez, beyan edilir. Kilit sonrası ilk 10 seansta ±%9,5 açılış boşluğu 0 (`post_lockout_first10_gaps`), yani kilit uzatılmaz.
  - **Gözlemde:** kilit ayrı bir bayrak değildir. Kilitli hücrede `feat_ok` = 0'dır (r20 ve vlog_z listelemeden sonra en az 20–40 seans ister), yani ajan kilidi `feat_ok` ile görür. Değişmez `test_marl_core` [3]'te. Kilitli ajanın eylemi etkisiz, kᵢ = 0; ödülü yalnız (1 − α)G. Bu etkisiz ajan gürültüsü kabul edilir, PPO kaybında maske yok (hakem-rl M2 ikinci okuma #8).
- **Açılış geçerli, kapanış eksik hücre** (satıcı veri boşluğu) şu an `valid = False` sayılır ve hisse donar; açılışta kapanışın eksik olacağı bilinmez, bu yüzden bu bir ileriye bakış sayılabilir (hakem-finans M2 S3). Muhafazakâr yön (işlem yapılmaz). Gerçek panelde **0 hücre** (`market.open_valid_close_missing_cells`, `measurements/phase10-m2/obs_health.json`); düzeltme gerekmez.
- **Aynı yaklaşım:** eski ortamdaki `missing_data_mode='own_last'` ve `evaluation/baselines.py` aynı taşıma mantığını kullanıyor.

### 2.4 Ölçüm: maliyet yaklaşımının bedeli

`test_marl_env` [7] aynı eylem dizisini iki modda koşar: 1e9 TL sermayeli tam sayı modu ve kesirli mod (sentetik panel, 79 adım).

| Maliyet | max \|R_tam − R_kesirli\| | Yorum |
|---|---|---|
| 0 bps | 7,5e-10 | yalnız tam sayı yuvarlaması |
| 20 bps | 1,0e-4 | maliyetin nakitten değil portföyden düşmesi ve son alımların komisyon kadar kısa kalması; mertebesi c·τ·\|g−1\| |

Fark günlük getiri ölçeğinde küçük, ama sıfır değil. Nihai backtest'ler tam sayı modunda koşulur (plan §3.2.2), bu yüzden raporlanan sayılar kesin muhasebeyle üretilir. Eğitimdeki kesirli mod bir yaklaşımdır ve böyle beyan edilir.

### 2.5 Yürütme ve veri beyanları (hakem-finans M2)

- **Fiyat tabanı — KARAR (2026-11-16, kullanıcı).** `data_fetcher.py` yfinance `history()`'yi `auto_adjust` vermeden çağırır; varsayılan True, yani OHLC bölünme ve temettüye göre düzeltilmiş. g_on ve g_day toplam getiri oranıdır; bu doğru. Tam sayı modunda fiyat seviyesi ve lot gerçek değildir, seviye sonraki şirket olaylarına bağlıdır. Tam sayı modu muhasebe doğruluğu (nakit/adet tutarlılığı) içindir, gerçek lot ve fiyat basamağı iddiası yapılmaz. `model_meta()['price_basis']`.
- **Fiyat limiti (tavan/taban).** KARAR (2026-11-16, kullanıcı): önce ölçülür; karar kuralı ölçümden önce yazıldı (§9 #9). **Ölçüm** (`measurements/phase10-m2/obs_health.json`, `market.open_gap_limit_cells['0.095']`, `RL_DATA_START`..2020-12-31, 56.919 geçerli gece geçişi):

  | Yön | Hücre | Oran | Eşik (ön kayıt) | Sonuç |
  |---|---|---|---|---|
  | Taban (açılış boşluğu ≤ −%9,5) | 42 | %0,074 | `frac_down` ≥ %0,05 | **satış engeli uygulandı** |
  | Tavan (≥ +%9,5) | 35 | %0,061 | `frac_up` ≥ %0,1 | engel yok, beyan |

  ±%19,5 boşlukta 1 tavan, 2 taban hücresi var. **Uygulama:** `fill_prices(limit_down_gap=0.095)` → `no_sell[t] = valid ∧ Oₜ/Cfₜ₋₁ ≤ 0,905`. Projeksiyon §3.2 e, K5 bekçisi §4. Ortamın gördüğü engelli hücre sayısı 38: ısınma ve kilit hücreleri düşülünce kalan; eğitim satırlarında 18, doğrulama satırlarında 20 (`market.no_sell_cells*`). Engel t+1 açılışında bilinir, yürütme de t+1 açılışında olur; bu yüzden sızıntı değil. Canlı karar `no_sell=None` verir, çünkü karar kapanışta verilir ve tabandaki satış emri borsada dolmaz. **Beyan:** tavan açılışta alım tam dolar (iyimser), oran ön kayıtlı eşiğin altında. Eşik bir açılış boşluğu yaklaşımıdır, gerçek limit fiyatı değildir: baz fiyat düzeltilmiş kapanıştır ve seans içinde tabana inen fiyat modellenmez.
- **Nakit ve takas.** Nakit getirisi brüt TCMB ağırlıklı ortalama fonlama maliyeti (TP.APIFON4, A8); stopaj ve repo spread'i yok. Satış geliri aynı açılışta yeniden kullanılır (T+2 takas modellenmez). Tek yön c açılış mezatı kaymasını da kapsıyor sayılır; ayrı kayma modeli yok. Sonuçlar D4 maliyet ızgarasıyla (0/10/20/25/50 bps) raporlanır.
- **Evren (hakem-finans M2 #2).** Panel dondurulmuş bir 30'lu liste (E0); tarihsel üyelik değil, 6 hissesi bugün XU030'da değil (`data/CLAUDE.md`). Evren sonradan seçilmiştir. **Mutlak performans iddiası yapılmaz;** yalnız aynı evrende göreli karşılaştırma (MARL ↔ SARL ↔ baseline). E1 üyelik maskesi hazır olunca eval'e bağlanır. Bu beyan tez metnine de girer.

---

## 3. Eylem ve risk projeksiyonu

### 3.1 Nakit logiti — KARAR (2026-11-16, kullanıcı)

- **Problem (M1 ölçümü):** z₀ = 0 sabit ve sınır ±5 iken 30 ajanla nakit en fazla 1/(1 + 30e⁻⁵) ≈ %83 olabiliyor; portföy tamamen nakde geçemiyor.
- **Karar:** ajan logit sınırı **B = 5 + ln N**, z₀ = 0 sabit kalır.
  - Tüm ajanlar −B'deyken nakit 1/(1 + e⁻⁵) ≈ **%99,33** olur ve bu değer N'den bağımsızdır. N = 30 için B = 8,40.
  - Ajanlar homojen kalır, parametre paylaşımı bozulmaz.
  - Plan §3.2.2'deki "z₀ = 0" varsayımı ve SARL-ağırlık kolunun aynı eylem temsili (§3.2.4) korunur.
- **Reddedilen seçenekler:**
  - Her ajanın nakde oy vermesi: eylem boyutu 2 olur, nakit kararının kredi ataması dağılır.
  - Ayrı nakit ajanı: heterojen olur, plan §2.2 madde 5'e aykırı.
- **Test:** `test_marl_core` [1], N ∈ {1, 5, 30, 100}.

### 3.2 Projeksiyon `project(u, w̃, tradable, Constraints, groups)`

Öğrenmeyen sabit bir işlemdir, merkezi bir dağıtıcı değildir (plan §3.2.2). Ortam ve canlı karar aynı fonksiyonu çağırır (K6). Adımların sırası sabittir:

| Adım | Kural |
|---|---|
| a. Dondurma | İşlem göremeyen hisse kaymış ağırlığını korur. Kalan kütle nakit ve işlem görenlere u oranında dağıtılır |
| b. Bant b | \|w*ᵢ − w̃ᵢ\| < b ise hisse w̃ᵢ'de kalır. Hisse toplamı 1'i aşarsa bantta kalmayan hisseler orantılı küçültülür |
| c. Pozisyon sınırı | İşlem gören hissede w*ᵢ ≤ max_position; fazlası **nakde** gider |
| d. Grup sınırı | Grubun işlem gören kısmı orantılı küçültülür, fazlası nakde gider. Dondurulmuş kısım sınırdan düşülür |
| e. Satış engeli (§9 #9) | `no_sell` hissesinin hedefi w̃ᵢ'nin altına düşerse hisse o gün **dondurulur** ve a–d baştan koşulur. Dondurulan küme yalnız büyür, en fazla N turda biter. Alım yönü açık kalır. Rapor `n_sell_blocked` |

**Tasarım varsayımları:**
- Sınırlar yalnız küçültür. Fazla kütle başka hisselere dağıtılmaz, nakde gider. Böylece projeksiyon tek geçişte ve deterministik olur; hedef uzun-yalnız kalır, toplamı 1'dir, nakit ≥ 0'dır.
- Dondurulmuş hisse sınırı aşabilir, çünkü o gün satılamaz. Bu durum `frozen_over_cap` ile raporlanır. Satış engelli hisse de dondurulmuş sayılır; grup sınırını aşabilir.
- Satış engeli yeni bir mekanizma değildir, dondurmanın yön koşullu hâlidir. `no_sell=None` ile sonuç eskisiyle bit-eştir (`test_marl_core` [2]).
- Sınır fazlası nakde gittiği ve nakit faiz aldığı için sınırlar oynaklığı ve fazla getiriyi sistematik etkiler. **Baseline'lar (1/N, al-tut) aynı `project` sınırları ve aynı rf ile de koşulur** ya da fark duyarlılık satırı olarak verilir (hakem-finans M2 #7; M3/M8).
- Grup kimliği E2 sektörüdür (`data.bist30_symbols.load_sector_groups`). Ortamda `groups=` parametresiyle verilir ve gözleme sektör one-hot'u olarak da girer.

**Test:** `test_marl_core` [2], 500 rastgele durum.

---

## 4. K5 — tam sayı emir üretimi `rebalance_orders`

- **Ortak kullanım:** ortamın tam sayı modu ve canlı karar aynı fonksiyonu çağırır.
- **Çıktı:** mevcut karar sözlüğü şeması (`interpret_actions_with_risk`) ile `target_weight` ve `current_weight` ek alanları.
- **Uyumluluk:** `portfolio.apply_decisions` değişmeden çalışır. Test aynı nakdi ve aynı adedi doğrular.
- **Satış engeli:** `no_sell` hissesinde hedef adet mevcut adedin altına inmez (`reason='satis engeli (taban acilis)'`). `project` aynı maskeyle hedefi zaten w̃ᵢ'de tutar; bu, yuvarlamaya karşı bekçidir.

**Nakit tanımları** CLAUDE.md "Kağıt portföy" ile aynı:
- BUY maliyeti komisyon **dahil**
- SELL geliri komisyon **düşülmüş**

**Algoritma (deterministik):**
1. V = nakit + Σ hᵢpᵢ.
2. Hedef adet nᵢ = ⌊w*ᵢ V / pᵢ⌋. İşlem göremeyen hisse için nᵢ = hᵢ.
3. Önce satışlar.
4. Sonra alımlar, eksik değeri büyükten küçüğe sırayla. Her alım nakdin yettiği kadarla sınırlıdır.

Hedef komisyonsuz V ile kurulduğu için son alımlar birkaç adet eksik kalabilir. Bu fark `summary.weight_gap` ile raporlanır (plan §3.2.2 "yuvarlama etkisi raporlanır").

**Tam sayı modunda fiyat:** fiyatlar yfinance auto_adjust düzeltmeli. Tam sayı adet düzeltilmiş fiyatla bir yaklaşımdır; temettü günü nakit akışı modellenmez. Kâğıt portföy de aynı varsayımla çalışıyor (beyan).

**Test:** `test_marl_core` [4]; `test_marl_env` [7].

---

## 5. Gözlem `marl-obs-v1`

### 5.1 Özellikler (hepsi ölçekten bağımsız)

| Ad | Tanım | Not |
|---|---|---|
| r1, r5, r20 | log(Cfₜ / Cfₜ₋ₖ) | |
| range | (H − L) / C | |
| gap | log(Oₜ / Cfₜ₋₁) | |
| vol20 | r1'in 20 günlük std'si | |
| rsi | RSI/100 − 0,5 | |
| cci | CCI/200 | |
| adx | ADX/100 | **'wilder'**. Legacy ADX (hep 0) `panel.build_arrays`'te reddedilir (K18) |
| macd_rel | MACD / C | K26: TL cinsinden MACD testte %75 doygundu |
| vlog_z | log(1+hacim)'in 60 günlük nedensel kayan z'si | K26 hacim adayı; sermaye artırımının yarattığı seviye kaymasına dayanıklı |

- **Türbülans yok.** K26 ölçümünde 30 sembolde tanh(·/2) %99,7 doymuştu.
- **Isınma:** 59 satır (`WARMUP_ROWS`). Bu satırlar yalnız bağlam olarak kullanılır; ortam penceresi ancak bu satırlardan sonra başlayabilir.

### 5.2 Normalizasyon (Y12, Y13)

- Her özelliğin havuzlanmış ortalama ve std'si **yalnız eğitim satırlarından** hesaplanır: `fit_norm_stats`, ortamda `norm_stats='fit'`.
- z = clip((x − μ)/σ, −5, 5). İstatistik model meta'sına `norm_stats` olarak yazılır.
- **Held-out ortam:** eğitim ortamının istatistiğini almak **zorundadır**.
  - `norm_stats=None` hata verir.
  - Eval ortamında `'fit'` hata verir.
  - Sessiz varsayılan yok; B9 dersi.
- **Geçersiz hücre** (eksik gün, ısınma, listelenmemiş): özellikler 0 olur ve `feat_ok` = 0 taşınır.

### 5.3 Ajanın yerel gözlemi ve eleştirmen durumu

**Yerel gözlem (aktör), sırayla:**
1. z özellikleri (F = 11)
2. feat_ok
3. wᵢ, w_nakit
4. ajan kimliği one-hot (N)
5. opsiyonel: grup one-hot (G)
6. opsiyonel S-obs: p_up_cal − 0,5, z(pred_logret), z(model_disagreement), pred_ok
7. opsiyonel ek özellikler (E): rejim veya makro, tüm ajanlara aynı değer

Boyut: F + 3 + N + G + 4·[S-obs] + E. N = 30'da ek özellik yokken 44.

**İleriye bakma koruması:** t+1 hücresinin geçerliliği gözleme **konmaz**, çünkü t+1 açılışında bilinir. Gözlem t'deki geçerliliği taşır.

**Eksik tahmin** maskelenir, 0,5'e düşülmez (plan §3.1.3).

**Küresel durum (MAPPO eleştirmeni):** w (N+1), z'nin kesitsel ortalaması (F), geçerli hücre oranı, E. Plan §3.2.3: eleştirmene yerel ve küresel özellikler birlikte verilir; yerel kısım BenchMARL'ın merkezi eleştirmeninde birleştirilir (M3).

### 5.4 K2 paritesi

- **İki yol:**
  - Ortam özellikleri tüm panel için bir kez hesaplar.
  - Canlı karar `build_obs_at(spec, panel[:t+1], norm_stats, w)` çağırır.
- **Bit-eşlik testleri:**
  - `test_marl_core` [5]: 6 tarih, tahmin, grup ve ek özelliklerle
  - `test_marl_env` [4]: 60 adım, ortamın kendi w'siyle
- **B0 aynı testlerde sınanıyor:** t sonrası satırlar değişince t gözlemi değişmez.
- **Mutasyon:** ileriye bakan bir özellik B0 kontrolünü kırar.

---

## 6. Veri hattı

`scripts/marl/prepare_panel_m2.py` ana venv'de koşar. Hat:

1. `stock_data_with_indicators.csv`
2. `with_adx('wilder')`
3. `prepare_rl_panel(drop_indicator_warmup=True)`. Rota ile aynı kapı: hayalet gün atılır, bayat hücre eksik sayılır.
4. `marl.panel.build_arrays`: takvim tam kalır, hücre NaN kalır.
5. A8 rf (`risk_free_for`; seri yoksa `RiskFreeUnavailable`).
6. `.npz` + SHA-256 + meta (kaynak SHA, commit, kapı raporu).

**Diğer kurallar:**
- Ortam yalnız `.npz` okur (`panel.load_npz`); M1 spike kalıbıyla aynı (K9).
- `--health-only`, rf olmadan da koşar ve gözlem sağlığını ölçer (K18: özellik başına kırpma oranı ve sabitlik; eğitim + doğrulama satırları).
- `.npz` yazımında `--json` aynı ölçüm alanlarına `npz` (göreli yol, SHA-256) ve `risk_free` (kapsam) ekler. Panel `data/marl/panel_m2.npz` (git dışı).
- Varsayılan pencere sonu 2020-12-31. 2021, m0p-v2 doğrulayıcı penceresi olduğu için ona dokunulmaz.

---

## 7. Ortam `MarlPortfolioEnv` (PettingZoo Parallel)

```python
tr = MarlPortfolioEnv(arrays, start_row, end_row, norm_stats='fit', mode='train', alpha=0.25,
                      cost_bps_roundtrip=20, constraints=Constraints(), groups=None,
                      reward_term='excess', reward_scale=100, episode_len=252,
                      integer_shares=False, pred=None, extras=None)
ev = MarlPortfolioEnv(arrays, s, e, norm_stats=tr.norm_stats, mode='eval', ...)
```

**Ajanlar ve uzaylar:**
- Ajanlar `agent_0..agent_{N−1}`. TorchRL bunları tek grup `agent` altında toplar; ajan ekseni tensör boyutudur (`test_marl_env` [8]).
- Eylem Box[−B, B]¹; gözlem Box[local_dim].
- `state()` ve `state_space` MAPPO eleştirmeni için.

**Bölümler:**
- `train` modu: pencere içinde rastgele başlangıç, uzunluk `episode_len`.
- `eval` modu: pencerenin tamamı.
- Her bölüm tam nakitle başlar.
- **`'differential'` ısınması:** DSR durumu her bölüm başında sıfırlanır ve ilk 1/η adımda Dₜ = 0 verir. Bölüm uzunluğu ≤ 1/η ise G bütün bölüm boyunca 0 olur ve ödül sessizce yalnız α·kᵢ'ye iner. Varsayılan η = 1/252 ve `episode_len` = 252 tam bu durumdaydı (öz-kontrol 2026-11-16: 100 ve 252 adımda sıfır olmayan pay 0, 300 adımda 0,16). Ortam bu durumda `ValueError` verir; bekçi yalnız `mode='train'`'de çalışır (eval'de ödül kullanılmaz; hakem-rl M2 #3). `'differential'` artık yalnız α = 0 ablasyonu (§2.2), bu yüzden η seçimi ablasyonun ön kaydında yapılır.

**Diğer:**
- `last_info` portföy düzeyindeki ayrıntıyı taşır: R, X, devir, nakit ağırlığı, projeksiyon raporu, katkı artığı. PettingZoo `infos` boş bırakılır; TorchRL spec'i sabit kalsın diye.
- `model_meta()`, plan §3.3 K1'deki ortam alanlarını üretir: kind, action_type, symbols, agent_ids, obs_spec, norm_stats, logit_bound, risk_projection, maliyet, yürütme, ödül, train_range. Eğitim betiği bunlara `run_id`, `seed`, `prediction_source` ve `comm_arm` ekler (M3).

---

## 8. Testler

| Test | Ortam | Kontrol | Kapsam |
|---|---|---|---|
| `python tests/test_marl_core.py` | `venv/` | 58 | logit sınırı, projeksiyon (500 rastgele, satış engeliyle), satış engeli (dondurma, alım açık, `None` bit-eş, maske eşiği + B0, K5 bekçisi), muhasebe (elle hesap, maliyet bir kez, katkı), K5 + `apply_decisions` gidiş-dönüş, gözlem B0 + mutasyon + K2, panel |
| `venv-marl\Scripts\python tests/test_marl_env.py` | `venv-marl/` | 36 | `parallel_api_test`, açık hatalar, tohum, K2 (60 adım), değer = Π(1+R), eksik hücre, listeleme kilidi (kesirli + tam sayı), satış engeli (kesirli + tam sayı; engel 0 kontrol kolu), held-out ölçek, tam sayı ↔ kesirli (§2.4), DSR (yalnız α = 0; eğitim bölümü ısınmadan kısaysa hata, eval'de değil; ısınma sonrası G ≠ 0), TorchRL sarmalayıcı |

**Mutasyon (2026-11-16):** `project`'te satış engeli döngüsü kapatılınca (`tr_eff &= ~hit` → `break`) `test_marl_core` [2]'de iki kontrol kırılır.

---

## 9. Açık maddeler (M2 kapanmadan önce)

| # | Madde | Neden açık | Sıradaki |
|---|---|---|---|
| 1 | ~~**Gerçek panelde ölçüm**~~ **yapıldı (2026-11-16, ev makinesi)** | `measurements/phase10-m2/obs_health.json` (kaynak CSV SHA `3aec450d…`). T = 1902, N = 30, 92 eksik günlü satır, eksik hücre %0,20. **Gözlem sağlığı:** 11 özelliğin hiçbiri sabit değil. Eğitim satırlarında (2013-08-27..2018-12-31) kırpma ≤ %0,45 (`range`). Eğitim istatistiğiyle doğrulama satırlarında (2019-01-02..2020-12-31) ≤ %1,12 (`range`), `macd_rel` %1,05, diğerleri ≤ %0,74. M2 için önceden yazılmış bir kırpma eşiği yoktu; sayı beyan edilir, sonradan eşik konmaz. Kilit, açılış-geçerli/kapanış-eksik ve fiyat limiti §2.3 ve §2.5'te. **rf ve panel:** `data/marl/panel_m2.npz` (git dışı), SHA `a5dcf9a6…` (`npz.sha256`); rf 1902 satırın 1901'inde sonlu, satır 0 tanım gereği yok (`risk_free`). Gerçek panelde eğitim + doğrulama ortamı uçtan uca koştu (duman, performans sayısı değil) | — |
| 2 | **S-obs ufuk hizası.** P tablosunun t satırı kapanışₜ → kapanışₜ₊₁ yönünü tahmin ediyor. t+1 açılışında yürütmede gece parçası eski ağırlıkla geçiyor, ajanın yeni hedefi bu parçayı yakalayamıyor | Plan §3.2.7 S-obs: "sinyal ufku ile ödül hizası sınanmalı" | PC060 pozitif kontrolü yeni ortamda: bilinen sinyalin değeri gece ve seans içi parçalarına ayrıştırılır |
| 3 | **PC060 yeni ortamda** | §9.37–9.39: eski ortamdaki politikalar bloktan değer üretmiyor | Madde 2 ile birlikte; M3 eğitimi öncesi |
| 4 | **BenchMARL görevi + eğitim betiği** | M3 kapsamı | `scripts/spike_m1/benchmarl_glue.py` kalıbıyla `marl/benchmarl_task.py` |
| 5 | **Canlı uç** (`POST /api/marl/daily-decision`, `RequireWriter`, `ws.use_workspace`) | Seçilecek MARL modeli yok (K27 ile aynı gerekçe) | Karar katmanında; K2/K5 fonksiyonları hazır |
| 6 | **M4 parametre eşitliği** (`none` 77.314 ↔ `attn` 27.906) | M4 kapsamı | M4 |
| 7 | **Hakem incelemesi** | Muhasebe ve ödül tanımı bir RL + finans iddiası | `hakem-rl` + `hakem-finans` paralel, sonra `hakem-2`; paket: bu belge + `marl/` + iki test |
| 8 | ~~DSR ısınması ↔ bölüm uzunluğu~~ **kapandı (karar 2026-11-16)** | Birincil ödül `'excess'`; `'differential'` yalnız α = 0 ablasyonu (§2.2). Isınma bekçisi yalnız eğitimde | η, ablasyonun ön kaydında |
| 9 | **Fiyat limiti — ÖN KAYIT (2026-11-16, kullanıcı; hakem-finans ikinci okuma #8), ölçümden ÖNCE.** **Uygulandı (2026-11-16):** `frac_down` %0,074 → satış engeli; `frac_up` %0,061 → beyan (§2.5) | Ölçü: `market.open_gap_limit_cells['0.095']`, pencere `RL_DATA_START`..2020-12-31, tüm geçerli hücreler. **Taban:** `frac_down` ≥ %0,05 → açılış boşluğu ≤ −%9,5 olan hücrede SATIŞ engeli. **Tavan:** `frac_up` ≥ %0,1 → açılış boşluğu ≥ +%9,5 olan hücrede ALIM engeli. Eşiğin altında kalan yön yalnız beyan. Kilit sonrası tavan serisi (`post_lockout_first10_gaps`) tavan kuralına dahil. Eşikler ölçüm görüldükten sonra değiştirilmez | Ev makinesinde ölçüm → kurala göre `project`'e yön sınırı (alt/üst) ya da beyan |
| 10 | **M3'e taşınan RL maddeleri** (hakem-rl M2 #4–#7) | Yerel gözleme piyasa bağlamı (kesitsel ortalama z, geçerli oran) ↔ yok; eleştirmene tam z (N×F) ↔ ortalama; TorchRL'de bölüm sonu `terminated=False` ve bootstrap V(sₜ₊₁) ≠ 0 duman testi; başlangıç log-std ~0,2–0,3 / tanh ezmesi, eval'de ortalama eylem | M3 görev + eğitim betiği; gözlem/eleştirmen kolları ablasyon olarak ön kayda |
| 11 | **Baseline'lar aynı sınırla** (hakem-finans M2 #7, ikinci okuma #10) | 1/N ve al-tut aynı `fill_prices(listing_lockout=20)`, aynı `project` ve aynı rf ile; sınırsız 1/N ayrı duyarlılık satırı | M3/M8 değerlendirme |
| 12 | **Evren beyanı tez metnine** (hakem-finans M2 #2) | Mutlak performans iddiası yok (§2.5); E1 maskesi hazır olunca eval'e bağlanır | Tez metni; E1 |

---

## 10. Hakem incelemesi (2026-11-16, uç `db8ef48`)

Paket istemin içinde verildi (`review-panel.md`). Maliyet: hakem-rl ~33 bin token / 5 araç; hakem-finans ~52 bin / 13. İkisi de Küçük revizyon, test koşmadı. Ölçümleri ana oturum yaptı (`synthetic_checks.json`).

| # | Kaynak | Cid. | Bulgu | Sonuç |
|---|---|---|---|---|
| R1 | hakem-rl | BÜYÜK | `'differential'`te α iki terimde aynı anlamı taşımıyor (ölçüldü: std oranı 242, `'excess'`te 6,3) | Karar: birincil `'excess'`, DSR yalnız α = 0 (§2.2); kodda zorlanıyor, test |
| R2 | hakem-rl | BÜYÜK | DSR (a, b) gözlemde yok → Markov değil | R1 kararıyla DSR yalnız ablasyon; ablasyon raporunda beyan |
| R3 | hakem-rl | KÜÇÜK | Isınma bekçisi eval'de de tetikleniyor | Düzeltildi (yalnız `mode='train'`), test |
| R4–R7 | hakem-rl | KÜÇÜK | Yerel gözlemde piyasa bağlamı yok; eleştirmen durumu zayıf; truncation/bootstrap çerçeveye bağlı; başlangıç gürültüsü devir üretir | §9 #10 (M3) |
| F1 | hakem-finans | BÜYÜK | Geç listelemede ilk gün açılıştan tam dolum (halka arz tavan serisi) | Karar: 20 seans kilit (§2.3); kod + B0 + mutasyon (kilit 0 → test kırılır) |
| F2 | hakem-finans | BÜYÜK | Evren sonradan seçilmiş; paketteki "bugünkü XU030" ifadesi yanlıştı | Beyan (§2.5), §9 #12 |
| F3 | hakem-finans | doğrulandı | Fiyatlar düzeltilmiş (yfinance `auto_adjust` varsayılanı True) | Karar: düzeltilmiş kalır + beyan (§2.5), `model_meta.price_basis` |
| F4 | hakem-finans | KÜÇÜK | Tavan/taban kilidi modellenmedi | Karar: önce ölçüm (§2.5, §9 #9); ölçüm alanı `prepare_panel_m2.py`'ye eklendi |
| F5 | hakem-finans | KÜÇÜK | Nakit/T+2/kayma varsayımları yazılı değil | Beyan (§2.5) |
| F6 | hakem-finans | KÜÇÜK | Katkı iddiası kapsamı; gece terimi önceki eyleme ait | Daraltıldı (§2.2 varsayım 3) |
| F7 | hakem-finans | KÜÇÜK | Sınır fazlası nakde → baseline aynı sınırla koşulmalı | §3.2, §9 #11 |
| F-S3 | hakem-finans | KÜÇÜK | `valid` kapanışı da içeriyor (açılışta bilinmez) | Beyan + ölçüm alanı (§2.3); sıfır değilse düzeltilir |

**İkinci okuma (SendMessage, yalnız `git diff db8ef48..28693d2`):** hakem-rl **Kabul**, hakem-finans **Kabul (koşullu)**. Yeni BÜYÜK yok.

| # | Kaynak | Cid. | Bulgu | Sonuç |
|---|---|---|---|---|
| R8 | hakem-rl | KÜÇÜK | Kilit gözlemde ayrı bayrak değil | Kilitli hücrede feat_ok = 0 (ana oturum gösterdi); değişmez testi `test_marl_core` [3]; etkisiz ajan gürültüsü kabul (§2.3) |
| R9 | hakem-rl | KÜÇÜK | Kilit kümesi panel kesimine bağlı | Panel hep `RL_DATA_START`'tan kurulur, pencere içinden kesilir (§2.3) |
| F8 | hakem-finans | KÜÇÜK (koşul) | Fiyat limitinde "seyrek" eşiği ölçümden önce yazılmalı; taban yönü ayrı | Ön kayıt yazıldı (kullanıcı kararı): taban %0,05, tavan %0,1 (§9 #9) |
| F9 | hakem-finans | KÜÇÜK | Tavan serisi 20 seanstan uzun sürebilir | Ölçüme `post_lockout_first10_gaps` eklendi (§2.3) |
| F10 | hakem-finans | KÜÇÜK | Baseline aynı kilit ve `tradable` kuralıyla | §9 #11 |

---

## Versiyon geçmişi

- **v0.5 (2026-11-16, ev makinesi):** §9 #1 gerçek panel ölçümü (`measurements/phase10-m2/obs_health.json`, doğrulama satırları dahil). §9 #9 ön kayıtlı kural uygulandı: taban açılışta satış engeli (`fill_prices(limit_down_gap)`, `project(no_sell)` adım e, K5 bekçisi); tavan beyan. Listeleme kilidi kümesinin artefakt olduğu beyan edildi (§2.3). `prepare_panel_m2.py` JSON'a yerel yol yazmıyor. Testler 58 + 36.
- **v0.4 (2026-11-16):** İkinci okuma: hakem-rl Kabul, hakem-finans Kabul (koşullu; §10). Fiyat limiti karar kuralı ön kaydı (§9 #9). Kilit ⊂ feat_ok = 0 değişmezi; panel kesimi ve baseline kilidi beyanı; ölçüme kilit sonrası tavan serisi. Testler 54 + 31.
- **v0.3 (2026-11-16):** Hakem-rl + hakem-finans ilk okuması (§10). Kullanıcı kararları: birincil ödül `'excess'` (DSR yalnız α = 0), listeleme kilidi 20 seans, fiyat limiti önce ölçülür, düzeltilmiş fiyat + beyan. Testler 53 + 31. §2.5 yürütme ve veri beyanları.
- **v0.2 (2026-11-16):** Öz-kontrol bulgusu: varsayılan η ve `episode_len` ile `'differential'` G hep 0'dı. Ortam artık hata veriyor; test 25 → 27 (§7, §9 #8).
- **v0.1 (2026-11-16):** İlk sürüm. Kullanıcı kararı: logit sınırı 5 + ln N (§3.1). `marl/` çekirdek + PettingZoo ortamı + iki test (49 + 25). Ölçülen: maliyet yaklaşımı ≤ 1,0e-4 günlük getiri farkı (sentetik, 20 bps). `venv-marl/` iş makinesinde CPU torch 2.6.0 ile kuruldu (GPU yok).
