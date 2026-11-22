# Faz 11 — Entegrasyon ve sanal parayla simülasyon: plan

**Sürüm:** v0.5 TASLAK · 2026-11-22 · S0, S2, S1 ve S3 bitti (dal `feat/phase11-s3`)
**Durum:** Taslak. Kullanıcı niyeti (2026-11-22): Faz 10 sonrasında MARL eğitimi panoya ve portföy sistemine entegre edilir, sanal parayla simülasyon yapılır. Karar bekleyen maddeler §6'da. **S0 bitti (§3.1). S2 bitti (§3.2). S1 bitti (§3.3). S3 bitti (§3.4).**
**Dayanak:**
- [Faz 10 planı](phase-10-marl-plan.md) §9 (araştırma kapanışı, devredilenler), §3.3 K1–K10 (karar katmanı), §6 K13, K14, K27, §8 #18.
- [M8 tasarımı](phase-10-m8-design.md) §6.7 (ileri veri), §7 (okuma; §7.3 b kural portföyleri, §7.5 tohum kümelemesi).
- `app/CLAUDE.md` (kimlik, çalışma alanı, kağıt portföy), `marl/CLAUDE.md` (iki katman K9, aynı fonksiyon ilkesi).

---

## 0. Özet

1. **Simülasyon ile ileri veri testi aynı iştir.** M8 ön kaydı (§6.7) 2026 katının seçilmiş politikalarını (kol başına 12; SHA'ları `m8_reading.json` → `forward_policies_2026`), dondurmadan sonraki ilk seanstan (2026-11-23) itibaren en az 126 seansla bir kez okur. Her gün çalışan sanal para simülasyonu bu günleri biriktirir. 126 seans ~6 ay eder, ~Haziran 2027. Bu süre kısaltılamaz.
2. **İki tür portföy ayrı tutulur.**
   - **Dondurulmuş (araştırma) defterleri:** M8'in 72 politikası (6 kol × 12 tohum, boş kol dahil; ön kayıt listesi `forward_policies_2026`) ve 4 kural portföyü (1/N, al-tut, min-varyans, HRP). Hiçbiri değiştirilmez, yeniden eğitilmez. İleri veri okuması bunlardan yapılır.
   - **Ürün portföyleri:** kullanıcının seçtiği ya da yeniden eğittiği stratejiler. İleri teste girmez.
3. **Kural portföyleri her ekranda öğrenen kolların yanında durur.** M8'de 20 bps'de 6 kat ortalamasında hiçbir öğrenen kol bunları geçmedi (M8 §7.3 b). Simülasyonda yalnız RL'yi göstermek yanıltıcı olur.
4. **Kabul kapısı parite:** simülasyon defteri, aynı politika ve aynı pencere için M8 değerlendirmesiyle (`eval_fold_m8.py`) aynı günlük getiri serisini üretmeli (§3, S3). Bu sağlanmadan simülasyon sonucu ileri veri sayılmaz.
5. **En büyük teknik engel:** M8 politikaları BenchMARL deney durumu olarak kayıtlı (`policy.pt`, `ckpt/*.pt`). Yalnız `venv-marl` ile yüklenebiliyorlar. K9 gereği üretim kodu `venv-marl`'a bağımlı olamaz. Bu yüzden politikalar sade bir ileri geçişe (numpy ya da düz torch) aktarılır ve parite testinden geçer (S2).

## 1. Neden şimdi

- Faz 10'un araştırma sorusu cevaplandı (plan §9). Devredilen ürün işleri (K13, K14, K27, §8 #18) bu fazda kapanır.
- İleri veri penceresi 2026-11-23'te açılıyor. Fiyatlar zaten kaydediliyor, politikalar sabit. Bu yüzden simülasyon geç başlasa da veri kaybolmaz: defterler geriye dönük olarak doldurulabilir (S4). Yine de günlük hattın erken kurulması operasyonel hataları ileri veri okumasından önce yakalar.
- Araştırma çıktıları şu an yalnız `results/marl_m8/` altında ve git dışı (yaklaşık 2,5 GB; yalnız 2026 katı 417 MB). Bu politikaların kalıcı ve yedekli bir yere alınması ileri veri okumasının ön koşulu (S0).

## 2. İlkeler

- **Aynı fonksiyon (K2/K5/K6):** simülasyonda gözlem `obs_spec.build_obs_at`, hedef `weights.target_weights` + `weights.project`, adım `accounting.step_fractional` ile hesaplanır. Kopya mantık yazılmaz.
- **Yürütme zamanı (K12, §8 #18):** karar t kapanışında, yürütme t+1 açılışında, aynı maliyet (20 bps gidiş-dönüş, tek yön c = 10 bps). Kağıt portföyün bugünkü "aynı gün kapanışı" davranışı simülasyon defterlerinde kullanılmaz.
- **Dondurma:** dondurulmuş defterlerin politika SHA'sı her gün denetlenir. SHA uyuşmazsa o gün yazılmaz ve hata verir.
- **Veri kapısı:** ileri panel M8 paneliyle aynı veri kapısından ve `m8-v1` düzeltme kurallarıyla kurulur (Faz 10 plan sürüm geçmişi 2026-11-20; M2 tasarımı §6.1). Geçmiş satırlar `panel_m8.npz` ile bit-eş kalır; yalnız yeni satır eklenir. Ek yalnızca sona yapılır.
- **Tez sonucu:** ileri veri okuması bir analiz betiğinden ve ayrı bir ön kayıt ekinden çıkar (M8 §6.7). Pano yalnız aynası olur. Parite testi (S3) ikisinin aynı sayıyı verdiğini garanti eder.
- **Kimlik ve çalışma alanı:** yeni yazma uçları `RequireWriter`/`RequireAdmin` taşır. Arka plan işi kullanıcı kimliğini `ws.use_workspace(user_id)` ile taşır. Dondurulmuş defterler kullanıcıya değil, ortak salt okunur bir araştırma alanına aittir (§6 K-F11-3).

## 3. İş paketleri

| # | Paket | İçerik | Kabul ölçütü |
|---|---|---|---|
| **S0** | Dondurulmuş politika arşivi | **BİTTİ (§3.1).** 2026 katının seçilmiş politikaları (72) + `meta.json`, `norm_stats.json` salt okunur bir yapıt deposuna kopyalanır; SHA manifesti `forward_policies_2026` ile karşılaştırılır. Tüm `results/marl_m8/` için yedek (git dışı) | Manifest SHA'ları 72/72 eşleşir. Yedekten geri yükleme bir kez denenir |
| **S1** | İleri panel | **BİTTİ (§3.3).** `prepare_panel_m2.py --data-fix m8-v1` hattıyla artımlı panel sürümü (`panel_m8_fwd`). Günlük ek: yfinance artımlı çekim → veri kapısı → `m8-v1` kuralları. 2026-11-09 sonrası fiyat marjı takvimi için `PRICE_MARGIN_VERIFIED_UNTIL` kaynakla yeniden denetlenir (`price_limit_audit_m8.py`) | Geçmiş satırlar `panel_m8.npz` ile bit-eş. Veri kapısı reddettiği günü yazmaz, sebebini kaydeder. Marj kapsamı dışındaki gün hata verir (sessiz geçmez) |
| **S2** | Politika aktarımı (K9) | **BİTTİ (§3.2).** BenchMARL politikasından ağırlıklar okunur, sade bir ileri geçişe aktarılır: M-none / S-esm / S-agr için paylaşımlı MLP, M-gat / M-self için `comm_models` ileri geçişi. Ana venv'de çalışır, `venv-marl` import etmez | Aynı gözlemde logit farkı ≤ 1e-6, 72/72 politika (`venv-marl` eval çıktısıyla). Önce bir spike: tek politika (M-none s201) |
| **S3** | Simülasyon motoru ve defter | **BİTTİ (§3.4).** Strateji başına defter (K14): kesirli mod, `accounting.step_fractional`, t+1 açılış yürütmesi, rf A8. Kural portföyleri `marl.rule_baselines.rule_series` ile aynı defter biçiminde | **Parite:** 2026 test penceresi (2026-01-01..2026-10-31) defterden yeniden oynatıldığında her politikanın günlük R serisi `eval_fold_m8.py` test serisiyle bit-eş (ya da ≤ 1e-12). Kural portföyleri `m8_reading.json` → `baselines.2026` ile aynı Sharpe |
| **S4** | Günlük iş ve geriye dönük doldurma | Seans kapanışından sonra (BIST kapanışı + veri gecikmesi) çalışan iş: panel eki → karar → defter. İdempotent; kaçırılan günleri sırayla doldurur. Tetik `app/services/background_jobs.py` ya da işletim sistemi zamanlayıcısı (§6 K-F11-4) | Aynı gün iki kez çalışınca defter değişmez. 5 gün boşluk sonrası doldurma, günlük çalışmayla bit-eş |
| **S5** | Pano (K27) | Strateji seçimi (eski tek ajan / SARL / MARL / kural), karşılaştırma görünümü (yalnız aynı pencere ve aynı maliyetteki defterler yan yana), tohum dağılımı (ortalama ± std, 12 tohum), N, "tez sonucu değildir" notu. Eski ortam modelleri ayrı sekmede, "doğrudan kıyaslanmaz" notuyla | `dashboard/CLAUDE.md` kuralları (tema, uzun işler). Yazma uçları RBAC'li (test) |
| **S6** | İleri veri ön kayıt eki | M8 §6.7 ölçütü sabit kalır. Ek, okumadan önce yazılır: tohum kümelemesi baştan beyan edilir ve kümeli istatistik eklenir (M8 §7.4, §7.5), panel sürümü ve SHA'lar sabitlenir | Ek, 126. seanstan önce commit'lenir ve dondurulur (etiket) |
| **S7** | Kağıt portföy uyumu (K13, §8 #18) | Mevcut tek ajan günlük kararı: eski modellerde normalizasyon uyarısı (K13). Kullanıcının elle "Uygula" akışı kalır ama yürütme fiyatı ertesi açılış olur ya da fark sınırlılık olarak gösterilir (§6 K-F11-5) | `test_daily_trading`, `test_daily_trading_tz`, `test_paper_portfolio` yeşil. Golden değişmez |

**Sıra:** S0 → S2 (spike, sonra 72 politika) → S1 → S3 → S6 → S4 → S5 → S7.
- S0 ve S6 düşük maliyetli ama geri dönüşü zor. Önce gelir.
- S3'ün parite kapısı geçmeden S4 ve S5'e başlanmaz.

### 3.1 S0 sonucu (2026-11-22; `measurements/phase11-s0/s0_archive.json`)

- **Arşiv:** `data/marl/frozen/m8-2026/` (git dışı, ortak). 72 politika × 5 dosya (seçilen kontrol noktası, `meta.json`, `norm_stats.json`, `train.json`, `fold_eval.json`) ve `manifest.json`.
  - Manifest kaynağı `m8_reading.json`; dosyanın SHA-256'sı manifestte kayıtlı.
  - Doğrulama (`marl/frozen.verify_archive`) üç şeyi denetler:
    - her dosyanın SHA-256'sı,
    - politika özeti `params_sha`: ana venv'de, eğitimdeki `state_digest` tanımıyla yeniden hesaplanır,
    - ön kayıt listesiyle birebir eşlik: 72/72.
  - Ana venv'de torch ile yüklenir; BenchMARL gerekmez. Bu, S2 için iyi işarettir.
- **Yedek:** harici diske (kullanıcı kararı: yerel HDD; bulut şimdilik yok), 8.660 dosya, 2,76 GB.
  - Kapsam: `results/marl_m8` ve `data/marl` (panel, İş Yatırım kaynak önbelleği, arşiv dahil).
  - Her dosya hedefte yeniden hash'lendi; hepsi eşleşti. Yedek artımlıdır: `archive_s0.py --backup <hedef>` yeniden koşulunca yalnız değişen dosyaları kopyalar.
- **Geri yükleme denemesi:** yedekten geçici dizine alınan arşiv, `verify_archive` ile 72/72 doğrulandı.
- **Kod:**
  - `marl/frozen.py`: `state_digest` artık tek tanım; `m3_runner` buradan ithal eder (`test_marl_m3` 20/20).
  - `scripts/marl/archive_s0.py`.
  - `tests/test_marl_frozen.py`: 3 kontrol.
- **S4 için kural:** günlük iş her çalışmadan önce `verify_archive(..., forward=...)` çağırır; uyuşmazlıkta o gün yazılmaz.

### 3.4 S3 sonucu (2026-11-22; `measurements/phase11-s3/s3_parity.json`)

- **Kabul kapısı geçti.** 2026 test penceresi (207 seans, 2026-01-02 → 10-30, satır 3150 → 3357) `marl/sim_ledger.py` defterinden yeniden oynatıldı.
  - 72/72 politikanın günlük R serisi M8 test kaydıyla (`eval_fold_m8.py`, venv-marl TorchRL ortamı) **bit-eş**; fark 0. X, devir ve nakit ağırlığı da 72/72 bit-eş.
  - Karşılaştırılan M8 kayıtlarının SHA'sı arşivdeki `fold_eval.json` ile aynı (72/72): ölçülen seri M8'in okuduğu seri.
  - Kural portföyleri (ew, bh, minvar, hrp) `rule_series` ile bit-eş. Sharpe `m8_reading.json` → `baselines.2026` ile birebir aynı (4/4).
  - Politika başına ~0,07 sn (207 gün). 72 politika + 4 kural için günlük adım maliyeti ihmal edilir.
- **Yapı (`marl/sim_ledger.py`, ana venv).**
  - Karar t kapanışında, yürütme t+1 açılışında. Hedef:
    - politikada `target_weights(eylem)`,
    - kuralda `rule_baselines` ile aynı karar (minvar/hrp ay başı ve pencerenin ilk satırı).
  - Muhasebe `accounting.step_fractional`, ortamın kullandığı fonksiyonun kendisi:
    - `fill_prices`: kilit 20, dönem bağımlı marj satış engeli,
    - `project`: varsayılan kısıt,
    - A8 rf,
    - maliyet 20 bps gidiş-dönüş.
  - Politika `FrozenPolicy` (S2) ile tek gözlemle çağrılır. Gözlem `obs_spec.local_obs`, politikanın kendi eğitim `norm_stats`'ıyla.
  - TorchRL/BenchMARL yok.
  - **Durum** (`LedgerState`): başlangıç satırı, sıradaki karar satırı, kapanış ağırlığı, değer, adım sayısı.
    - JSON'a kayıpsız yazılır.
    - Kaldığı yerden devam, tam koşuyla bit-eş (test 2). S4 günlük işi bunun üstüne kurulur.
  - Değer = sanal sermaye × ∏(1 + R). Kesirli defterde sermaye R'yi değiştirmez (test 2; K-F11-6 yalnız gösterim).
- **Kararlarla ilişki.** S3 kesirli dondurulmuş defterdir (K-F11-2 önerisinin dondurulmuş kısmı; backtest ile aynı). Tam sayı lotlu ürün portföyü ve sermaye seçimi S5'te.
- **Açık kalan (S4 ön koşulu, değişmedi).** İleri panelle `Market` kurulumu, marj doğrulama kapsamı (2026-11-09) dışındaki gün için hata verir (test 4). Defter ileri günleri işlemez.
- **Test:** `tests/test_marl_sim_ledger.py` 4 kontrol, ~4 sn:
  - seri paritesi (M-none ve S-agr M8 kaydıyla, minvar ve bh `rule_series` ile),
  - JSON durumla kaldığı yerden devam bit-eş,
  - mutasyon: maliyet, durumda 1e-12 kayma, kaydırılmış başlangıç seriyi değiştirir,
  - ileri panelde marj kapsamı hatası.

### 3.3 S1 sonucu (2026-11-22; `measurements/phase11-s1/s1_forward_panel.json`)

- **Kabul kapısı geçti.**
  - Geçmiş satırlar `panel_m8.npz` ile bit-eş: 3.364 satır × 10 dizi (OHLCV, RSI, CCI, ADX, MACD, rf).
  - Gerçek çekimle 9 yeni seans eklendi (2026-11-10 → 11-20). Panel `data/marl/forward/m8/panel_m8_fwd.npz` (git dışı), 3.373 satır.
  - İş Yatırım çapraz denetiminde yama gerekmedi; en büyük fark 1e-7.
  - Aynı gün ikinci çalıştırma dosyaları değiştirmedi.
  - Yeni 270 hücre geçerli. Eğitim `norm_stats`'ı ile gözlem kırpma oranı 11 özelliğin hepsinde 0; 2026 test penceresinde en çok %0,8.
- **Yapı (`marl/forward_panel.py`, `scripts/marl/forward_panel_m8.py`):**
  - **Dondurulmuş kaynak.** `data/marl/frozen/m8-source/` altında `panel_m8`'in kaynak CSV'si ve m8-v1 İş Yatırım dosyaları; SHA'lar panel meta'sıyla aynı.
    - Uygulamanın veri sayfası `data/bist/stock_data_with_indicators.csv`'yi artımlı güncelleyebilir; ileri hat o dosyayı okumaz.
  - **Panel her güncellemede baştan kurulur.** M8 hattının aynı fonksiyonlarıyla: `apply_m8_v1` → ileri satırlar → `recompute_indicators` → `with_adx('wilder')` → `prepare_rl_panel` → `build_arrays` → rf.
    - Önceki panelin tüm satırları bit-eş denetlenir; uyuşmazlıkta yazılmaz.
    - İleri satırlar yalnız eklenir (`rows.csv`, append-only).
  - **Zincir.** Yeni gün = son yazılı kapanış × taze yfinance (auto_adjust) oranı.
    - Temettü ve bölünme o günün getirisine yansır; geçmiş yeniden düzeltilmez.
    - Hacim aynı ölçekle bölünür (ciro korunur).
  - **Çapraz kaynak (m8-v1'in nedensel karşılığı).** Yeni günün getirisi İş Yatırım düzeltilmiş kapanış getirisinden %1'den (m8-v1 `CLOSE_TOL`) fazla saparsa:
    - kapanış İş Yatırım getirisiyle zincirlenir,
    - yüksek/düşük, İş Yatırım gün içi uçlarıyla genişletilir.
  - **Neden m8-v1'in kendisi değil.** m8-v1'in 11 seanslık merkezi medyanı 5 seans ileriye bakar. Geriye dönük bir medyan da kaynaklar arası kalıcı seviye adımında yanlış yama yapardı.
    - Ölçüm (2021+, panel ve İş Yatırım önbelleği): iki kaynağın günlük düzeltilmiş getiri farkı %1'i 11 hücrede aştı (geçmiş kesik çubuklar).
    - Son günde iki kaynağın ham kapanışı birebir aynı.
  - **Gün kapısı.** İlk engelde durur; sonraki günler yazılmaz, sebep `log.jsonl`'a yazılır. Sebepler:
    - seans değil (yfinance satırı sembollerin yarısından az: atlanır),
    - kesim saati öncesi (aynı gün 19:00 TRT'den önce),
    - İş Yatırım bekleniyor,
    - rf bayat (d−1'de bilinen fonlama gözlemi 5 günden eski),
    - marj bandı dışı (kapanış ya da açılış ±%10,5 dışında: veri hatası ya da marj değişikliği adayı, elle inceleme),
    - yfinance çapası yok.
  - **Risksiz faiz.** EVDS'ten çekilir, ileri deponun kendi kopyasına yazılır; ortak `data/macro/risk_free.csv` değişmez. Taze seri geçmiş rf'yi değiştirmedi (bit-eş).
  - Ham çekimler `fetch/<zaman>/` altında SHA'larıyla saklanır.
- **Açık kalan (S4 ön koşulu).** `PRICE_MARGIN_VERIFIED_UNTIL` hâlâ 2026-11-09; kapsam dışındaki ileri gün `limit_down_gaps`'te hata verir (test 4). Ölçüt bu; sessiz geçmez.
  - S4 defteri ileri günleri işlemeden önce kapsam uzatılmalı. Bunun için bir kayıt düzeneği gerekir:
    - kaynak denetimi tarihi (son: 2026-11-20, M2 tasarımı §2.5),
    - ve gün kapısının bant denetimi; 9 gün için 0 ihlal.
  - Sabiti her gün elle değiştirmek sürdürülemez; düzenek S4'te kurulur.
- **Test:** `tests/test_marl_forward_panel.py` 4 kontrol:
  - sıfır satırla bit-eş,
  - zincir kuralı (temettü, kesik çubuk, yama eşiği),
  - gün kapısının her sebebi,
  - gerçek kaynak + 3 sentetik gün: önceki satırlar bit-eş, sonraki gün öncekileri değiştirmez, mutasyonlar yakalanır, marj kapsamı hatası.

### 3.2 S2 sonucu (2026-11-22; `measurements/phase11-s2/s2_parity.json`)

- **Kabul kapısı geçti: 72/72.** Logit farkı ölçüt ≤ 1e-6. Ölçülen en büyük fark **0**: eylem 72/72 politikada bit-eş, loc (tanh öncesi) de 0.
  - Karşılaştırma iki girdi kümesinde yapıldı:
    - 2026 test penceresi: 207 gün × 30 ajan.
    - Rastgele gözlem: 256 adet, N(0, 3²), tohum 20261122. ±5 kırpma sınırının dışına taşar, tanh doygunluğunu da yoklar.
  - Mimari dağılımı: `mlp` 24 (M-none, null), `comm` 24 (M-gat, M-self), `rowshared` 12 (S-esm), `sarl_agr` 12 (S-agr).
- **Referansın kendisi denetlendi.** Referans `scripts/marl/s2_parity_ref.py` ile üretildi: venv-marl, BenchMARL `load_policy`, deterministik keşif. Referans penceresinin günlük R serisi, M8 test kaydıyla (`results/marl_m8/2026/.../test_argmax.npz`) 72/72 birebir aynı. Yani karşılaştırılan çıktı, M8'in okuduğu davranışın ta kendisi.
- **Kod:** `marl/frozen_policy.py` → `FrozenPolicy(dizin).act(gözlem[N, d])`.
  - Ana venv'de çalışır, düz torch kullanır; BenchMARL / TorchRL import etmez.
  - Deterministik eylem: `B · clamp(tanh(loc), ±(1 − ε))`. Bu, TorchRL 0.7 `TanhNormal.deterministic_sample` ile aynı işlem sırasıdır; ε float32 çözünürlüğü, B = `logit_bound`.
  - M5 güven terimli iletişim kolları aktarılmadı; M8'de yoklar, gelirlerse açık hata verir.
- **İki bulgu (kurala bağlandı):**
  1. **Çağrı biçimi.** Aynı gözlem kümesi toplu (`[T, N, d]`) ve tek tek verildiğinde eylem en çok 5,0e-6 farklı çıkıyor. En büyüğü S-agr'de: 1.320 girdili matris çarpımında float32 toplama sırası değişiyor.
     - Parite, referansla aynı çağrı biçiminde ölçüldü: pencere gün gün, rastgele küme toplu.
     - **Kural:** canlı ve simülasyon yolu politikayı TEK gözlemle çağırır; M8 değerlendirmesi de böyle çağırıyordu. Toplu yol yalnız çözümleme içindir (`frozen_policy` belgesi).
     - S3 seri paritesi bu kuralla ölçülecek.
  2. **Süreç içi durum.** BenchMARL görev yapılandırması sınıf düzeyinde tutuluyor (`MarlTask.PORTFOLIO.update_config` birleştirir). Bir SARL kolunun `single_agent` anahtarı, aynı süreçte yüklenen sonraki MARL koluna taşıyor ve yükleme hata veriyor.
     - M8 her politikayı ayrı süreçte yüklediği için M8 sonuçlarını etkilemez.
     - Referans betiği artık kol başına ayrı süreç açar.
     - Ana venv yolu (`frozen_policy`) bu durumu taşımaz.
- **Test:** `tests/test_marl_frozen.py` 4. kontrol. Her mimariden bir politika (6) referansla bit-eş; gat maskesi bozulunca çıktı ayrışır (mutasyon). Referans dosyaları yoksa kontrol atlanır.

**Kaba efor** (tek geliştirici, gün; ölçüm değil): S0 0,5 · S1 2–3 · S2 2–3 · S3 3–4 · S4 1–2 · S5 4–6 · S6 0,5 · S7 1–2 → ≈ 14–21.

## 4. Kapsam dışı

- Gerçek emir iletimi, aracı kurum bağlantısı, gerçek para. Yalnız sanal para.
- Yeni araştırma hipotezi. Ürün portföylerinde yeniden eğitim yapılabilir, ama sonuçları tez sonucu değildir ve deneme sayısına (DSR N) girmez. Yalnız ileri veri eki tez sonucudur.
- Makale 2 (B5 + K0–K4): ayrı iş (Faz 10 plan §4).

## 5. Riskler

| Risk | Etki | Azaltma |
|---|---|---|
| Politika aktarımında sessiz fark (S2) | İleri test başka bir politikayı ölçer | Logit paritesi + S3 seri paritesi; SHA günlük denetim |
| yfinance veri sorunları (kesik gün, yanlış düzeltme; M2 §6.1'de ölçüldü) | İleri panelde sahte getiri | Veri kapısı + `m8-v1` kuralları; ikinci kaynak (İş Yatırım) denetimi ileri pencere için de koşulur (`cross_source_audit_m8.py`) |
| Fiyat marjı kuralı değişir | Satış engeli yanlış | `PRICE_MARGIN_VERIFIED_UNTIL` sonrası hata verir; kaynakla yeniden denetim (S1) |
| Evren değişimi (BIST-30 üyeliği) | Panel sembolleri sabit, ürün beklentisi farklı | Dondurulmuş defterler M8 evreniyle kalır (beyan). Ürün portföyü evren kararı ayrı (§6) |
| Arşivin kaybı (`results/` git dışı) | İleri veri okunamaz | S0 yedeği ve SHA manifesti |

## 6. Karar bekleyen konular (kullanıcı)

| # | Konu | Öneri |
|---|---|---|
| K-F11-1 | Panoda varsayılan görünüm | M-none (12 tohumun ortalaması ± std) + 1/N + min-varyans yan yana. Diğer kollar seçilebilir |
| K-F11-2 | Tam sayı lot mu, kesirli mi | Dondurulmuş defterler **kesirli** (backtest ile aynı; ileri test için şart). Ürün portföyü isteğe bağlı tam sayı lot (K5 `rebalance_orders`); farkı panoda gösterilir |
| K-F11-3 | Dondurulmuş defterlerin sahibi | Ortak, salt okunur araştırma alanı (yönetici yazar, herkes okur). Kullanıcı çalışma alanlarına kopyalanmaz |
| K-F11-4 | Günlük tetik | Uygulama içi arka plan işi (sunucu açıkken) + açılışta geriye dönük doldurma. Sunucu kapalı günler S4 doldurmasıyla kapanır |
| K-F11-5 | Elle "Uygula" akışında yürütme fiyatı | Ertesi açılışta yeniden fiyatlama (K14 ile birlikte). Daha basit seçenek: fark sınırlılık olarak gösterilir |
| K-F11-6 | Başlangıç sermayesi | 100.000 TL sanal (bugünkü varsayılan). Kesirli defterlerde ölçek sonucu değiştirmez |

## Versiyon geçmişi

- **v0.5 (2026-11-22):** S3 bitti (§3.4): `marl/sim_ledger.py` kesirli defter; 2026 test penceresi 72/72 politika M8 test serisiyle bit-eş, 4/4 kural portföyü `rule_series` ile bit-eş ve M8 okumasındaki Sharpe ile aynı. Durum JSON'la kaldığı yerden devam bit-eş (S4 temeli). Marj kapsamı açık maddesi sürüyor.
- **v0.4 (2026-11-22):** S1 bitti (§3.3): ileri panel append-only, dondurulmuş kaynaktan baştan kurulum, önceki satırlar bit-eş; 9 gerçek seans eklendi (yama 0). Kapanış çapraz denetimi nedensel getiri kuralıyla (m8-v1 merkezi medyanı ileriye bakıyordu). Açık: marj doğrulama kapsamının uzatılması S4 ön koşulu.
- **v0.3 (2026-11-22):** S2 bitti (§3.2): sade ileri geçiş 72/72 bit-eş (logit farkı 0; ölçüt ≤ 1e-6); referans R serisi M8 test kaydıyla 72/72 aynı. Kurallar: tek gözlemle çağrı (toplu çağrı ≤ 5e-6 farklı), referans kol başına ayrı süreç.
- **v0.2 (2026-11-22):** S0 bitti (§3.1): arşiv 72/72 doğrulandı, harici disk yedeği ve geri yükleme denemesi geçti. Politika sayısı 60 → 72 düzeltildi (boş kol ön kayıt listesinde).
- **v0.1 (2026-11-22):** İlk taslak. Faz 10 araştırma kapanışından sonra kullanıcı niyetiyle yazıldı. Kod yok.
