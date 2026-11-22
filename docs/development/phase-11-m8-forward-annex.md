# M8 ileri veri ön kayıt eki (Faz 11 S6)

**Sürüm:** v0.1 TASLAK · 2026-11-22 · dondurulmadı
**Neyin eki:** M8 ön kaydı v1.0 (`phase-10-m8-design.md` §6, etiket `m8-prereg-v1`, commit 1f89895) §6.7 "ileri veri" maddesi. Ana metin değişmez; bu ek yalnız §6.7'nin bıraktığı uygulama ayrıntılarını okumadan önce sabitler.
**Dondurma:** kullanıcı onayıyla commit + etiket `m8-forward-annex-v1`. S4 günlük işi ilk ileri günü defterlemeden önce yapılır; plan (`phase-11-integration-plan.md` §3 S6) en geç 126. seanstan önce diyor.
**Okuma:** bir kez, §2'deki pencere tamamlanınca. Ara okuma yok (§9).

## 1. Değişmeyenler (M8 §6, aynen)

- **Ölçüt (§6.7):** ileri Δ̄ ve %95 GA aynı `scripts/marl/m8_stats.py` ile, tek kat olarak hesaplanır.
  - H1 ve H2'nin 6 kat etiketi FARK_VAR olduğu için üç sonuçlu kural uygulanır:
    - **destekler:** ileri Δ̄ aynı işaretli ve GA 6 kat Δ̄'sini kapsıyor,
    - **çelişir:** GA 0'ı ters işaretle dışlıyor,
    - aksi hâlde **belirsiz**.
  - H3 ve NK'nin 6 kat etiketi FARK_GÖZLENMEDİ; ileri okumaları yalnız betimseldir.
- **6 kat Δ̄ değerleri** elle kopyalanmaz. Kaynak `measurements/phase10-m8/m8_reading.json` → `contrasts.<H1|H2>.delta_bar` (argmax kuralı).
- **Politikalar:** 2026 katının seçilmiş (argmax) politikaları, kol başına 12, toplam 72. SHA'lar `m8_reading.json` → `forward_policies_2026` alanında; arşiv `data/marl/frozen/m8-2026` (S0, `verify_archive`).
- **Karşıtlıklar ve eşleme (§6.2, §6.4):**
  - H1: M-none ↔ S-esm.
  - H2: M-none ↔ S-agr.
  - H3: M-gat ↔ M-self.
  - NK (negatif kontrol): M-none ↔ boş kol.
  - Eş aynı tohumdur; NK'de 201↔213, …, 212↔224.
- **Diğerleri:** maliyet 20 bps gidiş-dönüş, A8 risksiz faiz, t kapanışında karar, t+1 açılışında yürütme, §6.8 yasakları.
- **Güç (§6.7, kaba):** 126 seans ve k 12 ile MDE ≈ 0,25 Sharpe. Yalnız büyük farklar doğrulanabilir.

## 2. Pencere

- **İlk yürütme günü:** 2026-11-23. Bu, ön kaydın dondurulduğu commit tarihinden (1f89895, 2026-11-21 17:23 +03) sonraki ilk seanstır (§6.7, §7.4).
  - 2026-11-10..11-20 satırları panelde var ama ileri veri sayılmaz. Yalnız gözlem bağlamı olarak kullanılır (özellik ısınması).
- **İlk karar satırı:** 2026-11-23'ten önceki son panel satırı. Kural `eval_fold_m8.decision_rows` ile aynı.
- **Başlangıç:** her defter tam nakitle başlar, M8 test penceresiyle aynı.
- **Uzunluk:** **tam olarak ilk 126 yürütme seansı.**
  - "En az 126" değil; okuma zamanı araştırmacının seçimine kalmaz.
  - 127. ve sonraki seanslar bu okumaya girmez. Defter çalışmaya devam edebilir; o seanslar ancak ayrı bir ön kayıtla okunur.
- **Takvim:** kabaca 2027 Mayıs sonu, tatillere bağlı. Tarih değil, seans sayısı belirleyicidir.

## 3. Panel sürümü ve §6.7'den sapma beyanı

- **Panel:** S1 ileri paneli (`marl/forward_panel.py`, `scripts/marl/forward_panel_m8.py`; plan §3.3).
  - Dondurulmuş kaynaktan her güncellemede baştan kurulur, önceki satırlar bit-eş denetlenir, ileri satırlar yalnız eklenir.
  - Sabitlenen SHA'lar:
    - kaynak CSV `3aec450d…`,
    - taban panel `panel_m8.npz` `2b7ea84c…` (kaynak manifest'i `data/marl/frozen/m8-source/manifest.json`),
    - kod dosyaları: dondurma commit'indeki hâlleri.
  - Her çekimin ham dosyası SHA'sıyla `data/marl/forward/m8/fetch/` altında, günlüğü `log.jsonl`'da.
- **Sapma: m8-v1 yerine nedensel karşılığı.** §6.7 ileri paneli "aynı veri kapısı ve `m8-v1` düzeltme kurallarıyla" kurmayı söyler. Uygulanan kural farklıdır:
  - **Neden:** m8-v1'in kapanış yaması 11 seanslık merkezi medyana dayanır, yani 5 seans ileriye bakar. İleri veride gün kapandığında bu bilgi yoktur; kural gerçek zamanlı uygulanamaz.
  - **Uygulanan:** aynı eşik (`panel_fix.CLOSE_TOL` %1), nedensel karşılaştırmayla. Yeni günün yfinance getirisi İş Yatırım düzeltilmiş kapanış getirisinden %1'den fazla saparsa:
    - kapanış İş Yatırım getirisiyle zincirlenir,
    - yüksek/düşük, İş Yatırım gün içi uçlarıyla genişletilir.
  - **Ne zaman karar verildi:** 2026-11-22, S1 sırasında. O tarihte hiçbir ileri gün hiçbir politika ya da kural portföyüyle defterlenmemişti (defter S3'te kuruldu; ileri günleri S4'ten önce işlemez).
  - **Etkisi:** 2026-11-10..20 için yama 0. Geçmiş satırlar (m8-v1'le düzeltilmiş) aynen korunur.
  - m8-v1'in tarihsel tek seferlik düzeltmeleri dondurulmuş kaynakta kalır: 2023-02-08 iptal seansı ve SKBNK çarpanları.

## 4. Defter (muhasebe)

- Defter: `marl/sim_ledger.py` (S3). 2026 penceresinde 72/72 politika M8 test serisiyle bit-eş, 4/4 kural `rule_series` ile bit-eş (`measurements/phase11-s3/s3_parity.json`).
- **Fiyat marjı:** satış engeli dönem bağımlı marjla (`accounting.limit_down_gaps`).
  - İleri günler için doğrulanmış kapsam S4 düzeneğiyle uzatılır: kaynak denetim tarihi ve gün kapısının ±%10,5 bant denetimi kaydedilir.
  - Pencere içinde Borsa İstanbul marjı değiştirirse yeni marj yürürlük tarihinden itibaren, kaynağıyla birlikte uygulanır. Bu, tüm stratejilere aynı anda uygulanan tarihli bir kayıttır.

## 5. Veri olayları (önceden kural)

- **Evren sabit:** panelin 30 sembolü (M8 evreni). XU030 üyelik değişikliği evreni değiştirmez; politikalar bu 30 sembolle eğitildi.
- **Eksik hücre:** `fill_prices` sembolü son kapanışıyla taşır ve hisse o gün işlem görmez (M2 kuralı).
  - Kalıcı işlem dışı kalan bir sembolün ağırlığı son fiyatında donar. Bu beyan edilir; hücre atılmaz.
- **Gün kapısı:** S1 kapısı (İş Yatırım bekleniyor, rf bayat, bant dışı, kesim saati) günü bekletir; atlamaz.
  - Bir seans kalıcı olarak işlenemezse elle karar `log.jsonl`'a gerekçesiyle yazılır. Bu, okumadan önce ve tüm stratejiler için aynı yapılır.
  - Tatil ve iptal seansı seans sayılmaz (yfinance satırı sembollerin yarısından azsa atlanır; S1).
- **Kaynak bozulması:** yfinance ya da İş Yatırım biçim değiştirirse çekim kodu düzeltilebilir.
  - Kural ve eşikler değişmez.
  - Önceki satırlar bit-eş kalmalıdır; S1 denetimi bunu zorlar.

## 6. İstatistik

- **Hücre metriği:** SR_a(s) = `marl.selection.excess_sharpe(R, rf = R − X)`. Hesap 126 seanslık ileri pencerede, yıllık; tanımsızsa 0 (§6.4).
- **Fark matrisi:** d = [1 × 12], d[0, s] = SR_a(s) − SR_b(s).
- **Birincil (§6.7):** `m8_stats.contrast(d)`. Değerleri: Δ̄, p_işaret (R 10.000, tohum 1), bootstrap %95 GA (B 10.000, tohum 0).
  - Ölçüt GA'yla okunur (§1). İki onaylayıcı karşıtlık (H1, H2) ayrı ayrı okunur.
  - GA tabanlı üç sonuçlu kural test değildir. Çoklu karşılaştırma düzeltmesi uygulanmaz; iki sonucun ortak yorumu yapılmaz.
- **Tohum kümelemesi (M8 §7.4 sözü).** Tek pencerede her eş farklı bir tohumdur. 12 eş, 12 kümedir.
  - Ek olarak verilecekler: tam işaret-çevirme (2¹² = 4.096 düzen) ve t-GA (serbestlik derecesi 11; `m8_robustness.seed_cluster` ile aynı tanım).
  - Ölçütü değiştirmez.
- **Zaman belirsizliği (betimsel).** 12 eş aynı 126 günlük piyasa yolunu paylaşır. §6.7 GA'sı yalnız eğitim (tohum) değişkenliğini ölçer, piyasa yolunun örneklem belirsizliğini ölçmez.
  - Betimsel ek: gün bloklu dairesel bootstrap. Günler tüm politikalar için ortak yeniden örneklenir; blok 10 seans (≈ √126); B 10.000, tohum 0; Δ̄ için %95 GA.
  - Blok indisleri `evaluation.regime_contrast.circular_block_indices` ile.
  - Ölçütü değiştirmez. GA'lar ayrışırsa iki GA da raporlanır.
- **Betimsel:**
  - kol başına ortalama ± std (12 tohum),
  - kural portföyleri (ew, bh, minvar, hrp; aynı defter),
  - 0 / 10 / 25 / 50 bps ex post (§6.5 D4),
  - devir, nakit ağırlığı, satış engeli,
  - XU030 referans satırı ("kıyas değil", D1),
  - gerçekleşen tohum sd'siyle MDE.

## 7. Okuma öncesi denetimler (biri bozuksa sonuçlar `UYARILI`)

- Arşiv `verify_archive` 72/72; politika SHA'ları `forward_policies_2026` ile aynı.
- Panel: ilk 3.364 satır `panel_m8.npz` ile bit-eş; kaynak manifest SHA'ları §3'teki gibi; `rows.csv` yalnız eklenmiş (günlükteki her güncellemede önceki satır sayısı korunmuş).
- Defter paritesi: `scripts/marl/s3_parity.py` okuma commit'inde 72/72 ve 4/4 geçer (muhasebe kodu değişmedi).
- 72 politika ve 4 kural defterinde tam 126 yürütme günü, aynı tarihler; ilk gün 2026-11-23.
- `log.jsonl`'da çözülmemiş gün kapısı durması yok; elle kararların hepsi okumadan önce tarihli.
- Okuma betiği ve self-testi okumadan önce commit'li; ağaç temiz.

## 8. Okuma kodu

`scripts/marl/analyze_forward_m8.py` S4'ten sonra yazılır, çünkü defter dosya biçimini S4 kurar. Self-test ile 126. seanstan önce commit'lenir; SHA'sı okuma JSON'una yazılır. Kod bu eke aykırıysa ek geçerlidir; düzeltme okumadan önce yapılır.

## 9. Sonuç görüldükten sonra yapılmayanlar

- **§6.8'in tamamı.** Ek olarak:
  - pencereyi uzatmak ya da kısaltmak,
  - 126. seanstan önce okumak (ara okuma),
  - politikaları değiştirmek ya da yeniden eğitmek,
  - kural portföylerini ölçüte katmak.
- **Pano (S5) ara görünümü.** Defterler panoda günlük görünür. Bu ara okuma sayılmaz, ama hiçbir karar ona göre değişmez: pencere, kollar, ölçüt, veri kuralı, marj kaydı.
  - §5'e göre yapılan tarihli veri kaydı tüm stratejilere aynı uygulanır. Okumada hepsi listelenir.
  - Araştırmacının ara görünümü gördüğü beyan edilir.

## Versiyon geçmişi

- **v0.1 (2026-11-22):** taslak. Pencere tam 126 seans (2026-11-23'ten). §6.7'den sapma beyanı: m8-v1 yerine nedensel kapanış denetimi. Tohum kümelemesi ve zaman belirsizliği betimsel ek. Okuma kodu S4 sonrası.
