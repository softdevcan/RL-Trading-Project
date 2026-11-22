# M8 ileri veri ön kayıt eki (Faz 11 S6)

**Sürüm:** v0.2 TASLAK · 2026-11-22 · dondurulmadı (hakem-finans + hakem-rl ilk okuma işlendi)
**Neyin eki:** M8 ön kaydı v1.0 (`phase-10-m8-design.md` §6, etiket `m8-prereg-v1`, commit 1f89895) §6.7 "ileri veri" maddesi. Ana metin değişmez; bu ek yalnız §6.7'nin bıraktığı uygulama ayrıntılarını okumadan önce sabitler.
**Dondurma:** kullanıcı onayıyla commit + etiket `m8-forward-annex-v1`. S4 günlük işi ilk ileri günü (2026-11-23) defterlemeden önce yapılır.
**Okuma:** bir kez, §2'deki pencere tamamlanınca. Ara okuma yok (§9).

## 1. Değişmeyenler (M8 §6, aynen)

- **Ölçüt (§6.7):** ileri Δ̄ ve %95 GA aynı `scripts/marl/m8_stats.py` ile, tek kat olarak hesaplanır.
  - H1 ve H2'nin 6 kat etiketi FARK_VAR olduğu için üç sonuçlu kural uygulanır:
    - **destekler:** ileri Δ̄ aynı işaretli ve GA 6 kat Δ̄'sini kapsıyor,
    - **çelişir:** GA 0'ı ters işaretle dışlıyor,
    - aksi hâlde **belirsiz**.
  - H3 ve NK'nin 6 kat etiketi FARK_GÖZLENMEDİ; ileri okumaları yalnız betimseldir. Etiketin nasıl yazılacağı §6.1'de.
- **6 kat Δ̄ değerleri** elle kopyalanmaz. Kaynak `measurements/phase10-m8/m8_reading.json` → `contrasts.<H1|H2>.delta_bar` (argmax kuralı).
- **Politikalar:** 2026 katının seçilmiş (argmax) politikaları, kol başına 12, toplam 72. SHA'lar `m8_reading.json` → `forward_policies_2026` alanında; arşiv `data/marl/frozen/m8-2026` (S0, `verify_archive`).
- **Çıkarım yolu (hakem-rl #3):** M8 test serisini üreten yolun aynısı. S3'te 2026 penceresinde 72/72 bit-eş gösterildi (`measurements/phase11-s3/s3_parity.json`); parite o pencereyle sınırlıdır.
  - **Gözlem:** `obs_spec.local_obs`, yalnız politikanın kendi eğitim `norm_stats`'ıyla ölçeklenir. İleri veriden hiçbir istatistik (ölçek, kırpma sınırı, seçim) politikaya girmez. Kırpma ±5 (eğitimle aynı).
  - **Eylem:** deterministik (`TanhNormal` ortalaması); `FrozenPolicy` tek gözlemle, CPU'da, float32.
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
- **Seans sayımı (hakem-finans #6, hakem-rl #8):**
  - Seans = ileri panele kabul edilmiş ve defterlenmiş satır.
  - Yarım gün seansı sayılır.
  - Tatil ve iptal seansı sayılmaz. S1 kuralı: yfinance satırı sembollerin yarısından azsa gün atlanır.
  - §5'e göre elle atlanan bir gün sayılmaz; elle kaydı log'dadır.
  - Günlük iş her çalışmada seans sayacını ve (oluşunca) 126. seansın tarihini `log.jsonl`'a yazar. Okumada log sayacı = defter satır sayısı denetlenir (§7).
- **Takvim:** kabaca 2027 Mayıs sonu, tatillere bağlı. Tarih değil, seans sayısı belirleyicidir.

## 3. Panel sürümü ve §6.7'den sapma beyanı

- **Panel:** S1 ileri paneli (`marl/forward_panel.py`, `scripts/marl/forward_panel_m8.py`; plan §3.3).
  - Dondurulmuş kaynaktan her güncellemede baştan kurulur, önceki satırlar bit-eş denetlenir, ileri satırlar yalnız eklenir.
  - Sabitlenen SHA'lar:
    - kaynak CSV `3aec450d…`,
    - taban panel `panel_m8.npz` `2b7ea84c…` (kaynak manifest'i `data/marl/frozen/m8-source/manifest.json`),
    - kod dosyaları: dondurma commit'indeki hâlleri.
  - Her çekimin ham dosyası SHA'sıyla `data/marl/forward/m8/fetch/` altında, günlüğü `log.jsonl`'da.
  - **Kaynağın geriye dönük düzeltmesi (hakem-finans #2) bit-eş denetimini kırmaz.** Panel yfinance geçmişinden değil, dondurulmuş kaynak + `rows.csv`'den kurulur. Yeni gün = son yazılı kapanış × taze yfinance oranı. yfinance'in sonradan geçmişi yeniden düzeltmesi yazılı satırları değiştirmez; yalnız o günkü oranı belirler.
- **Risksiz faiz (hakem-finans #3):**
  - Kaynak ve gecikme: EVDS fonlama oranı (`MacroDataFetcher().fetch_risk_free`), A8 gecikmesiyle (t'de bilinen oran).
  - Her çekim ileri deponun kendi kopyasına SHA'sıyla yazılır; ortak `data/macro/risk_free.csv` kullanılmaz.
  - Bayatlık eşiği: d−1'de bilinen son gözlem 5 günden eskiyse gün bekler (`rf_bayat`).
  - rf panelin bir dizisidir; bit-eş denetimine girer. EVDS geçmişi revize ederse denetim kırılır ve gün yazılmaz. Kural:
    - önceki satırların rf'si korunur,
    - yalnız yeni gözlemler eklenir,
    - olay tarihli olarak `log.jsonl`'a yazılır,
    - tüm stratejilere aynı uygulanır.
- **Sapma: m8-v1 yerine nedensel karşılığı.** §6.7 ileri paneli "aynı veri kapısı ve `m8-v1` düzeltme kurallarıyla" kurmayı söyler. Uygulanan kural farklıdır:
  - **Neden:** m8-v1'in kapanış yaması 11 seanslık merkezi medyana dayanır, yani 5 seans ileriye bakar. İleri veride gün kapandığında bu bilgi yoktur; kural gerçek zamanlı uygulanamaz.
  - **Uygulanan:** aynı eşik (`panel_fix.CLOSE_TOL` %1), nedensel karşılaştırmayla. Yeni günün yfinance getirisi İş Yatırım düzeltilmiş kapanış getirisinden %1'den fazla saparsa:
    - kapanış İş Yatırım getirisiyle zincirlenir,
    - yüksek/düşük, İş Yatırım gün içi uçlarıyla genişletilir.
  - **Ne zaman karar verildi:** 2026-11-22, S1 sırasında (S1 commit'leri, `phase-11-integration-plan.md` §3.3).
  - **Neden hiçbir okuma etkilemedi:** ileri gün defteri S4'te kuruluyor. Defter deposunun ilk kaydı ve ilk ileri günün işlenme zamanı `log.jsonl`'da; ikisi de bu ekin dondurma commit'inden sonradır (§7 denetimi).
  - **Sınır:** kural, 2026-11-10..20 panelini görmüş bir araştırmacı tarafından seçildi; o 9 günde yama 0. Kural yalnız veri doğruluğunu hedefler, politika ya da kol sonucuna bakmaz.
  - Geçmiş satırlar (m8-v1'le düzeltilmiş) aynen korunur.
  - m8-v1'in tarihsel tek seferlik düzeltmeleri dondurulmuş kaynakta kalır: 2023-02-08 iptal seansı ve SKBNK çarpanları.

## 4. Defter (muhasebe)

- Defter: `marl/sim_ledger.py` (S3). 2026 penceresinde 72/72 politika M8 test serisiyle bit-eş, 4/4 kural `rule_series` ile bit-eş (`measurements/phase11-s3/s3_parity.json`).
- **Fiyat marjı:** satış engeli dönem bağımlı marjla (`accounting.limit_down_gaps`).
  - İleri günler için kapsam S4 düzeneğiyle uzatılır: tarihli kaynak denetim kaydı + gün kapısının ±%10,5 bant denetimi.
  - Pencere içinde Borsa İstanbul marjı değiştirirse yeni marj yürürlük tarihinden itibaren, kaynağıyla birlikte uygulanır. Bu, tüm stratejilere aynı anda uygulanan tarihli bir kayıttır.

## 5. Veri olayları (önceden kural)

- **Evren sabit:** panelin 30 sembolü (M8 evreni). XU030 üyelik değişikliği evreni değiştirmez; politikalar bu 30 sembolle eğitildi.
- **Eksik hücre:** `fill_prices` sembolü son kapanışıyla taşır ve hisse o gün işlem görmez (M2 kuralı); hücre atılmaz.
- **Kalıcı işlem yasağı / işlem dışı kalma (hakem-finans #7):**
  - Birincil: ağırlık son fiyatında donar (M2 kuralı).
  - Bir sembol 20 seansı aşan kesintisiz süre işlem görmezse okumada ayrı satır verilir.
  - Duyarlılık: o sembolün donan değeri kesintinin 21. seansında 0 sayılır, tüm stratejilere aynı. Birincil sonucu değiştirmez; ayrışırsa beyan edilir.
- **Gün kapısı:** S1 kapısı (İş Yatırım bekleniyor, rf bayat, bant dışı, kesim saati) günü bekletir; atlamaz.
  - Bir seans kalıcı olarak işlenemezse elle karar `log.jsonl`'a yazılır: gerekçe, karar zamanı, o ana kadarki seans sayacı.
  - Defter satırları zaman damgalıdır. Karar anında panoda ne görüldüğü, o damgaya kadarki defterden yeniden kurulabilir (hakem-finans #8).
  - Okumadan önce ve tüm stratejiler için aynı yapılır.
- **Kaynak bozulması:** yfinance ya da İş Yatırım biçim değiştirirse çekim kodu düzeltilebilir.
  - Kural ve eşikler değişmez.
  - Önceki satırlar bit-eş kalmalıdır; S1 denetimi bunu zorlar.

## 6. İstatistik

- **Hücre metriği:** SR_a(s) = `marl.selection.excess_sharpe(R, rf = R − X)`. Hesap 126 seanslık ileri pencerede, yıllık; tanımsızsa 0 (§6.4).
- **Fark matrisi:** d = [1 × 12], d[0, s] = SR_a(s) − SR_b(s).
- **Birincil (§6.7):** `m8_stats.contrast(d)`. Değerleri: Δ̄, p_işaret (R 10.000, tohum 1), bootstrap %95 GA (B 10.000, tohum 0).
  - Ölçüt GA'yla okunur (§1). İki onaylayıcı karşıtlık (H1, H2) ayrı ayrı okunur.
  - GA tabanlı üç sonuçlu kural test değildir. Çoklu karşılaştırma düzeltmesi uygulanmaz; iki sonucun ortak yorumu yapılmaz.

### 6.1 Etiketin yazımı (hakem-rl #1, hakem-finans #1, #9; ölçütü değiştirmez)

§6.7 GA'sı 12 tohum eşinden gelir ve bu eşlerin hepsi aynı 126 günlük piyasa yolunu paylaşır. Bu yüzden GA yalnız eğitim (tohum) değişkenliğini ölçer, piyasa yolunun şansını ölçmez. Tohum kümelemesi de eşleri bağımsız sayar; ortak yol bu varsayımı zayıflatır ve GA'yı daraltabilir.

Bu, iki yönlü bir hataya kapı açar:
- **"Çelişir" şansla tetiklenebilir:** dar GA, yalnız piyasa şansıyla 0'ı ters işaretle dışlayabilir.
- **"Destekler" ucuz olabilir:** geniş GA hem 0'ı hem 6 kat Δ̄'sini kapsayabilir. O durumda "destekler", "çelişmedi ve işaret aynı" demektir; güçlü doğrulama değildir.

Bu yüzden H1/H2 etiketi her yerde (tablo, özet, tez) aynı satırda şunlarla verilir:
- (a) tohum GA'sı 0'ı kapsıyor mu: `ga0_kapsar` / `ga0_dislar`,
- (b) gün bloklu GA (§6.3) aynı üç sonuçlu kuralı aynı sonuçla veriyor mu: `yol_ayni` / `yol_farkli`,
- (c) GA genişliğinin 6 kat Δ̄'sine oranı.
- **Kurallar:**
  - `yol_farkli` ise tez metninde "destekler" ya da "çelişir" cümlesi kurulmaz. İki sonuç da yan yana raporlanır.
  - "destekler" + `ga0_kapsar`, "çelişmedi (işaret aynı)" diye yazılır.
- NK (aynı 126 günü paylaşan negatif kontrol) Δ̄ ve GA'sı her H1/H2 satırının yanında verilir. Bu, ileri verinin gürültü tabanının tek doğrudan ölçüsüdür (hakem-rl #7). H1/H2 GA'sı NK'ninkinden ayrışmıyorsa beyan edilir.

### 6.2 Tohum kümelemesi (M8 §7.4 sözü; hakem-rl #2)

- Tek pencerede her eş farklı bir tohumdur; 12 eş, 12 kümedir.
- Ek rapor, tek katlı tanımla: sm = d[0] (12 değer).
  - tam işaret-çevirme (2¹² = 4.096 düzen),
  - t-GA (serbestlik derecesi 11),
  - MDE₈₀.
- `m8_robustness.seed_cluster`'ın tohum etkisi F'si tek katta tanımsızdır (kat içi artık yok); raporlanmaz. Okuma betiği tek katlı ayrı bir fonksiyon kullanır; self-testi bunu sınar.
- Ölçütü değiştirmez.

### 6.3 Gün bloklu bootstrap (betimsel; §6.1(b) girdisi)

- Gün bloklu dairesel bootstrap. Günler 72 politika için ortak yeniden örneklenir.
  - Seçilen gün indisi o günün R, rf ve X'ini birlikte taşır (hakem-finans #5).
- Ayarlar:
  - blok 10 seans (≈ √126; ~13 blok, GA kaba),
  - B 10.000, tohum 0,
  - blok indisleri `evaluation.regime_contrast.circular_block_indices`,
  - her tekrarda hücre SR'si yeniden hesaplanır; tanımsızsa 0 (§6.4 ile aynı).
- Δ̄ için %95 yüzdelik GA.

### 6.4 Betimsel

- kol başına ortalama ± std (12 tohum),
- kural portföyleri (ew, bh, minvar, hrp; aynı defter),
- 0 / 10 / 25 / 50 bps ex post (§6.5 D4),
- devir, nakit ağırlığı, satış engeli,
- XU030 referans satırı ("kıyas değil", D1),
- gerçekleşen tohum sd'siyle MDE.
- **Girdi desteği (hakem-rl #5):** özellik başına, eğitim `norm_stats`'ıyla kırpılan gözlem oranı. Kötü bir ileri sonucun dağılım kaymasıyla birlikte olup olmadığı okunabilsin.

## 7. Okuma öncesi denetimler (biri bozuksa sonuçlar `UYARILI`)

- Arşiv `verify_archive` 72/72; politika SHA'ları `forward_policies_2026` ile aynı.
- **Panel:**
  - ilk 3.364 satır `panel_m8.npz` ile bit-eş,
  - kaynak manifest SHA'ları §3'teki gibi,
  - `rows.csv` yalnız eklenmiş: günlükteki her güncellemede önceki satır sayısı korunmuş,
  - geçmiş rf satırları korunmuş.
- **Defter:**
  - `scripts/marl/s3_parity.py` okuma commit'inde 72/72 ve 4/4 geçer (muhasebe kodu değişmedi),
  - 72 politika ve 4 kural defterinde tam 126 yürütme günü, aynı tarihler, ilk gün 2026-11-23,
  - log'daki seans sayacı ve 126. seans tarihi defterle aynı,
  - defter deposunun ilk kaydı bu ekin dondurma commit'inden sonra.
- **Günlük:** `log.jsonl`'da çözülmemiş gün kapısı durması yok; elle kararların hepsi tarihli ve okumadan önce.
- **Tanımsız hücre (§6.6, tek kat; hakem-finans #4, hakem-rl #4):**
  - kol başına tanımsız (tümüyle nakit) hücre sayısı raporlanır.
  - Bir kolda 2 ve üstü (≥ %10 kuralının 12 hücrede karşılığı) ya da iki kol arasındaki fark 2 ve üstüyse o karşıtlık için "tanımsızlar düşürülmüş" duyarlılık satırı verilir ve etikete `tanimsiz_uyarisi` eklenir.
- **Eksik hücre:** kol başına 12 hücre beklenir.
  - 2 ve üstü eksikse o kolun karşıtlıkları **EKSİK** olur (%10 kuralının 12 hücrede karşılığı).
  - 1 eksikse eşleşmeyen hücre düşer, sayısı raporlanır.
- **Kod:** okuma betiği ve self-testi okumadan önce commit'li; ağaç temiz.

## 8. Okuma kodu

`scripts/marl/analyze_forward_m8.py`, S4 defter biçimi kurulunca ve **ilk ileri gün defterlenmeden önce** self-testiyle commit'lenir (hakem-rl #6). SHA'sı okuma JSON'una yazılır.

- **Çıktı şeması (sabit):**
  - `checks` (§7),
  - `contrasts.<H1|H2|H3|NK>`: `delta_bar`, `ci95`, `p_signflip`, `p_bootstrap`, `verdict` (H1/H2), `ga0`, `path_ci95`, `path_verdict`, `width_ratio`, `cluster` {`p_exact`, `t_ci95`, `mde80`}, `undefined`, `missing`,
  - `descriptive` (§6.4),
  - `provenance`.
- Kod bu eke aykırıysa ek geçerlidir. Okumadan önce düzeltilebilir, ama ara görünümdeki bir sayıya göre değiştirilmez; her değişiklik commit mesajında gerekçelenir.

## 9. Sonuç görüldükten sonra yapılmayanlar

- **§6.8'in tamamı.** Ek olarak:
  - pencereyi uzatmak ya da kısaltmak,
  - 126. seanstan önce okumak (ara okuma),
  - politikaları değiştirmek ya da yeniden eğitmek,
  - kural portföylerini ölçüte katmak.
- **Pano (S5) ara görünümü.** Defterler panoda günlük görünür. Bu ara okuma sayılmaz, ama hiçbir karar ona göre değişmez: pencere, kollar, ölçüt, veri kuralı, marj kaydı.
  - §5'e göre yapılan tarihli veri kaydı tüm stratejilere aynı uygulanır. Okumada hepsi listelenir.
  - Araştırmacının ara görünümü gördüğü beyan edilir.

## Hakem okumaları

| Okuma | hakem-finans | hakem-rl |
|---|---|---|
| İlk (v0.1, 2026-11-22) | Küçük rev.: 2 BÜYÜK (#1 tohum GA'sı piyasa yolunu kapsamıyor, #2 sapma kanıtı + kaynak düzeltmesi kuralı), 7 KÜÇÜK | Küçük rev.: 3 BÜYÜK (#1 "destekler" geniş GA'da ucuz, #2 seed_cluster tek katta tanımsız, #3 çıkarım yolu sabitlenmemiş), 5 KÜÇÜK |
| İşleme (v0.2) | #1, #9 → §6.1; #2 → §3; #3 → §3; #4 → §7; #5 → §6.3; #6 → §2; #7 → §5; #8 → §5 | #1, #7 → §6.1; #2 → §6.2; #3 → §1; #4 → §7; #5 → §6.4; #6 → §8; #8 → §2 |

## Versiyon geçmişi

- **v0.2 (2026-11-22):** hakem ilk okumaları işlendi:
  - §6.1 etiket yazımı (tohum GA'sı + gün bloklu GA + NK yan yana; `yol_farkli`'da tez cümlesi yok),
  - §6.2 tek katlı kümeleme,
  - §1 çıkarım yolu,
  - §3 rf ve kaynak düzeltmesi kuralları,
  - §5 kalıcı işlem dışı kalma duyarlılığı,
  - §7 tanımsız/eksik hücre kuralları (12 hücrede),
  - §2 seans sayacı,
  - §8 okuma betiği ilk ileri günden önce, çıktı şeması.
- **v0.1 (2026-11-22):** taslak. Pencere tam 126 seans (2026-11-23'ten). §6.7'den sapma beyanı: m8-v1 yerine nedensel kapanış denetimi. Tohum kümelemesi ve zaman belirsizliği betimsel ek. Okuma kodu S4 sonrası.
