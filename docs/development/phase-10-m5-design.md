# Faz 10 M5 — Tahmin sinyali ve güvenle ağırlıklı iletişim: tasarım ve ön kayıt

**Sürüm:** v0.9 · 2026-11-19 · dal `feat/phase10-m5-mech`
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.1.3 (P5 ve "P5 durumu"), sinyal yolları tablosu, AQ2 ablasyonu
- [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) M5 (Y7 yeniden çerçeveleme, ablasyon kolları), M8 "Güven kolları" maddesi, §9.31 kullanıcı kararı
- [phase-10-m3-design.md](phase-10-m3-design.md) §5–§6 (pozitif kontrol PC060-K12, karıştırılmış kontrol, M3b KISMİ), §6.8 (M4 temeli)

**Durum (v0.8):** §4 mekanizma kolları KOŞULDU: **H2 KISMİ (`gurultu_tabani`), H3 SINANMADI** (§4.11). Ön kayıt (v0.7, DONDURULDU) değişmedi. Gerçek sinyal kolları ön kaydı §5 TASLAK (v0.9; kol seti kullanıcı onayı bekliyor, koşu kullanıcı takviminde). §3.6 ön kayıt öncesi kararlar. Aşağıdaki satır v0.3 durumudur.

**Durum (v0.3):** **Yalnız §2 yazıldı: P5c sonucunun okuma kuralı; açık kararlar K-a, K-b, K-c kapandı (§2.7).** Kural, gerçek sinyalli hiçbir MARL koşusu yapılmadan önce commit'lenir (plan §3.1.3: "Kapı geçmezse M5'in nasıl okunacağı M5 ön kaydında, sonuçtan önce yazılır"). §3 kollar ve bütçe **TASLAĞI** (v0.3): S-msg kodu yazıldı (`marl/comm_models.py` conf_mode, `tests/test_marl_m5_msg.py` 17/17); eşikler ve açık kararlar M4a Aşama 2 + 1b sonucundan sonra kapanır. **Koşu yok.**

---

## 1. Kapsam

M5, AQ2'yi sınar: tahmin sinyali yerel gözleme (S-obs), mesaja (S-msg) ya da ikisine birden verildiğinde politika ne kazanır? Ayrıca Faz 9 M5'teki güven mekanizmasını da sınar (H2: güvenle ağırlıklı > güvensiz dikkat; H3: kalibre > kalibresiz güven).

P5c ölçümü M5'in girdisini değiştirdi. Gerçek sinyal bilgi kapısından geçmiyor. Bu belge önce bu olguyu ve okumasını sabitler. Kolların ayrıntısı sonra gelir ve bu kurala uymak zorundadır.

## 2. P5c okuma kuralı (ön kayıt, sonuçtan önce)

### 2.1 Girdi olgusu

Kaynak: `docs/development/measurements/phase10-p5/gate_pit_v2_sensitivity.json` → `gate_dev_window`. Tablo `pit_v2_seed42.parquet` (SHA-256 `9116c571…`), hedef `open_to_open` (Oₜ₊₁ → Oₜ₊₂, K12 yürütme ufku), pencere 2013-06-01..2020-01-01, 47.199 satır. Aşağıdaki sayılar bu dosyadan yuvarlanmıştır.

| Ölçüt | Değer |
|---|---|
| Bilgili sembol (`n_informative`) / BH reddi (`n_bh_reject`), 30 sembol | 0 / 0 |
| Havuzlu AUC [%95 GA], tek yönlü p (tarih bloğu 20) | 0,5044 [0,4983; 0,5105], 0,090 |
| Blok içi AUC [%95 GA], tek yönlü p | 0,5043 [0,4984; 0,5090], 0,078 |
| Sembol başına MDE (AUC) medyanı, tek / aile (güç 0,8) | 0,534 / 0,552 |
| Havuzlu BSS / ECE | +0,0041 / 0,0005 |

**Kapsam sınırı:** Kapı yalnız `p_up_cal`'in **yön** ayırt ediciliğini ölçer. S-obs bloğundaki `pred_logret` (büyüklük) ve `model_disagreement` (taban modellerin dağılımı) kapı tarafından sınanmadı. "Sinyalde bilgi yok" ifadesi bu iki sütunu kapsamaz. Doğru ifade şudur: "yön bilgisi tespit edilmedi; tespit edilebilir en küçük etki tipik sembolde AUC ≈ 0,53."

### 2.2 Bu olgudan doğrudan çıkanlar (sonuca bağlı değil)

1. **"Kalibre" sıfatı kullanılmaz.** Faz 9 M8'deki "Güven kolları" kuralına göre (hakem-finans İ2) sıfat ve güven kolları yalnız kapıyı geçen sembolde kullanılır. Kapıyı geçen sembol yok. Gerçek sinyalin güveni her yerde "sıcaklık ölçekli, ayırt edici değil" diye adlandırılır. ECE'nin küçük olması (0,0005) bunu değiştirmez: taban orana yakın bir tahmin de iyi kalibre görünür.
2. **H3 gerçek sinyalle sınanamaz.** Kalibre ve kalibresiz güvenin karşılaştırılması, ikisinde de ayırt edici bilgi yokken "kalibrasyonun değeri" sorusuna yanıt vermez. Bu karşılaştırma gerçek sinyalde **koşulmaz** ve deneme sayısına (N) girmez. H3'ün nerede sınanacağı §2.6 K-b kararına bağlıdır.
3. **Beklenen etki önceden yazılır.** Gerçek sinyal kollarının bilgi kaynaklı etkisi için önsel beklenti ≈ 0'dır. Kıyas için sentetik pozitif kontrol PC060-K12'nin AUC'si 0,60'tır (M2 §9.1) ve o sinyal bile 20 bps'de yalnız KISMİ geçti (M3 §6.7). Ek bir girdi bloğu, bilgi taşımasa da öğrenmeyi yavaşlatabilir. Bu yüzden negatif bir etki de beklenen sonuçlar arasındadır.

### 2.3 Gerçek sinyal kolları neyi sınar

Gerçek sinyal kollarının sorusu yeniden adlandırılır.

> **AQ2-gerçek:** Yön bilgisi tespit edilemeyen gerçek bir tahmin sinyali yerel gözleme ya da mesaja eklendiğinde politika iyileşiyor mu, bozuluyor mu, yoksa değişmiyor mu?

Bu bir sağlamlık sorusudur, mekanizma sorusu değildir. Canlı kararda (plan §3.3) aynı sinyal kullanılacağı için yanıtı pratik olarak da gereklidir.

**Karşılaştırmalar (her yol için: S-obs, S-msg, ikisi birden):**
- **Birincil:** gerçek ↔ **karıştırılmış gerçek**. Karıştırılmış kol aynı mimariyi ve aynı girdi boyutunu kullanır. Satırlar (günler) M3'teki S1 ile aynı yöntemle permüte edilir (`scripts/marl/build_pred_m3.py --shuffle` kalıbı). Bu karşılaştırma bilgiyi kapasiteden ve girdi boyutundan ayırır.
- **İkincil:** gerçek ↔ **tahminsiz** (M4 temeli; M3 §6.8). Bu karşılaştırma girdi eklemenin toplam etkisini ölçer.

### 2.4 Okuma tablosu (önceden sabit)

Eşikler, güç (MDE) ve test §3'te, koşudan önce sabitlenir. "Fark var" ifadesi o eşiği geçmek demektir. Aşağıdaki okumalar sonuç görüldükten sonra değiştirilmez.

| # | Sonuç | Okuma | Yazılmayacak olan |
|---|---|---|---|
| O1 | Gerçek ↔ karıştırılmış: fark yok | "Bu sinyal ve bu güçle bilgi kaynaklı bir fark tespit edilmedi." MDE ile birlikte raporlanır | "Tahmin sinyali MARL'a değer katmaz", "güven mekanizması çalışmaz" |
| O2 | Gerçek < karıştırılmış ya da gerçek < tahminsiz | "Bilgisi tespit edilemeyen sinyal politikayı bozuyor." Sağlamlık bulgusu olarak raporlanır; canlı kullanım için uyarı | Mekanizmaya dair genel bir olumsuz hüküm |
| O3 | Gerçek ≈ karıştırılmış, ikisi de tahminsizden farklı | Etki bilgiden değil girdi boyutundan ya da kapasiteden geliyor. "Tahmin bilgisi" diye yorumlanmaz | "Tahmin destekli gözlem işe yarıyor" |
| O4 | Gerçek > karıştırılmış (eşik geçildi) | Kapı sonucuyla çelişir; tek başına bulgu sayılmaz. **Keşifsel** etiketiyle raporlanır, yalnız şu üç koşulun **hepsi** sağlanırsa: (a) doğrulama penceresinde işaret tekrarlanıyor, (b) tahminsiz kola karşı da pozitif, (c) etki S-obs'ta varsa `pred_logret`/`model_disagreement` çıkarıldığında ayrıca bakılır (betimsel). Koşullar sağlanırsa okuma "kapının ölçmediği bir bilgi (büyüklük ya da dağılım) olası"dır. Doğrulama için yeni bir ön kayıt gerekir | Tez bulgusu, H2/H3 desteği |

Her sonuç, ne çıkarsa çıksın raporlanır ve deneme kaydına (N) girer (plan §3.1.3 P5c ilkesi).

### 2.5 Yasak kurtarma yolları

Gerçek sinyal kollarının sonucu görüldükten sonra şunlar **yapılmaz**. Bunlardan biri gerekirse yeni bir ön kayıt yazılır ve deneme sayısına eklenir.
- `pit_v1`'e (Cₜ → Cₜ₊₁) dönmek ya da hedef ufkunu değiştirmek.
- Özellik kümesini, model kapsamını (K11) ya da yeniden eğitim periyodunu değiştirmek.
- Sonuca bakarak sembol alt kümesi seçmek. Kapı alt kümesi şu an boş; sonradan "iyi semboller" seçilmez.
- Aynı karşılaştırmayı ek tohumla yeniden koşmak (M3 §6.7 ile aynı ilke).
- Eşiği, testi ya da birincil karşılaştırmayı değiştirmek.

§9.31 kararındaki makro/çapraz özellikli 2018+ alt deneyi bu kuralın istisnası değildir. Ayrı bir sinyal tablosudur, kendi kapısından geçer ve kendi ön kaydı olur.

### 2.6 Mekanizma iddiaları nerede sınanır; açık kararlar

H2 (güvenle ağırlıklı > güvensiz dikkat) ve Faz 9 M5'teki β/geçit kolları bir **mekanizma** iddiasıdır. Bilgisiz sinyalde böyle bir mekanizmanın sınanması beklenemez, çünkü ağırlıklandırılacak bir bilgi yoktur. Bu yüzden mekanizma iddiaları yalnız **bilgili sinyalde**, yani sentetik pozitif kontrolde (PC060-K12 ailesi) sınanır ve sonuçları sentetik etiketiyle raporlanır. Gerçek sinyal kolları §2.3'teki sağlamlık sorusuyla sınırlıdır.

Kullanıcı kararı gerektiren üç madde var. Her biri §3'ten önce kapanır.

| # | Karar | Öneri | Gerekçe |
|---|---|---|---|
| K-a | Gerçek sinyal kolları koşulsun mu? | **Evet**, §2.3'teki sağlamlık sorusuyla, ön eleme bütçesiyle (3 tohum) | AQ2'nin dört koşullu tablosu boş kalmaz; canlı karar için O2 bilgisi gerekli. Maliyet: 3 yol × (gerçek + karıştırılmış) × 3 tohum |
| K-b | H3 nerede sınanır? | **Pozitif kontrolde**, kalibrasyonu bilerek bozulmuş bir sentetik güvenle (ör. sıcaklıkla bozulmuş p) | Gerçek sinyalde sınanamıyor (§2.2 #2). Alternatif: H3 makro alt deneyine ertelenir; o tablo da kapıdan geçmezse H3 kanıtsız kalır |
| K-c | Mekanizma kollarının maliyet koşulu | **Maliyetsiz birincil, 20 bps ikincil** | M3b'de PC060-K12 bile 20 bps'de KISMİ (M3 §6.7). 20 bps'de mekanizma farkı gürültüde kaybolabilir. Ancak birincil metrik (Faz 9 M8) 20 bps sonrası Sharpe; bu ayrım raporda beyan edilir |

### 2.7 Kullanıcı kararları (2026-11-19)

Üç karar da §2.6'daki öneri yönünde verildi.
- **K-a:** Gerçek sinyal kolları koşulur: S-obs, S-msg ve ikisi birden; her yolda gerçek ↔ karıştırılmış gerçek, ön eleme bütçesiyle 3 tohum. Soru §2.3'teki sağlamlık sorusudur; okuma §2.4'teki tabloya bağlıdır.
- **K-b:** H3 pozitif kontrolde sınanır. Kalibre kol PC060-K12 ailesinin sentetik güvenini kullanır; kalibresiz kolda aynı sinyalin kalibrasyonu bilerek bozulur (ör. sıcaklıkla). Bozma yöntemi ve büyüklüğü §3'te, koşudan önce sabitlenir. Sonuç sentetik etiketiyle raporlanır ve gerçek tahmin hattının kalibrasyonu hakkında bir iddia kurmaz.
- **K-c:** Mekanizma kollarında (H2, H3, β/geçit) birincil koşul maliyetsizdir, 20 bps ikincildir. Tezin birincil metriği (Faz 9 M8: 20 bps sonrası Sharpe) ile bu ayrım raporda beyan edilir. Gerçek sinyal kollarının (K-a) maliyet koşulu §3'te belirlenir.

## 3. Kollar, bütçe ve eşikler — TASLAK (v0.3)

Bu bölüm M4a sonucu (M4 tasarımı §3.6; Aşama 2 ve 1b sürüyor) gelmeden yazıldı. **Ön kayıt değildir:** eşikler, taban kol ve açık kararlar (§3.5) M4a okuması ve kullanıcı kararıyla kapanır, ardından hakem-rl (ve gerçek sinyal kolları için hakem-finans) okur.

### 3.1 S-msg mekanizması (kod)

`marl/comm_models.py` → `CommPolicy(conf_mode=...)`. Plan §3.1'deki tanım: puan_ij = q_i·k_j/√d + β·b_j.
- **`prob`** (plandaki biçim): b_j = log c_j, c_j = kırp(0,5 + |p_j − 0,5|, c_min, c_max). β = 1 DebUnc ölçeklemesidir (Yoffe vd.; Faz 9 Y7).
- **`zlogodds`** (hakem-2 #10: ölçekten bağımsız): b_j = |logit p_j| / z_scale. z_scale, eğitim satırlarındaki |logit p|'nin std'sidir ve dışarıdan verilir.
- Tahmini olmayan gönderici (pred_ok = 0) en düşük terimi alır (log c_min ya da 0); 0,5'e sessizce düşülmez.
- β: `fixed` (0 = güvensiz dikkat, 1 = DebUnc) ya da `learned` (skaler, 0'dan başlar).
- Alıcı geçidi: gᵢ = σ(a·dᵢ + b), dᵢ alıcının kendi güvensizliği (prob: 1 − cᵢ, plandaki biçim). Mesaj gᵢ ile ölçeklenir. Başlangıçta a = 0 ve b = 4,6, yani g ≈ 0,99: geçitli kol geçitsiz kola yakın başlar (*tasarım varsayımı*).
- Karıştırılmış güven: b_j ← b_π(j), π sabit bir derangement. Marjinal dağılım aynı kalır, gönderici ile güveni arasındaki bağ kopar (hakem-2 #10'un zorunlu kolu).
- Güven gözlemden okunur (p_up_cal − 0,5 ve pred_ok sütunları). `obs_spec` ve canlı karar paritesi (K2) değişmez. `conf_mode='none'` iken model, M4a'yı koşan commit'teki (`62b4023`) `CommPolicy` ile bit-eştir (`test_marl_m5_msg` [1]). Ek parametre: öğrenilen β +1, geçit +2.

### 3.2 Koşudan önce yapılan iki ölçüm (keşif; arşivlenecek)

Betik henüz commit'lenmedi (scratchpad `conf_informative.py`). Etiket yaklaşık: ham açılıştan 1[O_{t+2} > O_{t+1}]. M4a sonrası depoya alınıp temiz ağaçtan yeniden koşulacak, JSON'u arşivlenecek. **v0.5:** arşivlendi: `scripts/marl/conf_informative_m5.py` → `measurements/phase10-m5/conf_informative.json`; sayılar aynı çıktı (§4.3).
- **Güven, isabetle ilişkili (PC060-K12):** |p − 0,5|'in beşliklerine göre yön isabeti, en düşükten en yükseğe 0,51 → 0,53 → 0,56 → 0,59 → 0,65 (üç üreteçte aynı; corr(güven, isabet) ≈ +0,10). Karıştırılmış bloklarda 0,49–0,51, corr ≈ 0. Mekanizma kolları sınanabilir bir güven sinyali alıyor.
- **`prob` biçiminin dinamik aralığı çok dar:** |p − 0,5|'in medyanı 0,008. Böylece b_j = log c_j neredeyse sabit (≈ log 0,508). β = 1 (DebUnc) kolunda göndericiler arası terim farkı ~0,01–0,1 düzeyinde kalır ve dikkat puanlarına göre ihmal edilebilir. Bu, hakem-2 #10'un "kalibre güvenin dinamik aralığı dar" bulgusunun sayısal karşılığıdır. **Öneri:** birincil biçim `zlogodds`; `prob` yalnız DebUnc'a sadık betimsel kol.

### 3.3 H3 için tasarım bulgusu (K-b'yi etkiler)

K-b kararı: "kalibrasyonu bilerek bozulmuş sentetik güven (ör. sıcaklıkla)". `zlogodds` biçiminde **küresel bir sıcaklık bozulması tanım gereği etkisizdir**. Standartlaştırma (z_scale aynı veriden) ölçeği siler, b_j birebir aynı kalır (`test_marl_m5_msg` [4]). Öğrenilen β, `prob` biçiminde de küresel ölçeği büyük ölçüde soğurur. Anlamlı bir "kalibresiz" kol, **göndericiler arası karşılaştırılabilirliği** bozmalıdır: sembol başına farklı sıcaklık (sembol içi sıralama ve AUC aynı kalır, semboller arası güven sırası değişir; test [4]'te gösterildi). Bu, H3'ü şöyle daraltır: "kalibrasyon, göndericiler arası güvenin karşılaştırılabilirliğini sağlıyor mu?". Kullanıcı kararı gerekir (K-e).

### 3.4 Kol taslağı

**(a) Mekanizma kolları — pozitif kontrol** (K-b, K-c: maliyetsiz birincil). Taban: M4a komşu sinyali düzeni (σ, `pred_m4nb_g*`). Göndericinin taşıdığı bilgi alıcı için gerekli olduğu için güvenle ağırlıklandırmanın sınanabileceği düzen budur. Taban kol M4a'da geçen iletişim kolu (şu an gat, Aşama 2 bekleniyor; K-f). 3 tohum, F = 600 bin.

| Kol | Yapılandırma | Soru |
|---|---|---|
| G0 | β = 0 (= M4a G-X; yeni koşu gerekmez, `conf_mode='none'` bit-eş) | taban |
| GL | `zlogodds`, β öğrenilen | H2: güvenle ağırlık > güvensiz dikkat |
| GLS | GL + karıştırılmış güven | GL'nin avantajı güven bilgisinden mi (kontrol) |
| GLG | GL + alıcı geçidi | parça 3: öz-güven geçidi |
| GLM | GL, gözlemdeki p sembol başına farklı sıcaklıkla bozulmuş (K-e) | H3 (daraltılmış, §3.3) |
| GD | `prob`, β = 1 sabit | DebUnc'a sadık betimsel kol (dar aralık nedeniyle ≈ G0 beklenir) |

Yeni koşu: 5 kol × 3 tohum = 15, gat ~9,7 dk → **~2,4 saat**. ADMAC'tan esinlenen öğrenilen güvenilirlik geçidi (Faz 9 M5) bu turda yok (K-h).

**(b) Gerçek sinyal kolları** (K-a: 3 tohum; sağlamlık sorusu, §2.3). `pit_v2` bloğu ortam biçimine çevrilmeli (`build_pred_m3.py` kalıbı; henüz yok).

| Kol | Politika | Blok |
|---|---|---|
| T0 | none, tahminsiz (M4 temeli) | — |
| SO / SO-S | none (S-obs) | `pit_v2` / satırları karıştırılmış |
| SM / SM-S | taban iletişim kolu + GL yapılandırması (S-msg); gözlemde aynı blok | aynı |
| SB / SB-S | ikisi birden | aynı |

Not: S-msg'de güven gözlemden okunduğu için "yalnız S-msg" ile "ikisi birden" bu kodda ayrılmaz. Ayırmak için güvenin gözlemde değil yalnız mesaj yolunda olması gerekir (gözlem değişikliği, K2). K-g.

7 kol × 3 tohum = 21 koşu → **~3,2 saat**. Maliyet koşulu K-g.

### 3.5 Kararlar

**KARAR (2026-11-19, kullanıcı):** M4b pilotu PİLOT_İŞARETİ verdi (M4 tasarımı §6.4). Kullanıcı, M5 mekanizma ön kaydının yazılmasını ve K-d..K-i'nin öneri yönünde kapanmasını onayladı. Uygulamada iki nokta var:
- **K-f:** M4a'da GEÇTİ alan kol olmadı. Taban, M4b PİLOT_İŞARETİ'ne dayanarak **gat** seçildi. Bu bir kullanıcı kararıdır; GEÇTİ kuralının sonucu değildir ve ön kayıtta öyle beyan edilir.
- **K-i:** Öneri metni M4a kalıbını (δ eşikli) söylüyor. M4a'da δ'nın dayanağı zayıf çıktı (hakem-rl M4a #2) ve M4b δ'sız kalıpla ön kaydedildi. M5 ön kaydı yazılırken iki biçim (δ'lı / M4b gibi δ'sız) kullanıcıya **yeniden sunulacak**; bu not sonuçtan önce yazıldı.

| # | Karar | Öneri (kapandı) |
|---|---|---|

| K-d | Güven biçimi | `zlogodds` birincil; `prob` yalnız GD (DebUnc) kolu (§3.2) |
| K-e | H3'ün bozması | Sembol başına sıcaklık, log-tekdüze [0,5; 2], tohumlu ve önceden sabit (§3.3) |
| K-f | Taban iletişim kolu | M4a'da GEÇTİ alan kol; ikisi de alırsa gat (graf öncülü, daha ucuz). Hiçbiri almazsa mekanizma kolları koşulmaz (M4a durma kuralı) |
| K-g | Gerçek sinyal kolları: S-msg ↔ ikisi birden ayrımı ve maliyet | Ayrım için güven kanalını yalnız mesaja taşımak gözlem sürümünü değiştirir; bu turda "S-msg ⊂ ikisi birden" beyanı. Maliyet 20 bps birincil (sağlamlık ve canlı kullanım sorusu) |
| K-h | ADMAC'tan esinlenen kol | Ertelenir; GL ve GLG sonucu varsa eklenir |
| K-i | Eşikler | M4a kalıbı (D ≥ δ, ortak HAC + Holm, sıralama, doğrulama işareti, kontrol koşulu). δ M4a'nın gözlenen gat farkı görülmeden önce değil, M5 mekanizma kolları koşulmadan önce sabitlenir |

### 3.6 Ön kayıt öncesi kararlar (2026-11-19, kullanıcı)

Üçü de mekanizma kollarının koşusundan önce, hiçbir M5 sonucu görülmeden alındı.
- **K-i: δ yok, M4b kalıbı.** Soru evet/hayırdır. Etki büyüklüğü eşiği kullanılmaz, çünkü M4a'da δ'nın dayanağı zayıftı (hakem-rl M4a #2) ve şimdi konacak bir δ da keyfi olurdu. Gereken tohum sayısı yalnız betimsel verilir (§4.6).
- **GLG (alıcı öz-güven geçidi) bu turdan çıkarıldı.** Komşu sinyali düzeninde ajan i'nin gözlemindeki sinyal kendi hissesine değil, komşusu σ(i)'ye aittir. Geçidin girdisi olan "öz-güven" bu yüzden ajanın kendi kararıyla ilgisizdir. En iyi geçit "hep açık" olur ve GLG ≈ GL beklenir; kol bir şey sınamaz. Geçidi sınamak için ajanın hem kendi hem komşusunun sinyalini taşıdığı bir düzen gerekir. Bu ayrı bir ön kayıttır ve deneme sayısına (N) girmez.
- **GLM'de bozma yalnız güven teriminde.** Gönderici başına sıcaklık T_j yalnız dikkat puanına giren güven terimini bozar. Gözlemdeki olasılık ve mesajın içeriği GL ile aynı kalır. Gözlem bozulsaydı GL ↔ GLM farkı "bozuk girdi politikayı bozar" etkisini de taşırdı. Bu kol mekanizma etkisini ayırır; gerçek bir kalibresiz tahmin hattının toplam etkisini ölçmez (§4.7).
- **Not:** §3.4'teki "G0 için yeni koşu gerekmez" notu geçersizdir. G0 yeni tohumlarla (6–8) yeniden koşulur (§4.2).
- **GLM sıcaklık aralığı [0,25; 4]** (v0.6; hakem-rl M5 #4'ün istediği ölçümden sonra, koşudan önce). K-e'deki [0,5; 2] göndericilerin güven sırasını az bozuyordu: günlük kesitsel Spearman(b_GL, b_GLM) 0,87 / 0,88 / 0,91 (tohum 6 / 7 / 8). [0,25; 4] ile 0,68 / 0,71 / 0,76 (`measurements/phase10-m5/design_checks.json`). Sabit tek kaynakta: `marl/comm_models.py::GLM_TEMP_RANGE`.

---

## 4. Mekanizma kolları ön kaydı (v0.7, PİLOT, DONDURULDU) — sonuçtan önce

### 4.1 Soru ve kapsam

**H2 (mekanizma, sentetik):** Göndericinin tahmin güveni dikkat puanına eklendiğinde (güvenle ağırlıklı dikkat), gat, komşudaki bilgiyi güvensiz dikkatten daha iyi taşıyor mu? Taşıyorsa avantaj güvenin **bilgisinden** mi geliyor, yoksa ek serbestlikten mi?

**H3 (daraltılmış, §3.3):** Güven göndericiler arasında karşılaştırılabilir olmazsa (gönderici başına farklı sıcaklık) avantaj kayboluyor mu?

- Sinyal sentetik pozitif kontroldür (PC060-K12, AUC ≈ 0,60, **bilerek gelecekten**). Sonuçlar "sentetik" etiketiyle raporlanır. Gerçek tahmin hattı ya da onun kalibrasyonu hakkında iddia kurulmaz (§2.6, K-b).
- Bu bir **pilottur**: 3 tohum, ön eleme bütçesi. Etiketler karar kapısı değil işarettir (M4b ile aynı dil). Uzun koşu (M8 öncesi) ayrı bir ön kayıttır.
- Gerçek sinyal kolları (§3.4 b) bu ön kayda **girmez**. Kendi ön kaydı olur ve hakem-finans da okur.
- Taban kol gat'tir. Bu seçim M4b PİLOT_İŞARETİ'ne dayanan bir **kullanıcı kararıdır**, GEÇTİ kuralının sonucu değildir (K-f, §3.5).

### 4.2 Düzen (M4a/M4b ile aynı)

- **Komşu sinyali:** Ajan j'nin gözlemindeki tahmin bloğu kendi hissesine değil, gat grafında komşusu σ(j)'ye aittir. σ, M4a'da donduruldu (`measurements/phase10-m4/neighbor_map.json`). Hisse i'nin bilgisi yalnız komşu ajanın gözlemindedir; bu yüzden onu ancak iletişim taşıyabilir.
- **Bloklar:** `data/marl/pred_m4nb_g{6,7,8}.npz`, kayıt `measurements/phase10-m5/neighbor_map_m5.json` (kaynak bloklar `pred_blocks_m5.json`; karıştırılmışlar `pred_blocks_m5_shuf.json` yalnız `build_pred_m4` girdisi, bu ön kayıtta kol yok). Kaynak AUC 0,598 / 0,600 / 0,600. Sızıntı: komşu bloğunun ajanın kendi etiketiyle AUC'si 0,527 / 0,530 / 0,528. Yani iletişimsiz politika da komşu sinyalinden kısmi değer alabilir; bu GL ↔ G0 farkını küçültür, tutucudur.
- **Yeni tohumlar 6, 7, 8**, üreteç = tohum. M4b'nin 3–5'i yeniden kullanılmaz: onların sonucu görüldü ve taban seçimi o sonuca dayandı (seçim yanlılığı).
- **Sabitler:** gat (sector+mst, 65 kenar), hidden 110, heads 5, 1 tur; MAPPO, eleştirmen MLP (tüm yerel gözlem), α 0,25, **maliyetsiz** (K-c), F = 600.000, lr 5e-4, `init_std` 0,12, son ağırlıklar, deterministik tam pencere değerlendirme. Eğitim 2013-06..2018-12, doğrulama 2019–2020; 2021+ okunmaz.
- **Eşleşme:** Aynı tohumda GL, GLS ve GLM, G0 ile **birebir aynı başlangıç ağırlıklarından** (politika ve eleştirmen) başlar ve β = 0 olduğu için başlangıçta aynı işlevi görür (2026-11-19, sentetik panelde ölçüldü). Kollar arasındaki fark yalnız öğrenilen β'dan doğar.
- **GPU tekrarı bit-eş (v0.6):** G0 yapılandırması aynı tohumla GPU'da iki kez koşuldu (30 bin kare). Politika özeti birebir aynı çıktı (`measurements/phase10-m5/gpu_repeat_g0.json`). Bu yüzden aynı tohumlu tekrar koşu bir gürültü tabanı vermez. Taban, işlevi G0'a çok yakın ama yörüngesi farklı olan GD kolundan alınır (§4.4 (v)). Genel GPU bit-eşliği yine iddia edilmez (M3 §1); ölçüm yalnız bu yapılandırma ve sürücü içindir.

### 4.3 Kollar (5 kol × 3 tohum = 15 koşu)

Güven terimi b_j göndericiye aittir, alıcıdan bağımsızdır: puan_ij = q_i·k_j/√d + β·b_j (§3.1).

| Kol | Açık ad | Yapılandırma | Parametre | Soru |
|---|---|---|---|---|
| G0 | güvensiz gat | `--comm gat` (= M4a/M4b G-X) | 78.982 | taban |
| GL | güvenle ağırlıklı gat | + `zlogodds`, β öğrenilen (0'dan) | 78.983 | H2 |
| GLS | karıştırılmış güvenli gat | GL + güven sabit bir derangement ile göndericiler arasında karıştırılmış (tohum = eğitim tohumu) | 78.983 | H2 kontrolü: avantaj güven bilgisinden mi |
| GLM | bozuk kalibrasyonlu gat | GL + yalnız güven teriminde p_j ← σ(T_j·logit p_j), T_j ~ log-tekdüze [0,25; 4] (v0.6), üreteç = eğitim tohumu | 78.983 | H3 (daraltılmış) |
| GD | DebUnc gat | `prob`, β = 1 sabit | 78.982 | DebUnc'a sadık kol ve **gürültü tabanı** (§4.4 (v)); aralık dar (§3.2) → ≈ G0 beklenir |

- **Tasarım ölçümü** (v0.6, `measurements/phase10-m5/design_checks.json`, koşudan önce): GLS derangement'ı tohum 6 / 7 / 8'de 30 göndericinin 6 / 4 / 4'ünü bir gat komşusuna eşliyor; rastgele beklenti %14,9 (≈ 4,5). Karıştırma graf komşuluğuyla belirgin ilişkili değil. GLM bozması: günlük kesitsel Spearman(b_GL, b_GLM) 0,68 / 0,71 / 0,76 (ön kayıt kuralı: > 0,9 ise H3'ün gücü baştan düşük beyan edilir; geçmedi).

- **z_scale** (zlogodds ölçeği): eğitim satırlarında, tahmini olan hücrelerde |logit p|'nin std'si. Eğitici hesaplar ve kayda yazar. GLM'de bozulmuş terimden (|T_j·logit p|) ölçülür, böylece küresel ölçek GL ile aynı kalır ve fark yalnız göndericiler arası sıradan gelir. Duman testinde (tohum 6) GL 0,0288, GLM 0,0373.
- **Keşif ölçümü** (§3.2, arşiv `measurements/phase10-m5/conf_informative.json`, M4a blokları, eğitim satırları): güvenin beşliklerine göre isabet 0,51 → 0,65; corr(güven, isabet) ≈ +0,10; karıştırılmış bloklarda ~0,50 ve ≈ 0. Terim aralığı (p90 − p10): `prob` 0,036, `zlogodds` 2,5. Mekanizma sınanabilir bir güven sinyali alıyor; `prob` biçimi göndericileri neredeyse ayırmıyor.

### 4.4 Okuma kuralı (önceden sabit; `scripts/marl/analyze_m5.py`)

D(a, b) = ort_s[SR(a_s) − SR(b_s)], ortak HAC ile (`sharpe_diff_hac_pooled`), maliyetsiz X, SR yıllık, aynı tohum ve tarihler eşlenir. k = tohum sayısı, m = ⌈2k/3⌉ (3 tohumda 2).

**H2 (birincil):**
- **(i)** Eğitimde D(GL, G0) > 0 ve tek yönlü p < 0,05.
- **(ii)** Eğitimde D(GL, GLS) > 0 ve tek yönlü p < 0,05. Bu, fark farkına eşittir: D(GL, G0) − D(GLS, G0), aynı G0 koşularıyla. Doğrudan eşlenik test edildiği için M4b'deki bağımsızlık varsayımı gerekmez.
- **(iii)** Doğrulamada D(GL, G0) > 0 ve en az m tohumda Δ > 0.
- **(iv) Mekanizma kullanıldı** (v0.6, hakem-rl M5 #2): en az m tohumda eğitim sonu β_GL ≥ 0,04. Dayanak, sonuçtan önce ölçülen terim aralığıdır: `zlogodds` teriminin p90 − p10'u ≈ 2,5 (`conf_informative.json`), exp(0,04 × 2,5) ≈ 1,105. Yani en güvenli ve en güvensiz gönderici arasında en az %10 dikkat oranı farkı. Eşik bir mekanizma kullanım eşiğidir, getiri etkisi eşiği (δ) değildir.
- **(v) Gürültü tabanı** (v0.6, hakem-rl M5 #1): eğitimde D(GL, G0) > |D(GD, G0)|. GD'nin terimi göndericileri ~0,036 ayırır, işlevi G0'a çok yakındır ama yörüngesi farklıdır. D(GD, G0), "neredeyse etkisiz bir dokunuşun" aynı testte ne kadar fark üretebildiğini gösterir. GD'nin gerçek bir küçük etkisi varsa bu koşul tutucu kalır.
- **Etiketler:** PİLOT_İŞARETİ = (i)–(v) birden. KISMİ = (i) var ama hepsi değil. GÖZLENMEDİ = (i) yok. EKSİK = birincil kollardan biri (G0, GL, GLS, GD; eğitim ya da doğrulama) ya da GL'nin eğitim kaydındaki β eksik; o zaman etiket okunmaz. Betimsel bir çiftin eksikliği etiketi etkilemez. δ yok (§3.6).
- **Tek okuma (öncelik sırası; kod `reading_H2`):** KISMİ'de sağlanmayan ilk koşul okumayı belirler: (iv) → `guvenden_bagimsiz`, (ii) → `ek_serbestlik`, (v) → `gurultu_tabani`, (iii) → `dogrulamada_tutmuyor`. GÖZLENMEDİ'de (iv) varsa `guven_kullanildi_fark_yok`, yoksa `guven_kullanilmadi`. PİLOT_İŞARETİ'nde `bilgi_isareti`.

**H3 (ikincil), sabit sıra:** Yalnız H2 = PİLOT_İŞARETİ iken etiketlenir. Mekanizma güveni kullanmıyorsa kalibrasyonun değeri ölçülemez. Aksi durumda etiket SINANMADI'dır ve sayılar betimsel raporlanır. Sabit sıra (H2 → H3) aile hata oranını Holm'suz korur.
- **(i)** Eğitimde D(GL, GLM) > 0 ve p < 0,05. **(ii)** Doğrulamada D(GL, GLM) > 0 ve en az m tohum.
- PİLOT_İŞARETİ = ikisi. KISMİ = yalnız (i). Aksi GÖZLENMEDİ. H2 PİLOT_İŞARETİ iken GLM eksikse EKSİK.

**Betimsel:** Tohum başına β (GL, GLS, GLM; `conf_scalars_final`), ortalaması değil işaretleri de raporlanır.

**p'nin sınırı:** Ortak HAC p'si piyasa günü gürültüsünü taşır, eğitim tohumu varyansını taşımaz. Etiketin işaret olması, (iii)'teki tohum koşulu ve (v)'teki gürültü tabanı bu yüzdendir.

### 4.5 Önceden yazılan okumalar

Her sonuç bileşimi §4.4'teki öncelik sırasıyla aşağıdaki satırlardan **tek birine** düşer.

| Etiket / okuma anahtarı | Okuma | Yazılmayacak olan |
|---|---|---|
| H2 PİLOT_İŞARETİ / `bilgi_isareti` | "Sentetik pozitif kontrolde güvenle ağırlıklı dikkatin gat'a göre, güven bilgisinden gelen ve tek bir GD gerçekleşmesinin tabanını aşan bir işareti var." (v) bir gürültü dağılımı testi değildir. β_GLS de m tohumda ≥ 0,04 ise (`beta_GLS_ge_min_m_of_k`) (iv) güvene özgü değildir, bu yanına yazılır Uzun koşu ön kaydına aday | "H2 doğrulandı", gerçek tahmin hattına genelleme |
| H2 KISMİ / `guvenden_bagimsiz` ((iv) yok) | Getiri farkı var ama β mekanizma eşiğine ulaşmadı. Fark güvenden bağımsız; yörünge gürültüsü olası | "Güven işe yarıyor" |
| H2 KISMİ / `ek_serbestlik` ((iv) var, (ii) yok) | Avantaj karıştırılmış güvende de var. Kaynak güven bilgisi değil, ek serbestlik ya da öğrenme dinamiği | "Güven bilgisi işe yarıyor" |
| H2 KISMİ / `gurultu_tabani` ((iv), (ii) var, (v) yok) | Fark, neredeyse etkisiz kolun (GD) ürettiği farktan büyük değil. Yörünge gürültüsünden ayırt edilemedi | Mekanizmaya dair yön |
| H2 KISMİ / `dogrulamada_tutmuyor` (yalnız (iii) yok) | Eğitimde var, doğrulamada tutmuyor; aşırı uyum olası | Genel bir H2 hükmü |
| H2 GÖZLENMEDİ / `guven_kullanilmadi` ((iv) yok) | Mekanizma güveni kullanmadı (β öğrenilmedi ya da ters yönde öğrenildi) | "Güvenle ağırlık işe yaramaz" |
| H2 GÖZLENMEDİ / `guven_kullanildi_fark_yok` ((iv) var) | Güven kullanıldı ama getiri farkı bu güçle tespit edilmedi | "Etki yok" |
| H2 EKSİK | Birincil kol ya da β kaydı eksik; okunmaz, eksik koşu tamamlanır (ek tohum değil, aynı tohum) | Kısmi sonuçtan okuma |
| H3 PİLOT_İŞARETİ / `karsilastirilabilirlik_isareti` | "Sentetik kontrolde güvenin göndericiler arası karşılaştırılabilirliği avantajı taşıyor" | "Kalibrasyon MARL'a değer katar" (sentetik ve daraltılmış) |
| H3 KISMİ / `dogrulamada_tutmuyor` | Bozma eğitimde avantajı azaltıyor, doğrulamada tutmuyor | H3 hakkında yön |
| H3 GÖZLENMEDİ / `bozma_avantaji_silmedi` | Gönderici sırası bozulunca (Spearman ~0,7) avantaj kaybolmadı: avantaj sıradan değil, ya da bu güçle tespit edilmedi | "Kalibrasyon önemsiz" |
| H3 SINANMADI / `h2_isareti_yok` | H2 işareti olmadan H3 okunmaz; sayılar betimsel | H3 hakkında herhangi bir yön |
| H3 EKSİK | H2 işareti var ama GLM eksik; tamamlanır | — |
| GD ≈ G0 (betimsel) | Beklenen: `prob` terimi göndericileri ayırmıyor (aralık 0,036) | "DebUnc ölçeklemesi işe yaramaz" (yalnız bu sinyal aralığında) |

Her sonuç raporlanır ve deneme kaydına girer: H2 ve H3 birer deneme, GD betimsel.

### 4.6 Betimsel çıktılar (kararı değiştirmez)

D(GLS, G0), doğrulamada D(GD, G0), devir, tohum başına β, z_scale, eğitim kaydının git izi (`train_runs_all_clean`), GLM bozma gücü (`design_checks.json`), eğitim eğrileri (`curves_m4a.py`, her koşunun `run.log`'u). Tohum gereksinimi M4b formülüyle verilir (k = 3·((1,645 + 0,842)·SE / etki)²): (i) için gözlenen D ve %95 GA alt ucu, (ii) için gözlenen D(GL, GLS). Bağlayıcı olan çoğunlukla (ii)'dir, çünkü fark daha küçüktür. SE'nin 1/√k ile küçüldüğü varsayımı iyimserdir.

### 4.7 Sınırlar (beyan)

- **β'nın öğrenme hızı:** Adam bir skalere adım başına ≈ lr kadar adım attırır. 600.000 kare / 6.000 = 100 parti × 10 epoch = 1.000 güncelleme, yani |β| ≲ 0,5. b_j standartlaştırıldığı için (p90 − p10 ≈ 2,5) β = 0,5 göndericiler arasında ~1,25 logitlik puan farkı, yani ~3,5 kat dikkat oranı demektir. Mekanizmanın ifade gücü bu sınır içinde anlamlıdır, ama β yavaş öğrenilirse etki küçük kalır. Bu yüzden β manipülasyon denetimi olarak raporlanır. Duman testinde 20 güncellemede |β| ≈ 0,001–0,003.
- **Alıcıdan bağımsız terim:** Güvenli bir gönderici yalnız ilgili alıcıda değil, bütün komşularında öne çıkar (plan tanımı). İlgisiz alıcıda bu gürültüdür.
- **GLM mekanizmayı ayırır:** Gerçek bir kalibresiz tahminde içerik de bozulur. Bu kol o toplam etkiyi ölçmez (§3.6).
- **Komşu sızıntısı** (AUC ~0,53) G0'a da kısmi değer verir. Fark küçülür; okuma tutucudur.
- **Güç düşük:** M4b'nin tohum gereksinimi, GA alt ucuyla ~16. (iv) ve (v) koşulları gücü daha da düşürür. GÖZLENMEDİ "etki yok" diye okunmaz.
- **GLS tek çekim:** Her tohumda tek bir derangement çekilir (3 çekim). Hangi göndericiye hangi güvenin gittiği bu çekimlerle sınırlıdır. Komşuluk oranı rastgele beklentiye yakın (§4.3). Ayrıca GL'nin terimi alıcıdan bağımsız olduğu için GL ile GLS arasındaki fark küçülebilir ve (ii) güç yitirir.
- **Gürültü tabanı tek kol:** (v) GD'nin tek bir yörünge farkına dayanır (tohum başına bir). Bu bir dağılım değil, nokta karşılaştırmasıdır. GD'nin küçük gerçek etkisi tabanı büyütür (tutucu).

### 4.8 Yasak kurtarma yolları

Sonuç görüldükten sonra şunlar yapılmaz: ek tohum, eşik ya da testin değişmesi, β için ayrı öğrenme hızı ya da yeniden parametreleme, F'nin değişmesi, kol eklenmesi ya da çıkarılması, birincil karşılaştırmanın değişmesi. Gerekirse yeni bir ön kayıt yazılır ve deneme sayısına eklenir.

### 4.9 Süre ve komutlar

15 koşu, iki eş zamanlı (`--jobs 2`). M4b'de 12 koşu ~1,25 saat sürdü, buradan **~1,6 saat** beklenir. Duman testi (12 bin kare, 5 kol, tohum 6) uçtan uca geçti: eğitim + iki değerlendirme.

```
venv-marl\Scripts\python scripts/marl/run_m5.py --jobs 2          # arka planda; GPU dış yükte bekler
python scripts/marl/analyze_m5.py --out docs/development/measurements/phase10-m5/m5_results.json
python scripts/marl/curves_m4a.py --log results/marl_m5/*/s*/run.log --out docs/development/measurements/phase10-m5/m5_curves.json
```

Koşu sürerken ana ağaca commit atılmaz (eğitim kaydındaki git izi kirlenir).

### 4.10 Hakem okuması

hakem-rl (sonnet), koşudan önce, 2026-11-19. Paket: kollar, okuma kuralı, sınırlar, `analyze_m5.py`. **Karar: Küçük revizyon** (2 BÜYÜK, 4 KÜÇÜK). Hepsi koşudan önce işlendi (v0.6).

| # | Önem | Bulgu | Durum |
|---|---|---|---|
| 1 | BÜYÜK | H2 etiketinin sıfır altındaki yanlış-pozitif oranı bilinmiyor: GL ile G0 aynı başlangıçtan çıkıyor, ayrışan yörünge farkı eğitim gürültüsü olabilir, HAC p'si bunu taşımıyor | **Kabul.** Aynı tohumlu G0 tekrarı ölçüldü ve GPU'da bit-eş çıktı (`gpu_repeat_g0.json`), yani gürültü vermez. Taban GD'den alındı: (v) D(GL, G0) > \|D(GD, G0)\| (§4.4) |
| 2 | BÜYÜK | β denetimi etikete bağlı değil; eşik 0 | **Kabul.** (iv) en az m tohumda β_GL ≥ 0,04, terim aralığından türetildi; tohum başına β raporlanır; `guvenden_bagimsiz` okuması eklendi |
| 3 | KÜÇÜK | GLS tek derangement çekimi, grafla ilişkisi bilinmiyor; (ii)'nin gücü | **Kabul.** Komşuluk oranı ölçüldü (6 / 4 / 4 / 30, rastgele ≈ 4,5); sınır olarak beyan (§4.7) |
| 4 | KÜÇÜK | GLM bozmasının büyüklüğü ölçülmemiş | **Kabul.** Ölçüldü: [0,5; 2] ile Spearman 0,87–0,91 (zayıf). Kullanıcı kararıyla aralık [0,25; 4] (0,68–0,76) (§3.6) |
| 5 | KÜÇÜK | Okuma tablosunda eksik bileşimler, EKSİK tanımsız | **Kabul.** Öncelik sıralı tek okuma anahtarı ve 14 satırlık tablo (§4.5) |
| 6 | KÜÇÜK | Kod ↔ metin: betimsel çift eksikse birincil EKSİK oluyor, ≥ 2 sabit, self-test kapsamı, tohum gereksinimi yalnız (i) | **Kabul.** Birincil/betimsel ayrımı, m = ⌈2k/3⌉, self-test 10/10, (ii) için tohum gereksinimi |

Maliyet: ~38 bin token, 4 araç, ~0,9 dk (`review-panel.md` kaydı).

**İkinci okuma (SendMessage, aynı hakem): Kabul.** İki KÜÇÜK not işlendi: `bilgi_isareti` ifadesi "tek bir GD gerçekleşmesinin tabanını aşan" diye daraltıldı ((v) bir dağılım testi değil). β, Adam kaymasıyla da 0,04'e ulaşabilir; ayırıcı referans GLS'nin β'sı. `beta_GLS_ge_min_m_of_k` betimsel bayrağı eklendi ve okumada yanına yazılır, etiket değişmez. Maliyet ~48 bin token (önceki bağlamla), 5 araç, ~0,4 dk.

**DONDURULDU (v0.7, 2026-11-19).** Eşikler, kollar ve okuma bundan sonra değişmez; değişiklik = yeni ön kayıt.

### 4.11 Sonuç (2026-11-19): H2 KISMİ (`gurultu_tabani`), H3 SINANMADI

**Sade özet.** Ajanlara, komşudan gelen mesajı göndericinin ne kadar emin olduğuna göre ağırlıklandırma imkânı verdik (GL). Düz iletişimli kola (G0) göre biraz daha iyi karar verdiler. Güven karıştırılınca (GLS) bu avantaj azaldı, ve ajanlar güven ağırlığını yalnız güven gerçekken büyüttü. Ama güveni neredeyse hiç kullanmayan bir kontrol kolu (GD) da G0'ı aynı ölçüde, hatta biraz daha fazla geçti. Yani bu küçük denemede GL'nin avantajı, eğitimin kendi rastlantısal sapmasından ayırt edilemiyor. Ön kayıt bu durumu önceden "gürültü tabanı aşılmadı" diye yazmıştı. İpucu sentetik ve bilerek gelecekten; bu bir tez bulgusu değildir.

**Kaynak:** `docs/development/measurements/phase10-m5/m5_results.json` (okuma kodu `analyze_m5.py`, commit `1cd9d95`, temiz ağaç); eğriler `m5_curves.json` (`curves_m4a.py`, her koşunun `run.log`'u). Aşağıdaki sayılar bu dosyalardan yuvarlanmıştır.

**Koşu izi:** 15/15 koşu, F = 600.000, maliyetsiz, lr 5e-4, tohum 6–8, hepsi commit `1cd9d95`'ten, temiz ağaçla (`train_runs_all_clean`). İki eş zamanlı koşu; koşu başına eğitim 589–819 sn. Koşu sırasında GPU'da dış yük (tarayıcı vb.) ve son yarım saatte bir Docker imaj derlemesi vardı. Ayrı süreçler olduğu için eğitim matematiğini değiştirmez.

| Karşılaştırma (maliyetsiz X) | Pencere | D (SE) | Tek yönlü p | Tohum farkları (6, 7, 8) |
|---|---|---|---|---|
| GL ↔ G0 | eğitim | +0,13 (0,07) | 0,039 | +0,26 / +0,12 / −0,00 |
| GL ↔ GLS | eğitim | +0,09 (0,04) | 0,017 | +0,19 / +0,09 / −0,02 |
| GL ↔ G0 | doğrulama | +0,17 (0,11) | 0,065 | +0,11 / +0,18 / +0,21 |
| **GD ↔ G0 (gürültü tabanı)** | eğitim | **+0,18 (0,06)** | 0,001 | +0,16 / +0,27 / +0,11 |
| GLS ↔ G0 (betimsel) | eğitim | +0,04 (0,06) | 0,26 | +0,07 / +0,03 / +0,01 |
| GD ↔ G0 (betimsel) | doğrulama | +0,04 (0,10) | 0,33 | +0,04 / +0,18 / −0,09 |
| GL ↔ GLM (H3; betimsel, SINANMADI) | eğitim / doğrulama | +0,11 / +0,21 | 0,056 / 0,002 | eğitim +0,12 / +0,15 / +0,05 |

**Koşullar (§4.4):** (i) sağlandı (D > 0, p 0,039). (ii) sağlandı (p 0,017). (iii) sağlandı (doğrulamada D > 0, 3/3 tohum). (iv) sağlandı: β_GL 0,026 / 0,069 / 0,040, en az 2 tohumda ≥ 0,04. Üçüncü tohum eşiğin hemen üstünde (0,0403), yani sınırda. (v) **sağlanmadı**: D(GL, G0) 0,13 ≤ |D(GD, G0)| 0,18. Etiket **KISMİ**; öncelik sırasıyla ilk sağlanmayan koşul (v) olduğu için okuma **`gurultu_tabani`**.

**Okuma (§4.5'ten, önceden yazılmış):** Fark, neredeyse etkisiz kolun (GD) ürettiği farktan büyük değil; yörünge gürültüsünden ayırt edilemedi. Mekanizmaya dair bir yön yazılmaz. H3 kural gereği **SINANMADI** (`h2_isareti_yok`); GL ↔ GLM sayıları yalnız betimseldir.

**Betimsel (kararı değiştirmez):**
- **β güvene özgü öğrenildi:** β_GLS 0,002 / −0,002 / 0,013; `beta_GLS_ge_min_m_of_k` = yanlış. Gradyan β'yı yalnız güven gönderici ile eşleşikken büyüttü. Bu, mekanizmanın güven bilgisini *kullanmaya başladığını* gösterir. Getiri farkının gürültüden ayrıldığını göstermez. GLM'de β 0,046 / 0,003 / 0,006.
- **Gürültü tabanının kendisi "anlamlı" çıktı:** D(GD, G0) eğitimde p ≈ 0,001, doğrulamada ise +0,04 (p 0,33). İki açıklama bu tasarımla ayrılamaz: (a) GD'nin küçük bir gerçek etkisi var, (b) aynı başlangıçtan ayrışan yörüngeler ortak HAC testinde "anlamlı" bir fark üretebiliyor (§4.4 "p'nin sınırı"). (b), 3 tohumlu pilotlarda ortak HAC p'sinin tek başına okunmaması gerektiğini somutlaştırır. M8 öncesi uzun koşu ön kaydı için bir girdidir: gürültü tabanı tek bir gerçekleşme değil, bir dağılım olmalı (hakem-rl M5 2. okuma #1).
- **Eğriler:** Aynı tohumdaki kollar ilk 10 partide neredeyse aynı başlıyor (örneğin tohum 6'da 13,6–13,7). Bu, ortak başlangıç ağırlıklarının gözle görülen karşılığıdır. Sonra ayrışıyorlar: tohum 7'de partilerin ortasında G0 8,9, GD 12,8. Ayrışma yörünge gürültüsü okumasıyla tutarlıdır.
- **Devir:** G0 0,87, GD 0,87, GL 0,80, GLS 0,78 (eğitim, ortalama).
- **Tohum gereksinimi** (M4b formülü, iyimser): gözlenen etkiyle (i) için 7, (ii) için 5 tohum; (i)'nin GA alt ucu ≤ 0 olduğu için o biçimde hesaplanmaz. Bu sayılar gürültü tabanını hesaba katmaz.

**Yazılmayacak olan:** "Güvenle ağırlıklı iletişim işe yarıyor", "H2 desteklendi", "güven işe yaramaz", H3 hakkında herhangi bir yön, gerçek tahmin hattına genelleme.

**Yasak kurtarma (§4.8):** Aynı karşılaştırma ek tohumla, farklı bir gürültü tabanıyla ya da β için ayrı bir öğrenme hızıyla yeniden koşulmaz. Mekanizma sorusu yeniden sorulacaksa yeni bir ön kayıt yazılır (uzun koşu, gürültü tabanı dağılımı) ve deneme sayısına eklenir.

**Deneme kaydı:** H2 ve H3 birer deneme (H3 SINANMADI olarak), GD betimsel (§4.5).

## 5. Gerçek sinyal kolları ön kaydı (TASLAK v0.9) — sonuçtan önce

**Durum:** Taslak. Kol seti kullanıcı onayı bekliyor (öneri aşağıda). Koşu kullanıcının takviminde; 2026-11-19 kararı: "şu an koşu yapamayız, planlayalım". Hakem okuması (hakem-finans + hakem-rl) koşudan önce. Okuma kuralı §2.4 (O1–O4) ve yasak kurtarma yolları §2.5 **değişmez**; bu bölüm onları uygulanabilir eşiklere çevirir.

### 5.1 Soru

AQ2-gerçek (§2.3): yön bilgisi tespit edilemeyen gerçek bir tahmin sinyali gözleme ya da gözlem + mesaja eklendiğinde politika iyileşiyor mu, bozuluyor mu, değişmiyor mu? Bu bir **sağlamlık** sorusudur. Önsel beklenti, bilgi kaynaklı etkinin ≈ 0 olmasıdır (§2.2 #3). Ek girdi öğrenmeyi yavaşlatabileceği için negatif bir etki de beklenen sonuçlar arasındadır.

### 5.2 Düzen

- **Sinyal:** `pit_v2_seed42.parquet` (SHA-256 `9116c571…`, P5; hedef K12 ufku Oₜ₊₁ → Oₜ₊₂). Bilgi kapısından geçmedi (0/30, §2.1). **Komşu sinyali yok:** her ajan kendi hissesinin sinyalini görür.
- **Bloklar** (`scripts/marl/build_pred_m5real.py`, kayıt `measurements/phase10-m5/pred_blocks_real.json`): gerçek blok tek (tablo tek), karıştırılmış blok tohum s ile, ikinci karıştırma tohum 1000 + s ile (gürültü tabanı). Biçim M3 ile aynı (`build_pred_m3.to_arrays`).
- **Koşudan önce ölçülen (betimsel):** eğitim satırlarında K12 etiketiyle AUC gerçek 0,503, karıştırılmış 0,502–0,508. Kapsam (tahmini olan hücre payı) gerçek 0,949, karıştırılmış ~0,959. Permütasyon pencere dışındaki günleri pencereye taşıdığı için pay biraz farklıdır; bu küçük bir girdi farkıdır ve beyan edilir.
- **Sabitler:** 20 bps gidiş-dönüş (K-g; sağlamlık ve canlı kullanım sorusu), α 0,25, MAPPO, eleştirmen MLP (tüm yerel gözlemler), F = 600.000, lr 5e-4, `init_std` 0,12, son ağırlıklar, deterministik tam pencere değerlendirme. Eğitim 2013-06..2018-12, doğrulama 2019–2020; 2021+ okunmaz.
- **Tohumlar 9, 10, 11** (yeni). 3–5 M4b'de, 6–8 M5 mekanizma kollarında kullanıldı ve sonuçları görüldü.

### 5.3 Kollar (öneri: 8 kol × 3 tohum = 24 koşu)

| Kol | Politika | Tahmin bloğu | Rol |
|---|---|---|---|
| T0-N | none (MLP) | yok | gözlem yolunun tahminsiz karşılığı |
| SO | none | gerçek | S-obs |
| SO-S | none | karıştırılmış (tohum s) | S-obs kontrolü |
| SO-S2 | none | ikinci karıştırma (1000 + s) | S-obs gürültü tabanı |
| T0-G | gat (sector+mst) | yok | ikisi-birden yolunun tahminsiz karşılığı |
| SB | gat + GL (`zlogodds`, β öğrenilen) | gerçek | ikisi birden (gözlem + güvenli mesaj) |
| SB-S | gat + GL | karıştırılmış | ikisi-birden kontrolü |
| SB-S2 | gat + GL | ikinci karıştırma | ikisi-birden gürültü tabanı |

- K-g gereği "yalnız S-msg" kolu yok: güven gözlemden okunduğu için S-msg bu kodda ikisi birdenin içindedir (S-msg ⊂ ikisi birden).
- SB, GL yapılandırmasını kullanır. M5 mekanizma sonucu KISMİ'dir (§4.11). Bu kol mekanizmanın değerini sınamaz; yalnız gerçek sinyal bu yoldan verildiğinde ne olduğunu sınar.
- Alternatifler (kullanıcıya sunuldu, karar bekliyor): gürültü tabansız 6 kol (18 koşu), ya da yalnız gözlem yolu (4 kol, 12 koşu).

### 5.4 Okuma kuralı (önceden sabit; `scripts/marl/analyze_m5real.py`)

Her yol y ∈ {SO, SB} için, ortak HAC ile, maliyetli X, SR yıllık, eşli tohum ve tarih:
- **Birincil:** Dᵧ = D(gerçek, karıştırılmış) eğitimde; iki yönlü p, iki yol üzerinde Holm.
- **Gürültü tabanı:** Nᵧ = |D(karıştırılmış, karıştırılmış-2)| eğitimde. İki bilgisiz kol arasındaki fark yörünge ve girdi gürültüsünü ölçer (M5 §4.11 dersi).
- **"Fark var" (A, B karşılaştırmasında):** (a) Holm düzeltmeli iki yönlü p < 0,05, (b) |D(A, B)| > Nᵧ, (c) doğrulamada D aynı işaretli ve en az m = ⌈2k/3⌉ tohumda aynı işaret. Üçü birden.
- **İkincil:** D(gerçek, tahminsiz) ve D(karıştırılmış, tahminsiz), her yolda; dört test üzerinde Holm, aynı (a)–(c).

**Okuma, yol başına, öncelik sırasıyla (§2.4):**

| Sıra | Koşul | Okuma | Anahtar |
|---|---|---|---|
| 1 | Gerçek ↔ karıştırılmış fark var, D < 0 | O2: bilgisi tespit edilemeyen sinyal politikayı bozuyor (sağlamlık bulgusu; canlı kullanım uyarısı) | `O2_bozma` |
| 2 | Gerçek ↔ karıştırılmış fark var, D > 0, ve gerçek ↔ tahminsiz de fark var ve pozitif | O4 keşifsel: kapının ölçmediği bir bilgi (büyüklük ya da dağılım) olası. Doğrulama yeni ön kayıt ister; §2.4 O4 (c) bu turda koşulmaz | `O4_kesifsel` |
| 3 | Gerçek ↔ karıştırılmış fark var, D > 0, ama O4 koşulları eksik | Kapı sonucuyla çelişen ve tekrar etmeyen bir işaret; bulgu değil | `O4_eksik` |
| 4 | Gerçek ↔ karıştırılmış fark yok; gerçek ↔ tahminsiz ve karıştırılmış ↔ tahminsiz ikisi de fark var ve aynı işaretli | O3: etki bilgiden değil, girdi boyutundan ya da kapasiteden | `O3_girdi` |
| 5 | Gerçek ↔ karıştırılmış fark yok; gerçek ↔ tahminsiz fark var ve D < 0 (karıştırılmış ↔ tahminsiz değil) | O2 (tahminsiz maddesi): sinyal tahminsize göre bozuyor | `O2_tahminsize_gore` |
| 6 | Hiçbiri | O1: bu sinyal ve bu güçle bilgi kaynaklı fark tespit edilmedi; MDE ve tohum gereksinimiyle raporlanır | `O1_fark_yok` |

Eksik kol ya da eğitim kaydı → `EKSIK` (okunmaz). Yazılmayacaklar §2.4'teki tablodadır.

### 5.5 Betimsel (kararı değiştirmez)

Tohum başına β (SB, SB-S, SB-S2), z_scale, devir, kapsam, Nᵧ'nin tohum farkları, eğitim eğrileri, tohum gereksinimi (M4b formülü). Gerçek ↔ karıştırılmış için MDE yerine gözlenen SE ile "en küçük tespit edilebilir D" (1,96 + 0,84)·SE betimsel verilir.

### 5.6 Sınırlar (beyan)

- Tek bir tahmin tablosu (model tohumu 42). Sonuç başka bir tahmin hattına genellenmez.
- 3 tohum; ortak HAC p'si tohum varyansını taşımaz. Gürültü tabanı her yolda tohum başına bir gerçekleşmedir (3 çekim). Dağılım değildir, ama M5'teki tek kola göre daha iyidir.
- Kapsam farkı (gerçek 0,949, karıştırılmış ~0,959): küçük bir girdi farkı; karıştırılmış ↔ karıştırılmış-2 arasında yoktur.
- SB, mekanizması KISMİ olan GL yapılandırmasını taşır; "güvenli mesaj işe yarıyor" okunmaz.
- 20 bps'de pozitif kontrol bile KISMİ kalmıştı (M3 §6.7); bu düzenin gücü düşüktür.

### 5.7 Yasak kurtarma

§2.5 geçerli. Ek olarak: ek tohum, başka bir P tablosu, maliyetin değişmesi, kolların sonuca bakılarak çıkarılması, gürültü tabanının sonradan yeniden tanımlanması yapılmaz.

### 5.8 Süre ve komutlar

24 koşu, iki eş zamanlı. M5 mekanizma kolları (gat, 15 koşu) 102 dakikada bitti (~6,8 dk/koşu); `none` kolları daha hızlıdır (M4 süre ölçümü: 200 bin karede none 172 sn, gat 205 sn). Tahmin **≤ ~2,7 saat**. Koşu öncesi GPU/CPU boş olmalı.

```
python scripts/marl/build_pred_m5real.py --seeds 9,10,11 --json docs/development/measurements/phase10-m5/pred_blocks_real.json
venv-marl\Scripts\python scripts/marl/run_m5real.py --jobs 2
python scripts/marl/analyze_m5real.py --out docs/development/measurements/phase10-m5/m5real_results.json
```

### 5.9 Hakem okuması

hakem-finans (20 bps, gerçek sinyal, Sharpe çıkarımı) + hakem-rl (kollar, gürültü tabanı, okuma kodu), koşudan önce. Sonuç bu alt bölüme eklenir.

---

## Versiyon geçmişi

- **v0.9 (2026-11-19):** §5 gerçek sinyal kolları ön kaydı TASLAK: 8 kol (T0-N, SO, SO-S, SO-S2, T0-G, SB, SB-S, SB-S2), 20 bps, tohum 9–11, gürültü tabanı karıştırılmış ↔ karıştırılmış-2, okuma O1–O4 öncelik sırasıyla. Kod `build_pred_m5real.py`, `run_m5real.py`, `analyze_m5real.py` (self-test 7/7); bloklar arşivlendi. Koşu yok.
- **v0.8 (2026-11-19):** §4.11 sonuç: H2 KISMİ (`gurultu_tabani`; (i)–(iv) sağlandı, (v) D(GL, G0) 0,13 ≤ |D(GD, G0)| 0,18), H3 SINANMADI. β güvene özgü öğrenildi (GLS'de ~0), betimsel. Gürültü tabanının kendisi eğitimde p ≈ 0,001: uzun koşu ön kaydına girdi. Ön kayıt değişmedi.
- **v0.7 (2026-11-19):** hakem-rl 2. okuma Kabul; iki KÜÇÜK not işlendi (`bilgi_isareti` ifadesi daraltıldı, `beta_GLS_ge_min_m_of_k` betimsel). **DONDURULDU.** Koşu yok.
- **v0.6 (2026-11-19):** hakem-rl 1. okuma (Küçük rev., 2 BÜYÜK) işlendi: (iv) β ≥ 0,04 en az m tohumda, (v) gürültü tabanı GD (aynı tohumlu GPU tekrarı bit-eş), öncelik sıralı tek okuma ve 14 satırlık tablo, EKSİK tanımı, m = ⌈2k/3⌉. Tasarım ölçümü (`design_checks.json`). Kullanıcı kararı: GLM sıcaklık aralığı [0,25; 4]. Koşu yok.
- **v0.5 (2026-11-19):** §3.6 kararlar (K-i δ'sız; GLG çıkarıldı; GLM bozması yalnız güven teriminde; G0 yeni tohumlarla). §4 mekanizma kolları ön kaydı (PİLOT): G0/GL/GLS/GLM/GD, tohum 6–8, H2 (i)–(iii), H3 sabit sıra, β denetimi, sınırlar. Kod `run_m5.py`, `analyze_m5.py` (self-test 5/5), eğitici bayrakları, `conf_temp`; keşif ölçümü arşivlendi (`conf_informative.json`). Koşu yok.
- **v0.4 (2026-11-19):** Kullanıcı kararları: K-d..K-h öneri yönünde; taban gat (M4b PİLOT_İŞARETİ, kullanıcı kararı); K-i'nin biçimi ön kayıtta yeniden sunulacak (§3.5). Koşu yok.
- **v0.3 (2026-11-19):** §3 TASLAK: S-msg kodu (`conf_mode` prob/zlogodds, β sabit/öğrenilen, alıcı geçidi, karıştırılmış güven), iki keşif ölçümü (güven isabetle ilişkili; `prob` aralığı dar), H3 tasarım bulgusu (standartlaştırma sıcaklığı siler), kol taslağı ve açık kararlar K-d..K-i. Ön kayıt değil; koşu yok.
- **v0.2 (2026-11-19):** Kullanıcı kararları K-a (gerçek sinyal kolları 3 tohumla koşulur), K-b (H3 pozitif kontrolde, bozulmuş kalibrasyonla), K-c (mekanizma kolları maliyetsiz birincil) (§2.7). Okuma kuralı (§2.1–§2.5) değişmedi. Koşu yok.
- **v0.1 (2026-11-19):** §2 P5c okuma kuralı: gerçek sinyal (`pit_v2`) İ3 kapısından geçmiyor (0/30). Bu yüzden "kalibre" sıfatı kullanılmıyor, H3 gerçek sinyalde koşulmuyor ve gerçek sinyal kolları sağlamlık sorusuna (AQ2-gerçek) indiriliyor. Okuma tablosu O1–O4, yasak kurtarma yolları ve üç açık karar (K-a, K-b, K-c). Koşu yok.
