# Faz 10 M5 — Tahmin sinyali ve güvenle ağırlıklı iletişim: tasarım ve ön kayıt

**Sürüm:** v0.3 · 2026-11-19 · dal `feat/phase10-m5-arms` (worktree; M4a koşuları sürerken ana ağaca dokunulmadı)
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.1.3 (P5 ve "P5 durumu"), sinyal yolları tablosu, AQ2 ablasyonu
- [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) M5 (Y7 yeniden çerçeveleme, ablasyon kolları), M8 "Güven kolları" maddesi, §9.31 kullanıcı kararı
- [phase-10-m3-design.md](phase-10-m3-design.md) §5–§6 (pozitif kontrol PC060-K12, karıştırılmış kontrol, M3b KISMİ), §6.8 (M4 temeli)

**Durum:** **Yalnız §2 yazıldı: P5c sonucunun okuma kuralı; açık kararlar K-a, K-b, K-c kapandı (§2.7).** Kural, gerçek sinyalli hiçbir MARL koşusu yapılmadan önce commit'lenir (plan §3.1.3: "Kapı geçmezse M5'in nasıl okunacağı M5 ön kaydında, sonuçtan önce yazılır"). §3 kollar ve bütçe **TASLAĞI** (v0.3): S-msg kodu yazıldı (`marl/comm_models.py` conf_mode, `tests/test_marl_m5_msg.py` 17/17); eşikler ve açık kararlar M4a Aşama 2 + 1b sonucundan sonra kapanır. **Koşu yok.**

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

Betik henüz commit'lenmedi (scratchpad `conf_informative.py`). Etiket yaklaşık: ham açılıştan 1[O_{t+2} > O_{t+1}]. M4a sonrası depoya alınıp temiz ağaçtan yeniden koşulacak, JSON'u arşivlenecek.
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

### 3.5 Açık kararlar (M4a okumasından sonra)

| # | Karar | Öneri |
|---|---|---|
| K-d | Güven biçimi | `zlogodds` birincil; `prob` yalnız GD (DebUnc) kolu (§3.2) |
| K-e | H3'ün bozması | Sembol başına sıcaklık, log-tekdüze [0,5; 2], tohumlu ve önceden sabit (§3.3) |
| K-f | Taban iletişim kolu | M4a'da GEÇTİ alan kol; ikisi de alırsa gat (graf öncülü, daha ucuz). Hiçbiri almazsa mekanizma kolları koşulmaz (M4a durma kuralı) |
| K-g | Gerçek sinyal kolları: S-msg ↔ ikisi birden ayrımı ve maliyet | Ayrım için güven kanalını yalnız mesaja taşımak gözlem sürümünü değiştirir; bu turda "S-msg ⊂ ikisi birden" beyanı. Maliyet 20 bps birincil (sağlamlık ve canlı kullanım sorusu) |
| K-h | ADMAC'tan esinlenen kol | Ertelenir; GL ve GLG sonucu varsa eklenir |
| K-i | Eşikler | M4a kalıbı (D ≥ δ, ortak HAC + Holm, sıralama, doğrulama işareti, kontrol koşulu). δ M4a'nın gözlenen gat farkı görülmeden önce değil, M5 mekanizma kolları koşulmadan önce sabitlenir |

---

## Versiyon geçmişi

- **v0.3 (2026-11-19):** §3 TASLAK: S-msg kodu (`conf_mode` prob/zlogodds, β sabit/öğrenilen, alıcı geçidi, karıştırılmış güven), iki keşif ölçümü (güven isabetle ilişkili; `prob` aralığı dar), H3 tasarım bulgusu (standartlaştırma sıcaklığı siler), kol taslağı ve açık kararlar K-d..K-i. Ön kayıt değil; koşu yok.
- **v0.2 (2026-11-19):** Kullanıcı kararları K-a (gerçek sinyal kolları 3 tohumla koşulur), K-b (H3 pozitif kontrolde, bozulmuş kalibrasyonla), K-c (mekanizma kolları maliyetsiz birincil) (§2.7). Okuma kuralı (§2.1–§2.5) değişmedi. Koşu yok.
- **v0.1 (2026-11-19):** §2 P5c okuma kuralı: gerçek sinyal (`pit_v2`) İ3 kapısından geçmiyor (0/30). Bu yüzden "kalibre" sıfatı kullanılmıyor, H3 gerçek sinyalde koşulmuyor ve gerçek sinyal kolları sağlamlık sorusuna (AQ2-gerçek) indiriliyor. Okuma tablosu O1–O4, yasak kurtarma yolları ve üç açık karar (K-a, K-b, K-c). Koşu yok.
