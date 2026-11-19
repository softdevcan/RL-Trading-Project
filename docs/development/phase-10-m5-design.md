# Faz 10 M5 — Tahmin sinyali ve güvenle ağırlıklı iletişim: tasarım ve ön kayıt

**Sürüm:** v0.1 · 2026-11-19 · dal `feat/phase10-m5-prereg`
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.1.3 (P5 ve "P5 durumu"), sinyal yolları tablosu, AQ2 ablasyonu
- [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) M5 (Y7 yeniden çerçeveleme, ablasyon kolları), M8 "Güven kolları" maddesi, §9.31 kullanıcı kararı
- [phase-10-m3-design.md](phase-10-m3-design.md) §5–§6 (pozitif kontrol PC060-K12, karıştırılmış kontrol, M3b KISMİ), §6.8 (M4 temeli)

**Durum:** **Yalnız §2 yazıldı: P5c sonucunun okuma kuralı.** Kural, gerçek sinyalli hiçbir MARL koşusu yapılmadan önce commit'lenir (plan §3.1.3: "Kapı geçmezse M5'in nasıl okunacağı M5 ön kaydında, sonuçtan önce yazılır"). Kollar, bütçe, tohumlar ve eşikler (§3) M4'ten sonra yazılır. **Koşu yok.**

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

## 3. Kollar, bütçe ve eşikler (M4'ten sonra)

Yazılmadı. Bu bölüm şunları içerecek: kol listesi (AQ2 dört koşulu × gerçek/karıştırılmış, pozitif kontrol mekanizma kolları), tohumlar, kare bütçesi, A3 güç simülasyonuyla MDE, birincil test ve eşikler, durma kuralı, M4'ün iletişim kolundan devralınan yapılandırma. İnceleme hakem-rl ve hakem-finans tarafından bu bölüm yazılınca yapılır (review-panel.md: uzman okuması pahalı işin girdisinde).

---

## Versiyon geçmişi

- **v0.1 (2026-11-19):** §2 P5c okuma kuralı: gerçek sinyal (`pit_v2`) İ3 kapısından geçmiyor (0/30). Bu yüzden "kalibre" sıfatı kullanılmıyor, H3 gerçek sinyalde koşulmuyor ve gerçek sinyal kolları sağlamlık sorusuna (AQ2-gerçek) indiriliyor. Okuma tablosu O1–O4, yasak kurtarma yolları ve üç açık karar (K-a, K-b, K-c). Koşu yok.
