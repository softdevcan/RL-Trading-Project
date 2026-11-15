# Milestone 0' protokolü v2 (m0p-v2, yalın)

**Durum: TASLAK, 2026-11-15.** Hakem-2'nin kısa okuması ve kullanıcı onayıyla dondurulur. Bağlayıcı kuralların tümü [milestone0/protocol-v2.json](milestone0/protocol-v2.json)'da. Koşum betiği JSON'un ve bu belgenin SHA-256'sını deftere bağlar; biri değişirse koşmayı reddeder.

## Neden v2

**Ne için:** Milestone 0', tek ajanlı modelin (Ansari uyarlaması) BIST-30'da ne yaptığını ve tahmin eklemenin buna bir şey katıp katmadığını ölçen karşılaştırma çizgisidir. Tezin asıl sonucu değildir; asıl sonuç Faz 10'daki çok ajanlı karşılaştırmadan gelir.

**v1 neden bırakıldı:** [m0p-v1](milestone0-protocol.md) (etiket `m0-protocol-v1`) plandaki her kararı bir etken yapıp çarpıyordu. Sonuç 1.512 eğitim ve tek süreçte ~226–509 saatti. Bu, ana soru için orantısızdı (kullanıcı, 2026-11-15). Asıl sorunun cevabı da büyük ölçüde başka bir ölçümle zaten verilmişti: P tablosunda yön bilgisi yok (blok içi AUC 0,501; açık pencerede 0,502).

v1 **hiç koşulmadı ve hiçbir sonucu görülmedi.** Bu yüzden v2'ye geçmek bir sonuç görüp tasarımı değiştirmek değildir. v1'in kararları ve hakem-2 bulguları ([milestone0-protocol.md](milestone0-protocol.md) §12) v2'de de geçerlidir; bu belge yalnız farkları ve tasarımı yazar.

## 1. Sorular ve kollar

| Soru | Kol(lar) | Rol |
|---|---|---|
| Q1: tek ajan baseline'ı | DT (tahminsiz) | Keşifsel |
| Q2: tahmin katkısı | DT ↔ DTF (P tablosu `pit_v1_seed42`, P4 eşlemesi) | **Doğrulayıcı** |
| Q3: Q2'deki boş sonucu yorumlamak | PC060 (pozitif kontrol: aynı eşleme, sinyal **bilerek** gelecekten, AUC 0,60) ↔ DT | Keşifsel, doğrulayıcı aileye girmez |

**Sabit tutulanlar:**
- Algoritma: **PPO ve A2C**. TD3 çıkarıldı: paralel koşmuyor (8 eşzamanlı TD3'te ~1,6 kat hızlanma) ve hiçbir soruya hizmet etmiyor.
- Ödül: `simple` (faz 1). Faz 1'de temel oran ve makro gözlemi yok, bu yüzden `fundamentals_pit` açığı da yok.
- Fiyat ölçeği: `causal` (K25; test fiyatları destek içinde).
- Hiperparametreler: rota sabitleri; değerleri ve kökeni (R1 öncesi açık pencere) JSON'da, betik her hücrede denetler.
- Maliyet: komisyon 0,001 tek yön (20 bps gidiş-dönüş).

**Izgara:** 2 algoritma × 3 tahmin seviyesi × 6 kat (wf-v1, 2021–2026) × 5 tohum (1001–1005) = **180 eğitim**. Tek aşama; ön eleme ve budama yok.

**v1'den çıkarılanlar:** ödül terimi kıyası (K4), fiyat ölçeği kıyası (K25), TD3, ön eleme/budama, maliyet duyarlılığıyla yeniden eğitim, PC055. Gerekirse sonradan ayrı küçük deney olarak yapılır ve N'ye sayılır.

## 2. Çıkarım

- **Birincil metrik:** A8 risksiz faiz üstü, maliyet sonrası, nominal TL günlük getiriden yıllık Sharpe.
- **Test kaydı:** `last_checkpoint`. `argmax_validation` kayıtları keşifseldir.
- **Doğrulayıcı test (tek):** DT ↔ DTF.
  - Her algoritmada 5 tohumun günlük getirisi ortalanır.
  - İstatistik: iki algoritmanın yıllık Sharpe farklarının eşit ağırlıklı ortalaması.
  - Standart hata: birleşik seriden HAC (A3).
  - İki taraflı fark testi ve eşdeğerlik TOST'u. TOST sınırı ±0,5 yıllık Sharpe.
  - α = 0,05.
- **Pencere:** 2021 katı (bir duman okuması dışında okunmamış; düşük güçlü, ~1 yıllık Sharpe farkından küçüğü tespit edilemez) ve ön kayıt sonrası ileri veri. Tüm 6 kattaki kıyas keşifseldir.
- **Kurulamaz:**
  - Pozitif kontrol olmadan "tahmin katkı vermez".
  - RL'nin piyasayı, XU030'u ya da 1/N'yi yendiği.
  - BIST-30 bileşenlerine genelleme.
- **Zorunlu dipnotlar:** 2021 duman okuması (JSON `exposure_note`); evren hayatta kalan; eski ortam protokolde açık (`obs_health`, K26 = a).

## 3. Bütçe ve koşum

- **Ölçüm** (2026-11-15; CPU, 20.000 adım, 2024 katı, deneme kaydına yazılmadı):
  - 8 eşzamanlı PPO (faz 1): 42 sn duvar.
  - 8 eşzamanlı A2C (faz 1): 32 sn.
  - 1 PPO: 31 sn. 8 paralel süreç ~6–7 kat hızlandırıyor.
- **Tahmin:** 180 eğitim, 8 paralel süreçle **~2,4 saat** duvar süresi.

```bash
python scripts/run_milestone0.py --dry-run
python scripts/run_milestone0.py --scratch --workers 2 --max-steps 2000 --limit 4   # mekanizma denemesi, kayda yazmaz
CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --workers 8              # gerçek koşum (devam eder)
```

**İşçiler:**
- Her işçinin kendi sonuç, model ve log dizini var: `results/milestone0/m0p-v2/workers/w<k>/`. Model adındaki saniye damgası, deneme kaydı ve ETA geçmişi süreçler arasında yarışmaz.
- Ana defter işçi defterlerini birleştirir. Analizde N, işçi deneme kayıtlarının birleşimidir.
- Koşum kesilirse aynı komut kaldığı yerden devam eder.
