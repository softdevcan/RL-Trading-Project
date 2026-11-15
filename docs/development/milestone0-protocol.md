# Milestone 0' protokolü (m0p-v1)

**Durum: TASLAK, 2026-11-15.** Kapanış kapısında (hakem-2) ve kullanıcı onayıyla dondurulur. Dondurulunca bu belge ve [milestone0/protocol-v1.json](milestone0/protocol-v1.json) değişmez. Koşum betiği JSON'un SHA-256'sını deftere bağlar ve dosya değişirse koşmayı reddeder ([scripts/run_milestone0.py](../../scripts/run_milestone0.py)). Değişiklik yeni bir sürümdür (m0p-v2).

**Dayanak:** Faz 9 planı §4 "Milestone 0'", §5 M8 ve §9.22–§9.32 · Faz 10 planı §6 (K4, K7, K17, K20, K25, K26) · [vision-and-scope.md](../thesis/vision-and-scope.md) §7 Milestone 0.

## 1. Amaç ve yeri

Milestone 0' eski tek ajanlı ortamda (`env/trading_env.py`) Faz 9 altyapısıyla yeniden kurulan **baseline**'dır. Ürettiği şeyler:
- Ansari vd. (2024) uyarlamasının BIST-30 replikasyonu. **Sınırlılıkları beyanlı** bir sadık replikasyon (K26 = a).
- Vision §7 ablasyonu: **DT ↔ DTF** (tahminsiz ↔ nokta-zamanlı tahminli).
- K4 ödül terimi kıyası (`ema_level` ↔ `differential`) ve K25 fiyat ölçeği kıyası (`stats` ↔ `causal`).
- M8'in seçim kuralı etkeni (argmax ↔ son nokta).

**Tezin birincil kanıtı değildir.** Birincil karşıtlık yeni ortamda: MAPPO ↔ SARL-eşmimari (Faz 10 §2.3, K15). Eski ortamın hiçbir kolu protokolde temiz değil (`obs_health` türbülans, K26; Faz 9 §9.9). Milestone 0' sayıları bu açık maddelerle birlikte raporlanır.

## 2. Dondurulan girdiler

| Girdi | Değer | Kaynak |
|---|---|---|
| Evren | Panelin 30 hissesi, dondurulmuş, hayatta kalan (K7). Tarihsel üyelik maskesi kullanılmaz | `data/universe/bist30_panel.json`; E1 beyanı §9.32 |
| RL penceresi | 2013-06'dan (K17) | `RL_DATA_START` |
| Bölme | `wf-v1`, özet `33a196c602976de1`, 6 kat (test 2021..2026), purge 0 | `evaluation/splits.py`; `test_rl_walk_forward` |
| Panel | Koşum anındaki gösterge paneli; her kayıt `data_sha256` taşır (C3). Tam yeniden indirme geçmişi değiştirebilir, sonuçlar SHA'ya bağlıdır | §9.11, Faz 10 §2 |
| Tahmin tablosu (DTF) | `pit_v1_seed42`, SHA-256 `747bd39a…eae9`, eşleme P4 v1 | `pit_v1.json`, §9.28, §9.32 |
| Kod | Dondurma commit'i araştırma etiketi `m0-protocol-v1` alır | [versioning.md](versioning.md) "Araştırma etiketleri" |

## 3. Etkenler (ön eleme: 3 × 3 × 2 × 2 = 36 yapılandırma)

| Etken | Seviyeler | Neden (plan) |
|---|---|---|
| Algoritma | PPO, A2C, TD3 | Faz 9 §4 Milestone 0' tanımı |
| Ödül | `simple` (faz 1) · `psr_ema_level` (faz 2) · `psr_differential` (faz 2) | Ansari uyarlaması; K4 iki terimi de istiyor (§9.23) |
| Fiyat ölçeği | `stats` (B9) · `causal` (W = 252) | K25 = (b), iki kol |
| Tahmin | DT (tahminsiz) · DTF (P tablosu, P4) | Vision §7; §9.32 |

**Sabitler:** 200.000 adım (vision §7); başlangıç bakiyesi 1.000.000 TL; komisyon 0,001 tek yön (gidiş-dönüş 20 bps, M8 birincil); en fazla 100 pay/işlem; `missing_data_mode='own_last'`; 10 doğrulama kontrol noktası (R1, `N_CHECKPOINTS`). Hiperparametreler SB3 varsayılanları + rota öğrenme oranı: HPO koşulmaz. Her kayıt `hyperparams_source` taşır.

## 4. Seçim kuralı ve iki okuma (M8, R1)

- Her eğitim 10 kontrol noktasında doğrulamada ölçülür. Metrik doğrulama dönemi fazla getiri Sharpe'ı (A8 risksiz faiz).
- **Aynı eğitimden iki test kaydı:**
  - `selection_rule='argmax_validation'`: doğrulamada en iyi nokta.
  - `selection_rule='last_checkpoint'`: bütçenin sonu.
  - Her ikisi de N'ye sayılır ve `paired_trial_id` ile bağlıdır. Seçilen nokta zaten sonsa `same_policy=True` olur ve getiriler aynıdır.
  - Kabul: `test_rl_walk_forward` 98.
- Seçim metriğinin negatif fazla getiri rejimindeki davranışı (§9.12) ve `picks_agree` raporlanır.

## 5. Aşamalar ve budama kuralı

**Ön eleme** (`protocol-v1.json` → `prescreen`):
- 36 yapılandırma × **5 kat (2022–2026)** × 3 tohum (101, 202, 303) = **540 eğitim** (1.080 test kaydı).
- **2021 katı ön elemede koşulmaz.** Testi henüz okunmamış tek kat bu (`eval_window_unexposed`, §9.11). Rota her koşumda testi okuduğu için ön elemede koşmak bu pencereyi harcar.
- **Seçim yalnız doğrulamadan.** Yapılandırma puanı, seçilen kontrol noktasının doğrulama fazla getiri Sharpe'ının kat × tohum medyanı. Test metrikleri seçime girmez.

**Budama kuralı (öneri, KARAR 1):**
- Hipotez kıyaslarını taşıyan etkenler nihaiye **tümüyle** taşınır: tahmin, fiyat ölçeği, ödül.
- Yalnız algoritma budanır: doğrulama puanının yapılandırmalar üzerindeki medyanı en yüksek **tek** algoritma. Eşitlikte PPO.
- Nihai: 1 × 3 × 2 × 2 = **12 yapılandırma**.
- Kural mekanik uygulanır; çıktısı `protocol-v1-final.json` olarak ayrıca dondurulur.

**Nihai** (`final`):
- 12 yapılandırma × **6 kat (2021 dahil)** × 10 önceden listelenmiş tohum (1001–1010) = **720 eğitim**.
- **Maliyet duyarlılığı (hakem-rl İ3):** DT ve DTF'nin en iyi doğrulama yapılandırması 0 ve 50 bps ile yeniden eğitilir: 2 × 2 × 6 × 10 = 240 eğitim. 0–50 bps ex post ızgara (D4) yalnız raporlama içindir.

## 6. Çıkarım

- **Birincil metrik (M8):** A8 risksiz faiz üstü, 20 bps maliyet sonrası, nominal TL günlük getiriden yıllık Sharpe. İkincil: Sortino, Calmar, MDD, devir. Reel (TÜFE) ve USD birimlerinde **aylık** Sharpe (D3, §9.30 F4–F5).
- **Çıkarım birimi (K20):** yapılandırma × tohum başına katların örneklem dışı günlük getirilerinin birleştirilmiş serisi.
- **Kıyaslar:**
  - DT ↔ DTF, `stats` ↔ `causal`, `ema_level` ↔ `differential`, argmax ↔ son nokta.
  - Her kıyas eşleşik: öteki etkenler ve tohum aynı.
- **Testler:**
  - Ledoit-Wolf Sharpe farkı (A3, HAC + studentize blok bootstrap).
  - DSR (A1; N deneme kaydından, `protocol_open` dahil sayılır).
  - Tohum dağılımı (A5, IQM + GA).
  - MCS (A4). Kıyas ailesi başına BH-FDR (A7).
- **Doğrulayıcı kanıt (KARAR 2):** 2022–2026 katları R1 öncesi okundu, sonuçları **keşifseldir**. Doğrulayıcı kanıt yalnız 2021 katı ve ön kayıt sonrası ileri veri. 2021 tek yıldır (~250 seans), gücü düşüktür. MDE, A3 güç simülasyonuyla bu belgede sabitlenir (açık).
- **Rejim (B6):** birincil USD bazlı XU100 genişleyen tercil; nominal ve 5 yıllık kayan kollar duyarlılık. Blok bootstrap ile çıkarım; `episodes_min20`.
- **Baseline (D):** 1/N aylık, al-tut, XU030. Evren hayatta kalan olduğu için **üstünlük çıkarımı yapılmaz** (§9.30 F3). Tablo yalnız aynı dondurulmuş evrende kıyas içindir. XU030 "evren, dönem ve fiyat tanımı farklı, kıyas değil" dipnotuyla verilir (§9.31 #6).
- **Ödül kollarında ek tablo (M8, K4):** işlem sayısı, devir hızı, komisyon payı. Komisyon ödülde iki kez sayılıyor (beyan). K4 ölçek eşitlemesi (w₂ ızgarası) bu sürümde **yok**, terimler varsayılan w₂ ile kıyaslanır (KARAR 3).

## 7. Önceden yazılan beklentiler (hipotez değil)

- **DT ≈ DTF beklenir.** P tablosunda yön bilgisi tespit edilmedi (blok içi AUC 0,501, §9.31). Eşitler dışarıda havuzlu AUC 0,503. DTF gözleminde getiri ve güven ~0; yön ikili ve tam ölçekli (§9.32).
  - DT ile DTF arasında bir fark çıkarsa önce yön özelliğinin **gürültü etkisi** düşünülür.
  - Bunu ayırmak için pozitif kontrol kolu bu sürümde yok (KARAR 4).
- `causal` kolunun test fiyat özelliklerini desteğe alması beklenir (%98,9 → %0,03 havuzlu destek dışı, §9.9). Başarıma etkisi önceden yazılmaz.

## 8. Beyan listesi (açık maddeler, koşumlarla birlikte raporlanır)

- **Eski ortam her kolda protokolde açık:** `obs_health` (türbülans doygun), `features_in_support`/`dims_in_support` (MACD, hacim, ADX; K26 = a). Duman koşumunda (PPO, 2021, `stats`) dört açık uyarı.
- **Faz 2:** temel oranlar tek anlık görüntü (`fundamentals_pit` açık); makro 2018'den önce 0. Eski ödülde r_f sabiti 0,15 (G2 yeni ortamda).
- **Ödül:** komisyon iki kez; `ema_level` düşük maruziyette −1'de doyar (§9.23).
- **Evren hayatta kalan;** tarihsel üyelik geçici (E1 §9.32).
- **DTF sinyali bilgisiz;** yön özelliği ikili (§9.32).
- **C1:** GPU belirlenimsiz. Koşumlar CPU'da (`CUDA_VISIBLE_DEVICES=-1`); bit-eşlik yalnız CPU'da iddia edilir. Ortam kaydı manifestte.
- **Golden kırmızı** (K1); Milestone 0' ile bir kez yenilenir. Tahmin hattı bu koşumlara P tablosundan girer, golden'ı kullanmaz.

## 9. Bütçe (ölçüm)

- **Duman koşumu** (2026-11-15; PPO, `simple`, `stats`, DT, kat 2021, tohum 101; 30 sembol, CPU):
  - 20.000 adım, 10 doğrulama noktası dahil 30,7 sn (adım başına 0,0015 sn).
  - Deneme kaydına yazılmadı (geçici dizin).
- **Doğrusal izdüşüm:** 200.000 adımlık eğitim ~307 sn. TD3 ve faz 2 ölçülmedi; ETA katsayıları TD3'ü ~1,45 kat yavaş veriyor.

| Aşama | Eğitim | Tek süreç (PPO hızıyla) |
|---|---|---|
| Ön eleme | 540 | ~46 sa |
| Nihai | 720 | ~61 sa |
| Maliyet duyarlılığı | 240 | ~20 sa |
| **Toplam** | **1.500** | **~127 sa** (4 paralel süreçle ~32 sa) |

## 10. Açık kararlar (kullanıcı; kapanış kapısında hakem-2'ye de soru)

1. **Budama kuralı:** yalnız algoritma budanır (öneri), ya da ön elemede en iyi k yapılandırma taşınır.
2. **Doğrulayıcı kanıt tanımı:** 2021 katı + ileri veri (öneri). MDE güç simülasyonu dondurmadan önce yapılır.
3. **K4 ölçek eşitlemesi:** varsayılan w₂ (öneri, beyanla), ya da kol başına küçük w₂ ızgarası (N'ye sayılır, maliyet ×k).
4. **DTF pozitif kontrol kolu:** sentetik kehanetli P tablosu (etiketten, AUC 0,55 / 0,60). DT ≈ DTF boş sonucunun "RL sinyali kullanamaz" mı, "sinyal yok" mu olduğunu ayırır. Planda yalnız M8/H3 için var (§9.30 R2). Milestone 0''a eklenmesi yeni karar.
5. **Yön özelliği:** ikili (P4 v1, onaylı) ↔ sürekli `p_up_cal`. Değişirse P4 v2, yeni tablo eşlemesi.

## 11. Koşum

```bash
python scripts/run_milestone0.py --dry-run                 # hücre sayısı ve ETA
python scripts/run_milestone0.py --smoke --steps 20000      # adım maliyeti; deneme kaydına yazmaz
CUDA_VISIBLE_DEVICES=-1 python scripts/run_milestone0.py --stage prescreen [--limit N] [--user <id>]
```

Defter: `results_dir()/milestone0/m0p-v1/ledger.jsonl`. İlk satırda protokol SHA'sı; tamamlanan hücre yeniden koşulmaz.
