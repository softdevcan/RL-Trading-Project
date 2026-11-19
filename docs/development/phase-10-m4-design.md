# Faz 10 M4 — İletişim kolları: tasarım ve M4a ön kaydı

**Sürüm:** v0.3 · 2026-11-19 · dal `feat/phase10-m5-prereg`
**Üst belgeler:**
- [phase-10-marl-plan.md](phase-10-marl-plan.md) §3.2.5 (kollar, eşit bütçe, GAT grafı), §8 hakem-2 #14 (kenar kuralı M4 öncesi sabitlenir) ve #15 (`obs_spec` sabit)
- [phase-9-research-validity-plan.md](phase-9-research-validity-plan.md) M4 (AQ1, G1; eşit bütçe)
- [phase-10-m3-design.md](phase-10-m3-design.md) §2 (hiperparametreler), §5 (M3a kalıbı), §6.8 (M4 temeli: MAPPO, tüm yerel gözlemler, α = 0,25; varsayılan, kanıta dayanmaz)
- [phase-10-m5-design.md](phase-10-m5-design.md) §2.6 (mekanizma iddiaları yalnız bilgili sinyalde)

**Durum:** **M4a ön kaydı v0.3 DONDURULDU (2026-11-19): hakem-rl ilk okuma Büyük revizyon, ikinci okuma Kabul (§5). Koşu yok.** Kod, ölçümler ve okuma betiği commit'lendi; kullanıcı kararları K-M4-1..4 (§4); hakem tablosu §5. Okuma betiği bu sürümü taşır (`PREREG = 'phase-10-m4-design.md §3 v0.3'`). Eşikler ve okuma kuralı bundan sonra değişmez; değişiklik gerekirse yeni sürüm (v0.4) ve yeni ön kayıt olur.

Kod ile belge farklıysa kod hatalıdır.

---

## 0. Kapsam ve soru

| Madde | Burada | Kod / ölçüm |
|---|---|---|
| Koşucu ve ön kayıtlı okuma (`--self-test` 17/17) | §3 | `scripts/marl/run_m4a.py`, `scripts/marl/analyze_m4a.py` |
| `attn` ve `gat` politika modelleri, eşit bütçe | §1 | `marl/comm_models.py`, `tests/test_marl_m4.py` (14) |
| GAT graf kenar kuralı (hakem-2 #14) | §2 | `scripts/marl/graph_m4.py` → `measurements/phase10-m4/graph_rules.json` |
| **M4a ön kaydı: iletişim boru hattı kapısı** (komşu sinyali pozitif kontrolü) | §3 | `scripts/marl/build_pred_m4.py` → `measurements/phase10-m4/neighbor_map.json` |
| Süre ölçümü | §3.5 | `measurements/phase10-m4/timing_{none,attn,gat}.json` |

**M4'ün sorusu (AQ1, H1):** İletişim, eşit bütçede iletişimsiz politikaya göre bir şey kazandırıyor mu? Nihai H1 sınaması M8 ön kaydındadır.

**Bu belgedeki ilk adım (M4a):** M3a'nın iletişim için karşılığı. Gerçek sinyalli hiçbir iletişim koşusundan önce şu sınanır: iletişim kolları, yalnız başka bir ajanın gözleminde bulunan bilgiyi doğru ajana taşıyabiliyor mu? Gerçek sinyal (`pit_v2`) kapıdan geçmediği için (M5 §2.1) gerçek sinyalle bir iletişim farkının boş çıkması, bu kapı olmadan "iletişim boru hattı çalışmıyor" ile "taşınacak bilgi yok" arasında ayırt edilemez. M4a bir geliştirme kapısıdır; sinyal bilerek gelecekten türetilir, sonuç tez bulgusu değildir.

---

## 1. Kollar ve eşit bütçe

| Kol | Politika | Parametre (gözlem 48, eylem 1 → çıktı 2) | Kaynak |
|---|---|---|---|
| `none` | BenchMARL MLP [256, 256], tanh; ajan başına, tam paylaşımlı + ajan kimliği (K24) | 78.850 | M3 ile aynı |
| `attn` | kodlayıcı 48 → 110 (tanh) → 1 tur çok başlı öz-dikkat (5 baş, artık + LayerNorm), tam bağlı → [kendi kod ‖ mesaj sonrası] → 110 (tanh) → 2 | 78.982 (+%0,17) | TarMAC tipi (Das vd. 2019); M1 spike `AttnComm` yapısı |
| `gat` | `attn` ile **aynı katmanlar ve aynı parametre sayısı**; dikkat sabit graf komşuları + kendisiyle sınırlı (ikili maske, §2) | 78.982 (+%0,17) | Veličković vd. 2018; MAGIC (Niu vd. 2021); bkz. K-M4-1 |

- **Eşit bütçe** (plan §3.2.5): parametre ±%10 sağlanıyor (`test_marl_m4` [2]; formül `comm_param_count`). Eleştirmen üç kolda aynı BenchMARL MLP'dir (girdi 30 × 48 = 1440, 434.945 parametre; M3 §6.4). M4a'da HPO yoktur, yani deneme sayısı her kolda 0'dır. Tohumlar ve kare bütçesi kollar arasında aynıdır (§3).
- **Mekanizma testi:** `gat`'ta komşu olmayan bir ajanın gözlemi değişince ajan i'nin çıktısı bit-eş kalıyor, komşununki değişince değişiyor. `attn`'de herhangi bir ajan etkiliyor (`test_marl_m4` [3]).
- **Hiperparametreler** M3 §2 ile aynı: MAPPO, lr 5e-4, 10 epoch, bölünmeyen parti 6000, başlangıç std 0,12, `'excess'`, α = 0,25, ölçek 100, 252 adımlık bölüm.

**K-M4-1 (KARAR 2026-11-19, kullanıcı; öneri yönünde):** `gat` kolu Veličković vd.'nin toplamsal puanını (LeakyReLU(aᵀ[Whᵢ ‖ Whⱼ])) değil, `attn` ile aynı ölçekli nokta çarpımı puanını graf maskesiyle kullanır. **Öneri: bu biçim.** Gerekçe: plan §2'deki karşılaştırmanın iddiası "`attn` ile `gat` arasındaki fark graf öncülünden gelir". Puan fonksiyonu da değişirse iki etken birlikte değişir ve parametre sayısı ayrıca eşlenmek zorunda kalır. Bedel: kol, özgün GAT'ın birebir uygulaması değildir. Metinde "graf maskeli dikkat (GAT tipi)" diye adlandırılır, toplamsal puan bir ablasyon olarak kalır.

## 2. GAT graf kenar kuralı

Kaynak: `docs/development/measurements/phase10-m4/graph_rules.json` (komut `python scripts/marl/graph_m4.py --out …`, commit `6d6e57c`, temiz ağaç). Graf **yalnız eğitim penceresinden** kurulur: 2013-08-27..2018-12-31, 1342 getiri günü (B0 mantığı; plan §3.2.5). Doğrulamada aynı graf değişmeden kullanılır. Getiri log(Cₜ/Cₜ₋₁), çift bazında tam gözlemli günlerle Pearson; uzaklık d = √(2(1 − ρ)) (Mantegna 1999). Sektör kenarları E2 gruplarından gelir (`data/universe/sectors.csv`, 7 grup; 5 sembollü `DIGER` bir grup sayılmaz). Kararlılık ölçüsü: eğitim penceresinin iki yarısında (2013-08-28..2016-04-28 ve 2016-04-29..2018-12-31) ayrı kurulan grafların Jaccard benzerliği.

Ortalama çiftli ρ 0,383 (medyan 0,357, aralık 0,10–0,89). Hakem-2 #14'teki 0,398 farklı bir pencereden (2013-06..2018-07) gelir.

| Kural | Kenar | Yoğunluk | Bileşen | Derece min/medyan/maks | Sektör içi pay | Yarılar Jaccard |
|---|---|---|---|---|---|---|
| sector | 47 | 0,108 | 12 | 0 / 2 / 7 | 1,00 | 1,00 |
| mst | 29 | 0,067 | 1 | 1 / 1 / 8 | 0,38 | 0,21 |
| thr 0,3 | 311 | 0,715 | 1 | 2 / 22 / 28 | 0,14 | 0,63 |
| thr 0,5 | 77 | 0,177 | 14 | 0 / 1 / 13 | 0,32 | 0,49 |
| topk 3 | 82 | 0,189 | 1 | 3 / 3 / 19 | 0,32 | 0,23 |
| **sector + mst** | **65** | **0,149** | **1** | **1 / 3 / 12** | **0,72** | **0,65** |
| sector + topk 3 | 103 | 0,237 | 1 | 3 / 5 / 19 | 0,46 | 0,44 |

Tablonun tamamı (thr 0,4, topk 2 ve 5, sector + topk 2 ve 5) JSON'dadır.

**Öneri ve KARAR (K-M4-3, 2026-11-19, kullanıcı): `sector + mst`** (plan §3.2.5'teki "E2 sektör kenarları ∪ korelasyon kenarları" tanımının MST biçimi).
- **Bağlantılı:** Her ajanın en az bir komşusu var. Yalnız sektör grafında 5 `DIGER` sembolü yalıtık kalıyor (12 bileşen).
- **Seyrek:** Yoğunluk %14,9. Böylece `attn` (tam bağlı) ile `gat` arasında gerçek bir öncül farkı oluşur. Eşik 0,3'te graf %71,5 dolu, yani `attn`'e çok yakın (hakem-2 #14).
- **Ayar parametresi yok:** τ ya da k seçmek bir hiperparametre olur ve deneme sayısına (N) girer. MST'de seçilecek bir değer yoktur.
- **Seyrek kurallar içinde en kararlısı.**

**Beyan:** `sector + mst`'nin kararlılığı (0,65) sektör kenarlarının sabit olmasından gelir. Korelasyon kısmı (MST) tek başına 0,21'de kalıyor, yani pencere yarıları arasında MST kenarlarının çoğu değişiyor. Graf her eğitim penceresinde bir kez kurulur ve sabit kalır (plan §3.2.5); dinamik graf kapsam dışıdır. Derece eşit dağılmıyor: bankalar merkezde (VAKBN 12, AKBNK ve HALKB 9, GARAN 8). M8'de walk-forward katlarının her biri kendi eğitim penceresinden yeniden kurulur ve yoğunluk her kat için raporlanır.

## 3. M4a ön kaydı taslağı — komşu sinyali pozitif kontrolü

### 3.1 Tasarım

- **Sinyal:** M3'teki PC060-K12 blokları (AUC 0,60, etiket 1[O_{t+2} > O_{t+1}]; M3 §4), bu kez **sütun permütasyonuyla**. Ajan j'nin gözlemine kendi hissesinin değil, `sector + mst` grafında bir komşusu σ(j)'nin bloğu konur. σ bir bijeksiyondur ve sabit noktası yoktur. Grafta böyle bir σ var: iki parçalı eşleme 30/30 (`neighbor_map.json` → `sigma`). Hisse i'nin bilgisi yalnız komşu ajan σ⁻¹(i)'nin gözlemindedir.
- **Neden `none` kontrol:** Paylaşımlı MLP, ajan j'nin eylemini yalnız j'nin gözlemiyle belirler. Hisse j'nin bilgisini yalnız iletişimle alabilir. Aynı gözlem, aynı parametre bütçesi ve aynı eleştirmen kullanılır; tek fark ajanlar arası bilgi yoludur.
- **Bilinen sızıntı:** Komşular korele olduğu için σ(j)'nin yön sinyali j'nin etiketi hakkında da kısmi bilgi taşır. `none` bu yüzden sıfır değil, kısmi bir değer alabilir. Sızıntı koşudan önce ölçülür (`neighbor_map.json` → `auc_holder_vs_own_label`; §3.4). Kapının birincil karşılaştırması `none`'ın eriştiği bu vekil bilgiyi zaten içerir.
- **Graf `gat` lehine hizalı:** σ kenarlar boyunca seçildiği için `gat` doğru göndericiyi graf öncülünden bilir. `attn` ise onu 29 ajan içinden öğrenmek zorundadır. M4a bu yüzden **`attn` ile `gat`'ı sıralamaz.** Yalnız her iletişim kolunun bilgiyi taşıyıp taşıyamadığını sınar.
- **Maliyet:** maliyetsiz (M3a ve M5 K-c ile aynı gerekçe). 20 bps yalnız betimsel olarak sonradan uygulanır.

### 3.2 Kollar ve aşamalar (K-M4-2: iki aşama)

| Kol | Politika | Tahmin bloğu |
|---|---|---|
| N-X | `none` | komşu sinyali (`pred_m4nb_g{s}.npz`) |
| A-X | `attn` | aynı |
| G-X | `gat` | aynı |
| N-S, A-S, G-S | aynı üç politika | aynı permütasyon, satırları karıştırılmış blok (`pred_m4nb_g{s}_shuf.npz`; M3 S1 kalıbı) |

- **Aşama 1:** N-X, A-X, G-X, lr 5e-4 (M3 §2) (9 koşu).
- **Aşama 1b — lr duyarlılığı (hakem-rl M4a #1):** Aşama 1'de (i)–(iii)'ü sağlamayan her iletişim kolu, durma kararından **önce** N-X ile birlikte lr = 2e-4'te yeniden koşulur (`results/marl_m4a/lr2e-4/`; en çok 9 koşu). Gerekçe: dikkat modeli MLP'nin lr'siyle ayar yapılmadan eğitiliyor. Tek bir lr noktası, "öğrenemedi" ile "lr artefaktı"nı ayıramaz (M3a #4 ile aynı hata biçimi). 2e-4 seçimi önceden yapıldı: dikkat modelleri için yaygın aralığın içinde, 5e-4'ün altında. Bu bir **tasarım varsayımıdır**, kaynak taranmadı. 1b yalnız aşağı yönlü tek bir lr'yi dener: 1b'den sonra GÖZLENMEDİ, "bu yönde lr duyarlılığı yok" diye okunur, "hiçbir lr'de yok" diye okunmaz (hakem-rl M4a ikinci okuma #2). Okuma aynı (i)–(iii). Holm ailesi, Aşama 1 ve 1b'deki tüm D_k testleridir (en çok 4). Aşama 1 kararları değişmez. 1b'de geçen kol Aşama 2'ye o lr ile gider. **Bedel (ikinci okuma #1):** 1b, geçmeyen kola ikinci bir atış verir ve kapıyı geçme olasılığını artırır. Holm yalnız (i)'in p'sini düzeltir; (ii) ve (iii) düzeltilmez. M4a'nın GEÇTİ'si bu yüzden yalnız bir geliştirme kapısı kararıdır, tez bulgusu sayılmaz.
- **Aşama 2:** Aşama 1 ya da 1b'de (i)–(iii)'ü sağlayan iletişim kolunun S'si ve aynı lr'de N-S. Hiçbir kol sağlamazsa Aşama 2 koşulmaz (`run_m4a.py` reddeder).

Tohumlar s ∈ {0, 1, 2}, üreteç g_s ∈ {20261115, 1, 2} (M3 §5 ile aynı eşleme). Kare bütçesi F = 600.000 (M3 F_min). Son ağırlıklar kullanılır; kontrol noktası seçimi ve erken durdurma yoktur. Değerlendirme deterministik ve tam pencere: eğitim 1343 gün, doğrulama 500 gün; 2021+ yok (M3 §3).

### 3.3 Okuma kuralı (δ = 0,25, K-M4-4; kod `analyze_m4a.py`)

Her iletişim kolu k ∈ {A, G} için Δ_{k,s} = SR(k-X_s) − SR(N-X_s) ve D_k = ort_s Δ_{k,s}. SR yıllık, maliyetsiz X üzerinde. Kalıp M3a §5 ile aynı:
- **GEÇTİ = üçü birden:**
  - (i) Eğitimde D_k ≥ δ ve ortak HAC tek yönlü p < 0,05. p, iki kol üzerinde Holm düzeltmesiyle.
  - (ii) Eğitimde min_s SR(k-X_s) > max_s SR(N-X_s).
  - (iii) Doğrulamada D_k > 0 ve en az 2/3 tohumda Δ_{k,s} > 0.
- (i)–(iii) sağlanırsa kol Aşama 2'ye geçer. Nihai GEÇTİ için ayrıca:
  - **(iv) Kontrol, Aşama 2 (hakem-rl M4a #3):** D_k^S = ort_s[SR(k-S_s) − SR(N-S_s)], eğitimde. Fark G = D_k − D_k^S ≥ δ/2 **ve** z = G / √(se_X² + se_S²) ≥ 1,645 (tek yönlü %5). se'ler ortak HAC'tan alınır. X ve S koşuları ayrı bloklarla yapıldığından bağımsızlık varsayılır. Ortak piyasa günleri iki fark arasında pozitif korelasyon verdiği için bu varsayım SE'yi büyütür, yani tutucudur. Koşul, iletişim kolunun avantajının taşınan bilgiden geldiğini, iletişimin kendisinden (ör. kesitsel teknik özellik birleştirme) gelmediğini sınar. Sağlanmazsa etiket KISMİ olur: avantaj var ama taşınan bilgiye bağlanamadı. HAC SE'leri tohum varyansını içermez (bilinen açık), bu yüzden 3 tohumla z yaklaşıktır. Tohum başına G_s = Δ_{X,s} − Δ_{S,s} betimsel olarak raporlanır (`gap_per_seed`; ikinci okuma #3).
- **KISMİ / GÖZLENMEDİ:** M3a §5 ile aynı tanımlar. GÖZLENMEDİ "bu bütçede iletişimle bilgi taşınması gözlenmedi" diye okunur, "iletişim çalışmaz" diye okunmaz. **δ'ya ulaşılamazsa** ama D_k > 0 ve Holm p < 0,05 ise etiket KISMİ olur ve karar kullanıcıya döner (hakem-rl M4a #2; kod `analyze_m4a.py::_read_arm`).
- **attn ↔ gat (hakem-rl M4a #5):** σ grafa hizalı olduğu için gat GEÇTİ alıp attn almazsa boru hattı çalışıyor demektir. attn'in sonucu "bu bütçede göndericiyi tanımlayamadı" diye okunur (ajan kimliği gözlemde olduğu için tanımlama olanaklı ama zor). attn'in M5'te koşulup koşulmayacağı kullanıcı kararıdır; otomatik elenmez. Grafa hizalı olmayan bir σ ile ek gat kolu kurulmadı (hakemin isteğe bağlı önerisi).
- **Durma kuralı:** Aşama 1b'den sonra da GEÇTİ alamayan kol için gerçek sinyalli iletişim koşuları (M5 S-msg) başlamaz; karar kullanıcıya döner.
- **Betimsel, kararı değiştirmez:** kendi sinyalli tavan = M3a A1 (aynı bloklar, aynı hiperparametre, maliyetsiz; `m3a_results.json`, okuma JSON'unda `ceiling_m3a_A1`). Ayrıca aynı maliyetsiz 1/N ve al-tut SR'si, devir ve nakit ağırlığı.
- **Beyanlar:**
  - **Fark = iletişim + mimari (hakem-rl M4a #4).** Eşit bütçe yalnız parametre sayısıdır. `none` [256, 256] MLP'dir; iletişim kolları 110 genişlikli kodlayıcı + dikkat + başlıktır. Kendi içinde iletişimsiz bir `rounds = 0` kolu yoktur. (iv)'teki S kontrolü bu karışmayı kısmen ayırır; tam ayrım M8'e kadar açık sınırlamadır.
  - **Eleştirmen her kolda aynıdır ve tüm gözlemi görür** (BenchMARL MLP, 1440 girdi). Komşu sinyalleri değer tahminine her kolda eşit girer; D_k'yi yanlı kılmaz, ama 30 ajanlı ortak ödülde gücü düşürebilir. Etkisi ölçülmedi.

### 3.4 Koşudan önce ölçülecek: sızıntı ve δ

Kaynak: `docs/development/measurements/phase10-m4/neighbor_map.json` (komut `python scripts/marl/build_pred_m4.py --seeds 20261115,1,2`, commit `b67e90e`, temiz ağaç). Hiçbir eğitim koşusundan önce ölçüldü.

| Tohum (üreteç) | Kendi sinyali → kendi etiketi (M3) | Komşu sinyali → kendi etiketi (`none`'ın vekili) | Sızıntı payı |
|---|---|---|---|
| 0 (20261115) | 0,595 | 0,527 | 0,28 |
| 1 (1) | 0,601 | 0,536 | 0,36 |
| 2 (2) | 0,600 | 0,534 | 0,34 |
| karıştırılmış bloklar | 0,500–0,504 | 0,497–0,505 | — |

Sızıntı payı = (komşu AUC − 0,5) / (kendi AUC − 0,5). Ortalaması ~0,33: `none`, hissesinin yön bilgisinin yaklaşık üçte birine komşu korelasyonu üzerinden erişiyor.

**δ = 0,25 (KARAR K-M4-4, 2026-11-19, kullanıcı; öneri yönünde).** Kaba tavan: M3a'nın eğitim farkı 0,76 (A1 ↔ S1, M3 §5.2), sızıntıdan sonra kalan pay ~0,67, yani tam taşımada D_k ≈ 0,5. Bu hesap, SR farkının AUC fazlasıyla doğrusal ölçeklendiğini varsayar. Bu bir **tasarım varsayımıdır**, ölçülmedi. δ = 0,25 bu tavanın yarısıdır: iletişim kolunun komşudaki bilginin en az yaklaşık yarısını taşımasını ister. M3b'de eğitim D'nin SE'si ~0,09 idi (M3 §6.7). Bu düzeyde 0,25'lik bir fark tek yönlü p < 0,05'e ulaşır. **Hakem-rl M4a #2:** Tavanın türetildiği fark (A1 − S1, sinyal ↔ sinyalsiz) kapının ölçtüğü farkla (iletişim ↔ iletişimsiz, aynı sızıntılı gözlem) aynı nicelik değildir. Tek turlu tek bir MHA'nın komşu bilgisinin yarısını taşıyacağı beklentisinin dayanağı yoktur. Bu yüzden δ ulaşılamaz çıkabilir. Ulaşılamazsa okuma §3.3'tedir (KISMİ, kullanıcı kararı); δ sonuçtan sonra değiştirilmez.

### 3.5 Süre

Kaynak: `docs/development/measurements/phase10-m4/timing_{none,attn,gat}.json`. 30.000 kare, RTX 4060, paralel toplama (10 ortam), komşu sinyali olmadan PC060-K12 g1 bloğuyla (gözlem 48), maliyetsiz. Ölçüm sırasında GPU'da Steam açıktı (%33 kullanım, cs2 yok). attn ve gat JSON'larında `git_status` alanında yalnız önceki ölçümün izlenmeyen JSON'u görünür; yalnız süre ölçümüdür.

| Kol | sn / 200 bin kare | 600 bin kareye ölçeklenmiş |
|---|---|---|
| none | 172,4 | ~8,6 dk (M3a gerçek: 455–495 sn) |
| attn | 209,9 | ~10,5 dk |
| gat | 204,9 | ~10,2 dk |

30 bin kareden ölçeklemek başlangıç yükünü de çarptığı için süreyi abartıyor (M3a'da `none` gerçekte ~7,8 dk sürdü). **Toplam duvar süresi (3 tohum, sıralı):** Aşama 1 (9 koşu) ~1,4 saat. Aşama 1b en çok 9 koşu (~1,4 saat), yalnız bir kol geçmezse. Aşama 2 en çok 9 koşu (~1,4 saat), yalnız bir kol geçerse. En kötü durum ~4,2 saat; her aşama ayrı başlatılır.

## 4. Kararlar

Hepsi 2026-11-19'da kullanıcı tarafından öneri yönünde kararlaştırıldı.

| # | Karar | Sonuç |
|---|---|---|
| K-M4-1 | `gat` puan fonksiyonu | Nokta çarpımı + graf maskesi (§1) |
| K-M4-2 | M4a'da S kolları (kontrol koşulu iv) | **İki aşamalı:** Aşama 1 X kolları (9 koşu, ~1,4 sa). (i)–(iii)'ü sağlayan bir iletişim kolu varsa Aşama 2'de o kolun S'si ve N-S koşulur ve (iv) okunur. Hiçbir kol sağlamazsa S koşulmaz (M3b durma kuralı kalıbı). v0.3'te hakem-rl #1 gereği araya Aşama 1b (lr duyarlılığı) eklendi |
| K-M4-4 | δ | 0,25 (§3.4) |
| K-M4-3 | Graf kenar kuralı | `sector + mst` (§2) |

## 5. Hakem incelemesi

**hakem-rl, M4a ön kaydı v0.2 (2026-11-19, uç `88a1709`).** Paket ~1,5 bin kelime; komut koşmadı. ~39 bin token / 9 araç / ~1,3 dk. **Büyük revizyon:** iki BÜYÜK bulgu da ön kayıt metniyle ilgili; kod ya da koşu gerektirmiyor.

| # | Düzey | Bulgu | Yanıt (v0.3) |
|---|---|---|---|
| 1 | BÜYÜK | Durma kuralı ayarsız tek bir lr noktasına bağlı; "öğrenemedi" ile "lr artefaktı" ayrılamıyor | Aşama 1b: geçmeyen kol + N-X lr 2e-4'te, durmadan önce; Holm ailesi 4 (§3.2) |
| 2 | BÜYÜK | δ'nın tavanı başka bir farktan (A1 − S1) türetilmiş; δ ulaşılamaz çıkabilir | Tavan M3a A1 ile betimsel raporlanır; δ'ya ulaşılamazsa KISMİ + kullanıcı kararı açıkça yazıldı (§3.3, §3.4) |
| 3 | KÜÇÜK | (iv)'ün gürültü payı yok (eşik ≈ 1 SE) | (iv) = G ≥ δ/2 ve tek yönlü z ≥ 1,645 (bağımsızlık, tutucu) |
| 4 | KÜÇÜK | Eşit bütçe yalnız parametre; fark = iletişim + mimari | Beyan (§3.3) |
| 5 | KÜÇÜK | attn başarısızlığı σ hizasıyla karışır; durma kuralı attn'i M5'ten otomatik eliyordu | gat GEÇTİ ve attn değilse "tanımlayamadı" okuması, attn için karar kullanıcıda (§3.3) |

Hakemin soru yanıtları: (1) sızıntı kontrolü bozmaz, `none` ve iletişim kolu aynı vekili alır; tavan kolu M3a A1 olarak zaten var. (5) eleştirmen her kolda aynı; beyan yeterli.

**İkinci okuma** (SendMessage, yalnız `88a1709..fa04db2` farkı): **Kabul.** #1–#5 karşılandı; `m3a_results.json`'da `ceiling()` için gereken alanlar var. Üç yeni KÜÇÜK madde metne işlendi: 1b'nin kapıyı geçme olasılığını artırması ve (ii)/(iii)'ün düzeltilmemesi (geliştirme kapısı, tez bulgusu değil); 1b'nin yalnız aşağı yönlü lr denemesi; tohum başına G'nin betimsel raporu.

---

## Versiyon geçmişi

- **v0.3 (2026-11-19):** hakem-rl ilk okuması (Büyük revizyon) işlendi (§5): Aşama 1b lr duyarlılığı, (iv) z testi, δ ulaşılamazsa okuma, attn ↔ gat okuması, mimari ve eleştirmen beyanları, betimsel tavan. `analyze_m4a.py --self-test` 17/17. hakem-rl ikinci okuması Kabul; üç KÜÇÜK not metne işlendi, ön kayıt DONDURULDU. Koşu yok.
- **v0.2 (2026-11-19):** Kullanıcı kararları K-M4-1..4 (öneri yönünde). Koşucu `run_m4a.py` ve ön kayıtlı okuma betiği `analyze_m4a.py` (`--self-test` 11/11) eklendi. Koşu yok.
- **v0.1 (2026-11-19):** Kollar ve eşit bütçe (§1), graf kenar kuralı ölçümü (§2), M4a ön kaydı taslağı (§3). Koşu yok.
