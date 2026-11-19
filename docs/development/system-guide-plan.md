# Sistem Rehberi ve Sözlük — Plan

**Sürüm:** v0.6 · 2026-11-19 · **Durum:** **ONAYLANDI (2026-11-19, kullanıcı; R-K1..R-K7 öneri yönünde).** **R1 bitti** (§11), **R2 ve R3 kodu bitti** (§12, §13; görsel denetim ve konteyner açılışı bekliyor); sonra R4.
**Kaynak istek (kullanıcı, 2026-11-19):** "Sistemimizin bir sözlüğünü çıkartalım. Hatta arayüzde tüm sistemi anlatan bir sayfa ekleriz, bu dokümandan yola çıkarak."
**Tetikleyen sorun:** Tasarım belgeleri ve raporlar hakem için yazıldı. Kol kodları (G-X, Z-S), istatistik kısaltmaları (D, HAC, Holm) ve aşama adları (M4a, K12) tanımsız ya da dağınık tanımlı. Kullanıcı sonuçları takip edemiyor ve sistemi konuyu bilmeyen birine anlatamıyor.

---

## 0. Amaç ve kapsam

1. **Sözlük:** Sistemin bütün terimleri tek bir kaynakta toplanır. Her terimin iki katmanı vardır: **sade** (konuyu bilmeyen biri için, 1–2 cümle) ve **teknik** (geliştirici ve hakem için, koddaki tanıma bağlı).
2. **Rehber sayfası:** Panoda (`/dash/guide`) sistemi baştan sona anlatan, yalnız okunur bir sayfa. Sistem ne yapar, veri nasıl akar, hangi aşamadayız, deneyler ne sordu ve ne buldu, ne değildir. Sayfa sözlükte aranabilir.
3. **Kapsam dışı:** tez metni (tez kuralları ayrı, `docs/thesis/CLAUDE.md`), mevcut pano sayfalarının yeniden yazımı, İngilizce sürüm (şema izin verir, içerik sonra).

**Hangi işe hizmet eder:** Ürün tarafında K27 (pano) ailesinden bir madde; anlatım tarafında danışman, jüri ve seminer sunumları. Bir tez hipotezini sınamaz, bir deneyin sonucunu değiştirmez. M5 ve M8 koşularını bloklamaz. GPU gerektirmediği için GPU koşuları sürerken ayrı bir worktree'de geliştirilebilir.

## 1. Ölçülmüş kısıtlar

| Kısıt | Kanıt | Plana etkisi |
|---|---|---|
| Konteyner imajı `docs/`, `*.md` (README hariç), `scripts/`, `tests/` dizinlerini almıyor | `.dockerignore` (`docs/`, `*.md`, `!README.md`, `scripts/`, `tests/`) | Sayfa çalışma anında `docs/` altından okuyamaz. Tek kaynak imaja giren `dashboard/content/` altında durur. Ölçüm JSON'ları da imajda yok; sayılar derleme anında, kaynaklarıyla birlikte içerik dosyasına yazılır |
| Pano kodu konteynere kopyalanıyor | oturum notu: rebuild olmadan değişiklik gerçek değil | Her içerik değişikliği imaj yeniden derlemesi ister. Kabul ölçütüne "konteynerde açıldı" eklenir |
| Renk değeri yalnız token dosyasında | `dashboard/CLAUDE.md`; `tests/test_theme_contrast.py` (92 kontrol) | Sayfada hex yok. Yeni `dcc` bileşenlerinin (arama kutusu) sınıf ailesi `custom.css`'e bağlanır |
| `dbc` bileşenleri tanımadığı argümanda patlıyor | `tests/test_dash_props.py` | Yeni çağrılar bu testten geçer |
| Menü tek modelden türetiliyor | `dashboard/components/sidebar.py::NAV_GROUPS`, `topbar.py::ROUTE_INDEX`, `tests/test_topbar.py` | Yeni madde `NAV_GROUPS`'a eklenir. Kırıntı kendiliğinden gelir. `NEXT_STEP` yalnız var olan rotaya işaret eder |
| Yönlendirme tek fonksiyonda | `dashboard/app.py::display_page` | `/guide` dalı ve `register_callbacks` eklenir |
| Yeni yazma ucu RBAC ister | kök `CLAUDE.md` "Do NOT" | Sayfa yalnız okur, yeni API ucu yoktur. Erişim giriş yapmış her rol (izleyici dahil); `AUTH_ENABLED` açık kalır |
| Tez terminolojisi kuralları | `docs/thesis/CLAUDE.md` (ör. "EMA Sharpe düzeyi" denir, "diferansiyel Sharpe" denmez; "sızıntı yok" denmez) | Sözlük bu adlandırmalara uyar; çakışmada tez kuralı kazanır. Yasaklı ifadeler testle taranır |
| Kaynak bellekten yazılmaz | global ve proje kuralı | Sözlükteki atıf yalnız `docs/thesis/literature-table.md`'de bulunan "Yazar (yıl)" dizesiyle yapılır. Yeni kaynak `/kaynak-ekle` ile |
| Rakam elle kopyalanmaz | global kural | Rehberdeki her sayı, kaynak JSON + alan + commit ile üretilmiş bir anlık görüntüden gelir |
| Koşu sürerken depoya yazmak koşu kaydını kirletir | `train.json` → `git_status_porcelain` | Geliştirme, GPU koşusu varken ayrı worktree'de yapılır (M5 S-msg kodu böyle yazıldı) |

## 2. Mimari: tek kaynak, iki çıktı

```
dashboard/content/glossary.yaml   (elle yazılır: terimler)
dashboard/content/guide.yaml      (elle yazılır: rehber bölümleri, sade anlatım, deney kartlarının sayı listesi)
        │
        ▼  scripts/build_guide.py   (geliştirme makinesinde; imajda yok)
        │    - şemayı doğrular
        │    - guide.yaml'daki sayı listesini ölçüm JSON'larından okur → değer + kaynak yol + alan + commit
        │    - system-map.md "Durum tablosu"ndan aşama çizelgesini okur
        ├──► dashboard/content/snapshot.json   (üretilir; imaja girer; elle düzenlenmez)
        └──► docs/sozluk.md                    (üretilir; GitHub'da okunur; elle düzenlenmez)

dashboard/pages/guide.py  →  yalnız dashboard/content/*.yaml|json okur (çalışma anında docs/ yok)
```

- **Neden YAML:** Elle yazılacak ve PR'da okunacak bir metin. PyYAML zaten bağımlılıklarda (`requirements.txt`: `PyYAML==6.0.3`).
- **Neden üretilmiş Markdown:** Sözlük GitHub'da ve depoda okunabilsin. Ama tek kaynak YAML olsun, iki kopya ayrışmasın. `literature-table.md` ile aynı ilke: üretilen dosya elle düzenlenmez. İstenirse guard hook'una eklenir.

## 3. Terim şeması

```yaml
- id: sharpe                      # tekil, a-z0-9-, çapa olarak da kullanılır (/dash/guide#sharpe)
  terim: Sharpe oranı
  kisaltma: SR                    # isteğe bağlı
  kategori: degerlendirme         # §4'teki kategorilerden biri
  sade: >-
    Alınan her birim risk için ne kadar kazanıldığı. Yüksek olması, aynı dalgalanmayla daha çok kazanmak demek.
  teknik: >-
    Risksiz faiz (A8) üstü günlük getirinin ortalaması / standart sapması, √252 ile yıllıklandırılmış.
  kodda: [scripts/marl/analyze_m3.py::sr, evaluation/sharpe_tests.py]   # testle varlığı denetlenir
  belgede: [docs/development/phase-10-m3-design.md]                      # isteğe bağlı
  ilgili: [fazla-getiri, d-farki, hac]                                  # başka terim id'leri
  kaynak: ["Sharpe (1966)"]                                             # literature-table.md'de geçmeli
  benzetme: null                  # isteğe bağlı, sade katmanda
  uyari: null                     # isteğe bağlı: ör. "‘diferansiyel Sharpe’ denmez; EMA Sharpe düzeyi"
```

**Yazım kuralları:**
- Sade tanım en fazla 2 cümle; kısaltma ya da başka bir terim kullanıyorsa o terim `ilgili`'de olmalı.
- Teknik tanım koddaki tanımla aynı olmalı; `kodda` alanı o tanımın yeridir.
- Sözlükte sayı yoktur. Deneylerin sayıları yalnız `snapshot.json`'dan gelir.

## 4. Kapsam: terim envanteri

Sayılar kabaca tahmindir; R1'de envanter çıkarılınca kesinleşir.

| Kategori | Örnek terimler | Tahmini sayı | R1 (ilk sürüm) |
|---|---|---|---|
| Sistem ve aşamalar | Faz 1–10, Milestone 0', M1–M8, M3a/M3b/M4a/M4b, P paketi, K-kararları (K12 vb.), ön kayıt, hakem paneli, geliştirme kapısı | ~25 | evet |
| Deney kodları ve etiketler | A1/S1, A1c/S1c, N/A/G/Z × X/S kol kodları, GEÇTİ / KISMİ / GÖZLENMEDİ / PİLOT_İŞARETİ, pozitif kontrol, karıştırılmış kontrol, komşu sinyali (σ) | ~15 | evet |
| Değerlendirme ve istatistik | Sharpe, fazla getiri, risksiz faiz, D, ortak HAC, Holm, BH-FDR, tek yönlü p, güç, MDE, tohum, eğitim/doğrulama/test penceresi, 1/N, al-tut, DSR, PBO, MCS | ~25 | evet (ilk 15) |
| MARL | ajan, MAPPO, IPPO, CTDE, eleştirmen, parametre paylaşımı, iletişim kolları (none/attn/gat/self), graf (sektör + MST), α (ödül karışımı), logit sınırı, S-obs / S-msg / S-reg, β, güven geçidi | ~25 | evet |
| Tek ajan RL | politika, ödül, gözlem, eylem, bölüm, kare, PPO, A2C, TD3, SAC, HPO, walk-forward, EMA Sharpe düzeyi, Moody–Saffell | ~20 | kısmen |
| Tahmin hattı | ensemble, GBM, BiLSTM, TFT, TATS, meta-öğrenici, kalibrasyon (sıcaklık), p_up_cal, pit_v1 / pit_v2, bilgi kapısı (İ3), AUC, BSS, ECE | ~25 | kısmen (AUC, pit_v2, kapı) |
| Veri | BIST-30, evren E0/E1/E2, üyelik, yfinance, EVDS, makro, panel, eksik hücre, listeleme kilidi, fiyat marjı / satış engeli, nokta-zamanlılık (B0) | ~20 | kısmen |
| Yazılım ve ürün | çalışma alanı, roller (yönetici/kullanıcı/izleyici), pano sayfaları, golden test, model meta (K1), sürüm | ~15 | hayır (R4) |

**R1 seti (~60 terim):** Kullanıcının son raporları takip etmesini engelleyen terimler: aşamalar, deney kodları, MARL ve değerlendirmenin çekirdeği.

## 5. Rehber sayfası (`/dash/guide`)

| Bölüm | İçerik | Kaynak |
|---|---|---|
| 1. Bu sistem ne yapar? | 30 saniyelik anlatım (BIST-30, hisse başına ajan, akşam karar / sabah işlem, deneme-yanılmayla öğrenme) | `guide.yaml` |
| 2. Veri nasıl akar? | Basit akış şeması: veri → tahmin → karar veren ajanlar → değerlendirme → günlük karar. Mermaid panoda çizilmediği için token renkli HTML/CSS kutu-ok şeması kullanılır | `guide.yaml` |
| 3. Neredeyiz? | Aşama çizelgesi (bitti / sürüyor / planlı) | `snapshot.json` ← `system-map.md` durum tablosu |
| 4. Deneyler: ne sorduk, ne bulduk? | Her kapı için bir kart: sade soru, benzetme, etiket, en fazla 2 sayı (kaynak satırıyla), zorunlu "tez bulgusu değil" rozeti | `guide.yaml` + `snapshot.json` |
| 5. Bu sistem ne değildir? | Dürüst sınırlar: gerçek sinyal şu an yazı-turadan farksız, sonuçlar para kazandırma iddiası değil, 2021+ veriye dokunulmadı | `guide.yaml` |
| 6. Sözlük | Arama kutusu + kategori filtresi; her terim kartında sade tanım, açılır teknik tanım, ilgili terim bağlantıları | `glossary.yaml` |
| 7. Sık sorulanlar | "Para kazandırıyor mu?", "Neden sahte ipucu?", "Tohum neden 3?" | `guide.yaml` |

- **Menü:** "Analiz" grubunda, Dashboard'un altında "Rehber" (`bi bi-book`).
- **Üst çubuk araması (R4):** Terimler komut paletine eklenir ("Sözlük · Sharpe oranı" → `/dash/guide#sharpe`).
- **Uzun iş yok:** Sayfa yoklama yapmaz, `dcc.Interval` kullanmaz.

## 6. Doğrulama

Yeni test: `tests/test_guide.py` (ana venv, ağ yok).

| # | Kontrol |
|---|---|
| 1 | Şema: zorunlu alanlar, `id` tekil ve biçimi geçerli, kategori listede |
| 2 | `ilgili` bağlantıları var olan bir id'ye gidiyor; sade tanımda geçen her sözlük kısaltması `ilgili`'de |
| 3 | `kodda` yolları depoda var (`::ad` verilmişse dosyada geçiyor) |
| 4 | `kaynak` dizeleri `literature-table.md`'de geçiyor |
| 5 | Yasaklı ifade taraması: "sızıntı yok", "diferansiyel Sharpe" (EMA terimi için), "ilk", "literatürde yok", "anlamlı olarak daha iyi" (istatistiksel kaynak olmadan). **Mutasyon:** listeye uyan bir ifade eklenince test kırılır |
| 6 | Tazelik: `build_guide.py --check` ile `snapshot.json` ve `docs/sozluk.md` üreticinin çıktısıyla bit-eş; kaynak JSON SHA'ları uyuşuyor |
| 7 | Sayfa kuruluyor: `guide.layout()` hatasız; arama "sharpe" → Sharpe kartı görünür, "zzz" → boş durum mesajı |
| 8 | Her deney kartının sayısı `snapshot.json`'da bir kaynağa bağlı (kaynaksız sayı yok) |

Mevcut testler: `test_theme_contrast.py`, `test_dash_props.py`, `test_topbar.py`. Son adımda iki temada (açık/koyu) elle görsel denetim ve konteynerde açılış.

## 7. Bakım kuralı

Kök `CLAUDE.md`'ye, sistem haritası kuralının yanına bir kural eklenir: **yeni bir kod adı, kol kodu, etiket ya da aşama adı ekleyen PR, aynı PR'da `dashboard/content/glossary.yaml`'a terimi ekler ve `build_guide.py`'yi koşar.** `test_guide.py` [6] üretilmiş dosyalar bayatsa kırılır. Deney sonucu ekleyen PR, `guide.yaml`'daki kartı ve sayı listesini günceller.

## 8. İş paketleri

| Paket | İş | Kabul ölçütü | Tahmini süre |
|---|---|---|---|
| **R1** | Şema + `glossary.yaml` (~60 terim) + `build_guide.py` (Markdown üretimi) + `test_guide.py` [1–6] → `docs/sozluk.md`; `docs/README.md` indeksine bağlantı | Test geçiyor; kullanıcı sözlüğü okuyup son M4 raporunu takip edebiliyor | 1 oturum |
| **R2** | `dashboard/pages/guide.py`: bölüm 1, 2, 5, 6 (anlatım, akış, sınırlar, sözlük araması); menü + yönlendirme; tema ve prop testleri | `test_guide` [7] + mevcut UI testleri geçiyor; iki temada görsel denetim; konteynerde açılıyor | 1 oturum |
| **R3** | `snapshot.json`: deney kartları (bölüm 4) ve aşama çizelgesi (bölüm 3); tazelik testi [6, 8] | Kartlardaki her sayının kaynağı gösteriliyor; `system-map.md` değişince test bayatlığı yakalıyor | 0,5–1 oturum |
| **R4** | Kalan ~110 terim (tek ajan RL, tahmin, veri, yazılım); üst çubuk araması; SSS (bölüm 7); `CLAUDE.md` bakım kuralı | Envanterdeki her kategori dolu; arama terimi buluyor | 1 oturum |

Sıra R1 → R2 → R3 → R4. R1 tek başına değer üretir (okunabilir sözlük). R2–R3 sayfayı getirir.

## 9. Kararlar

**KARAR (2026-11-19, kullanıcı):** plan onaylandı; aşağıdaki yedi karar öneri yönünde kapandı. R1'den başlanır.

| # | Karar | Öneri |
|---|---|---|
| R-K1 | Sayfanın adı ve yeri | "Rehber", Analiz grubunda Dashboard'un altında |
| R-K2 | İki katman (sade + teknik) | Evet; sade katman her zaman görünür, teknik katman açılır |
| R-K3 | Dil | Türkçe; şema `en` alanına izin verir (makale için sonra) |
| R-K4 | Deney kartlarında sayı | En fazla 2 sayı + kaynak satırı; etiket ve "tez bulgusu değil" rozeti zorunlu |
| R-K5 | Erişim | Giriş yapmış her rol (izleyici dahil); anonim erişim yok |
| R-K6 | Türkçe karakter | Rehber içeriğinde tam Türkçe karakter. Panonun mevcut ASCII etiketleri ("Egitim" vb.) bu planın kapsamı dışında |
| R-K7 | Eski seminer belgeleri (`docs/thesis/seminar-basic-guide.md`, `seminar-overview.md`) | Tarihsel kalır. İçlerinden rehbere taşınacak her cümle güncel kurallara göre denetlenir (Faz 1–3 dönemi iddiaları) |

## 10. Riskler

- **Bayatlama:** Rehber sistemden geri kalabilir. Önlem: tek kaynak, üretici ve tazelik testi; bakım kuralı (§7).
- **Fazla iddia:** Sade dil, temkinli iddiayı basitleştirirken güçlendirebilir ("çalışıyor" ↔ "işaret var"). Önlem: yasaklı ifade testi, deney kartlarında etiket + rozet zorunluluğu, kart metinleri M-belgelerinin "Okuma" paragrafından türetilir.
- **Kapsam şişmesi:** ~170 terim tek seferde yazılırsa kalite düşer. Önlem: R1'de ~60 terim; kalan R4'te.
- **Konteyner farkı:** `dashboard/content/` imaja girmezse sayfa boş kalır. Önlem: R2 kabul ölçütünde konteyner açılışı; `.dockerignore` `*.md` kuralı YAML ve JSON'u etkilemez, R2'de doğrulanır.

## 11. R1 durumu (2026-11-19)

- **Sözlük:** `dashboard/content/glossary.yaml`, 73 terim, 7 kategori (yazılım ve ürün kategorisi R4'te). Üretilmiş [`docs/sozluk.md`](../sozluk.md) (`python scripts/build_guide.py`; `--check` bayatlığı yakalar).
- **Test:** `tests/test_guide.py` 6/6 ([1]–[6]; her kontrol mutasyonla birlikte). 93 kod/belge yolu ve 13 kaynak dizesi denetleniyor.
- **Plandan sapmalar (beyan):** (a) §6 [5]'teki "ilk" yasağı tek kelime değil özgünlük kalıbı olarak uygulandı (`ilk kez|defa|çalışma`, `literatürde yok|ilk`); "ilk gün" gibi olağan kullanımlar engellenmez. (b) Yasaklı ifade taraması `uyari` alanını atlar, çünkü o alan yasaklı ifadeyi "denmez" diye anmak için var. (c) Sözlükte deney sonucu (etiket, ölçülen sayı) yok; yalnız tanım ve tasarım parametresi. Sonuçlar R3'teki deney kartlarına (`snapshot.json`) kalır. (d) M5 kol kodları (G0/GL/GLS/GLM/GD) M5 mekanizma ön kaydına göre yazıldı (GLG yok).
- **Açık:** üretilmiş dosyanın elle düzenlenmesini engelleyen guard hook'u (§2, isteğe bağlı) eklenmedi; `--check` + test [6] bayatlığı yakalıyor. Bakım kuralı (§7) kök `CLAUDE.md`'ye R4'te girer.

## 12. R2 durumu (2026-11-19)

- **Sayfa:** `dashboard/pages/guide.py` → `/dash/guide`, menüde "Analiz" grubunda Dashboard'un altında "Rehber" (`bi bi-book`). Bölümler: 1 (ne yapar), 2 (veri akışı; token renkli HTML/CSS kutu-ok şeması, ≤ 820px'te oklar gizlenir), 5 (ne değildir), 6 (sözlük: arama + kategori süzgeci, sade tanım görünür, teknik ayrıntı `<details>` ile açılır). Yalnız okur; API ucu ve `dcc.Interval` yok.
- **İçerik:** `dashboard/content/guide.yaml` (bölüm 1, 2, 5; sayı yok) + `glossary.yaml`. Arama Türkçe harfe duyarsız ("ogrenme" ~ "öğrenme").
- **Stil:** `custom.css` `guide-` ailesi, yalnız token; `test_theme_contrast` [10] artık `guide-` önekini de tarıyor.
- **Test:** `test_guide` 8/8 ([7] iki kontrol: kuruluş + arama + bağlantı + yasaklı ifade mutasyonu + menü/rota); `test_theme_contrast` 92/92, `test_dash_props` 4/4, `test_topbar` 39/39.
- **Plandan sapma (beyan):** terim çapası `#terim-<id>` (planda `#<id>`), çünkü kısa terim id'leri (ör. `panel`, `kol`) sayfa bileşen id'leriyle çakışabilir. R4'teki üst çubuk araması bu çapayı kullanır. Sayfa yüklenirken URL'deki çapaya kaydırma yok (içerik Dash ile sonradan çiziliyor); R4'te istemci tarafı bir geri çağrıyla eklenir.
- **Açık (kabul ölçütünün kalanı):** iki temada (açık/koyu) görsel denetim ve konteynerde açılış. Bu oturumda Docker çalışmıyordu ve tarayıcı aracı yoktu; ölçülmedi.

## 13. R3 durumu (2026-11-19)

- **Anlık görüntü:** `dashboard/content/snapshot.json` (üretilir, elle düzenlenmez). Deney kartları `guide.yaml` → `deneyler.kartlar` içinde tanımlanır. Her kartın etiketi ve en fazla 2 sayısı üretici tarafından kaynak JSON'un belirtilen alanından okunur ve değer, kaynak yolu ve alan adıyla birlikte yazılır. Kaynak özeti SHA-256'dır; varsa ölçümü üreten kodun commit'i (JSON'daki `provenance.commit` ya da `commit`) eklenir. Sayılar Türkçe ondalık virgülle biçimlenir.
- **Kartlar (5):** P5 bilgi kapısı, M3a, M3b, M4a, M4b. Her kartta etiket, zorunlu "Tez bulgusu değil" rozeti, sade soru, okuma cümlesi (tasarım belgesindeki okuma paragrafından) ve belge yolu var. M5 kartı sonuç gelince eklenir.
- **Aşama çizelgesi:** `system-map.md` "Durum tablosu"ndan; durum kalın baş ya da ilk ayraç öncesi; sınıf anahtar sözcükle (sürüyor / bitti / planlı ya da açık). Yalnız tablonun özeti kaydedilir, haritanın başka bir satırı rehberi bayatlatmaz.
- **Test:** `test_guide` 9/9 ([6] snapshot tazeliği; [8] her sayı kaynak alanına eşit, etiket kaynaktan, rozet; mutasyon: çözülmeyen alan, 3 sayılı kart, değiştirilmiş kaynak JSON, değiştirilmiş harita satırı). Tema 92, props 4, üst çubuk 39.
- **Plandan sapmalar (beyan):** (a) Planda "değer + kaynak yol + alan + commit" yazıyor. Dosyanın son git commit'i yerine üretim commit'i ve SHA kullanıldı; kaynak ile snapshot aynı commit'e girince son git commit'i kendini bayatlatırdı. M3a JSON'u provenance taşımadığı için onun commit alanı boş. (b) P5 kartının etiketi bir kuraldan türetilir: BH reddi 0 ise "GEÇMEDİ". Kaynak alanı kartta gösterilir. (c) Haritanın M2 satırı bayattı ("sürüyor"); kanıtıyla birlikte "bitti" yapıldı (`review-panel.md`, M2 kapanış ikinci okuması Kabul). "Faz 10 M4–M8" satırı M5 sonucuyla M5 PR'ında güncellenecek.
- **Bakım kuralı:** kök `CLAUDE.md`'ye eklendi: harita tablosu ya da kart kaynağı değişince `python scripts/build_guide.py` koşulur.

## 14. R4 durumu (kısmi, 2026-11-19)

- **Yapıldı:** Sık sorulanlar (bölüm 7; `guide.yaml` → `sss`, 6 soru, terim bağlantılı, yasaklı ifade ve bağlantı denetimine dahil). Sayfa açılırken URL'deki çapaya (`#terim-<id>`, `#deney-<id>`) bir kez kaydırma (istemci tarafı geri çağrı; arama yeniden çizince kaydırmaz).
- **Kalan:** ~110 terim (tek ajan RL, tahmin, veri, yazılım ve ürün); üst çubuk aramasına sözlük terimleri. Bu ikincisi `go_to_page` geri çağrısına `url.hash` çıktısı eklemeyi ve `test_topbar` [5]'in çıktı imzasını güncellemeyi gerektirir, ayrı yapılacak.

---

## Versiyon geçmişi

- **v0.6 (2026-11-19):** R4 kısmi (§14): SSS bölümü, çapaya kaydırma. Kalan: terimler, üst çubuk araması.
- **v0.5 (2026-11-19):** R3 kodu (§13): `snapshot.json`, 5 deney kartı, aşama çizelgesi, test 9/9; bakım kuralı CLAUDE.md'de.
- **v0.4 (2026-11-19):** R2 kodu (§12): Rehber sayfası, `guide.yaml`, CSS, menü/rota, test 8/8. Görsel denetim ve konteyner bekliyor.
- **v0.3 (2026-11-19):** R1 bitti (§11): 73 terim, üretici, test 6/6, `docs/sozluk.md`. Plandan sapmalar beyan edildi.
- **v0.2 (2026-11-19):** Kullanıcı onayı: plan ve R-K1..R-K7 öneri yönünde. Sıradaki iş R1.
- **v0.1 (2026-11-19):** İlk plan (kullanıcı isteği). Ölçülmüş kısıtlar, tek kaynak mimarisi, şema, envanter, sayfa bölümleri, doğrulama, iş paketleri R1–R4, kararlar R-K1..R-K7. Kod yok.
