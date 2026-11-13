# Literatür Taraması Doğrulama Raporu

**Doğrulanan belge:** `docs/reference/Literatür Taraması Tahmin-Destekli İletişimli MARL ile BIST-30 Portföy Yönetimi (Ekim 2026).md` (claude.ai Research çıktısı, 121 kaynak)
**Tarih:** 2026-10-06
**Kapsam:** (1) Her kaynağın gerçekten var olduğunun doğrulanması, (2) özgünlük tehdidi olarak sunulan iddiaların tam metinle kontrolü, (3) eksik kalan güçlü/güncel kaynakların eklenmesi, (4) önerilerin mevcut koda etkisi.

> **2026-10-07 notu:** Tezin kaynak tabanı artık [literature-table.md](literature-table.md)'dir (169 kayıt değerlendirildi, 148 seçildi). Bu rapor 2026-10-06 doğrulamasının kaydı olarak korunuyor. İkinci turun bulguları ve düzeltmeleri §8'de. Buradaki bazı kaynaklar (ör. MACTAS, SAMP-HDRL, RL-BHRP, Rahimikia vd.) seçim kuralıyla elendi; gerekçeler tablonun Ek A'sında.

---

## 1. Sonuç

- **Uydurma kaynak yok.** Belgedeki 33 DOI'nin tamamı Crossref'te, 51 arXiv kimliğinin tamamı arXiv API'sinde bulundu. Kimliği olmayan 30+ kaynak yayıncı sayfası, kurumsal depo veya Crossref başlık aramasıyla tek tek doğrulandı. Doğrulanamayan kaynak kalmadı.
- **Belgenin kendi tespiti de doğru:** vizyon belgesindeki 4 kaynak (Preprints 202511.0447, Sci Rep "DNN+RL döviz", Decision Science Letters 2025, MDPI AI NIFTY 50) gerçekten bulunamıyor → **tezde kullanılmamalı.**
- **Belge kaynak kalitesini sistematik olarak küçümsemiş:** 9 kaynak "ön baskı / ⚠" diye işaretlenmiş ama hakemli yayınlanmış (KDD, AAAI, ICLR, CIKM, ESWA...). 15+ eksik yazar/DOI bulundu.
- **Birkaç zayıf kaynak "Oku" listesine fazla yüksek yerleştirilmiş** (PPO-HRAP, Research Square ön baskısı, SPIE bildirisi). Daha güçlü muadilleri bulundu.
- **Belgenin kaçırdığı 1 kritik kaynak:** BIST30'da DRL portföy optimizasyonu yapan hakemli bir 2026 makalesi (Beyhan vd., EPFAD). G4 iddiasını doğrudan etkiliyor.
- **Mevcut sisteme etkisi büyük:** belgenin "benim değerlendirmem" diye eklediği sızıntı uyarılarının **üçü kodda doğrulandı** (ayrıştırma, makro yayın gecikmesi, statik temel oranlar); doğrulama sırasında dördüncüsü bulundu (makro `bfill` — varsayılan hatta etkin). Ayrıntı ve hangisinin varsayılan hatta etkin olduğu Bölüm 6'da. Kod değişikliği yapılmadı; bulgular planlama için kayıt altında.

---

## 2. Doğrulama yöntemi

| Adım | Araç | Kapsam | Sonuç |
|---|---|---|---|
| DOI çözümleme | Crossref REST API | 33 DOI | 33/33 gerçek; başlık, yazar, cilt/sayfa karşılaştırıldı |
| arXiv kimlikleri | arXiv export API | 51 kimlik | 51/51 gerçek; yayın yeri (journal_ref / comment) okundu |
| Diğer URL'ler | Sayfa başlığı + `citation_*` meta | 58 URL | Erişilebilenlerin hepsi künyeyle uyumlu |
| Bot korumalı / kimliksiz | Crossref başlık araması, web arama, PDF metni | ~30 kaynak | Hepsi bulundu |
| Kritik iddialar | Tam metin (Europe PMC XML, arXiv HTML, PDF) | 10 iddia | Bölüm 4 |

---

## 3. Künye düzeltmeleri

### 3.1 Belgede "ön baskı / ⚠" denmiş ama hakemli yayınlanmış (kalite yükseltmesi)

| # | Kaynak | Belgede | Doğrusu |
|---|---|---|---|
| #7 | Fang vd. 2023, Intention-Aware Communication | Ö, yayın yeri ⚠ | **KDD 2023**, DOI 10.1145/3580305.3599856 — G1 tehdidi daha da güçlü |
| #2 | Chen, Li, Wang 2025, MARS | Ö | **AAAI 2026 Main Track** |
| #33 | DeepClair 2024 | Ö, yazar ⚠ | **CIKM 2024**; Choi, Kim, Gim, Lee, Kang; DOI 10.1145/3627673.3680008 |
| #34 | RTS-PnO 2025 | Ö, yazar ⚠ | **KDD 2025 ADS**; Lyu vd.; DOI 10.1145/3711896.3737268 |
| #25 | ExpoComm 2025 | Ö, yazar ⚠ | **ICLR 2025**; Li, Wang, Bai, Zhang |
| #42 | RVFL walk-forward 2021 | Ö, yazar ⚠ | **ESWA 2022**; Gao, Du, Suganthan vd.; DOI 10.1016/j.eswa.2022.117784 |
| #39 | ML Pitfalls rehberi | Ö, yazar ⚠ | **Patterns 2024**; Lones; DOI 10.1016/j.patter.2024.101046 |
| #38 | Forecast Evaluation Pitfalls | Ö, dergi ⚠ | **Data Mining and Knowledge Discovery 37:788–832 (2023)**; Hewamalage, Ackermann, Bergmeir; DOI 10.1007/s10618-022-00894-5 |
| #41 | EMD yanlılığı 2017 | D, yazar ve yayın yeri ⚠ | **Expert Systems with Applications 82:19–26 (2017)**; Carnelossi Furlaneto, Oliveira, Menotti, Cavalcanti; DOI 10.1016/j.eswa.2017.03.053 — Makale 2'nin en güçlü sızıntı kaynağı |
| #23 | Goal-Oriented Communication overview | Ö | **IEEE Open J. Comm. Society 7 (2026)**; Charalambous vd. |
| #71 | Benchmarking Robustness of DRL OPM | K ⚠, yazar ⚠ | **IEEE INISTA 2023**; Velay, Doan, Rimmel, Popineau, Daniel; DOI 10.1109/INISTA59065.2023.10310402 |
| J | HedgeAgents / QuantAgents / FinAgent | ⚠ | **WWW 2025** (arXiv 2502.13165) / **EMNLP 2025** (2510.04643) / **KDD 2024** (DOI 10.1145/3637528.3671801) |

### 3.2 Belgede yanlış ya da eksik künye

| Kaynak | Sorun | Doğrusu |
|---|---|---|
| Sattar vd. 2025 (BibTeX) | İkinci yazar "Asma Sarwar" yazılmış; cilt/sayfa yok | **Amna** Sarwar; IEEE Access **13:42813–42835** |
| Ansari vd. 2024 | "4 mü 6 mı yazar" belirsiz bırakılmış | Crossref: **6 yazar** (Ansari, Gillani, Bukhari, Lee, Maqsood, Rho) |
| Ayari 2025 | DOI ⚠ | Quantitative Finance **26:119–136**, DOI **10.1080/14697688.2025.2592822** |
| Bai vd. 2025 | Dergi DOI'si ⚠ | Annu. Rev. Stat. Appl. **12:209–232**, DOI **10.1146/annurev-statistics-112723-034423** |
| Macrì & Lillo | Belgede bir yerde 2024, bir yerde 2025 | Crossref: Applied Mathematical Finance 31(5), **2024** |
| Neurocomputing 2024 "Teammate Modeling" | Yazar/DOI ⚠ | Jiang, Zhang, Liu, Xu, Zhang; 586:127638; DOI **10.1016/j.neucom.2024.127638** |
| #17 ADMAC | Yazar ⚠ | Yu, Qiu, Yao, Shen, Zhang, Wang (Tsinghua); arXiv **2312.11545** |
| #19 IMMAC | Yazar ⚠ | Sun, Wu, Wang, Hu, Yang, Cong (ISCAS), AAMAS 2021 genişletilmiş özet |
| #28 MAGIC | Yazar ⚠ | Niu, Paleja, Gombolay, AAMAS 2021, ss. 964–973 |
| #5 IPS (MMEP) | Yazar ⚠ | Khemlichi, Khamlichi, Elhaj Ben Ali |
| #11 Factor-MCLS | Yazar ⚠ | Sun, Stefanidis, Jiang, Su — Sun vd. (2025, CAIS) ve SAMP-HDRL ile **aynı grup** |
| #32 StockFormer | DOI yok | IJCAI 2023, ss. 4766–4774, DOI 10.24963/ijcai.2023/530 |
| #55 MTX | Yazar ⚠ | Hsieh, Lai, Yuan; IJFS 14:67 (2026) |
| #59 Smart Tangency | Yazar ⚠ | Yu, Chang; IJFS 13:227 (2025) |
| #60 Zhang, Zohren, Roberts | Dergi ⚠ | J. Financial Data Science 2(2):25–40 (2020), DOI 10.3905/jfds.2020.1.030 |
| #69 Tohum duyarlılığı ön baskısı | Yazar ⚠ | Djomo Goïmba (tek yazar, aivancity), Research Square, Mayıs 2026 |
| #77 TJEECS 2022 | Yazar ⚠ | Altuner, Kilimci |
| #78 ASYU 2025 | Yazar ⚠ | Akyapak, Tunç |
| #79 Borsa Istanbul Review 2026 | Yazar ⚠ | Karacam, Ulukus |
| #80 Erdağ vd. 2026 | Sayfa yok | Review of Behavioral Finance 18:546–566, DOI 10.1108/rbf-12-2025-0531 |
| #13 HAELT (V13) | Yazar ⚠ | Bui (tek yazar) — zaten kapsam dışı |
| V9 MACTAS | v3'te başlık değişti dendi | v4 başlığı: *Communication-Enhanced Tutoring for Efficient Decentralized MARL*, AAAI 2027'ye gönderilmiş; vizyon belgesindeki "MACTAS" adı artık güncel değil |

### 3.3 Çıkarılması / geri çekilmesi gereken kaynaklar

| Kaynak | Neden |
|---|---|
| #27 DCMAC (arXiv 2409.07127) | arXiv yorumunda yazarların kendisi: *"Paper has errors and needs to be revised"* → **çıkar** |
| #58 PPO-HRAP (arXiv 2610.01325) | 1 Ekim 2026 tarihli 8 sayfalık ön baskı; tek varlık (SPY); "ödülünüzün doğrudan öncülü" iddiası abartılı. **Okuma listesinden çıkar**, "Atıf yeter"e indir |
| #69 Research Square tohum çalışması | Tek yazarlı, hakemsiz. Tohum argümanını taşıyamaz; argümanı **Agarwal vd. (2021) + Henderson vd. (2018)** taşımalı |
| #40 Zhang D. vd. 2025 (SPIE, OIOE2024) | Düşük kademeli bildiri. Makale 2'nin ana sorusunu buna dayandırma; yerine **Yang vd. 2024 (Sci Rep)** ve **Carnelossi Furlaneto vd. 2017 (ESWA)** |
| #54 Ilieva 2026 | **Lisans** tezi (TU Delft repository'de "Bachelor") → yalnızca arka plan |
| V7, V14, V16, V17 | Var olmayan kaynaklar → vizyon belgesinden de silinmeli |

---

## 4. Kritik iddiaların tam metin kontrolü

| İddia (belgede) | Kontrol | Hüküm |
|---|---|---|
| **Zhang (2025, Sci Rep):** güven = son tahmin isabetinin EMA'sı, portföy önerilerinin konsensüsünde kullanılıyor | Europe PMC tam metin: *"confidence level calculated from recent prediction accuracy using exponential moving average"* → konsensüs ağırlığı | ✅ Doğru. Mesaj dikkatini modüle etmiyor → G2 **kısmen** açık |
| Zhang (2025): tohum yok, MAPPO karşılaştırması yok | Tam metinde "seed" 0, "MAPPO" 0 geçiyor | ✅ Doğru |
| Zhang (2025): 17. kaynak yanlış yazarlı | Kaynakça: *"Yang, K., Chen, L. & Wu, X. ... Complex Intell. Syst. 11(3), 2847–2865 (2024)"*; gerçek makale Sun, Xi, Stefanidis, Jiang, Su (2025) | ✅ Doğru. **Hakemli bir dergide hatalı/uydurma atıf** — eleştirirken bunu nesnel dille belirtin |
| **TATS = Kazemdehbashi (2026)** | arXiv HTML: ikili trend sınıflandırıcısı (LR, NB, KNN, SVM, DT, RF, **XGBoost** denenmiş) + LSTM/Bi-LSTM düzeltmesi | ✅ Doğru. Ek bulgu: `prediction/tats.py` repoya **2026-03-27**'de girmiş; makale **2026-01-19**'da yayımlanmış. Yöntem bu makaleden türetilmiş görünüyor → **atıf zorunlu**, kendi katkı olarak sunulamaz |
| Fang vd. (2023): finansta CommNet/TarMAC/iletişimsiz karşılaştırması | Özet: multi-order execution, öğrenilmiş çok turlu niyet iletişimi; KDD 2023 | ✅ Doğru; yayın yeri belgede eksik (KDD = A*) |
| ADMAC: alıcı tarafında mesaj güvenilirliği | AAAI 2024 kaydı: *"estimates the reliability of received messages and adjusts their impact"* | ✅ Doğru; amaç düşmanca saldırıya karşı savunma |
| IMMAC: belirsizlik/merak tabanlı dikkat | AAMAS 2021 PDF: "Communicate what surprises you"; içsel değere dayalı dikkat + geçit | ✅ Doğru |
| DebUnc: belirsizliğe göre dikkat ölçekleme | Özet: *"confidence conveyed through a modified attention mechanism that adjusts token weights"* | ✅ Doğru; LLM tartışması, RL değil |
| Epistemik sürüleşme (2605.06988): ters-entropi ağırlıklı birleştirme sürüleşmeye yol açabiliyor | Tam metin: *"weights each by its inverse uncertainty"* protokolleri (C2/C3) test ediliyor; sürüleşme ve **entropi geçitlemesi** ana konu | ⚠️ Destekleniyor ama bağlam farklı: RL değil, Bayesçi inanç füzyonlu dağıtık arama. "Analoji" olarak atıf yapın |
| #72 Nawathe vd. (2024): "%1 maliyetle eşit ağırlıklı Al-Tut en iyi" → DRL 1/N'yi geçemiyor | arXiv HTML: Equal Buy-and-Hold yalnızca **kıyas stratejileri arasında** en iyi; bazı ajan yapılandırmaları (ör. CNN politikası) S&P endeksini geçiyor; özet ajanın kıyasları geçebildiğini söylüyor | ❌ **Yanlış okuma** — "DRL basit baseline'ı geçemez" argümanına kaynak olarak kullanılmamalı |
| #70 Aarnio (2025): 1/N'ye karşı anlamlı üstünlük yok | Tez künyesi doğrulandı (Turku YL, 2025-09-18; PPO/A2C/DDPG vs 1/N, 10 ABD hissesi); **sonuç cümlesine erişilemedi** (PDF bot korumalı) | ⚠️ Sonuç doğrulanmadı — atıf yapmadan önce tam metinden teyit edin. 1/N argümanı için klasik kaynak: DeMiguel, Garlappi & Uppal (RFS 2009, DOI 10.1093/rfs/hhm075) |
| Shi & Hollifield (2025): iletişim değeri piyasaya bağlı | Özet doğruluyor; 5 LLM ajanı, 450 deney | ✅ Doğru; "NeurIPS 2025 çalıştayı" bilgisi arXiv'de yok → doğrulanmadı |
| MPLS: üst ajan sektör dağıtımı | IIETA sayfası: *"High-Level Agent (HLA) for sector allocation"* + GNN + PPO füzyon | ✅ Doğru; yazarlar sayfadan çıkarılamadı |
| tez-final.pdf: %46,44 / %146,44 tutarsızlığı | Tablo 46.44, metin "%146.44" | ✅ Doğru |
| tez-final.pdf: "b seviyesine" kalıntısı | Metinde aynen var | ✅ Doğru |
| tez-final.pdf: RMS = Root Mean Square yorumu | Metin: *"Root Mean Square - tabanlı normalizasyon"* | ✅ Doğru, düzeltilmeli |
| tez-final.pdf: 5 Optuna denemesi | *"Her algoritma için 5 farklı deneme (trial)"* | ✅ Doğru |
| tez-final.pdf: "8447.2" işlem sayısı | Tabloda **844**; "7.2." sonraki bölüm başlığı. PDF metin çıkarımının yarattığı yanılsama | ❌ **Yanlış iddia** — tezde böyle bir hata yok |
| tez-final.pdf: Faz 1'de yıllık getiri < kümülatif | A2C 28.97 → 24.10; test dönemi ~11 ay, yıllık > kümülatif olmalı | ✅ Doğru (Al-ve-Tut 46.44 → 41.47 de aynı sorun) |

---

## 5. Eklenecek yeni kaynaklar (hepsi doğrulandı)

| Öncelik | Künye | Neden önemli | Bölüm |
|---|---|---|---|
| **Zorunlu** | Beyhan, H., Ergin, E., Eren, B. S. (2026). Dynamic Portfolio Optimization with Deep Reinforcement Learning: Evidence from Borsa Istanbul. *Ekonomi Politika ve Finans Araştırmaları Dergisi* 11(1):106–. DOI 10.30784/epfad.1811319 | BIST30 + DDPG; Markowitz, eşit ağırlık ve BIST30 endeksine karşı. Belge kaçırmış. **G4'ün DRL kısmı için en güncel hakemli öncül** (Candar 2022 bir tezdi) | 2.13, G4 |
| **Zorunlu** | Yang, X., Li, J., Jiang, X. (2024). Research on information leakage in time series prediction based on empirical mode decomposition. *Scientific Reports* 14. DOI 10.1038/s41598-024-80018-9 (35 atıf) | Ayrıştırma sızıntısını gösteriyor ve **3 sızıntısız strateji** öneriyor (SW-EMD, STMP-EMD, MTMP-EMD). SPIE bildirisinin yerine | 2.8, Makale 2 |
| **Zorunlu** | Zhang, Peng, Song (2024). Realized Volatility Forecasting for Stocks and Futures Indices with Rolling CEEMDAN and Machine Learning Models. *Computational Economics* 66:1215–1268. DOI 10.1007/s10614-024-10732-z | Finans verisinde **kayan pencereli CEEMDAN** — bizim ICEEMDAN düzeltmesinin doğrudan yöntem kaynağı | 2.8, Makale 2 |
| Yüksek | Kim, S.-H., Lee, K.-H. (2025). Multi-Asset Multi-Agent Reinforcement Learning for Portfolio Management (MAMA). *IEEE Access* 13:194456–194474. DOI 10.1109/ACCESS.2025.3632201 | Varlık sınıfı başına ajan + GNN tabanlı durum temsili + faiz oranı + kavram kayması. Ansari ile aynı dergide. **G3 için öncül** (grup tabanlı ajan granülaritesi) | 2.4, G3 |
| Yüksek | Rout, A. K., Acharya, U. D., Aithal, P. K., Sharma, A. (2026). Systematic review of reinforcement learning for automated equity portfolio management from single agent to multi agent systems. *Discover Computing* 29. DOI 10.1007/s10791-026-10336-1 | PRISMA tabanlı, 156 hakemli makale, 2018–2025; **tek ajandan çok ajana** evrim — tezin anlatısıyla birebir | 2.2, 2.4 |
| Yüksek | Carnelossi Furlaneto vd. (2017), ESWA 82:19–26 (belgede ⚠ idi) | Finans verisinde EEMD sızıntısını gösteren Q1 dergi makalesi | 2.8, Makale 2 |
| Yüksek | Hewamalage, Ackermann, Bergmeir (2023), DMKD 37:788–832 (belgede ön baskı idi) | Sızıntı + tahmin değerlendirme rehberi, hakemli sürüm | 2.8, 2.11 |
| Orta | Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018). Deep Reinforcement Learning that Matters. AAAI 2018 (arXiv 1709.06560) | Tohum varyansı argümanının klasik kaynağı; Research Square ön baskısının yerine | 2.11 |
| Opsiyonel | Şimşek, Doğan, Taşkıran (2026). Analysis of information leakage in decomposition-based GNSS coordinate time series prediction. *Advances in Space Research* 78:9058–9074 | Türk yazarlı güncel sızıntı analizi (finans dışı) | 2.8 |

> Bölüm 5'teki tüm satırlar Crossref, arXiv API veya yayıncı sayfasıyla doğrulandı.

---

## 6. Mevcut sisteme katkı: kodda doğrulanan bulgular

Belgenin 2.3 ve 7. bölümdeki önerileri koda karşı kontrol edildi (2026-10-06, `main` @ b82fedb).

> **Kapsam notu.** Bu bölüm bir **bulgu kaydıdır, iş planı değildir.** Kod değişikliği yapılmadı ve henüz planlanmadı; düzeltmelerin sırası ve zamanı ayrı bir planlama turunda belirlenecek. "Önerilen yön" sütunu yalnızca literatürün işaret ettiği çözümü kaydeder.
>
> **Düzeltme (aynı gün, ikinci doğrulama turu):** İlk sürümde "istatistiksel test yok" ve "tek tohum" yazılmıştı; ikisi de eksik/yanlıştı. Aşağıdaki P1 tablosu düzeltilmiş hâlidir. Ayrıca varsayılan hatta etkin olan sızıntı ile yalnızca opt-in gruplarda oluşan sızıntı artık ayrı gösteriliyor.

### P0 — Sızıntı (Makale 2 ve tez sonuçları için yayın engelleyici)

| # | Bulgu | Varsayılan hatta etkin mi? | Kanıt | Önerilen yön (literatür) |
|---|---|---|---|---|
| 1 | **ICEEMDAN serinin tamamına uygulanıyor; hedef de filtrelenmiş fiyattan türetiliyor** | Hayır — `iceemdan` grubu `default: False`. Ama Makale 2 tam olarak bu grubu kullanacak | [feature_engineer.py:86-88](../../prediction/feature_engineer.py#L86-L88) `data['close']` filtreli seriyle değiştiriliyor; [feature_engineer.py:574](../../prediction/feature_engineer.py#L574) hedef bu `close`'tan; [feature_groups.py:72](../../prediction/feature_groups.py#L72) | Kayan pencereli ayrıştırma (Yang vd. 2024 SW-EMD; Zhang, Peng & Song 2024 rolling CEEMDAN); hedef **ham** fiyattan |
| 2 | **Makro seride `bfill`** — serinin başındaki boşluk gelecekteki değerle dolduruluyor | **Evet** (`macro_tr`, `macro_global` varsayılan açık). Yerel önbellekte (2026-10-06) `oil_wti`, `gold_vix`, `silver`, `us_real_rate` sütunlarının ilk gerçek değeri **2026-04-23**; bu değer 2018-01-01'e kadar geriye kopyalanmış | [macro_fetcher.py:275](../../data/macro_fetcher.py#L275), [macro_fetcher.py:454](../../data/macro_fetcher.py#L454) | `bfill` yerine başlangıç boşluğunu düşürme / eksik bayrağı |
| 3 | **TÜFE/ÜFE yayın gecikmesi modellenmiyor** — referans aya tarihli değer yalnızca 1 iş günü kaydırılıyor; resmi açıklama ertesi ayın başında (~1 ay ileriye bakma) | **Gizli**: yerel önbellekte EVDS verisi yok — `policy_rate` fallback sabiti 50.0, `cpi_inflation`/`ppi_inflation` sabit 0.0 (değişim sayısı 0). EVDS bağlandığı anda sızıntı devreye girer | [macro_fetcher.py:200](../../data/macro_fetcher.py#L200) `resample('D').ffill()`; [feature_engineer.py:459-463](../../prediction/feature_engineer.py#L459-L463) yalnızca `shift(1)`; [macro_fetcher.py:245](../../data/macro_fetcher.py#L245) fallback | Değeri açıklanma tarihine hizalama |
| 4 | **Temel oranlar bugünkü tek bir anlık değer** — `ticker.info` skaler olarak tüm geçmişe yayılıyor (ileriye bakma + sembol kimliği gibi davranan sabit) | Hayır — `fundamental` grubu `default: False` | [fundamental_fetcher.py:71](../../data/fundamental_fetcher.py#L71); [feature_engineer.py:484-503](../../prediction/feature_engineer.py#L484-L503) | Point-in-time çeyreklik veri ya da grubu kapalı tutup sınırlılık olarak beyan |

**Ek veri kalitesi bulgusu (P0 #3 ile bağlantılı):** Yerel makro önbellekte TCMB politika faizi ve enflasyon fiilen yok. Bu, "Türkiye'ye özgü makro dinamikleri modelliyoruz" iddiasını o önbellekle üretilen sonuçlar için geçersiz kılar. Eğitim manifestindeki `data_quality` bayrağı (`fallback_constant_50.0`) bunu yakalamak için var; sonuç raporlarında okunmalı.

### P1 — Değerlendirme protokolü (tez + makaleler)

| # | Bulgu | Kanıt | Önerilen yön (literatür) |
|---|---|---|---|
| 5 | **Risksiz faiz üç yerde üç farklı değer** | [evaluator.py:450](../../prediction/evaluator.py#L450) `0.0`; [reward_functions.py:37](../../env/reward_functions.py#L37) `0.15` ("TCMB ~%15" yorumu); [model_analysis.py:40](../../app/services/model_analysis.py#L40) `0.02` | Tek kaynak: politika faizi / TLREF zaman serisi; nominal + TÜFE-reel + USD bazında raporlama |
| 6 | **Hayatta kalma yanlılığı** — sembol evreni bugünkü BIST-30 | [bist30_symbols.py](../../data/bist30_symbols.py) sabit liste | Dönemsel endeks bileşenleri ya da sınırlılık olarak açık beyan |
| 7 | **RL eğitiminde tohum yok; tahmin hattında tek tohum** — pano eğitim rotası SB3 modellerini `seed` vermeden kuruyor (koşum tekrar üretilemez, tohum kaydedilmez); tahmin hattı `GLOBAL_SEED = 42` | [trading.py:1267](../../app/api/routes/trading.py#L1267) ve sonraki `model_class(...)` çağrıları; [hyperopt.py:150](../../app/api/routes/hyperopt.py#L150) `seed=None`; [trainer.py:251](../../prediction/trainer.py#L251) | Her koşumda tohumu kaydetme; ön eleme 3–5, nihai ≥10 tohum; IQM + bootstrap CI (Agarwal vd. 2021; Henderson vd. 2018) |
| 8 | **İstatistiksel testler var ama bağlı değil ve zayıf biçimde uygulanmış** — DM testi HAC/Newey-West varyansı ve Harvey-Leybourne-Newbold düzeltmesi olmadan; t-testi eşleştirilmemiş (`ttest_ind`) ve otokorelasyonu yok sayıyor; iki fonksiyon da hiçbir rotadan/scriptten çağrılmıyor. Sharpe farkı testi (Ledoit-Wolf), MCS ve çoklu karşılaştırma düzeltmesi yok | [evaluator.py:268](../../prediction/evaluator.py#L268); [model_analysis.py:204-223](../../app/services/model_analysis.py#L204-L223) | DM → HAC'li; Sharpe farkı → Ledoit-Wolf; çok yapılandırma → MCS; BH-FDR |
| 9 | **Eksik basit baseline'lar** — mevcut: Al-Tut (başlangıçta eşit ağırlık, yeniden dengeleme yok) ve XU030 (indirilemezse eşit ağırlıklı sentetik endeks) | [benchmark_strategies.py:13](../../scripts/benchmarking/benchmark_strategies.py#L13), [benchmark_strategies.py:154](../../scripts/benchmarking/benchmark_strategies.py#L154) | Periyodik yeniden dengelenen 1/N, min-varyans (Ledoit-Wolf daraltmalı), HRP; Beyhan vd. (2026) ve Candar (2022) tipi DDPG/PPO |

### P2 — Küçük / zaten karşılanan

| # | Bulgu | Durum |
|---|---|---|
| 10 | PyEMD `noise_seed()` sabitlenmeli | **Zaten yapılıyor** — [iceemdan_processor.py:75](../../prediction/iceemdan_processor.py#L75) `noise_seed(42)` |
| 11 | İşlem maliyeti 20–30 bps gidiş-dönüş | [trading_env.py:43](../../env/trading_env.py#L43) `0.001` (tek yön) = 20 bps → uyumlu. Eksik: 0/10/25/50 bps duyarlılık analizi |
| 12 | TATS atfı | [tats.py](../../prediction/tats.py) docstring'ine ve tez/Makale 2 metnine Kazemdehbashi (2026) atfı eklenmeli |
| 13 | ICEEMDAN adlandırması | PyEMD belgesine göre `CEEMDAN` sınıfı Colominas (2014) iyileştirmelerini içeriyor; makalede "Colominas vd. (2014) iyileştirmelerini içeren PyEMD CEEMDAN" ifadesi daha güvenli |

### Tez tasarımına etkiler (MARL, henüz kod yok)

- **G2'yi daralt:** "Harici, kalibre bir topluluk tahmincisinin güveni (a) gönderen tarafında dikkat logitine log-güven önseli, (b) alıcı tarafında öz-güven geçidi olarak." Ablation kolları: güvensiz dikkat / ADMAC tipi öğrenilmiş güvenilirlik / DebUnc tipi ters belirsizlik ölçekleme / karıştırılmış (shuffled) güven.
- **Kalibrasyon zorunlu:** Mekanizmanın değeri güvenin kalibre olmasına bağlı → ECE ve güvenilirlik diyagramı. Topluluk şu an güveni yön başlığı + model uyumundan türetiyor; kalibrasyon ölçülmüyor.
- **Sürüleşme riski:** Güven ağırlığına kırpma veya entropi geçidi (Farr vd. 2026, analoji olarak).
- **Zorunlu MARL baseline'ları:** IPPO, MAPPO ve **MPLS tipi hiyerarşik sektör dağıtıcı** (G3).
- **G4 yeni ifadesi:** "BIST'te DRL portföy çalışmaları mevcuttur (Candar 2022; Beyhan vd. 2026); erişilebilen literatürde MARL uygulaması bulunamamıştır." YÖK Tez Merkezi taraması elle tamamlanmalı.
- **Rejim etiketi:** Kural tabanlı ve önceden tanımlı (XU100 200g trend + gerçekleşmiş oynaklık); model girdisi olarak yalnızca HMM'nin filtrelenmiş olasılıkları. Viterbi ve düzleştirilmiş olasılıklar geleceği kullanır.

---

## 7. Belgenin genel değerlendirmesi

| Boyut | Not |
|---|---|
| Kaynakların gerçekliği | **Çok iyi** — uydurma yok; önceki taramanın uydurmalarını da yakaladı |
| Künye doğruluğu | Orta — 25+ eksik/yanlış alan; birkaç BibTeX hatası (Sattar) |
| Kaynak kalitesi sınıflaması | **Zayıf** — 9+ hakemli yayını ön baskı saymış, 3 zayıf ön baskıyı "Oku" listesine almış |
| Kapsam | İyi; ancak en güncel BIST DRL makalesini (2026) ve en güçlü sızıntı dergi makalelerini kaçırmış |
| Özgünlük analizi | **Çok değerli** — Zhang 2025, Fang 2023, ADMAC, TATS tespitleri tam metinle doğrulandı |
| Metodoloji önerileri | **Çok değerli** — sızıntı uyarılarının üçü kodda gerçek çıktı |
| tez-final.pdf eleştirisi | 8/9 doğru; "8447.2" iddiası PDF çıkarım yanılsaması |

**Tavsiye edilen kullanım:** Belgeyi literatür matrisi olarak kullanın; ama künye ve kalite sütunlarını bu raporun 3. bölümüyle düzeltin, 3.3'teki kaynakları çıkarın, 5. bölümdekileri ekleyin. Nihai sayımı Zotero'da yapın.

---

## 8. İkinci tur (2026-10-07): seçilmiş kaynakça

Bu turun çıktısı [literature-table.md](literature-table.md). Kısaca:

### 8.1 Yöntem

- **İleri atıf taraması:** 25 çekirdek makaleye 2025–2026'da atıf yapan 247 çalışma tarandı (OpenAlex `cites:`).
- **Kapsam denetimi:** Kodda kullanılan her yöntemin ve tezdeki her iddianın kaynağı arandı.
- **Seçim kuralı:** SCIE/SSCI dergi ya da CORE A*/A konferans; Semantic Scholar atıf eşiği. Kural, zorunlu atıf ve yöntem kaynaklarını kademeden bağımsız tutar.
- **İndeks kontrolü:** Clarivate MJL'de elle kontrol + Wikidata.

Sonuç: 169 kayıt değerlendirildi, 148 seçildi, 21 elendi. Kapsam matrisinde K1/K2 kaynağı olmayan bileşen kalmadı.

### 8.2 Bu raporun önceki bölümlerine düzeltmeler

| Bölüm | Bu raporda | Doğrusu |
|---|---|---|
| §5, §6 (G4) | Candar (2022) yalnızca İTÜ YL tezi | Ayrıca **Candar & Üstündağ (2022)**, INFUS 2022, LNNS 505, DOI 10.1007/978-3-031-09176-6_16 — birincil atıf |
| §2.2 / §4 (DebUnc) | "Ön baskı" | **Findings of EMNLP 2025**, DOI 10.18653/v1/2025.findings-emnlp.1265 |
| §3.2 (Sattar vd. 2025) | Yalnızca BibTeX künye düzeltmesi | Ansari ekibinin devam çalışması (*IEEE Access*, SCIE). Başlıktaki **RMS = Reports + Max drawdown + Sentiment**; §4'teki tez-final.pdf "Root Mean Square" bulgusunun nedeni bu |
| §5 (Yang vd. 2024) | "Ayrıştırma sızıntısını gösteriyor" | Veri **su kalitesi** — yöntem kaynağı, finans kanıtı değil. Finans kanıtı Carnelossi Furlaneto (2017) ve **Xiong vd. (2025, *Applied Intelligence*)** |
| §3 (StockMARL) | EMSS 2025 | *Procedia Computer Science*, DOI 10.1016/j.procs.2025.12.091 |
| §3 (CommFormer) | Yalnızca ICLR 2024 | Dergi sürümü *IEEE TKDE* 2026 |
| §4 (G2) | DebUnc "kavramsal olarak en yakın" | **Matematiksel olarak eşdeğer:** softmax sonrası ağırlığı cⱼ ile çarpıp yeniden normalize etmek, logite log cⱼ eklemekle aynıdır |
| §4 (ADMAC) | Alıcı tarafında güvenilirlik | Kestiricinin etiketleri **ayrık eylem** varsayıyor; sürekli eylemde uyarlama gerekir |

### 8.3 Yeni yakın çalışmalar (2025–2026)

| Çalışma | Neden önemli | Durum |
|---|---|---|
| Hung & Huang (2026), LNCS (IEA/AIE) | Portföyde öz-dikkatli MARL — G1 | Tam metin okunmadı |
| Yuan vd. (2026), *Eng. Appl. of AI* (SCIE) | Çok-ajanlı DRL hisse alım-satım | Tam metin okunmadı |
| Lin & Zeng (2025), CAROC, *IJGS* (SCIE) | Akran güvenilirliğiyle seçici iletişim — G2 | Özet okundu |
| Öztürk (2026), SAUCIS (TR Dizin) | Güvenle ağırlıklı DRL topluluğu — G2 ifadesi, Türk jürisi | Özet okundu |
| Xiong vd. (2025), *Applied Intelligence* (SCIE) | Hisse tahmininde tek seferlik gürültü giderme sızıntısı — Makale 2 | Tam metin okunmadı |
| Taşkıran (2026), *Borsa Istanbul Review* (SSCI) | Türkiye'de enflasyon sürprizi — S3 | Başlık |
| Zhu, Dastani & Wang (2024), *JAAMAS* (SCIE) | İletişimli MARL derlemesi (9 boyut) — G1 konumlandırması | Tam metin |
| Pippas vd. (2025), *ACM Computing Surveys* (SCIE) | Nicel finansta RL derlemesi (167 yayın) | Özet |

### 8.4 Hâlâ açık

- §8.3'teki dört tam metin (Hung & Huang, Yuan vd., CAROC, Xiong vd.).
- YÖK Tez Merkezi taraması (G4).
- Aarnio tezinin sonucunun teyidi (§4) — 1/N argümanı DeMiguel vd. (2009) üzerine kurulduğu için artık kritik değil.
- Seçilmiş kaynaklarda 11 derginin MJL teyidi (literature-table §4.2; 2026-11-13'te BMC Bioinformatics eklendi).
- R1 seçim protokolü kaynaklarının üç tam metni (§8.5).

### 8.5 Faz 9 / R1 seçim protokolü kaynakları (2026-11-13)

Künyeler bellekten değil birincil kayıttan doğrulandı. Rol gerekçeleri yalnızca **özetten**; tam metin okunmadı.

| Anahtar | Doğrulama | Seçim kuralı | Dayandırdığı karar |
|---|---|---|---|
| Varma2006_CVBias | Crossref `api.crossref.org/works/10.1186/1471-2105-7-91` — *BMC Bioinformatics* 7:91 (2006) | SCIE dergi (MJL teyidi bekliyor, §4.2) · Y | Seçimde kullanılan tahmin seçilen modelin hatası için yanlı → test seçime katılmaz |
| Cawley2010_SelectionBias | `jmlr.org/papers/v11/cawley10a.html` — *JMLR* 11:2079–2107 (2010) | SCIE dergi [MJL] · Y | Model seçim ölçütüne aşırı uyum ve değerlendirmede seçim yanlılığı |
| Eimer2023_RLHPO | `proceedings.mlr.press/v202/eimer23a.html` — ICML 2023, PMLR 202:9104–9149; arXiv API 2306.01324 | CORE A* konf. · Y | RL HPO'da ayar ve sınama tohumlarının ayrılması (`hpo_seed_reused`) |

**Hat notu:** `resolve.py` her kaydı yeniden çözüyor. Toptan çözüm 2026-11-13'te diğer kayıtların atıf sayılarını değiştiriyor ve önceden hatalı 4 kaydı (GAT, IPPO, PPO, Transformer) dolduruyordu. Bu yüzden `resolved.json` ve `s2.json`'a yalnızca bu üç kayıt birleştirildi; geri kalanı 2026-10-07 anlık görüntüsünde kaldı. Toptan yenileme ayrı bir iş.
