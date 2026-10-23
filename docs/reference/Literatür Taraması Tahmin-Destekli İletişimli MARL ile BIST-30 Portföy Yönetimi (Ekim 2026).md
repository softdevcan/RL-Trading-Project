# Literatür Taraması: Tahmin-Destekli İletişimli MARL ile BIST-30 Portföy Yönetimi

Oct 6, 2026 · @Can

> **Doğrulama notu (2026-10-06):** Bu belge claude.ai Research çıktısıdır ve özgün hâliyle korunmuştur. Bağımsız doğrulama: [../thesis/literature-review-verification.md](../thesis/literature-review-verification.md). Özet: uydurma kaynak yok; ancak 9+ hakemli yayın "ön baskı" sayılmış, 25+ künye alanı eksik/yanlış (ör. BibTeX'te Sattar vd. ikinci yazar *Amna* Sarwar), DCMAC (#27) yazarlarınca hatalı ilan edilmiş, #72 Nawathe vd. yanlış okunmuş, §2.4'teki "8447.2" iddiası PDF çıkarım yanılsaması. Tezde kullanılacak doğrulanmış künyeler: [../thesis/vision-and-scope.md §9](../thesis/vision-and-scope.md).

## Önce en önemli sonuç

Altı boşluk iddiasından hiçbiri tek bir çalışmayla **kapanmış** değil; ancak dördü (G1, G2, G3, G5) bugünkü ifadeleriyle savunmada zorlanacak ölçüde daraldı. G4'ün DRL kısmı yeniden yazılmalı; G6 açık ama ayrıştırma sızıntısı nedeniyle yüksek riskli.

| Boşluk | Tehdit eden çalışma | Neyi daraltıyor |
| --- | --- | --- |
| G2 | [Zhang (2025), Scientific Reports](https://doi.org/10.1038/s41598-025-32408-w) | Ajanların portföy önerilerini, son tahmin isabetinin üstel hareketli ortalamasından hesaplanan güvenle ağırlıklandırıyor; mesajları öğrenilmiş dikkatle topluyor |
| G2 | [ADMAC (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29708) | Alıcı, gelen mesajın güvenilirliğini tahmin edip karar üzerindeki etkisini buna göre azaltıyor |
| G2 | [DebUnc (2024)](https://arxiv.org/abs/2407.06426) | LLM tartışmasında daha az belirsiz ajanın yanıtına giden dikkat ağırlıkları artırılıyor |
| G1 | [Fang vd. (2023)](https://arxiv.org/abs/2307.03119) | Finansta (emir yürütme) CommNet, TarMAC ve iletişimsiz yöntemleri aynı düzenekte karşılaştırıyor; GNN yok |
| G5 | [Shi & Hollifield (2025)](https://arxiv.org/abs/2511.13614) | LLM ajanlarında iletişimin değerinin piyasa türüne göre değiştiğini gösteriyor |
| G4 | [Candar (2022), İTÜ tezi](https://polen.itu.edu.tr/items/88affa11-5549-4dbe-8db7-20e02bf507e3) | BIST30'da DDPG/PPO ile temel veri kullanan portföy optimizasyonu; atıf zorunlu |

Ayrıntılar 5. bölümde. Bu doküman, 6 Ekim 2026'da iki oturumda yapılan taramanın birleşik hâlidir.

## 0. Tarama protokolü

Tarama 6 Ekim 2026'da iki oturumda yapıldı: 64 web araması ve 3 tam sayfa getirme. Bu bölüm tezin 2. bölümünde yöntem alt başlığı olarak kullanılabilir.

**Erişilen kaynaklar.** arXiv (q-fin.PM, q-fin.TR, cs.LG, cs.MA), PubMed Central, IEEE Xplore, ACM DL, Springer, Elsevier/ScienceDirect, Wiley, Taylor & Francis, MDPI, OpenAlex, SSRN, RePEc (IDEAS, EconPapers), NeurIPS/ICLR/ICML/AAAI/IJCAI/AAMAS bildiri sayfaları, DergiPark, AVESİS, İTÜ Polen ve diğer üniversite depoları, GitHub sürüm sayfaları, PyPI ve resmi dokümantasyon siteleri.

**Erişilemeyenler.** Scopus/Web of Science'a doğrudan erişim yoktu. YÖK Ulusal Tez Merkezi dinamik arayüzü nedeniyle taranamadı; BIST tezleri yalnızca üniversite depolarından bulundu. Retraction Watch veritabanı sorgulanamadı; ziyaret edilen yayıncı sayfalarında geri çekme bildirimi görülmedi.

**Örnek sorgu dizeleri (eksen bazında)**

| Eksen | Örnek sorgular |
| --- | --- |
| A | `graph attention heterogeneous multi-agent DRL portfolio`, `collaborative multi-agent RL portfolio`, `sector-level agents multi-agent RL portfolio`, `hierarchical DRL multi-agent portfolio` |
| B | `uncertainty-aware communication MARL`, `trust-aware / confidence-weighted message passing MARL`, `epistemic uncertainty sender confidence message aggregation`, `MARL communication CommNet TarMAC GNN comparison` |
| C | `forecast-augmented DRL portfolio prediction uncertainty ablation` |
| D | `decomposition-based forecasting look-ahead bias CEEMDAN VMD whole series`, `revisiting efficacy of signal decomposition`, `PyEMD CEEMDAN Colominas`, `TATS trend-adjusted time series`, `time series foundation models daily stock returns` |
| E | `regime-aware RL portfolio HMM bull bear` |
| F | `differential Sharpe ratio reward transaction cost turnover penalty`, `CVaR-constrained RL portfolio` |
| G | `DRL portfolio fails equal weight buy-and-hold reproducibility`, `statistical precipice IQM`, `deflated Sharpe`, `probability of backtest overfitting`, `model confidence set`, `Ledoit Wolf Sharpe test` |
| H | `attention is not explanation` |
| I | `Borsa Istanbul DRL BIST 30 PPO`, `derin pekiştirmeli öğrenme Borsa İstanbul portföy dergipark`, `Turkish stock market / BIST 100 reinforcement learning` |
| J | `TradingAgents FinCon FinMem FinAgent survey` |
| K | `PettingZoo releases`, `RLlib new API stack RLModule`, `BenchMARL TorchRL releases` |
| Tohum | 2.1 ve 2.2'deki her başlık ve arXiv kimliği ayrı ayrı |

**Dahil etme ölçütleri**

1. Konu A–K eksenlerinden biriyle doğrudan ilgili.
2. Ağırlık Ocak 2023 – Ekim 2026; daha eskiler yalnızca "temel" etiketiyle.
3. Yazar, yıl, başlık ve yayın yeri kaynağın kendi sayfasından ya da güvenilir bir indeksleyiciden görülmüş; görülmeyen alan ⚠ ile işaretli.
4. Çalışan bir DOI ya da arXiv/URL var.

**Dışlama ölçütleri.** Blog ve Medium yazıları; ders projesi raporları (ör. Stanford CS224R); lise dergileri ve popüler siteler; kapsam dışı konular (HFT, kripto, türev; yalnızca konumlandırma gerekiyorsa tutuldu); künyesi hiçbir sayfada görülemeyen kayıtlar.

**PRISMA benzeri akış (el ile sayım, yaklaşık)**

| Aşama | Sayı | Not |
| --- | --- | --- |
| Arama sonuçlarında görülen kayıt | ≈ 595 | Tekrarlar dahil, sonuç parçası düzeyinde |
| Tekrar ayıklandıktan sonra benzersiz kayıt | ≈ 245 |  |
| Başlık/özetle elenen | ≈ 105 | Konu dışı, popüler içerik, kripto/HFT |
| Ayrıntılı incelenen | ≈ 140 | Tam metin: Zhang (2025), Fang vd. (2023), de-la-Rica-Escudero vd. (2025) ve birkaç arXiv HTML sürümü; geri kalanı özet sayfası |
| Ayrıntılı incelemede elenen | ≈ 30 | Ders raporları, doğrulanamayanlar, kapsam dışı |
| Dahil edilen | 121 | 31 tohum + 90 yeni kaynak (6. bölümdeki matris) |

**Sınırlılıklar**

- Kaynakların çoğu "yalnız özet" düzeyinde okundu; matriste her satır için belirtildi.
- Atıf sayıları (Semantic Scholar / Google Scholar) sistematik olarak toplanamadı; yalnızca sayfada göründüğü durumlarda yazıldı.
- JCR/Scimago çeyrek bilgisi kaynakla doğrulanmadı; matriste "doğrulanmadı" yazıyor.
- Sayılar el ile tutulduğu için yaklaşıktır; tezde kullanmadan önce bir referans yöneticisinde (Zotero) yeniden sayılmalıdır.

## 1. Yönetici özeti

Alan 2025–2026'da hiyerarşik ve rol tabanlı çok-ajanlı yapılara, belirsizliğe duyarlı karar katmanlarına ve LLM ajanlarına kaydı; buna karşın değerlendirme disiplini zayıf kaldı. Tezin özgün çekirdeği hâlâ savunulabilir, ancak daha dar ifade edilmeli.

1. **MARL-portföy literatürü büyüdü ama iletişimi deney değişkeni yapmadı.** Rol tabanlı ([Zhang, 2025](https://doi.org/10.1038/s41598-025-32408-w)), hiyerarşik ([Sun vd., 2025](https://doi.org/10.1007/s40747-025-01884-y); [SAMP-HDRL](https://arxiv.org/abs/2512.22895)) ve sektör–hisse iki düzeyli ([RL-BHRP](https://arxiv.org/abs/2508.11856)) yapılar öne çıkıyor. Hiçbirinde iletişimsiz / dikkat / GNN aynı düzenekte karşılaştırılmıyor.
2. **G2'nin bileşenleri dağınık hâlde mevcut.** Güvenle ağırlıklı birleştirme (Zhang, 2025), alıcı tarafında öğrenilen mesaj güvenilirliği ([ADMAC](https://ojs.aaai.org/index.php/AAAI/article/view/29708)) ve belirsizliğe göre dikkat ölçekleme ([DebUnc](https://arxiv.org/abs/2407.06426)) ayrı ayrı yapılmış. Harici bir tahmincinin kalibre güveninin öğrenilen dikkat logitlerine girmesi ve alıcının öz-güven geçidi bulunamadı. 2026 tarihli bir çalışma, ters-entropi ağırlıklı mesaj birleştirmenin sürüleşmeye yol açabildiğini gösteriyor ([arXiv:2605.06988](https://arxiv.org/abs/2605.06988)).
3. **Makale 2 için iki kritik bulgu var.** Birincisi, ayrıştırmayı seri bütününe uygulayıp sonra bölmek sızıntıdır; nedensel işlendiğinde ayrıştırmanın katkısı azalıyor ([Zhang D. vd., 2025](https://doi.org/10.1117/12.3045386)). İkincisi, **TATS özgün bir yöntem olarak yayımlanmış**: XGBoost trend sınıflandırıcısıyla LSTM/Bi-LSTM tahminini düzelten Trend-Adjusted Time Series modeli ([Kazemdehbashi, 2026](https://arxiv.org/abs/2601.12706)). TATS'a atıf zorunlu; kendi katkınız olarak sunamazsınız.
4. **DRL'nin basit baseline'ları geçtiği iddiası zayıflıyor.** 1/N'ye karşı anlamlı üstünlük bulunamayan çalışmalar ([Aarnio, 2025](https://www.utupub.fi/bitstreams/7d12624f-27e2-4448-99c9-2520ec3009f3/download)) ve PPO'da tohumlar arası getiri standart sapmasının ortalamaya yakın çıktığı tanı çalışmaları ([Research Square ön baskısı](https://assets-eu.researchsquare.com/files/rs-9824202/v1_covered_2dbf3593-a58c-4edb-bb2a-bb85e54443cb.pdf)) 3 tohumun yetersiz olduğunu gösteriyor.
5. **Zaman serisi temel modelleri günlük getiride GBM'leri geçmiyor.** Sıfır-atış modeller günlük fazla getiride CatBoost ve LightGBM'in gerisinde kalıyor ([arXiv:2511.18578](https://arxiv.org/abs/2511.18578)). GBM + TFT seçiminiz savunulabilir.
6. **LLM tabanlı çok-ajanlı işlem sistemleri öne çıktı.** [TradingAgents](https://arxiv.org/abs/2412.20138) gibi sistemler doğal dille koordine oluyor; [Shi & Hollifield (2025)](https://arxiv.org/abs/2511.13614) iletişimin değerinin piyasa türüne göre değiştiğini gösteren ilk karşılaştırmayı yaptı.
7. **BIST'te DRL var, MARL yok.** BIST30 üzerinde DDPG/PPO portföy tezi ([Candar, 2022](https://polen.itu.edu.tr/items/88affa11-5549-4dbe-8db7-20e02bf507e3)) ve birkaç tez/bildiri bulundu; MARL çalışması bulunamadı.

## 2. Kritik uyarılar

Üç tür uyarı var: özgünlüğü daraltan çalışmalar, hatalı ya da uydurma tohum kaynaklar ve metodolojide literatürün kusur saydığı noktalar.

### 2.1 Özgünlüğü daraltan çalışmalar

Ayrıntı 5. bölümde. En kritik beş kaynak: Zhang (2025) → G2, G3, G5; Fang vd. (2023) → G1; Shi & Hollifield (2025) → G5; Candar (2022) → G4; Kazemdehbashi (2026) → G6 (TATS).

### 2.2 Tohum kaynak sorunları

**Bulunamayan dört kaynak (kullanmayın):** Preprints.org 202511.0447; Scientific Reports 2025 "DNN + RL for exchange rate forecasting"; Decision Science Letters 2025 "Convolutional DRL for emerging stock market"; MDPI AI 2025 "DRL framework for NIFTY 50". Bunlar önceki yapay zekâ taramasının uydurduğu kaynaklar olabilir.

**Künyesi hatalı kaynaklar**

| Kaynak | Hata | Doğrusu |
| --- | --- | --- |
| Lwele vd. 2025 | "Quantitative Finance" dergisi değil | arXiv ön baskısı, kategori q-fin.PM ([2511.11481](https://arxiv.org/abs/2511.11481)); hakemsiz |
| Macrì & Lillo 2025 | "Quantitative Finance" değil | Applied Mathematical Finance 31(5):312–342 ([10.1080/1350486X.2025.2490157](https://www.tandfonline.com/doi/abs/10.1080/1350486X.2025.2490157)) |
| "ICCBDA 2025" | Konferans adı yanlış | CIBDA 2025, ACM, ss. 1208–1214 |
| IJCAI 2023 "Scalable Communication" | Başlık eksik | "…via Transformer-Based Email Mechanism" |
| Neurocomputing 2024 "Explicit Teammate Modeling" | Başlık farklı | "Multi-agent cooperative strategy with explicit teammate modeling and targeted informative communication", 586:127638 |
| arXiv 2511.12120 | Yeni çalışma değil | ICAIF 2020 makalesinin 2025'te arXiv'e yüklenmiş hâli |
| HAELT | Kapsam dışı | Yüksek frekanslı AAPL verisi (HFT) |

"Quantitative Finance" etiketli iki kaynağın arXiv kategori adından türetildiği anlaşılıyor; Ayari (2025) ise gerçekten Quantitative Finance dergisinde.

**Bütünlük uyarısı: Zhang (2025).** Kaynakçadaki 17 numaralı atıf "Yang, K., Chen, L. & Wu, X." yazarlarıyla Complex & Intelligent Systems 2024 olarak verilmiş; oysa aynı başlıklı gerçek makale Sun, Xi, Stefanidis, Jiang ve Su'ya ait ve 2025'te yayımlandı. Kaynakça kısmen otomatik üretilmiş olabilir. Bu makaleye "en yakın çalışma" olarak atıf yapın, ama metodolojik eleştirinizi açıkça yazın: kod yalnız talep üzerine, tohum sayısı ve işlem maliyeti raporlanmamış.

### 2.3 Metodolojide literatürün kusur saydığı noktalar

1. **Ayrıştırma sızıntısı.** ICEEMDAN serinin tamamına uygulanıp sonra bölünüyorsa bu yayın engelleyicidir. Çözüm: yalnızca geçmişle, ileri yürüyen (walk-forward) ayrıştırma. PyEMD `CEEMDAN` belirlenimci değil; tekrarlanabilirlik için `noise_seed()` gerekir.
2. **TATS'ın sunumu.** TATS Kazemdehbashi (2026) tarafından aynı mimariyle (XGBoost trend sınıflandırıcısı + regresör) yayımlandı. Makale 2'de bunu kendi bileşeniniz gibi değil, uyarladığınız yöntem olarak sunun.
3. **Tohum sayısı.** 3 tohum yetersiz; az koşulda aralık tahmini ve IQM raporlanmalı ([Agarwal vd., 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)).
4. **Testlerin kapsamı.** Diebold-Mariano tahmin isabeti içindir. Sharpe farkı için Ledoit-Wolf testi, çok sayıda yapılandırma için Model Confidence Set kullanılmalı.
5. **Hayatta kalma yanlılığı (benim değerlendirmem).** 2018–2026 için bugünkü BIST-30 bileşenleri kullanılıyorsa evren seçiminde yanlılık oluşur. Tarihsel bileşen listesini dönem dönem kullanın.
6. **Temel oranlarda ileriye bakma (benim değerlendirmem).** yfinance temel oranları "o tarihte bilinen" (point-in-time) veri sunmaz; sonradan açıklanan değerler geçmişe sızabilir.
7. **Makro yayın gecikmesi (benim değerlendirmem).** TÜFE ve ÜFE referans aydan sonra açıklanır; açıklanma tarihine göre hizalayın.
8. **Nominal getiri ve risksiz faiz (benim değerlendirmem).** Yüksek enflasyonda nominal TL getirisi yanıltıcıdır. Sonuçları nominal TL, TÜFE ile reelleştirilmiş TL ve USD bazında raporlayın; Sharpe'ta risksiz faiz olarak politika faizini ya da TLREF'i kullanın.

### 2.4 Önceki çalışmanın (tez-final.pdf) jürinin göreceği tutarsızlıkları

- **Al-ve-Tut getirisi iki farklı değerle yazılmış:** Tablo 2'de %46,44, 7.2'deki metinde "%146,44".
- **Yıllıklaştırma tutarsız:** Faz 1 satırlarında yıllık getiri kümülatiften düşük, PPO (Faz 2) satırında yüksek.
- **Karşılaştırılamaz satırlar aynı tabloda:** PPO (Faz 2) BIST-30 evreninde, Al-ve-Tut 5 hisseyle hesaplanmış.
- **Taslak kalıntısı:** "Sharpe oranını ise b seviyesine".
- **Yanlış atıf:** SSDAE "Yue vd. ve Bai vd. tarafından geliştirilen" diye yazılmış; Bai vd. bir derleme, SSDAE Yue vd.'ye ait.
- **"RMS" yanlış yorumlanmış:** Sattar vd. (2025)'te RMS, Root Mean Square değil; kazanç raporları (Reports), maksimum düşüş ödülü (Max drawdown) ve duyarlılık (Sentiment) bileşenlerinin baş harfleri ([IEEE Xplore](https://ieeexplore.ieee.org/document/10904473/)). Önceki metindeki "RMS tabanlı normalizasyon" açıklaması düzeltilmeli.
- **Lwele (2025) yanlış özetlenmiş:** Çalışmada ajan oynaklığı stabilize ediyor ama aşırı tutucu politikaya yakınsadığı için risk-ayarlı getirisi baseline'ın altında kalıyor. "Riske duyarlı ödülün önemini gösterdi" demek bulguyu tersine çevirir.
- **Zayıf deney tasarımı:** tek tohum, algoritma başına 5 Optuna denemesi, yalnız boğa piyasasını kapsayan tek test yılı.
- **İşlem sayısı belirsiz:** tablodaki "8447.2" muhtemelen "844" ile bölüm numarası "7.2"nin birleşmesi.

## 3. Tohum kaynak doğrulama tablosu

2.1'deki 14 kaynağın tamamı doğrulandı; 2.2'deki 17 kaynaktan 13'ü bulundu (5'i düzeltilerek), 4'ü bulunamadı.

### 3.1 Önceki kaynakça (2.1)

| # | Kaynak | Durum | Düzeltilmiş künye / not | DOI |
| --- | --- | --- | --- | --- |
| 1 | [Ansari vd. 2024](https://doaj.org/article/8d14a5ea52744304ae52085bec6f9e15) | Doğrulandı | Ansari, Y., Gillani, S., Bukhari, M., Lee, B., Maqsood, M., Rho, S. IEEE Access 12:90041–90060. DOAJ kaydı dört yazar, Springer atıfları altı yazar listeliyor; yayıncı PDF'inden teyit edin. 30 Dow Jones hissesinde test | 10.1109/ACCESS.2024.3418510 |
| 2 | [Sattar vd. 2025](https://ieeexplore.ieee.org/document/10904473/) | Doğrulandı | Sattar, A., Sarwar, A., Gillani, S., Bukhari, M., Rho, S., Faseeh, M. IEEE Access. PPO/A2C/DDPG\_RMS; en iyi model kümülatif %27, Sharpe 0,66 (^DJI'ye karşı) | 10.1109/ACCESS.2025.3546099 |
| 3 | [Ayari 2025](https://www.zeppelin-university.com/chairs/finance/news/publication-drl-portfolio-management-oct2025.php) | Doğrulandı | Quantitative Finance (T&F). TD3 + BARRA + DES ödülü; 1.000 Monte Carlo portföyü; FDR denetimli permütasyon testi. DOI ⚠ görülmedi | SSRN 5595570 |
| 4 | [Bai vd. 2025](https://arxiv.org/abs/2411.12746) | Yazarlar doğrulandı | Bai, Y., Gao, Y., Wan, R., Zhang, S., Song, R. Annual Review of Statistics and Its Application; dergi DOI'si ⚠ görülmedi | arXiv:2411.12746 |
| 5 | [Chaudhari & Mahajan 2025](https://pubmed.ncbi.nlm.nih.gov/40207066/) | Doğrulandı | MethodsX 14:103260. A2C, PPO2, SAC; PACTRO yöntemi; Kaggle verisi | 10.1016/j.mex.2025.103260 |
| 6 | [de-la-Rica-Escudero vd. 2025](https://pubmed.ncbi.nlm.nih.gov/39820096/) | Doğrulandı | de-la-Rica-Escudero, A., Garrido-Merchán, E. C., Coronado-Vaca, M. PLOS ONE 20(1):e0315528. PPO + SHAP/LIME; [kod açık](https://github.com/aleedelarica/XDRL-for-finance) | 10.1371/journal.pone.0315528 |
| 7 | [Lwele vd. 2025](https://arxiv.org/abs/2511.11481) | **Düzeltildi** | Lwele, E., Emmanuel, S., Sitali, S. G. arXiv ön baskısı (dergi değil). PPO; ajan baseline'ın altında kaldı | arXiv:2511.11481 |
| 8 | [González Cortés vd. 2024](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13667) | Doğrulandı | González Cortés, D., Onieva, E., Pastor, I., Trinchera, L., Wu, J. Expert Systems 41(11):e13667. CAC-40; dikkat katmanlı açıklanabilir RL | 10.1111/exsy.13667 |
| 9 | [Park vd. 2024](https://ieeexplore.ieee.org/document/10419157) | Doğrulandı | Park, J.-H., Kim, J.-H., Huh, J.-H. IEEE Access 12:20705–20725. Tek hisse (AAPL), PPO; S&P 500 ve ABD faizi eklenmiş model | 10.1109/ACCESS.2024.3361035 |
| 10 | [Macrì & Lillo 2024](https://www.tandfonline.com/doi/abs/10.1080/1350486X.2025.2490157) | **Düzeltildi** | Applied Mathematical Finance 31(5):312–342 (QF değil). DDQN; zamanla değişen likiditede optimal yürütme | 10.1080/1350486X.2025.2490157 |
| 11 | [Millea & Edalat 2022](https://ideas.repec.org/a/gam/jijfss/v11y2022i1p10-d1019580.html) | Doğrulandı | IJFS (MDPI) 11(1):10. HRP/HERC alt ajanları + üst DRL. DOI MDPI kalıbından | 10.3390/ijfs11010010 |
| 12 | [Park & Lee 2021](https://scite.ai/reports/practical-algorithmic-trading-using-state-aXK42x9j) | Doğrulandı | Park, D.-Y., Lee, K.-H. IEEE Access 9:152310–152321. SIRL-Trader: SRL + TD3 + davranış klonlama; yalnız uzun | 10.1109/ACCESS.2021.3127209 |
| 13 | [Thongkairat & Yamaka 2025](https://doi.org/10.3390/math13030461) | Doğrulandı | Mathematics 13(3):461 (MDPI). DRL + Iterative Model Combining Algorithm; SET50 | 10.3390/math13030461 |
| 14 | [Yue vd. 2022](https://www.mdpi.com/2079-9292/11/9/1506) | Doğrulandı | Yue, H., Liu, J., Tian, D., Zhang, Q. Electronics 11(9):1506 (MDPI). SSDAE + A2C | 10.3390/electronics11091506 |

### 3.2 Vizyon belgesindeki kaynaklar (2.2)

| # | Vizyon belgesindeki kayıt | Durum | Doğru künye | DOI / URL |
| --- | --- | --- | --- | --- |
| 1 | Collaborative MARL, ICCBDA 2025 | **Düzeltildi** | Hao, J., Gao, M., Han, Y., Gao, Q. CIBDA 2025, ACM, ss. 1208–1214. Varlık başına ajan; bireysel + küresel kâr ödülü; vadeli işlemler | [10.1145/3746709.3746915](https://doi.org/10.1145/3746709.3746915) |
| 2 | GAT heterojen MARL, Sci Rep 2025 | Doğrulandı | Zhang, B. Scientific Reports 16:2674 (27 Aralık 2025). Bkz. 2.2 bütünlük uyarısı | [10.1038/s41598-025-32408-w](https://doi.org/10.1038/s41598-025-32408-w) |
| 3 | Hierarchical DRL Multi-Agent, CAIS 2025 | **Düzeltildi** | Sun, R., Xi, Y., Stefanidis, A., Jiang, Z., Su, J. "A novel multi-agent dynamic portfolio optimization learning system based on hierarchical deep reinforcement learning". Complex & Intelligent Systems, 2025-05-29. Cilt/sayfa ⚠ | [10.1007/s40747-025-01884-y](https://doi.org/10.1007/s40747-025-01884-y) |
| 4 | StockMARL, EMSS 2025 | Doğrulandı | Zou, P., Siebers, P.-O. 37th EMSS 2025 (I3M 2025). Simülasyon platformu; işbirlikçi portföy MARL'ı değil. DOI ⚠ | [PDF](https://people.cs.nott.ac.uk/pszps/resources/zou-siebers-emss2025-corrected.pdf) |
| 5 | HARLF, arXiv 2507.18560 | Doğrulandı | Coriat, B., Benhamou, E. Ön baskı | [arXiv:2507.18560](https://arxiv.org/abs/2507.18560) |
| 6 | RL-BHRP, arXiv 2508.11856 | Doğrulandı | Kang, S., Tian, Z. Ön baskı | [arXiv:2508.11856](https://arxiv.org/abs/2508.11856) |
| 7 | Preprints.org 202511.0447 | **Bulunamadı** | — | — |
| 8 | MDPI Electronics 2025, sektör rotasyonu | Doğrulandı | Yan, Y., Zhang, C., An, Y., Zhang, B. Electronics 14(5):1036 | [10.3390/electronics14051036](https://www.mdpi.com/2079-9292/14/5/1036) |
| 9 | MACTAS, arXiv 2508.13661 | Doğrulandı | Wojtala, M., Stefańczyk, B., Bogucki, D., Lepak, Ł., Strykowski, J., Wawrzyński, P. Ön baskı; v3'te başlık değişti | [arXiv:2508.13661](https://arxiv.org/abs/2508.13661) |
| 10 | IJCAI 2023 Scalable Communication | **Düzeltildi** | Guo, X., Shi, D., Fan, W. TEM | [10.24963/ijcai.2023/15](https://doi.org/10.24963/ijcai.2023/15) |
| 11 | Neurocomputing 2024 Teammate Modeling | Kısmen doğrulandı | Başlık düzeltildi; 586:127638. Yazarlar ve DOI ⚠ (yalnızca başka bir makalenin kaynakçasından görüldü) | — |
| 12 | arXiv 2511.12120 | **Düzeltildi** | Yang, H., Liu, X.-Y., Zhong, S., Walid, A. ICAIF 2020; 30 Dow Jones hissesi | [10.1145/3383455.3422540](https://openfin.engineering.columbia.edu/sites/default/files/content/publications/ensemble.pdf) |
| 13 | HAELT, arXiv 2506.13981 | Başlık doğrulandı; yazarlar ⚠ | Yüksek frekanslı; kapsam dışı | [arXiv:2506.13981](https://arxiv.org/abs/2506.13981) |
| 14 | Sci Rep 2025 DNN + RL döviz kuru | **Bulunamadı** | — | — |
| 15 | JEEP 2024 BIST endeksleri | Doğrulandı | Sümerli Sarıgül, S., Aldemir, R., Uzunoğlu, H. JOEEP 9(2):96–106; TR Dizin. DOI ⚠ | [DergiPark](https://dergipark.org.tr/en/pub/joeep/article/1462937) |
| 16 | Decision Science Letters 2025 | **Bulunamadı** | — | — |
| 17 | MDPI AI 2025 NIFTY 50 | **Bulunamadı** | — | — |

**Sorunuz: Yang vd. ensemble strategy ICAIF 2020 mi, 2021 mi?** Makale ICAIF '20 bildirisi; DOI 10.1145/3383455.3422540. 2020 olarak atıf yapın.

## 4. Eksen bazında sentez (A–K)

Atıflar (Yazar, Yıl) biçiminde; tam künyeler 6. bölümdeki matriste ve 11. bölümdeki BibTeX'te.

### A. MARL ile portföy yönetimi ve alım-satım

Literatürde dört ajan granülaritesi öne çıkıyor. Varlık başına ajan yaklaşımında MAPS her ajanı farklı davranmaya yönlendirilen bağımsız bir yatırımcı olarak kurar (Lee vd., 2020; temel); MSPM her varlık için bir modül kullanır (Huang & Tanaka, 2022; temel); Hao vd. (2025) bu çizgiyi vadeli işlemlere taşır. Rol tabanlı yaklaşımda Zhang (2025) risk, getiri tahmini ve piyasa algısı ajanlarını; MASA iki işbirlikçi ajan ile bir piyasa gözlemcisini (Li vd., 2024); MARS risk profilleri güvenlik eleştirmeniyle zorlanan heterojen bir topluluğu kullanır (Chen vd., 2025).

Hiyerarşik ve sektör düzeyli yaklaşımlar 2025'te hızla çoğaldı. Sun vd. (2025) DJIA'dan 29 hisseyle hiyerarşik DRL sistemi kuruyor; SAMP-HDRL piyasayı dinamik gruplara ayırıp üst ve alt ajanlarla yönetiyor (Ren vd., 2025); MPLS sektör dağılımını üst ajana, sektör içi kararları alt ajanlara veriyor (2025); RL-BHRP riski sektörler ve hisseler arasında iki düzeyde dağıtıyor (Kang & Tian, 2025); He vd. (2025) bir makro ajanla sektörleri ağırlıklandırıyor.

Ödül tasarımında size en yakın öncül Hao vd. (2025): bireysel ve küresel kâr oranının etkisini inceleyip dinamik bir oran ayarı öneriyorlar. Bu, `α · bireysel + (1−α) · portföy Sharpe` tasarımınızın doğrudan öncülüdür. Kredi ataması ve CTDE (merkezi eğitim–dağıtık yürütme, centralized training with decentralized execution) çoğu çalışmada açıkça tanımlanmamış. Piyasalar ağırlıkla ABD ve Çin A-hisse; gelişmekte olan bir hisse piyasasında MARL-portföy çalışması bulunamadı. Metodolojik kalite düşük-orta: çoklu tohum nadir, maliyet seviyesi çoğu özette yok, iletişim yalnızca emir yürütmede (Fang vd., 2023) deney değişkeni olarak izole edilmiş.

### B. MARL'de öğrenilmiş iletişim

Mimari soy çizgisi CommNet → TarMAC → graf dikkatli MAGIC → transformer/graf tabanlı yöntemler şeklinde ilerliyor. TarMAC'ta ajanlar kime ne göndereceklerini yalnızca görev ödülüyle öğrenir (Das vd., 2019; temel). 2024–2026'da standart, dikkat/transformer tabanlı iletişim ya da öğrenilen seyrek graflar: CommFormer iletişim grafiğini politika ile birlikte optimize ediyor (Hu vd., 2024); f-MAT komşu ajanlar arasında transformerla iletişim kuruyor (Fan vd., 2025); MACTAS öz-dikkati tek başına iletişim aracı olarak kullanıyor (Wojtala vd., 2025); SOPS ve ExpoComm büyük ölçekte seyrek topoloji öğreniyor. 8–30 ajan ölçeğinde tam dikkatin maliyeti ihmal edilebilir; seyreltme yöntemlerine gerek yok.

G2 için kritik alt hat belirsizlik ve güvenilirlik. IMMAC ajanın içsel belirsizlik ve merakıyla gelen mesajlar için dikkat ağırlığı hesaplıyor (AAMAS 2021); ADMAC alıcı tarafında öğrenilen bir güvenilirlik kestiricisiyle güvenilmez mesajların etkisini azaltıyor (AAAI 2024); Sun vd. (2023) budanmış mesaj kümelerinden oluşan bir topluluk politikasıyla sertifikalı sağlamlık öneriyor. Ortak nokta: güven ya ajanın kendi gizli durumundan ya da alıcının yerel gözleminden öğreniliyor; harici, denetimli bir tahmincinin kalibre güveni hiçbirinde yok. 2026 tarihli bir ön baskı, gönderenin entropisiyle ters orantılı ağırlıklandırmanın sürüleşmeye yol açabildiğini gösteriyor; güven ağırlığını kırpmanız (clipping) ya da geçitlemeniz gerekebilir.

Dikkatin yorumlanabilirliği TarMAC'tan beri iddia ediliyor, ancak NLP'deki "dikkat açıklama mıdır" tartışmasından muaf değil (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019). Wiegreffe ve Pinter'in önerdiği testlerden ikisi (tek biçimli dikkat baseline'ı ve tohumlar arası dikkat varyansı) yorumlanabilirlik iddianız için doğrudan uygulanabilir.

### C. Tahmin-destekli RL

Tahmin RL'e dört yoldan giriyor. Gözlem en yaygın yol: Ansari vd. (2024), StockFormer (Gao vd., 2023) ve DeepClair (2024, LoRA ile aktarılan tahminci). Karar kısıtı: RTS-PnO'da uyarlanabilir belirsizlik kısıtı sabit kısıttan altı veri setinde daha iyi (2025). Öğrenmeyle güçlendirilmiş çevrimiçi portföy seçimi: Zhang, Zhao ve Zomaya (2025) tahminler hatalı olsa bile alt sınırı koruyan bir kural kanıtlıyor. İletişim yolu neredeyse boş: Zhang (2025)'te getiri tahmin ajanının güven aralıkları konsensüse giriyor, mesaj dikkatine değil.

Tahmin belirsizliğinin kullanımı 2025'te arttı (RTS-PnO; Bayesçi sinir ağlı IPS, 2025; Zhang, 2025). Tahmin katmanının ablation ile ölçülmesi ise yaygın değil. AQ2 için dört koşullu ablation (tahminsiz / yalnız gözlem / yalnız mesaj / ikisi birden) literatürde boş kalan bir hücreyi dolduruyor.

### D. Tahmin katmanı (Makale 2)

**Seri bütününe ayrıştırma sızıntı yaratıyor mu?** Evet; literatür bu konuda uzlaşmış durumda. Genel tahmin rehberleri, bölmeden önce seri bütününe ayrıştırma veya normalizasyon uygulamayı dolaylı sızıntı sayıyor (Forecast Evaluation Pitfalls, 2022). Mekanizma şu: ayrıştırılmış bileşenler fiilen evrişim çıktısıdır ve evrişime gelecek veri girer (RVFL çalışması, 2021). Finans özelinde 2017 tarihli bir çalışma, EEMD'nin parlak sonuçlarının ileriye bakma yanlılığından geldiğini ve bu düzeltildiğinde kazancın kaybolduğunu gösteriyor. Zhang D. vd. (2025) hatanın yaygın olduğunu ve nedensel işlemede ayrıştırmanın katkısının azaldığını bildiriyor; ancak oran veren sistematik bir sayım bulunamadı. Sızıntısız yöntemler: veriye dayalı dolgu, kayan pencere, ileri yürüyen ayrıştırma ve Multi-Front yöntemi.

**ICEEMDAN ve PyEMD.** CEEMDAN Torres vd. (2011), ICEEMDAN Colominas vd. (2014) kaynaklıdır. PyEMD'nin `CEEMDAN` sınıfı Torres (2011)'e atıf yapıyor ama Colominas (2014) iyileştirmelerini de içerdiğini belirtiyor ([belge](https://pyemd.readthedocs.io/en/latest/ceemdan.html)). Algoritmik fark (eğitim bilgime dayanıyor, özgün makaleden teyit edilmeli): ICEEMDAN ham beyaz gürültü yerine gürültünün k-ıncı EMD modunu ekler ve kalıntıyı yerel ortalamaların ortalaması olarak hesaplar. Makale 2'de ya kaynak koddan teyit edip "ICEEMDAN" deyin ya da "Colominas vd. (2014) iyileştirmelerini içeren PyEMD CEEMDAN" ifadesini kullanın.

**TATS özgün bir kaynakta mevcut.** Kazemdehbashi (2026), tahmini iki alt göreve ayırıp bir ikili sınıflandırıcının trend tahminiyle LSTM/Bi-LSTM çıktısını düzelten Trend-Adjusted Time Series modelini önerdi ve günlük altın fiyatında LSTM ile Bi-LSTM'den düşük hata raporladı ([arXiv:2601.12706](https://arxiv.org/abs/2601.12706)). Ön baskı, tek varlık, tek seri. Sizin katkınız TATS'ın kendisi değil; onun stacking topluluğuna ve çok varlıklı BIST verisine uyarlanması ve RL gözlemine aktarılması olabilir.

**Zaman serisi temel modelleri.** 94 ülkenin 34 yıllık günlük fazla getiri panelinde hazır modeller sıfır-atışta CatBoost ve LightGBM'in gerisinde kalıyor (Re(Visiting) TSFMs, 2025). Noguer i Alonso & Pereira Franklin (2026) rastgele yürüyüşe karşı kazançların küçük ve seyrek olduğunu, DM testinin yalnız iki durumda anlamlı çıktığını buluyor. Łaniewski & Ślepaczuk (2025) sıfır-atış Chronos'un rekabetçi isabetine karşın işlem getirisinin orta düzeyde kaldığını gösteriyor. GBM + TFT seçiminiz savunulabilir; sıfır-atış bir temel model ucuz bir sağlamlık kontrolü olarak eklenebilir.

### E. Rejime duyarlı RL ve durağan olmama

Rejim tanımında Gizli Markov modeli (HMM) baskın: Zhang (2025) 252 günlük kayan pencerede dört durumlu HMM, Nixon Raj (2025) HMM rejim olasılıklarını gözleme ekliyor, Verma vd. (2026) üç durumlu Gauss HMM'yi BIC ile seçiyor. Kural tabanlı ve ikili ajan yaklaşımları da var (boğa/ayı uzmanı iki ajan ve bir rejim seçici; MTX çalışması, 2026).

Rejim bazında raporlama yaygınlaşıyor ama bileşen katkısı rejime göre ayrıştırılmıyor; SAMP-HDRL üç rejimde raporluyor, ablation'ı ise küresel. Tohum disiplini için iyi bir örnek, 10 tohumla tutarlılık raporlayan rejime duyarlı PPO çalışması (Ilieva, 2026). En savunulabilir tanım (benim değerlendirmem): değerlendirme etiketini önceden ve kural tabanlı tanımlayın (200 günlük trend yönü + gerçekleşmiş oynaklık terciliyle "yatay"); model girdisi olarak HMM'nin yalnızca filtrelenmiş olasılıklarını kullanın. Viterbi ve düzleştirilmiş olasılıklar geleceği kullanır.

### F. Riske duyarlı ödül, pozisyon boyutlandırma ve işlem maliyeti

Riske duyarlı ödülün temeli Moody & Saffell'in doğrudan pekiştirme ve diferansiyel Sharpe çalışmasıdır (2001; temel). 2025–2026 çalışmalarında eğilim, ödüle birden çok sürtünme terimini birlikte koymak: küresel hisse portföyünde diferansiyel Sharpe + 2 baz puan maliyet + L1 dönüşüm cezası + yoğunlaşma cezası ([arXiv:2605.17307](https://arxiv.org/html/2605.17307v1)); Dirichlet politikalı dikkatli bir ajanda dönüşüm maliyeti + karesel varyans cezası (Xue & Ye, 2025). En yakın öncül PPO-HRAP: log getiri, VIX'e koşullu düşüş cezası, hedef maruziyetten sapma ve dönüşüm maliyetini birleştiriyor; beş tohumda Sharpe 0,62 ± 0,06 raporluyor ([arXiv:2610.01325](https://arxiv.org/pdf/2610.01325)). Bu, ablation'daki `Sortino − λ·turnover − μ·hedeften sapma` varyantınızın doğrudan öncülü.

Kuyruk riski tarafında CVaR kısıtlı politika gradyanı ve aktör-eleştirmen algoritmaları temel kaynaklardır (Chow & Ghavamzadeh, 2014; Chow vd., 2018). Ayari (2025) Differential Expected Shortfall ödülüyle maksimum düşüşte anlamlı iyileşme gösteriyor. Uyarıcı bulgu: Lwele vd. (2025)'te riske duyarlı ödül oynaklığı düşürüyor ama aşırı tutucu politikaya yakınsıyor. Maliyet duyarlılığı analizi için örnek, farklı maliyet oranlarında Sharpe'ı raporlayan Zhang, Zohren ve Roberts (2020) çalışması. Kelly tabanlı boyutlandırma için 2023–2026 RL literatürü bu taramada bulunamadı; bu alt konu ince.

### G. Değerlendirme metodolojisi ve backtest güvenilirliği

Dört eleştiri kanıtla destekleniyor. (1) Tek koşu: tohum duyarlılığı ön baskısı PPO'da çok yüksek tohum varyansı raporluyor. (2) Basit baseline'lar: %1 maliyetle eşit ağırlıklı al-tut en iyi çıkıyor (Multimodal DRL, 2024); klasik DRL algoritmalarının çoğu backtest'te hızla bozuluyor (Benchmarking Robustness, 2023); 1/N'ye karşı anlamlı fark yok (Aarnio, 2025). (3) Derlemeler açıklanabilirlik, MDP modellemesi ve sağlamlığı engel sayıyor (Bai vd., 2025; 167 makalelik sistematik derleme, 2025). (4) Olumlu örnekler: Ayari (2025) permütasyon + FDR; Xue & Ye (2025) arındırılmış (purged) walk-forward.

İstatistiksel araç seti her soruya ayrı yanıt veriyor: seçim yanlılığı için Deflated Sharpe Ratio (Bailey & López de Prado, 2014); backtest aşırı uyumu için PBO ve CSCV (Bailey vd., 2017); çok yapılandırma arasında en iyi küme için Model Confidence Set (Hansen vd., 2011); iki Sharpe farkı için Ledoit-Wolf testi (2008); tohumlar arası varyans için katmanlı bootstrap, IQM ve iyileşme olasılığı (Agarwal vd., 2021). 3 tohum yetersiz; öneriler 7. bölümde.

### H. Açıklanabilirlik (XAI)

Portföy DRL'ında XAI ağırlıkla sonradan (post hoc) yapılıyor: SHAP, LIME ve özellik önemi (de-la-Rica-Escudero vd., 2025); entegre gradyanlar (Guan & Liu, 2021); dikkat katmanları (González Cortés vd., 2024); ajan katmanları arası SHAP (SAMP-HDRL, 2025). MARL'de dikkat ağırlıkları "kim kimi dinliyor" diye okunabilir, ancak bu iki kontrolle savunulabilir: tek biçimli dikkat baseline'ı ve tohumlar arası dikkat varyansı. Daha nedensel bir alternatif, bir ajanın mesajı maskelendiğinde değer fonksiyonunun değişimini ölçen karşı-olgusal mesaj değeri (MUTE, 2026, ön baskı).

### I. BIST, Türkiye ve gelişmekte olan piyasalar

BIST'te RL/DRL çalışmaları var ama az ve çoğu tez ya da bildiri: BIST30'da DDPG/PPO portföy tezi (Candar, 2022); 8 hissede DQN türevleri (Eğe, 2024); BerTurk duyarlılığı ve sosyal bilgi grafiğiyle DQN (TJEECS, 2022); Türk makro göstergeleriyle Q-öğrenme/SARSA (ASYU, 2025); aracı kurumların gizli ödüllerini ters RL ile çıkaran çalışma (Borsa Istanbul Review, 2026). Makine öğrenmesiyle tahmin tarafında Sümerli Sarıgül vd. (2024), Ergenç & Aktaş (2025) ve Erdağ vd. (2026) var. Gelişmekte olan piyasalar için SET50 (Thongkairat & Yamaka, 2025) ve Kore ETF'leri (Chang, 2026) örnek.

Türkiye'ye özgü etkenler (faiz, enflasyon, kur) bazı çalışmalarda girdi olarak kullanılıyor, ancak getirilerin reelleştirilmesi ya da risksiz faizin enflasyona göre ayarlanması hiçbir özette görülmedi. Kalite düşük: tek hisse, kısa test, bazı yüksek isabet iddiaları sızıntı şüphesi uyandırıyor (ör. BIST100 yön tahmininde %77,29; Ayyıldız, 2025). YÖK Tez Merkezi taranamadı; jüri sorusu açısından bu tarama sizin tarafınızdan tamamlanmalı.

### J. LLM tabanlı çok-ajanlı işlem sistemleri (konumlandırma)

Ortak fark: bu sistemlerde ajanlar doğal dille ve istem düzeyinde koordine olur; sizde mesajlar ödül gradyanıyla öğrenilen sürekli vektörlerdir.

| Çalışma | Yayın | Bizden farkı |
| --- | --- | --- |
| [TradingAgents](https://arxiv.org/abs/2412.20138) | AAAI 2025 MARW çalıştayı | Rol tabanlı LLM ajanları metin tartışmasıyla karar verir; mesajlar öğrenilmez |
| FinCon | NeurIPS 2024 (⚠ derleme üzerinden) | Hiyerarşik yönetici-analist yapısı; inanç sözel olarak güncellenir |
| FinMem | 2023 (⚠) | Katmanlı bellekli tek LLM tüccarı |
| FinAgent | 2024 (⚠) | Çok kipli, araç destekli tek ajan |
| FinRobot | 2024 (⚠) | LLM ajan platformu; politika öğrenimi yok |
| HedgeAgents | 2025 (⚠) | Dengeye duyarlı çoklu LLM ajanı |
| QuantAgents / TradingGroup | 2025 (⚠) | Simüle işlem ve öz-yansıtma |
| [Shi & Hollifield](https://arxiv.org/abs/2511.13614) | NeurIPS 2025 çalıştayı | İletişim yapılarını karşılaştırır; konuşma kalitesi getiriyle ilişkisiz. Konuya en yakın, ama LLM ve kesitsel |
| [He vd.](https://arxiv.org/abs/2510.21147) | 2025, ön baskı | LLM + RL hibrit; RL yalnız birleştirme katmanında |
| [HARLF](https://arxiv.org/abs/2507.18560) | 2025, ön baskı | LLM yalnız duyarlılık skorlayıcısı |

Jüriye önerilen yanıt dört gerekçeye dayanır: gecikme ve maliyet, tekrarlanabilirlik (sıcaklık ve model sürümü sonuçları değiştirir), öğrenilmiş sayısal kanalın ablation ile ölçülebilirliği ve konuşma kalitesinin getiriyle ilişkisiz olduğu bulgusu.

### K. Araç ve çerçeve durumu

PettingZoo 1.27.0 en son sürüm; ortam kaydı (registry) ekledi, eski oluşturma API'sini kullanımdan kaldırdı ve Python 3.9 desteğini düşürdü ([sürüm notları](https://github.com/Farama-Foundation/PettingZoo/releases)). RLlib'de yeni API yığını artık varsayılan; özel modeller RLModule ile yazılıyor ve ajanlar arası paylaşımlı iletişim için `MultiRLModule` alt sınıfı öneriliyor ([Ray 2.59.0 belgesi](https://docs.ray.io/en/latest/rllib/rl-modules.html)). BenchMARL MAPPO/IPPO ve hazır bir GNN modeliyle geliyor ve TorchRL üzerine kurulu. Ayrıntı 8. bölümde.

## 5. Boşluk / özgünlük analizi (G1–G6)

Hiçbir boşluk tamamen kapanmadı; dördü kısmen kapandı ve G6'nın bir bileşeni (TATS) özgün olmaktan çıktı.

| Kod | Hüküm | En yakın çalışmalar | Önerilen dar iddia |
| --- | --- | --- | --- |
| G1 | **Kısmen kapandı** (finans genelinde); portföy + GNN için **açık** | Fang vd. 2023; Zhang 2025; Hu vd. 2024 (finans dışı) | Portföy yönetiminde iletişimsiz / dikkat / GAT'ın eşit bütçe, çoklu tohum ve aralık tahminiyle ilk karşılaştırması; ilk gelişmekte olan piyasa örneği |
| G2 | **Kısmen kapandı** | Zhang 2025; ADMAC 2024; IMMAC 2021; DebUnc 2024; sürüleşme 2026 | Harici kalibre tahmincinin güvenini hem gönderen tarafında logit önseli hem alıcı tarafında öz-güven geçidi olarak kullanmak |
| G3 | **Kısmen kapandı** (hiyerarşik sektör); eşdüzeyli sektör ajanları için **açık** | MPLS 2025; RL-BHRP 2025; He vd. 2025; Yan vd. 2025 | Merkezi dağıtıcı olmadan, öğrenilmiş iletişimle koordine olan eşdüzeyli sektör ajanları |
| G4 | MARL: **açık** (YÖK taraması bekliyor). DRL: iddia **yanlış ifade edilmiş** | Candar 2022; Eğe 2024; TJEECS 2022; ASYU 2025; BIR 2026 | Erişilebilen literatürde BIST-30'da ilk MARL uygulaması |
| G5 | **Kısmen kapandı** (LLM, kesitsel) | Shi & Hollifield 2025; SAMP-HDRL 2025; Zhang 2025 | Öğrenilmiş sayısal iletişimin marjinal katkısının, önceden tanımlı zamansal rejimlere koşullu ölçümü |
| G6 | **Açık ama daraldı**: TATS özgün değil | Kazemdehbashi 2026; sızıntı eleştirisi literatürü; StockFormer; DeepClair | Sızıntısız ICEEMDAN altında ayrıştırma ve TATS'ın katkısının tahmin ve RL katmanında ölçülmesi |

### G1. İletişim mimarilerinin karşılaştırması

[Fang vd. (2023)](https://arxiv.org/abs/2307.03119) her ajanın bir emri yürüttüğü multi-order execution görevinde PPO, DDQN, CommNet, TarMAC, niyet paylaşımı ve kendi yöntemlerini Çin A-hisse ve ABD verisinde aynı düzenekte karşılaştırıyor. Fark: görev emir yürütme, portföy değil; GNN, ajanlar arası belirgin graf yapısı olmadığı gerekçesiyle dışarıda; amaç yeni bir protokol önermek. Zhang (2025) GAT'i grafik evrişimi veya korelasyonla değiştiren bir ablation yapıyor ama iletişimsiz MAPPO ile karşılaştırmıyor. Fang vd.'ye "finanstaki tek öncül" olarak açıkça atıf yapın.

### G2. Güven ağırlıklı dikkat

- **Zhang (2025):** Güven, ajanların nihai portföy önerilerini birleştiren konsensüs ağırlığında kullanılıyor ve geçmiş isabetin hareketli ortalamasından geliyor; mesaj dikkatini modüle etmiyor.
- **ADMAC (2024):** Güvenilirlik alıcının yerel bilgisinden öğreniliyor; amaç düşmanca mesajlara karşı savunma.
- **IMMAC (2021):** Dikkat ajanın içsel belirsizlik ve merakından türetiliyor; harici tahminci yok.
- **DebUnc (2024):** Kavramsal olarak en yakın mekanizma; ajanın belirsizliğinin tersiyle token ağırlıkları ölçekleniyor. Ancak LLM tartışmasında, eğitim ve RL yok.
- **Sürüleşme çalışması (2026):** Güvene dayalı ağırlıklandırmanın sistematik bir başarısızlık biçimi olabileceğini gösteriyor.

Bulunamayan iki bileşen: (i) kalibre edilmiş harici topluluk tahmincisinin güveninin öğrenilen dikkat logitlerine log-güven yanlılığı olarak eklenmesi; (ii) alıcının kendi düşük güveninin gelen mesajlara verdiği toplam ağırlığı artırması. Ablation'da üç alternatife karşı sınanmalı: öğrenilmiş güvenilirlik (ADMAC tipi), ters belirsizlikle sabit ölçekleme (DebUnc tipi) ve güvensiz standart dikkat. Güvenin kalibrasyonunu (ECE, güvenilirlik diyagramı) raporlamak zorunlu; mekanizmanın değeri buna dayanıyor.

### G3. Sektör tabanlı ajan granülaritesi

MPLS (2025) sektör dağılımını üst ajana veriyor; RL-BHRP (2025) sektör–hisse hiyerarşisini tek bir RL ajanıyla kuruyor; He vd. (2025) sektör seçimini LLM tabanlı makro ajana bırakıyor; Yan vd. (2025) sektör rotasyonunu DRL ile yapıyor; SAMP-HDRL sabit sektör yerine dinamik gruplama kullanıyor. Bulunamayan: eşdüzeyli sektör ajanlarının merkezi dağıtıcı olmadan öğrenilmiş iletişimle koordinasyonu. MPLS tipi hiyerarşik yapıyı yedek seçenek değil, zorunlu baseline yapın.

### G4. BIST / Türkiye

BIST'te MARL çalışması bulunamadı; YÖK Tez Merkezi taraması tamamlanmadan "yok" değil, "erişilebilen literatürde bulunamamıştır" deyin. DRL tarafında [Candar (2022)](https://polen.itu.edu.tr/items/88affa11-5549-4dbe-8db7-20e02bf507e3) sizin tek-ajan baseline'ınızın doğrudan öncülü: BIST30, PPO/DDPG, temel veriler, BIST30 endeksi baseline'ı. Tezde farklar açıkça yazılmalı: makro ve tahmin özellikleri, walk-forward, çoklu algoritma ve çok-ajanlı yapı.

### G5. İletişimin rejime göre değeri

[Shi & Hollifield (2025)](https://arxiv.org/abs/2511.13614) iletişimin değerinin piyasaya göre değiştiğini gösteriyor; ancak ajanlar LLM, "piyasa koşulu" zamansal rejim değil kesitsel hisse evreni, iletişim doğal dil. Zhang (2025) rejimleri girdi olarak kullanıyor ama iletişim ablation'ını rejime göre ayrıştırmıyor. Shi & Hollifield'ı motivasyon olarak kullanın: hipoteziniz, onların LLM bulgusunun öğrenilmiş sayısal iletişimde ve zamansal rejimlerde de geçerli olup olmadığını sınamak olsun.

### G6. Makale 2

ICEEMDAN filtreli stacking topluluğunun ve bir trend düzelticinin çıktısını RL gözlemine veren bir BIST çalışması bulunamadı; boşluk teknik olarak açık. Ancak TATS [Kazemdehbashi (2026)](https://arxiv.org/abs/2601.12706) tarafından aynı mimariyle (XGBoost trend sınıflandırıcısı + regresör) yayımlandığı için "TATS düzeltmesi" artık bir katkı değil, bir bileşendir. Ayrıca hakem ayrıştırma sızıntısı literatürünü bilecektir.

**Konumlandırma önerisi.** Makale 2'nin ana sorusu şu olsun: "Sızıntısız (ileri yürüyen) ICEEMDAN altında ayrıştırmanın ve TATS düzeltmesinin tahmin ve RL performansına katkısı ne kalıyor?" Aynı boru hattını iki koşulda koşturun: (i) seri bütününe ayrıştırma, (ii) yalnız geçmişle ayrıştırma. Farkı tahmin katmanında Diebold-Mariano, RL katmanında Deflated Sharpe ile raporlayın. Bu tasarım, Zhang D. vd. (2025) ve 2017 tarihli EMD yanlılığı çalışmasının bulgularını gelişmekte olan bir piyasada RL'e taşır ve Aralık 2026 hedefi için daha savunulabilirdir. TATS'ı ilk kez çok varlıklı bir hisse evreninde ve stacking topluluğu üzerine uygulamak ikincil katkı olarak sunulabilir.

## 6. Literatür matrisi

Matris, tohum seti dışındaki 90 yeni kaynağı listeler (tohumlar 3. bölümde). 2025–2026 tarihli kaynak sayısı 45 (%50); tohumlar dahil edildiğinde oran %58.

**Okuma anahtarı**

- **Puan (0–6):** her ölçüt için +1 — (1) işlem maliyeti, (2) zamansal OOS / walk-forward, (3) çoklu tohum veya varyans raporu, (4) güçlü basit baseline'lar, (5) istatistiksel test, (6) kod veya veri erişimi. Yalnız ampirik finans çalışmalarına verildi; "—" uygulanmaz demek. Puanların çoğu **yalnız özetten** çıkarıldı; tam metinle teyit edin.
- **Kalite:** JCR/Scimago çeyreği bu taramada kaynakla doğrulanmadı; tüm satırlar için "doğrulanmadı". Atıf sayıları toplanamadı.
- **Tür:** D = dergi, K = konferans, Ö = ön baskı (hakemsiz), T = tez, Y = yazılım. ⚠ = alan doğrulanamadı.
- **Öncelik:** Oku / Atıf yeter / Arka plan.

### 6.1 MARL ile portföy ve alım-satım (Eksen A)

| # | Künye | Tür | Piyasa & veri | Yöntem | Puan | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Li, Z., Tam, V. vd. (2024). [MASA](https://arxiv.org/abs/2402.00515) | K (AAMAS) | CSI 300, DJIA, S&P 500; 10 yıl | RL getiri ajanı + çözücü risk ajanı + piyasa gözlemcisi | 2 | Üç ajanlı yapı RL baseline'larını geçiyor | AQ1, G3; rol tabanlı öncül | Oku |
| 2 | Chen, J., Li, J., Wang, G. (2025). [MARS](https://arxiv.org/abs/2508.01173) | Ö | ABD hisseleri | Güvenlik eleştirmenli heterojen ajan topluluğu + meta denetleyici | 2 | Risk profilleri ayrışan topluluk rejimlere uyuyor | G5; rejim | Atıf yeter |
| 3 | Ren, X., Abudurexiti, N. vd. (2025). [SAMP-HDRL](https://arxiv.org/abs/2512.22895) | Ö | Hisse; 2019–2021, üç rejim | Dinamik gruplama + üst/alt ajan + SHAP | 3 | 9 geleneksel ve 9 DRL baseline'ını geçiyor | G3, G5, H | Oku |
| 4 | MPLS (2025). [Modular Portfolio Learning System](https://iieta.org/node/20741) ⚠ yazar | D (IIETA) | ⚠ | Sektör dağıtan üst ajan + sektör içi alt ajanlar | ⚠ | Hiyerarşik sinyal füzyonu | G3 için en yakın | Oku |
| 5 | IPS (2025). [Hierarchical MAS with BNNs](https://iieta.org/journals/mmep/paper/10.18280/mmep.120417) ⚠ yazar | D (IIETA MMEP) | ⚠ | Hiyerarşik çoklu ajan + Bayesçi sinir ağı + PPO | 2 | Belirsizliğe duyarlı karar | G2 (belirsizlik) | Atıf yeter |
| 6 | He vd. (2025). [Hierarchical AI Multi-Agent Fundamental Investing](https://arxiv.org/abs/2510.21147) | Ö | Çin A-hisse | LLM ajanları + RL portföy ajanı | 2 | Makro ajan sektörleri ağırlıklandırıyor | G3, J | Atıf yeter |
| 7 | Fang, Y., Tang, Z., Ren, K. vd. (2023). [Intention-Aware Communication](https://arxiv.org/abs/2307.03119) | Ö (yayın yeri ⚠) | Çin A-hisse, ABD | Çok turlu niyet iletişimi; CommNet, TarMAC karşılaştırması | 3 | Niyet iletişimi koordinasyonu iyileştiriyor | **G1 tehdidi** | Oku |
| 8 | Lee, J., Kim, R., Yi, S.-W., Kang, J. (2020). [MAPS](https://doi.org/10.24963/ijcai.2020/623) | K (IJCAI) | ABD, 12 yıl | Çeşitlilik kayıplı bağımsız yatırımcı ajanlar | 2 | Ajan sayısı arttıkça Sharpe artıyor | Temel; varlık başına ajan | Arka plan |
| 9 | Huang, Z., Tanaka, F. (2022). [MSPM](https://doi.org/10.1371/journal.pone.0263689) | D (PLOS ONE) | Hisse | Varlık başına modül + portföy modülü | 2 | Modüler, yeniden kullanılabilir yapı | Temel | Arka plan |
| 10 | Xue, P., Ye, Y. (2025). [Attention-Enhanced RL](https://arxiv.org/abs/2510.06466) | Ö | S&P 500, 2000–2025 | Dirichlet politika + kesitsel dikkat | 4 | Arındırılmış walk-forward ile EW'yi geçiyor | G, F; iyi uygulama örneği | Oku |
| 11 | Factor-MCLS (2025). [arXiv:2504.11874](https://arxiv.org/abs/2504.11874) ⚠ yazar | Ö | ⚠ | Ödül faktör matrisi + çok eleştirmen | ⚠ | Risk kısıtlı çok eleştirmenli öğrenme | Arka plan | Arka plan |

### 6.2 MARL iletişimi ve yorumlanabilirlik (Eksen B, H)

| # | Künye | Tür | Yöntem | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | Das, A., Gervet, T., Romoff, J. vd. (2019). [TarMAC](https://arxiv.org/abs/1810.11187) | K (ICML), temel | İmza tabanlı hedefli dikkat | Hedefli ve çok turlu iletişim | AQ1 dikkat kolu | Oku |
| 13 | Hu, S., Shen, L., Zhang, Y., Tao, D. (2024). [CommFormer](https://arxiv.org/abs/2405.08550) | K (ICLR) | İletişim grafiği + politika birlikte | TarMAC, MAGIC, CommNet'i geçiyor | AQ1, G1 (finans dışı) | Oku |
| 14 | Fan, W., Yu, Z., Ma, C. vd. (2025). [f-MAT](https://doi.org/10.1609/aaai.v39i16.33813) | K (AAAI) | Faktör tabanlı transformer iletişimi | Komşular arası verimli mesajlaşma | AQ1 | Atıf yeter |
| 15 | Liu, Z., Wan, L., Sui, X. vd. (2023). [DHCG](https://doi.org/10.24963/ijcai.2023/24) | K (IJCAI) | DAG iletişim grafiği | Döngüsüz graf ile yakınsama | B arka plan | Arka plan |
| 16 | Yuan, L., Wang, J., Zhang, F. vd. (2022). [MAIC](https://doi.org/10.1609/aaai.v36i9.21179) | K (AAAI) | Takım arkadaşı modeli + teşvik mesajı | Hedefli seyrek iletişim | B arka plan | Arka plan |
| 17 | ADMAC (2024). [Robust Communicative MARL](https://ojs.aaai.org/index.php/AAAI/article/view/29708) ⚠ yazar | K (AAAI) | Alıcı taraflı güvenilirlik kestiricisi | Güvenilmez mesajların etkisini azaltıyor | **G2 tehdidi** | Oku |
| 18 | Sun, Y., Zheng, R., Hassanzadeh, P. vd. (2023). [Certifiably Robust](https://arxiv.org/abs/2206.10158) | K (ICLR) | Budanmış mesaj topluluğu | Sertifikalı sağlamlık | G2 | Atıf yeter |
| 19 | IMMAC (2021). [AAMAS genişletilmiş özet](https://archive.illc.uva.nl/AAMAS-2021/pdfs/p1668.pdf) ⚠ yazar | K (AAMAS) | Belirsizlik/merak tabanlı dikkat | İçsel değerle mesaj ağırlığı | **G2 tehdidi** | Oku |
| 20 | DebUnc (2024). [arXiv:2407.06426](https://arxiv.org/abs/2407.06426) ⚠ yazar | Ö | LLM tartışmasında ters belirsizlikle dikkat ölçekleme | Güvenli ajana daha çok dikkat | **G2 tehdidi** (kavramsal) | Oku |
| 21 | Epistemik sürüleşme (2026). [arXiv:2605.06988](https://arxiv.org/abs/2605.06988) ⚠ yazar | Ö | Ters entropi ağırlıklı füzyon | Güven ağırlığı sürüleşmeye yol açabiliyor | G2 risk | Oku |
| 22 | Communication Methods in MARL (2026). [arXiv:2601.12886](https://arxiv.org/abs/2601.12886) ⚠ yazar | Ö (derleme) | Derleme | Dikkat tabanlı iletişim sınıflaması | B | Atıf yeter |
| 23 | Goal-Oriented Communication overview (2025). [arXiv:2508.07720](https://arxiv.org/abs/2508.07720) ⚠ yazar | Ö (derleme) | Derleme | Dikkat tabanlı mesaj önceliklendirme | B | Arka plan |
| 24 | Deng, Z., Liang, F., Zhang, Y. vd. (2026). [SOPS](https://icml.cc/virtual/2026/poster/65907) | K (ICML) | Üstel graf omurgası + seyrek bağlantı | Büyük ölçekte ölçeklenebilir iletişim | B | Arka plan |
| 25 | ExpoComm (2025). [arXiv:2502.19717](https://arxiv.org/abs/2502.19717) ⚠ yazar | Ö | Üstel topoloji | Sıfır-atış aktarım | B | Arka plan |
| 26 | Sun, C., He, P., Ji, Q. vd. (2026). [M2I2](https://doi.org/10.1609/aaai.v40i30.39764) | K (AAAI) | Maskeli durum modelleme + niyet çıkarımı | Verimli iletişim | B | Arka plan |
| 27 | Huo, D., Zhang, H., Hao, Y. vd. (2024). [DCMAC](https://arxiv.org/abs/2409.07127) | Ö | Talebe duyarlı çapraz dikkat | Kısıtlı bantta iyi performans | B | Arka plan |
| 28 | MAGIC (2021). [AAMAS](https://archive.illc.uva.nl/AAMAS-2021/pdfs/p964.pdf) ⚠ yazar | K (AAMAS), temel | Zamanlayıcı + GAT mesaj işleyici | Graf dikkatli iletişim | AQ1 GAT kolu | Oku |
| 29 | MUTE (2026). [arXiv:2607.03473](https://www.emergentmind.com/papers/2607.03473) ⚠ yazar | Ö | Karşı-olgusal mesaj değeri | Mesaj katkısının ölçümü | H, iletişim açıklanabilirliği | Atıf yeter |
| 30 | Jain, S., Wallace, B. (2019). Attention is not Explanation | K (NAACL), temel | Dikkat eleştirisi | Dikkat ≠ açıklama | H | Atıf yeter |
| 31 | Wiegreffe, S., Pinter, Y. (2019). [Attention is not not Explanation](https://preview.aclanthology.org/menus/D19-1002/) | K (EMNLP-IJCNLP), temel | Dört tanı testi | Tek biçimli baseline ve tohum varyansı önerisi | H; doğrudan uygulanabilir | Oku |

### 6.3 Tahmin-destekli RL ve tahmin katmanı (Eksen C, D)

| # | Künye | Tür | Yöntem | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | Gao, S., Wang, Y., Yang, X. (2023). StockFormer | K (IJCAI) | Öngörü kodlama + aktör-eleştirmen | Tahmin modülü RL'i destekliyor | C; gözlem yolu | Oku |
| 33 | DeepClair (2024). [arXiv:2407.13427](https://arxiv.org/abs/2407.13427) ⚠ yazar | Ö | Tahminci → RL, LoRA aktarımı | Ablation ile tahmin katkısı | C, AQ2 | Atıf yeter |
| 34 | RTS-PnO (2025). [arXiv:2505.24835](https://arxiv.org/abs/2505.24835) ⚠ yazar | Ö | Uyarlanabilir belirsizlik kısıtı | Sabit kısıttan iyi | C, G2 | Oku |
| 35 | Zhang, Z., Zhao, T., Zomaya, A. (2025). [OPS with ML Predictions](https://papers.neurips.cc/paper_files/paper/2025/hash/76981cb7a06b5ad53639f5bbe9924388-Abstract-Conference.html) | K (NeurIPS) | Tahmin destekli çevrimiçi portföy | Hatalı tahminlere karşı alt sınır | C; teorik dayanak | Atıf yeter |
| 36 | Li, X. vd. (2019). [Optimistic Bull or Pessimistic Bear](https://arxiv.org/abs/1907.01503) | Ö, temel | Tahmin hatasına göre uyarlanan DDPG | Tahmin hatası politika eğilimini değiştiriyor | C | Arka plan |
| 37 | Kazemdehbashi, S. (2026). [TATS](https://arxiv.org/abs/2601.12706) | Ö | Trend sınıflandırıcısı + LSTM düzeltmesi | Altın fiyatında LSTM'den düşük hata | **G6: TATS özgün kaynak** | Oku |
| 38 | Forecast Evaluation Pitfalls (2022). [arXiv:2203.10716](https://arxiv.org/abs/2203.10716) ⚠ yazar | Ö (dergi sürümü ⚠) | Rehber | Bütün seriye ayrıştırma sızıntıdır | D, Makale 2 | Oku |
| 39 | ML Pitfalls rehberi (2021). [arXiv:2108.02497](https://arxiv.org/abs/2108.02497) ⚠ yazar | Ö | Rehber | Bölme öncesi ön işleme ileriye bakma yaratır | D | Atıf yeter |
| 40 | Zhang, D., Jiang, K., Wu, C. vd. (2025). [Revisiting Signal Decomposition](https://doi.org/10.1117/12.3045386) | K (SPIE) | Nedensel ayrıştırma testi | Sızıntı yaygın; nedensel işlemede katkı azalıyor | **G6, Makale 2** | Oku |
| 41 | EMD yanlılığı (2017). [Bias effect on predicting market trends with EMD](https://educapes.capes.gov.br/handle/capes/1017359?mode=full) ⚠ yazar, yayın yeri | D ⚠ | EEMD sızıntı testi | Sızıntı düzeltilince kazanç kayboluyor | G6 (finans) | Oku |
| 42 | RVFL walk-forward (2021). [arXiv:2107.14385](https://arxiv.org/abs/2107.14385) ⚠ yazar | Ö | İleri yürüyen ayrıştırma | Sızıntı mekanizması ve çözümleri | Makale 2 yöntemi | Oku |
| 43 | Multi-Front yöntemi (2023). [arXiv:2303.06394](https://arxiv.org/abs/2303.06394) ⚠ yazar | Ö | EWT + MF | Sızıntıyı önleyen ayrıştırma | Makale 2 | Atıf yeter |
| 44 | Trend-encoded probabilistic model (2025). [arXiv:2502.08144](https://arxiv.org/abs/2502.08144) ⚠ yazar | Ö | İnceleme | Ayrıştırma-topluluk yöntemlerinde sızıntı | D | Arka plan |
| 45 | Torres, M. E., Colominas, M. A., Schlotthauer, G., Flandrin, P. (2011). CEEMDAN | K (ICASSP), temel | CEEMDAN | — | D | Atıf yeter |
| 46 | Colominas, M. A., Schlotthauer, G., Torres, M. E. (2014). [ICEEMDAN](https://github.com/macolominas/CEEMDAN) | D (BSPC 14:19–29), temel | ICEEMDAN | — | D | Atıf yeter |
| 47 | Laszuk, D. [PyEMD belgesi](https://pyemd.readthedocs.io/en/latest/ceemdan.html) | Y | CEEMDAN (Colominas iyileştirmeleriyle) | Belirlenimci değil; tohum gerekli | D, tekrarlanabilirlik | Atıf yeter |
| 48 | Re(Visiting) TSFMs in Finance (2025). [arXiv:2511.18578](https://arxiv.org/abs/2511.18578) ⚠ yazar | Ö | TSFM vs GBM, 94 ülke | Sıfır-atış TSFM'ler GBM'in gerisinde | D | Oku |
| 49 | Noguer i Alonso, M., Pereira Franklin, R. (2026). [Pretrained TSFMs](https://arxiv.org/abs/2606.27100) | Ö | TSFM vs sıfırdan modeller, DM testi | Kazançlar küçük ve seyrek | D | Atıf yeter |
| 50 | Łaniewski, Ślepaczuk (2025). [Chronos daily index](https://doi.org/10.62036/ISD.2025.48) | K (ISD) | Chronos, kayan pencere | Rekabetçi isabet, orta getiri | D | Atıf yeter |
| 51 | Widener, M. vd. (2025). [arXiv:2508.20437](https://arxiv.org/abs/2508.20437) | Ö | ARIMA, GBM, Chronos, Llama | Oynak alanlarda GBM üstün | D | Arka plan |

### 6.4 Rejim, ödül ve maliyet (Eksen E, F)

| # | Künye | Tür | Piyasa & veri | Yöntem | Puan | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 52 | Nixon Raj, G. (2025). [Regime-Aware RL](https://arxiv.org/abs/2509.14385) | Ö | Stilize ortam + makro rejim | HMM olasılıkları gözlemde | 1 | Rejim bağlamı istikrarı artırıyor | E, AQ3 | Atıf yeter |
| 53 | Verma, A. K., Putri, N. S., Lesupi, N. P. (2026). [HMM + RL](https://arxiv.org/abs/2605.27848) | Ö | SPY/TLT/GLD 2004–2025 | 3 durumlu Gauss HMM (BIC) + tablosal RL | 2 | RL en yüksek Sharpe, düşük düşüş | E; rejim tanımı | Oku |
| 54 | Ilieva, T. B. (2026). [Regime-Aware Pairs Trading](https://repository.tudelft.nl/record/uuid:4f4f2e2e-ca16-4fd1-91a6-b17a9580ed73) | T (lisans) | V/MA, V/JKHY | HMM + PPO, 10 tohum | 3 | Rejime duyarlı ajan tutarlı biçimde üstün | E, G (tohum disiplini) | Atıf yeter |
| 55 | MTX çalışması (2026). [IJFS 14(3):67](https://www.mdpi.com/2227-7072/14/3/67) ⚠ yazar | D (MDPI) | Tayvan vadeli | Boğa/ayı uzmanı iki ajan + rejim seçici | ⚠ | Rejim ayrıştırmalı ajan | E | Atıf yeter |
| 56 | Moody, J., Saffell, M. (2001). Learning to trade via direct reinforcement | D (IEEE TNN 12(4):875–889), temel | — | Doğrudan pekiştirme, diferansiyel Sharpe | — | — | F temeli | Atıf yeter |
| 57 | Global equity DRL (2026). [arXiv:2605.17307](https://arxiv.org/html/2605.17307v1) ⚠ yazar | Ö | Küresel hisse | Diferansiyel Sharpe + 2 bps + dönüşüm cezası | 3 | Sürtünmeli ödül tasarımı | F | Oku |
| 58 | PPO-HRAP (2026). [arXiv:2610.01325](https://arxiv.org/pdf/2610.01325) ⚠ yazar | Ö | SPY | Düşüş + hedef sapma + dönüşüm ödülü, 5 tohum | 4 | Sharpe 0,62 ± 0,06 | **F; ablation ödülünüzün öncülü** | Oku |
| 59 | Smart Tangency Portfolio (2025). [IJFS 13(4):227](https://www.mdpi.com/2227-7072/13/4/227) ⚠ yazar | D (MDPI) | 12 ABD varlığı, DJIA | DRL + fayda topluluğu | 4 | HAC alfa ve bootstrap Sharpe testi anlamlı | F, G | Atıf yeter |
| 60 | Zhang, Z., Zohren, S., Roberts, S. (2020). [Deep RL for Trading](https://arxiv.org/abs/1911.10107) | D (JFDS ⚠) | Vadeli işlemler | DQN, PG, A2C | 3 | Maliyet oranına göre Sharpe duyarlılığı | F, G | Atıf yeter |
| 61 | Chow, Y., Ghavamzadeh, M. (2014). [CVaR in MDPs](https://arxiv.org/pdf/1406.3339v3.pdf) | K (NeurIPS), temel | — | Ortalama-CVaR PG ve AC | — | — | F | Arka plan |
| 62 | Chow, Y. vd. (2018). [Percentile Risk Criteria](https://arxiv.org/pdf/1512.01629) | D (JMLR ⚠), temel | — | CVaR kısıtlı AC | — | — | F | Arka plan |
| 63 | Chang, M. (2026). [Recursive Utility RL](https://arxiv.org/abs/2603.22880) | Ö | Kore ETF'leri, 10 kronolojik bölme | Özyinelemeli fayda + PPO/A2C | 2 | Naif baseline'ı geçiyor | F; gelişmekte olan piyasa | Arka plan |

### 6.5 Değerlendirme metodolojisi ve XAI (Eksen G, H)

| # | Künye | Tür | Yöntem | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. G. (2021). [Statistical Precipice](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) | K (NeurIPS), temel | IQM, bootstrap CI, performans profili | Az koşulda nokta tahmini yanıltıcı | G; tohum protokolü | Oku |
| 65 | Bailey, D. H., López de Prado, M. (2014). [Deflated Sharpe Ratio](https://jpm.iijournals.com/highwire/citation/9993/refworks-tagged) | D (JPM 40(5):94–107), temel | DSR | Seçim yanlılığını düzeltir | G | Oku |
| 66 | Bailey, D. H., Borwein, J., López de Prado, M., Zhu, Q. J. (2017). [PBO](https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting) | D (J. Comput. Finance 20(4)), temel | CSCV | Aşırı uyum olasılığı | G | Atıf yeter |
| 67 | Hansen, P. R., Lunde, A., Nason, J. M. (2011). [Model Confidence Set](https://doi.org/10.3982/ECTA5771) | D (Econometrica 79(2):453–497), temel | MCS | En iyi model kümesi | G | Atıf yeter |
| 68 | Ledoit, O., Wolf, M. (2008). [Sharpe testi](https://ideas.repec.org/a/eee/empfin/v15y2008i5p850-859.html) | D (JEF 15:850–859), temel | HAC/bootstrap Sharpe farkı | İki strateji karşılaştırması | G | Oku |
| 69 | PPO tohum duyarlılığı tanısı (2026 ⚠). [Research Square](https://assets-eu.researchsquare.com/files/rs-9824202/v1_covered_2dbf3593-a58c-4edb-bb2a-bb85e54443cb.pdf) ⚠ yazar | Ö | Çok tohumlu tanı | PPO getirisi %48,2 ± %51,0; EW %33,6 | **G; 3 tohum eleştirisi** | Oku |
| 70 | Aarnio, J. (2025). [DRL vs 1/N](https://www.utupub.fi/bitstreams/7d12624f-27e2-4448-99c9-2520ec3009f3/download) | T (YL, Turku) | DRL vs 1/N, 2020–2024 OOS | Anlamlı üstünlük yok | G | Atıf yeter |
| 71 | Benchmarking Robustness of DRL OPM (2023). [HAL](https://hal-universite-paris-saclay.archives-ouvertes.fr/LISN/hal-04473989) ⚠ yazar | K ⚠ | Sağlamlık değerlendirmesi | DRL algoritmaları hızla bozuluyor | G | Atıf yeter |
| 72 | Multimodal DRL for Portfolio (2024). [arXiv:2412.17293](https://arxiv.org/abs/2412.17293) ⚠ yazar | Ö | %1 maliyetli kıyas | Eşit ağırlıklı al-tut en iyi | G; baseline zorunluluğu | Atıf yeter |
| 73 | RL in Financial Decision Making (2025). [arXiv:2512.10913](https://arxiv.org/abs/2512.10913) ⚠ yazar | Ö (derleme) | 167 makalelik sistematik derleme | Alan bazında performans örüntüleri | G, arka plan | Atıf yeter |
| 74 | Guan, M., Liu, X.-Y. (2021). [Explainable DRL for PM](https://dl.acm.org/doi/10.1145/3490354.3494415) | K (ICAIF) | Entegre gradyan | DRL çok adımlı öngörü gücü gösteriyor | H | Atıf yeter |

### 6.6 BIST ve Türkiye (Eksen I)

| # | Künye | Tür | Piyasa & veri | Yöntem | Puan | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75 | Candar, M. (2022). [BIST30 RL portföy](https://polen.itu.edu.tr/items/88affa11-5549-4dbe-8db7-20e02bf507e3) | T (YL, İTÜ) | BIST30; 10 yıl eğitim, 1 yıl test | DDPG, PPO; temel veri | 2 | BIST30 endeksine karşı ağırlıklandırma | **G4; tek-ajan öncülü** | Oku |
| 76 | Eğe, Ş. (2024). [BIST RL + duyarlılık](https://polen.itu.edu.tr/items/951baa9f-14a3-4f7f-a0ba-fa851e866a8a) | T (YL, İTÜ) | 8 hisse, 2021–2024 | DQN, DDQN, D3QN | 1 | Hisseye göre değişen performans | G4 | Atıf yeter |
| 77 | TJEECS (2022). [DRL + bilgi grafiği](https://journals.tubitak.gov.tr/elektrik/vol30/iss4/22) ⚠ yazar | D (TJEECS 30(4):1506–1524) | GARAN, AKBNK, ISCTR | DQN + BerTurk + sosyal graf | 1 | Duyarlılık destekli DQN | G4 | Atıf yeter |
| 78 | ASYU (2025). [RL + makro](https://doi.org/10.1109/ASYU67174.2025.11208408) ⚠ yazar | K (IEEE ASYU) | TTRAK, 2024 | Q-öğrenme, SARSA + Türk makrosu | 0 | Makro göstergeler kararı etkiliyor | G4; Türkiye etkenleri | Atıf yeter |
| 79 | Borsa Istanbul Review (2026). [Inverse RL on brokers](https://research.itu.edu.tr/en/publications/inferring-latent-trading-motivations-in-borsa-istanbul-a-comparat/) ⚠ yazar | D (BIR 26(4):100834) | 2.506 işlem günü | Doğrusal ve maks. entropi IRL | 2 | Yerli ve yabancı aracıların ödülleri ilişkisiz | G4; BIST'te RL | Atıf yeter |
| 80 | Erdağ, A., Uçar, M. F., Tarhan, İ. E. (2026). [BIST100 ML/DL](https://avesis.iku.edu.tr/yayin/e7569442-c9a7-4ca1-b41b-90a611ad469f/structural-drivers-and-forecasting-dynamics-of-the-bist100-index-a-hybrid-ml-dl-approach-integrating-liquidity-sentiment-and-macroeconomic-factors) | D (Rev. Behav. Finance; ESCI) | BIST100, 2010–2022 | ML/DL + makro, likidite, duyarlılık | — | Temeller, duyarlılık ve likidite birlikte belirleyici | I; Türkiye etkenleri | Atıf yeter |
| 81 | Ergenç, C., Aktaş, R. (2025). [BIST100 supervised ML](https://reference-global.com/article/10.2478/revecp-2025-0005) | D (Rev. Econ. Perspectives) | BIST100, 2014–2024 | XGBoost, RF, CNN, LSTM | — | Tek model tutarlı biçimde üstün değil | I | Arka plan |
| 82 | Ayyıldız, N. (2025). [E-7 yön tahmini](https://ideas.repec.org/a/acc/malfin/v40y2025i123p92-111.html) | D (J. Finance Letters) | 7 gelişmekte olan endeks | H2O derin öğrenme, %80/20 | — | BIST100'de %77,29 isabet (sızıntı şüphesi) | I; kalite eleştirisi | Arka plan |

### 6.7 LLM ajanları ve araçlar (Eksen J, K)

| # | Künye | Tür | Ana bulgu | Teze ilişkisi | Öncelik |
| --- | --- | --- | --- | --- | --- |
| 83 | TradingAgents (2024/2025). [arXiv:2412.20138](https://arxiv.org/abs/2412.20138) ⚠ yazar | Ö; AAAI 2025 MARW | Rol tabanlı LLM ajanları baseline'ları geçiyor | J; konumlandırma | Atıf yeter |
| 84 | Shi, J., Hollifield, B. (2025). [Market-Dependent Communication](https://arxiv.org/abs/2511.13614) | K (NeurIPS 2025 çalıştayı) | İletişimin değeri piyasa türüne bağlı; kod açık | **G5 tehdidi** | Oku |
| 85 | LLM Agent in Financial Trading: A Survey (2024). [arXiv:2408.06361](https://arxiv.org/abs/2408.06361) ⚠ yazar | Ö (derleme) | 27 makalelik derleme | J | Atıf yeter |
| 86 | Integrating LLMs in Financial Investments (2025). [arXiv:2507.01990](https://arxiv.org/abs/2507.01990) ⚠ yazar | Ö (derleme) | FinCon ve TradingAgents özetleri | J | Arka plan |
| 87 | Agentic AI in Finance (2026). [arXiv:2604.21672](https://arxiv.org/abs/2604.21672) ⚠ yazar | Ö (derleme) | Fang vd.'yi MARL iletişim örneği olarak ele alıyor | J, G1 | Arka plan |
| 88 | PettingZoo (2021). [arXiv:2009.14471](https://arxiv.org/abs/2009.14471) ⚠ yazar | K (NeurIPS D&B ⚠) | AEC ve Parallel API | K; ortam arayüzü | Atıf yeter |
| 89 | BenchMARL (2024). [JMLR 25](https://www.jmlr.org/papers/volume25/23-1612/23-1612.pdf) ⚠ yazar | D (JMLR) | TorchRL tabanlı standart MARL kıyaslama | K; alternatif çerçeve | Atıf yeter |
| 90 | RLlib RLModule belgesi. [Ray 2.59.0](https://docs.ray.io/en/latest/rllib/rl-modules.html) | Y | Paylaşımlı iletişim için MultiRLModule | K; uygulama | Atıf yeter |

## 7. Değerlendirme protokolü için somut öneriler

En önemli üç değişiklik: tohum sayısını artırmak, testleri soruya göre seçmek ve ayrıştırmayı sızıntısız yapmak.

### 7.1 Tohum ve raporlama

1. **Nihai yapılandırmalar için en az 10 tohum** kullanın; ön elemede 3–5 tohum yeterli. Tohumlar arası dağılımı IQM, katmanlı bootstrap güven aralığı ve iyileşme olasılığıyla raporlayın (Agarwal vd., 2021). PPO'da tohum varyansının ortalamaya yakın çıkabildiği gösterildi (matris #69); 10 tohumlu rejim çalışması örnek (#54).
2. **Tek sayı değil, dağılım raporlayın:** her yapılandırma için Sharpe'ın medyanı, IQM'si ve %95 aralığı.
3. **Toplam deneme sayısını kaydedin** (Optuna denemeleri × yapılandırma × tohum). Deflated Sharpe bu sayıya ihtiyaç duyar.

### 7.2 Baseline'lar

1. **Zorunlu basit baseline'lar:** eşit ağırlık (aylık yeniden dengeleme), al-tut, piyasa değeri ağırlıklı BIST-30 (XU030), minimum varyans (Ledoit-Wolf daraltmalı), risk paritesi / HRP (Millea & Edalat, 2022).
2. **Zorunlu öğrenen baseline'lar:** tek-ajan PPO (mevcut baseline), IPPO ve MAPPO (iletişimsiz), hiyerarşik sektör dağıtıcı (MPLS tipi, G3 için).
3. **G2 ablation baseline'ları:** güvensiz standart dikkat, öğrenilmiş güvenilirlik (ADMAC tipi), ters belirsizlikle sabit ölçekleme (DebUnc tipi), karıştırılmış güven (güven sinyalinin bilgi taşıyıp taşımadığını sınar).
4. **Önceki çalışmanın öncülü:** Candar (2022) kurulumuna benzer tek-ajan BIST30 PPO.

### 7.3 İstatistiksel testler (soruya göre)

| Soru | Test | Kaynak |
| --- | --- | --- |
| Tahmin modeli A, B'den isabetli mi? | Diebold-Mariano (HAC varyans) | Tahmin katmanı |
| Strateji A'nın Sharpe'ı B'den yüksek mi? | Ledoit-Wolf studentize bootstrap | Ledoit & Wolf (2008) |
| Çok sayıda yapılandırmadan hangileri en iyi? | Model Confidence Set | Hansen vd. (2011) |
| Seçilen Sharpe şans eseri mi? | Deflated Sharpe Ratio | Bailey & López de Prado (2014) |
| Backtest aşırı uyumlu mu? | PBO / CSCV | Bailey vd. (2017) |
| Tohumlar arası fark anlamlı mı? | Bootstrap CI + iyileşme olasılığı; Wilcoxon yalnız eşleştirilmiş tohumlarda | Agarwal vd. (2021) |
| Çok sayıda karşılaştırma | Benjamini-Hochberg FDR | Ayari (2025) uygulaması |

t-testi, getirilerin otokorelasyonu ve kalın kuyrukları nedeniyle tek başına yetersiz; bootstrap tabanlı testlerle destekleyin.

### 7.4 Walk-forward ve sızıntı

1. **Ayrıştırmayı yalnızca geçmişle yapın.** Her tahmin noktasında son W günlük pencereye ICEEMDAN uygulayın; yalnızca son noktanın bileşenlerini özellik olarak kullanın. Kenar etkisi nedeniyle son noktadaki bileşenler kararsızdır; bunu ayrı bir duyarlılık analiziyle raporlayın.
2. **PyEMD tohumunu sabitleyin** (`noise_seed()`) ve ayrıştırma tohumunu RL tohumundan bağımsız tutun.
3. **Makale 2 ablation'ı:** sızıntılı (tüm seri) ve sızıntısız (ileri yürüyen) ayrıştırmayı aynı boru hattında karşılaştırın; TATS'lı ve TATS'sız koşulları da ekleyin.
4. **Purge ve embargo:** 5 gün purge, 3 gün embargo makul; tahmin ufku ve özellik pencereleri uzunsa bunları ufukla orantılı büyütün. Hiperparametre seçimi yalnızca iç doğrulama katlarında yapılmalı.
5. **Evren ve veri sızıntısı (benim değerlendirmem):** tarihsel BIST-30 bileşenleri; temel oranlarda açıklanma tarihine göre hizalama; TÜFE/ÜFE için açıklanma gecikmesi.

### 7.5 Rejim tanımı

1. **Değerlendirme etiketi:** önceden ve kural tabanlı tanımlı, sızıntısız. Önerim: XU100'ün 200 günlük hareketli ortalamasına göre yön (boğa/ayı) ve 60 günlük gerçekleşmiş oynaklığın alt tercili + düşük trend eğimiyle "yatay".
2. **Model girdisi:** HMM'nin yalnızca filtrelenmiş olasılıkları; durum sayısı BIC ile seçilsin (Verma vd., 2026).
3. **Sağlamlık:** sonuçları iki farklı rejim tanımıyla tekrarlayın; iletişim katkısının işareti değişmiyorsa G5 bulgusu güçlenir.
4. **Örneklem uyarısı:** BIST'te ayı dönemleri kısa olabilir; rejim başına gün sayısını ve bağımsız dönem sayısını raporlayın.

### 7.6 İşlem maliyeti ve getiri tanımı

1. **Maliyet seviyesi:** ana senaryo komisyon + vergi + yayılma için gidiş-dönüş yaklaşık 20–30 baz puan (benim varsayımım; aracı kurum tarifesiyle belgeleyin); duyarlılık için 0, 10, 25, 50 baz puan (Zhang, Zohren & Roberts, 2020 örneği).
2. **Getiri birimi:** nominal TL, TÜFE ile reel TL ve USD bazlı sonuçları birlikte raporlayın.
3. **Risksiz faiz:** TCMB politika faizi ya da TLREF; Sharpe'ı sıfır faizle değil, bu faizle hesaplayın.
4. **Dönüşüm ve maliyet sürüklenmesi:** yıllık dönüşüm oranını ve maliyet sürüklenmesini ayrı raporlayın (Smart Tangency örneği, #59).

### 7.7 Yorumlanabilirlik

1. Dikkat ağırlıklarını yorumlarken tek biçimli dikkat baseline'ı ve tohumlar arası dikkat varyansını raporlayın (Wiegreffe & Pinter, 2019).
2. Mesaj katkısı için karşı-olgusal maskeleme testi ekleyin (MUTE yaklaşımı): bir sektör ajanının mesajı sıfırlandığında portföy Sharpe'ı ne kadar düşüyor?
3. SHAP'ı tahmin katmanında ve politika girdilerinde ayrı ayrı raporlayın.

## 8. Araç durumu tablosu (Eksen K)

Mevcut yol (PettingZoo + RLlib) güncel ve bakımda; özel iletişim modülü RLlib'de `MultiRLModule` alt sınıfıyla yazılıyor. İletişim mimarisi karşılaştırması için BenchMARL güçlü bir yedek. Sürüm bilgileri resmi sayfalardan 6 Ekim 2026'da alındı; "⚠" olanlar bu taramada doğrulanmadı.

| Araç | Son görülen sürüm / tarih | Bakım | IPPO / MAPPO | Özel iletişim modülü | Kaynak |
| --- | --- | --- | --- | --- | --- |
| PettingZoo | 1.27.0, 13 Ağustos (yıl sayfada yok; 1.26.x Nisan 2026'dan sonra) | Aktif | Ortam API'si; algoritma yok | Ortam tarafında gözlem/mesaj alanı tanımlanır | [Sürüm notları](https://github.com/Farama-Foundation/PettingZoo/releases) |
| RLlib (Ray) | Belge sürümü Ray 2.59.0 | Aktif; yeni API yığını varsayılan | PPO çok ajanlı; IPPO bağımsız politikalarla, MAPPO merkezi eleştirmen için özel modül gerekir | `MultiRLModule` alt sınıfı; ajanlar arası paylaşımlı iletişim için önerilen yol | [RLModule belgesi](https://docs.ray.io/en/latest/rllib/rl-modules.html) |
| BenchMARL | 1.5.2 (TorchRL 0.11 ile eşli) | Aktif | MAPPO ve IPPO hazır | Hazır GNN modeli; özel model arayüzü | [Sürümler](https://github.com/facebookresearch/BenchMARL/releases), [JMLR makalesi](https://www.jmlr.org/papers/volume25/23-1612/23-1612.pdf) |
| TorchRL | Gece sürümleri Temmuz 2026'da yayımlanıyor; kararlı sürüm ⚠ | Aktif | MAPPO, IPPO, MultiAgentGAE | TensorDict tabanlı modüllerle esnek | [PyPI](https://pypi.org/project/torchrl-nightly/2026.7.20/) |
| JaxMARL | ⚠ | ⚠ (2026 kaynak listesinde yer alıyor) | Baseline algoritmalar JAX ile | JAX'ta özel ağ yazımı gerekir | [Kaynak listesi](https://github.com/TimeBreaker/MARL-resources-collection) |
| EPyMARL | ⚠ | ⚠ | IPPO, MAPPO ve diğerleri | PyMARL tarzı modüler yapı | [Kaynak listesi](https://github.com/TimeBreaker/MARL-resources-collection) |
| MARLlib | ⚠ | ⚠ | ⚠ | ⚠ | Bu taramada kontrol edilmedi |
| FinRL / FinRL-Meta | ⚠ | ⚠ | Tek ajan odaklı | — | Bu taramada kontrol edilmedi |
| PyTorch Geometric | ⚠ | ⚠ | — | GATConv katmanı GAT kolu için | Bu taramada kontrol edilmedi |

**Öneri.** PettingZoo Parallel API + RLlib yeni yığın yolunu sürdürün; üç iletişim kolunu tek bir `MultiRLModule` içinde, iletişim katmanı değiştirilebilir biçimde (yok / dikkat / GAT) yazın. PettingZoo 1.27.0 eski ortam oluşturma API'sini kullanımdan kaldırdığı için yeni kayıt (registry) arayüzünü kullanın. RLlib'de takılırsanız BenchMARL + TorchRL, hazır MAPPO/IPPO ve GNN modeliyle en hızlı yedek yoldur.

## 9. Okuma listesi: önce okunacak 15 kaynak

Sıra, savunma riskine göre: önce özgünlüğü tehdit edenler, sonra yöntemi değiştirecek olanlar.

1. [Zhang (2025), Scientific Reports](https://doi.org/10.1038/s41598-025-32408-w) — G2, G3, G5 için en yakın finans çalışması; güven ağırlıklı konsensüsünü ve kaynakça sorunlarını tam metinden görmeniz gerekiyor.
2. [Fang vd. (2023)](https://arxiv.org/abs/2307.03119) — finansta iletişim mimarilerini aynı düzenekte karşılaştıran tek çalışma; G1'in sınırını belirliyor.
3. [ADMAC (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29708) — alıcı tarafında öğrenilen mesaj güvenilirliği; G2 ablation'ındaki zorunlu karşılaştırma.
4. [DebUnc (2024)](https://arxiv.org/abs/2407.06426) — ters belirsizlikle dikkat ölçekleme; mekanizmanız kavramsal olarak buna çok yakın.
5. [Epistemik sürüleşme (2026)](https://arxiv.org/abs/2605.06988) — güven ağırlıklı birleştirmenin başarısızlık biçimi; tasarımınıza kırpma/geçitleme eklemenin gerekçesi.
6. [Shi & Hollifield (2025)](https://arxiv.org/abs/2511.13614) — iletişim değerinin piyasaya bağlılığı; G5'in motivasyonu ve sınırı.
7. [Kazemdehbashi (2026), TATS](https://arxiv.org/abs/2601.12706) — TATS'ın özgün kaynağı; Makale 2'nin katkı iddiasını yeniden yazmanız gerekiyor.
8. [Zhang D. vd. (2025), SPIE](https://doi.org/10.1117/12.3045386) — ayrıştırma sızıntısının etkisi; Makale 2'nin ana sorusunu kurar.
9. [Forecast Evaluation Pitfalls (2022)](https://arxiv.org/abs/2203.10716) — sızıntının kaynakla gösterilmesi ve tahmin değerlendirme rehberi.
10. [Agarwal vd. (2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf) — tohum sayısı, IQM ve bootstrap raporlaması; değerlendirme protokolünün omurgası.
11. [Bailey & López de Prado (2014), DSR](https://jpm.iijournals.com/highwire/citation/9993/refworks-tagged) — çok sayıda yapılandırma denediğiniz için Sharpe'ı düzeltmek zorundasınız.
12. [Candar (2022), İTÜ tezi](https://polen.itu.edu.tr/items/88affa11-5549-4dbe-8db7-20e02bf507e3) — BIST30'daki tek-ajan öncülünüz; jüri bunu bilebilir.
13. [MPLS (2025)](https://iieta.org/node/20741) — hiyerarşik sektör dağıtıcının en yakın örneği; G3 baseline'ı.
14. [PPO-HRAP (2026)](https://arxiv.org/pdf/2610.01325) — hedef sapma + dönüşüm cezalı ödül; ablation ödülünüzün öncülü.
15. [Wiegreffe & Pinter (2019)](https://preview.aclanthology.org/menus/D19-1002/) — dikkat ağırlıklarını yorumlanabilirlik kanıtı olarak sunmadan önce yapmanız gereken testler.

## 10. Tez 2. bölüm iskeleti

Numaralar: `#n` = 6. bölümdeki matris satırı; `T n` = 3.1 tablosu (önceki kaynakça); `V n` = 3.2 tablosu (vizyon belgesi). Bulunamayan V7, V14, V16, V17 hiçbir yerde kullanılmamalı.

| Alt başlık | İçerik | Atanan kaynaklar |
| --- | --- | --- |
| 2.1 Tarama yöntemi | Veritabanları, sorgular, ölçütler, PRISMA akışı (0. bölüm) | — |
| 2.2 Finansta pekiştirmeli öğrenmenin temelleri | MDP formülasyonu, aktör-eleştirmen algoritmaları, derlemeler | T4, V12, #56, #73, T12, T13 |
| 2.3 Tek ajanlı DRL ile portföy yönetimi | Durum uzayı zenginleştirme, temel ve makro veriler, risk odaklı ödüller | T1, T2, T3, T7, T9, T11, T14, #10, #60 |
| 2.4 Çok-ajanlı RL ile portföy yönetimi | Varlık, rol, hiyerarşi ve sektör granülaritesi; ödül paylaşımı ve CTDE | V1, V2, V3, V5, V6, V8, #1–#9, #11 |
| 2.5 MARL'de öğrenilmiş iletişim | CommNet'ten transformer/graf iletişimine; ölçeklenebilirlik | #12–#16, #22–#28, V9, V10, V11, #7 |
| 2.6 Belirsizlik ve güvenle ağırlıklı iletişim | Güvenilirlik, belirsizlik tabanlı dikkat, sürüleşme riski (G2'nin konumlandığı yer) | #17–#21, #5, V2 |
| 2.7 Tahmin-destekli RL | Tahminin gözlem, kısıt ve iletişim yoluyla RL'e girişi | #32–#36, T5 |
| 2.8 Tahmin katmanı: topluluk, ayrıştırma ve sızıntı | Stacking, ICEEMDAN, TATS, ayrıştırma sızıntısı, temel modeller | #37–#51 |
| 2.9 Piyasa rejimleri ve durağan olmama | Rejim tanımları, rejime duyarlı RL, rejim bazlı raporlama | #52–#55, #2, #3 |
| 2.10 Ödül tasarımı, risk ve işlem maliyeti | Diferansiyel Sharpe, CVaR, dönüşüm ve hedef sapma cezaları | #56–#63, T3, T7 |
| 2.11 Değerlendirme metodolojisi ve backtest güvenilirliği | Tohum, baseline'lar, DSR, PBO, MCS, Sharpe testleri | #64–#73, T3 |
| 2.12 Açıklanabilirlik | SHAP/LIME, dikkatin açıklama değeri, mesaj katkısı | T6, T8, #74, #29–#31, #3 |
| 2.13 Borsa İstanbul ve gelişmekte olan piyasalar | BIST'te RL ve ML çalışmaları, Türkiye'ye özgü etkenler, diğer gelişmekte olan piyasalar | #75–#82, V15, T13, #63 |
| 2.14 LLM tabanlı çok-ajanlı sistemler (konumlandırma) | Doğal dil iletişimi vs öğrenilmiş sayısal iletişim | #83–#87, V5 |
| 2.15 Araştırma boşlukları ve tezin konumu | G1–G6'nın daraltılmış ifadeleri (5. bölüm) | 5. bölümdeki en yakın çalışmalar |

Araç ve kütüphaneler (#88–#90, PettingZoo, RLlib, BenchMARL) Bölüm 2'de değil, yöntem bölümünde yer almalı. T10 (Macrì & Lillo, emir yürütme) yalnızca 2.2'de kapsam dışı bir uygulama alanı örneği olarak bir cümleyle anılabilir.

## 11. BibTeX

Anahtar biçimi `soyadYILkelime`. `note` alanında ⚠ olan girdilerde yazar, yayın yeri ya da DOI bu taramada doğrulanamadı; tezde kullanmadan önce yayıncı sayfasından tamamlayın. Bulunamayan dört vizyon kaynağı bilerek dışarıda bırakıldı.

### 11.1 Tohum kaynaklar (doğrulananlar)

```bibtex
@article{ansari2024multifaceted,
  author = {Ansari, Yasmeen and Gillani, Saira and Bukhari, Maryam and Lee, Byeongcheon and Maqsood, Muazzam and Rho, Seungmin},
  title = {A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning},
  journal = {IEEE Access}, volume = {12}, pages = {90041--90060}, year = {2024},
  doi = {10.1109/ACCESS.2024.3418510},
  note = {Yazar sayısı kaynaklar arasında farklı (4 veya 6); yayıncı PDF'inden teyit edin}
}
@article{sattar2025rms,
  author = {Sattar, Asma and Sarwar, Asma and Gillani, Saira and Bukhari, Maryam and Rho, Seungmin and Faseeh, Muhammad},
  title = {A Novel RMS-Driven Deep Reinforcement Learning for Optimized Portfolio Management in Stock Trading},
  journal = {IEEE Access}, year = {2025}, doi = {10.1109/ACCESS.2025.3546099},
  note = {Yazar adlarının açık yazımı kısmen ⚠}
}
@article{ayari2025feature,
  author = {Ayari, Rayan},
  title = {Feature Configuration Effects in {DRL} Portfolio Management: A Risk-Focused Evaluation under Market Stress},
  journal = {Quantitative Finance}, year = {2025},
  note = {DOI ⚠; SSRN 5595570}
}
@article{bai2025review,
  author = {Bai, Yahui and Gao, Yuhe and Wan, Runzhe and Zhang, Sheng and Song, Rui},
  title = {A Review of Reinforcement Learning in Financial Applications},
  journal = {Annual Review of Statistics and Its Application}, year = {2025},
  note = {Dergi DOI ⚠; arXiv:2411.12746}
}
@article{chaudhari2025prediction,
  author = {Chaudhari, Archana Y. and Mahajan, Smita},
  title = {Prediction of Stock Market Using Sentiment Analysis and Ensemble Learning},
  journal = {MethodsX}, volume = {14}, pages = {103260}, year = {2025}, doi = {10.1016/j.mex.2025.103260}
}
@article{delaricaescudero2025explainable,
  author = {de-la-Rica-Escudero, Alejandra and Garrido-Merch{\'a}n, Eduardo C. and Coronado-Vaca, Mar{\'i}a},
  title = {Explainable Post Hoc Portfolio Management Financial Policy of a Deep Reinforcement Learning Agent},
  journal = {PLOS ONE}, volume = {20}, number = {1}, pages = {e0315528}, year = {2025}, doi = {10.1371/journal.pone.0315528}
}
@misc{lwele2025riskaware,
  author = {Lwele, Emmanuel and Emmanuel, Sabuni and Sitali, Sitali Gabriel},
  title = {Risk-Aware Deep Reinforcement Learning for Dynamic Portfolio Optimization},
  year = {2025}, eprint = {2511.11481}, archivePrefix = {arXiv},
  note = {Ön baskı; dergi yayını değil}
}
@article{gonzalezcortes2024portfolio,
  author = {Gonz{\'a}lez Cort{\'e}s, Daniel and Onieva, Enrique and Pastor, Iker and Trinchera, Laura and Wu, Jian},
  title = {Portfolio Construction Using Explainable Reinforcement Learning},
  journal = {Expert Systems}, volume = {41}, number = {11}, pages = {e13667}, year = {2024}, doi = {10.1111/exsy.13667}
}
@article{park2024robots,
  author = {Park, J.-H. and Kim, J.-H. and Huh, J.-H.},
  title = {Deep Reinforcement Learning Robots for Algorithmic Trading: Considering Stock Market Conditions and {U.S.} Interest Rates},
  journal = {IEEE Access}, volume = {12}, pages = {20705--20725}, year = {2024}, doi = {10.1109/ACCESS.2024.3361035}
}
@article{macri2024execution,
  author = {Macr{\`i}, Andrea and Lillo, Fabrizio},
  title = {Reinforcement Learning for Optimal Execution When Liquidity Is Time-Varying},
  journal = {Applied Mathematical Finance}, volume = {31}, number = {5}, pages = {312--342}, year = {2024}, doi = {10.1080/1350486X.2025.2490157}
}
@article{millea2022hrp,
  author = {Millea, A. and Edalat, A.},
  title = {Using Deep Reinforcement Learning with Hierarchical Risk Parity for Portfolio Optimization},
  journal = {International Journal of Financial Studies}, volume = {11}, number = {1}, pages = {10}, year = {2022}, doi = {10.3390/ijfs11010010},
  note = {DOI MDPI kalıbından ⚠}
}
@article{park2021sirl,
  author = {Park, Deog-Yeong and Lee, Ki-Hoon},
  title = {Practical Algorithmic Trading Using State Representation Learning and Imitative Reinforcement Learning},
  journal = {IEEE Access}, volume = {9}, pages = {152310--152321}, year = {2021}, doi = {10.1109/ACCESS.2021.3127209}
}
@article{thongkairat2025combined,
  author = {Thongkairat, Sukrit and Yamaka, Woraphon},
  title = {A Combined Algorithm Approach for Optimizing Portfolio Performance in Automated Trading: A Study of {SET50} Stocks},
  journal = {Mathematics}, volume = {13}, number = {3}, pages = {461}, year = {2025}, doi = {10.3390/math13030461}
}
@article{yue2022antirisk,
  author = {Yue, Han and Liu, Jiapeng and Tian, Dongmei and Zhang, Qin},
  title = {A Novel Anti-Risk Method for Portfolio Trading Using Deep Reinforcement Learning},
  journal = {Electronics}, volume = {11}, number = {9}, pages = {1506}, year = {2022}, doi = {10.3390/electronics11091506}
}
@inproceedings{hao2025collaborative,
  author = {Hao, Jinling and Gao, Minghan and Han, Youyun and Gao, Qiang},
  title = {Collaborative Multi-Agent Reinforcement Learning Model for Portfolio Management},
  booktitle = {Proceedings of the 6th International Conference on Computer Information and Big Data Applications (CIBDA 2025)},
  publisher = {ACM}, pages = {1208--1214}, year = {2025}, doi = {10.1145/3746709.3746915}
}
@article{zhang2025graph,
  author = {Zhang, Bing},
  title = {Graph Attention-Based Heterogeneous Multi-Agent Deep Reinforcement Learning for Adaptive Portfolio Optimization},
  journal = {Scientific Reports}, volume = {16}, pages = {2674}, year = {2025}, doi = {10.1038/s41598-025-32408-w}
}
@article{sun2025hierarchical,
  author = {Sun, Ruoyu and Xi, Yue and Stefanidis, Angelos and Jiang, Zhengyong and Su, Jionglong},
  title = {A Novel Multi-Agent Dynamic Portfolio Optimization Learning System Based on Hierarchical Deep Reinforcement Learning},
  journal = {Complex \& Intelligent Systems}, year = {2025}, doi = {10.1007/s40747-025-01884-y},
  note = {Cilt/sayfa ⚠}
}
@inproceedings{zou2025stockmarl,
  author = {Zou, Peiyan and Siebers, Peer-Olaf},
  title = {{StockMARL}: A Novel Multi-Agent Reinforcement Learning System},
  booktitle = {37th European Modeling \& Simulation Symposium (EMSS 2025)}, year = {2025},
  note = {DOI ⚠}
}
@misc{coriat2025harlf,
  author = {Coriat, Benjamin and Benhamou, Eric},
  title = {{HARLF}: Hierarchical Reinforcement Learning and Lightweight {LLM}-Driven Sentiment Integration for Financial Portfolio Optimization},
  year = {2025}, eprint = {2507.18560}, archivePrefix = {arXiv}
}
@misc{kang2025rlbhrp,
  author = {Kang, Shaofeng and Tian, Zeying},
  title = {Optimal Portfolio Construction -- A Reinforcement Learning Embedded Bayesian Hierarchical Risk Parity ({RL-BHRP}) Approach},
  year = {2025}, eprint = {2508.11856}, archivePrefix = {arXiv}
}
@article{yan2025sector,
  author = {Yan, Yuxiao and Zhang, Changsheng and An, Yang and Zhang, Bin},
  title = {A Deep-Reinforcement-Learning-Based Multi-Source Information Fusion Portfolio Management Approach via Sector Rotation},
  journal = {Electronics}, volume = {14}, number = {5}, pages = {1036}, year = {2025}, doi = {10.3390/electronics14051036}
}
@misc{wojtala2025mactas,
  author = {Wojtala, Maciej and Stefa{\'n}czyk, Bogusz and Bogucki, Dominik and Lepak, {\L}ukasz and Strykowski, Jakub and Wawrzy{\'n}ski, Pawe{\l}},
  title = {{MACTAS}: Self-Attention-Based Inter-Agent Communication in Multi-Agent Reinforcement Learning with Action-Value Function Decomposition},
  year = {2025}, eprint = {2508.13661}, archivePrefix = {arXiv}
}
@inproceedings{guo2023tem,
  author = {Guo, Xudong and Shi, Daming and Fan, Wenhui},
  title = {Scalable Communication for Multi-Agent Reinforcement Learning via Transformer-Based Email Mechanism},
  booktitle = {Proceedings of IJCAI-23}, year = {2023}, doi = {10.24963/ijcai.2023/15}
}
@article{neurocomputing2024teammate,
  title = {Multi-Agent Cooperative Strategy with Explicit Teammate Modeling and Targeted Informative Communication},
  journal = {Neurocomputing}, volume = {586}, pages = {127638}, year = {2024},
  note = {Yazarlar ve DOI ⚠}
}
@inproceedings{yang2020ensemble,
  author = {Yang, Hongyang and Liu, Xiao-Yang and Zhong, Shan and Walid, Anwar},
  title = {Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy},
  booktitle = {Proceedings of the First ACM International Conference on AI in Finance (ICAIF '20)},
  year = {2020}, doi = {10.1145/3383455.3422540}
}
@misc{haelt2025,
  title = {{HAELT}: A Hybrid Attentive Ensemble Learning Transformer Framework for High-Frequency Stock Price Forecasting},
  year = {2025}, eprint = {2506.13981}, archivePrefix = {arXiv},
  note = {Yazarlar ⚠; kapsam dışı (HFT)}
}
@article{sumerlisarigul2024bist,
  author = {S{\"u}merli Sar{\i}g{\"u}l, Sevgi and Aldemir, Ramazan and Uzuno{\u{g}}lu, Hayrettin},
  title = {Stock Market Index Prediction Using Machine Learning Techniques: Application of {BIST} Indices},
  journal = {Journal of Emerging Economies and Policy}, volume = {9}, number = {2}, pages = {96--106}, year = {2024},
  note = {DOI ⚠; DergiPark 1462937}
}
```

### 11.2 Yeni kaynaklar

```bibtex
@inproceedings{li2024masa, author = {Li, Zhenglong and Tam, Vincent and others}, title = {Developing A Multi-Agent and Self-Adaptive Framework with Deep Reinforcement Learning for Dynamic Portfolio Risk Management}, booktitle = {Proceedings of AAMAS 2024}, year = {2024}, eprint = {2402.00515}, note = {Üçüncü yazar ⚠}}
@misc{chen2025mars, author = {Chen, Jiayi and Li, Jing and Wang, Guiling}, title = {{MARS}: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management}, year = {2025}, eprint = {2508.01173}, archivePrefix = {arXiv}}
@misc{ren2025samphdrl, author = {Ren, Xiaotian and Abudurexiti, Nuerxiati and Jiang, Zhengyong and Stefanidis, Angelos and Liu, Hongbin and Su, Jionglong}, title = {{SAMP-HDRL}: Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning}, year = {2025}, eprint = {2512.22895}, archivePrefix = {arXiv}}
@article{mpls2025, title = {{MPLS}: A Modular Portfolio Learning System for Adaptive Portfolio Optimization}, year = {2025}, url = {https://iieta.org/node/20741}, note = {Yazarlar ve dergi adı ⚠ (IIETA)}}
@article{ips2025bnn, title = {Hierarchical Multi-Agent System with Bayesian Neural Networks for Portfolio Optimization}, journal = {Mathematical Modelling of Engineering Problems}, volume = {12}, number = {4}, year = {2025}, doi = {10.18280/mmep.120417}, note = {Yazarlar ⚠; DOI URL'den}}
@misc{he2025hierarchicalai, title = {Hierarchical {AI} Multi-Agent Fundamental Investing: Evidence from China's A-Share Market}, year = {2025}, eprint = {2510.21147}, archivePrefix = {arXiv}, note = {İlk yazar soyadı He; tam yazar listesi ⚠}}
@misc{fang2023intention, author = {Fang, Yuchen and Tang, Zhenggang and Ren, Kan and Liu, Weiqing and Zhao, Li and Bian, Jiang and Li, Dongsheng and Zhang, Weinan and Yu, Yong and Liu, Tie-Yan}, title = {Learning Multi-Agent Intention-Aware Communication for Optimal Multi-Order Execution in Finance}, year = {2023}, eprint = {2307.03119}, archivePrefix = {arXiv}, note = {Hakemli yayın yeri ⚠}}
@inproceedings{lee2020maps, author = {Lee, Jinho and Kim, Raehyun and Yi, Seok-Won and Kang, Jaewoo}, title = {{MAPS}: Multi-Agent Reinforcement Learning-Based Portfolio Management System}, booktitle = {Proceedings of IJCAI-20}, pages = {4520--4526}, year = {2020}, doi = {10.24963/ijcai.2020/623}}
@article{huang2022mspm, author = {Huang, Zhenhan and Tanaka, Fumihide}, title = {{MSPM}: A Modularized and Scalable Multi-Agent Reinforcement Learning-Based System for Financial Portfolio Management}, journal = {PLOS ONE}, year = {2022}, doi = {10.1371/journal.pone.0263689}}
@misc{xue2025attention, author = {Xue, Pei and Ye, Yuanchun}, title = {Attention-Enhanced Reinforcement Learning for Dynamic Portfolio Optimization}, year = {2025}, eprint = {2510.06466}, archivePrefix = {arXiv}}
@misc{factormcls2025, title = {Factor-{MCLS}: Multi-Agent Learning System with Reward Factor Matrix and Multi-Critic Framework for Dynamic Portfolio Optimization}, year = {2025}, eprint = {2504.11874}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{das2019tarmac, author = {Das, Abhishek and Gervet, Th{\'e}ophile and Romoff, Joshua and Batra, Dhruv and Parikh, Devi and Rabbat, Michael and Pineau, Joelle}, title = {{TarMAC}: Targeted Multi-Agent Communication}, booktitle = {ICML}, year = {2019}, eprint = {1810.11187}}
@inproceedings{hu2024commformer, author = {Hu, Shengchao and Shen, Li and Zhang, Ya and Tao, Dacheng}, title = {Learning Multi-Agent Communication from Graph Modeling Perspective}, booktitle = {ICLR}, year = {2024}, eprint = {2405.08550}}
@inproceedings{fan2025fmat, author = {Fan, Wenzhe and Yu, Zishun and Ma, Chengdong and Li, Changye and Yang, Yaodong and Zhang, Xinhua}, title = {Towards Efficient Collaboration via Graph Modeling in Reinforcement Learning}, booktitle = {Proceedings of AAAI}, volume = {39}, number = {16}, pages = {16505--16513}, year = {2025}, doi = {10.1609/aaai.v39i16.33813}}
@inproceedings{liu2023dhcg, author = {Liu, Zeyang and Wan, Lipeng and Sui, Xue and Chen, Zhuoran and Sun, Kewu and Lan, Xuguang}, title = {Deep Hierarchical Communication Graph in Multi-Agent Reinforcement Learning}, booktitle = {Proceedings of IJCAI-23}, pages = {208--216}, year = {2023}, doi = {10.24963/ijcai.2023/24}}
@inproceedings{yuan2022maic, author = {Yuan, Lei and Wang, Jianhao and Zhang, Fuxiang and Wang, Chenghe and Zhang, Zongzhang and Yu, Yang and Zhang, Chongjie}, title = {Multi-Agent Incentive Communication via Decentralized Teammate Modeling}, booktitle = {Proceedings of AAAI}, year = {2022}, doi = {10.1609/aaai.v36i9.21179}}
@inproceedings{admac2024, title = {Robust Communicative Multi-Agent Reinforcement Learning with Active Defense}, booktitle = {Proceedings of AAAI}, year = {2024}, eprint = {2312.11545}, note = {Yazarlar ve DOI ⚠}}
@inproceedings{sun2023certifiably, author = {Sun, Yanchao and Zheng, Ruijie and Hassanzadeh, Parisa and Liang, Yongyuan and Feizi, Soheil and Ganesh, Sumitra and Huang, Furong}, title = {Certifiably Robust Policy Learning against Adversarial Multi-Agent Communication}, booktitle = {ICLR}, year = {2023}, eprint = {2206.10158}}
@inproceedings{immac2021, title = {Intrinsic Motivated Multi-Agent Communication (Extended Abstract)}, booktitle = {Proceedings of AAMAS 2021}, year = {2021}, url = {https://archive.illc.uva.nl/AAMAS-2021/pdfs/p1668.pdf}, note = {Yazarlar ⚠}}
@misc{debunc2024, title = {{DebUnc}: Improving Large Language Model Agent Communication With Uncertainty Metrics}, year = {2024}, eprint = {2407.06426}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{herding2026, title = {The Cost of Consensus: Malignant Epistemic Herding and Adaptive Gating in Distributed Multi-Agent Search}, year = {2026}, eprint = {2605.06988}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{commsurvey2026, title = {Communication Methods in Multi-Agent Reinforcement Learning}, year = {2026}, eprint = {2601.12886}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{goaloriented2025, title = {Toward Goal-Oriented Communication in Multi-Agent Systems: An Overview}, year = {2025}, eprint = {2508.07720}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{deng2026sops, author = {Deng, Zhibo and Liang, Feng and Zhang, Yong and Zhang, Xiaoxi and Hu, Xiping}, title = {Sparse Topology-Aware Pairwise Scoring for Large-Scale Multi-Agent Reinforcement Learning}, booktitle = {ICML}, year = {2026}}
@misc{expocomm2025, title = {Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning}, year = {2025}, eprint = {2502.19717}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{sun2026m2i2, author = {Sun, Chuxiong and He, Peng and Ji, Qirui and Zang, Zehua and others and Wang, Wei}, title = {{M2I2}: Learning Efficient Multi-Agent Communication via Masked State Modeling and Intention Inference}, booktitle = {Proceedings of AAAI}, year = {2026}, doi = {10.1609/aaai.v40i30.39764}}
@misc{huo2024dcmac, author = {Huo, Dongkun and Zhang, Huateng and Hao, Yixue and Ye, Yuanlin and Hu, Long and Wang, Rui and Chen, Min}, title = {{DCMAC}: Demand-aware Customized Multi-Agent Communication via Upper Bound Training}, year = {2024}, eprint = {2409.07127}, archivePrefix = {arXiv}}
@inproceedings{magic2021, title = {Multi-Agent Graph-Attention Communication and Teaming}, booktitle = {Proceedings of AAMAS 2021}, year = {2021}, note = {Yazarlar ⚠}}
@misc{mute2026, title = {{MUTE}: Return-Preserving Communication Unlearning for Efficient Multi-Agent Coordination}, year = {2026}, eprint = {2607.03473}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{jain2019attention, author = {Jain, Sarthak and Wallace, Byron C.}, title = {Attention is not Explanation}, booktitle = {Proceedings of NAACL-HLT}, year = {2019}, note = {Ad açılımları, sayfa ve DOI ⚠}}
@inproceedings{wiegreffe2019attention, author = {Wiegreffe, Sarah and Pinter, Yuval}, title = {Attention is not not Explanation}, booktitle = {Proceedings of EMNLP-IJCNLP}, pages = {11--20}, year = {2019}, note = {ACL Anthology D19-1002}}
@inproceedings{gao2023stockformer, author = {Gao, Siyu and Wang, Yunbo and Yang, Xiaokang}, title = {{StockFormer}: Learning Hybrid Trading Machines with Predictive Coding}, booktitle = {Proceedings of IJCAI-23}, pages = {4766--4774}, year = {2023}, note = {Künye ikincil kaynaktan ⚠}}
@misc{deepclair2024, title = {{DeepClair}: Utilizing Market Forecasts for Effective Portfolio Selection}, year = {2024}, eprint = {2407.13427}, archivePrefix = {arXiv}, note = {Yazarlar ve hakemli sürüm ⚠}}
@misc{rtspno2025, title = {Timing is Important: Risk-aware Fund Allocation based on Time-Series Forecasting}, year = {2025}, eprint = {2505.24835}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{zhang2025ops, author = {Zhang, Ziliang and Zhao, Tianming and Zomaya, Albert}, title = {Online Portfolio Selection with {ML} Predictions}, booktitle = {NeurIPS}, year = {2025}}
@misc{li2019adaptive, author = {Li, Xinyi and others}, title = {Optimistic Bull or Pessimistic Bear: Adaptive Deep Reinforcement Learning for Stock Portfolio Allocation}, year = {2019}, eprint = {1907.01503}, archivePrefix = {arXiv}}
@misc{kazemdehbashi2026tats, author = {Kazemdehbashi, Sina}, title = {Trend-Adjusted Time Series Models with an Application to Gold Price Forecasting}, year = {2026}, eprint = {2601.12706}, archivePrefix = {arXiv}}
@misc{forecastpitfalls2022, title = {Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices}, year = {2022}, eprint = {2203.10716}, archivePrefix = {arXiv}, note = {Yazarlar ve dergi sürümü ⚠}}
@misc{mlpitfalls2021, title = {How to Avoid Machine Learning Pitfalls: A Guide for Academic Researchers}, year = {2021}, eprint = {2108.02497}, archivePrefix = {arXiv}, note = {Yazar ⚠}}
@inproceedings{zhang2025revisiting, author = {Zhang, Dan and Jiang, Kexin and Wu, Chuhan and Peng, Yan and Han, Zhaolong and Chen, Yaoran}, title = {Revisiting the Efficacy of Signal Decomposition in {AI}-Based Time Series Prediction}, booktitle = {Proc. SPIE 13513 (OIOE2024)}, pages = {135130M}, year = {2025}, doi = {10.1117/12.3045386}}
@misc{emdbias2017, title = {Bias Effect on Predicting Market Trends with {EMD}}, year = {2017}, url = {https://educapes.capes.gov.br/handle/capes/1017359}, note = {Yazarlar ve dergi ⚠}}
@misc{rvfl2021walkforward, title = {Random Vector Functional Link Neural Network Based Ensemble Deep Learning for Short-Term Load Forecasting}, year = {2021}, eprint = {2107.14385}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{multifront2023, title = {Multi-Front decomposition for intricate time series}, year = {2023}, eprint = {2303.06394}, archivePrefix = {arXiv}, note = {Tam başlık ve yazarlar ⚠}}
@misc{trendencoded2025, title = {Trend-encoded Probabilistic Multi-order Model: A Non-Machine Learning Approach for Enhanced Stock Market Forecasts}, year = {2025}, eprint = {2502.08144}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{torres2011ceemdan, author = {Torres, Mar{\'i}a E. and Colominas, Marcelo A. and Schlotthauer, Gast{\'o}n and Flandrin, Patrick}, title = {A Complete Ensemble Empirical Mode Decomposition with Adaptive Noise}, booktitle = {Proceedings of IEEE ICASSP}, pages = {4144--4147}, year = {2011}, note = {DOI ⚠}}
@article{colominas2014iceemdan, author = {Colominas, Marcelo A. and Schlotthauer, Gast{\'o}n and Torres, Mar{\'i}a E.}, title = {Improved Complete Ensemble {EMD}: A Suitable Tool for Biomedical Signal Processing}, journal = {Biomedical Signal Processing and Control}, volume = {14}, pages = {19--29}, year = {2014}}
@misc{laszuk_pyemd, author = {Laszuk, Dawid}, title = {{PyEMD}: Python Implementation of Empirical Mode Decomposition}, howpublished = {\url{https://pyemd.readthedocs.io}}, note = {Erişim: 2026-10-06}}
```

```bibtex
@misc{revisitingtsfm2025, title = {Re(Visiting) Time Series Foundation Models in Finance}, year = {2025}, eprint = {2511.18578}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{noguer2026tsfm, author = {Noguer i Alonso, Miquel and Pereira Franklin, Rodolfo}, title = {Pretrained Time-Series Foundation Models for Financial Return Forecasting}, year = {2026}, eprint = {2606.27100}, archivePrefix = {arXiv}}
@inproceedings{laniewski2025chronos, author = {{\L}aniewski and {\'S}lepaczuk}, title = {Evaluating the Chronos Foundation Model for Daily Stock Index Forecasting}, booktitle = {Proceedings of ISD 2025}, year = {2025}, doi = {10.62036/ISD.2025.48}, note = {Ad açılımları ⚠}}
@misc{widener2025foundation, author = {Widener, Michael and others}, title = {On Identifying Why and When Foundation Models Perform Well on Time-Series Forecasting Using Automated Explanations and Rating}, year = {2025}, eprint = {2508.20437}, archivePrefix = {arXiv}}
@misc{nixonraj2025regime, author = {Nixon Raj, Gabriel}, title = {Adaptive and Regime-Aware {RL} for Portfolio Optimization}, year = {2025}, eprint = {2509.14385}, archivePrefix = {arXiv}}
@misc{verma2026hmm, author = {Verma, Ajay Kumar and Putri, Nunik Srikandi and Lesupi, Neo Paul}, title = {Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning}, year = {2026}, eprint = {2605.27848}, archivePrefix = {arXiv}}
@thesis{ilieva2026regime, author = {Ilieva, T. B.}, title = {Regime-Switching Reinforcement Learning for Portfolio Allocation in Pairs Trading}, school = {TU Delft}, type = {Bachelor Thesis}, year = {2026}}
@article{mtx2026, title = {A Novel {AI}-Based Trading Framework for Futures Markets: Evidence from the {MTX} Case Study}, journal = {International Journal of Financial Studies}, volume = {14}, number = {3}, pages = {67}, year = {2026}, note = {Yazarlar ⚠}}
@article{moody2001direct, author = {Moody, John and Saffell, Matthew}, title = {Learning to Trade via Direct Reinforcement}, journal = {IEEE Transactions on Neural Networks}, volume = {12}, number = {4}, pages = {875--889}, year = {2001}}
@misc{globalequity2026, title = {Deep Reinforcement Learning Framework for Diversified Portfolio Management Across Global Equity Markets}, year = {2026}, eprint = {2605.17307}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{ppohrap2026, title = {{PPO-HRAP}: Proximal Policy Optimization with a Hybrid Regime-Aware Policy for Risk-Controlled Trading}, year = {2026}, eprint = {2610.01325}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@article{smarttangency2025, title = {Smart Tangency Portfolio: Deep Reinforcement Learning for Dynamic Rebalancing and Risk--Return Trade-Off}, journal = {International Journal of Financial Studies}, volume = {13}, number = {4}, pages = {227}, year = {2025}, note = {Yazarlar ⚠}}
@misc{zhang2020drltrading, author = {Zhang, Zihao and Zohren, Stefan and Roberts, Stephen}, title = {Deep Reinforcement Learning for Trading}, year = {2020}, eprint = {1911.10107}, archivePrefix = {arXiv}, note = {Dergi sürümü (J. Financial Data Science) ⚠}}
@inproceedings{chow2014cvar, author = {Chow, Yinlam and Ghavamzadeh, Mohammad}, title = {Algorithms for {CVaR} Optimization in {MDPs}}, booktitle = {NeurIPS}, year = {2014}, eprint = {1406.3339}}
@misc{chow2018percentile, author = {Chow, Yinlam and others}, title = {Risk-Constrained Reinforcement Learning with Percentile Risk Criteria}, year = {2018}, eprint = {1512.01629}, archivePrefix = {arXiv}, note = {Dergi (JMLR) ⚠}}
@misc{chang2026recursive, author = {Chang, Minkey}, title = {Portfolio Optimization under Recursive Utility via Reinforcement Learning}, year = {2026}, eprint = {2603.22880}, archivePrefix = {arXiv}}
@inproceedings{agarwal2021precipice, author = {Agarwal, Rishabh and Schwarzer, Max and Castro, Pablo Samuel and Courville, Aaron and Bellemare, Marc G.}, title = {Deep Reinforcement Learning at the Edge of the Statistical Precipice}, booktitle = {NeurIPS}, year = {2021}, eprint = {2108.13264}}
@article{bailey2014deflated, author = {Bailey, David H. and L{\'o}pez de Prado, Marcos}, title = {The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality}, journal = {The Journal of Portfolio Management}, volume = {40}, number = {5}, pages = {94--107}, year = {2014}, doi = {10.3905/jpm.2014.40.5.094}}
@article{bailey2017pbo, author = {Bailey, David H. and Borwein, Jonathan M. and L{\'o}pez de Prado, Marcos and Zhu, Qiji Jim}, title = {The Probability of Backtest Overfitting}, journal = {The Journal of Computational Finance}, volume = {20}, number = {4}, year = {2017}, note = {DOI ⚠}}
@article{hansen2011mcs, author = {Hansen, Peter R. and Lunde, Asger and Nason, James M.}, title = {The Model Confidence Set}, journal = {Econometrica}, volume = {79}, number = {2}, pages = {453--497}, year = {2011}, doi = {10.3982/ECTA5771}}
@article{ledoit2008sharpe, author = {Ledoit, Oliver and Wolf, Michael}, title = {Robust Performance Hypothesis Testing with the Sharpe Ratio}, journal = {Journal of Empirical Finance}, volume = {15}, pages = {850--859}, year = {2008}, doi = {10.1016/j.jempfin.2008.03.002}}
@misc{seedsensitivity2026, title = {{PPO} seed sensitivity in portfolio allocation: a reproducible diagnostic}, howpublished = {Research Square preprint rs-9824202}, note = {Tam başlık, yazarlar ve yıl ⚠}}
@thesis{aarnio2025drl, author = {Aarnio, Joni}, title = {{AI}-Driven Portfolio Management: A Comparative Research of Deep Reinforcement Learning Techniques against the 1/N Portfolio Strategy}, school = {University of Turku}, type = {Master's thesis}, year = {2025}}
@misc{benchmarkingrobustness2023, title = {Benchmarking Robustness of Deep Reinforcement Learning Approaches to Online Portfolio Management}, year = {2023}, howpublished = {HAL hal-04473989}, note = {Yazarlar ve yayın yeri ⚠}}
@misc{multimodaldrl2024, title = {Multimodal Deep Reinforcement Learning for Portfolio Optimization}, year = {2024}, eprint = {2412.17293}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{rlfinreview2025, title = {Reinforcement Learning in Financial Decision Making: A Systematic Review of Performance, Challenges, and Implementation Strategies}, year = {2025}, eprint = {2512.10913}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@inproceedings{guan2021explainable, author = {Guan, Mao and Liu, Xiao-Yang}, title = {Explainable Deep Reinforcement Learning for Portfolio Management: An Empirical Approach}, booktitle = {Proceedings of the Second ACM International Conference on AI in Finance}, pages = {1--9}, year = {2021}, doi = {10.1145/3490354.3494415}}
@thesis{candar2022bist30, author = {Candar, Mert}, title = {Equity Portfolio Optimization Using Reinforcement Learning: An Emerging Market Case}, school = {{\.I}stanbul Teknik {\"U}niversitesi}, type = {Y{\"u}ksek lisans tezi}, year = {2022}}
@thesis{ege2024bist, author = {E{\u{g}}e, {\c{S}}eyma}, title = {Predicting Stock Prices in {BIST}: A Reinforcement Learning and Sentimental Analysis Approach}, school = {{\.I}stanbul Teknik {\"U}niversitesi}, type = {Y{\"u}ksek lisans tezi}, year = {2024}}
@article{tjeecs2022drlkg, title = {A Novel Deep Reinforcement Learning Based Stock Price Prediction Using Knowledge Graph and Community Aware Sentiments}, journal = {Turkish Journal of Electrical Engineering and Computer Sciences}, volume = {30}, number = {4}, pages = {1506--1524}, year = {2022}, note = {Yazarlar ve DOI ⚠}}
@inproceedings{asyu2025macro, title = {A Reinforcement Learning Approach to Stock Trading with Macroeconomic Indicators}, booktitle = {Innovations in Intelligent Systems and Applications Conference (ASYU 2025)}, year = {2025}, doi = {10.1109/ASYU67174.2025.11208408}, note = {Yazarlar ⚠}}
@article{bir2026irl, title = {Inferring Latent Trading Motivations in Borsa Istanbul: A Comparative Inverse Reinforcement Learning Approach}, journal = {Borsa Istanbul Review}, volume = {26}, number = {4}, pages = {100834}, year = {2026}, note = {Yazarlar ve DOI ⚠}}
@article{erdag2026bist100, author = {Erda{\u{g}}, A. and U{\c{c}}ar, M. F. and Tarhan, {\.I}. E.}, title = {Structural Drivers and Forecasting Dynamics of the {BIST100} Index: A Hybrid {ML/DL} Approach Integrating Liquidity, Sentiment and Macroeconomic Factors}, journal = {Review of Behavioral Finance}, year = {2026}, doi = {10.1108/rbf-12-2025-0531}}
@article{ergenc2025bist100, author = {Ergen{\c{c}}, Cansu and Akta{\c{s}}, Rafet}, title = {A Supervised Machine Learning in Financial Forecasting: Identifying Effective Models for the {BIST100} Index}, journal = {Review of Economic Perspectives}, volume = {25}, number = {1}, pages = {66--90}, year = {2025}, doi = {10.2478/revecp-2025-0005}}
@article{ayyildiz2025e7, author = {Ayy{\i}ld{\i}z, Nazif}, title = {Predicting Market Direction With Deep Learning: An Application on {E-7} Country Stock Markets}, journal = {Journal of Finance Letters (Maliye ve Finans Yaz{\i}lar{\i})}, volume = {40}, number = {123}, pages = {92--111}, year = {2025}}
@misc{tradingagents2024, title = {{TradingAgents}: Multi-Agents {LLM} Financial Trading Framework}, year = {2024}, eprint = {2412.20138}, archivePrefix = {arXiv}, note = {AAAI 2025 MARW çalıştayı; yazarlar ⚠}}
@misc{shi2025marketdependent, author = {Shi, Jerick and Hollifield, Burton}, title = {Market-Dependent Communication in Multi-Agent Alpha Generation}, year = {2025}, eprint = {2511.13614}, archivePrefix = {arXiv}, note = {NeurIPS 2025 çalıştayı}}
@misc{llmagentsurvey2024, title = {Large Language Model Agent in Financial Trading: A Survey}, year = {2024}, eprint = {2408.06361}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{llminvest2025, title = {Integrating Large Language Models in Financial Investments and Market Analysis: A Survey}, year = {2025}, eprint = {2507.01990}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{agenticfinance2026, title = {Agentic Artificial Intelligence in Finance: A Comprehensive Survey}, year = {2026}, eprint = {2604.21672}, archivePrefix = {arXiv}, note = {Yazarlar ⚠}}
@misc{pettingzoo2021, title = {{PettingZoo}: Gym for Multi-Agent Reinforcement Learning}, year = {2021}, eprint = {2009.14471}, archivePrefix = {arXiv}, note = {Yazarlar ve yayın yeri ⚠}}
@article{benchmarl2024, title = {{BenchMARL}: Benchmarking Multi-Agent Reinforcement Learning}, journal = {Journal of Machine Learning Research}, volume = {25}, year = {2024}, url = {https://www.jmlr.org/papers/volume25/23-1612/23-1612.pdf}, note = {Yazarlar ⚠}}
@misc{rllib_rlmodule, title = {{RLlib} {RL} Modules documentation}, howpublished = {\url{https://docs.ray.io/en/latest/rllib/rl-modules.html}}, note = {Ray 2.59.0; erişim: 2026-10-06}}
```

### 11.3 Temel (klasik) referanslar

GAT, GraphSAGE, TFT, PPO, TD3 ve A3C künyeleri bu taramada başka makalelerin kaynakçalarında görüldü; CommNet, DIAL/RIAL, MAAC, Mean-Field MARL, MAPPO ve FinRL künyeleri doğrudan teyit edilmedi. Hepsini yayıncı sayfasından son kez kontrol edin.

```bibtex
@misc{schulman2017ppo, author = {Schulman, John and others}, title = {Proximal Policy Optimization Algorithms}, year = {2017}, eprint = {1707.06347}, archivePrefix = {arXiv}}
@inproceedings{mnih2016a3c, author = {Mnih, Volodymyr and others}, title = {Asynchronous Methods for Deep Reinforcement Learning}, booktitle = {ICML}, year = {2016}}
@inproceedings{fujimoto2018td3, author = {Fujimoto, Scott and others}, title = {Addressing Function Approximation Error in Actor-Critic Methods}, booktitle = {ICML}, year = {2018}, eprint = {1802.09477}}
@inproceedings{velickovic2018gat, author = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua}, title = {Graph Attention Networks}, booktitle = {ICLR}, year = {2018}}
@inproceedings{hamilton2017graphsage, author = {Hamilton, Will and Ying, Zhitao and Leskovec, Jure}, title = {Inductive Representation Learning on Large Graphs}, booktitle = {NeurIPS}, volume = {30}, year = {2017}}
@article{lim2021tft, author = {Lim, Bryan and Ar{\i}k, Sercan {\"O}. and Loeff, Nicolas and Pfister, Tomas}, title = {Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting}, journal = {International Journal of Forecasting}, volume = {37}, number = {4}, pages = {1748--1764}, year = {2021}}
@inproceedings{sukhbaatar2016commnet, author = {Sukhbaatar, Sainbayar and Szlam, Arthur and Fergus, Rob}, title = {Learning Multiagent Communication with Backpropagation}, booktitle = {NeurIPS}, year = {2016}, note = {⚠ doğrudan teyit edilmedi}}
@inproceedings{foerster2016dial, author = {Foerster, Jakob and Assael, Yannis M. and de Freitas, Nando and Whiteson, Shimon}, title = {Learning to Communicate with Deep Multi-Agent Reinforcement Learning}, booktitle = {NeurIPS}, year = {2016}, eprint = {1605.06676}, note = {⚠ doğrudan teyit edilmedi}}
@inproceedings{iqbal2019maac, author = {Iqbal, Shariq and Sha, Fei}, title = {Actor-Attention-Critic for Multi-Agent Reinforcement Learning}, booktitle = {ICML}, year = {2019}, note = {⚠ doğrudan teyit edilmedi}}
@inproceedings{yang2018meanfield, author = {Yang, Yaodong and others}, title = {Mean Field Multi-Agent Reinforcement Learning}, booktitle = {ICML}, year = {2018}, note = {⚠ doğrudan teyit edilmedi}}
@inproceedings{yu2022mappo, author = {Yu, Chao and others}, title = {The Surprising Effectiveness of {PPO} in Cooperative Multi-Agent Games}, booktitle = {NeurIPS Datasets and Benchmarks}, year = {2022}, note = {⚠ doğrudan teyit edilmedi}}
@misc{liu2020finrl, author = {Liu, Xiao-Yang and others}, title = {{FinRL}: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance}, year = {2020}, note = {⚠ yayın yeri ve yıl teyit edilmedi}}
@misc{liu2022finrlmeta, author = {Liu, Xiao-Yang and others}, title = {{FinRL-Meta}: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning}, year = {2022}, eprint = {2112.06753}, archivePrefix = {arXiv}, note = {⚠ hakemli yayın yeri teyit edilmedi}}
```
