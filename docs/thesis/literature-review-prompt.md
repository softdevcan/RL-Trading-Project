# Literatür Taraması — Claude (web) için araştırma promptu

> Kullanım: claude.ai'de **Araştırma (Research)** modu ve web araması açıkken aşağıdaki
> çizginin altındaki metnin tamamını yapıştır. Varsa `docs/reference/tez-final.pdf` ve
> `docs/thesis/vision-and-scope.md` dosyalarını da ekle (zorunlu değil; prompt kendi
> başına yeterli).
> Hazırlanma: 2026-10-06.

---

## Rol ve görev

Finansal makine öğrenmesi ve çoklu-ajan pekiştirmeli öğrenme (MARL) alanlarında uzman bir akademik araştırma asistanı olarak çalışmanı istiyorum. Görevin, aşağıda tanıttığım yüksek lisans tezi için **Ekim 2026 itibarıyla güncel, doğrulanmış ve eleştirel** bir literatür taraması hazırlamak.

Bugünün tarihi **6 Ekim 2026**. Eğitim verindeki bilgiye güvenme: her kaynağı web'de bul, künyesini kaynağın kendi sayfasından teyit et. Bana soru sormadan başla; belirsiz kaldığın yerde makul bir varsayım yap ve varsayımı açıkça yaz.

Bu taramanın üç amacı var, önem sırasıyla:

1. **Tezin özgünlük iddialarını sınamak.** Bölüm 4'teki boşlukları kapatan ya da daraltan bir çalışma varsa bunu bilmem gerekiyor. Bu bölüm, raporun en önemli kısmı.
2. Tezin 2. bölümünü (Literatür Taraması) yazabileceğim, en az 60 kaynaklı ve sentezlenmiş bir zemin kurmak.
3. Değerlendirme protokolümü ve erken yayın planımı (aşağıda "Makale 2") literatürdeki en iyi uygulamalara göre düzeltmek.

---

## 1. Tez bağlamı

### 1.1 Çalışma başlığı

*"Çoklu-Ajan Pekiştirmeli Öğrenmede Tahmin-Destekli İletişim: BIST-30 Portföy Yönetimi İçin Mimari Karşılaştırma"*

Danışman onaylı yol haritası Haziran 2026'dan Haziran 2027'deki savunmaya kadar uzanıyor. Şu anda tek-ajan baseline donduruldu; tek-ajan Stable-Baselines3 kodundan PettingZoo + RLlib'e geçiş sürüyor.

### 1.2 Araştırma sorusu

> BIST-30 portföy yönetiminde her sektörün bağımsız bir ajan olarak modellendiği ve ajanlar arasında **tahmin-destekli iletişimin** bulunduğu bir MARL çerçevesi, tek-ajanlı alternatiflere göre (i) risk-ayarlı getiri, (ii) piyasa rejimlerine uyum ve (iii) yorumlanabilirlik açısından ne düzeyde üstünlük sağlar?

- **AQ1 (Mimari):** Hangi iletişim mimarisi BIST-30'da en iyi sonucu verir: iletişimsiz (IPPO/MAPPO), dikkat tabanlı (TarMAC benzeri) ya da graf tabanlı (GAT)?
- **AQ2 (Tahmin sinyali):** Topluluk (ensemble) tahmin sisteminin güven, yön ve rejim çıktılarını iletişim kanalına beslemek koordinasyonu nasıl etkiler?
- **AQ3 (Rejim):** İletişimin değeri boğa, ayı ve yatay piyasada nasıl değişir?

### 1.3 Mevcut sistem (tamamlandı, baseline olarak kullanılacak)

Ana referans **Ansari et al. (2024), IEEE Access**. Metodolojisi BIST-30'a uyarlandı ve genişletildi.

- **Veri:** BIST-30 günlük OHLCV (yfinance, 2018–2026). Makro veriler TCMB EVDS'den (politika faizi, TÜFE, ÜFE) ve yfinance'ten (USD/TRY, EUR/TRY, BIST100, VIX, US10Y, DXY, altın). Temel oranlar yfinance'ten (ROE, ROA, F/K, PD/DD, borç/özsermaye).
- **Tahmin katmanı:**
  - 10 özellik grubu, hepsi en az 1 gün gecikmeli.
  - Özellik seçimi: mutual information + permutation importance.
  - Modeller: XGBoost, LightGBM, CatBoost, BiLSTM ve Temporal Fusion Transformer.
  - Stacking meta-öğrenici (Ridge veya XGBoost); kronolojik 60/20/20 bölme ve OOF.
  - (I)CEEMDAN gürültü filtresi (PyEMD / EMD-signal).
  - TATS: XGBoost tabanlı bir trend sınıflandırıcısının tahmini düzelttiği "trend-adjusted" bileşen.
  - Optuna HPO (TPE + MedianPruner, TimeSeriesSplit).
  - Walk-forward eğitim; 5 günlük purge ve 3 günlük embargo.
  - SHAP ile açıklanabilirlik.
- **RL katmanı:**
  - Gymnasium tabanlı, çok hisseli, yalnız uzun pozisyonlu (long-only) ortam; komisyon modellenmiş.
  - Ödül: PSR (Ansari Eq. 1).
  - Algoritmalar: A2C, PPO ve TD3 (Stable-Baselines3).
  - Gözlem uzayı: bakiye, eldeki hisseler, OHLCV ve teknik göstergeler. Buna hisse başına 4 tahmin özelliği eklenir: tahmini getiri, yön, güven ve model uyumu.
  - Opsiyonel pozisyon boyutlandırma: ATR tabanlı ve çeyrek-Kelly.
- **Değerlendirme:** Sharpe, Sortino, Calmar, Deflated Sharpe Ratio, turnover, maksimum düşüş (MaxDD), yön doğruluğu, profit factor ve IC.

### 1.4 Tezin planlanan katkısı

- **Sektör tabanlı MARL:** Yaklaşık 8 ajan (banka, enerji, sanayi, perakende, holding, telekom, teknoloji, diğer). Sektör içindeki dağılım eşit ağırlıkla ya da risk paritesiyle yapılır. Yedek seçenekler: hiyerarşik yapı (sektör dağıtıcısı + hisse seçici) veya hisse başına ajan.
- **Üç iletişim mimarisinin karşılaştırması:** iletişimsiz (IPPO/MAPPO), dikkat tabanlı ve GNN/GAT. Öğrenilmiş ayrık protokoller (DIAL/RIAL) ileri çalışmaya bırakıldı.
- **Özgün çekirdek — tahmin-destekli iletişim:**
  - Her ajan kendi tahmin güvenini ve yönünü mesaj olarak gönderir.
  - Dikkat ağırlıkları bu güvenle modüle edilir (güven ağırlıklı / *trust-weighted* attention): kendi tahmini zayıf olan ajan diğerlerinin mesajına daha çok ağırlık verir.
  - Meta-öğrenicinin ürettiği rejim sinyali tüm ajanlara yayınlanır.
- **Ödül:** Temel tasarım `α · bireysel getiri + (1−α) · portföy Sharpe`. Ablation için bir varyant daha var: Sortino − λ·turnover − μ·hedef ağırlıktan sapma.
- **Değerlendirme:** Walk-forward, rejim bazlı ablation ve istatistiksel testler (t-testi, Wilcoxon, Diebold-Mariano). Her yapılandırma 3 tohumla (seed) koşulur.

### 1.5 Kapsam dışı (bu konularda derin tarama YAPMA, yalnızca konumlandırma için kısa değin)

Gün içi işlem / HFT, türev ürünler, açığa satış, canlı işlem, BIST dışı borsalarda deney, LLM tabanlı haber/duyarlılık analizi, DIAL/RIAL ve derin feodal (feudal) hiyerarşik MARL.

### 1.6 Yayın hedefleri (taramanın bunları da beslemesi gerekiyor)

- **Makale 1:** *Prediction-Augmented Communicating Agents for Portfolio Management in Emerging Markets*. Hedef ESWA veya Applied Soft Computing.
- **Makale 2 (erken yayın, Aralık 2026):** *Ensemble Prediction with ICEEMDAN Denoising and TATS Correction for RL Trading: A BIST-30 Study*. Hedef Neurocomputing veya KBS.
- **Makale 3:** *Trust-Weighted Message Passing in Multi-Agent Portfolio Management*. Hedef Information Sciences veya IEEE TNNLS.
- **Makale 4:** Türkçe, ulusal dergi.

---

## 2. Zaten bildiğim kaynaklar (tohum seti)

Bunları literatür matrisinde yeniden "yeni bulgu" diye sunma. Bunlar yerine şunları yap:

- (a) Künyelerini doğrula.
- (b) Geri çekilmiş (retracted) olup olmadıklarını kontrol et.
- (c) Bu kaynaklardan **ileri atıf zinciri** (onları atıf yapan 2025–2026 çalışmaları) ve **geri atıf zinciri** ile yeni kaynak bul.

### 2.1 Önceki çalışmamın kaynakçası (künyeler büyük ölçüde doğru, DOI'ler eksik)

1. Ansari, Y. et al. (2024). *A Multifaceted Approach to Stock Market Trading Using Reinforcement Learning*. IEEE Access.
2. Sattar, A. et al. (2025). *A Novel RMS-Driven Deep Reinforcement Learning for Optimized Portfolio Management in Stock Trading*. IEEE Access.
3. Ayari, R. (2025). *Feature configuration effects in DRL portfolio management: a risk-focused evaluation under market stress*. Quantitative Finance.
4. Bai, Y. et al. (2025). *A Review of Reinforcement Learning in Financial Applications*. Annual Review of Statistics and Its Application.
5. Chaudhari, A. Y. & Mahajan, S. (2025). *Prediction of stock market using sentiment analysis and ensemble learning*. MethodsX.
6. de-la-Rica-Escudero, A. et al. (2025). *Explainable post hoc portfolio management financial policy of a Deep Reinforcement Learning agent*. PLOS One.
7. Lwele, E. et al. (2025). *Risk-Aware Deep Reinforcement Learning for Dynamic Portfolio Optimization*. Quantitative Finance.
8. González Cortés, D. et al. (2024). *Portfolio construction using explainable reinforcement learning*. Expert Systems.
9. Park, J.-H. et al. (2024). *Deep Reinforcement Learning Robots for Algorithmic Trading: Considering Stock Market Conditions and U.S. Interest Rates*. IEEE Access.
10. Macrì, A. & Lillo, F. (2025). *Reinforcement Learning for Optimal Execution When Liquidity Is Time-Varying*. Quantitative Finance.
11. Millea, A. & Edalat, A. (2022). *Using deep reinforcement learning with hierarchical risk parity for portfolio optimization*. Int. J. Financial Studies.
12. Park, D. Y. & Lee, K. H. (2021). *Practical Algorithmic Trading Using State Representation Learning and Imitative Reinforcement Learning*. IEEE Access.
13. Thongkairat, S. & Yamaka, W. (2025). *A Combined Algorithm Approach for Optimizing Portfolio Performance in Automated Trading: A Study of SET50 Stocks*. Mathematics.
14. Yue, H., Liu, J., Tian, D. & Zhang, Q. (2022). *A Novel Anti-Risk Method for Portfolio Trading Using Deep Reinforcement Learning*. Electronics.

### 2.2 Vizyon belgemdeki kaynaklar (künyeler EKSİK, önceki bir yapay zekâ taramasından geldi)

Her birinin gerçekten var olup olmadığını, tam yazar listesini, yılını, yayın yerini ve DOI'sini doğrula. Bulamadığını açıkça "bulunamadı" diye işaretle. **Bu liste uydurma kaynak içeriyor olabilir.**

- *Collaborative Multi-Agent RL Model for Portfolio Management*. ACM, ICCBDA 2025.
- *Graph Attention-Based Heterogeneous Multi-Agent DRL for Adaptive Portfolio*. Scientific Reports, 2025. (Rol tabanlı üç ajan: risk, getiri tahmini, piyasa algısı; GAT kullanıyor.)
- *Hierarchical DRL Multi-Agent Portfolio Optimization*. Complex & Intelligent Systems, 2025.
- *StockMARL*. Zou & Siebers, EMSS 2025.
- arXiv 2507.18560, *HARLF: Hierarchical RL + LLM sentiment*.
- arXiv 2508.11856, *RL-BHRP: RL with Bayesian Hierarchical Risk Parity*.
- Preprints.org 202511.0447, *Data-Aware MARL with Adaptive Risk Control*.
- MDPI Electronics 2025, *DRL multi-source sector rotation portfolio*.
- arXiv 2508.13661, *MACTAS: self-attention module for inter-agent communication*.
- IJCAI 2023, *Scalable Communication for Multi-Agent RL*.
- Neurocomputing 2024, *Explicit Teammate Modeling and Targeted Informative Communication*.
- arXiv 2511.12120, *DRL for Automated Stock Trading: Ensemble Strategy* (güncel sürüm?).
- arXiv 2506.13981, *HAELT: Hybrid Attentive Ensemble Learning Transformer*.
- Scientific Reports 2025, *DNN + RL for exchange rate forecasting*.
- Journal of Emerging Economies and Policy 2024, *Stock market index prediction: BIST indices*.
- Decision Science Letters 2025, *Convolutional DRL for emerging stock market*.
- MDPI AI 2025, *DRL framework for NIFTY 50 index trading*.

### 2.3 Temel (klasik) referanslar (yeniden listeleme, yalnızca künye ve BibTeX ver)

- RL algoritmaları: PPO (Schulman 2017), A3C/A2C (Mnih 2016), TD3 (Fujimoto 2018).
- Finansal RL: Yang et al. *ensemble strategy* (ICAIF 2020 mi, 2021 mi? doğrula), FinRL ve FinRL-Meta (Liu et al.).
- MARL iletişim mimarileri: CommNet (Sukhbaatar 2016), DIAL/RIAL (Foerster 2016), TarMAC (Das 2019), MAAC (Iqbal & Sha 2019), Mean-Field MARL (Yang 2018).
- Çok-ajanlı RL algoritmaları ve araçları: MAPPO (Yu et al. 2022, *The Surprising Effectiveness of PPO in Cooperative MARL*), PettingZoo (Terry 2021).
- Graf sinir ağları: GAT (Veličković 2018), GraphSAGE (Hamilton 2017).
- Tahmin ve sinyal işleme: TFT (Lim et al. 2021), CEEMDAN (Torres 2011), ICEEMDAN (Colominas 2014).
- Backtest güvenilirliği: Deflated Sharpe Ratio (Bailey & López de Prado 2014), *Probability of Backtest Overfitting* (Bailey et al.).

---

## 3. Tarama eksenleri

Her eksende önce verdiğim sorulara yanıt ara, sonra kaynak topla. Anahtar kelimeler başlangıç noktası; çeşitlendir.

### A. MARL ile portföy yönetimi ve alım-satım (2023–2026), çekirdek eksen

- **Anahtar kelimeler:** `multi-agent reinforcement learning portfolio management`, `MARL stock trading`, `MAPPO portfolio`, `cooperative multi-agent trading`, `hierarchical multi-agent portfolio optimization`, `sector rotation reinforcement learning`, `multi-agent deep reinforcement learning asset allocation`
- **Sorular:**
  - Ajan granülaritesi nedir: varlık başına, sektör başına, rol başına ya da hiyerarşik mi?
  - Ödül paylaşımlı mı, bireysel mi, karma mı? Kredi atama (credit assignment) nasıl çözülmüş?
  - CTDE kullanılıyor mu?
  - Ajanlar arasında iletişim var mı, varsa hangi türde?
  - Hangi piyasa üzerinde çalışılmış? Not: Çin A-hisse piyasasında çok sayıda MARL çalışması var; bunları atlama, ama piyasa etiketini yaz.

### B. MARL'de öğrenilmiş iletişim (genel ML literatürü, 2022–2026)

- **Anahtar kelimeler:** `communication in multi-agent reinforcement learning survey`, `transformer-based communication MARL`, `graph neural network communication MARL`, `targeted communication`, `uncertainty-aware communication multi-agent`, `trust-aware multi-agent reinforcement learning`, `confidence-weighted message passing`, `intention sharing MARL`, `communication-efficient MARL`, `heterogeneous agents communication`
- **Sorular:**
  - 2025–2026'da standart sayılan iletişim mimarisi hangisi?
  - Mesajın gönderenin belirsizliği ya da güveniyle ağırlıklandırıldığı çalışmalar var mı? (Bölüm 4'teki G2 için kritik.)
  - Dikkat ağırlıklarının yorumlanabilirliği tartışılmış mı?
  - 8–30 ajan ölçeğinde hangi mimari öneriliyor?

### C. Tahmin-destekli RL (forecast-augmented / prediction-guided RL, finans)

- **Anahtar kelimeler:** `forecast-augmented reinforcement learning trading`, `prediction-guided deep reinforcement learning portfolio`, `hybrid forecasting reinforcement learning stock`, `uncertainty-aware forecasting reinforcement learning portfolio`, `ensemble forecasting reinforcement learning trading`, `temporal fusion transformer reinforcement learning`
- **Sorular:**
  - Tahmin çıktısı RL'e hangi yoldan giriyor: gözleme mi, ödüle mi, eylem kısıtına mı, iletişime mi?
  - Tahmin belirsizliği ya da güveni kullanılıyor mu?
  - Tahmin katmanı ablation ile ölçülmüş mü?

### D. Tahmin katmanı (Makale 2 için)

- **Anahtar kelimeler:** `CEEMDAN stock forecasting`, `ICEEMDAN stock price prediction`, `decomposition-ensemble forecasting look-ahead bias`, `data leakage signal decomposition forecasting`, `stacking ensemble stock return prediction`, `trend-adjusted forecast correction`, `temporal fusion transformer stock`
- **Sorular:**
  - **Kritik soru:** Ayrıştırmayı (EMD/CEEMDAN/VMD) serinin tamamına uygulayıp sonra eğitim/test diye bölmek ileriye bakma yanlılığı (look-ahead bias) yaratıyor mu? Bu eleştiriyi yapan çalışmaları ve sızıntısız (kayan pencereli / çevrimiçi) ayrıştırma yöntemlerini bul. Ayrıştırmalı tahmin makalelerinin ne kadarının bu hatayı yaptığına dair sistematik bir analiz varsa onu da getir.
  - ICEEMDAN ile CEEMDAN arasındaki fark nedir? PyEMD (EMD-signal) paketindeki `CEEMDAN` uygulaması hangisine karşılık geliyor?
  - "TATS" adıyla ya da "trend-adjusted" bir düzeltici olarak yayımlanmış özgün bir kaynak var mı? Yoksa en yakın yöntemler hangileri?
  - Kısa bağlam: Zaman serisi temel modelleri (Chronos, TimesFM, Moirai, TimeGPT ve benzerleri) günlük hisse getirisinde gradient boosting veya TFT'yi geçiyor mu? Birkaç güçlü kaynak yeterli.

### E. Rejime duyarlı RL ve durağan olmama

- **Anahtar kelimeler:** `regime-aware reinforcement learning portfolio`, `market regime detection hidden Markov model deep reinforcement learning`, `regime switching DRL trading`, `non-stationary reinforcement learning finance`
- **Sorular:**
  - Rejim nasıl tanımlanıyor: HMM, volatilite eşiği, trend kuralı ya da kümeleme mi?
  - Sonuçları rejim bazında raporlamak yaygın mı?
  - Hangi rejim tanımı akademik olarak en savunulabilir?

### F. Riske duyarlı ödül, pozisyon boyutlandırma ve işlem maliyeti

- **Anahtar kelimeler:** `risk-sensitive reinforcement learning portfolio CVaR`, `differential Sharpe ratio reward`, `drawdown penalty reward shaping`, `Kelly criterion reinforcement learning`, `transaction cost aware deep reinforcement learning`, `turnover penalty portfolio RL`

### G. Değerlendirme metodolojisi ve backtest güvenilirliği (eleştirel eksen)

- **Anahtar kelimeler:** `backtest overfitting`, `deflated Sharpe ratio`, `combinatorial purged cross-validation`, `reproducibility deep reinforcement learning finance`, `statistical significance trading strategy Diebold-Mariano`, `superior predictive ability test`, `model confidence set`, `rliable interquartile mean RL evaluation` (Agarwal et al. 2021), `FinRL contest benchmark`, `deep reinforcement learning trading fails to beat buy and hold`
- **Sorular:**
  - DRL ile işlem yapan çalışmalara yöneltilen başlıca metodolojik eleştiriler neler? Örnekler: işlem maliyetinin yok sayılması, tek tohum kullanımı, zayıf baseline seçimi, hayatta kalma yanlılığı (survivorship bias), hiperparametre sızıntısı.
  - Basit baseline'ları (al-tut, eşit ağırlık, ortalama-varyans) geçemediğini gösteren çalışmalar hangileri?
  - 3 tohum yeterli mi? Literatür kaç tohum öneriyor?

### H. Açıklanabilirlik (XAI) — DRL ve MARL'de, finans bağlamında

- **Anahtar kelimeler:** `explainable reinforcement learning finance`, `SHAP deep reinforcement learning portfolio`, `attention weights interpretability multi-agent`, `explainable multi-agent reinforcement learning`

### I. BIST, Türkiye ve gelişmekte olan piyasalar

- **İngilizce anahtar kelimeler:** `Borsa Istanbul deep reinforcement learning`, `BIST 30 machine learning`, `BIST 100 deep learning prediction`, `Turkish stock market reinforcement learning`, `high inflation stock market machine learning`, `emerging market reinforcement learning portfolio` (NIFTY, Bovespa, SET50, JSE, KOSPI)
- **Türkçe kaynaklar:** DergiPark ve TR Dizin'de `pekiştirmeli öğrenme borsa`, `derin pekiştirmeli öğrenme portföy`, `BIST derin öğrenme`, `Borsa İstanbul makine öğrenmesi` aramalarını yap. **YÖK Ulusal Tez Merkezi**'nde 2020–2026 arası BIST + RL/DRL konulu yüksek lisans ve doktora tezlerini tara; jüri ulusal literatürü sorabilir.
- **Sorular:**
  - BIST'te DRL ya da MARL çalışması var mı?
  - Türkiye'ye özgü etkenler (enflasyon, kur, TCMB politikası) modellenmiş mi?
  - Bu çalışmaların metodolojik kalitesi ne düzeyde?

### J. LLM tabanlı çok-ajanlı işlem sistemleri (kapsam dışı, ama konumlandırma zorunlu)

- **Anahtar kelimeler:** `LLM multi-agent trading framework`, `LLM agents stock trading`, `LLM reinforcement learning hybrid trading`
- **Kapsam:** 2024–2026'nın en çok atıf alan 8–12 çalışmasını listele. Her biri için tek cümleyle bizim yaklaşımımızdan farkını yaz: biz ajanlar arasında doğal dil değil, gradyanla öğrenilen sayısal mesaj kullanıyoruz. Jüri "neden LLM ajanları değil?" diye soracak.

### K. Araç ve çerçeve durumu (Ekim 2026; literatür değil, kısa bir tablo yeterli)

- **Araçlar:** PettingZoo, RLlib (yeni API yığını ve çok-ajanlı destek), BenchMARL, JaxMARL, EPyMARL, MARLlib, TorchRL, FinRL / FinRL-Meta, PyTorch Geometric.
- **Her biri için:** son sürüm ve tarihi, bakım durumu, IPPO/MAPPO desteği, özel iletişim modülü yazmanın kolaylığı.
- Kaynak olarak resmi depo ve sürüm notlarını kullan.

---

## 4. Özgünlük / boşluk tehdit analizi (en önemli bölüm)

Tezin dayandığı boşluk iddiaları aşağıda. Her biri için **iddiayı çürütmeye çalış**: en yakın çalışmaları bul ve bir hüküm ver: **Açık**, **Kısmen kapandı** ya da **Kapandı**.

| Kod | İddia |
|---|---|
| G1 | MARL-finans literatüründe iletişim mimarilerinin (iletişimsiz / dikkat / GNN) aynı deney düzeneğinde sistematik karşılaştırması yok. |
| G2 | Tahmin modelinin güveninin ya da belirsizliğinin ajanlar arası mesaja beslenmesi ve dikkatin bu güvenle modüle edilmesi (*trust/confidence-weighted attention*) ne finansta ne de genel MARL'de doğrudan yapılmış. |
| G3 | Finansta sektör tabanlı ajan granülaritesi az çalışılmış. |
| G4 | BIST ya da Türkiye üzerinde MARL çalışması yok; DRL çalışmaları da sınırlı. |
| G5 | İletişimin katkısının piyasa rejimine göre (boğa/ayı/yatay) nasıl değiştiği analiz edilmemiş. |
| G6 | ICEEMDAN filtreli stacking topluluk tahmini ile trend düzelticinin çıktılarını RL gözlemine veren bir BIST çalışması yok (Makale 2). |

Her boşluk için şunları ver:

- Hüküm.
- En yakın 2–5 çalışma, künyeleriyle.
- Her birinin bizden tam olarak hangi noktada farklı olduğu.
- Tezi nasıl konumlandırmam gerektiği; gerekirse iddianın daha dar ve savunulabilir bir hâlini öner.

**İddiayı çürüten bir çalışma bulursan bunu raporun en başında açıkça yaz.** Bunu erken duymak, savunmada duymaktan çok daha iyi.

---

## 5. Kaynak kalitesi kuralları

- **Zaman penceresi:**
  - Ağırlık Ocak 2023 – Ekim 2026 aralığında olsun.
  - Dahil edilen kaynakların **en az yarısı 2025–2026** tarihli olmalı.
  - Daha eski kaynakları yalnızca "temel" etiketiyle al.
- **Hedef sayı:** Tohum seti hariç 60–90 benzersiz kaynak.
- **Öncelik sırası:**
  1. Q1/Q2 SCIE/SSCI dergiler. Örnekler: ESWA, Applied Soft Computing, KBS, Neurocomputing, Information Sciences, IEEE TNNLS, EAAI, Quantitative Finance, EJOR, Finance Research Letters, J. Financial Data Science.
  2. Güçlü konferanslar: NeurIPS, ICML, ICLR, AAAI, IJCAI, AAMAS, KDD, CIKM, ACM ICAIF.
  3. Preprintler (arXiv q-fin.PM, q-fin.TR, cs.LG, cs.MA ve SSRN): yalnızca 2025–2026 tarihli ve doğrudan ilgiliyse al, **"Preprint – hakemsiz"** etiketiyle. Hakemli sürümü çıkmışsa onu kullan.
- **Dergi sınıfı:** Çeyrek ya da indeks bilgisini bir kaynağa dayandır (JCR / Scimago, yılıyla). Doğrulayamadıysan "doğrulanmadı" yaz; tahmin etme.
- **Dışla:**
  - Yağmacı (predatory) ya da kaçırılmış (hijacked) dergiler.
  - Geri çekilmiş makaleler (Retraction Watch'u kontrol et).
  - Backtest detayı vermeyen çalışmalar.
- **Toplu yayın yapan yayınevleri:** MDPI, Hindawi, Frontiers gibi yayınevlerinde dergi bazında karar ver ve yayınevini etiketle.
- **Uydurma kaynak kesinlikle yasak:**
  - Her kaynak bu oturumda gerçekten bulunmuş olmalı.
  - Her kaynakta çalışan bir DOI ya da arXiv/URL bulunmalı.
  - Yazar, yıl, başlık ve yayın yeri kaynağın kendi sayfasından teyit edilmiş olmalı.
  - Teyit edemediğin her alanı **⚠ doğrulanamadı** diye işaretle.
  - Boşluğu tahminle doldurma; "bulunamadı" yazmak kabul edilebilir bir sonuç.
- **Okuma derinliği:** Çıkarımı yalnızca özetten yaptıysan bunu belirt ("yalnız özet").
- **Atıf sayısı:** Semantic Scholar ya da Google Scholar'dan, bakış tarihiyle birlikte yaz.

---

## 6. Literatür matrisi (her kaynak için bir satır)

| Sütun | İçerik |
|---|---|
| # | Sıra no |
| Künye | Yazarlar (ilk 3 + "et al."), yıl, başlık (özgün dilde) |
| Yayın yeri | Dergi / konferans; tür (Dergi / Konferans / Preprint / Tez) |
| Kalite | Q / CORE sınıfı ve kaynağı; atıf sayısı (tarihli) |
| DOI / URL | Çalışan bağlantı |
| Eksen | A–K (birden fazla olabilir) |
| Piyasa & veri | Piyasa, varlık sayısı, dönem, frekans |
| Yöntem | Algoritma; ajan yapısı; iletişim türü; tahmin entegrasyonu |
| Değerlendirme | Metrikler; walk-forward var mı; işlem maliyeti var mı; istatistiksel test var mı |
| Metodolojik puan (0–6) | Her ölçüt için +1: (1) işlem maliyeti, (2) zamansal out-of-sample / walk-forward, (3) çoklu tohum veya varyans raporu, (4) güçlü basit baseline'lar, (5) istatistiksel anlamlılık testi, (6) kod veya veri erişilebilir |
| Ana bulgu | 1–2 cümle, mümkünse sayılarla |
| Teze ilişkisi | Hangi AQ / G ile ilgili; destekliyor mu, çeliştiriyor mu |
| Öncelik | **Oku** / **Atıf yeter** / **Arka plan** |

---

## 7. Çıktı formatı (bu sırayla)

0. **Tarama protokolü.**
   - Taranan veritabanları, kullanılan sorgu dizeleri ve tarama tarihi.
   - Dahil etme ve dışlama ölçütleri.
   - PRISMA benzeri akış sayıları: bulunan → tekrar ayıklanan → başlık/özetle elenen → tam metinle elenen → dahil edilen.
   - Bunu tezin 2. bölümünde yöntem olarak kullanacağım.
1. **Yönetici özeti** (en fazla 1 sayfa). 2025–2026'da alan nereye gitti? 5–7 ana bulgu.
2. **Kritik uyarılar.** Özgünlüğü tehdit eden bir çalışma, geri çekilmiş ya da var olmayan bir tohum kaynak veya metodolojimizde literatürün açıkça kusur saydığı bir nokta varsa, burada.
3. **Tohum kaynak doğrulama tablosu.** Bölüm 2.1 ve 2.2'deki her kaynak için: doğrulandı / düzeltildi / bulunamadı, düzeltilmiş tam künye ve DOI.
4. **Eksen bazında sentez (A–K).**
   - Her eksen için 2–4 paragraf akademik Türkçe.
   - Kaynakları tek tek özetleme. Eğilimleri, uzlaşılan noktaları, çelişkileri ve açık soruları sentezle.
   - Atıf biçimi: (Yazar, Yıl).
5. **Boşluk / özgünlük analizi** (Bölüm 4'teki tablonun doldurulmuş hâli).
6. **Literatür matrisi** (Bölüm 6).
7. **Değerlendirme protokolümüz için somut öneriler.**
   - Madde madde ve kaynaklı.
   - Örnek konular: tohum sayısı, eklenmesi gereken baseline'lar, kullanılması gereken istatistiksel testler, rejim tanımı, işlem maliyeti seviyesi, ayrıştırma sızıntısının nasıl önleneceği.
8. **Araç durumu tablosu** (eksen K).
9. **Okuma listesi:** Önce okunması gereken 15 kaynak, her biri için tek cümlelik gerekçeyle.
10. **Tez 2. bölüm iskeleti:** Alt başlıklar ve her alt başlığa atanacak kaynak numaraları.
11. **BibTeX:** Dahil edilen tüm kaynaklar ve doğrulanan tohum kaynaklar. Anahtar biçimi `soyadYILkelime`.

---

## 8. Çalışma düzeni ve dil

- **Dil:**
  - Rapor Türkçe, akademik üslupta olsun.
  - Teknik terimleri ilk geçtiği yerde İngilizce karşılığıyla ver, örneğin "merkezi eğitim–dağıtık yürütme (CTDE)".
  - Makale başlıklarını özgün dillerinde bırak.
- **Uzunluk:** Çıktı tek yanıta sığmazsa yanıtı 0–5. bölümlerle ver. Sonra "devam" dediğimde sırasıyla 6., 7–10. ve 11. bölümleri ver. Her yanıtın sonuna `DEVAM → <sonraki bölüm>` satırını ekle.
- **Dürüstlük:** Az kaynak bulduğun bir eksende kaynak sayısını şişirme; "bu eksende literatür ince" demek de bir bulgudur.
