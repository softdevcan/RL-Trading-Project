# Seminer Sunumu — Sayfa Sayfa Plan

---

## NotebookLM İçin Bilgilendirme Notu (Bu bölümü kaynağın başında bırak)

> **Bu doküman ne için?** BIST-30 üzerinde derin pekiştirmeli öğrenme tabanlı bir portföy
> optimizasyonu çalışmasının **seminer dersi sunumu** için sayfa-sayfa plandır. **Bu bir tez
> savunması değildir;** ileride yapılacak yüksek lisans tezinin bir **ön çalışması / hazırlık
> sunumudur** — mevcut sistemi tanıtır ve tezde nereye gidileceğini özetler. 14 slayttan oluşur.
> Her slaytta üç parça vardır: **Başlık**, **Maddeler** (slayt üzerinde görünecek özet) ve
> **Konuşma Notu** (sunucunun sözlü anlatımı). Bazı slaytlarda ayrıca **Görsel** açıklaması
> (diyagram/grafik tarifi) bulunur.
>
> **Kapsam sınırı:** Bu bir *seminer dersi / ön çalışma* sunumudur, projenin tamamının teknik
> dökümü ya da tez savunması değildir. Üç şeye odaklanır: (1) **sistem mimarisi**, (2) **tanımlar**
> — teknik indikatörler ve finans terimleri, (3) **veri kaynakları**. Çoklu-ajan (MARL) kısmı
> yalnızca *gelecekteki tez hedefi* olarak Slayt 12'de geçer; mevcut sistem **tek-ajanlıdır**.
>
> **NotebookLM'den beklenen kullanım:** Bu plandan (a) sözlü anlatım metni / podcast özeti
> üretmek, (b) dinleyicilerin sorabileceği soruları türetmek, (c) terim sözlüğü çıkarmak, (d)
> slaytları daha da sadeleştirmek. Konuşma notları zaten doğal konuşma dilinde yazıldığı için
> seslendirme kaynağı olarak doğrudan kullanılabilir.
>
> **Doğruluk taahhüdü (önemli):** Bu plandaki tüm teknik iddialar proje kaynak koduyla
> karşılaştırılıp doğrulanmıştır. Özellikle dikkat:
> - **PSR = Portfolio-Sharpe-Returns** (kod: `env/reward_functions.py`). Ödül *tek terimli değil*,
>   **6 bileşenli ağırlıklı toplamdır** (getiri + Differential Sharpe + MDD cezası + volatilite
>   cezası + işlem sıklığı ± komisyon). Slayt 9 bu gerçek formülü verir.
> - **RL ajanında "attention" YOKTUR.** Attention yalnızca tahmin modeli TFT'nin içinde ve
>   *gelecek* MARL iletişim vizyonunda geçer. Açıklanabilirlik (Slayt 10) **SHAP** ile yapılır.
> - **Aksiyon uzayı süreklidir** (`Box[-1,+1]`, hisse başına). Pozitif=al, negatif=sat, ~0=tut;
>   süreklilik TD3'ün de çalışmasını sağlar.
>
> **Türkçe kullanım:** Yerleşik İngilizce teknik terimler (ensemble, attention, baseline, walk-forward,
> drawdown, overfitting, state space) korunmuştur; Türkçesi parantezle verilir. Akademik üslup,
> doğru imla.

---

> **Sunum hedefi:** ~14 slayt, ~15–20 dk. Hedef kitle: seminer dersi öğretim üyesi + akademik dinleyiciler (RL uzmanı şart değil).
> **Slayt 4 ve 12 aynı mimari diyagramını kullanır** (çapa + özet). Slayt başına en fazla 1 grafik;
> grafik daima metinle birebir uyumlu olmalı (her slaytta "Görsel" notu bunu tarifler).

---

## Slayt 1 — Kapak

- **Başlık:** Derin Pekiştirmeli Öğrenme ile BIST-30 Portföy Optimizasyonu
- **Alt Başlık:** Çok Modelli Tahmin Entegrasyonu, Risk Yönetimi ve Açıklanabilirlik
- **Sunucu:** Can Akyıldırım
- **Danışman:** Doç. Dr. Deniz Özonur
- **Görsel:** Sade kapak — kurum logosu + bir BIST-30 fiyat grafiği silüeti (arka plan, soluk). Metni gölgelemesin.
- **Konuşma Notu:** "Değerli hocalarım, kıymetli danışman hocam Doç. Dr. Deniz Özonur ve sevgili arkadaşlar, hoş geldiniz. Bugün sizlere seminer dersi kapsamında, ileride yürüteceğim yüksek lisans tezinin bir ön çalışması olan 'Derin Pekiştirmeli Öğrenme ile BIST-30 Portföy Optimizasyonu' başlıklı sunumumu gerçekleştireceğim. Bu çalışma; makine öğrenmesi tahminlerini, risk yönetimini ve derin pekiştirmeli öğrenme karar mekanizmalarını tek bir çatı altında birleştiriyor."

---

## Slayt 2 — Problem ve Motivasyon

- **Başlık:** Karar Problemi ve Zorluklar
- **Maddeler:**
  - **Temel soru:** Gürültülü, durağan olmayan (non-stationary) ve yüksek volatiliteli bir piyasada, bir yapay zekâ ajanı en uygun al-sat kararlarını nasıl öğrenir?
  - **Geleneksel sınırlar:** Klasik modeller (Markowitz, Risk Parity) getirilerin normal dağıldığı ve durağan olduğu gibi gerçekçi olmayan parametrik varsayımlara dayanır.
  - **DRL çözümü:** Ajan, piyasayı önceden modellemeden getiri–risk ödünleşimini doğrudan deneyimleyerek (model-free) öğrenir.
- **Görsel:** BIST-30 fiyat çizgisi üzerinde "?" işaretli al/sat noktaları — karar problemini somutlaştırır. Tek grafik, formülsüz.
- **Konuşma Notu:** "Portföy yönetimi, sürekli değişen bir piyasada risk ve getiri dengesini kurma problemidir. Klasik modellerin katı varsayımları, kriz anlarında ve BIST gibi gelişen piyasalarda çöker. Bu noktada derin pekiştirmeli öğrenme, piyasayı önceden kısıtlayıcı formüllere sokmadan, doğrudan deneyimleyerek en uygun stratejiyi öğrenen güçlü bir alternatif sunar."

---

## Slayt 3 — Literatürdeki Boşluk (Research Gap)

- **Başlık:** Araştırma Boşluğu (Research Gap)
- **Maddeler:**
  - Geleneksel RL çalışmaları çoğunlukla yalnızca fiyat (OHLCV) verilerini kullanır.
  - Makroekonomik ve temel (fundamental) veriler genellikle dışarıda bırakılır.
  - Çoğu çalışma gelişmiş piyasalara (S&P 500, NASDAQ) odaklanır.
  - BIST-30 üzerinde risk-ayarlı, çok modelli (ensemble) RL çalışmaları oldukça sınırlıdır.
  - **Sonuç (bu çalışmanın ve tezin hedefi):** Çok kaynaklı veri entegrasyonu + ensemble tahmin katmanı + risk duyarlı ödül fonksiyonu + BIST-30 odağı ile bu boşluğu doldurmak.
- **Görsel:** 2×2 matris veya basit "kapsananlar ✓ / eksikler ✗" tablosu — sol sütun mevcut literatür, sağ sütun bu çalışma. Metindeki 4 maddeyle birebir hizalı.
- **Konuşma Notu:** "Peki literatürde ne eksikti de bu çalışmaya yöneldik? Mevcut RL çalışmaları genellikle yalnızca geçmiş fiyatlara bakar ve çoğunlukla Amerikan borsalarına odaklanır; makroekonomik şokların ve şirket sağlığının etkisi göz ardı edilir. Bu çalışma; çok kaynaklı veriyi, ensemble tahmin mimarisini ve BIST-30 özelindeki risk dinamiklerini tek bir potada eriterek bu somut boşluğu doldurmayı hedefliyor."

---

## Slayt 4 — Sistem Mimarisi (ana slayt)

- **Başlık:** Uçtan Uca Sistem Mimarisi
- **Görsel (sade, kavramsal akış — merkezde):**

```
Veri Kaynakları → Öznitelik Mühendisliği → Ensemble Katmanı → DRL Ajanı → Al/Sat Kararı → Değerlendirme
```

- **Maddeler (akışın altında, her kutuyu tek satırla açar):**
  - **Veri kaynakları:** OHLCV · makro · fundamental · global göstergeler
  - **Öznitelik mühendisliği:** 10 özellik grubu, ICEEMDAN gürültü filtresi, ≥1 gün gecikme (leakage yok)
  - **Ensemble katmanı:** 5 model → tahmini getiri + yön + güven skoru
  - **DRL ajanı:** durumu gözler → aksiyon (al/sat/tut)
  - **Değerlendirme:** Sharpe, Sortino, Calmar, Maks. Drawdown
- **Detaylı sürüm (yedek / teknik soru gelirse — kod haritası olarak):** `seminar-overview.md` §4'teki modül-adlı tam diyagram. Sunumda sade akışı göster; "hangi dosya?" diye sorulursa yedek slayta geç.
- **Konuşma Notu:** "Geliştirdiğimiz sistemin kavramsal akışı ekranda gördüğünüz gibidir. Ham veriler toplanır, öznitelik mühendisliği ile işlenir, ensemble katmanında bir tahmine dönüştürülür ve nihayetinde DRL ajanına beslenir. Ajan bu sinyallere göre al-sat kararını verir ve performansı risk metrikleriyle değerlendirilir. Kod veya dosya karmaşasından uzak, net ve modüler bir mimari tasarladık."

---

## Slayt 5 — Sinyal Üretimi: Fiyat ve Teknik İndikatörler

- **Başlık:** Fiyat ve Teknik İndikatörler
- **Maddeler:**
  - **OHLCV temeli:** Açılış, En Yüksek, En Düşük, Kapanış fiyatları ve Hacim.
  - **Teknik dönüşümler:** Ham fiyatın karmaşasını anlamlı sinyallere çeviren matematiksel filtreler.
  - **Kullanılan indikatörler (Ansari et al. 2024 seti):**
    - **MACD & ADX** — trendin yönü ve gücü
    - **RSI & CCI** — momentum, aşırı alım/satım sapmaları
    - **Turbulence endeksi** — piyasa genelindeki kesitsel kriz/stres anlarının tespiti (Mahalanobis tabanlı)
- **Görsel:** Üstte bir fiyat çizgisi, altında hizalı RSI paneli (70/30 bantlı). İsteğe bağlı: yanına küçük MACD histogramı. Grafikteki indikatör adları maddelerle aynı olsun.
- **Konuşma Notu:** "Modelin ilk veri katmanını fiyat ve ondan türetilen teknik sinyaller oluşturuyor. Yalnızca ham fiyat kullanmak yerine; MACD ile trendi, RSI ile momentumu ve Turbulence endeksi ile olası piyasa krizlerini matematiksel olarak algılayıp ajanımızın durum uzayına (state space) entegre ediyoruz."
- *Doğruluk notu:* Turbulence tüm hisseler üzerinden kesitsel (Mahalanobis) hesaplanır; MACD/RSI/CCI/ADX hisse bazında. (`data/technical_indicators.py`)

---

## Slayt 6 — Ekonomiyi Anlamak: Makro ve Temel Veriler

- **Başlık:** Makro ve Temel (Fundamental) Veriler
- **Maddeler:**
  - **Makroekonomik rejim:**
    - **TCMB politika faizi & enflasyon (TÜFE/ÜFE)** — sermaye maliyeti ve yerel dinamikler (kaynak: TCMB EVDS)
    - **VIX & USD/TRY** — global korku endeksi ve kur baskısı (kaynak: yfinance); ayrıca US10Y, DXY
  - **Temel analiz (şirket sağlığı):**
    - **Değerleme:** F/K, PD/DD
    - **Kârlılık & likidite:** ROE, ROA, Cari Oran (ayrıca Borç/Özkaynak, Kâr Marjı)
- **Görsel:** İki sütunlu ikon tablosu — solda "Makro (rüzgâr)", sağda "Fundamental (gemi)". Her satır metindeki bir göstergeyle eşleşsin. Sayısal grafik yerine kavramsal ikonlar daha okunur.
- **Konuşma Notu:** "Bir önceki slayttaki teknik analiz bize zamanlamayı verirken, makro ve temel veriler 'yatırım yaptığımız gemi ne kadar sağlam?' ve 'rüzgâr ne yönden esiyor?' sorularını yanıtlar. Yalnızca ekrana değil, faizlere ve şirket bilançolarına da bakan bir model, BIST-30 gibi volatil bir piyasada rejim değişimlerine çok daha hızlı uyum sağlar."

---

## Slayt 7 — Tahmin Katmanı 1: Ensemble Öğrenme Nedir?

- **Başlık:** Ensemble Öğrenme Nedir? Neden Çoklu Model?
- **Maddeler:**
  - **Neden tek model değil?** Finansal verilerde sinyal–gürültü oranı çok düşüktür; tek bir algoritma kolayca ezberler (overfitting) veya yanılır.
  - **Ağaç + derin öğrenme sinerjisi:**
    - **Karar ağaçları (XGBoost/LightGBM/CatBoost):** doğrusal olmayan tablo verilerinde ve anlık şoklarda hızlı ve güçlü.
    - **Derin öğrenme (BiLSTM/TFT):** uzun vadeli zamansal bağımlılıkları (temporal patterns) hatırlar.
  - **Amaç:** Modellerin zayıf yönlerini birbiriyle telafi ederek DRL ajanına pürüzsüz ve yüksek güvenilirlikli bir yön/getiri sinyali sunmak.
- **Görsel:** "Tek model = dağınık tahmin / Ensemble = toplanmış tahmin" karşılaştırması; ok-darbe metaforu veya 5 zayıf okun bir hedefte toplanması. Metindeki "kolektif zekâ" fikriyle uyumlu.
- **Konuşma Notu:** "Bu çalışmanın en güçlü yanlarından biri ensemble, yani çoklu model katmanıdır. Finans gibi gürültülü bir alanda tek bir algoritmaya güvenemezsiniz. Bu nedenle hem anlık tablo verilerinde çok güçlü olan ağaç modellerini hem de geçmişi iyi hatırlayan derin öğrenme modellerini bir araya getirerek, zayıflıkları izole eden bir kolektif zekâ oluşturduk."

---

## Slayt 8 — Tahmin Katmanı 2: Stacking Ensemble Mimarisi

- **Başlık:** Stacking Ensemble Mimarisi
- **Görsel (hiyerarşik yapı — yukarıdan aşağı):**

```
   Base Models:  XGBoost | LightGBM | CatBoost | BiLSTM | TFT
                              │
                              ▼
        Meta-Learner (Ridge / XGBoost)
                              │
                              ▼
   Tahmin: Yön · Getiri · Güven Skoru
```

- **Maddeler (görselin yanında, kısa):**
  - 5 base model paralel eğitilir; her biri kendi tahminini üretir.
  - **Meta-learner** (üst akıl) hangi modele ne kadar güveneceğini öğrenir.
  - **Data leakage'a karşı:** 3-yönlü kronolojik bölme (60/20/20) + walk-forward (purge gap + embargo).
- **Konuşma Notu:** "Mimarimiz şöyle çalışıyor: 5 farklı base model piyasayı analiz edip kendi tahminlerini üretiyor. Ardından bir meta-learner, yani üst akıl, o anki piyasa koşullarında hangi modele ne kadar güvenmesi gerektiğini öğreniyor. Ortaya çıkan rafine getiri, yön ve güven skoru sinyalleri, son karar mercii olan DRL ajanımıza iletiliyor. Veri sızıntısını (data leakage) önlemek için tüm bölmeler kronolojik yapılır."
- *Doğruluk notu:* Görselle metin uyumlu — 5 model, meta-learner, 3 çıktı; hepsi `prediction/models/ensemble.py` ile birebir.

---

## Slayt 9 — Karar Katmanı: DRL Ajanı Ne Öğreniyor?

- **Başlık:** DRL Ajanı Ne Öğreniyor? (Ödül Optimizasyonu)
- **Kavramsal formül (büyük, ortada):**

  $$ \text{Ödül} = \text{Getiri} \;-\; \lambda \cdot \text{Risk} $$

- **Gerçek ödül fonksiyonu (alt satır, küçük puntoyla — "uygulamada"):**

  $$ R_t = w_1 R_t^{\text{port}} + w_2 \,\text{DSR}_t + w_3 \,\text{MDD}_t + w_4 \,\text{Vol}_t + w_5 \,\text{İşlem}_t - \text{Komisyon}_t $$

  (varsayılan ağırlıklar: getiri 0.50, Sharpe 0.30, MDD 0.10, volatilite 0.05, işlem sıklığı 0.05)

- **Maddeler:**
  - Getiri **ödüllendirilir** (maksimize).
  - Risk **cezalandırılır** (minimize): Differential Sharpe + Maks. Drawdown cezası + volatilite cezası.
  - Amaç salt kâr değil, **risk-ayarlı performans**: ödül fonksiyonumuzun adı **PSR — Portfolio-Sharpe-Returns** (Ansari et al. 2024).
  - **Kullanılan ajanlar:** PPO, A2C, TD3 — **sürekli aksiyon uzayı** (`Box[-1, +1]`, hisse başına): pozitif = al, negatif = sat, ~0 = tut.
- **Görsel:** Üstte basit "Getiri − λ·Risk" denge terazisi; altta gerçek 6 bileşenli formülün bar gösterimi (getiri büyük, riskler negatif). Formüldeki terimlerle maddeler birebir aynı.
- **Konuşma Notu:** "Burada akıllara gelen en kritik soru şu: Ajan tam olarak neyi optimize ediyor? Kavramsal olarak ekrandaki gibi — getiriyi maksimize, riski minimize ediyor. Ancak uygulamada ödülümüz tek terimli değil; getiri, Differential Sharpe, Maks. Drawdown cezası, volatilite cezası ve işlem sıklığından oluşan ağırlıklı bir toplam. Bu fonksiyona PSR, yani Portfolio-Sharpe-Returns diyoruz. Sayesinde ajan 'kazan ama güvenli kazan' politikasını öğreniyor."
- *Doğruluk notu (kritik):* "Return − λ·Risk" pedagojik sadeleştirmedir; **gerçek formül 6 bileşenlidir** (`env/reward_functions.py`). PSR burada **Portfolio-Sharpe-Returns** (kod ile tutarlı). Aksiyon uzayı **süreklidir** — bu yüzden TD3 de çalışır.

---

## Slayt 10 — Açıklanabilirlik Katmanı: Şeffaf Yapay Zekâ (XAI)

- **Başlık:** Şeffaf Yapay Zekâ (XAI) ve Yorumlanabilirlik
- **Maddeler:**
  - **Kara kutu (black-box) problemi:** Finans sektörü, nedenini açıklayamadığı yapay zekâ kararlarına güvenmez.
  - **SHAP analizi (post-hoc):** Her al/sat kararında hangi özelliğin yüzde kaç pozitif/negatif katkı yaptığını ölçer (`prediction/explainability.py`, `/prediction/explain/{symbol}` API).
  - **Özellik önemi:** Modelin o gün fiyata mı, enflasyona mı, yoksa RSI'a mı tepki verdiğini sayısal olarak gösterir.
  - *(Gelecek vizyonu)* MARL aşamasında ajanlar arası **attention/iletişim** ağırlıkları da yorumlanabilir hale gelecek.
- **Görsel:** Tek bir SHAP "force plot" veya bar grafiği — bir hisse için en etkili 5–6 özellik (kırmızı=negatif, mavi=pozitif). Gerçek bir çıktı ekran görüntüsü en ikna edicisi.
- **Konuşma Notu:** "Finansal otoriteler, nedenini açıklayamayan kara kutu bir yapay zekâya güvenmez. Bu çalışmada, tahmin katmanının bir hisse için neden o yönde sinyal ürettiğini SHAP analizleriyle açıklıyoruz. Böylece modelin o gün RSI'a mı yoksa yaklaşan faiz kararına mı tepki verdiğini bilimsel olarak görebiliyoruz. İleride, tez kapsamındaki çoklu-ajan aşamasında, ajanların birbirine ne ölçüde 'kulak verdiğini' de görselleştirmeyi hedefliyoruz."
- *Doğruluk notu (kritik):* Mevcut RL ajanı (PPO/A2C/TD3) **attention içermez** — bu nedenle "ajanın attention'ı" mevcut sistemde yoktur; XAI yalnızca **SHAP** ile yapılır. Attention; tahmin modeli TFT'nin içinde ve *gelecek* MARL iletişiminde geçer. Önceki taslaktaki "ajan attention ile odaklanır" ifadesi düzeltildi.

---

## Slayt 11 — Değerlendirme: Finansal Başarı ve Risk Metrikleri

- **Başlık:** Finansal Başarı ve Risk Metrikleri
- **Formül (ortada):**

  $$ \text{Sharpe} = \frac{R_p - R_f}{\sigma_p} $$

- **Maddeler:**
  - **Yüksek Sharpe:** Aynı riskte daha fazla getiri.
  - **Düşük Sharpe:** Alınan riske karşılık tatmin etmeyen getiri.
  - **Maksimum Drawdown (MDD):** Sermayedeki zirveden dibe en büyük kayıp (aşağı yönlü risk).
  - **Sortino & Calmar:** Özellikle negatif dalgalanmaları (zararları) cezalandıran tamamlayıcı metrikler.
  - *(Ek)* Deflated Sharpe Ratio, Profit Factor, Turnover — `prediction/evaluator.py`.
- **Görsel:** İki portföy eğrisi (yüksek vs düşük Sharpe) + birinde işaretli drawdown bölgesi. Formüldeki σ (volatilite) ile eğrilerin dalgalanması görsel olarak örtüşsün.
- **Konuşma Notu:** "Sistemimizin başarısını kanıtlamak için klasik Sharpe oranına bakıyoruz. Ekrandaki formülde payda — sigma, yani volatilite — büyüdükçe oran küçülür. Yani yüksek Sharpe, aynı stresi çekerek daha fazla kazandığımız anlamına gelir. Maksimum Drawdown ile de en kötü senaryoda paramızın ne kadarının eridiğini ölçüyoruz. Sortino ve Calmar, özellikle zararlı dalgalanmaları öne çıkararak tabloyu tamamlıyor."

---

## Slayt 12 — Gelecek Vizyonu: Neredeyiz ve MARL Hedefi

- **Başlık:** Projede Neredeyiz ve Çoklu-Ajan (MARL) Hedefi
- **Maddeler:**
  - **Tamamlanan:** Çalışan, risk-ayarlı, **tek-ajanlı** (baseline) ve ensemble destekli al-sat sistemi + dashboard.
  - **Tezin asıl hedefi (Multi-Agent RL):**
    - Sistemi **sektör-bazlı çoklu-ajan** yapısına evirmek (~8 ajan: Banka, Enerji, Sanayi, Perakende, Holding, Telekom, Teknoloji, Diğer).
    - Ajanlar arası **tahmin-destekli iletişim**: düşük güvenli ajanın, yüksek güvenli ajanlara daha çok "kulak vermesi" (güven-ağırlıklı mesaj).
- **Görsel:** Slayt 4 mimarisinin "çoğaltılmış" hali — tek DRL kutusu yerine 8 sektör ajanı, aralarında çift yönlü iletişim okları. Önce/sonra kontrastı net olsun (tek ajan → ağ).
- **Konuşma Notu:** "Bugüne kadar güçlü, çalışan ve çok modelli tek-ajanlı bir temel sistem inşa ettik. Tezin bundan sonraki aşaması, bu yapıyı sektör bazlı çoklu-ajan mimarisine taşımaktır. Örneğin enerji ajanı ile bankacılık ajanı, geliştirdiğimiz ensemble tahmin güven skorlarını birbirleriyle paylaşarak piyasayı çok daha kolektif bir zekâyla yönetecek."

---

## Slayt 13 — Sonuç: Çalışmanın Katkıları ve Tez Yol Haritası

- **Başlık:** Bu Ön Çalışmanın Katkıları ve Tezin Yönü
- **Maddeler:**
  - BIST-30 için çok kaynaklı (makro + fundamental) veri mimarisi.
  - Ensemble (çoklu model) tahmin katmanının RL'e entegrasyonu.
  - Risk duyarlı DRL al-sat optimizasyonu (PSR ödülü).
  - XAI (açıklanabilir yapay zekâ, SHAP) ile şeffaflaştırılmış karar mekanizması.
  - **Tezde devamı:** çoklu-ajan (MARL) iletişime genişlemeye hazır, ölçeklenebilir altyapı.
- **Görsel:** 5 katkıyı 5 ikonla özetleyen yatay şerit; Slayt 4 mimarisinin renk koduyla aynı (veri=mavi, tahmin=yeşil, RL=turuncu, XAI=mor). Tutarlılık dinleyiciye "her şeyi bağladık" hissi verir.
- **Konuşma Notu:** "Sonuç olarak bu seminer ön çalışmasıyla; BIST-30 özelinde makro veriyi anlayan, ensemble mimarisiyle tahmin üreten, riski formül düzeyinde cezalandıran ve kararlarını açıklanabilir yapay zekâ ile gerekçelendirebilen çok boyutlu bir DRL altyapısı kurulmuş oldu. Yüksek lisans tezimde bu altyapıyı çoklu-ajan ve ajanlar arası iletişim yönünde geliştirmeyi hedefliyorum."

---

## Slayt 14 — Kapanış

- **Başlık:** Teşekkürler ve Soru-Cevap
- **Görsel:** Sade — yalnızca logo + iletişim (ad, e-posta). İsteğe bağlı: küçük bir QR kod (dashboard demo / repo).
- **Konuşma Notu:** "Beni dinlediğiniz için teşekkür ederim. Değerli sorularınızı, eleştirilerinizi ve katkılarınızı almaktan büyük memnuniyet duyarım."

---

## Sunum Notları (sunucu için)

- **Tanım slaytları (5, 6, 9, 11) bu konuşmanın kalbidir** — dinleyici terminoloji ve mimari anlamak istiyor, RL teorisi değil.
- **Slayt 4 = çapa, Slayt 12 = aynı diyagramın çoğaltılmışı.** Aynı renk kodunu Slayt 13'te de kullan.
- **Her slaytta görsel ↔ metin uyumu zorunlu:** grafikteki her etiket, maddelerdeki bir terimle birebir eşleşmeli; eşleşmeyen öğeyi grafikten çıkar.
- **Soru-cevaba karşı sağlamlık (kritik üç düzeltme):**
  1. **Ödül:** "Return − λ·Risk" sadeleştirme; gerçekte 6 bileşenli. Slayt 9 ikisini de gösterir.
  2. **PSR = Portfolio-Sharpe-Returns** (kod ile tutarlı). Ansari makalesinde geçen "Probabilistic Sharpe Ratio" sorulursa: "kod tabanımızda Portfolio-Sharpe-Returns olarak adlandırdık" de.
  3. **Attention ajanI da değil** — XAI yalnızca SHAP. Attention TFT'de ve gelecek MARL'de.
- **Aksiyon uzayı sürekli** (`Box[-1,+1]`): pozitif al / negatif sat / sıfır tut; TD3'ün çalışmasını da bu sağlar.
- Zaman darsa Slayt 7'yi kısalt veya Slayt 8 ile birleştir; **tanım ve mimari slaytlarını asla atma**.
- Q&A için yedek slaytlar (RSI/MACD/Sharpe formülleri, Slayt 4 detaylı modül diyagramı) Slayt 14'ten sonra dursun.
</content>
