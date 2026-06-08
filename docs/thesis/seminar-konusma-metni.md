# Seminer Sunumu — Konuşma Metni (Anlatım İçeriği)

> **Bu doküman ne?** `Transparent_BIST-30_DRL_Optimizer.pptx` sunumundaki 14 slaytın her biri için
> doğal Türkçe anlatım metnidir. Sunum sırasında doğrudan konuşma kaynağı olarak kullanılabilir.
>
> **Üslup kuralı:** Finansa ilgisi olmayan bir dinleyici de takip edebilsin diye, geçen her teknik
> ve finansal terim (Sharpe oranı, drawdown, ensemble, RSI, volatilite, ...) birkaç kelimeyle,
> günlük dilden bir benzetmeyle açıklanır. Cümleler kısa ve sözlü anlatıma uygundur.
>
> **Bağlam:** Bu bir tez savunması değil; seminer dersi için yapılan, ileriki yüksek lisans tezinin
> bir ön çalışmasıdır. Mevcut sistem tek-ajanlıdır; çoklu-ajan kısmı gelecekteki tez hedefidir.
>
> ⏱️ Toplam hedef: ~15–20 dakika. Her slayt için yaklaşık süre başlıkta belirtilmiştir.

---

## Slayt 1 — Kapak  ⏱️ ~45 sn

"Değerli hocalarım, kıymetli danışman hocam Doç. Dr. Deniz Özonur ve sevgili arkadaşlar, hoş geldiniz.

Ben Hayri Can Akyıldırım. Bugün sizlere, seminer dersi kapsamında hazırladığım ve ileride yürüteceğim yüksek lisans tezimin bir ön çalışması olan projeyi anlatacağım. Başlığımız: *'Derin Pekiştirmeli Öğrenme ile BIST-30 Portföy Optimizasyonu'.*

Kısaca şunu yaptık: Borsa İstanbul'un en büyük 30 şirketi için, kendi kendine *al-sat-bekle* kararı verebilen bir yapay zekâ sistemi geliştirdik. Bu sistem üç şeyi bir araya getiriyor — geleceğe yönelik tahminler, riski kontrol altında tutan bir yapı, ve kararlarını bize *neden böyle karar verdiğini* açıklayabilen şeffaf bir mekanizma.

İzninizle başlayalım."

> 💡 *Tanım hatırlatması (gerekirse):* "Portföy = elimizdeki paranın hangi hisselere ne kadar dağıtıldığı. BIST-30 = İstanbul Borsası'ndaki en büyük 30 şirket."

---

## Slayt 2 — Karar Problemi ve Zorluklar  ⏱️ ~1.5 dk

"Önce çözmeye çalıştığımız problemi netleştirelim.

Borsada para kazanmak aslında tek bir soruya iniyor: *Hangi hisseyi, ne zaman alıp ne zaman satmalıyım?* Kulağa basit geliyor ama değil. Çünkü piyasa sürekli değişiyor, dalgalı ve tahmin edilmesi çok zor. Buna teknik dilde *'durağan olmayan'* diyoruz — yani dünün kuralları yarın geçerli olmayabilir.

Peki klasik yöntemler bunu nasıl çözüyor? Markowitz gibi geleneksel finans modelleri var. Ama bunların bir zayıf noktası şu: Piyasanın 'düzgün ve öngörülebilir' davrandığını varsayıyorlar. Gerçek hayatta, özellikle kriz anlarında ve Türkiye gibi gelişen, dalgalı piyasalarda bu varsayım çöküyor.

Bizim çözümümüz *derin pekiştirmeli öğrenme*. Şöyle düşünün: Bir çocuğa bisiklet sürmeyi formülle değil, deneyerek-düşerek öğretirsiniz. Bizim yapay zekâ ajanımız da piyasayı önceden formüllerle tanımlamaya çalışmıyor; binlerce kez deneyerek, hatalarından öğrenerek en iyi stratejiyi kendisi keşfediyor.

Ekrandaki grafikte gördüğünüz o iniş çıkışlar ve soru işaretleri işte tam da bu zorluğu temsil ediyor: her an bir karar noktası."

> 💡 *Açıklanan terimler:* **Durağan olmayan (non-stationary)** = kurallarının zamanla değiştiği ortam. **Volatilite** = fiyatların ne kadar sert iniş-çıkış yaptığı, yani 'dalgalılık'. **Pekiştirmeli öğrenme** = deneme-yanılmayla, ödül-ceza alarak öğrenme.

---

## Slayt 3 — Araştırma Boşluğu ve Yaklaşımımız  ⏱️ ~1.5 dk

"Peki bu konuda zaten çok çalışma varken, biz neyi farklı yaptık? İşte literatürdeki, yani daha önce yapılmış akademik çalışmalardaki dört önemli eksik.

Birincisi: Çoğu çalışma sadece geçmiş fiyatlara bakıyor. Yani sadece grafiğe. Oysa bir hissenin değerini sadece grafiği değil, ülkenin faiz oranı, enflasyon, döviz kuru ve şirketin mali sağlığı da belirler.

İkincisi: Çalışmaların büyük kısmı Amerikan borsasına odaklanıyor — S&P 500 gibi. Türkiye gibi gelişen piyasalar göz ardı ediliyor.

Üçüncüsü: Genelde tek bir tahmin modeli kullanılıyor. Biz birden fazla modeli birlikte çalıştırıyoruz — buna birazdan geleceğiz.

Dördüncüsü: Risk çoğu zaman ihmal ediliyor; sadece 'ne kadar kazandın' diye bakılıyor, 'bunu ne kadar risk alarak kazandın' diye bakılmıyor.

Bizim çalışmamız bu dört boşluğu birden dolduruyor: Çok kaynaklı veri kullanıyoruz, BIST-30'a odaklanıyoruz, birden fazla tahmin modelini birleştiriyoruz ve riski ödül-ceza mekanizmasına dahil ediyoruz. Tablonun sağ sütununda gördüğünüz yeşil tikler bizim katkılarımız."

> 💡 *Açıklanan terimler:* **Literatür** = bir konuda daha önce yayımlanmış bilimsel çalışmaların tümü. **Fundamental (temel) veri** = bir şirketin kârı, borcu gibi mali sağlık göstergeleri. **Gelişen piyasa** = Türkiye gibi, gelişmiş ülkelere göre daha oynak ve büyüyen borsalar.

---

## Slayt 4 — Uçtan Uca Sistem Mimarisi  ⏱️ ~1.5 dk  ⭐ (ana slayt)

"Şimdi sistemin bütününe kuş bakışı bakalım. Soldan sağa doğru bir üretim bandı gibi düşünün.

İlk durak, **Veri Kaynakları**. Sisteme ham bilgileri topluyoruz: hisse fiyatları, makroekonomik veriler — yani faiz, enflasyon gibi — ve şirketlerin mali tabloları.

İkinci durak, **Öznitelik Mühendisliği**. Burada ham veriyi makinenin anlayacağı, anlamlı sinyallere dönüştürüyoruz. Bu aşamada veriyi temizliyoruz ve önemli bir kural uyguluyoruz: model asla geleceği görmüyor, sadece geçmiş bilgiyle karar veriyor. Buna 'veri sızıntısının önlenmesi' diyoruz — yani sınavda cevap anahtarını önceden görmemek gibi.

Üçüncü durak, **Ensemble Katmanı**. Burada beş farklı tahmin modeli, ertesi gün için bir tahmin üretiyor: fiyat yukarı mı gider aşağı mı, ne kadar gider ve buna ne kadar güveniyoruz.

Dördüncü durak, sistemin beyni: **DRL Ajanı**. Tüm bu bilgilere bakıp nihai kararı veriyor — al, sat, ya da bekle.

Son durak, **Değerlendirme**. Verdiği kararların ne kadar başarılı olduğunu, hem kâr hem de risk açısından ölçüyoruz.

Bu beş kutu, sunumun geri kalanının da haritası. Şimdi her birini tek tek açacağız."

> 💡 *Açıklanan terimler:* **Öznitelik (feature)** = modele verilen her bir bilgi parçası, örneğin 'dünkü fiyat' veya 'enflasyon oranı'. **Ensemble** = birden fazla modeli aynı anda çalıştırıp ortak akıl oluşturma. **DRL Ajanı** = kararı veren yapay zekâ; 'ajan' burada 'karar veren oyuncu' demek.

---

## Slayt 5 — Sinyal Üretimi: Fiyat ve Teknik İndikatörler  ⏱️ ~1.5 dk

"İlk veri katmanımız fiyat ve fiyattan türetilen sinyaller.

En temelde **OHLCV** var. Bu beş harf, bir hissenin bir günlük özetidir: Açılış fiyatı, gün içi en yüksek, en düşük, kapanış fiyatı ve o gün el değiştiren hisse miktarı — yani işlem hacmi.

Ama ham fiyat tek başına yeterli değil. Onun için **teknik indikatörler** kullanıyoruz. Bunlar fiyat grafiğini matematiksel filtrelerden geçirip işe yarar sinyaller çıkaran araçlar. Üç gruba ayıralım:

Birincisi, **trend göstergeleri** — MACD ve ADX. Bunlar bize fiyatın hangi yöne gittiğini ve bu hareketin ne kadar güçlü olduğunu söyler. Yani 'yukarı mı gidiyor, ve bu yükseliş sağlam mı?'

İkincisi, **momentum göstergeleri** — RSI ve CCI. Bunlar fiyatın çok fazla mı yükseldiğini yoksa çok mu düştüğünü ölçer. Mesela RSI sıfırla yüz arasında bir değer verir; yetmişin üstü 'aşırı alınmış, geri dönebilir', otuzun altı 'aşırı satılmış, toparlayabilir' anlamına gelir.

Üçüncüsü, ve özellikle önemlisi, **Turbulence — yani çalkantı endeksi**. Bu, tüm piyasada aynı anda bir panik, bir kriz havası olup olmadığını yakalıyor. Adeta piyasanın ateşini ölçen bir termometre.

Tüm bu sinyalleri ajanımızın gördüğü bilgi havuzuna ekliyoruz."

> 💡 *Açıklanan terimler:* **İndikatör** = fiyattan hesaplanan yardımcı gösterge. **Trend** = fiyatın genel gidiş yönü. **Momentum** = hareketin hızı/şiddeti. **RSI** = 0–100 arası 'aşırı alım/satım' ölçer. **İşlem hacmi** = o gün kaç adet hissenin alınıp satıldığı.

---

## Slayt 6 — Ekonomiyi Anlamak: Makro ve Temel Veriler  ⏱️ ~1.5 dk

"Bir önceki slaytta teknik analiz bize *zamanlamayı* verdi. Bu slayttaki veriler ise *resmin bütününü* veriyor. İki güzel benzetmeyle anlatalım.

Soldaki **'rüzgâr'**: makroekonomik koşullar. Bir gemiyi düşünün — ne kadar sağlam olursa olsun, rüzgâr ters eserse zorlanır. Borsa için bu rüzgâr; Merkez Bankası'nın faiz kararları, enflasyon, ve döviz kuru. Ayrıca VIX dediğimiz bir 'küresel korku endeksi' var — dünya piyasalarındaki gerginliği ölçer. Bu veriler özellikle Türkiye gibi kura ve faize çok duyarlı bir piyasada hayati.

Sağdaki **'gemi'**: şirketin kendi sağlığı, yani temel analiz. Burada şirketin karnesine bakıyoruz. Birkaç örnek: F/K oranı, yani Fiyat/Kazanç — hisse, kazancına göre pahalı mı ucuz mu? ROE, yani özkaynak kârlılığı — şirket kendi parasıyla ne kadar verimli kâr ediyor? Cari oran — şirketin kısa vadeli borçlarını ödeyebilecek nakdi var mı?

Özetle: Sadece grafiğe değil, hem rüzgâra hem de geminin sağlamlığına bakan bir model kuruyoruz. Bu da onu ani değişimlere karşı çok daha dayanıklı yapıyor."

> 💡 *Açıklanan terimler:* **Makroekonomi** = bir ülkenin geneli; faiz, enflasyon, kur. **Faiz** = paranın maliyeti. **F/K oranı** = hissenin pahalı/ucuz olduğunu gösteren temel ölçü. **ROE** = şirketin kârlılık verimi. **VIX** = piyasalardaki korku/gerginlik düzeyi.

---

## Slayt 7 — Ensemble Öğrenme: Neden Çoklu Model?  ⏱️ ~1.5 dk

"Şimdi tahmin katmanına geldik. Buradaki temel fikir çok basit ama çok güçlü.

Soru şu: Neden tek bir tahmin modeli kullanmıyoruz? Çünkü finansal veride 'sinyal-gürültü oranı' çok düşük. Yani gerçek bilgiyle anlamsız dalgalanma iç içe geçmiş durumda. Tek bir model bu gürültüye kapılıp yanılabilir, ya da geçmişi ezberleyip gelecekte çuvallayabilir. Buna *aşırı öğrenme* deniyor — sınava çalışırken soruları anlamak yerine cevapları ezberleyen öğrenci gibi; yeni soru gelince takılır.

Çözümümüz: Tek bir uzmana güvenmek yerine, bir uzmanlar kuruluna danışmak. Buna *ensemble*, yani topluluk öğrenmesi diyoruz. İki farklı uzman tipini bir araya getiriyoruz:

Birinci grup, **ağaç tabanlı modeller** — XGBoost ve LightGBM gibi. Bunlar tablo halindeki veride, ani şoklarda çok hızlı ve isabetli.

İkinci grup, **derin öğrenme modelleri** — BiLSTM ve TFT gibi. Bunlar geçmişi uzun süre hatırlama konusunda çok iyi; zaman içindeki gizli örüntüleri yakalıyorlar.

Birinin zayıf olduğu yerde diğeri güçlü. Hepsini birleştirince ortaya, tek tek hepsinden daha güvenilir bir 'kolektif zekâ' çıkıyor."

> 💡 *Açıklanan terimler:* **Ensemble** = birden çok modelin ortak kararı, 'uzmanlar kurulu'. **Aşırı öğrenme (overfitting)** = ezberleyip genelleyememe. **Gürültü** = veride işe yaramayan, yanıltıcı rastgelelik. **Derin öğrenme** = beyindeki sinir ağlarından esinlenmiş güçlü model ailesi.

---

## Slayt 8 — Stacking Ensemble Mimarisi  ⏱️ ~1.5 dk

"Peki bu uzmanlar kurulu pratikte nasıl çalışıyor? Ekrandaki şemaya bakalım.

En üstte beş bağımsız model var — az önce bahsettiğimiz XGBoost, LightGBM, CatBoost, BiLSTM ve TFT. Her biri piyasaya kendi açısından bakıp kendi tahminini üretiyor.

Ama burada akıllıca bir adım var. Bu beş tahmini körlemesine ortalamıyoruz. Bunun yerine, ortada gördüğünüz bir **'Meta-Learner', yani üst akıl** devreye giriyor. Bu üst aklın görevi şu: O günkü piyasa koşullarında hangi modele ne kadar güvenmesi gerektiğini öğrenmek. Mesela kriz dönemlerinde belki ağaç modellerine, sakin dönemlerde derin öğrenme modellerine daha çok kulak veriyor.

Sonuçta üç net çıktı üretiyoruz: Birincisi rafine bir getiri tahmini, ikincisi yön — yukarı mı aşağı mı, üçüncüsü ve önemlisi bir 'güven skoru' — yani bu tahmine ne kadar güveniyoruz.

Sağ tarafta da kritik bir detay var: Veri sızıntısını önlüyoruz. Veriyi her zaman zaman sırasına göre bölüyoruz ki model geleceği asla önceden görmesin. Bu, çalışmanın bilimsel güvenilirliği için olmazsa olmaz."

> 💡 *Açıklanan terimler:* **Base model** = kuruldaki tek tek uzmanlar. **Meta-Learner (üst akıl)** = uzmanların hangisine ne kadar güvenileceğine karar veren üst model. **Güven skoru** = tahminin ne kadar emin olduğunu gösteren sayı. **Veri sızıntısı (data leakage)** = modelin yanlışlıkla geleceği görmesi; bunu engelliyoruz.

---

## Slayt 9 — DRL Ajanı Ne Öğreniyor? (Ödül Optimizasyonu)  ⏱️ ~2 dk  ⭐

"Şimdi sistemin beynine, karar veren ajana geldik. Buradaki en kritik soru: Ajan tam olarak neyi başarmaya çalışıyor?

En tepedeki basit formül bunu özetliyor: **Ödül = Getiri eksi, lambda çarpı Risk.** Türkçesi şu: Ajanı kazandığı para için ödüllendiriyoruz, ama aldığı risk için cezalandırıyoruz. Yani sadece 'çok kazan' demiyoruz; 'akıllıca, güvenli kazan' diyoruz. Buradaki lambda, riske ne kadar önem verdiğimizi ayarlayan bir düğme gibi.

Ama dürüst olmak gerekirse, gerçekte kullandığımız ödül bundan biraz daha zengin. Alttaki çubuklarda gördüğünüz gibi, ödül altı parçadan oluşan bir denge: Kazançtan artı puan; ama büyük düşüşlerden, aşırı dalgalanmadan, gereksiz çok işlem yapmaktan ve işlem masraflarından eksi puan alıyor. Bu fonksiyona biz PSR adını verdik — yani Portföy-Sharpe-Getiri ödülü. Bütün bu parçalar birleşince, ajan 'kazan ama dikkatli kazan' politikasını kendiliğinden öğreniyor.

Karar mekanizması da şöyle: Ajan her hisse için eksi birle artı bir arasında bir değer üretiyor. Artıysa al, eksiyse sat, sıfıra yakınsa bekle. Üç farklı yapay zekâ algoritması deniyoruz — PPO, A2C ve TD3. Bunlar pekiştirmeli öğrenmenin farklı 'öğrenme stilleri' gibi düşünülebilir."

> 💡 *Açıklanan terimler:* **Ödül fonksiyonu** = ajana 'iyi iş' ya da 'kötü iş' diyen puanlama sistemi; ajan bu puanı en yükseğe çıkarmaya çalışır. **Risk** = işlerin ters gitme ihtimali; burada dalgalanma ve düşüşle ölçülür. **Drawdown** = paranın tepe noktasından ne kadar geri çekildiği. **PPO/A2C/TD3** = farklı öğrenme algoritmaları.
>
> ⚠️ *Soru gelirse:* "Üstteki formül anlatım kolaylığı için sadeleştirilmiştir; sistemde gerçekten altı bileşenli ödül kullanılıyor. PSR'yi biz kodda 'Portfolio-Sharpe-Returns' olarak adlandırdık."

---

## Slayt 10 — Şeffaf Yapay Zekâ (XAI) ve Yorumlanabilirlik  ⏱️ ~1.5 dk

"Çok kritik bir konu: güven. Bir yapay zekâ size 'bu hisseyi al' dese ama nedenini söyleyemese, paranızı ona emanet eder miydiniz? Etmezdiniz. Finans dünyası da etmez.

İşte bu 'nedenini söyleyemeyen' yapay zekâya **kara kutu** diyoruz — içinde ne olduğunu göremediğimiz bir kutu. Biz bu kutuyu açmak istedik.

Bunun için **SHAP** adında bir teknik kullanıyoruz. SHAP bize her karar için şunu söylüyor: 'Bu tahmini verirken hangi bilgi ne kadar etkili oldu?' Ekrandaki grafikte görüyorsunuz — mesela bir kararda enflasyon yüzde şu kadar olumsuz etki yapmış, RSI göstergesi şu kadar olumlu katkı vermiş.

Yani artık 'model şunu dedi, sebebini bilmiyoruz' demiyoruz. 'Model bugün bu kararı şu üç sebepten dolayı verdi' diyebiliyoruz. Bu da hem bize, hem de sisteme güvenecek kişilere şeffaflık sağlıyor.

İleride, tez aşamasında çoklu-ajan yapısına geçtiğimizde, ajanların birbirine ne kadar 'kulak verdiğini' de aynı şekilde görselleştirmeyi planlıyoruz."

> 💡 *Açıklanan terimler:* **XAI (Açıklanabilir Yapay Zekâ)** = kararlarının gerekçesini gösterebilen yapay zekâ. **Kara kutu** = nasıl karar verdiği görünmeyen model. **SHAP** = her bilginin karara katkısını ölçen açıklama yöntemi.

---

## Slayt 11 — Finansal Başarı ve Risk Metrikleri  ⏱️ ~2 dk

"Peki sistemin başarılı olup olmadığını nasıl ölçüyoruz? Sadece 'kâr etti mi' demek yetmez, çünkü çok büyük risk alarak da kâr edilebilir. Bunun için birkaç ölçüt kullanıyoruz.

En önemlisi **Sharpe oranı**. Şöyle açıklayayım: İki kişi de yılda yüzde yirmi kazanmış olsun. Ama biri bunu sakin, istikrarlı bir şekilde yapmış; diğeri her gün büyük iniş-çıkışlarla, kalbi ağzında. Sharpe oranı işte bu farkı ölçer — *aldığınız her birim risk başına ne kadar getiri elde ettiniz.* Yüksek Sharpe iyidir: aynı stresle daha çok para. Formülde de görüyorsunuz, paydadaki sigma yani dalgalanma büyüdükçe oran düşer.

İkinci ölçüt **Maksimum Drawdown**. Bu, en kötü senaryoyu ölçer: Paranız en yüksek noktasından en dibe kadar en fazla yüzde kaç eridi? Bir yatırımcı için bu, 'en kötü ne kadar acı çekerim' sorusunun cevabı.

Son olarak **Sortino ve Calmar** oranları var. Bunlar Sharpe'a benzer, ama özellikle *zarar* getiren dalgalanmalara odaklanırlar. Çünkü yukarı yönlü dalgalanma aslında sevindiricidir; bizi asıl üzen aşağı yönlü olanıdır. Bu metrikler tam da onu cezalandırır.

Özetle: Hem ne kadar kazandığımıza, hem de bunu ne kadar güvenli kazandığımıza birlikte bakıyoruz."

> 💡 *Açıklanan terimler:* **Sharpe oranı** = risk başına getiri; 'ne kadar stresle ne kadar kazandın'. **Risk-ayarlı getiri** = çıplak kâr değil, alınan riske göre düzeltilmiş kâr. **Drawdown** = tepeden dibe en büyük kayıp yüzdesi. **Sortino/Calmar** = özellikle zararları öne çıkaran ölçütler.

---

## Slayt 12 — Tezin Gelecek Hedefi: Çoklu-Ajan (MARL) Ekosistemi  ⏱️ ~1.5 dk

"Buraya kadar anlattığım sistem, şu an çalışan halidir. Sol tarafta görüyorsunuz: tek bir akıllı ajan, tüm kararları veriyor. Bu bizim sağlam temelimiz — kuruldu, çalışıyor.

Tezde asıl hedefim ise sağ taraftaki yapı: **çoklu-ajan sistemi.** Fikir şu: Tek bir ajanın 30 farklı hisseyi yönetmesi yerine, her sektör için ayrı bir uzman ajan olsun. Bir Banka ajanı, bir Enerji ajanı, bir Teknoloji ajanı gibi — yaklaşık sekiz ajan. Çünkü banka hisseleriyle teknoloji hisseleri çok farklı davranır; her birini kendi uzmanına bırakmak daha mantıklı.

Ve en heyecan verici kısım: Bu ajanlar **birbiriyle konuşacak.** Mesela bir ajan kendi tahmininden çok emin değilse, diğer ajanların görüşüne daha çok kulak verecek. Buna 'tahmin-destekli iletişim' diyoruz. Yani sadece bireysel değil, kolektif bir akılla piyasayı yönetecekler.

Kısacası: Bugün tek bir uzmanımız var; tezde bir uzmanlar ekibi kuracağız — ve bu ekip birbiriyle iletişim kuracak."

> 💡 *Açıklanan terimler:* **Çoklu-ajan (Multi-Agent)** = tek bir karar verici yerine, işbirliği yapan birden çok yapay zekâ. **Sektör** = benzer iş yapan şirket grubu, örneğin bankalar. **Tek-ajanlı (baseline)** = şu anki, tek karar vericili sistem.

---

## Slayt 13 — Katkılar ve Tez Yol Haritası  ⏱️ ~1 dk

"Toparlayalım. Bu seminer ön çalışmasıyla ortaya ne çıktı? Beş ana katkı var, soldan sağa:

Birincisi, BIST-30 için sadece fiyata değil, makro ve temel verilere de bakan **çok kaynaklı bir veri mimarisi.**

İkincisi, beş modeli birleştiren **ensemble tahmin katmanı.**

Üçüncüsü, riski cezalandıran, yani güvenli kazanmayı öğreten **risk-duyarlı karar mekanizması.**

Dördüncüsü, kararlarının nedenini açıklayabilen **şeffaf yapay zekâ.**

Beşincisi ve geleceğe bakanı: Tüm bu yapı, **çoklu-ajan iletişimine genişlemeye hazır** şekilde, ölçeklenebilir biçimde kuruldu.

Yani bugün size çalışan, sağlam bir temel sundum. Yüksek lisans tezimde bu temeli, ajanların birbiriyle konuştuğu çok daha gelişmiş bir sisteme dönüştürmeyi hedefliyorum."

> 💡 *Açıklanan terim:* **Ölçeklenebilir** = sistem büyüdüğünde, yeni parçalar eklendiğinde çökmeden çalışabilen yapı.

---

## Slayt 14 — Kapanış: Sistem Demosu ve Araştırma Çıktıları  ⏱️ ~45 sn

"Sunumumun sonuna geldim. Dilerseniz bu noktada sistemin canlı arayüzünden kısa bir demo gösterebilirim — verinin nasıl aktığını, tahminlerin ve kararların ekranda nasıl göründüğünü.

Beni dinlediğiniz için çok teşekkür ederim. Sorularınızı, eleştirilerinizi ve katkılarınızı almaktan büyük memnuniyet duyarım."

> 💡 *Demo notu:* Vakit varsa dashboard'u aç; yoksa bir-iki ekran görüntüsü göster. Soru-cevaba hazır ol — özellikle Slayt 9 (ödül), Slayt 11 (Sharpe) ve Slayt 8 (veri sızıntısı) konularına.

---

## Sunucu İçin Hızlı Hatırlatmalar

- **Tempo:** Tanım slaytlarında (5, 6, 9, 11) yavaşla; dinleyici terimleri burada öğreniyor.
- **Her teknik terimi ilk geçtiğinde bir cümleyle tanımla** — yukarıdaki 💡 kutuları bunun içindir.
- **Üç hassas konuya hazırlıklı ol:**
  1. Ödül "Getiri − λ·Risk" sadeleştirmedir; gerçekte 6 bileşenli (Slayt 9).
  2. PSR = Portfolio-Sharpe-Returns (kodla tutarlı).
  3. Açıklanabilirlik SHAP iledir; mevcut ajanda 'attention' yoktur (o gelecekteki çoklu-ajan hedefi).
- **Bağlamı koru:** Bu bir tez savunması değil — "seminer ön çalışması", "ileride tezde…" dilini kullan.
</content>
