# Faz 11 — Entegrasyon ve sanal parayla simülasyon: plan

**Sürüm:** v0.17 · 2026-11-26 · Açılır listeler ve takvim, koyu tema §3.16; arayüz metinleri ve kullanılabilirlik §3.15; ana sayfa ve arayüz güncel duruma §3.14; simülasyon sayfası yeniden tasarımı §3.13; ileri veri deposunun VDS'ye devri §3.12; S0–S7 bitti; S6 eki etiketle donduruldu; ilk gerçek çalışma bulgusu §3.9; lot tablosu §3.10; İş Yatırım eski yanıt §3.11
**Durum:** Taslak. Kullanıcı niyeti (2026-11-22): Faz 10 sonrasında MARL eğitimi panoya ve portföy sistemine entegre edilir, sanal parayla simülasyon yapılır. Karar bekleyen maddeler §6'da. **S0 bitti (§3.1). S2 bitti (§3.2). S1 bitti (§3.3). S3 bitti (§3.4). S4 bitti (§3.5).** Kullanıcı (2026-11-22): K-F11-1..6'da öneriler uygulanır.
**Dayanak:**
- [Faz 10 planı](phase-10-marl-plan.md) §9 (araştırma kapanışı, devredilenler), §3.3 K1–K10 (karar katmanı), §6 K13, K14, K27, §8 #18.
- [M8 tasarımı](phase-10-m8-design.md) §6.7 (ileri veri), §7 (okuma; §7.3 b kural portföyleri, §7.5 tohum kümelemesi).
- `app/CLAUDE.md` (kimlik, çalışma alanı, kağıt portföy), `marl/CLAUDE.md` (iki katman K9, aynı fonksiyon ilkesi).

---

## 0. Özet

1. **Simülasyon ile ileri veri testi aynı iştir.** M8 ön kaydı (§6.7) 2026 katının seçilmiş politikalarını (kol başına 12; SHA'ları `m8_reading.json` → `forward_policies_2026`), dondurmadan sonraki ilk seanstan (2026-11-23) itibaren en az 126 seansla bir kez okur. Her gün çalışan sanal para simülasyonu bu günleri biriktirir. 126 seans ~6 ay eder, ~Haziran 2027. Bu süre kısaltılamaz.
2. **İki tür portföy ayrı tutulur.**
   - **Dondurulmuş (araştırma) defterleri:** M8'in 72 politikası (6 kol × 12 tohum, boş kol dahil; ön kayıt listesi `forward_policies_2026`) ve 4 kural portföyü (1/N, al-tut, min-varyans, HRP). Hiçbiri değiştirilmez, yeniden eğitilmez. İleri veri okuması bunlardan yapılır.
   - **Ürün portföyleri:** kullanıcının seçtiği ya da yeniden eğittiği stratejiler. İleri teste girmez.
3. **Kural portföyleri her ekranda öğrenen kolların yanında durur.** M8'de 20 bps'de 6 kat ortalamasında hiçbir öğrenen kol bunları geçmedi (M8 §7.3 b). Simülasyonda yalnız RL'yi göstermek yanıltıcı olur.
4. **Kabul kapısı parite:** simülasyon defteri, aynı politika ve aynı pencere için M8 değerlendirmesiyle (`eval_fold_m8.py`) aynı günlük getiri serisini üretmeli (§3, S3). Bu sağlanmadan simülasyon sonucu ileri veri sayılmaz.
5. **En büyük teknik engel:** M8 politikaları BenchMARL deney durumu olarak kayıtlı (`policy.pt`, `ckpt/*.pt`). Yalnız `venv-marl` ile yüklenebiliyorlar. K9 gereği üretim kodu `venv-marl`'a bağımlı olamaz. Bu yüzden politikalar sade bir ileri geçişe (numpy ya da düz torch) aktarılır ve parite testinden geçer (S2).

## 1. Neden şimdi

- Faz 10'un araştırma sorusu cevaplandı (plan §9). Devredilen ürün işleri (K13, K14, K27, §8 #18) bu fazda kapanır.
- İleri veri penceresi 2026-11-23'te açılıyor. Fiyatlar zaten kaydediliyor, politikalar sabit. Bu yüzden simülasyon geç başlasa da veri kaybolmaz: defterler geriye dönük olarak doldurulabilir (S4). Yine de günlük hattın erken kurulması operasyonel hataları ileri veri okumasından önce yakalar.
- Araştırma çıktıları şu an yalnız `results/marl_m8/` altında ve git dışı (yaklaşık 2,5 GB; yalnız 2026 katı 417 MB). Bu politikaların kalıcı ve yedekli bir yere alınması ileri veri okumasının ön koşulu (S0).

## 2. İlkeler

- **Aynı fonksiyon (K2/K5/K6):** simülasyonda gözlem `obs_spec.build_obs_at`, hedef `weights.target_weights` + `weights.project`, adım `accounting.step_fractional` ile hesaplanır. Kopya mantık yazılmaz.
- **Yürütme zamanı (K12, §8 #18):** karar t kapanışında, yürütme t+1 açılışında, aynı maliyet (20 bps gidiş-dönüş, tek yön c = 10 bps). Kağıt portföyün bugünkü "aynı gün kapanışı" davranışı simülasyon defterlerinde kullanılmaz.
- **Dondurma:** dondurulmuş defterlerin politika SHA'sı her gün denetlenir. SHA uyuşmazsa o gün yazılmaz ve hata verir.
- **Veri kapısı:** ileri panel M8 paneliyle aynı veri kapısından ve `m8-v1` düzeltme kurallarıyla kurulur (Faz 10 plan sürüm geçmişi 2026-11-20; M2 tasarımı §6.1). Geçmiş satırlar `panel_m8.npz` ile bit-eş kalır; yalnız yeni satır eklenir. Ek yalnızca sona yapılır.
- **Tez sonucu:** ileri veri okuması bir analiz betiğinden ve ayrı bir ön kayıt ekinden çıkar (M8 §6.7). Pano yalnız aynası olur. Parite testi (S3) ikisinin aynı sayıyı verdiğini garanti eder.
- **Kimlik ve çalışma alanı:** yeni yazma uçları `RequireWriter`/`RequireAdmin` taşır. Arka plan işi kullanıcı kimliğini `ws.use_workspace(user_id)` ile taşır. Dondurulmuş defterler kullanıcıya değil, ortak salt okunur bir araştırma alanına aittir (§6 K-F11-3).

## 3. İş paketleri

| # | Paket | İçerik | Kabul ölçütü |
|---|---|---|---|
| **S0** | Dondurulmuş politika arşivi | **BİTTİ (§3.1).** 2026 katının seçilmiş politikaları (72) + `meta.json`, `norm_stats.json` salt okunur bir yapıt deposuna kopyalanır; SHA manifesti `forward_policies_2026` ile karşılaştırılır. Tüm `results/marl_m8/` için yedek (git dışı) | Manifest SHA'ları 72/72 eşleşir. Yedekten geri yükleme bir kez denenir |
| **S1** | İleri panel | **BİTTİ (§3.3).** `prepare_panel_m2.py --data-fix m8-v1` hattıyla artımlı panel sürümü (`panel_m8_fwd`). Günlük ek: yfinance artımlı çekim → veri kapısı → `m8-v1` kuralları. 2026-11-09 sonrası fiyat marjı takvimi için `PRICE_MARGIN_VERIFIED_UNTIL` kaynakla yeniden denetlenir (`price_limit_audit_m8.py`) | Geçmiş satırlar `panel_m8.npz` ile bit-eş. Veri kapısı reddettiği günü yazmaz, sebebini kaydeder. Marj kapsamı dışındaki gün hata verir (sessiz geçmez) |
| **S2** | Politika aktarımı (K9) | **BİTTİ (§3.2).** BenchMARL politikasından ağırlıklar okunur, sade bir ileri geçişe aktarılır: M-none / S-esm / S-agr için paylaşımlı MLP, M-gat / M-self için `comm_models` ileri geçişi. Ana venv'de çalışır, `venv-marl` import etmez | Aynı gözlemde logit farkı ≤ 1e-6, 72/72 politika (`venv-marl` eval çıktısıyla). Önce bir spike: tek politika (M-none s201) |
| **S3** | Simülasyon motoru ve defter | **BİTTİ (§3.4).** Strateji başına defter (K14): kesirli mod, `accounting.step_fractional`, t+1 açılış yürütmesi, rf A8. Kural portföyleri `marl.rule_baselines.rule_series` ile aynı defter biçiminde | **Parite:** 2026 test penceresi (2026-01-01..2026-10-31) defterden yeniden oynatıldığında her politikanın günlük R serisi `eval_fold_m8.py` test serisiyle bit-eş (ya da ≤ 1e-12). Kural portföyleri `m8_reading.json` → `baselines.2026` ile aynı Sharpe |
| **S4** | Günlük iş ve geriye dönük doldurma | **BİTTİ (§3.5).** Seans kapanışından sonra (BIST kapanışı + veri gecikmesi) çalışan iş: panel eki → karar → defter. İdempotent; kaçırılan günleri sırayla doldurur. Tetik `app/services/background_jobs.py` ya da işletim sistemi zamanlayıcısı (§6 K-F11-4) | Aynı gün iki kez çalışınca defter değişmez. 5 gün boşluk sonrası doldurma, günlük çalışmayla bit-eş |
| **S5** | Pano (K27) | **BİTTİ (§3.7).** Strateji seçimi (eski tek ajan / SARL / MARL / kural), karşılaştırma görünümü (yalnız aynı pencere ve aynı maliyetteki defterler yan yana), tohum dağılımı (ortalama ± std, 12 tohum), N, "tez sonucu değildir" notu. Eski ortam modelleri ayrı sekmede, "doğrudan kıyaslanmaz" notuyla | `dashboard/CLAUDE.md` kuralları (tema, uzun işler). Yazma uçları RBAC'li (test) |
| **S6** | İleri veri ön kayıt eki | **Ek v0.3 + okuma betiği hazır (§3.6); dondurma bekliyor.** M8 §6.7 ölçütü sabit kalır. Ek, okumadan önce yazılır: tohum kümelemesi baştan beyan edilir ve kümeli istatistik eklenir (M8 §7.4, §7.5), panel sürümü ve SHA'lar sabitlenir | Ek, 126. seanstan önce commit'lenir ve dondurulur (etiket) |
| **S7** | Kağıt portföy uyumu (K13, §8 #18) | **BİTTİ (§3.8).** Mevcut tek ajan günlük kararı: eski modellerde normalizasyon uyarısı (K13). Kullanıcının elle "Uygula" akışı kalır ama yürütme fiyatı ertesi açılış olur ya da fark sınırlılık olarak gösterilir (§6 K-F11-5) | `test_daily_trading`, `test_daily_trading_tz`, `test_paper_portfolio` yeşil. Golden değişmez |

**Sıra:** S0 → S2 (spike, sonra 72 politika) → S1 → S3 → S6 → S4 → S5 → S7.
- S0 ve S6 düşük maliyetli ama geri dönüşü zor. Önce gelir.
- S3'ün parite kapısı geçmeden S4 ve S5'e başlanmaz.

### 3.1 S0 sonucu (2026-11-22; `measurements/phase11-s0/s0_archive.json`)

- **Arşiv:** `data/marl/frozen/m8-2026/` (git dışı, ortak). 72 politika × 5 dosya (seçilen kontrol noktası, `meta.json`, `norm_stats.json`, `train.json`, `fold_eval.json`) ve `manifest.json`.
  - Manifest kaynağı `m8_reading.json`; dosyanın SHA-256'sı manifestte kayıtlı.
  - Doğrulama (`marl/frozen.verify_archive`) üç şeyi denetler:
    - her dosyanın SHA-256'sı,
    - politika özeti `params_sha`: ana venv'de, eğitimdeki `state_digest` tanımıyla yeniden hesaplanır,
    - ön kayıt listesiyle birebir eşlik: 72/72.
  - Ana venv'de torch ile yüklenir; BenchMARL gerekmez. Bu, S2 için iyi işarettir.
- **Yedek:** harici diske (kullanıcı kararı: yerel HDD; bulut şimdilik yok), 8.660 dosya, 2,76 GB.
  - Kapsam: `results/marl_m8` ve `data/marl` (panel, İş Yatırım kaynak önbelleği, arşiv dahil).
  - Her dosya hedefte yeniden hash'lendi; hepsi eşleşti. Yedek artımlıdır: `archive_s0.py --backup <hedef>` yeniden koşulunca yalnız değişen dosyaları kopyalar.
- **Geri yükleme denemesi:** yedekten geçici dizine alınan arşiv, `verify_archive` ile 72/72 doğrulandı.
- **Kod:**
  - `marl/frozen.py`: `state_digest` artık tek tanım; `m3_runner` buradan ithal eder (`test_marl_m3` 20/20).
  - `scripts/marl/archive_s0.py`.
  - `tests/test_marl_frozen.py`: 3 kontrol.
- **S4 için kural:** günlük iş her çalışmadan önce `verify_archive(..., forward=...)` çağırır; uyuşmazlıkta o gün yazılmaz.

### 3.16 Açılır listeler ve takvim, koyu tema (2026-11-26; kullanıcı bildirimi; `tests/test_theme_contrast.py` 97)

- **Kullanıcı bildirimi.** Koyu temada açılır liste seçimlerinde kötü deneyim. Önce ölçüldü: yerel sunucu + Edge, koyu ve açık tema, 7 açılır liste (tekli, çoklu, aramalı, üst çubuk araması) ve 2 tarih seçici; ekran görüntüsü ve hesaplanmış stil.
- **Ölçülen sorunlar ve çözüm (`dashboard/assets/custom.css`).**
  - Arama satırının kapsayıcısı (`.dash-dropdown-search-container`) Dash varsayılanıyla beyazdı (255,255,255): koyu menünün ortasında beyaz çerçeve + Dash'in mor odak kenarı. → yüzey tokenı, alt çizgi, birincil renkte odak.
  - Seçenek metninin span'i satırdan ayrı bir zemin alıyordu (metnin arkasında koyu kutu). → iç span'ler saydam.
  - Çoklu seçimde seçili satırların hepsi "üzerine gelinmiş" renginde: fare nerede, belli değildi. → çoklu seçimde satır boyanmaz, durum onay kutusunda (birincil renk); üzerine gelinen satır `surface-hover`; tekli seçimde seçili satır birincil tonda + solda şerit, gereksiz radyo düğmesi gizli.
  - Takvim: Dash 4 `td.dash-datepicker-calendar-date-*` çiziyor; eski `[role="gridcell"] button` kuralları uymuyordu. Seçili gün ve odak çerçevesi Dash morunda (127,75,196). → birincil renk, ay dışı günler soluk, gezinme düğmeleri tema renginde.
- **Türkçe iç metinler.**
  - `dcc.Dropdown`: "Search", "Select All", "Deselect All", "N selected", "No options found" → `dashboard/components/dropdown_tr.py` (`labels` varsayılanı, `app.py` sayfalardan önce kurar; çağrıda açık `labels` kazanır). Sayı yer tutucusu `{num_selected}`: belgede yok, tarayıcıda denenerek bulundu (`{num}`, `{count}` çizilmiyor).
  - Takvim: ay ve gün adları bileşende sabit İngilizce, dil seçeneği yok, tarayıcı dili (tr-TR ile denendi) etkilemiyor. `dashboard/assets/datepicker-tr.js` yalnız takvim kapsamında tam eşleşen metni çevirir (SVG grafik eksenleri kapsam dışı). İlk sürümde Salı "Sa" İngilizce Saturday "Sa" ile çakışıp ikinci turda "Ct" oldu (tarayıcıda yakalandı); üç harfli kısaltmalar (Pzt, Sal, Çar, Per, Cum, Cmt, Paz). Hafta pazartesi başlar (`first_day_of_week=1`).
- **Test.** `test_theme_contrast` 92 → 97 (bölüm 11): arama kapsayıcısı ve seçili gün token kullanıyor, çoklu seçimde seçili satır boyanmıyor, Dropdown Türkçe varsayılanı + `{num_selected}`, gün çevirisi çakışmasız (mutasyon: "Sa" → "Sa" yakalanır). `test_dash_props` 4, `test_topbar` 43, `test_account_profile` 84, `test_guide` 9. Görsel: iki tema, 9 bileşen; konsol hatası 0.
- **Sınır.** Yerel `<select>` (Kullanıcılar sayfasında rol) tarayıcının kendi açılır penceresi; `color-scheme: dark` ile koyu çizilir, ekran görüntüsüne girmediği için görsel olarak doğrulanmadı.

### 3.15 Arayüz metinleri ve kullanılabilirlik (2026-11-25; kullanıcı isteği; mevcut testler, beklentileri güncellendi)

- **Kullanıcı isteği (2026-11-25).** §3.14'te kapsam dışı bırakılan eski sayfalar: "yapabileceğimiz diğer arayüz iyileştirmelerini de yapalım". Önce bütün sayfaların ekran görüntüsü alındı (yerel sunucu, Edge); iş o taramadan çıkan somut sorunlarla sınırlı.
- **Türkçe metin (9 sayfa + bileşenler).** Ekranda görünen dizeler Türkçe karakterle yazıldı. Kör bul-değiştir yapılmadı:
  - görünen dize konumları AST ile çıkarıldı (bileşen children, label, title, placeholder; sütun adları; etiket sözlükleri) ve kesin karşılığı olan kelime haritası uygulandı;
  - kimlik, className, value, href, sözlük anahtarı, karşılaştırmaya giren dize ve tanımlayıcı biçimli dizeler (`gram_altin_try`) dışarıda tutuldu; karşılaştırma çakışması 0;
  - birden çok satıra yayılan dizeler elle düzeltildi; fark satır satır okundu.
  - `<html lang="tr">`: CSS büyük harf dönüşümü "takip"i "TAKIP" yapıyordu; dil işaretiyle "TAKİP".
  - Üst çubuk görünüm düğmesi "Aydınlık".
- **Ortak biçim (`dashboard/components/fmt.py`).** Para "100.000,00 TL" (binlik nokta, ondalık virgül, gerçek eksi; simülasyon sayfasıyla aynı), yüzde "+%5,20", saat TRT. Günlük karar, Modeller ve Tahmin `f"₺{x:,.2f}"` yerine bunu kullanır.
- **Saat.** Hesabım ve Kullanıcılar sayfasında kimlik kayıtları (naive UTC) TRT'ye çevrilip "TRT" etiketiyle yazılıyor. Kullanıcılar sayfasında son giriş daha önce etiketsiz UTC'ydi.
- **Hata: hiperparametre durum rozeti.** Kod `"complete"` arıyordu, API `"completed"` döndürüyor: tamamlanan çalışmalar gri ve İngilizce "COMPLETED" görünüyordu. `fmt.status` ikisini de tanır ("Tamamlandı", yeşil); deneme satırları ve detay da aynı eşlemeyi kullanır.
- **Veri sayfası dosya listesi.** `data/**/*.csv`'nin tamamı (ileri veri önbelleği dahil 364 dosya) alt alta yazılıyordu: sayfa ~15.000 px. Aynı ad (`raw_stock_data.csv`) farklı klasörlerde ayırt edilemiyordu. Artık klasör sütunlu, sıralanır, süzülür, 12'lik sayfalı tablo; dosya/klasör/toplam boyut özeti. Sayfa ~2.000 px.
- **Boş durumlar.** Akademik sayfada model yokken dört boş grafik ve boş tablo yerine tek kart ("Eğitime git", "Dashboard"). Modeller ve Günlük karar'da boş grafik 450 → 160 px, ne yapılacağını söyleyen metinle.
- **Kapsam notu.** Tek ajanlı sayfalarda (Modeller, Akademik, Eğitim, Hiperparametre, Günlük karar) başlık altında: bu sayfa Faz 1–9 tek ajanlı modeller içindir; M8 ve simülasyon Dashboard / Simülasyon / Rehber'de. Tahmin sayfasında: bilgi kapısında (P5) yön bilgisi tespit edilmedi, Rehber kartına bağlantı. `create_page_header(note=...)`.
- **Başlıklar.** "Günlük Trading" → "Günlük Karar" (menüyle aynı), "Hiper Parametre" → "Hiperparametre", Günlük karar kartı "Trading Kararları" → "Karar Ayrıntısı".
- **Test.** Davranış değişmedi; metin beklentileri güncellendi: `test_account_profile` 84 (3 beklenti: "bu tarayıcı", "Giriş", "boş olamaz"), `test_training_status` 19 (1: "eğitiliyor"). Ayrıca `test_dash_props` 4, `test_theme_contrast` 92 (`page-scope-note`), `test_topbar` 43, `test_delete_artifacts` 32, `test_theme_preference` 31, `test_paper_execution` 4, `test_paper_portfolio` 43, `test_workspace_isolation` 20, `test_training_eta` 57, `test_guide` 9. Görsel kontrol: 9 sayfa yeniden çekildi; konsol hatası 0.
- **Kapsam dışı.** `dcc.Dropdown`'ın kendi metni ("30 selected") Dash bileşeninden geliyor; tek ajanlı sayfaların iç akışı (form alanları, metrikler) değiştirilmedi.

### 3.14 Ana sayfa ve arayüzün güncel duruma getirilmesi (2026-11-25; kullanıcı isteği; `tests/test_marl_forward_api.py` 6, `tests/test_guide.py` 9)

- **Kullanıcı isteği (2026-11-25).** Pano ana sayfasında simülasyon sonuçları ve en iyi performans veren model gösterilsin; arayüz sistemin bugünkü hâlini yansıtsın. Ana sayfa Faz 1–3 döneminin tek ajanlı modellerini "en iyi model" diye gösteriyordu; rehberde M8 yoktu ve "son sınav dönemine bakılmadı" yazıyordu.
- **"En iyi model" nasıl gösterilir (ön kayıt sınırı).** M8 en iyi kolu seçmez (M8 §6.5); "MARL tek ajandan iyi mi" sorusunu yanıtlar, "basit kurallardan iyi mi" sorusunu yanıtlamaz (§7.3(b)). Bu yüzden kart bir sıralama değil:
  - öğrenen kollar arasında 72 hücre ortalaması en yüksek olan (boş kol hariç; o negatif kontrol) M-none, yanında aynı maliyette en yüksek kural portföyü (min-varyans) ve "4 kural portföyünün 4'ü öğrenen kolların hepsinin üstünde" cümlesi;
  - çubuk grafik 6 kol + 4 kural, 6 yıl ortalaması fazla Sharpe (20 bps), ± std yalnız ipucunda;
  - "Ön kayıtlı nihai deney" rozeti; etiketler Rehber'deki M8 kartlarında.
  - İleri veri (126 seansı dolmamış) "en iyi" seçiminde KULLANILMAZ; simülasyon kutusu ara görünüm uyarısını taşır (ek §9).
- **Sayıların kaynağı.** Pano ölçüm dosyası açmaz (imajda `docs/` yok). `scripts/build_guide.py` `snapshot.json`'a `m8_ozet` yazar: kol ortalamaları `descriptive.arms.<kol>.mean_sr`, kural ortalamaları `baselines.<kat>.<kural>.sr` 6 kat ortalaması (M8 §7.3(b) ile aynı: 1/N 0,54, al-tut 0,57, min-var 0,61, HRP 0,61; M-none 0,31). Kaynak SHA ve üretim commit'i `kaynaklar`'da; test kaynağa eşitliği denetler.
- **Ana sayfa (`dashboard/pages/home.py`).** Sıra: durum şeridi (ileri veri seansı n/126, son günlük çalışma TRT, marj denetimi ufku, sürüm) → öne çıkan model (M8) ve simülasyon kutusu (ileri veri / 2026 test; kol 12 tohum ort. ± std, 1/N, min-varyans) → katlanır "önceki fazların modelleri" (M8 ile aynı protokolde değil, karşılaştırılmaz). Yenileme 30 sn → 5 dk (defter günde bir değişir); her model için metrik isteği atılan eski "en iyi model" kartları kaldırıldı.
- **Rehber (`guide.yaml`, `glossary.yaml`).** M8 H1/H2/H3 kartları "Ön kayıtlı nihai deney" rozetiyle (yeni kart alanı `kademe: nihai`; R-K4 geliştirme kapıları için aynen geçerli, `system-guide-plan.md` §16). Bayat metin düzeltildi: "son sınav dönemine bakılmadı", "son sınav henüz açılmadı", "M6–M8 sonraki aşamalar", "simülasyonu hızlandırmak planda". Yeni maddeler: simülasyon panosu sonuç değildir; öğrenen modeller basit kurallardan iyi değildir; ana sayfadaki kart bir sıralama değildir (SSS). Sözlüğe `m8`, `sarl`, `ileri-veri-testi`, `kural-portfoyleri`.
- **Düzeltmeler.**
  - Son çalışma saati: günlük `finished` damgası saat dilimsiz `datetime.now()`; VDS konteyneri UTC olduğu için pano 19:16'yı 16:16 gösteriyordu. `marl_forward.info()` `finished_trt` ekler (sunucu kendi günlüğünü okur: naive → yerel → TRT). Defter ve günlük biçimi değişmedi.
  - Menü Türkçe karakterle; Simülasyon Analiz grubunda Dashboard'un altında; "Trading" → "Günlük karar". Üst çubuk "sıradaki adım": Dashboard → Simülasyon → Rehber.
  - Simülasyon sayfası: lejant çizim alanının üstüne alındı (4 satıra sarılınca çizgilere biniyordu); kıyas etiketleri Türkçe (`forward_views.ARM_LABELS`, yalnız görüntü etiketi).
  - `test_marl_forward_api` callback bulucusu kimlik sınırında eşleşir: `sim-chart.figure` ana sayfanın `home-sim-chart.figure`'ını seçip kontrol 5'i yanlış düşürüyordu.
- **Test.** `test_marl_forward_api` 6 (yeni 6: ana sayfa + öne çıkan M-none + 4 kural üstte + grafik = snapshot özeti + 2026 test kutusu + TRT damgası; arşivli makinede koşuldu, §3.13'ün koşulamayan 5. kontrolü de burada geçti), `test_guide` 9 (kademe rozeti, M8 özeti kaynağa eşit, bilinmeyen kademe mutasyonu), `test_theme_contrast` 92 (`home-` sınıf ailesi taranıyor), `test_dash_props` 4, `test_topbar` 43, `test_marl_sim_book` 5. Görsel kontrol: yerel sunucu + Edge (Playwright), açık ve koyu tema; ana sayfa, simülasyon ve rehber; konsol hatası 0.
- **Kapsam dışı (ayrı paket).** Eski sayfaların (Eğitim, Tahmin, Modeller, Akademik, Günlük karar, Veri, Hiperparametre) iç metinleri hâlâ Türkçe karaktersiz ve tek ajanlı döneme göre yazılı; ayrı bir arayüz paketi olarak önerildi.

### 3.13 Simülasyon sayfası yeniden tasarımı (2026-11-25; kullanıcı isteği; `tests/test_marl_sim_book.py` 5)

- **Kullanıcı isteği (2026-11-25).** Sayfada portföyde hangi hisselerin olduğu, günlük kazanç ve toplam K/Z anlaşılmıyor; grafikler zayıf. Çatı değiştirilmedi (Next.js değerlendirildi, reddedildi: 13 sayfa ve tema/RBAC bekçileri Dash'e bağlı; sorun bilgi tasarımıydı). Önce örnek veriyle etkileşimli prototip yapıldı ve onaylandı, sonra Dash'e uygulandı.
- **Sayfa (`dashboard/pages/simulation.py`).** Sıra:
  1. Pencere bandı: ileri pencerede "ara görünüm — tez sonucu değildir" ve seans ilerlemesi n/126; son çalışma ve marj denetimi aynı satırda.
  2. İncelenen portföy: kol + tohum ya da kural, sermaye (K-F11-6). Tam lot (K-F11-2); kol ortalamasının lot karşılığı olmadığı yazılır.
  3. Beş kart: değer, gün K/Z, toplam K/Z (maliyet dahil), en büyük düşüş (tepe → dip tarihi), 1/N kıyası (puan).
  4. Ortak zaman eksenli üç panel: değer (incelenen tam lot + kıyas çizgileri kesirli, kol bandı ± std), günlük K/Z çubukları, tepeden düşüş. Aralık 1A/3A/Tümü.
  5. Gün özeti: günün K/Z'si, bir cümle (en çok kazandıran / kaybettiren), hisse bazında katkı çubukları, emir listesi (alış/satış, lot × fiyat), nakit faizi ve maliyet.
  6. "Portföyümde ne var": halka grafik + tablo (lot, ort. maliyet, kapanış, gün %, piyasa değeri, ağırlık, gün K/Z, toplam K/Z, o günkü işlem; nakit ve toplam satırı).
  7. Dağılım zamanla (yığılmış alan), strateji karşılaştırma tablosu, 12 tohumun dağılımı.
  - **Gün seçimi:** grafiğe tıklama ya da önceki/sonraki düğmesi; kartlar, gün özeti, tablo, halka ve dağılım o günün kapanışına geçer. Karşılaştırma tablosu ve tohum grafiği pencere sonunu gösterir (yazılı).
- **Veri (yalnız okunan; ileri test, dondurulmuş defter ve S3 paritesi değişmez).**
  - `orders.step_integer` yanıtına `p_exec`, `p_close` eklendi (zaten hesaplanan değerler; hesaba girmez).
  - `sim_ledger.step` tam sayı satırı `fills` (gerçekleşen emirler: simge, yön, adet, yürütme fiyatı, komisyon; `integer_fills`) ve `p_close` taşır. Kesirli yol ve defter deposu (`rows.csv`) dokunulmadı.
  - `forward_views.daily_book`: gün gün lot, değerleme fiyatı, ortalama maliyet, birikimli gerçekleşen K/Z, hisse katkısı, nakit faizi, komisyon, emir listesi. `product` yanıtına `book`; `view` kol satırına `members` (tohum başına toplam getiri ve Sharpe).
  - **Tanımlar.** Ortalama maliyet komisyon HARİÇ hareketli ortalama (kâğıt portföyün `avg_cost`'u komisyon dahil; ikisi ayrı modül, sayfada yazılı). Hisse katkısı = gün sonu pozisyon değeri − önceki gün sonu pozisyon değeri − pozisyona giren net nakit. Özdeşlik: Σ katkı + nakit faizi − komisyon = günlük değer farkı.
- **Ön kayıt.** İleri pencerede tohum grafiğinin notu kıyas yorumu yapmaz ("126. seansta bir kez okunur"); test penceresinde 1/N üstündeki tohum sayısı yazılır. Ürün portföyü ileri teste girmez, kayıt tutmaz (değişmedi).
- **Test.**
  - `test_marl_sim_book` 5 (veri gerektirmez): gerçek `step_integer` adımlarıyla sentetik 59 gün. Özdeşlik ve lot × fiyat + nakit = değer en büyük göreli fark 2,9e-16. Ort. maliyetin bağımsız denetimi: birikimli katkı = lot × (fiyat − ort. maliyet) + gerçekleşen, 1,8e-16. İşlem göremeyen hissede emir yok. Mutasyon: emir fiyatı +0,01 → 6e-7; gerçekleşen yok sayılınca 6e-2. Render ve ön kayıt dili.
  - `test_marl_forward_api` 5 yeni callback'lere uyarlandı: günlük defter muhasebesi, 12 tohum satırı, grafik 9 iz. **Bu makinede arşiv yok; 5. kontrol arşivli makinede (ev / VDS) koşulmalı.**
  - `test_dash_props` 4, `test_theme_contrast` 92 (`sim-` sınıf ailesi taranıyor), `test_topbar` 43, `test_marl_core` 63, `test_guide` 9.
  - Sentetik veriyle ayrı Dash sürecinde görsel kontrol: sayfa açıldı, önceki gün düğmesi bütün bölümleri o güne taşıdı.
- **Sınırlılık.** Fiyatlar yfinance düzeltilmiş (marl/CLAUDE.md: tam sayı modunda lot/fiyat seviyesi iddiası yok). Ürün portföyü her istekte baştan hesaplanır (test penceresi ilk istekte ~saniyeler).

### 3.12 İleri veri deposunun VDS'ye devri (2026-11-25; `tests/test_marl_forward_job.py` 5, `scripts/marl/handover_parity.py`)

- **Kullanıcı kararı (2026-11-25).** Defterin tek yazarı VDS olur: sunucu sürekli açık, pano her yerden erişilir. Önceki karar (2026-11-24, `docs/deployment/ci-cd.md` §4: "şimdilik yerelde") bununla değişti.
- **Neden ayrı paket.** Ön kayıtlı ileri testin verisi pencere içinde makine değiştiriyor. İki koşul: (1) iki makine aynı anda yazmamalı (iki ayrı defter), (2) yeni makine aynı günler için aynı sayıyı üretmeli.
- **Tek yazar (`marl/forward_job.writer_check`).** `data/marl/forward/m8/WRITER` dosyası varsa içindeki kimlik `MARL_FORWARD_WRITER_ID` ile aynı değilse iş hiçbir dosyaya (defter, günlük, kilit) dokunmadan `yazar_degil` döner. Devirde iki kopyaya da `WRITER=vds` yazılır; VDS `.env`'i `MARL_FORWARD_WRITER_ID=vds`, `MARL_FORWARD_AUTORUN=True`. Yereldeki kopya salt okunur ayna olur.
- **Devir paritesi (`scripts/marl/handover_parity.py`).** Diskteki defteri aynı ileri panel ve dondurulmuş arşivden geçici dizinde baştan kurar; her satırı (`written_at` hariç) ve durum JSON'unu karşılaştırır. Gerçek depoya ve günlüğe dokunmaz. Yerelde 2026-11-25 (2 seans, 76 strateji, 152 satır) `bit_es`; mutasyon (bir `R` değerinde 1e-17) `fark_var`. Kabul: VDS imajında (torch 2.9.0 CPU; yerel 2.6.0) aynı betik `bit_es` der; demezse devir durur, fark ve ortam bu bölüme yazılır.
- **Taşınan küme (~160 MB).** `frozen/m8-2026` (72 politika), `frozen/m8-source`, `forward/m8` (panel, defter, çekimler, günlük), `panel_m8.npz`. Kopya her dosyanın SHA-256'sıyla iki tarafta karşılaştırılır (`--digest`). Eğitim koşuları (`results/marl_m8`) taşınmaz.
- **İmaj.** Günlük işin okuduğu `scripts/marl/forward_panel_m8.py`, `m8_reading.json` ve devir betiği imaja alınır (`.dockerignore`). İmajda git yok: çekim kaydındaki commit derlemede gömülen `GIT_COMMIT`'ten (önceden `git` yoksa iş çökerdi). VDS'de `./data/marl` bind mount (`docker-compose.prod.yml`), sahibi uid 10001.
- **Veri kuralı ve ek (S6) değişmedi.** Politikalar, kapı, %1 çapraz denetim, okuma kuralları ve pencere aynı; hesaplama ortamı değişikliği okumada beyan edilir.
- **Açık maddeler.** (a) Yedek yönü tersine döndü: defterin aslı VDS'de; E: diskine yedek VDS'den çekilerek alınmalı. (b) Aylık marj denetimi kaydı (`marl/margin_source_checks.json`) imajın içinde: her denetimden sonra yeni sürüm dağıtılır (sonraki ufuk 2026-12-21).
- **Devir sonucu (2026-11-25 00:50 TRT; `measurements/phase11-handover/`).**
  - Kopya: devir kümesi 1.045 dosya, her dosyanın SHA-256'sı iki tarafta aynı (`local_before.json`, `vds_after.json` → `digest`). Arşiv doğrulaması VDS'de 72/72.
  - Defter paritesi VDS'de **`fark_var`** (kabul ölçütü bit-eşlik; sağlanmadı). Ortam: yerel Python 3.12.5 / torch 2.6.0+cu124 / Windows; VDS Python 3.13.18 / torch 2.9.0+cpu / Linux; numpy ikisinde 2.2.6. Fark yalnız dikkatli iki MARL kolunda (M-gat, M-self; 152 satırın 53'ü): portföy değeri en çok 3,9e-4 TL (100.000 TL'de ~4e-9 göreli), günlük getiri `R` 2,6e-9, devir 2,5e-7. M-none, S-agr, S-esm, boş kol, 1/N ve al-tut bit-eş; min-varyans ve HRP ≤ 3e-11 TL.
  - Politika çıktısı (`logits_compare.json`, `handover_parity.py --logits-out`; her politika iki makinede aynı iki gözlemle): en büyük `loc` farkı 1,64e-7 (M-gat), 1,49e-7 (M-self), diğer kollar 0; eylem farkı ≤ 1,3e-6. S2 kabul ölçütü (logit farkı ≤ 1e-6) VDS'de de sağlanıyor.
  - **Kullanıcı kararı (2026-11-25): beyanla devir.** 2026-11-23 ve 2026-11-24 seansları yerelde yazıldığı gibi kalır; 2026-11-25'ten itibaren VDS yazar. **Sapma beyanı (ileri veri okumasına girer):** 126 seanslık pencerenin ilk 2 seansı ile kalanı farklı hesaplama ortamında üretildi; ortam farkı dikkatli iki kolda float32 son basamak düzeyinde (yukarıdaki büyüklükler), karar kuralı, politika ağırlıkları, veri ve maliyet aynı. Ortamı yerel ile eşitlemek (temel imaj Python 3.12 / torch 2.6) Windows ↔ Linux işlemci kütüphanesi farkı yüzünden bit-eşliği garanti etmediği için seçilmedi.
  - VDS `.env`: `MARL_FORWARD_WRITER_ID=vds`, `MARL_FORWARD_AUTORUN=True` (v2.25.1). Yerel kopyada `WRITER=vds`: yerel iş `yazar_degil` döner.

### 3.11 İş Yatırım eski yanıt ve zaman aşımı (2026-11-24; `tests/test_marl_forward_panel.py` 6)

- **Ne oldu.** 2026-11-24 seansı 20:46, 21:16 ve 21:46 denemelerinde `isy_bekleniyor` ile durdu. 22:16 denemesinde tek bir İş Yatırım isteği (EKGYO) 120 sn'de zaman aşımına düştü ve bu hata bütün panel güncellemesini düşürdü (log `panel_update.error`, 22:18). Defter 2026-11-23'te kaldı; hiçbir gün yanlış yazılmadı, kapı tasarlandığı gibi durdu.
- **Neden (ham yanıtlar `data/marl/forward/m8/fetch/<zaman>/`).** Gecikme sembol sembol yayımlama değil. Aynı URL istekten isteğe farklı yanıt veriyor: kimi yanıt 24 Kasım'ı içeriyor, kimi bir önceki günde bitiyor (ARCLK 20:46'da var, 21:16'da yok, 21:46'da var; TSKB tersi). Üç denemenin birleşiminde 30 sembolün 23'ünde 24 Kasım en az bir kez geldi. Kapı tek denemede 30/30 istediği için bu düzende neredeyse hiç açılmaz. Ayrıca tek isteğin hatası yeniden denenmeden bütün işi düşürüyordu; `urlopen(timeout=120)` her okumada sıfırlandığı için istek başına toplam süre sınırı da yoktu.
- **Düzeltme (`scripts/marl/forward_panel_m8.py::fetch_isy`).** Beklenen gün (yfinance'ta bugüne kadarki son aday seans) yanıtta yoksa sembol en çok 6 kez, 3 sn arayla yeniden istenir; yeniden denemede URL'ye zaman damgası eklenir (ilk istek eskisiyle aynı). İstek başına toplam süre 60 sn. Hata ya da hep eski yanıt sembolü eksik bırakır (kapı `isy_bekleniyor` der); önceden bir hata bütün işi düşürüyordu. Denemeler log'a `isy_fetch` olarak yazılır.
- **Veri kuralı değişmedi.** Kaynak, %1 çapraz denetim, kapı ve ham yanıtın SHA'sıyla saklanması aynı; yalnız eski önbellekten gelen yanıt reddediliyor. S6 eki okuma kuralları ve pencere değişmedi.
- **Seçenekler (kullanıcıyla 2026-11-24).** Başka kaynak ya da yalnız yfinance, dondurulmuş çapraz denetim kuralını pencere içinde değiştirirdi (sapma beyanı gerekir); önce bu düzeltme denendi.

### 3.10 Ürün portföyü lot tablosu (2026-11-23; `tests/test_marl_forward_api.py` 5)

- **Kullanıcı isteği:** hangi hissede kaç lot olduğu tabloda görünsün.
- **Sayfa** (`/dash/simulation`, Ürün portföyü → Hesapla): kartların altında "Eldeki hisseler" tablosu. Sütunlar: hisse, lot (1 lot = 1 pay), kapanış, piyasa değeri, tam lot ağırlığı, kesirli defterin aynı günkü ağırlığı, son gün işlemi (alınan + / satılan − adet); altta nakit ve toplam satırı.
- **Veri:** `forward_views.product` yanıtına `holdings`, `holdings_date`, `cash_integer_tl`, `cash_weight_frac` eklendi. `sim_ledger.step` tam sayı modunun gün satırı eldeki adedi ve nakdi taşıyor; kesirli yol ve defter deposu değişmedi.
- **Sınırlılık:** fiyatlar yfinance düzeltilmiş kapanış (marl/CLAUDE.md: tam sayı modunda lot/fiyat seviyesi iddiası yok). Ürün portföyü anlık hesaptır, ileri teste girmez.
- **Test:** S5 kontrolüne muhasebe özdeşliği eklendi: Σ lot × kapanış + nakit = tam sayı defterinin son değeri (göreli 1e-6), adetler tam sayı ve negatif değil; ürün callback'i tabloyu içeriyor.

### 3.9 İlk gerçek günlük çalışma bulgusu (2026-11-23; `tests/test_marl_forward_panel.py` 5)

- **Ne oldu.** 2026-11-23 21:30 TRT ilk gerçek ileri gün çalışmasında panel güncellemesi `check_prefix` bekçisinde durdu: "ileri panel önceki satırları değiştirdi". Defter kurulmadı (`baslamadi`), hiçbir dosya yazılmadı; bekçi tasarlandığı gibi çalıştı.
- **Neden.** `scripts/marl/forward_panel_m8.py` önceki ileri satırları `rows.csv`'den pandas'ın varsayılan float ayrıştırıcısıyla okuyordu. Bu ayrıştırıcı son basamakta (1 ULP) sapabiliyor: 2026-11-10..20 arasındaki 9 seansta 12–69 hücre, en büyük fark 6,8e-13 (CCI), fiyatlarda 5,7e-14. Geçmiş 3.364 satır etkilenmedi. 2026-11-22'deki ikinci çalışmada yeni gün olmadığı için panel yeniden kurulmamış, hata ilk yeni günde görüldü.
- **Doğrulama (salt okuma, dosya yazmadan).** Aynı `rows.csv` `float_precision='round_trip'` ile okununca yeniden kurulan panel diskteki panelle 10 dizide bit-eş (fark 0).
- **Düzeltme.** `marl/forward_panel.py::read_rows` (round_trip) tek okuma yolu; S4 defteri zaten `round_trip` okuyordu (`forward_job.py`). Davranış değişikliği yok: diskteki panel ve satırlar aynı, yalnız geri okuma kayıpsız.
- **Test boşluğu kapandı.** S1 test 4 ileri satırları bellekten veriyordu; CSV'ye yazma ve geri okuma yolu denenmiyordu. Yeni kontrol 4a: yaz → `read_rows` → yeniden kurulum bit-eş; varsayılan `read_csv` ile aynı yol `check_prefix`'te hata veriyor (bekçinin bu kaybı yakaladığının kanıtı).
- **Ek (S6) ile ilişki.** Okuma kuralları ve pencere değişmedi; defter deposu yine ilk ileri gün panelde olunca, etiketten sonra kurulur.

### 3.8 S7 sonucu (2026-11-22; `tests/test_paper_execution.py` 4)

- **"Uygula" ertesi açılışta yürütür (K-F11-5, öneri uygulandı; plan §8 #18).**
  - `POST /trading/apply-decision?execution=next_open` varsayılan.
  - Karar t kapanışında verilir; işlem t+1 seansının açılış fiyatıyla yapılır, simülasyon defteriyle aynı yürütme zamanı (`portfolio.reprice_to_open`, `daily_trading.next_session_opens`).
  - Adet aynı kalır. Maliyet, gelir ve komisyon açılış fiyatıyla ve kararın kendi komisyon oranıyla yeniden hesaplanır.
  - Açılışı olmayan sembolde işlem yapılmaz. Nakit yetmezse mevcut bekçi işlemi atlar ve bildirir.
  - **Ertesi seans henüz yoksa 409.** Pano bunu hata değil bekleme olarak gösterir (mavi bilgi kutusu).
  - Eski davranış yalnız açıkça `execution=decision_close` ile; yanıtta sınırlılık notu var.
- **Canlı gözlem uyarıları (K13; davranış değişmedi).** `/daily-decision` özeti `observation_warnings` taşır; pano karar kartlarının altında gösterir.
  - **Bulgu:** canlı karar (`build_live_state`) eğitim `price_stats`'ını ŞU AN KULLANMIYOR. Fiyatlar 50/50 yedek ölçekle z-skorlanıyor, gösterge legacy ADX.
  - Bu yüzden her modelde en az bir uyarı çıkar. Ek uyarılar: `causal` fiyat tanımı, `wilder` ADX, meta yok.
  - **Düzeltme bu paketin kapsamı dışı** (canlı karar davranışını değiştirir). Faz 10 K13 kararı yeni modellerde istatistiğin kullanılmasını öngörüyor → açık ürün işi.
- **Test:**
  - `test_paper_execution` 4,
  - `test_paper_portfolio` 43, `test_daily_trading_tz` 18, `test_panel_freshness` 21, `test_trade_universe` 22,
  - `test_prediction_regression` GEÇTİ (golden değişmedi),
  - `test_dash_props` 4, `test_theme_contrast` 92.
  - `tests/test_daily_trading.py` otomatik test değil: açık sunucuya ve sabit model adına bağlı, `input()` bekliyor. Kabul ölçütündeki yeri `test_paper_execution`'la karşılandı.

### 3.7 S5 sonucu (2026-11-22; `tests/test_marl_forward_api.py` 5, `tests/test_marl_sim_ledger.py` 5)

- **Sayfa** `/dash/simulation` ("Simulasyon", İşlem grubu; `dashboard/pages/simulation.py`).
  - **Durum kartları:** ileri seans n/126, son defter günü, son çalışma, marj denetiminin geçerlilik sonu.
  - **Yönetici "Şimdi çalıştır" düğmesi.** Sayfa açılışta işin durumunu sorar; iş çalışıyorsa yoklama açılır (pano uzun iş kuralı).
- **Karşılaştırma yalnız aynı pencere ve aynı maliyetteki defterler arasında.** Pencere seçimi iki seçenekli:
  - **İleri veri** (2026-11-23 →; ara görünüm notuyla: tez sonucu değil, ek §9),
  - **2026 test penceresi** (M8'de okunmuş geçmiş; S3 defteriyle yeniden oynatım, arşiv SHA'sıyla anahtarlı önbellek).
- **Kol çizgisi** 12 tohumun ortalaması ± std bandı, N gösterilir. **Varsayılan görünüm (K-F11-1):** MARL iletişimsiz + 1/N + min-varyans; diğer kollar ve kurallar seçilebilir.
  - Özet tablo: toplam getiri, fazla getiri Sharpe, en büyük düşüş, devir ve nakit; her biri ortalama ± std.
- **Ürün portföyü (K-F11-2):** seçilen strateji, kullanıcının sermayesiyle (varsayılan 100.000 TL, K-F11-6) tam sayı lotla oynatılır ve kesirli defterle karşılaştırılır.
  - Gösterilenler: son değer farkı, yıllık izleme hatası, ortalama ağırlık açığı (yuvarlama), emir sayısı.
  - Ürün portföyü ileri teste girmez ve kayıt tutmaz (hesap anlık).
  - Kullanıcı başına kalıcı ürün portföyü bu pakette yok. Kâğıt portföy akışı S7'de.
  - Tam sayı muhasebesi ortamla aynı fonksiyon: `orders.step_integer`, `MarlPortfolioEnv._step_integer`'dan taşındı. main'deki ortamla 80 adım bit-eş, `test_marl_env` 36; M2 tasarımı v0.12.
- **Eski tek ajan modelleri** ayrı kartta, "doğrudan kıyaslanmaz" notuyla (Modeller ve Günlük karar sayfalarına bağlantı).
- **Uçlar:**
  - `GET /api/marl/forward/view?window=&keys=` ve `GET /api/marl/forward/product?sid=&capital=&window=` (CurrentUser; geçersiz girdi 422).
  - Veri katmanı `marl/forward_views.py`.
- **Ölçüm (2026 penceresi; betimsel, yeni iddia değil):**
  - 1/N fazla getiri Sharpe'ı defterden −1,045; `m8_reading.json` baselines.2026 ile aynı.
  - Tam lot 1e9 sermayede kesirliye yakınsar (< 1e-4, test 5).
- **Test:**
  - `test_marl_forward_api` 5: 401/200/422, sayfa ve iki callback 200, grafik 5 iz.
  - `test_marl_sim_ledger` 5: tam sayı modu.
  - `test_dash_props` 4, `test_theme_contrast` 92, `test_topbar` 43, `test_account_profile` 84.

### 3.6 S6 durumu (2026-11-22; ek `phase-11-m8-forward-annex.md` v0.3)

- **Ek v0.3.** M8 §6.7'nin uygulama ayrıntıları:
  - pencere tam ilk 126 seans (2026-11-23'ten),
  - m8-v1 yerine nedensel kapanış denetimi (sapma beyanı),
  - etiketin yazımı: tohum GA'sı + gün bloklu GA + NK yan yana; `yol_farkli`'da tez cümlesi kurulmaz,
  - tek katlı tohum kümelemesi, tanımsız ve eksik hücre kuralları (12 hücrede ≥ 2).
- **Hakemler.** hakem-finans + hakem-rl ilk okuma Küçük rev. (2 + 3 BÜYÜK) → işlendi → ikinci okuma ikisi de **Kabul**. Maliyet kaydı `review-panel.md`.
- **Okuma betiği** `scripts/marl/analyze_forward_m8.py`, ilk ileri günden önce commit'lendi (ek §8). Öz-test 9/9:
  - vektör SR = `excess_sharpe`,
  - pencere tam 126 seans,
  - karar kuralları,
  - şema,
  - gün bloklu GA ortak indis,
  - tanımsız / EKSİK,
  - çıkarım yolu M8 kaydıyla bit-eş,
  - tek okuma.

  Gerçek okuma 126 seanstan önce durur; çıktı dosyası varsa durur.
- **Dondurma:** kullanıcı onayıyla etiket `m8-forward-annex-v1`, ilk ileri gün defterlenmeden önce (2026-11-23 19:00 TRT sonrası iş). Okuma denetimi (§7) depo kuruluşunun etiketten sonra olduğunu denetler.

### 3.5 S4 sonucu (2026-11-22; `tests/test_marl_forward_job.py`, `tests/test_marl_forward_api.py`)

- **Kabul kapısı geçti** (test 2–3; gerçek kaynak + 6 sentetik ileri gün, 2 politika + 2 kural):
  - aynı gün ikinci çalışma dosyalara dokunmuyor (SHA aynı),
  - 4 gün boşluk sonrası tek çalışma = her gün çalışma = tek seferde = `sim_ledger` doğrudan koşusu (satırlar ve durum bit-eş).
- **İş (`marl/forward_job.py`, ana venv).** `run_daily` sırası:
  1. arşiv doğrulama (72 SHA),
  2. S1 panel güncellemesi (ağ hatası, panelde zaten olan günleri bekletmez),
  3. marj kapsamı,
  4. defterlerin kaldığı yerden ilerlemesi (72 politika + ew, bh, minvar, hrp).
  - Kilit dosyası iki çalışmanın üst üste binmesini engeller.
  - Her çalışma `data/marl/forward/m8/log.jsonl`'a `kind: ledger` kaydı yazar: kapsam, seans sayısı, durum, hata.
- **Defter deposu** `data/marl/forward/m8/ledger/` (git dışı; ortak araştırma alanı, K-F11-3):
  - `state.json`: `LedgerState`, atomik yazım,
  - `rows.csv`: yalnız eklenir; `%.17g` yazılır, `round_trip` okunur, kayıpsız.
  - Çökme kurtarma: durumdan fazla satır atılır; eksik satır bozulma sayılır, hata verir.
  - Depo yalnız ilk ileri gün (2026-11-23) panelde olunca kurulur (ek §3: ilk defter kaydı dondurmadan sonra). Bugünkü gerçek çalışma: `baslamadi`, depo yok.
- **Marj kapsamı (S1 açık maddesi kapandı; M2 tasarımı §2.5 v0.11).**
  - Kapsam = min(panel son günü, son tarihli kaynak denetimi + 31 gün).
  - Kaynak denetimi `marl/margin_source_checks.json`: elle okunur, kayıt eklenir. İlk kayıt 2026-11-20, kapsam ufku 2026-12-21. **Ayda bir yeniden denetim gerekir;** yoksa defter durur ve sebep log'a yazılır.
  - Veri denetimi S1 gün kapısı (±%10,5).
  - `limit_down_gaps(verified_until=None)` varsayılanı değişmedi: `test_marl_core` 63, `test_marl_env` 36.
- **Tetik (K-F11-4, öneri uygulandı).** `app/services/marl_forward.py`:
  - sunucu açılışında 60 sn sonra bir kez (kaçırılan günleri doldurur),
  - sonra 30 dakikada bir: hafta içi 19:00 TRT sonrası, günde en çok 4 deneme; günün seansı deftere girince o gün biter.
  - Ayar `MARL_FORWARD_AUTORUN` varsayılan KAPALI: testler uygulamayı açar, Docker imajında `data/marl` yok. `run_server.py` açar.
  - Elle: `POST /api/marl/forward/run` (RequireAdmin + denetim kaydı) ya da `scripts/marl/forward_daily_m8.py`.
  - Okuma: `GET /api/marl/forward/status` (CurrentUser).
- **Test:**
  - `test_marl_forward_job` 4: kapsam kuralı, bit-eş doldurma, idempotent, kurtarma ve bekçiler.
  - `test_marl_forward_api` 4: ayar kapalı, 401/200, viewer 403 / admin 200 + denetim, zamanlayıcı kuralı.

### 3.4 S3 sonucu (2026-11-22; `measurements/phase11-s3/s3_parity.json`)

- **Kabul kapısı geçti.** 2026 test penceresi (207 seans, 2026-01-02 → 10-30, satır 3150 → 3357) `marl/sim_ledger.py` defterinden yeniden oynatıldı.
  - 72/72 politikanın günlük R serisi M8 test kaydıyla (`eval_fold_m8.py`, venv-marl TorchRL ortamı) **bit-eş**; fark 0. X, devir ve nakit ağırlığı da 72/72 bit-eş.
  - Karşılaştırılan M8 kayıtlarının SHA'sı arşivdeki `fold_eval.json` ile aynı (72/72): ölçülen seri M8'in okuduğu seri.
  - Kural portföyleri (ew, bh, minvar, hrp) `rule_series` ile bit-eş. Sharpe `m8_reading.json` → `baselines.2026` ile birebir aynı (4/4).
  - Politika başına ~0,07 sn (207 gün). 72 politika + 4 kural için günlük adım maliyeti ihmal edilir.
- **Yapı (`marl/sim_ledger.py`, ana venv).**
  - Karar t kapanışında, yürütme t+1 açılışında. Hedef:
    - politikada `target_weights(eylem)`,
    - kuralda `rule_baselines` ile aynı karar (minvar/hrp ay başı ve pencerenin ilk satırı).
  - Muhasebe `accounting.step_fractional`, ortamın kullandığı fonksiyonun kendisi:
    - `fill_prices`: kilit 20, dönem bağımlı marj satış engeli,
    - `project`: varsayılan kısıt,
    - A8 rf,
    - maliyet 20 bps gidiş-dönüş.
  - Politika `FrozenPolicy` (S2) ile tek gözlemle çağrılır. Gözlem `obs_spec.local_obs`, politikanın kendi eğitim `norm_stats`'ıyla.
  - TorchRL/BenchMARL yok.
  - **Durum** (`LedgerState`): başlangıç satırı, sıradaki karar satırı, kapanış ağırlığı, değer, adım sayısı.
    - JSON'a kayıpsız yazılır.
    - Kaldığı yerden devam, tam koşuyla bit-eş (test 2). S4 günlük işi bunun üstüne kurulur.
  - Değer = sanal sermaye × ∏(1 + R). Kesirli defterde sermaye R'yi değiştirmez (test 2; K-F11-6 yalnız gösterim).
- **Kararlarla ilişki.** S3 kesirli dondurulmuş defterdir (K-F11-2 önerisinin dondurulmuş kısmı; backtest ile aynı). Tam sayı lotlu ürün portföyü ve sermaye seçimi S5'te.
- **Açık kalan (S4 ön koşulu, değişmedi).** İleri panelle `Market` kurulumu, marj doğrulama kapsamı (2026-11-09) dışındaki gün için hata verir (test 4). Defter ileri günleri işlemez.
- **Test:** `tests/test_marl_sim_ledger.py` 4 kontrol, ~4 sn:
  - seri paritesi (M-none ve S-agr M8 kaydıyla, minvar ve bh `rule_series` ile),
  - JSON durumla kaldığı yerden devam bit-eş,
  - mutasyon: maliyet, durumda 1e-12 kayma, kaydırılmış başlangıç seriyi değiştirir,
  - ileri panelde marj kapsamı hatası.

### 3.3 S1 sonucu (2026-11-22; `measurements/phase11-s1/s1_forward_panel.json`)

- **Kabul kapısı geçti.**
  - Geçmiş satırlar `panel_m8.npz` ile bit-eş: 3.364 satır × 10 dizi (OHLCV, RSI, CCI, ADX, MACD, rf).
  - Gerçek çekimle 9 yeni seans eklendi (2026-11-10 → 11-20). Panel `data/marl/forward/m8/panel_m8_fwd.npz` (git dışı), 3.373 satır.
  - İş Yatırım çapraz denetiminde yama gerekmedi; en büyük fark 1e-7.
  - Aynı gün ikinci çalıştırma dosyaları değiştirmedi.
  - Yeni 270 hücre geçerli. Eğitim `norm_stats`'ı ile gözlem kırpma oranı 11 özelliğin hepsinde 0; 2026 test penceresinde en çok %0,8.
- **Yapı (`marl/forward_panel.py`, `scripts/marl/forward_panel_m8.py`):**
  - **Dondurulmuş kaynak.** `data/marl/frozen/m8-source/` altında `panel_m8`'in kaynak CSV'si ve m8-v1 İş Yatırım dosyaları; SHA'lar panel meta'sıyla aynı.
    - Uygulamanın veri sayfası `data/bist/stock_data_with_indicators.csv`'yi artımlı güncelleyebilir; ileri hat o dosyayı okumaz.
  - **Panel her güncellemede baştan kurulur.** M8 hattının aynı fonksiyonlarıyla: `apply_m8_v1` → ileri satırlar → `recompute_indicators` → `with_adx('wilder')` → `prepare_rl_panel` → `build_arrays` → rf.
    - Önceki panelin tüm satırları bit-eş denetlenir; uyuşmazlıkta yazılmaz.
    - İleri satırlar yalnız eklenir (`rows.csv`, append-only).
  - **Zincir.** Yeni gün = son yazılı kapanış × taze yfinance (auto_adjust) oranı.
    - Temettü ve bölünme o günün getirisine yansır; geçmiş yeniden düzeltilmez.
    - Hacim aynı ölçekle bölünür (ciro korunur).
  - **Çapraz kaynak (m8-v1'in nedensel karşılığı).** Yeni günün getirisi İş Yatırım düzeltilmiş kapanış getirisinden %1'den (m8-v1 `CLOSE_TOL`) fazla saparsa:
    - kapanış İş Yatırım getirisiyle zincirlenir,
    - yüksek/düşük, İş Yatırım gün içi uçlarıyla genişletilir.
  - **Neden m8-v1'in kendisi değil.** m8-v1'in 11 seanslık merkezi medyanı 5 seans ileriye bakar. Geriye dönük bir medyan da kaynaklar arası kalıcı seviye adımında yanlış yama yapardı.
    - Ölçüm (2021+, panel ve İş Yatırım önbelleği): iki kaynağın günlük düzeltilmiş getiri farkı %1'i 11 hücrede aştı (geçmiş kesik çubuklar).
    - Son günde iki kaynağın ham kapanışı birebir aynı.
  - **Gün kapısı.** İlk engelde durur; sonraki günler yazılmaz, sebep `log.jsonl`'a yazılır. Sebepler:
    - seans değil (yfinance satırı sembollerin yarısından az: atlanır),
    - kesim saati öncesi (aynı gün 19:00 TRT'den önce),
    - İş Yatırım bekleniyor,
    - rf bayat (d−1'de bilinen fonlama gözlemi 5 günden eski),
    - marj bandı dışı (kapanış ya da açılış ±%10,5 dışında: veri hatası ya da marj değişikliği adayı, elle inceleme),
    - yfinance çapası yok.
  - **Risksiz faiz.** EVDS'ten çekilir, ileri deponun kendi kopyasına yazılır; ortak `data/macro/risk_free.csv` değişmez. Taze seri geçmiş rf'yi değiştirmedi (bit-eş).
  - Ham çekimler `fetch/<zaman>/` altında SHA'larıyla saklanır.
- **Açık kalan (S4 ön koşulu).** `PRICE_MARGIN_VERIFIED_UNTIL` hâlâ 2026-11-09; kapsam dışındaki ileri gün `limit_down_gaps`'te hata verir (test 4). Ölçüt bu; sessiz geçmez.
  - S4 defteri ileri günleri işlemeden önce kapsam uzatılmalı. Bunun için bir kayıt düzeneği gerekir:
    - kaynak denetimi tarihi (son: 2026-11-20, M2 tasarımı §2.5),
    - ve gün kapısının bant denetimi; 9 gün için 0 ihlal.
  - Sabiti her gün elle değiştirmek sürdürülemez; düzenek S4'te kurulur.
- **Test:** `tests/test_marl_forward_panel.py` 4 kontrol:
  - sıfır satırla bit-eş,
  - zincir kuralı (temettü, kesik çubuk, yama eşiği),
  - gün kapısının her sebebi,
  - gerçek kaynak + 3 sentetik gün: önceki satırlar bit-eş, sonraki gün öncekileri değiştirmez, mutasyonlar yakalanır, marj kapsamı hatası.

### 3.2 S2 sonucu (2026-11-22; `measurements/phase11-s2/s2_parity.json`)

- **Kabul kapısı geçti: 72/72.** Logit farkı ölçüt ≤ 1e-6. Ölçülen en büyük fark **0**: eylem 72/72 politikada bit-eş, loc (tanh öncesi) de 0.
  - Karşılaştırma iki girdi kümesinde yapıldı:
    - 2026 test penceresi: 207 gün × 30 ajan.
    - Rastgele gözlem: 256 adet, N(0, 3²), tohum 20261122. ±5 kırpma sınırının dışına taşar, tanh doygunluğunu da yoklar.
  - Mimari dağılımı: `mlp` 24 (M-none, null), `comm` 24 (M-gat, M-self), `rowshared` 12 (S-esm), `sarl_agr` 12 (S-agr).
- **Referansın kendisi denetlendi.** Referans `scripts/marl/s2_parity_ref.py` ile üretildi: venv-marl, BenchMARL `load_policy`, deterministik keşif. Referans penceresinin günlük R serisi, M8 test kaydıyla (`results/marl_m8/2026/.../test_argmax.npz`) 72/72 birebir aynı. Yani karşılaştırılan çıktı, M8'in okuduğu davranışın ta kendisi.
- **Kod:** `marl/frozen_policy.py` → `FrozenPolicy(dizin).act(gözlem[N, d])`.
  - Ana venv'de çalışır, düz torch kullanır; BenchMARL / TorchRL import etmez.
  - Deterministik eylem: `B · clamp(tanh(loc), ±(1 − ε))`. Bu, TorchRL 0.7 `TanhNormal.deterministic_sample` ile aynı işlem sırasıdır; ε float32 çözünürlüğü, B = `logit_bound`.
  - M5 güven terimli iletişim kolları aktarılmadı; M8'de yoklar, gelirlerse açık hata verir.
- **İki bulgu (kurala bağlandı):**
  1. **Çağrı biçimi.** Aynı gözlem kümesi toplu (`[T, N, d]`) ve tek tek verildiğinde eylem en çok 5,0e-6 farklı çıkıyor. En büyüğü S-agr'de: 1.320 girdili matris çarpımında float32 toplama sırası değişiyor.
     - Parite, referansla aynı çağrı biçiminde ölçüldü: pencere gün gün, rastgele küme toplu.
     - **Kural:** canlı ve simülasyon yolu politikayı TEK gözlemle çağırır; M8 değerlendirmesi de böyle çağırıyordu. Toplu yol yalnız çözümleme içindir (`frozen_policy` belgesi).
     - S3 seri paritesi bu kuralla ölçülecek.
  2. **Süreç içi durum.** BenchMARL görev yapılandırması sınıf düzeyinde tutuluyor (`MarlTask.PORTFOLIO.update_config` birleştirir). Bir SARL kolunun `single_agent` anahtarı, aynı süreçte yüklenen sonraki MARL koluna taşıyor ve yükleme hata veriyor.
     - M8 her politikayı ayrı süreçte yüklediği için M8 sonuçlarını etkilemez.
     - Referans betiği artık kol başına ayrı süreç açar.
     - Ana venv yolu (`frozen_policy`) bu durumu taşımaz.
- **Test:** `tests/test_marl_frozen.py` 4. kontrol. Her mimariden bir politika (6) referansla bit-eş; gat maskesi bozulunca çıktı ayrışır (mutasyon). Referans dosyaları yoksa kontrol atlanır.

**Kaba efor** (tek geliştirici, gün; ölçüm değil): S0 0,5 · S1 2–3 · S2 2–3 · S3 3–4 · S4 1–2 · S5 4–6 · S6 0,5 · S7 1–2 → ≈ 14–21.

## 4. Kapsam dışı

- Gerçek emir iletimi, aracı kurum bağlantısı, gerçek para. Yalnız sanal para.
- Yeni araştırma hipotezi. Ürün portföylerinde yeniden eğitim yapılabilir, ama sonuçları tez sonucu değildir ve deneme sayısına (DSR N) girmez. Yalnız ileri veri eki tez sonucudur.
- Makale 2 (B5 + K0–K4): ayrı iş (Faz 10 plan §4).

## 5. Riskler

| Risk | Etki | Azaltma |
|---|---|---|
| Politika aktarımında sessiz fark (S2) | İleri test başka bir politikayı ölçer | Logit paritesi + S3 seri paritesi; SHA günlük denetim |
| yfinance veri sorunları (kesik gün, yanlış düzeltme; M2 §6.1'de ölçüldü) | İleri panelde sahte getiri | Veri kapısı + `m8-v1` kuralları; ikinci kaynak (İş Yatırım) denetimi ileri pencere için de koşulur (`cross_source_audit_m8.py`) |
| Fiyat marjı kuralı değişir | Satış engeli yanlış | `PRICE_MARGIN_VERIFIED_UNTIL` sonrası hata verir; kaynakla yeniden denetim (S1) |
| Evren değişimi (BIST-30 üyeliği) | Panel sembolleri sabit, ürün beklentisi farklı | Dondurulmuş defterler M8 evreniyle kalır (beyan). Ürün portföyü evren kararı ayrı (§6) |
| Arşivin kaybı (`results/` git dışı) | İleri veri okunamaz | S0 yedeği ve SHA manifesti |

## 6. Karar bekleyen konular (kullanıcı)

| # | Konu | Öneri |
|---|---|---|
**Karar (2026-11-22, kullanıcı): K-F11-1..6'da öneriler uygulanır.** S4 (K-F11-3, -4), S5 (K-F11-1, -2, -6) ve S7 (K-F11-5) buna göre kuruldu.

| K-F11-1 | Panoda varsayılan görünüm | M-none (12 tohumun ortalaması ± std) + 1/N + min-varyans yan yana. Diğer kollar seçilebilir |
| K-F11-2 | Tam sayı lot mu, kesirli mi | Dondurulmuş defterler **kesirli** (backtest ile aynı; ileri test için şart). Ürün portföyü isteğe bağlı tam sayı lot (K5 `rebalance_orders`); farkı panoda gösterilir |
| K-F11-3 | Dondurulmuş defterlerin sahibi | Ortak, salt okunur araştırma alanı (yönetici yazar, herkes okur). Kullanıcı çalışma alanlarına kopyalanmaz |
| K-F11-4 | Günlük tetik | Uygulama içi arka plan işi (sunucu açıkken) + açılışta geriye dönük doldurma. Sunucu kapalı günler S4 doldurmasıyla kapanır |
| K-F11-5 | Elle "Uygula" akışında yürütme fiyatı | Ertesi açılışta yeniden fiyatlama (K14 ile birlikte). Daha basit seçenek: fark sınırlılık olarak gösterilir |
| K-F11-6 | Başlangıç sermayesi | 100.000 TL sanal (bugünkü varsayılan). Kesirli defterlerde ölçek sonucu değiştirmez |

## Versiyon geçmişi

- **v0.17 (2026-11-26):** Açılır listeler ve takvim koyu temada (§3.16; kullanıcı bildirimi): beyaz arama satırı, metin arkası kutu, seçili/hover ayrımı, takvim morları; Dropdown ve takvim Türkçe. Test 97.
- **v0.16 (2026-11-25):** Arayüz metinleri ve kullanılabilirlik (§3.15; kullanıcı isteği): 9 sayfa Türkçe karakter (AST ile görünen dizeler), `fmt.py` Türkçe para/saat/durum, veri dosya listesi sayfalı tablo, boş durumlar, tek ajanlı sayfalara kapsam notu, hiperparametre durum rozeti hatası.
- **v0.15 (2026-11-25):** Ana sayfa ve arayüz güncel duruma (§3.14; kullanıcı isteği): M8 özeti (sıralama değil; kural portföyleri yanında) + simülasyon kutusu + durum şeridi; rehberde M8 kartları (`kademe: nihai`), bayat metinler; son çalışma saati TRT; menü Türkçe. Test 6 + 9.
- **v0.14 (2026-11-25):** Simülasyon sayfası yeniden tasarımı (§3.13; kullanıcı isteği): eldeki hisseler, gün özeti, gün seçimi; `forward_views.daily_book`, tam sayı satırında `fills`/`p_close`. Test 5.
- **v0.13 (2026-11-25):** İleri veri deposunun VDS'ye devri (§3.12; kullanıcı kararı): tek yazar `WRITER`, devir paritesi betiği, imajda günlük iş dosyaları ve `GIT_COMMIT`.
- **v0.12 (2026-11-24):** İş Yatırım eski yanıt / zaman aşımı (§3.11): `fetch_isy` beklenen günü yeniden ister, istek başına toplam süre sınırı; veri kuralı aynı. Test 5.
- **v0.11 (2026-11-23):** Ürün portföyü lot tablosu (§3.10; kullanıcı isteği).
- **v0.10 (2026-11-23):** İlk gerçek günlük çalışmada `rows.csv` geri okuma 1 ULP kaybı prefix bekçisinde yakalandı (§3.9); `read_rows` round_trip, test 4a. Davranış değişmedi.
- **v0.9 (2026-11-22):** S7 bitti (§3.8): "Uygula" ertesi açılışta (K-F11-5), canlı gözlem uyarıları (K13; canlı karar price_stats'ı kullanmıyor bulgusu, açık ürün işi). K-F11-1..6 kullanıcı kararı yazıldı.
- **v0.8 (2026-11-22):** S5 bitti (§3.7): simülasyon panosu, aynı pencere karşılaştırması, kol ort. ± std, ürün portföyü tam lot (K-F11-1, -2, -6 önerileri); `orders.step_integer` (ortam ve defter aynı fonksiyon).
- **v0.7 (2026-11-22):** S6 eki v0.3 (hakem Kabul) + okuma betiği (öz-test 9/9); dondurma onayı bekliyor.
- **v0.6 (2026-11-22):** S4 bitti (§3.5): günlük iş idempotent, boşluk doldurma bit-eş; marj kapsamı düzeneği (tarihli kaynak denetimi + 31 gün; S1 açık maddesi kapandı); uygulama içi tetik (K-F11-4 öneri). K-F11-1..6 kullanıcı: öneriler uygulanır. S6 eki v0.3 taslak, hakem Kabul, dondurma onayı bekliyor.
- **v0.5 (2026-11-22):** S3 bitti (§3.4): `marl/sim_ledger.py` kesirli defter; 2026 test penceresi 72/72 politika M8 test serisiyle bit-eş, 4/4 kural portföyü `rule_series` ile bit-eş ve M8 okumasındaki Sharpe ile aynı. Durum JSON'la kaldığı yerden devam bit-eş (S4 temeli). Marj kapsamı açık maddesi sürüyor.
- **v0.4 (2026-11-22):** S1 bitti (§3.3): ileri panel append-only, dondurulmuş kaynaktan baştan kurulum, önceki satırlar bit-eş; 9 gerçek seans eklendi (yama 0). Kapanış çapraz denetimi nedensel getiri kuralıyla (m8-v1 merkezi medyanı ileriye bakıyordu). Açık: marj doğrulama kapsamının uzatılması S4 ön koşulu.
- **v0.3 (2026-11-22):** S2 bitti (§3.2): sade ileri geçiş 72/72 bit-eş (logit farkı 0; ölçüt ≤ 1e-6); referans R serisi M8 test kaydıyla 72/72 aynı. Kurallar: tek gözlemle çağrı (toplu çağrı ≤ 5e-6 farklı), referans kol başına ayrı süreç.
- **v0.2 (2026-11-22):** S0 bitti (§3.1): arşiv 72/72 doğrulandı, harici disk yedeği ve geri yükleme denemesi geçti. Politika sayısı 60 → 72 düzeltildi (boş kol ön kayıt listesinde).
- **v0.1 (2026-11-22):** İlk taslak. Faz 10 araştırma kapanışından sonra kullanıcı niyetiyle yazıldı. Kod yok.
