# Hakem paneli

**Durum:** 2026-11-06'da kuruldu. İlk kullanım K18 (2026-11-09).

**Neden:** Faz 9 B9 incelemesinde tek bir hakem-2 koşumu **~301 bin token, 66 araç çağrısı ve ~23 dakika** harcadı (görev bildirimindeki `subagent_tokens`, `tool_uses`, `duration_ms`). Maliyetin iki ana kaynağı var:

- **Bağlam:** ajan sıfırdan başlıyor ve büyük belgeleri baştan sona okuyor. Faz 9/10 planları, vision-and-scope ve literatür tablosu kabaca 300 KB'ı aşıyor. Okunan bağlam sonraki her araç turunda yeniden gönderiliyor.
- **Tur sayısı ve kendi ölçümleri:** ajan kendi ölçüm betiklerini yazıp koşturuyor; her tur maliyeti büyütüyor.

Uzmanlaşmış ve daha ucuz hakemler, ortak bir inceleme paketi ve son kapıda tek bir hakem-2 ile aynı derinliği daha az token'la almak hedefleniyor. Gerçekleşen maliyet her incelemeden sonra aşağıdaki kayda yazılır.

## Kim

| Hakem | Model | Kapsam | Ne zaman |
|---|---|---|---|
| `hakem-finans` | sonnet | Nokta-zamanlılık, BIST gerçekliği, maliyet ve yürütme, risksiz faiz ve fazla getiri, Sharpe çıkarımı, baseline | Finansal iddia, metrik ya da veri hattı değişikliği |
| `hakem-rl` | sonnet | MDP ve ödül, eğitim bütçesi, HPO ↔ nihai eğitim, değerlendirme protokolü, tohum, RL'ye özgü sızıntı, MARL | Env, ödül, eğitim ya da değerlendirme protokolü değişikliği |
| `hakem-2` | opus | Son kapı. Uzman bulgularını doğrular; iddia–kanıt, iç tutarlılık, özgünlük, rakam izi ve tekrar üretimi denetler | Faz kapanış kapısı ve teze/makaleye girecek sonuç (2026-11-15'ten beri; önceden her paket) |

**Kademeli düzen (kullanıcı kararı, 2026-11-15; Faz 9 §9.22).** Hakem-2 her pakette çalışmaz. Kurallar:
- Uzmanlar (sonnet) yalnız **pahalı bir işin girdisini üreten** pakette çağrılır: veri hattı, ödül, kalibrasyon, tahmin tablosu, baseline. Pahalı iş P hesaplaması ya da Milestone 0' koşumlarıdır. Hata o iş başlamadan bulunmalı.
- **Hakem-2 (opus) fazın kapanışında bir kez** çalışır. Kapanış paketi uzman tablolarını, beyan listesini ve protokolü birlikte verir. Paketler arası tutarlılık ve iddia–kanıt denetimi burada yapılır.
- **İkinci okuma yalnız BÜYÜK bulguda** yapılır. KÜÇÜK bulgular düzeltilip kapanış paketinde listelenir.
- **Hakemsiz paket:** testin bozulan üretim koduna karşı kırıldığı gösterilir (mutasyon) ve öz-kontrol yapılır. Kapanışta hakem-2 bunları örnekleyerek okur.
- **Gerekçe** (aşağıdaki kayıt): sonnet uzmanlar dar kapsamda BÜYÜK bulgular buldu (K18'de 3, R1'de 2). Hakem-2 uzmanların kaçırdığını buldu (K18'de 4 yeni bulgu, E'de bir kuralın öncülü). Opus tek seferlik kapıda tutulur.
- Faz 9'un kalan noktaları (İ1–İ3 + kapanış) §9.22'de.

**Plan paketlerine eşleme** (2026-11-15'ten önceki düzen; B7–E için tarihsel kayıt):

| Paket | Panel |
|---|---|
| B7 | hakem-2 |
| B8, B10 | hakem-finans + hakem-2 |
| K18 (+ K25) | hakem-rl + hakem-2 |
| R1 | hakem-rl + hakem-finans + hakem-2 |
| E | hakem-2 |
| G (İ1) | hakem-rl → kapanışta hakem-2 |
| F + B6 (İ2) | hakem-finans → kapanışta hakem-2 |
| P + D (İ3) | hakem-finans + hakem-rl → kapanışta hakem-2 |
| M (MARL) | hakem-rl paket başına; hakem-2 Milestone kapılarında |

## Akış

1. **İnceleme paketi.** Ana oturum, aşağıdaki şablonla en fazla ~5 bin token'lık bir paket yazar. Paket oturumun geçici dizininde durur; hakemlere yolu verilir. Uzun ölçümleri ana oturum yapar ve JSON'ları depoda arşivler (örnek: `docs/development/measurements/phase9-b9/`). Hakemler bu JSON'ları okur, ölçümü kendileri tekrarlamaz.
2. **Uzmanlar.** Paralel ve arka planda çalışırlar; yalnızca alanına giren pakette çağrılırlar.
3. **hakem-2.** Paketi ve uzman tablolarını okur, son kararı verir. Bedeli biraz bağımsızlık kaybı: hakem-2 uzmanların bulduğunu yeniden keşfetmez, doğrular.
4. **Birleştirme.** Bulgular plan §9'daki tek tabloda birleşir; "Kaynak" sütunu bulgunun hangi hakemden geldiğini gösterir.
5. **Düzeltme sonrası ikinci okuma.** Aynı hakem `SendMessage` ile sürdürülür; bağlamı korunur ve yalnızca değişen satırları okur. Yeni ajan başlatılmaz.
6. **Maliyet kaydı.** Bildirimdeki token, araç sayısı ve süre aşağıdaki tabloya ve ilgili §9 inceleme başlığına yazılır.

**Ajan tanımları da her çağrıda bağlama girer.** `.claude/agents/*.md` dosyalarını kısa tut.

## İnceleme paketi şablonu

```markdown
# İnceleme paketi — <paket> (<tarih>, dal <dal>, taban <commit>)
## Kapsam ve kabul ölçütü
<plan maddesi ve satırı; kabul ölçütü tek cümle>
## Değişen dosyalar
- <dosya:satır aralığı> — <tek satır amaç>
## İddialar (her biri kaynağıyla)
- <iddia> — kaynak: <test adı | JSON alanı | komut>
## Komutlar ve son çıktıları
- `<komut>` → <sonuç satırı>
## Ölçüm JSON'ları
- <yol> (commit, panel SHA-256)
## Bilinen açık işler (yeniden itiraz edilmesin)
- <madde> → <plan yeri>
## Bu hakeme özel sorular (en fazla 5)
1. ...
```

## Maliyet kaydı

| Tarih | Paket | Hakem (model) | Token | Araç çağrısı | Süre | Not |
|---|---|---|---|---|---|---|
| 2026-11-06 | B9 | hakem-2 (opus) | ~301 bin | 66 | ~23 dk | Panel öncesi, paket yok, planlar bütün okundu |
| 2026-11-06 | B7 | hakem-2 (opus) | ~144 bin | 25 | ~12 dk | İnceleme paketiyle; küçük paket, B9 ile kıyas tam değil. 1 BÜYÜK bulgu (türbülans 0 başı) |
| 2026-11-09 | K18 + K25 | hakem-rl (sonnet), ilk okuma | ~50 bin | 7 | ~1,5 dk | Panelin ilk kullanımı. Python koşmadı, JSON + aralıklı kod. 3 BÜYÜK bulgu (doygunluk kapı dışı, eğitim bağlamı, B7 düzeltmesinin yanlış negatifi) |
| 2026-11-09 | K18 + K25 | hakem-rl (sonnet), ikinci okuma (SendMessage) | ~57 bin | 3 | ~0,5 dk | Yalnız değişen satırlar; Küçük revizyon, 1 yeni KÜÇÜK bulgu (bağlamda bayat hücre) |
| 2026-11-09 | K18 + K25 | hakem-2 (opus), son kapı | ~143 bin | 39 | ~8,6 dk | Paket + uzman tablosu. Büyük revizyon (RED yok): 2 BÜYÜK (sembol havuzlama, MACD varsayılanı karar dışı), 6 KÜÇÜK. Kendi ölçümü: sembol başına destek (persym.py), VAKBN yeniden üretimi. Uzman bulgularını doğruladı, 4 yenisini buldu |
| 2026-11-09 | K18 + K25 | hakem-2 (opus), ikinci okuma (SendMessage) | ~165 bin (önceki bağlamla) | 8 | ~1,7 dk | Kabul; 2 KÜÇÜK not (seri başı kenar durumu, ifade) |
| 2026-11-13 | R1 | hakem-rl (sonnet), ilk okuma | ~57 bin | 9 | ~1 dk | Paketle; Python koşmadı. Küçük revizyon: 3 BÜYÜK (seçimin değeri, pasif referans, `stats` kolunda destek dışı seçim), 5 KÜÇÜK |
| 2026-11-13 | R1 | hakem-finans (sonnet), ilk okuma | ~66 bin | 9 | ~2 dk | Küçük revizyon: 2 BÜYÜK (negatif fazla getiride Sharpe oynaklığı ödüllendirir; nakit 0 ↔ rf düşülüyor), 5 KÜÇÜK. Ana oturum bunu gerçek panelde ölçtü (selection.json) |
| 2026-11-14 | R1 | hakem-2 (opus), son kapı | ~123 bin | 33 | ~7,9 dk | Paket + §9.11–§9.12. Büyük revizyon (RED yok): 2 BÜYÜK (2024–2026 testi R1 öncesi okunmuş; F1 yanıtı kanıttan fazlası), 7 KÜÇÜK. Testi kendi koştu (82), rakam izini örnekleyerek denetledi |
| 2026-11-14 | R1 | hakem-2 (opus), ikinci okuma (SendMessage) | ~153 bin (önceki bağlamla) | 18 | ~2,8 dk | H2–H9 kapandı; H1'in olgusu yanlış bulundu (R1 öncesi 23 dosya işlem tarihi taşıyor). Düzeltme yeniden okutulmadı |
| 2026-11-14 | Y28 | hakem-2 (opus), son kapı | ~70 bin | 25 | ~2,9 dk | Paketle (~450 kelime), uzman yok. Küçük revizyon: 5 KÜÇÜK (son satır değişmezi kodda zorlanmıyor, SHAP ucu farklı matris → Y30, gerekçe cümlesi, ICEEMDAN uç etkisi, DL yolu dolaylı). Testi kendi koştu (41) |
| 2026-11-14 | Y29 | hakem-2 (opus), son kapı (Y28 bağlamıyla SendMessage) | ~40 bin ek (toplam ~110 bin) | 20 | ~3,1 dk | Paketle. Küçük revizyon: 6 KÜÇÜK (saate bağlı kontrol, v1.10.0 değerlendirmesi, üzerine yazma, düz yfinance satırı, pano, ex-ante). Testi kendi koştu |
| 2026-11-15 | E (E1 geçici) | hakem-2 (opus), son kapı | ~85 bin | 18 | ~5,7 dk | Paketle, uzman yok. Büyük revizyon (RED yok): 2 BÜYÜK (tutarlılık kuralının öncülü yanlış; #11503 dipnotu ayrıştırıcıda kayboluyordu, "sitede kanıt yok" iddiası yanlış), 5 KÜÇÜK. 14 çeyreği ham metinle elle karşılaştırdı; test koşmadı |
| 2026-11-15 | E (E1 geçici) | hakem-2 (opus), ikinci okuma (SendMessage) | ~118 bin (önceki bağlamla) | 5 | ~1,1 dk | Küçük revizyon, PR'ı engellemez: 3 KÜÇÜK (çapa günü katı modda None, "üst sınır" koşullu, strict ≠ doğruluk). Testi kendi koştu (32/0) |
| 2026-11-15 | İ1 (G1 + G3) | hakem-rl (sonnet), ilk okuma | ~37 bin | 6 | ~0,7 dk | Kademeli düzenin ilk uzman noktası; paket ~1,8 bin kelime + ölçüm JSON'u. Küçük revizyon: 2 BÜYÜK (K4 w₂ ölçeği; nakitte Dₜ — ana oturum ölçtü: max\|D\| 293), 4 KÜÇÜK. Test koşmadı |
| 2026-11-15 | İ1 (G1 + G3) | hakem-rl (sonnet), ikinci okuma (SendMessage) | ~41 bin (önceki bağlamla) | 3 | ~0,2 dk | Kabul; yalnız git diff + grep. BÜYÜK bulgu olduğu için yapıldı (kademeli kural) |
| 2026-11-15 | İ2 (F + B6) | hakem-finans (sonnet), ilk okuma | ~37 bin | 7 | ~1,1 dk | Paket ~1,9 bin kelime + 2 JSON. Küçük revizyon: 2 BÜYÜK (ECE tek başına kalibrasyon kanıtı değil; GA'sız karşılaştırma), 3 KÜÇÜK; rejim için USD bazlı tanım önerisi (kullanıcı kabul etti). Test koşmadı |
| 2026-11-15 | İ2 (F + B6) | hakem-finans (sonnet), ikinci okuma (SendMessage) | ~46 bin (önceki bağlamla) | 3 | ~0,4 dk | Kabul; 2 engelleyici olmayan not (kapı testte seçiliyor → P'de çoklu karşılaştırma; USD bir gün kayma) |
| 2026-11-15 | İ3 (P + D) | hakem-finans (sonnet) | ~51 bin | 10 | ~1,8 dk | Büyük revizyon (çıkarım katmanı): güç/MDE, blok içi AUC, hayatta kalan evren; 5 KÜÇÜK |
| 2026-11-15 | İ3 (P + D) | hakem-rl (sonnet) | ~41 bin | 10 | ~0,9 dk | Küçük revizyon: 2 BÜYÜK Milestone 0' tasarımına (S-msg boş sonuç beyanı, pozitif kontrol), 3 KÜÇÜK |
| 2026-11-15 | İ3 (P + D) | hakem-finans (sonnet), ikinci okuma (YENİ ajan; ilk bağlam kayboldu) | ~33 bin | 7 | ~0,9 dk | Küçük revizyon: 7 KÜÇÜK. Eşit kapanış duyarlılığı (#4) ana oturumda ölçüldü: havuzlu 0,511'in anlamlılığı eşitlere bağlı. Test koşmadı |
| 2026-11-15 | Faz 9 kapanış kapısı (§9.22–9.33 + Milestone 0' protokolü) | hakem-2 (opus) | ~132 bin | 36 | ~6 dk | Paketle. Büyük revizyon: 8 BÜYÜK (2021 duman okuması, ön elemede 2021 doğrulaması + argmax yanlılığı, "SB3 varsayılanları" yanlış, MDE yok, doğrulayıcı kıyas yok, K4↔M8, pozitif kontrol/hakem-rl, veri dondurma), 7 KÜÇÜK. 6/8 iddia örneklemesi tuttu. 1–7. maddeler kapanıyor, 8 açık |
| 2026-11-15 | Faz 9 kapanış kapısı | hakem-2 (opus), ikinci okuma (SendMessage) | ~177 bin (önceki bağlamla) | 10 | ~2,2 dk | 14/15 kapandı; kalan BÜYÜK #5 metin düzeyinde (seçim kuralı, birleştirme, TOST sınırı); 3 KÜÇÜK. "Düzeltilince Kabul, yeni okuma gerekmez" |
| 2026-11-15 | Milestone 0' m0p-v2 (yalnız v1 farkları) | hakem-2 (opus), kısa okuma | ~52 bin | 13 | ~2,1 dk | Paket ~1,1 bin kelime. Küçük revizyon: 2 BÜYÜK (`--scratch` 2021'i yeniden hesapladı; Q3 okuma kuralı/penceresi yok), 5 KÜÇÜK. Ölçüm istemedi; ρ'lu MDE sonuçla |
| 2026-11-15 | Milestone 0' m0p-v2 | hakem-2 (opus), ikinci okuma (SendMessage) | ~64 bin (önceki bağlamla) | 3 | ~0,5 dk | Kabul; 7/7 kapandı. İstenen tek kanıt (scratch yalnız 2024) ana oturumda koşuldu: 4/4 `fold=2024` |
| 2026-11-16 | Milestone 0' m0p-v2 sonuç okuması | hakem-2 (opus) | ~79 bin | 19 | ~5 dk | Paketle. Büyük revizyon (yalnız okuma katmanı): 5 BÜYÜK (arşivde yasak 'eşdeğer' etiketi — sonuç sonrası eklenmişti; Q3'te pencere koruması yok; havuzlu TOST heterojenliği gizliyor; SE tohum varyansını içermiyor; PC060'ın ekonomik değeri ölçülmedi), 6 KÜÇÜK. Kendi duyarlılık ve algoritma bazında ölçümlerini yaptı; ana oturum birebir yeniden üretti |
| 2026-11-16 | Milestone 0' m0p-v2 sonuç okuması | hakem-2 (opus), ikinci okuma (SendMessage) | ~94 bin (önceki bağlamla) | 4 | ~0,8 dk | Kabul; öz sınamayı kendi koştu (10/10). 3 KÜÇÜK not (özet niteleyicisi, sabit forward_data, izinli dal sınanmıyor) |
| 2026-11-16 | Faz 10 M2 çekirdeği | hakem-rl (tanımda sonnet; oturumda opus göründü), ilk okuma | ~33 bin | 5 | ~1,4 dk | Paket istemin İÇİNDE (dosya okunmadı), yalnız satır aralıkları. Küçük revizyon: 2 BÜYÜK (`'differential'`te α anlamsız — ana oturum ölçtü: portföy terimi std'si bireysel terimin 242 katı, `'excess'`te 6,3; DSR (a, b) gözlemde yok → Markov değil), 5 KÜÇÜK. Test koşmadı |
| 2026-11-16 | Faz 10 M2 çekirdeği | hakem-finans (sonnet), ilk okuma | ~52 bin | 13 | ~2,3 dk | Paket istemin içinde. Küçük revizyon: 2 BÜYÜK (halka arz/geç listelemede ilk gün açılıştan tam dolum; evren ex-post, paketteki 'bugünkü XU030' ifadesi yanlıştı), 1 doğrulanmalı (yfinance `auto_adjust` varsayılanı True → OHLC düzeltilmiş; ana oturum doğruladı), 4 KÜÇÜK. Test koşmadı |
| 2026-11-16 | Faz 10 M2 çekirdeği | hakem-rl, ikinci okuma (SendMessage) | ~40 bin (önceki bağlamla) | 3 | ~0,4 dk | Kabul; yalnız git diff. 2 KÜÇÜK (kilit gözlemde yok → ana oturum gösterdi: kilit ⊂ feat_ok = 0, değişmez test; panel kesimine bağlılık → belge) |
| 2026-11-16 | Faz 10 M2 kapanış kapısı (gerçek panel + satış engeli + §9.1 PC060) | hakem-2 (opus) | ~104 bin | 25 | ~6,3 dk | Paket ~2,3 bin kelime, yalnız 096ae80..5d81571 farkı. Küçük revizyon: 3 BÜYÜK (fiyat marjı veri döneminde ±%20 → satış engelinin dayanağı yanlış, web kaynağıyla; 0,17 bu testin MDE'si değil; tohum yayılımı — kendi ölçümü, ana oturum birebir yeniden üretti), 7 KÜÇÜK. Testleri koştu |
| 2026-11-16 | Faz 10 M3a ön kaydı (eğitimden önce) | hakem-rl (sonnet) | ~28 bin | 6 | ~0,9 dk | Paket ~1,3 bin kelime; Python koşmadı. Küçük revizyon: 4 BÜYÜK (ortak HAC tohum varyansını katmıyor; eğitim penceresinde ezber ↔ sinyal ayrılamıyor → karıştırılmış PC kontrolü; "aktarıyor" genelleme ister; lr 5e-5 ile KALDI artefakt olabilir), 3 KÜÇÜK. Kullanıcının ~1 saat sınırıyla 25 koşu yerine 6 koşu + 3'e 3 sıralama kuralı |
| 2026-11-16 | Faz 10 M3a ön kaydı | hakem-rl, ikinci okuma (SendMessage) | ~40 bin (önceki bağlamla) | 2 | ~0,4 dk | Küçük revizyon; 7/7 karşılandı. 1 BÜYÜK (adım tabanı yok → F_min = 600 bin ön kayda), 2 KÜÇÜK (p = 0,05 eşitliği; doğrulama yalnız işaret kontrolü). Yalnız git diff |
| 2026-11-18 | Faz 10 M3b ön kaydı v1.0 (eğitimden önce) | hakem-rl (sonnet) | ~51 bin | 17 | ~1,3 dk | Küçük revizyon: 1 BÜYÜK (Aşama 2 seçimi tohum gürültüsüne karşı kalibresiz), 4 KÜÇÜK (eleştirmen kapasitesi, M4 varsayılanı, öz test, kapı denetimi) |
| 2026-11-18 | Faz 10 M3b ön kaydı v1.1 | hakem-rl, ikinci okuma (SendMessage) | ~65 bin (önceki bağlamla) | 1 | ~0,3 dk | Kabul; 5/5 karşılandı, yeni BÜYÜK yok. Yalnız git diff |
| 2026-11-19 | Faz 10 M4a ön kaydı v0.2 (eğitimden önce) | hakem-rl (sonnet) | ~39 bin | 9 | ~1,3 dk | Paket ~1,5 bin kelime; komut koşmadı. Büyük revizyon: 2 BÜYÜK (durma kuralı ayarsız tek lr'ye bağlı → Aşama 1b; δ'nın tavanı başka bir farktan türetilmiş), 3 KÜÇÜK ((iv) gürültü payı, mimari karışması, attn ↔ σ hizası) |
| 2026-11-19 | Faz 10 M4a ön kaydı v0.3 | hakem-rl, ikinci okuma (SendMessage) | bildirimde yok | — | — | Kabul; 5/5 karşılandı. 3 KÜÇÜK (1b ikinci atış bedeli, tek yönlü lr, tohum başına G) metne işlendi. Yalnız git diff |
| 2026-11-16 | Faz 10 M2 kapanış kapısı | hakem-2 (opus), ikinci okuma (SendMessage) | ~122 bin (önceki bağlamla) | 6 | ~0,8 dk | Kabul (tek satırlık koşul: duman betiği yeniden adlandırma sonrası çöküyordu — koşturup yakaladı); 1 KÜÇÜK beyan (bant dışı açılış taban sayılıyor). Duyarlılık ve hücre sayımını kendi ölçümüyle doğruladı |
| 2026-11-16 | Faz 10 M2 çekirdeği | hakem-finans, ikinci okuma (SendMessage) | ~67 bin (önceki bağlamla) | 4 | ~0,7 dk | Kabul (koşullu): fiyat limiti 'seyrek' eşiği ölçümden ÖNCE yazılmalı; 3 KÜÇÜK (eşik ön kaydı, kilit sonrası tavan serisi ölçümü, baseline aynı kilitle) |
