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
| `hakem-2` | opus | Son kapı. Uzman bulgularını doğrular; iddia–kanıt, iç tutarlılık, özgünlük, rakam izi ve tekrar üretimi denetler | Her paket |

**Plan paketlerine önerilen eşleme:**

| Paket | Panel |
|---|---|
| B7 | hakem-2 |
| B8, B10 | hakem-finans + hakem-2 |
| K18 (+ K25) | hakem-rl + hakem-2 |
| R1 | hakem-rl + hakem-finans + hakem-2 |
| D, E, F | hakem-finans + hakem-2 |
| G | hakem-rl + hakem-finans + hakem-2 |
| P | hakem-finans + hakem-rl + hakem-2 |
| M (MARL) | hakem-rl + hakem-2 |

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
