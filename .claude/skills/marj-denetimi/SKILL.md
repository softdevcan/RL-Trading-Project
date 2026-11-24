---
name: marj-denetimi
description: Faz 11 ileri veri simulasyonunun aylik fiyat marji kaynak denetimi — Borsa Istanbul pay fiyat marjinin (2020/20 duyurusu, %10) hala yururlukte oldugunu birincil kaynaktan okur ve marl/margin_source_checks.json'a tarihli kayit ekler. Kullanici "marj denetimi", "fiyat marji kontrolu", "marj kaydi" dediginde ya da son denetimin 31 gunluk ufku (horizon_end) yaklastiginda/gectiginde kullan. Kayit yoksa gunluk defter ufuk sonunda durur.
---

# Marj denetimi (ayda bir)

Simulasyon defteri taban acilistaki satis engelini Borsa Istanbul fiyat marjiyla (`marl/accounting.py`
`PRICE_MARGIN_SCHEDULE`, bugun %10) uygular. Kod kuralin degismedigini bilemez; bu yuzden
`marl/margin_coverage.py` son kaynak denetimine 31 gun guvenir (`CHECK_HORIZON_DAYS`). Yeni kayit yoksa
ufuk sonunda defter durur ve log'a sebep yazar (veri kaybi yok; kayit eklenince bosluk doldurulur).

## 1. Durumu oku
- `marl/margin_source_checks.json` son kaydi: `checked`, kaynaklar. Ufuk = `checked` + 31 gun.
- `data/marl/forward/m8/log.jsonl` son kayitlari: `margin_coverage.horizon_end`, `limited_by: check` var mi
  (defter durmus mu), gun kapisinda `marj bandi disi` sebebiyle durmus gun var mi (veri hatasi ya da marj
  degisikligi adayi — once onu incele).

## 2. Birincil kaynagi OKU (bellekten yazma)
Her sayfayi gercekten ac (WebFetch ya da web-fetch ajani); okumadigin sayfayi kayda yazma.
- https://borsaistanbul.com/piyasalar/pay-piyasasi/piyasa-isleyisi — fiyat marji/fiyat limitleri bolumu.
- https://borsaistanbul.com/piyasalar/pay-piyasasi/pazarlar — pazar bazinda marj.
- Borsa Istanbul duyurulari, son denetimden bu yana: "fiyat marji", "fiyat limiti", "2020/20" gecen duyuru.
- Gerekirse Pay Piyasasi Proseduru'nun son revizyonu (Tablo-4 dipnotu; revizyon tablosu).
Aranan tek soru: 13/03/2020'den beri gecerli %10 pay fiyat marji degisti mi, yeni bir duyuru var mi?

## 3a. Degismediyse: kayit ekle
`marl/margin_source_checks.json` listesine EN SONA ekle (eski kayitlara dokunma):
```json
{"checked": "<BUGUN YYYY-MM-DD>", "schedule_unchanged": true,
 "result": "<ne okundu, tek cumle; ornek: 2020/20 (%10) yururlukte, yeni duyuru yok>",
 "sources": ["<gercekten okunan URL'ler>"],
 "ref": "docs/development/phase-10-m2-design.md §2.5 (2020/20 dayanak)"}
```
- `python tests/test_marl_forward_job.py` (4) kos; kapsam kurali yeni kayitla gecmeli.
- Dal ac (main korumali), commit + push + PR. Birlestirme kullanicinin acik sozuyle.
- Sunucu yeniden baslatilmaz: kapsam her calismada dosyadan okunur; durmus gunler sonraki tetikte dolar.

## 3b. Degistiyse ya da emin degilsen: DUR
- Kaydi `schedule_unchanged: false` ile yazma ve takvimi kendin degistirme. Kullaniciya kaynakla birlikte bildir.
- Degisiklik varsa is: `PRICE_MARGIN_SCHEDULE` + M2 tasarimi §2.5 yeni tarihli alt madde + `test_marl_core`
  (plan disi duzeltme -> once plana ekle; kok CLAUDE.md). Bu ileri test icin bir sapma beyanidir (S6 eki).

## Sonra
- Kullaniciya yeni ufku soyle (`checked` + 31 gun) ve bellek notundaki hatirlatma tarihini bir ay ileri al.
