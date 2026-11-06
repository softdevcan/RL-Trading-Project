---
name: kaynak-ekle
description: Tezin kaynak tabanina (docs/thesis/literature-table.md + vision-and-scope.md §9) yeni kaynak ekler — kunyeyi Crossref/arXiv/OpenAlex ile dogrular, seed.tsv → resolve → s2_batch → meta.py → coverage.py → build.py hattini sirayla yurutur. Kullanici bir makale/DOI/arXiv kimligi eklemek, bir kaynagi tabloya almak ya da literatur tablosunu guncellemek istediginde kullan. literature-table.md'yi asla elle duzenleme.
---

# Kaynak ekle

`docs/thesis/literature-table.md` ve `vision-and-scope.md` §9 **URETILEN** dosyalardir —
elle duzenleme. Tek yol bu hat (`scripts/literature/build.py` docstring'i ile ayni).
Onceki bir YZ taramasi 4 var olmayan kaynak uretti: **kunyeyi bellekten asla yazma.**

## 0. Girdi
Kullanicidan (veya baglamdan) her kaynak icin: DOI, arXiv kimligi ya da tam baslik;
hangi bilesene/bosluga (G1–G5, AQ) dayanak olacagi. Eksikse sor.

## 1. Varlik dogrulamasi (aga cikmadan once yapma — her kaynak icin)
- DOI: `https://api.crossref.org/works/<doi>` → baslik, yazarlar, yayin yeri, yil, tur.
- arXiv: `http://export.arxiv.org/api/query?id_list=<id>` → baslik, `journal_ref`/`comment`
  (hakemli yayinlanmis mi? Yayinlanmissa DOI'sini bul ve DOI ile ekle).
- Yalniz baslik: Crossref `query.bibliographic=` aramasi; ilk 3 sonucu basliga karsilastir.
- Bulunamazsa **DUR**, kullaniciya "dogrulanamadi" de, tabloya ekleme.
- Ozet gerekirse: `python -I -X utf8 scripts/literature/oa_abs.py <doi> ...`

## 2. Secim kurali (tablonun Ek A'siyla tutarli)
SCIE/SSCI dergi ya da CORE A*/A konferans + atif esigi. Istisna: zorunlu atif (Z) ve
yontem kaynagi (Y) kademeden bagimsiz alinir. Kurali gecmiyorsa kullaniciya soyle ve
rol B/I olarak eklenip eklenmeyecegini sor. Zenodo/Research Square/preprint → hakemli sayma.

## 3. Hat
Anahtar bicimi: `SoyadYIL_KisaAd` (ornek `Lee2020_MAPS`). Once `grep` ile anahtarin ve
DOI'nin `scripts/literature/data/seed.tsv`'de olmadigini kontrol et (cift kayit yok).

1. `scripts/literature/data/seed.tsv`'ye TAB ayrimli satir: `key  group  doi:...|arxiv:...|title:...  not`
   (grup: dosyadaki mevcut grup adlarindan biri — once `cut -f2 seed.tsv | sort -u`).
2. Cozumle (ag gerekir):
   ```
   python -I -X utf8 scripts/literature/resolve.py scripts/literature/data/seed.tsv scripts/literature/data/resolved.json
   python -I -X utf8 scripts/literature/s2_batch.py scripts/literature/data/resolved.json scripts/literature/data/s2.json
   ```
   Yeni anahtar resolved.json'da cikti mi, baslik dogru mu kontrol et.
3. `scripts/literature/meta.py` → `ROLE["<key>"] = ("Z|Y|B|I", "tek satir gerekce")` (zorunlu).
   Gerekirse `VENUE` / `INDEX` / `YEAR` / `PDF` duzeltmeleri. Tam metin okunmadiysa gerekceye
   "tam metin okunmali" ekle.
4. `scripts/literature/coverage.py` → kaynagi ilgili bilesen satirinin anahtar listesine ekle.
5. `python -I -X utf8 scripts/literature/build.py`
6. `git diff --stat docs/thesis/` — yalnizca beklenen satirlar degismis olmali.

## 4. Kayit
`docs/thesis/literature-review-verification.md` sonuna kisa not: tarih, anahtar, nasil
dogrulandi (Crossref/arXiv URL'si), secim kurali sonucu.

## 5. Rapor
Kullaniciya: eklenen anahtarlar, rol, yayin yeri/indeks, atif sayisi, dogrulama kaynagi,
reddedilen/dogrulanamayanlar ve nedenleri. Ozette olmayan bir iddiayi kaynaga yukleme.
