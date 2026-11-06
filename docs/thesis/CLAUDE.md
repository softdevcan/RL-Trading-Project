# CLAUDE.md — docs/thesis/ (tez, literatur, akademik yazim)

Kok `CLAUDE.md`'den tasindi; docs/thesis/ altinda calisirken yuklenir.

### Akademik gecerlilik ve literatur (ONEMLI — 2026-10-06)
- **Kanonik tez belgesi `docs/thesis/vision-and-scope.md` (v0.3).** Literatur iddialari
  ve bosluk ifadeleri (G1–G5) oradan alinir.
- **Kaynak tabani `docs/thesis/literature-table.md`** (148 secilmis kayit). Tablo ve
  vision §9 `python scripts/literature/build.py` ile URETILIR — ikisini de elle
  duzenleme; kaynak eklemek icin `scripts/literature/build.py` docstring'indeki adimlar
  (seed.tsv → resolve → s2_batch → meta.py ROLE → coverage.py → build). Secim kurali:
  SCIE/SSCI dergi ya da CORE A*/A konferans + atif esigi; zorunlu atif (Z) ve yontem
  kaynagi (Y) kademeden bagimsiz tutulur. Tam metni okunmamis 2026 kaynaklarindan
  (Hung & Huang, Yuan vd., CAROC, Xiong vd.) ozette olmayan iddia tasima. Seminer belgeleri
  (`docs/thesis/seminar-*.md`) Mayis 2026 anlik goruntusudur; oradan iddia tasima.
- **"Leakage yok" DEME.** 1 gunluk `shift` fiyat turevli ozellikler icin yeterli ama
  kodda dogrulanmis sizinti kaynaklari var: makro `bfill` (varsayilan hatta etkin),
  TUFE/UFE aciklanma gecikmesi yok, statik temel oranlar (opt-in), seri butunune
  ICEEMDAN + hedef filtrelenmis fiyattan (opt-in). Protokol aciklari: RL egitiminde
  tohum yok, DM/t-testi bagli degil, risksiz faiz 3 yerde 3 deger, 1/N/min-var/HRP yok.
  Liste: `docs/development/roadmap.md` → "Bilinen Metodolojik Aciklar"; kapatilma plani
  Faz 9 + Faz 10'da (asagida). Plandaki paket sirasinin disinda "duzeltmeye" girisme.
- **Faz 9 ve Faz 10 planlari ONAYLANDI (2026-11-06):**
  `docs/development/phase-9-research-validity-plan.md`, `docs/development/phase-10-marl-plan.md`
  (§8 hakem-2 eklemeleriyle). Kararlar K1 (varsayilani cevir, golden Milestone 0' ile),
  K17 (RL penceresi 2013-06), K23 (anahtar koddan kaldirildi, yenilenmeyecek).
  **Ilk kapi (V + A + C) yapildi**; durum Faz 9 planinda "Uygulama durumu" bolumunde.
  Tezde istatistik icin `evaluation/` paketi kullanilir. DSR eski haliyle yanlisti (makale
  ornegi 0,9004 → repo 0,4031); duzeltildi, ama tez-final.pdf'te eski fonksiyonla
  raporlanmis bir DSR/DM/t-testi varsa yeniden hesaplanmali. Diger bulgular gecerli:
  oduldeki "Differential Sharpe" turev degil (G); G2'nin log-guven onseli DebUnc dikkat
  olceklemesiyle esdeger.
- **RL sonuclari henuz tez sayisi degil.** Bu kapidan ONCE uretilenler 2000–2018 penceresi,
  komsu sembol kopyasi ve tohumsuz egitimle; SONRA uretilenler de test olcegi test penceresinden
  (Y13/B9), gosterge bfill (Y15/B7) ve ADX = 0 (K18) ile uretildi. Ikisi de yalnizca "on sonuc /
  sinirlilik" olarak anilabilir (Faz 9 §9.2 #1). Deneme kaydinda `protocol_open` etiketlidirler.
  Tam metinler `docs/reference/papers/` (PDF'ler git disi, icindekiler README.md).
- **TATS ozgun degil:** Kazemdehbashi (2026, arXiv 2601.12706). Akademik metinde atif
  zorunlu, katki olarak sunulmaz.
- **Kaynak eklerken dogrula.** Onceki bir YZ taramasi 4 var olmayan kaynak uretmisti.
  Her kunye Crossref/arXiv/yayinci sayfasiyla kontrol edilir; dogrulama kaydi:
  `docs/thesis/literature-review-verification.md`.

### Yazim kurallari
- Her olgusal iddia ya bir kaynaga (literature-table anahtari) ya da repodaki bir olcume
  (dosya/komut) baglanir. Baglanamiyorsa iddia yazilmaz, "dogrulanmadi" diye isaretlenir.
- Kaynak eklemek icin `/kaynak-ekle` skill'ini kullan; kunyeyi asla bellekten yazma.
- Bir bolumu bitirince `hakem-2` alt ajanina okut (metodoloji + iddia denetimi).
- Sonuc rakamlari metne elle kopyalanmaz; hangi `results/` / manifest dosyasindan geldigi yazilir.
- Ozgunluk dili temkinli: "bildigimiz kadariyla", tarama kapsami ve tarihiyle birlikte.
