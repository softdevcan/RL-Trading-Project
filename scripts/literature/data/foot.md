
## 5. Künye düzeltmeleri (bu çalışmada bulunanlar)

| Kayıt | Önceki belgelerde | Doğrusu | Kaynak |
|---|---|---|---|
| Candar (2022) | Yalnızca İTÜ YL tezi | Ayrıca **Candar & Üstündağ (2022)**, *Intelligent and Fuzzy Systems* (INFUS 2022), LNNS 505, DOI 10.1007/978-3-031-09176-6_16 — birincil atıf bu | Crossref |
| DebUnc (Yoffe vd.) | "Ön baskı, arXiv 2407.06426" | **Findings of EMNLP 2025**, DOI 10.18653/v1/2025.findings-emnlp.1265 | Semantic Scholar / DBLP |
| StockMARL (Zou & Siebers) | "EMSS 2025" | *Procedia Computer Science*, DOI 10.1016/j.procs.2025.12.091 | Crossref |
| MAETS (Chaudhari 2026) | — | OpenAlex TNNLS diyor, **Crossref'te TNNLS kaydı yok**; Zenodo deposu. Elendi | Crossref |
| Ansari ekibi (Sattar vd. 2025) | Doğrulama raporunda yalnızca BibTeX künyesi | *IEEE Access* (SCIE), 25+ atıf. Başlıktaki **RMS = Reports + Max drawdown + Sentiment**. tez-final.pdf'teki "Root Mean Square" yorumu bu nedenle yanlış | Özet |
| Yang vd. (2024), Sci Rep | Sızıntının finans kanıtı gibi anılıyor | Veri **su kalitesi** — yöntem kaynağı. Finans kanıtı Carnelossi Furlaneto (2017) ve **Xiong vd. (2025)** | Tam metin |
| "Advances in Financial Machine Learning" | — | OpenAlex *Quantitative Finance*'taki bir **kitap incelemesine** eşliyor; asıl kaynak Wiley 2018 kitabı | Elle |
| The Journal of Financial Data Science | MJL kontrolünde "Journal of Finance and Data Science" ile karıştı | İki ayrı dergi (§4.1) | Crossref / ISSN |
| Hu vd. CommFormer | Yalnızca ICLR 2024 | Dergi sürümü: *IEEE TKDE* 2026 (SCIE) | Crossref |

## 6. Sınırlılıklar

- **İndeks:** Wikidata'da kaydı olmayan ilk tur dergileri MJL'de elle kontrol edildi. İkinci turda eklenen 10 derginin durumu bilgiye dayalı (§4.2); hepsi köklü dergiler ama elle teyit edilmeli. İndeks durumu yıllık değişebilir; makale gönderiminden önce yeniden bakılmalı.
- **Atıf sayıları** anlık görüntüdür (2026-10-07) ve kaynaklar arasında farklıdır.
- **Tarama kapsamı:** arXiv, OpenAlex, Crossref, Semantic Scholar ve DergiPark kullanıldı. Scopus, Web of Science, IEEE Xplore ve YÖK Tez Merkezi taranmadı; YÖK taraması G4 ("BIST'te MARL bulunamadı") iddiası için hâlâ gerekli.
- **Tam metni okunmamış 2026 kayıtları:** Hung & Huang (LNCS), Yuan vd. (EAAI), Lin & Zeng (CAROC), Xiong vd. (Applied Intelligence). Rolleri özete ya da başlığa dayanıyor; G1/G2 ifadeleri kesinleşmeden önce okunmalı.
- **Anahtar kelimeler** otomatik atanmıştır; tezde yazar anahtar kelimeleri gerekirse makale sayfasından alınmalı.

## 7. Sonraki adım: belgelerin güncellenmesi

Hepsi 2026-10-07'de yapıldı:

1. ✅ `vision-and-scope.md` **v0.3**: §2 yeniden yazıldı. G1/G2 yeni öncüllere göre gözden geçirildi (Hung & Huang 2026, CAROC 2025, Öztürk 2026; G2'de DebUnc eşdeğerliği). §6.2 Makale 2'ye Xiong vd. 2025 eklendi. §9 bu tablodan üretiliyor.
2. ✅ `literature-review-verification.md` §8: bu turun düzeltmeleri ve yeni yakın çalışmalar.
3. ✅ `phase-9-research-validity-plan.md`: elenen kaynak atıfları seçilmiş muadilleriyle değiştirildi (Xue & Ye → Jiang vd. 2017; Verma → Hamilton 1989 + Nystrup 2018; Farr → tasarım varsayımı; Niyogi → Xiong vd. 2025).
4. ✅ `docs/reference/papers/README.md`: 26 yeni PDF; elle getirilecekler listesine 7 kaynak.

**Yeniden üretim:** `python scripts/literature/build.py`. Hem bu tabloyu hem vision §9'u yazar; ikisi de elle düzenlenmemeli. Kaynak ekleme adımları betiğin açıklamasında.
