# 📚 Dokümantasyon

Bu klasör, **RL Trading System** projesinin tüm dokümantasyonunu içerir. Faz 1 → Faz 3 süreci tamamlandıktan sonra yeniden organize edilmiştir.

## 📁 Klasör Yapısı

```
docs/
├── guides/            # Kullanım kılavuzları (aktif)
├── development/       # Geliştirme notları ve mimari (aktif)
├── thesis/            # Tez vizyonu, kapsam ve yol haritası (aktif)
├── reference/         # Harici referans materyaller (PDF'ler, tez)
└── archive/           # Tamamlanmış faz/sprint dokümanları (arşiv)
```

---

## 📖 Guides — Kullanım Kılavuzları

Günlük kullanım için başvuru dokümanları:

- [**ALGORITHMS.md**](guides/ALGORITHMS.md) — RL algoritma karşılaştırması (A2C, PPO, TD3, SAC)
- [**ACADEMIC_GUIDE.md**](guides/ACADEMIC_GUIDE.md) — Akademik analiz, metrik hesaplamaları, LaTeX çıktısı
- [**API_HYPEROPT_GUIDE.md**](guides/API_HYPEROPT_GUIDE.md) — Hiperparametre optimizasyonu (Optuna + RL algoritmaları)
- [**GPU_PERFORMANCE_GUIDE.md**](guides/GPU_PERFORMANCE_GUIDE.md) — GPU performans testleri ve ipuçları

---

## 🔧 Development — Geliştirme Dokümanları

Proje mimarisi ve roadmap:

- [**system-map.md**](development/system-map.md) — **Sistem akış haritası (canlı):** veri akışı (kaynak → veri → tahmin → RL → değerlendirme → ürün) ve geliştirme aşamaları (Faz 1–10, Milestone 0', Makale 2, tez) durum renkleriyle; durum değiştiren her PR günceller
- [**roadmap.md**](development/roadmap.md) — Geliştirme planı ve faz takibi (Faz 1-3 + Faz 6 + Faz 7 + Faz 8 tamamlandı)
- [**prediction-system.md**](development/prediction-system.md) — Gelişmiş tahmin sistemi mimarisi (ensemble, feature engineering, veri katmanı)
- [**phase-3-implementation.md**](development/phase-3-implementation.md) — Faz 3 uygulama detayları (bug fix'ler, ICEEMDAN, TATS, ATR/Kelly, SHAP)
- [**phase-6-backend-performance.md**](development/phase-6-backend-performance.md) — Faz 6: backend performans & eğitim throughput sprint'i (DL ince ayar, sembol paralelliği, eğitim manifesti, güvenilirlik) + GPU test/kapanış kaydı
- [**phase-7-auth.md**](development/phase-7-auth.md) — Faz 7: kimlik doğrulama, roller, kullanıcı bazlı çalışma alanları (hibrit izolasyon), kurulum
- [**phase-8-ui-theming.md**](development/phase-8-ui-theming.md) — Faz 8: aydınlık/koyu/sistem teması ve token katmanı (A–E), profil sayfası (F), üst çubuk + bildirimler (G), dar ekranda ikon rayı (H), model ve optimizasyon kaydı silme (I). Kapsam özeti belgenin sonundaki "Kapanış özeti"nde.
- [**rl-stability-portfolio-analysis.md**](development/rl-stability-portfolio-analysis.md) — RL eğitim stabilite riskleri + arayüz akışı (veri indir → eğit → analiz) + portföy yönetimi hazırlık değerlendirmesi (salt analiz, 2026-05-12)
- [**phase-9-research-validity-plan.md**](development/phase-9-research-validity-plan.md) — Faz 9 planı (2026-10-07; **onaylandı 2026-11-06**, ilk kapı V + A + C uygulandı → §9): sızıntı ve değerlendirme açıklarının kapatılması (A–G), Makale 2 deney tasarımı, MARL tasarım kararları (M1–M8); tam metinle doğrulanan 10 yeni bulgu (DSR formül hatası, G2 ≡ DebUnc eşdeğerliği dahil)
- [**milestone0-protocol.md**](development/milestone0-protocol.md) — Milestone 0' protokolü (m0p-v1, taslak): etkenler, aşamalar, budama kuralı, çıkarım, beyan listesi, bütçe; koşum `scripts/run_milestone0.py`
- [**phase-10-marl-plan.md**](development/phase-10-marl-plan.md) — Faz 10 planı (**onaylandı 2026-11-06**): çoklu-ajan yenileme (hisse başına ajan, CTDE), tahmin → RL sinyal tablosu (P), karar katmanı (K), veri bütünlüğü kapısı (V); hakem-2 itirazları §8, kararlar §6 (K1, K17, K23 işlendi)
- [**phase-10-m2-design.md**](development/phase-10-m2-design.md) — Faz 10 M2 tasarımı (v0.1, 2026-11-16): zaman indeksli muhasebe denklemi, logit sınırı 5 + ln N, risk projeksiyonu, K5 tam sayı emir, `obs_spec` v1, K2 paritesi; kod `marl/`, açık maddeler §9
- [**phase-10-m3-design.md**](development/phase-10-m3-design.md) — Faz 10 M3 tasarımı ve ön kayıtları: M3a boru hattı kapısı (GEÇTİ, §5.2), M3b 20 bps kapısı (KISMİ, §6.7), M4 temeli (§6.8)
- [**sozluk.md**](sozluk.md) — **Sistem sözlüğü** (K28 R1, 73 terim): aşama adları, deney kodları, etiketler, istatistik ve MARL terimleri; her terimde sade + teknik tanım. ÜRETİLİR (`python scripts/build_guide.py`), tek kaynak `dashboard/content/glossary.yaml`
- [**system-guide-plan.md**](development/system-guide-plan.md) — Sistem rehberi ve sözlük planı (v0.3, 2026-11-19; onaylı, K28; R1 bitti): tek kaynaklı sözlük (sade + teknik), panoda "Rehber" sayfası, iş paketleri R1–R4
- [**phase-10-m4-design.md**](development/phase-10-m4-design.md) — Faz 10 M4 tasarımı (v0.1, 2026-11-19): `attn`/`gat` eşit bütçe (78.982 ↔ 78.850), GAT graf kuralı ölçümü (`sector + mst`), M4a komşu sinyali pozitif kontrolü ön kayıt taslağı
- [**phase-10-m5-design.md**](development/phase-10-m5-design.md) — Faz 10 M5 ön kaydı (v0.8, 2026-11-19; mekanizma kolları koşuldu: H2 KISMİ, gürültü tabanı aşılmadı, §4.11). İlk sürüm (v0.1): P5c okuma kuralı — kapıdan geçmeyen gerçek sinyalin neyi sınadığı, okuma tablosu O1–O4, yasak kurtarma yolları; kollar M4'ten sonra
- [**review-panel.md**](development/review-panel.md) — Hakem paneli (2026-11-06): `hakem-finans` + `hakem-rl` (sonnet, dar kapsam) ve son kapı `hakem-2` (opus); paket → panel eşlemesi, inceleme paketi şablonu, token maliyet kaydı (B9: tek hakem-2 ~301 bin token)
- [**measurements/**](development/measurements/) — Plan §9'daki sayıların kaynağı olan ölçüm JSON'ları (komut, commit, panel SHA-256); ilk: `phase9-b9/`
- [**prediction-timeseries-models.md**](development/prediction-timeseries-models.md) — Ensemble'a ARIMA/SARIMAX eklemenin değerlendirmesi: getiri vs. volatilite yapısı ve mevsimsellik ölçümü (30 sembol, 76.058 gözlem); GARCH ve yön meta-learner'ı önerileri (salt analiz, 2026-08-30); §8 TimesFM-3 deneyi (`experiment/timesfm` dalı) ve `predict_next` son-satır iyileştirmesi (2026-11-06)

---

## 🎓 Thesis — Tez Vizyonu ve Yol Haritası

Tez kapsamını, akademik konumu ve süreç planını içeren dokümanlar:

- [**vision-and-scope.md**](thesis/vision-and-scope.md) — **Kanonik tez belgesi (v0.2, 2026-10-06).** Tez vizyonu, doğrulanmış literatür konumu ve G1–G5 boşluk tablosu, tasarım seçenekleri, kapsam (in/out), değerlendirme protokolü, başarı kriterleri, milestone haritası, makale hedefleri, doğrulanmış kaynakça
- [**literature-table.md**](thesis/literature-table.md) — **Seçilmiş kaynakça (2026-10-07): 169 kayıt değerlendirildi, 148 seçildi** (85 SCIE/SSCI dergi, 41 üst düzey konferans). Her kaynak için başlık, yazar, yıl, yayın yeri, indeks, atıf, anahtar kelimeler, erişim ve rol; tez/kod bileşenlerini kaynaklara bağlayan kapsam matrisi (açık: 0); elenenler ve yerlerine kullanılacaklar. **Tezin ve makalelerin kaynak tabanı budur**
- [**literature-review-verification.md**](thesis/literature-review-verification.md) — Ekim 2026 literatür taramasının bağımsız doğrulaması: kaynak gerçekliği (Crossref/arXiv), künye düzeltmeleri, tam metin iddia kontrolleri, eklenen kaynaklar ve kodda doğrulanan sızıntı/protokol açıkları (2026-10-06)
- [**literature-review-prompt.md**](thesis/literature-review-prompt.md) — Literatür taramasını üreten araştırma promptu (tarama yönteminin kaydı)
- [**seminar-overview.md**](thesis/seminar-overview.md) — Danışman görüşmesi / seminer için proje tanıtım belgesi (Faz 1-3 özeti + tez vizyonu + tartışma noktaları). *Mayıs 2026 anlık görüntüsü; geçerliliğini yitiren ifadeler belgenin başındaki sürüm notunda*
- [**seminar-presentation.md**](thesis/seminar-presentation.md) — 10 slaytlık seminer sunumu içeriği (her slayt: başlık + maddeler + konuşma metni + görsel önerisi; Notebook LM için). *Mayıs 2026 anlık görüntüsü* — diğer seminer materyalleri (`seminar-*.md`) de aynı durumda

---

## 📑 Reference — Referans Materyaller

Harici dokümanlar ve tez:

- [**tez-final.pdf**](reference/tez-final.pdf) — *BIST-30 Endeksinde Derin Pekiştirmeli Öğrenme ile Algoritmik Ticaret Stratejilerinin Karşılaştırmalı Analizi* (Faz 1-2 sonuçları). Doğrulamada bulunan tutarsızlıklar: [literature-review-verification.md §4](thesis/literature-review-verification.md)
- [**Literatür Taraması ... (Ekim 2026).md**](<reference/Literatür Taraması Tahmin-Destekli İletişimli MARL ile BIST-30 Portföy Yönetimi (Ekim 2026).md>) — 121 kaynaklı literatür taraması (claude.ai Research çıktısı). Künye/kalite düzeltmeleri için doğrulama raporu esas alınmalı
- [**papers/**](reference/papers/README.md) — 66 makalenin tam metin arşivi (PDF'ler git dışı) + elle getirilecek 19 kaynağın bağlantıları
- [**evds-python-kilavuz.pdf**](reference/evds-python-kilavuz.pdf) — TCMB EVDS Python kılavuzu
- [**evds-web-servis-kilavuz.pdf**](reference/evds-web-servis-kilavuz.pdf) — TCMB EVDS Web Servis kılavuzu

---

## 🗄️ Archive — Arşivlenmiş Dokümanlar

Tamamlanmış faz/sprint dokümanları. Güncel geliştirmede referans için saklanır; **günlük kullanımda bu klasöre bakmayın.**

- [archive/phase2/](archive/phase2/) — Faz 2 sprint dokümanları (gereksinimler, analiz)
- [archive/sprint-notes/](archive/sprint-notes/) — Faz 2 sprint tamamlama notları (PSR reward, hyperopt, UI updates)
- [archive/issues/](archive/issues/) — Issue tracker (Batch 0-8 sırasında kullanıldı, 62/63 issue kapatıldı)

---

## 🔗 Hızlı Başlangıç

**Yeni başlayanlar:**
1. Ana [README.md](../README.md) — proje genel tanıtımı ve kurulum
2. [guides/ALGORITHMS.md](guides/ALGORITHMS.md) — RL algoritmalarını tanıyın
3. [development/roadmap.md](development/roadmap.md) — proje yol haritası

**Akademik kullanım:**
1. [guides/ACADEMIC_GUIDE.md](guides/ACADEMIC_GUIDE.md) — metrik ve rapor üretimi
2. `python scripts/analysis/generate_academic_report.py`

**Hiperparametre optimizasyonu:**
1. [guides/API_HYPEROPT_GUIDE.md](guides/API_HYPEROPT_GUIDE.md)
2. [guides/GPU_PERFORMANCE_GUIDE.md](guides/GPU_PERFORMANCE_GUIDE.md)

**Tahmin sistemi (Faz 2/3):**
1. [development/prediction-system.md](development/prediction-system.md) — mimari
2. [development/phase-3-implementation.md](development/phase-3-implementation.md) — uygulama detayları

---

## 📝 Katkı

Yeni dokümantasyon eklerken:

1. **Aktif mi arşiv mi?** Belirli bir faz/sprint'e özgü notlarsa `archive/` altına; kalıcı referanssa `guides/` veya `development/` altına.
2. **İndeksi güncelleyin** — bu dosyaya (`docs/README.md`) eklediğiniz dosyanın linkini koyun.
3. **Ana README'deki referansları** kontrol edin.
