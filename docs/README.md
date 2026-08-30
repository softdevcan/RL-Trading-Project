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

- [**roadmap.md**](development/roadmap.md) — Geliştirme planı ve faz takibi (Faz 1-3 + Faz 6 + Faz 7 tamamlandı)
- [**prediction-system.md**](development/prediction-system.md) — Gelişmiş tahmin sistemi mimarisi (ensemble, feature engineering, veri katmanı)
- [**phase3-implementation.md**](development/phase3-implementation.md) — Faz 3 uygulama detayları (bug fix'ler, ICEEMDAN, TATS, ATR/Kelly, SHAP)
- [**phase-6-backend-performance.md**](development/phase-6-backend-performance.md) — Faz 6: backend performans & eğitim throughput sprint'i (DL ince ayar, sembol paralelliği, eğitim manifesti, güvenilirlik) + GPU test/kapanış kaydı
- [**phase7-auth.md**](development/phase7-auth.md) — Faz 7: kimlik doğrulama, roller, kullanıcı bazlı çalışma alanları (hibrit izolasyon), kurulum
- [**phase-8-ui-theming.md**](development/phase-8-ui-theming.md) — Faz 8: aydınlık/koyu/sistem teması, token katmanı, bileşen sadeleştirme, kontrast denetimi
- [**rl-stability-portfolio-analysis.md**](development/rl-stability-portfolio-analysis.md) — RL eğitim stabilite riskleri + arayüz akışı (veri indir → eğit → analiz) + portföy yönetimi hazırlık değerlendirmesi (salt analiz, 2026-05-12)

---

## 🎓 Thesis — Tez Vizyonu ve Yol Haritası

Tez kapsamını, akademik konumu ve süreç planını içeren dokümanlar:

- [**vision-and-scope.md**](thesis/vision-and-scope.md) — Tez vizyonu, literatür konumu, tasarım seçenekleri değerlendirmesi, kapsam (in/out), başarı kriterleri, milestone haritası ve makale hedefleri
- [**seminar-overview.md**](thesis/seminar-overview.md) — Danışman görüşmesi / seminer için proje tanıtım belgesi (Faz 1-3 özeti + tez vizyonu + tartışma noktaları)
- [**seminar-presentation.md**](thesis/seminar-presentation.md) — 10 slaytlık seminer sunumu içeriği (her slayt: başlık + maddeler + konuşma metni + görsel önerisi; Notebook LM için)

---

## 📑 Reference — Referans Materyaller

Harici dokümanlar ve tez:

- [**tez-final.pdf**](reference/tez-final.pdf) — Tez final sürümü
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
2. [development/phase3-implementation.md](development/phase3-implementation.md) — uygulama detayları

---

## 📝 Katkı

Yeni dokümantasyon eklerken:

1. **Aktif mi arşiv mi?** Belirli bir faz/sprint'e özgü notlarsa `archive/` altına; kalıcı referanssa `guides/` veya `development/` altına.
2. **İndeksi güncelleyin** — bu dosyaya (`docs/README.md`) eklediğiniz dosyanın linkini koyun.
3. **Ana README'deki referansları** kontrol edin.
