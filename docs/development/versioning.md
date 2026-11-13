# Sürümleme

**Tek kaynak:** `app/__init__.py` → `__version__`. `app/core/config.py` `API_VERSION`'ı oradan alır ve
ortam değişkeniyle ezilmez (eski `.env`'lerdeki `API_VERSION` satırı yok sayılır). Sürüm `/health`,
API başlığı, RL ortam kaydı (`evaluation/reproducibility.py::environment_record` → `app_version`) ve
tahmin manifestinde (`prediction/manifest.py` → `app_version`) görünür; git commit ayrıca kayıtlıdır.

## Kural (SemVer)

| Artış | Ne zaman | Örnek |
|---|---|---|
| **MAJOR** | Eğitilmiş modelleri ya da kayıtlı sonuçları bozan değişiklik | RL durum uzayı, golden yenileme (Milestone 0'), Faz 10 MARL ortamı |
| **MINOR** | Plan paketi kapanır (davranış değişebilir; PR'da beyan edilir) | B11, Y28 |
| **PATCH** | Davranışı değiştirmeyen düzeltme, belge, test | yazım, test onarımı |

Başlangıç: **v1.9.0** (2026-11-14; Faz 9 sürüyor, main `cc0a0e6` + bu değişiklik).

## Yayın adımları (her birleştirmeden sonra)

1. Sürümü yükselten değişiklik paketin PR'ında `app/__init__.py`'de yapılır.
2. Birleştirmeden sonra main'de:
   ```bash
   git checkout main && git pull
   git tag -a v1.10.0 -m "v1.10.0: <paket>"
   git push origin v1.10.0
   gh release create v1.10.0 --generate-notes --title "v1.10.0 — <paket>"
   ```
   Sürüm notu PR başlıklarından otomatik üretilir; elle CHANGELOG tutulmaz. Golden'ı ya da eğitilmiş
   modelleri etkileyen değişiklik notun başına elle yazılır.

## Araştırma etiketleri

Tez ya da makaleye giren sonuç, ürettiği kodun commit'ine **adlandırılmış** bir tag alır
(`m0-baseline`, `paper2-submission`, …). Ölçüm JSON'ları ve manifestler zaten commit taşır; tag,
"bu tablo hangi kodla" sorusunu tek etikete bağlar. Bu tag'ler silinmez ve taşınmaz.

## main koruması

GitHub'da main korumalı (2026-11-14): birleştirme yalnız PR ile (onay şartı 0, tek geliştirici),
force push ve dal silme kapalı, yönetici muafiyeti açık.
