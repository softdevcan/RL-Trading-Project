# Faz 9 / B9 ölçümleri

[Faz 9 §9.5](../../phase-9-research-validity-plan.md)'teki sayıların kaynağı. Dosyalar elle düzenlenmez. Yeniden üretmek için komutu depo kökünden koşturun; çıktıyı bu dizine yazmak için `--out` verin.

| Dosya | Komut | İçerik |
|---|---|---|
| `price_scale_saturation.json` | `python scripts/analysis/measure_price_scale.py --mode saturation --out docs/development/measurements/phase9-b9/price_scale_saturation.json` | Doğrulama ve test pencerelerinde fiyat z kırpması, eğitim desteği dışı oran, blok başına sabit boyut oranı. Ölçek iki türlü: eğitim penceresi (B9) ve pencerenin kendisi (eski davranış) |
| `price_scale_effect.json` | `python scripts/analysis/measure_price_scale.py --mode effect --algos A2C,PPO --seeds 1,2,3 --steps 50000 --out docs/development/measurements/phase9-b9/price_scale_effect.json` | Tohum başına aynı model, test penceresinde iki ölçekle. Ham ve fazla getiri Sharpe'ı, eşleştirilmiş farkın %95 GA'sı, işlem sayısı, blok başına kırpma, günlük getiriler. Bağlam: al-tut ve risksiz birikim |

Her JSON şunları taşır:
- `command`;
- `git_commit` (ölçümü üreten kod);
- `panel_sha256` (`evaluation.reproducibility.frame_sha256`, RL penceresindeki panel);
- bölme tarihleri.

Panel tazelenince SHA değişir. O durumda sayılar §9.5'teki değerlerle karşılaştırılamaz; yeniden ölçülür.
