# 🔧 Scripts

Bu klasör tüm utility scriptleri içerir.

## 📁 Klasör Yapısı

### `/training` - Model Eğitimi
- **train_a2c_phase1.py** - Faz 1 A2C model eğitimi (standalone)

**Kullanım:**
```bash
python scripts/training/train_a2c_phase1.py
```

---

### `/benchmarking` - Performans Testleri
- **benchmark_strategies.py** - Buy-and-Hold ve BIST-30 Index stratejileri
- **test_benchmarks.py** - Benchmark testleri

**Kullanım:**
```bash
# Benchmark stratejilerini test et
python scripts/benchmarking/test_benchmarks.py

# Modülü kullan
from scripts.benchmarking.benchmark_strategies import BuyAndHoldStrategy
```

---

### `/analysis` - Analiz ve Raporlama
- **generate_academic_report.py** - Akademik kalite raporlar oluşturur
- **extract_comparison_table.py** - Model karşılaştırma tablosu
- **probe_timeseries_structure.py** - Getiri/volatilite otokorelasyonu + mevsimsellik ölçümü (ARIMA/SARIMAX/GARCH kararına veri sağlar; `statsmodels` gerektirmez)

**Kullanım:**
```bash
# Tüm modelleri karşılaştır ve rapor oluştur
python scripts/analysis/generate_academic_report.py

# Karşılaştırma tablosu çıkar
python scripts/analysis/extract_comparison_table.py

# Zaman serisi yapısını ölç (bulgular: docs/development/prediction-timeseries-models.md)
python scripts/analysis/probe_timeseries_structure.py
```

**Çıktılar:**
- `results/figures/` - Grafikler (PDF, 300 DPI)
- `results/latex/` - LaTeX tabloları
- `results/data/` - CSV ve JSON

---

### `/debug` - Debug Araçları
- **debug_model_actions.py** - Model action debugging
- **fix_learning_rates.py** - Learning rate düzeltmeleri

**Kullanım:**
```bash
# Model aksiyon debug
python scripts/debug/debug_model_actions.py

# Learning rate sorunlarını düzelt
python scripts/debug/fix_learning_rates.py
```

---

## 🚀 Hızlı Başlangıç

### Web Dashboard Üzerinden (Önerilen)
```bash
python run_server.py
# http://localhost:8000
```

### Standalone Training
```bash
python scripts/training/train_a2c_phase1.py
```

### Benchmark Karşılaştırması
```bash
python scripts/benchmarking/test_benchmarks.py
```

### Akademik Rapor
```bash
python scripts/analysis/generate_academic_report.py
```

---

## ⚠️ Import Notları

Scriptler proje root'undan çalıştırılmalıdır:

```bash
# ✅ Doğru
python scripts/training/train_a2c_phase1.py

# ❌ Yanlış
cd scripts/training
python train_a2c_phase1.py  # Import hataları verir
```

Eğer import sorunları yaşıyorsanız:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
```

---

## 📝 Yeni Script Ekleme

1. Uygun kategori klasörüne ekleyin
2. Bu README'yi güncelleyin
3. Import path'leri kontrol edin
4. Kullanım örneği ekleyin
