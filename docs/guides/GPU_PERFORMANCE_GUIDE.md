# 🚀 GPU Performance Testing Guide

RTX 4060 ve Ryzen 7 7800X3D sisteminde GPU kullanımını test etmek için rehber.

## 📋 Ön Hazırlık

### 1. Monitoring Araçlarını Yükle
```bash
pip install psutil nvidia-ml-py3
```

### 2. CUDA PyTorch Kontrolü
PyTorch'un CUDA versiyonunu kontrol et:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Eğer `False` dönerse, CUDA versiyonunu yükle:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧪 Test Adımları

### Test 1: Hızlı GPU Kontrolü (30 saniye)

```bash
python hyperparameter_optimization/test_gpu_performance.py
```

**Beklenen Çıktı:**
```
✅ PyTorch Version: 2.x.x
✅ CUDA Available: True
✅ GPU Device: NVIDIA GeForce RTX 4060
✅ MODEL IS USING GPU!
```

**Sorun Çözme:**
- `CUDA Available: False` → CUDA PyTorch yükle (yukarıdaki komut)
- `GPU available but not used` → `device="auto"` parametresi eksik (zaten eklendi)

---

### Test 2: Real-time Monitoring

İki terminal açın:

**Terminal 1:** Resource Monitor
```bash
python hyperparameter_optimization/monitor_resources.py
```

**Terminal 2:** Optimization Çalıştır
```bash
# Web UI üzerinden veya:
python hyperparameter_optimization/run_optimization.py
```

**Beklenen Davranış:**
| Phase | CPU % | GPU % | GPU Mem | Açıklama |
|-------|-------|-------|---------|----------|
| Başlangıç | 10-20% | 0% | ~0 MB | Veri yükleniyor |
| Model Init | 30-50% | 10-30% | 500-1000 MB | Model GPU'ya taşınıyor |
| Training | 50-80% | **70-100%** | **2000-4000 MB** | ✅ Ana training loop |
| Evaluation | 40-60% | 50-80% | 2000-4000 MB | Validation |

**🎯 GPU Tam Kullanılıyor mu?**
- ✅ GPU % = 70-100% → Mükemmel!
- ⚠️  GPU % = 30-70% → Batch size artırılabilir
- ❌ GPU % = 0-30% → CPU'da çalışıyor olabilir

---

### Test 3: Windows Task Manager

En basit yöntem:

1. **Task Manager** aç (Ctrl + Shift + Esc)
2. **Performance** sekmesi
3. **GPU 0** seçin

**Training sırasında görülmeli:**
- 🔥 GPU Utilization: 70-100%
- 📊 GPU Memory: 2-4 GB kullanımda
- 🌡️ GPU Temperature: 60-80°C arası

---

### Test 4: NVIDIA-SMI (En Detaylı)

Windows PowerShell veya CMD:

```bash
nvidia-smi
```

**Tek Seferlik Bilgi:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx       CUDA Version: 12.x  |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
| 30%   65C    P2    80W / 115W |   3500MiB /  8192MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

**Real-time Monitoring (1 saniye refresh):**
```bash
nvidia-smi -l 1
```

**Beklenen Değerler Training Sırasında:**
- **GPU-Util**: 70-100%
- **Memory-Usage**: 2000-4000 MiB
- **Temp**: 60-80°C
- **Pwr**: 80-115W (RTX 4060 için)

---

## 📊 Performans Benchmark

### CPU-Only vs GPU Karşılaştırması

Test parametreleri:
- Algorithm: PPO
- Timesteps: 50,000
- Stocks: 2 (THYAO, SAHOL)

| Device | Time per Trial | GPU Util | GPU Mem | Expected |
|--------|---------------|----------|---------|----------|
| **CPU Only** | ~150-180 sec | 0% | 0 MB | Baseline |
| **RTX 4060** | **40-60 sec** | 80-95% | 3000 MB | ✅ 3x faster |

---

## 🎯 Optimization İçin Best Practices

### 1. Batch Size Artırın (GPU için)
Eğer GPU memory %50'nin altındaysa:

`search_spaces.py` içinde:
```python
"batch_size": {
    "choices": [128, 256, 512, 1024],  # 1024 ekledik
}
```

### 2. Paralel Trials (CPU için)
Optuna n_jobs artırılabilir (GPU için değil):
```python
# GPU kullanırken:
n_jobs = 1  # ✅ Doğru (GPU sequentially daha hızlı)

# CPU kullanırken:
n_jobs = 4  # ✅ 4 trial parallel (Ryzen 7 7800X3D için)
```

### 3. Mixed Precision Training
Daha hızlı training için (opsiyonel):
```python
# policy_kwargs içine ekle:
policy_kwargs = dict(
    net_arch=net_arch,
    # activation_fn=nn.ReLU,  # Faster than default
)
```

---

## 🔧 Sorun Giderme

### Problem 1: GPU Kullanılmıyor
**Belirtiler:**
- nvidia-smi'da Python process görünmüyor
- GPU % = 0
- Training CPU gibi yavaş

**Çözüm:**
```bash
# 1. CUDA PyTorch yükle
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Test et
python -c "import torch; print(torch.cuda.is_available())"

# 3. Model oluşturmayı test et
python hyperparameter_optimization/test_gpu_performance.py
```

### Problem 2: Out of Memory (OOM)
**Belirtiler:**
- RuntimeError: CUDA out of memory

**Çözüm:**
```python
# Batch size küçült veya:
# Model size'ı küçült
"net_arch_size": "small"  # medium yerine
```

### Problem 3: GPU Utilization Düşük (<50%)
**Olası Nedenler:**
- Environment step slow (CPU bottleneck)
- Batch size çok küçük
- Network çok küçük

**Çözüm:**
- Batch size artır: 256 → 512
- Network size: small → medium

---

## ✅ Başarı Kriterleri

Optimizasyon sırasında:
- ✅ **GPU Util**: 70-100%
- ✅ **GPU Mem**: 2-4 GB (8GB RTX 4060 için)
- ✅ **Time per Trial**: 40-60 saniye (50K timesteps)
- ✅ **CPU**: 50-80% (data preprocessing için)

---

## 🚀 Hızlı Komutlar

```bash
# Tek komutla test
python hyperparameter_optimization/test_gpu_performance.py

# Real-time monitoring
python hyperparameter_optimization/monitor_resources.py

# NVIDIA monitoring
nvidia-smi -l 1

# Full optimization (10 trials)
# Web UI: http://localhost:8000/hyperopt.html
# n_trials=10, timesteps=50000
```

---

**🎉 İyi Optimizasyonlar!**
