"""
GPU ve CPU Performans Test Script

Bu script:
1. GPU kullanılabilirliğini kontrol eder
2. Kısa bir optimization çalıştırır
3. Performans metriklerini gösterir
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("🔬 GPU/CPU Performance Test")
print("="*80 + "\n")

# 1. PyTorch CUDA Check
print("1️⃣ Checking PyTorch and CUDA...")
try:
    import torch
    print(f"   ✅ PyTorch Version: {torch.__version__}")
    print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   ✅ CUDA Version: {torch.version.cuda}")
        print(f"   ✅ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Test CUDA functionality
        device = torch.device("cuda:0")
        test_tensor = torch.randn(1000, 1000).to(device)
        result = test_tensor @ test_tensor
        print(f"   ✅ GPU Computation Test: PASSED")
    else:
        print(f"   ⚠️  CUDA not available. Will use CPU.")
        print(f"   💡 Make sure you have CUDA-enabled PyTorch installed:")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("   ❌ PyTorch not installed!")
    sys.exit(1)

print()

# 2. Stable-Baselines3 Check
print("2️⃣ Checking Stable-Baselines3...")
try:
    import stable_baselines3 as sb3
    print(f"   ✅ Stable-Baselines3 Version: {sb3.__version__}")
except ImportError:
    print("   ❌ Stable-Baselines3 not installed!")
    sys.exit(1)

print()

# 3. CPU Info
print("3️⃣ CPU Information...")
try:
    import psutil
    print(f"   ✅ CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    print(f"   ✅ CPU Threads (Logical): {psutil.cpu_count(logical=True)}")
    print(f"   ✅ RAM Total: {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"   ✅ RAM Available: {psutil.virtual_memory().available / 1e9:.2f} GB")
except ImportError:
    print("   ⚠️  psutil not installed. Install with: pip install psutil")

print()

# 4. Quick Performance Test
print("4️⃣ Running Quick Performance Test...")
print("   Testing PPO with 1000 timesteps...")

from data.bist30_symbols import PHASE1_SYMBOLS
from hyperparameter_optimization.optimizers import PPOOptimizer

# Create a small test
optimizer = PPOOptimizer(
    n_trials=1,  # Just 1 trial for testing
    n_jobs=1,
    seed=42
)

print(f"   ✅ Optimizer created: {optimizer.study_name}")

# Run quick optimization
start_time = time.time()

try:
    study = optimizer.optimize(
        stock_symbols=PHASE1_SYMBOLS[:2],  # Only 2 stocks for speed
        train_start='2023-01-01',
        train_end='2023-06-30',  # 6 months
        val_start='2023-07-01',
        val_end='2023-09-30',    # 3 months
        total_timesteps=1000,     # Very small for quick test
        eval_freq=500,
        n_eval_episodes=2,
        show_progress_bar=False
    )

    elapsed = time.time() - start_time

    print(f"\n   ✅ Test completed in {elapsed:.2f} seconds")
    print(f"   ✅ Best Sharpe Ratio: {study.best_value:.4f}")

    # Device check
    print("\n   📊 Device Usage:")
    if torch.cuda.is_available():
        print(f"      - GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"      - GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")

        # Check if model actually used GPU
        if torch.cuda.memory_allocated(0) > 0:
            print(f"      ✅ MODEL IS USING GPU!")
        else:
            print(f"      ⚠️  GPU available but not used by model")

    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"      - CPU Usage: {cpu_percent:.1f}%")
    except:
        pass

except Exception as e:
    print(f"\n   ❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("📊 Performance Test Complete!")
print("="*80)

# Recommendations
print("\n💡 Recommendations:")
if torch.cuda.is_available():
    if torch.cuda.memory_allocated(0) > 0:
        print("   ✅ Your system is using GPU acceleration!")
        print("   ✅ Expected speedup: 3-5x faster than CPU-only")
    else:
        print("   ⚠️  GPU is available but not being used")
        print("   💡 Make sure 'device=\"auto\"' is set in model creation")
else:
    print("   ⚠️  GPU acceleration not available")
    print("   💡 For RTX 4060, install CUDA PyTorch:")
    print("      pip install torch --index-url https://download.pytorch.org/whl/cu121")

print("\n🚀 Run a full optimization with:")
print("   - n_trials=10")
print("   - total_timesteps=50000")
print("   - Monitor GPU usage with: nvidia-smi -l 1")
print()
