"""
Real-time Resource Monitoring

Hyperparameter optimization çalışırken CPU ve GPU kullanımını izler.
"""

import time
import sys
from datetime import datetime

print("\n" + "="*80)
print("📊 Real-time Resource Monitor")
print("="*80)
print("Press Ctrl+C to stop\n")

# Check dependencies
try:
    import psutil
except ImportError:
    print("❌ psutil not installed. Install with: pip install psutil")
    sys.exit(1)

try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}\n")
    else:
        print("⚠️  No GPU detected\n")
except ImportError:
    print("⚠️  PyTorch not installed\n")
    gpu_available = False

# Headers
print(f"{'Time':<10} {'CPU %':<10} {'RAM %':<10} {'RAM GB':<10}", end="")
if gpu_available:
    print(f" {'GPU %':<10} {'GPU Mem':<15} {'GPU Temp':<10}", end="")
print()
print("-" * 80)

try:
    while True:
        timestamp = datetime.now().strftime("%H:%M:%S")

        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used_gb = ram.used / 1e9

        print(f"{timestamp:<10} {cpu_percent:<10.1f} {ram_percent:<10.1f} {ram_used_gb:<10.2f}", end="")

        # GPU (if available)
        if gpu_available:
            try:
                # GPU utilization (requires nvidia-ml-py3)
                try:
                    import pynvml
                    if not hasattr(monitor_resources, 'nvml_initialized'):
                        pynvml.nvmlInit()
                        monitor_resources.nvml_initialized = True

                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                    gpu_util = utilization.gpu
                    gpu_mem_used = memory_info.used / 1e6  # MB
                    gpu_mem_total = memory_info.total / 1e6  # MB
                    gpu_temp = temperature

                    print(f" {gpu_util:<10.1f} {f'{gpu_mem_used:.0f}/{gpu_mem_total:.0f} MB':<15} {gpu_temp:<10.0f}", end="")

                except ImportError:
                    # Fallback to PyTorch
                    mem_allocated = torch.cuda.memory_allocated(0) / 1e6
                    mem_reserved = torch.cuda.memory_reserved(0) / 1e6
                    print(f" {'N/A':<10} {f'{mem_allocated:.0f}/{mem_reserved:.0f} MB':<15} {'N/A':<10}", end="")

            except Exception as e:
                print(f" {'Error':<10} {'N/A':<15} {'N/A':<10}", end="")

        print()  # New line
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n✅ Monitoring stopped")
    if gpu_available and hasattr(monitor_resources, 'nvml_initialized'):
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except:
            pass

print("\n💡 Tips:")
print("   - CPU should be 70-100% during training")
print("   - RAM usage should be stable")
if gpu_available:
    print("   - GPU should be 70-100% during neural network training")
    print("   - GPU memory usage should increase during model creation")
    print("   - For detailed GPU stats, use: nvidia-smi -l 1")
print()
