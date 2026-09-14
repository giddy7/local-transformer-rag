import os
import sys
import platform
import psutil
import torch

def get_disk_space():
    try:
        usage = psutil.disk_usage('.')
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        return f"{free_gb:.1f} GB free of {total_gb:.1f} GB"
    except Exception:
        return "Unknown"

def recommend_preset(has_cuda, vram_gb, sys_ram_gb):
    if not has_cuda:
        if sys_ram_gb < 16:
            return "SMALL"
        else:
            return "SMALL"  # CPU benefit from SMALL model
    else:
        if vram_gb < 4.0:
            return "SMALL"
        elif vram_gb < 8.0:
            return "MEDIUM"
        else:
            return "LARGE"

def check_env():
    os_name = platform.system() + " " + platform.release()
    python_ver = sys.version.split()[0]
    pytorch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        gpu_str = gpu_name
        vram_str = f"{vram_gb:.2f} GB"
    else:
        gpu_name = "None (CPU Mode)"
        vram_gb = 0.0
        gpu_str = "None"
        vram_str = "N/A"
        
    cpu_info = platform.processor() or platform.machine()
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    disk_info = get_disk_space()
    
    preset = recommend_preset(cuda_avail, vram_gb, ram_gb)

    print("=========================================")
    print("LOCAL RAG SYSTEM ENVIRONMENT CHECK")
    print("=========================================")
    print(f"Operating System: {os_name}")
    print(f"Python:           {python_ver}")
    print(f"PyTorch:          {pytorch_ver}")
    print(f"CUDA Available:   {'Yes' if cuda_avail else 'No'}")
    print(f"GPU:              {gpu_str}")
    print(f"GPU Memory:       {vram_str}")
    print(f"CPU:              {cpu_info}")
    print(f"CPU Cores:        {cpu_cores}")
    print(f"RAM:              {ram_gb:.1f} GB")
    print(f"Disk Space:       {disk_info}")
    print("-----------------------------------------")
    print(f"Recommended Configuration Preset: {preset}")
    print("=========================================\n")

    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"Selected PyTorch Device: {device}")
    return preset, device

if __name__ == "__main__":
    check_env()
