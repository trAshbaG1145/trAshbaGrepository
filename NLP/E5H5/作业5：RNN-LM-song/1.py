import torch

if torch.cuda.is_available():
    print("太好了！PyTorch 已经配置了 GPU (CUDA) 支持。")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"检测到的 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch 已安装，但目前是 CPU-only 版本。")