import torch
import flwr
print(flwr.__version__)
print(torch.__version__)          # 确认 PyTorch 版本
print(torch.cuda.is_available())  # 应输出 True
print(torch.version.cuda)         # 显示 CUDA 版本
print(torch.backends.cudnn.version())  # 显示 cuDNN 版本