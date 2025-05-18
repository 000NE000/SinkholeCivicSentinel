import torch
print(torch.backends.mps.is_available())  # True면 MPS 지원됨
print(torch.backends.mps.is_built())      # True면 빌드에 포함됨