import torch
print("torch version:", torch.__version__)
print("cuda compiled:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
