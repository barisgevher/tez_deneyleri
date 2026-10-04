import torch

print("PyTorch version:", torch.__version__)
print("Compiled CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory GB:",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    z = x @ y

    print("Matrix multiplication successful")
    print("Output device:", z.device)
else:
    print("PyTorch CUDA GPU görmüyor.")