"""Quick environment sanity check for cellpose + pytorch + GPU."""

import cellpose
import torch

print(f"cellpose version: {cellpose.version if hasattr(cellpose, 'version') else cellpose.__version__}")
print(f"pytorch version:  {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA available:   {cuda_available}")

if cuda_available:
    print(f"CUDA version:     {torch.version.cuda}")
    print(f"GPU count:        {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Run a tiny op on the GPU to confirm we can actually talk to it
    x = torch.rand(3, 3, device="cuda")
    y = x @ x
    print(f"GPU tensor test OK, result device: {y.device}")
else:
    print("No GPU detected - running on CPU.")
