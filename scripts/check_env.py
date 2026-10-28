import sys
import torch

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "| built for CUDA", torch.version.cuda)
print("GPU found   :", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU         :", props.name, "| compute capability", torch.cuda.get_device_capability(0))
    print("GPU memory  :", round(props.total_memory / 1e9, 1), "GB")
    print("bfloat16    :", torch.cuda.is_bf16_supported())
    x = torch.randn(2048, 2048, device="cuda")   # create a matrix directly on the GPU
    y = x @ x                                      # run a real GPU computation
    torch.cuda.synchronize()                       # wait for the GPU to finish
    print("GPU maths   : OK", tuple(y.shape))

import transformers, sklearn, pandas
print("transformers:", transformers.__version__, "| sklearn:", sklearn.__version__, "| pandas:", pandas.__version__)
