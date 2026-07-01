from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline


def main() -> None:
    print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
    print("CUDA_HOME", os.environ.get("CUDA_HOME"))
    cpp_source = r"""
#include <torch/extension.h>

extern "C" void launch_add_one(float* x, long long n);

torch::Tensor add_one(torch::Tensor x) {
  auto y = x.contiguous().clone();
  launch_add_one(y.data_ptr<float>(), y.numel());
  return y;
}
"""
    cuda_source = r"""
#include <cuda_runtime.h>

__global__ void add_one_kernel(float* x, long long n) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += 1.0f;
  }
}

extern "C" void launch_add_one(float* x, long long n) {
  add_one_kernel<<<(n + 255) / 256, 256>>>(x, n);
}
"""
    module = load_inline(
        name="cortex3_cuda128_smoke",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["add_one"],
        extra_cuda_cflags=["-allow-unsupported-compiler"],
        no_implicit_headers=True,
        verbose=False,
    )
    x = torch.arange(4, device="cuda", dtype=torch.float32)
    y = module.add_one(x)
    torch.cuda.synchronize()
    result = [float(value) for value in y.cpu().tolist()]
    print("result", result)
    if result != [1.0, 2.0, 3.0, 4.0]:
        raise RuntimeError(f"unexpected CUDA extension result: {result!r}")


if __name__ == "__main__":
    main()
