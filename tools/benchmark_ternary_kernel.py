from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from cortex3_ternary import BitLinear, BitLinearConfig, native_ternary_cuda_available


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _time_cuda(fn, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(1, repeat))


def benchmark_case(
    *,
    batch: int,
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    layer = BitLinear(
        BitLinearConfig(
            in_features,
            out_features,
            activation_bits=0,
            residual_runtime=False,
            require_native_cuda_kernel=True,
            log_prefix="bench-native-ternary",
        )
    ).cuda()
    x = torch.randn(batch, in_features, device="cuda", dtype=dtype)
    layer._sync_quantized_buffers_from_weight(record_decision=False)

    def native() -> torch.Tensor:
        return layer._native_cuda_packed_output(x)

    def torch_unpacked() -> torch.Tensor:
        weight = layer._packed_runtime_weight(dtype=x.dtype, device=x.device)
        bias = layer.bias.to(x.dtype) if layer.bias is not None else None
        return F.linear(x, weight, bias)

    native_out = native()
    unpacked_out = torch_unpacked()
    torch.cuda.synchronize()
    max_abs_error = float((native_out.float() - unpacked_out.float()).abs().max().detach().cpu())
    native_ms = _time_cuda(native, warmup=warmup, repeat=repeat)
    unpacked_ms = _time_cuda(torch_unpacked, warmup=warmup, repeat=repeat)
    return {
        "batch": int(batch),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "dtype": str(dtype).replace("torch.", ""),
        "native_backend": "native_int2_cupy_cuda",
        "native_ms": native_ms,
        "torch_unpack_linear_ms": unpacked_ms,
        "speedup_vs_torch_unpack_linear": unpacked_ms / max(native_ms, 1e-9),
        "packed_weight_bytes": int(layer.packed_codes.numel()),
        "unpacked_weight_bytes_at_dtype": int(layer.float_weight.numel() * torch.tensor([], dtype=dtype).element_size()),
        "compression_ratio_vs_unpacked_weight": (
            int(layer.float_weight.numel() * torch.tensor([], dtype=dtype).element_size())
            / max(1, int(layer.packed_codes.numel()))
        ),
        "max_abs_error": max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Cortex-3 native packed ternary CUDA kernel.")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native ternary kernel benchmarking")
    if not native_ternary_cuda_available():
        raise RuntimeError("CuPy native ternary CUDA kernel is unavailable")

    started = time()
    report = benchmark_case(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=DTYPES[args.dtype],
        warmup=args.warmup,
        repeat=args.repeat,
    )
    report["elapsed_seconds"] = time() - started
    report["device"] = torch.cuda.get_device_name()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
