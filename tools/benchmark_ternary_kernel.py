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
from cortex3_ternary import (
    clear_native_ternary_autotune_cache,
    load_native_ternary_autotune_cache,
    native_ternary_autotune_cache_snapshot,
    save_native_ternary_autotune_cache,
)


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
    kernel_variant: str,
    enable_autotune: bool,
    autotune_warmup: int,
    autotune_repeat: int,
    autotune_cache_path: Path | None,
    autotune_cache_write: bool,
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
            native_cuda_kernel_variant=kernel_variant,
            native_cuda_autotune=enable_autotune,
            native_cuda_autotune_warmup=autotune_warmup,
            native_cuda_autotune_repeat=autotune_repeat,
            native_cuda_autotune_cache_path=str(autotune_cache_path) if autotune_cache_path is not None else None,
            native_cuda_autotune_cache_write=autotune_cache_write,
            log_prefix="bench-native-ternary",
        )
    ).cuda()
    legacy_layer = BitLinear(
        BitLinearConfig(
            in_features,
            out_features,
            activation_bits=0,
            residual_runtime=False,
            require_native_cuda_kernel=True,
            native_cuda_kernel_variant=kernel_variant,
            native_cuda_autotune=enable_autotune,
            native_cuda_autotune_warmup=autotune_warmup,
            native_cuda_autotune_repeat=autotune_repeat,
            native_cuda_autotune_cache_path=str(autotune_cache_path) if autotune_cache_path is not None else None,
            native_cuda_autotune_cache_write=autotune_cache_write,
            use_fast_ste_autograd=False,
            log_prefix="bench-legacy-ste",
        )
    ).cuda()
    with torch.no_grad():
        legacy_layer.float_weight.copy_(layer.float_weight)
        if layer.bias is not None and legacy_layer.bias is not None:
            legacy_layer.bias.copy_(layer.bias)
    x = torch.randn(batch, in_features, device="cuda", dtype=dtype)
    x_train = x.detach().clone().requires_grad_(True)
    legacy_x_train = x.detach().clone().requires_grad_(True)
    target = torch.randn(batch, out_features, device="cuda", dtype=dtype)
    layer._sync_quantized_buffers_from_weight(record_decision=False)
    legacy_layer._sync_quantized_buffers_from_weight(record_decision=False)

    def native() -> torch.Tensor:
        return layer._native_cuda_packed_output(x)

    def torch_unpacked() -> torch.Tensor:
        weight = layer._packed_runtime_weight(dtype=x.dtype, device=x.device)
        bias = layer.bias.to(x.dtype) if layer.bias is not None else None
        return F.linear(x, weight, bias)

    def ste_dense() -> torch.Tensor:
        weight = layer._runtime_weight_ste()
        bias = layer.bias
        if x.dtype in {torch.float16, torch.bfloat16}:
            weight = weight.to(x.dtype)
            bias = bias.to(x.dtype) if bias is not None else None
        return F.linear(x, weight, bias)

    def bitlinear_forward() -> torch.Tensor:
        return layer(x)

    def bitlinear_forward_backward() -> torch.Tensor:
        layer.zero_grad(set_to_none=True)
        x_train.grad = None
        loss = (layer(x_train).float() - target.float()).square().mean()
        loss.backward()
        return loss

    def legacy_forward_backward() -> torch.Tensor:
        legacy_layer.zero_grad(set_to_none=True)
        legacy_x_train.grad = None
        loss = (legacy_layer(legacy_x_train).float() - target.float()).square().mean()
        loss.backward()
        return loss

    native_out = native()
    unpacked_out = torch_unpacked()
    ste_out = ste_dense()
    forward_out = bitlinear_forward()
    torch.cuda.synchronize()
    max_abs_error = float((native_out.float() - unpacked_out.float()).abs().max().detach().cpu())
    max_abs_error_vs_ste = float((native_out.float() - ste_out.float()).abs().max().detach().cpu())
    max_abs_error_forward_vs_ste = float((forward_out.float() - ste_out.float()).abs().max().detach().cpu())
    native_ms = _time_cuda(native, warmup=warmup, repeat=repeat)
    unpacked_ms = _time_cuda(torch_unpacked, warmup=warmup, repeat=repeat)
    ste_dense_ms = _time_cuda(ste_dense, warmup=warmup, repeat=repeat)
    full_forward_ms = _time_cuda(bitlinear_forward, warmup=warmup, repeat=repeat)
    full_forward_backward_ms = _time_cuda(bitlinear_forward_backward, warmup=max(1, warmup // 2), repeat=max(1, repeat // 2))
    legacy_forward_backward_ms = _time_cuda(legacy_forward_backward, warmup=max(1, warmup // 2), repeat=max(1, repeat // 2))
    legacy_training_forward_ms = native_ms + ste_dense_ms
    return {
        "batch": int(batch),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "dtype": str(dtype).replace("torch.", ""),
        "native_backend": f"native_int2_cupy_cuda_{layer._last_native_kernel_variant}",
        "kernel_variant": layer._last_native_kernel_variant,
        "kernel_family": layer._last_native_kernel_family,
        "autotuned": bool(layer._last_native_autotuned),
        "autotune_cache_hit": bool(layer._last_native_autotune_cache_hit),
        "autotune_candidate_ms": dict(layer._last_native_autotune_candidate_ms),
        "autotune_cache_entries": int(native_ternary_autotune_cache_snapshot()["entry_count"]),
        "native_ms": native_ms,
        "torch_unpack_linear_ms": unpacked_ms,
        "ste_dense_ms": ste_dense_ms,
        "full_bitlinear_forward_ms": full_forward_ms,
        "full_bitlinear_forward_backward_ms": full_forward_backward_ms,
        "legacy_dense_ste_forward_backward_ms": legacy_forward_backward_ms,
        "legacy_training_forward_native_plus_ste_dense_ms": legacy_training_forward_ms,
        "estimated_training_forward_native_plus_ste_ms": legacy_training_forward_ms,
        "ste_dense_over_native_ratio": ste_dense_ms / max(native_ms, 1e-9),
        "fast_ste_autograd_enabled": bool(layer.config.use_fast_ste_autograd),
        "full_forward_speedup_vs_legacy_native_plus_ste_dense": legacy_training_forward_ms / max(full_forward_ms, 1e-9),
        "full_forward_backward_speedup_vs_legacy_dense_ste": legacy_forward_backward_ms / max(full_forward_backward_ms, 1e-9),
        "full_forward_over_native_ratio": full_forward_ms / max(native_ms, 1e-9),
        "speedup_vs_torch_unpack_linear": unpacked_ms / max(native_ms, 1e-9),
        "speedup_vs_ste_dense": ste_dense_ms / max(native_ms, 1e-9),
        "packed_weight_bytes": int(layer.packed_codes.numel()),
        "unpacked_weight_bytes_at_dtype": int(layer.float_weight.numel() * torch.tensor([], dtype=dtype).element_size()),
        "compression_ratio_vs_unpacked_weight": (
            int(layer.float_weight.numel() * torch.tensor([], dtype=dtype).element_size())
            / max(1, int(layer.packed_codes.numel()))
        ),
        "max_abs_error": max_abs_error,
        "max_abs_error_vs_ste": max_abs_error_vs_ste,
        "max_abs_error_forward_vs_ste": max_abs_error_forward_vs_ste,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Cortex-3 native packed ternary CUDA kernel.")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--kernel-variant", choices=("auto", "tiled", "warp"), default="auto")
    parser.add_argument("--disable-autotune", action="store_true")
    parser.add_argument("--autotune-warmup", type=int, default=1)
    parser.add_argument("--autotune-repeat", type=int, default=3)
    parser.add_argument("--autotune-cache", type=Path, default=None)
    parser.add_argument("--clear-autotune-cache", action="store_true")
    parser.add_argument("--no-autotune-cache-write", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native ternary kernel benchmarking")
    if not native_ternary_cuda_available():
        raise RuntimeError("CuPy native ternary CUDA kernel is unavailable")
    if args.clear_autotune_cache:
        clear_native_ternary_autotune_cache()
    if args.autotune_cache is not None:
        load_native_ternary_autotune_cache(args.autotune_cache, merge=True)

    started = time()
    report = benchmark_case(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=DTYPES[args.dtype],
        kernel_variant=args.kernel_variant,
        enable_autotune=not args.disable_autotune,
        autotune_warmup=args.autotune_warmup,
        autotune_repeat=args.autotune_repeat,
        autotune_cache_path=args.autotune_cache,
        autotune_cache_write=not args.no_autotune_cache_write,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    report["elapsed_seconds"] = time() - started
    report["device"] = torch.cuda.get_device_name()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.autotune_cache is not None and not args.no_autotune_cache_write:
        save_native_ternary_autotune_cache(args.autotune_cache)


if __name__ == "__main__":
    main()
