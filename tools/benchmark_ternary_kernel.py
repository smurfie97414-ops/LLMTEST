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

from cortex3_llm import ResourceUsageMonitor
from cortex3_ternary import (
    BitLinear,
    BitLinearConfig,
    clear_native_grad_kernel_counts,
    last_native_grad_input_kernel,
    last_native_grad_weight_kernel,
    native_grad_input_kernel_counts,
    native_grad_weight_kernel_counts,
    native_ternary_cuda_available,
    native_ternary_cuda_extension_available,
)
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
DEFAULT_MATRIX_SHAPES = ((64, 128, 128), (128, 256, 256), (256, 512, 512))


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
    native_backend: str,
    enable_autotune: bool,
    autotune_warmup: int,
    autotune_repeat: int,
    autotune_cache_path: Path | None,
    autotune_cache_write: bool,
    warmup: int,
    repeat: int,
    sustain_seconds: float,
    sustain_op: str,
    sustain_sync_every: int,
) -> dict[str, Any]:
    clear_native_grad_kernel_counts()
    layer = BitLinear(
        BitLinearConfig(
            in_features,
            out_features,
            activation_bits=0,
            residual_runtime=False,
            require_native_cuda_kernel=True,
            native_cuda_backend=native_backend,
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
            native_cuda_backend=native_backend,
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
    torch_requant_layer = BitLinear(
        BitLinearConfig(
            in_features,
            out_features,
            activation_bits=0,
            residual_runtime=False,
            use_native_cuda_kernel=False,
            native_cuda_autotune=False,
            log_prefix="bench-torch-requantize",
        )
    ).cuda()
    with torch.no_grad():
        legacy_layer.float_weight.copy_(layer.float_weight)
        torch_requant_layer.float_weight.copy_(layer.float_weight)
        if layer.bias is not None and legacy_layer.bias is not None:
            legacy_layer.bias.copy_(layer.bias)
        if layer.bias is not None and torch_requant_layer.bias is not None:
            torch_requant_layer.bias.copy_(layer.bias)
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

    def native_requantize_pack() -> torch.Tensor:
        layer._sync_quantized_buffers_from_weight(record_decision=False)
        return layer.scales

    def torch_requantize_pack() -> torch.Tensor:
        torch_requant_layer._sync_quantized_buffers_from_weight(record_decision=False)
        return torch_requant_layer.scales

    sustained_workloads = {
        "forward": bitlinear_forward,
        "forward_backward": bitlinear_forward_backward,
        "native": native,
        "requantize": native_requantize_pack,
    }
    if sustain_op not in sustained_workloads:
        raise ValueError(f"unsupported sustain_op={sustain_op!r}; choose from {sorted(sustained_workloads)}")

    def sustained_workload() -> dict[str, Any]:
        if sustain_seconds <= 0:
            return {"enabled": False}
        fn = sustained_workloads[sustain_op]
        target_seconds = float(sustain_seconds)
        sync_every = max(1, int(sustain_sync_every))
        start_wall = time()
        iterations = 0
        while True:
            fn()
            iterations += 1
            if iterations % sync_every == 0:
                torch.cuda.synchronize()
                if time() - start_wall >= target_seconds:
                    break
        torch.cuda.synchronize()
        elapsed = time() - start_wall
        return {
            "enabled": True,
            "op": sustain_op,
            "target_seconds": target_seconds,
            "duration_seconds": elapsed,
            "iterations": iterations,
            "sync_every": sync_every,
            "iterations_per_second": iterations / max(elapsed, 1e-9),
        }

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
    native_requantize_pack_ms = _time_cuda(native_requantize_pack, warmup=warmup, repeat=repeat)
    torch_requantize_pack_ms = _time_cuda(torch_requantize_pack, warmup=warmup, repeat=repeat)
    sustained = sustained_workload()
    legacy_training_forward_ms = native_ms + ste_dense_ms
    return {
        "batch": int(batch),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "dtype": str(dtype).replace("torch.", ""),
        "native_backend": f"native_int2_{layer._last_native_cuda_backend}_cuda_{layer._last_native_kernel_variant}",
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
        "native_requantize_pack_ms": native_requantize_pack_ms,
        "torch_requantize_pack_ms": torch_requantize_pack_ms,
        "sustained_workload": sustained,
        "native_grad_weight_backend_counts": dict(layer.ledger.native_ternary_grad_weight_backend_counts),
        "native_grad_input_kernel": last_native_grad_input_kernel(),
        "native_grad_weight_kernel": last_native_grad_weight_kernel(),
        "native_grad_input_kernel_counts": native_grad_input_kernel_counts(),
        "native_grad_weight_kernel_counts": native_grad_weight_kernel_counts(),
        "native_extension_grad_weight_dispatches": int(
            layer.ledger.native_ternary_grad_weight_backend_counts.get("extension", 0)
        ),
        "requantize_pack_speedup_vs_torch": torch_requantize_pack_ms / max(native_requantize_pack_ms, 1e-9),
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


def _parse_shape(raw: str) -> tuple[int, int, int]:
    normalized = raw.lower().replace(",", "x").replace(":", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be BATCHxIN_FEATURESxOUT_FEATURES")
    try:
        batch, in_features, out_features = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if min(batch, in_features, out_features) < 1:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return batch, in_features, out_features


def _resource_metrics(report: dict[str, Any]) -> dict[str, Any]:
    usage = dict(report.get("resource_usage") or {})
    metrics = dict(usage.get("metrics") or {})
    out: dict[str, Any] = {}
    for key in (
        "cpu_total_percent",
        "process_cpu_percent_of_total",
        "gpu_utilization_percent",
        "gpu_memory_utilization_percent",
        "gpu_memory_used_mb",
        "gpu_power_draw_watts",
    ):
        stats = metrics.get(key)
        if isinstance(stats, dict):
            out[key] = {
                "avg": float(stats.get("avg", 0.0) or 0.0),
                "max": float(stats.get("max", 0.0) or 0.0),
            }
    return out


def _case_strict_extension_only(report: dict[str, Any]) -> bool:
    counts = {str(key): int(value) for key, value in dict(report.get("native_grad_weight_backend_counts") or {}).items()}
    return (
        str(report.get("native_backend", "")).startswith("native_int2_extension_cuda_")
        and int(counts.get("extension", 0)) > 0
        and not any(value > 0 and backend != "extension" for backend, value in counts.items())
    )


def _benchmark_case_with_resource_monitor(*, resource_interval: float, min_resource_samples: int, **kwargs: Any) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    monitor = ResourceUsageMonitor(device=device, interval_seconds=max(0.01, float(resource_interval)))
    monitor.start()
    try:
        report = benchmark_case(**kwargs)
    finally:
        resource_usage = monitor.stop()
    report["resource_usage"] = resource_usage
    report["resource_metrics"] = _resource_metrics(report)
    report["resource_sample_count_passed"] = int(resource_usage.get("sample_count", 0) or 0) >= int(min_resource_samples)
    report["strict_extension_only"] = _case_strict_extension_only(report)
    return report


def _matrix_summary(cases: list[dict[str, Any]], *, min_resource_samples: int) -> dict[str, Any]:
    speedups = [float(case.get("full_forward_backward_speedup_vs_legacy_dense_ste", 0.0) or 0.0) for case in cases]
    forward_ms = [float(case.get("full_bitlinear_forward_ms", 0.0) or 0.0) for case in cases]
    forward_backward_ms = [float(case.get("full_bitlinear_forward_backward_ms", 0.0) or 0.0) for case in cases]
    gpu_avg = [
        float(dict(dict(case.get("resource_metrics") or {}).get("gpu_utilization_percent") or {}).get("avg", 0.0) or 0.0)
        for case in cases
    ]
    gpu_power_avg = [
        float(dict(dict(case.get("resource_metrics") or {}).get("gpu_power_draw_watts") or {}).get("avg", 0.0) or 0.0)
        for case in cases
    ]
    cpu_avg = [
        float(dict(dict(case.get("resource_metrics") or {}).get("process_cpu_percent_of_total") or {}).get("avg", 0.0) or 0.0)
        for case in cases
    ]
    sample_counts = [int(dict(case.get("resource_usage") or {}).get("sample_count", 0) or 0) for case in cases]
    strict_extension_only = all(bool(case.get("strict_extension_only")) for case in cases)
    speedup_passed = all(speedup > 1.0 for speedup in speedups)
    resource_samples_passed = all(count >= int(min_resource_samples) for count in sample_counts)
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "strict_extension_only": strict_extension_only,
        "speedup_vs_legacy_dense_ste_passed": speedup_passed,
        "resource_samples_passed": resource_samples_passed,
        "passed": strict_extension_only and speedup_passed and resource_samples_passed,
        "min_resource_samples_required": int(min_resource_samples),
        "min_resource_sample_count": min(sample_counts) if sample_counts else 0,
        "min_full_forward_backward_speedup_vs_legacy_dense_ste": min(speedups) if speedups else 0.0,
        "avg_full_forward_backward_speedup_vs_legacy_dense_ste": sum(speedups) / len(speedups) if speedups else 0.0,
        "avg_full_bitlinear_forward_ms": sum(forward_ms) / len(forward_ms) if forward_ms else 0.0,
        "avg_full_bitlinear_forward_backward_ms": sum(forward_backward_ms) / len(forward_backward_ms) if forward_backward_ms else 0.0,
        "avg_gpu_utilization_percent": sum(gpu_avg) / len(gpu_avg) if gpu_avg else 0.0,
        "avg_gpu_power_draw_watts": sum(gpu_power_avg) / len(gpu_power_avg) if gpu_power_avg else 0.0,
        "avg_process_cpu_percent_of_total": sum(cpu_avg) / len(cpu_avg) if cpu_avg else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Cortex-3 native packed ternary CUDA kernel.")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=512)
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--matrix", action="store_true", help="run a short multi-shape benchmark matrix")
    parser.add_argument("--shape", action="append", type=_parse_shape, default=[], help="matrix case as BATCHxIN_FEATURESxOUT_FEATURES; may be repeated")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--native-backend", choices=("auto", "extension", "rawkernel"), default="extension")
    parser.add_argument("--kernel-variant", choices=("auto", "tiled", "warp"), default="auto")
    parser.add_argument("--disable-autotune", action="store_true")
    parser.add_argument("--autotune-warmup", type=int, default=1)
    parser.add_argument("--autotune-repeat", type=int, default=3)
    parser.add_argument("--autotune-cache", type=Path, default=None)
    parser.add_argument("--clear-autotune-cache", action="store_true")
    parser.add_argument("--no-autotune-cache-write", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--resource-interval", type=float, default=0.05)
    parser.add_argument("--min-resource-samples", type=int, default=1)
    parser.add_argument("--sustain-seconds", type=float, default=0.0)
    parser.add_argument("--sustain-op", choices=("forward", "forward_backward", "native", "requantize"), default="forward_backward")
    parser.add_argument("--sustain-sync-every", type=int, default=8)
    parser.add_argument("--allow-speedup-failure", action="store_true", help="write a matrix report even if a strict extension case is not faster than dense STE")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native ternary kernel benchmarking")
    if args.native_backend == "extension" and not native_ternary_cuda_extension_available():
        raise RuntimeError("Cortex ternary CUDA extension backend is unavailable")
    if args.native_backend != "extension" and not native_ternary_cuda_available():
        raise RuntimeError("CuPy native ternary CUDA kernel is unavailable")
    if args.clear_autotune_cache:
        clear_native_ternary_autotune_cache()
    if args.autotune_cache is not None:
        load_native_ternary_autotune_cache(args.autotune_cache, merge=True)

    started = time()
    shapes = list(args.shape)
    if args.matrix and not shapes:
        shapes = list(DEFAULT_MATRIX_SHAPES)
    common = {
        "dtype": DTYPES[args.dtype],
        "kernel_variant": args.kernel_variant,
        "native_backend": args.native_backend,
        "enable_autotune": not args.disable_autotune,
        "autotune_warmup": args.autotune_warmup,
        "autotune_repeat": args.autotune_repeat,
        "autotune_cache_path": args.autotune_cache,
        "autotune_cache_write": not args.no_autotune_cache_write,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "sustain_seconds": args.sustain_seconds,
        "sustain_op": args.sustain_op,
        "sustain_sync_every": args.sustain_sync_every,
    }
    if shapes:
        cases = [
            _benchmark_case_with_resource_monitor(
                resource_interval=args.resource_interval,
                min_resource_samples=args.min_resource_samples,
                batch=batch,
                in_features=in_features,
                out_features=out_features,
                **common,
            )
            for batch, in_features, out_features in shapes
        ]
        report = {
            "schema_version": 1,
            "mode": "matrix",
            "device": torch.cuda.get_device_name(),
            "dtype": args.dtype,
            "native_backend_requested": args.native_backend,
            "kernel_variant_requested": args.kernel_variant,
            "sustain_seconds": float(args.sustain_seconds),
            "sustain_op": args.sustain_op,
            "min_resource_samples": int(args.min_resource_samples),
            "elapsed_seconds": time() - started,
            "summary": _matrix_summary(cases, min_resource_samples=args.min_resource_samples),
            "cases": cases,
        }
    else:
        report = _benchmark_case_with_resource_monitor(
            resource_interval=args.resource_interval,
            min_resource_samples=args.min_resource_samples,
            batch=args.batch,
            in_features=args.in_features,
            out_features=args.out_features,
            **common,
        )
        report["elapsed_seconds"] = time() - started
        report["device"] = torch.cuda.get_device_name()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if shapes and not report["summary"]["passed"] and not args.allow_speedup_failure:
        raise RuntimeError("strict extension matrix failed speedup or backend checks")
    if args.autotune_cache is not None and not args.no_autotune_cache_write:
        save_native_ternary_autotune_cache(args.autotune_cache)


if __name__ == "__main__":
    main()
