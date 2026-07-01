from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from math import fsum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cortex3 import CostTrace, TernaryBlock, ZeroState, ternarize_values


import torch
import torch.nn as nn
import torch.nn.functional as F


_CUPY_TERNARY_KERNEL_SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void ternary_matmul_fp32(
    const float* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    float* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m_rows * n_cols;
  if (idx >= total) return;
  int m = idx / n_cols;
  int n = idx - m * n_cols;
  float acc = has_bias ? bias[n] : 0.0f;
  float scale = scales[n];
  const unsigned char* packed_row = packed + n * packed_stride;
  for (int k = 0; k < k_cols; ++k) {
    unsigned char byte = packed_row[k >> 2];
    unsigned int code = (byte >> ((k & 3) << 1)) & 3;
    float sign = code == 1 ? -1.0f : (code == 2 ? 1.0f : 0.0f);
    float w = sign * scale;
    if (use_residual) w += residual[n * k_cols + k];
    acc += x[m * k_cols + k] * w;
  }
  out[idx] = acc;
}

extern "C" __global__ void ternary_matmul_fp16(
    const __half* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __half* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m_rows * n_cols;
  if (idx >= total) return;
  int m = idx / n_cols;
  int n = idx - m * n_cols;
  float acc = has_bias ? bias[n] : 0.0f;
  float scale = scales[n];
  const unsigned char* packed_row = packed + n * packed_stride;
  for (int k = 0; k < k_cols; ++k) {
    unsigned char byte = packed_row[k >> 2];
    unsigned int code = (byte >> ((k & 3) << 1)) & 3;
    float sign = code == 1 ? -1.0f : (code == 2 ? 1.0f : 0.0f);
    float w = sign * scale;
    if (use_residual) w += residual[n * k_cols + k];
    acc += __half2float(x[m * k_cols + k]) * w;
  }
  out[idx] = __float2half(acc);
}

extern "C" __global__ void ternary_matmul_bf16(
    const __nv_bfloat16* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __nv_bfloat16* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = m_rows * n_cols;
  if (idx >= total) return;
  int m = idx / n_cols;
  int n = idx - m * n_cols;
  float acc = has_bias ? bias[n] : 0.0f;
  float scale = scales[n];
  const unsigned char* packed_row = packed + n * packed_stride;
  for (int k = 0; k < k_cols; ++k) {
    unsigned char byte = packed_row[k >> 2];
    unsigned int code = (byte >> ((k & 3) << 1)) & 3;
    float sign = code == 1 ? -1.0f : (code == 2 ? 1.0f : 0.0f);
    float w = sign * scale;
    if (use_residual) w += residual[n * k_cols + k];
    acc += __bfloat162float(x[m * k_cols + k]) * w;
  }
  out[idx] = __float2bfloat16(acc);
}

#define C3_BLOCK_M 16
#define C3_BLOCK_N 16
#define C3_BLOCK_K 32
#define C3_WARPS_PER_BLOCK 8

__device__ __forceinline__ float c3_decode_weight(
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    int n,
    int k,
    int packed_stride,
    int k_cols,
    int use_residual) {
  const unsigned char* packed_row = packed + n * packed_stride;
  unsigned char byte = packed_row[k >> 2];
  unsigned int code = (byte >> ((k & 3) << 1)) & 3;
  float sign = code == 1 ? -1.0f : (code == 2 ? 1.0f : 0.0f);
  float w = sign * scales[n];
  if (use_residual) w += residual[n * k_cols + k];
  return w;
}

__device__ __forceinline__ float c3_warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

extern "C" __global__ void ternary_matmul_tiled_fp32(
    const float* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    float* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  __shared__ float x_tile[C3_BLOCK_M][C3_BLOCK_K];
  __shared__ float w_tile[C3_BLOCK_N][C3_BLOCK_K];
  int local_m = threadIdx.y;
  int local_n = threadIdx.x;
  int tid = local_m * blockDim.x + local_n;
  int global_m = blockIdx.y * C3_BLOCK_M + local_m;
  int global_n = blockIdx.x * C3_BLOCK_N + local_n;
  float acc = (global_n < n_cols && has_bias) ? bias[global_n] : 0.0f;
  for (int k0 = 0; k0 < k_cols; k0 += C3_BLOCK_K) {
    for (int index = tid; index < C3_BLOCK_M * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int lm = index / C3_BLOCK_K;
      int lk = index - lm * C3_BLOCK_K;
      int m = blockIdx.y * C3_BLOCK_M + lm;
      int k = k0 + lk;
      x_tile[lm][lk] = (m < m_rows && k < k_cols) ? x[m * k_cols + k] : 0.0f;
    }
    for (int index = tid; index < C3_BLOCK_N * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int ln = index / C3_BLOCK_K;
      int lk = index - ln * C3_BLOCK_K;
      int n = blockIdx.x * C3_BLOCK_N + ln;
      int k = k0 + lk;
      w_tile[ln][lk] = (n < n_cols && k < k_cols)
          ? c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual)
          : 0.0f;
    }
    __syncthreads();
    if (global_m < m_rows && global_n < n_cols) {
      #pragma unroll
      for (int kk = 0; kk < C3_BLOCK_K; ++kk) {
        if (k0 + kk < k_cols) acc += x_tile[local_m][kk] * w_tile[local_n][kk];
      }
    }
    __syncthreads();
  }
  if (global_m < m_rows && global_n < n_cols) {
    out[global_m * n_cols + global_n] = acc;
  }
}

extern "C" __global__ void ternary_matmul_tiled_fp16(
    const __half* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __half* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  __shared__ float x_tile[C3_BLOCK_M][C3_BLOCK_K];
  __shared__ float w_tile[C3_BLOCK_N][C3_BLOCK_K];
  int local_m = threadIdx.y;
  int local_n = threadIdx.x;
  int tid = local_m * blockDim.x + local_n;
  int global_m = blockIdx.y * C3_BLOCK_M + local_m;
  int global_n = blockIdx.x * C3_BLOCK_N + local_n;
  float acc = (global_n < n_cols && has_bias) ? bias[global_n] : 0.0f;
  for (int k0 = 0; k0 < k_cols; k0 += C3_BLOCK_K) {
    for (int index = tid; index < C3_BLOCK_M * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int lm = index / C3_BLOCK_K;
      int lk = index - lm * C3_BLOCK_K;
      int m = blockIdx.y * C3_BLOCK_M + lm;
      int k = k0 + lk;
      x_tile[lm][lk] = (m < m_rows && k < k_cols) ? __half2float(x[m * k_cols + k]) : 0.0f;
    }
    for (int index = tid; index < C3_BLOCK_N * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int ln = index / C3_BLOCK_K;
      int lk = index - ln * C3_BLOCK_K;
      int n = blockIdx.x * C3_BLOCK_N + ln;
      int k = k0 + lk;
      w_tile[ln][lk] = (n < n_cols && k < k_cols)
          ? c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual)
          : 0.0f;
    }
    __syncthreads();
    if (global_m < m_rows && global_n < n_cols) {
      #pragma unroll
      for (int kk = 0; kk < C3_BLOCK_K; ++kk) {
        if (k0 + kk < k_cols) acc += x_tile[local_m][kk] * w_tile[local_n][kk];
      }
    }
    __syncthreads();
  }
  if (global_m < m_rows && global_n < n_cols) {
    out[global_m * n_cols + global_n] = __float2half(acc);
  }
}

extern "C" __global__ void ternary_matmul_tiled_bf16(
    const __nv_bfloat16* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __nv_bfloat16* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  __shared__ float x_tile[C3_BLOCK_M][C3_BLOCK_K];
  __shared__ float w_tile[C3_BLOCK_N][C3_BLOCK_K];
  int local_m = threadIdx.y;
  int local_n = threadIdx.x;
  int tid = local_m * blockDim.x + local_n;
  int global_m = blockIdx.y * C3_BLOCK_M + local_m;
  int global_n = blockIdx.x * C3_BLOCK_N + local_n;
  float acc = (global_n < n_cols && has_bias) ? bias[global_n] : 0.0f;
  for (int k0 = 0; k0 < k_cols; k0 += C3_BLOCK_K) {
    for (int index = tid; index < C3_BLOCK_M * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int lm = index / C3_BLOCK_K;
      int lk = index - lm * C3_BLOCK_K;
      int m = blockIdx.y * C3_BLOCK_M + lm;
      int k = k0 + lk;
      x_tile[lm][lk] = (m < m_rows && k < k_cols) ? __bfloat162float(x[m * k_cols + k]) : 0.0f;
    }
    for (int index = tid; index < C3_BLOCK_N * C3_BLOCK_K; index += C3_BLOCK_M * C3_BLOCK_N) {
      int ln = index / C3_BLOCK_K;
      int lk = index - ln * C3_BLOCK_K;
      int n = blockIdx.x * C3_BLOCK_N + ln;
      int k = k0 + lk;
      w_tile[ln][lk] = (n < n_cols && k < k_cols)
          ? c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual)
          : 0.0f;
    }
    __syncthreads();
    if (global_m < m_rows && global_n < n_cols) {
      #pragma unroll
      for (int kk = 0; kk < C3_BLOCK_K; ++kk) {
        if (k0 + kk < k_cols) acc += x_tile[local_m][kk] * w_tile[local_n][kk];
      }
    }
    __syncthreads();
  }
  if (global_m < m_rows && global_n < n_cols) {
    out[global_m * n_cols + global_n] = __float2bfloat16(acc);
  }
}

extern "C" __global__ void ternary_matmul_warp_fp32(
    const float* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    float* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * n_cols;
  if (output_index >= total) return;
  int m = output_index / n_cols;
  int n = output_index - m * n_cols;
  float acc = 0.0f;
  for (int k = lane; k < k_cols; k += 32) {
    acc += x[m * k_cols + k] * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    if (has_bias) acc += bias[n];
    out[output_index] = acc;
  }
}

extern "C" __global__ void ternary_matmul_warp_fp16(
    const __half* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __half* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * n_cols;
  if (output_index >= total) return;
  int m = output_index / n_cols;
  int n = output_index - m * n_cols;
  float acc = 0.0f;
  for (int k = lane; k < k_cols; k += 32) {
    acc += __half2float(x[m * k_cols + k]) * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    if (has_bias) acc += bias[n];
    out[output_index] = __float2half(acc);
  }
}

extern "C" __global__ void ternary_matmul_warp_bf16(
    const __nv_bfloat16* x,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    const float* bias,
    __nv_bfloat16* out,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual,
    int has_bias) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * n_cols;
  if (output_index >= total) return;
  int m = output_index / n_cols;
  int n = output_index - m * n_cols;
  float acc = 0.0f;
  for (int k = lane; k < k_cols; k += 32) {
    acc += __bfloat162float(x[m * k_cols + k]) * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    if (has_bias) acc += bias[n];
    out[output_index] = __float2bfloat16(acc);
  }
}

extern "C" __global__ void ternary_grad_input_warp_fp32(
    const float* grad_out,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    float* grad_x,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * k_cols;
  if (output_index >= total) return;
  int m = output_index / k_cols;
  int k = output_index - m * k_cols;
  float acc = 0.0f;
  for (int n = lane; n < n_cols; n += 32) {
    acc += grad_out[m * n_cols + n]
        * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    grad_x[output_index] = acc;
  }
}

extern "C" __global__ void ternary_grad_input_warp_fp16(
    const __half* grad_out,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    __half* grad_x,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * k_cols;
  if (output_index >= total) return;
  int m = output_index / k_cols;
  int k = output_index - m * k_cols;
  float acc = 0.0f;
  for (int n = lane; n < n_cols; n += 32) {
    acc += __half2float(grad_out[m * n_cols + n])
        * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    grad_x[output_index] = __float2half(acc);
  }
}

extern "C" __global__ void ternary_grad_input_warp_bf16(
    const __nv_bfloat16* grad_out,
    const unsigned char* packed,
    const float* scales,
    const float* residual,
    __nv_bfloat16* grad_x,
    int m_rows,
    int n_cols,
    int k_cols,
    int packed_stride,
    int use_residual) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int output_index = blockIdx.x * C3_WARPS_PER_BLOCK + warp;
  int total = m_rows * k_cols;
  if (output_index >= total) return;
  int m = output_index / k_cols;
  int k = output_index - m * k_cols;
  float acc = 0.0f;
  for (int n = lane; n < n_cols; n += 32) {
    acc += __bfloat162float(grad_out[m * n_cols + n])
        * c3_decode_weight(packed, scales, residual, n, k, packed_stride, k_cols, use_residual);
  }
  acc = c3_warp_sum(acc);
  if (lane == 0) {
    grad_x[output_index] = __float2bfloat16(acc);
  }
}
"""

_CUPY_TERNARY_KERNEL_NAMES = {
    "tiled": {
        torch.float32: "ternary_matmul_tiled_fp32",
        torch.float16: "ternary_matmul_tiled_fp16",
        torch.bfloat16: "ternary_matmul_tiled_bf16",
    },
    "warp": {
        torch.float32: "ternary_matmul_warp_fp32",
        torch.float16: "ternary_matmul_warp_fp16",
        torch.bfloat16: "ternary_matmul_warp_bf16",
    },
}
_CUPY_TERNARY_GRAD_INPUT_KERNEL_NAMES = {
    torch.float32: "ternary_grad_input_warp_fp32",
    torch.float16: "ternary_grad_input_warp_fp16",
    torch.bfloat16: "ternary_grad_input_warp_bf16",
}
_CUPY_TERNARY_KERNEL_CACHE: dict[Any, Any] = {}
_NATIVE_TERNARY_AUTOTUNE_CACHE: dict[tuple[Any, ...], tuple[str, tuple[tuple[str, float], ...]]] = {}
_NATIVE_TERNARY_AUTOTUNE_LOADED_PATHS: set[str] = set()
_CUPY_IMPORT_ERROR: Exception | None = None


def _autotune_key_to_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    (
        dtype,
        rows,
        in_features,
        out_features,
        residual_runtime,
        has_bias,
        device_index,
        device_name,
        compute_major,
        compute_minor,
    ) = key
    return {
        "dtype": str(dtype),
        "rows": int(rows),
        "in_features": int(in_features),
        "out_features": int(out_features),
        "residual_runtime": bool(residual_runtime),
        "has_bias": bool(has_bias),
        "device_index": int(device_index),
        "device_name": str(device_name),
        "compute_major": int(compute_major),
        "compute_minor": int(compute_minor),
    }


def _autotune_key_from_dict(data: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(data["dtype"]),
        int(data["rows"]),
        int(data["in_features"]),
        int(data["out_features"]),
        int(bool(data.get("residual_runtime", False))),
        int(bool(data.get("has_bias", True))),
        int(data.get("device_index", 0)),
        str(data["device_name"]),
        int(data["compute_major"]),
        int(data["compute_minor"]),
    )


def clear_native_ternary_autotune_cache() -> int:
    removed = len(_NATIVE_TERNARY_AUTOTUNE_CACHE)
    _NATIVE_TERNARY_AUTOTUNE_CACHE.clear()
    _NATIVE_TERNARY_AUTOTUNE_LOADED_PATHS.clear()
    return removed


def native_ternary_autotune_cache_snapshot() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for key, (selected, candidates) in sorted(_NATIVE_TERNARY_AUTOTUNE_CACHE.items(), key=lambda item: repr(item[0])):
        entry = _autotune_key_to_dict(key)
        entry.update({
            "selected": selected,
            "candidate_ms": [
                {"variant": str(name), "ms": float(ms)}
                for name, ms in candidates
            ],
        })
        entries.append(entry)
    return {
        "schema_version": 1,
        "entry_count": len(entries),
        "entries": entries,
    }


def save_native_ternary_autotune_cache(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(native_ternary_autotune_cache_snapshot(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_native_ternary_autotune_cache(path: str | Path, *, merge: bool = True) -> int:
    source = Path(path)
    if not source.exists():
        return 0
    payload = json.loads(source.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported native ternary autotune cache schema: {payload.get('schema_version')!r}")
    if not merge:
        _NATIVE_TERNARY_AUTOTUNE_CACHE.clear()
    loaded = 0
    for raw_entry in payload.get("entries", ()):
        entry = dict(raw_entry)
        selected = str(entry.get("selected", ""))
        if selected not in {"tiled", "warp"}:
            raise ValueError(f"invalid autotune selected variant: {selected!r}")
        candidates = tuple(
            (str(item["variant"]), float(item["ms"]))
            for item in entry.get("candidate_ms", ())
            if str(item.get("variant", "")) in {"tiled", "warp"}
        )
        if not candidates:
            raise ValueError("autotune cache entry has no candidate_ms values")
        _NATIVE_TERNARY_AUTOTUNE_CACHE[_autotune_key_from_dict(entry)] = (selected, candidates)
        loaded += 1
    return loaded


def _load_cupy() -> Any:
    global _CUPY_IMPORT_ERROR
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on optional CUDA runtime
        _CUPY_IMPORT_ERROR = exc
        raise RuntimeError("cupy-cuda12x is required for native ternary CUDA kernels") from exc
    _CUPY_IMPORT_ERROR = None
    return cp


def _cupy_ternary_kernel(dtype: Any, variant: str = "tiled") -> tuple[Any, Any]:
    if variant not in _CUPY_TERNARY_KERNEL_NAMES:
        raise RuntimeError(f"native ternary CUDA kernel variant {variant!r} is not supported")
    names = _CUPY_TERNARY_KERNEL_NAMES[variant]
    if dtype not in names:
        raise RuntimeError(f"native ternary CUDA kernel does not support dtype {dtype}")
    cp = _load_cupy()
    key = (variant, dtype)
    if key not in _CUPY_TERNARY_KERNEL_CACHE:
        _CUPY_TERNARY_KERNEL_CACHE[key] = cp.RawKernel(
            _CUPY_TERNARY_KERNEL_SOURCE,
            names[dtype],
            options=("--std=c++17",),
        )
    return cp, _CUPY_TERNARY_KERNEL_CACHE[key]


def _cupy_ternary_grad_input_kernel(dtype: Any) -> tuple[Any, Any]:
    if dtype not in _CUPY_TERNARY_GRAD_INPUT_KERNEL_NAMES:
        raise RuntimeError(f"native ternary CUDA grad-input kernel does not support dtype {dtype}")
    cp = _load_cupy()
    key = ("grad_input_warp", dtype)
    if key not in _CUPY_TERNARY_KERNEL_CACHE:
        _CUPY_TERNARY_KERNEL_CACHE[key] = cp.RawKernel(
            _CUPY_TERNARY_KERNEL_SOURCE,
            _CUPY_TERNARY_GRAD_INPUT_KERNEL_NAMES[dtype],
            options=("--std=c++17",),
        )
    return cp, _CUPY_TERNARY_KERNEL_CACHE[key]


def native_ternary_cuda_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _cupy_ternary_kernel(torch.float32, "tiled")
        _cupy_ternary_kernel(torch.float32, "warp")
        _cupy_ternary_grad_input_kernel(torch.float32)
    except Exception:
        return False
    return True


def _native_packed_ternary_grad_input_cuda(
    grad_output_flat: Any,
    packed_codes: Any,
    scales: Any,
    residual_weight: Any,
    in_features: int,
    *,
    residual_runtime: bool,
) -> Any:
    if not grad_output_flat.is_cuda:
        raise RuntimeError("native ternary CUDA grad-input kernel requires CUDA grad_output")
    if grad_output_flat.dtype not in _CUPY_TERNARY_GRAD_INPUT_KERNEL_NAMES:
        raise RuntimeError(f"native ternary CUDA grad-input kernel does not support dtype {grad_output_flat.dtype}")
    cp, kernel = _cupy_ternary_grad_input_kernel(grad_output_flat.dtype)
    grad_out = grad_output_flat.detach().contiguous()
    m_rows = int(grad_out.shape[0])
    n_cols = int(grad_out.shape[1])
    k_cols = int(in_features)
    packed = packed_codes.to(device=grad_out.device, dtype=torch.uint8).contiguous()
    scale_values = scales.detach().to(device=grad_out.device, dtype=torch.float32).contiguous().view(-1)
    residual = (
        residual_weight.detach().to(device=grad_out.device, dtype=torch.float32).contiguous()
        if residual_runtime
        else torch.empty((0,), device=grad_out.device, dtype=torch.float32)
    )
    grad_x = torch.empty((m_rows, k_cols), device=grad_out.device, dtype=grad_out.dtype)
    warps_per_block = 8
    total_outputs = m_rows * k_cols
    blocks = ((total_outputs + warps_per_block - 1) // warps_per_block,)
    threads = (warps_per_block * 32,)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, message="ExternalStream is deprecated.*")
        stream = cp.cuda.ExternalStream(torch.cuda.current_stream(grad_out.device).cuda_stream)
    with stream:
        kernel(
            blocks,
            threads,
            (
                cp.from_dlpack(grad_out),
                cp.from_dlpack(packed),
                cp.from_dlpack(scale_values),
                cp.from_dlpack(residual),
                cp.from_dlpack(grad_x),
                m_rows,
                n_cols,
                k_cols,
                int(packed.shape[1]),
                int(bool(residual_runtime)),
            ),
        )
    return grad_x


def _unpack_packed_ternary_weight(
    packed: Any,
    scales: Any,
    residual: Any,
    in_features: int,
    *,
    dtype: Any,
    device: Any,
    residual_runtime: bool,
) -> Any:
    packed = packed.to(device=device, dtype=torch.uint8)
    words = packed.to(torch.int64)
    codes = torch.stack(
        (
            words & 0x03,
            (words >> 2) & 0x03,
            (words >> 4) & 0x03,
            (words >> 6) & 0x03,
        ),
        dim=-1,
    ).reshape(packed.shape[0], -1)[:, :in_features]
    zeros = torch.zeros_like(codes, dtype=dtype, device=device)
    neg = -torch.ones_like(codes, dtype=dtype, device=device)
    pos = torch.ones_like(codes, dtype=dtype, device=device)
    weight = torch.where(codes == 1, neg, torch.where(codes == 2, pos, zeros)) * scales.to(device=device, dtype=dtype)
    if residual_runtime:
        weight = weight + residual.to(device=device, dtype=dtype)
    return weight


class _PackedTernarySTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Any,
        float_weight: Any,
        bias: Any,
        packed_output: Any,
        packed_codes: Any,
        scales: Any,
        residual_weight: Any,
        has_bias: bool,
        in_features: int,
        residual_runtime: bool,
    ) -> Any:
        ctx.input_shape = tuple(int(dim) for dim in x.shape)
        ctx.has_bias = bool(has_bias)
        ctx.in_features = int(in_features)
        ctx.residual_runtime = bool(residual_runtime)
        ctx.float_weight_dtype = float_weight.dtype
        ctx.save_for_backward(x, packed_codes.detach(), scales.detach(), residual_weight.detach())
        return packed_output.detach()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        x, packed_codes, scales, residual_weight = ctx.saved_tensors
        grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_flat = x.reshape(-1, x.shape[-1])
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output_flat.is_cuda and grad_output_flat.dtype in _CUPY_TERNARY_GRAD_INPUT_KERNEL_NAMES:
                grad_x = _native_packed_ternary_grad_input_cuda(
                    grad_output_flat,
                    packed_codes,
                    scales,
                    residual_weight,
                    ctx.in_features,
                    residual_runtime=ctx.residual_runtime,
                ).reshape(ctx.input_shape)
            else:
                input_weight = _unpack_packed_ternary_weight(
                    packed_codes,
                    scales,
                    residual_weight,
                    ctx.in_features,
                    dtype=grad_output_flat.dtype,
                    device=grad_output_flat.device,
                    residual_runtime=ctx.residual_runtime,
                )
                grad_x = grad_output_flat.matmul(input_weight).reshape(ctx.input_shape)
        if ctx.needs_input_grad[1]:
            weight_grad = grad_output_flat.transpose(0, 1).matmul(x_flat.to(dtype=grad_output_flat.dtype))
            grad_weight = weight_grad.to(dtype=ctx.float_weight_dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_flat.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None, None


@dataclass(frozen=True)
class ActivationQuantization:
    original: tuple[float, ...]
    quantized: tuple[int, ...]
    dequantized: tuple[float, ...]
    bits: int
    scale: float
    saturated: int = 0
    total_values: int | None = None

    @property
    def activation_bits(self) -> float:
        return (self.total_values if self.total_values is not None else len(self.quantized)) * self.bits


@dataclass(frozen=True)
class CompressionDecision:
    block_id: str
    source: str
    original_count: int
    active_count: int
    provisional_zero_count: int
    certified_zero_count: int
    scale: float
    threshold: float
    estimated_bits: float
    residual_l1: float
    note: str = ""

    @property
    def zero_count(self) -> int:
        return self.provisional_zero_count + self.certified_zero_count


@dataclass(frozen=True)
class ExpertActivation:
    expert_id: str
    reason: str
    cost: float = 1.0


@dataclass(frozen=True)
class KVModeEvent:
    segment_id: str
    mode: str
    bytes_used: float
    exact_anchors: int = 0
    note: str = ""


@dataclass(frozen=True)
class MTPFSPEvent:
    block_id: str
    horizon: int
    accepted: bool
    confidence: float
    contract_revision: int = 0
    reason: str = ""


@dataclass(frozen=True)
class LayerForwardEvent:
    layer_id: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    active_weights: int
    total_weights: int
    estimated_weight_bits: float
    activation_bits: float
    note: str = ""


@dataclass(frozen=True)
class PackedTernaryDispatch:
    layer_id: str
    backend: str
    device: str
    packed_weight_bytes: int
    active_weights: int
    total_weights: int
    max_abs_error_vs_ste: float
    used_residual: bool
    note: str = ""
    native_kernel: bool = False
    kernel_variant: str = ""
    autotuned: bool = False
    autotune_cache_hit: bool = False
    autotune_candidate_ms: tuple[tuple[str, float], ...] = ()


@dataclass
class CompressionTraceLedger:
    compression_decisions: list[CompressionDecision] = field(default_factory=list)
    activation_quantizations: list[ActivationQuantization] = field(default_factory=list)
    expert_activations: list[ExpertActivation] = field(default_factory=list)
    kv_events: list[KVModeEvent] = field(default_factory=list)
    mtp_fsp_events: list[MTPFSPEvent] = field(default_factory=list)
    layer_forward_events: list[LayerForwardEvent] = field(default_factory=list)
    packed_ternary_dispatches: list[PackedTernaryDispatch] = field(default_factory=list)
    retention_limit: int | None = None
    total_compression_decisions: int = 0
    total_activation_quantizations: int = 0
    total_expert_activations: int = 0
    total_kv_events: int = 0
    total_mtp_fsp_events: int = 0
    total_layer_forward_events: int = 0
    total_packed_ternary_dispatches: int = 0
    total_native_ternary_kernel_dispatches: int = 0
    total_torch_packed_ternary_dispatches: int = 0
    total_native_ternary_autotuned_dispatches: int = 0
    total_native_ternary_autotune_cache_hits: int = 0
    total_weight_bits_read: float = 0.0
    total_activation_bits: float = 0.0
    total_kv_bytes: float = 0.0
    total_packed_weight_bytes: float = 0.0

    def _trim(self, events: list[Any]) -> None:
        if self.retention_limit is None:
            return
        limit = max(0, int(self.retention_limit))
        if len(events) > limit:
            del events[: len(events) - limit]

    def record_compression(self, decision: CompressionDecision) -> None:
        self.total_compression_decisions += 1
        self.total_weight_bits_read += float(decision.estimated_bits)
        self.compression_decisions.append(decision)
        self._trim(self.compression_decisions)

    def record_activation(self, quantization: ActivationQuantization) -> None:
        self.total_activation_quantizations += 1
        self.total_activation_bits += float(quantization.activation_bits)
        self.activation_quantizations.append(quantization)
        self._trim(self.activation_quantizations)

    def record_expert(self, expert_id: str, reason: str, cost: float = 1.0) -> None:
        self.total_expert_activations += 1
        self.expert_activations.append(ExpertActivation(expert_id, reason, cost))
        self._trim(self.expert_activations)

    def record_kv(self, segment_id: str, mode: str, bytes_used: float, exact_anchors: int = 0, note: str = "") -> None:
        self.total_kv_events += 1
        self.total_kv_bytes += float(bytes_used)
        self.kv_events.append(KVModeEvent(segment_id, mode, bytes_used, exact_anchors, note))
        self._trim(self.kv_events)

    def record_mtp_fsp(self, block_id: str, horizon: int, accepted: bool, confidence: float, contract_revision: int = 0, reason: str = "") -> None:
        self.total_mtp_fsp_events += 1
        self.mtp_fsp_events.append(MTPFSPEvent(block_id, horizon, accepted, confidence, contract_revision, reason))
        self._trim(self.mtp_fsp_events)

    def record_layer_forward(self, event: LayerForwardEvent) -> None:
        self.total_layer_forward_events += 1
        self.layer_forward_events.append(event)
        self._trim(self.layer_forward_events)

    def record_packed_ternary_dispatch(self, event: PackedTernaryDispatch) -> None:
        self.total_packed_ternary_dispatches += 1
        if event.native_kernel or event.backend.startswith("native_"):
            self.total_native_ternary_kernel_dispatches += 1
            if event.autotuned:
                self.total_native_ternary_autotuned_dispatches += 1
            if event.autotune_cache_hit:
                self.total_native_ternary_autotune_cache_hits += 1
        else:
            self.total_torch_packed_ternary_dispatches += 1
        self.total_packed_weight_bytes += float(event.packed_weight_bytes)
        self.packed_ternary_dispatches.append(event)
        self._trim(self.packed_ternary_dispatches)

    @property
    def cost_trace(self) -> CostTrace:
        return CostTrace(
            weight_bits_read=self.total_weight_bits_read,
            activation_bits=self.total_activation_bits,
            kv_bytes=self.total_kv_bytes,
            experts_activated=self.total_expert_activations,
        )

    def explain_failure(self, reason: str = "") -> list[str]:
        hints: list[str] = []
        if self.kv_events and any(event.mode != "exact" and event.exact_anchors == 0 for event in self.kv_events):
            hints.append("kv_mode_may_have_lost_exact_anchors")
        if self.mtp_fsp_events and any(event.horizon > 1 and event.accepted for event in self.mtp_fsp_events):
            hints.append("accepted_mtp_horizon_may_have_overshot")
        if self.activation_quantizations and any(item.bits <= 4 and item.saturated for item in self.activation_quantizations):
            hints.append("activation_quantization_saturated")
        if self.compression_decisions:
            most_zeroed = max(self.compression_decisions, key=lambda decision: decision.zero_count / max(decision.original_count, 1))
            zero_rate = most_zeroed.zero_count / max(most_zeroed.original_count, 1)
            if zero_rate > 0.5:
                hints.append(f"block_{most_zeroed.block_id}_zero_rate_{zero_rate:.2f}")
        if self.expert_activations and "expert" in reason.lower():
            hints.append("expert_routing_involved")
        return hints or ["no_compression_culprit_logged"]

    def to_dict(self) -> dict[str, Any]:
        native_variants = tuple(sorted({
            item.kernel_variant
            for item in self.packed_ternary_dispatches
            if (item.native_kernel or item.backend.startswith("native_")) and item.kernel_variant
        }))
        return {
            "compression_decisions": [asdict(item) for item in self.compression_decisions],
            "activation_quantizations": [asdict(item) for item in self.activation_quantizations],
            "expert_activations": [asdict(item) for item in self.expert_activations],
            "kv_events": [asdict(item) for item in self.kv_events],
            "mtp_fsp_events": [asdict(item) for item in self.mtp_fsp_events],
            "layer_forward_events": [asdict(item) for item in self.layer_forward_events],
            "packed_ternary_dispatches": [asdict(item) for item in self.packed_ternary_dispatches],
            "retention_limit": self.retention_limit,
            "retained_event_counts": {
                "compression_decisions": len(self.compression_decisions),
                "activation_quantizations": len(self.activation_quantizations),
                "expert_activations": len(self.expert_activations),
                "kv_events": len(self.kv_events),
                "mtp_fsp_events": len(self.mtp_fsp_events),
                "layer_forward_events": len(self.layer_forward_events),
                "packed_ternary_dispatches": len(self.packed_ternary_dispatches),
            },
            "total_event_counts": {
                "compression_decisions": self.total_compression_decisions,
                "activation_quantizations": self.total_activation_quantizations,
                "expert_activations": self.total_expert_activations,
                "kv_events": self.total_kv_events,
                "mtp_fsp_events": self.total_mtp_fsp_events,
                "layer_forward_events": self.total_layer_forward_events,
                "packed_ternary_dispatches": self.total_packed_ternary_dispatches,
                "native_ternary_kernel_dispatches": self.total_native_ternary_kernel_dispatches,
                "torch_packed_ternary_dispatches": self.total_torch_packed_ternary_dispatches,
                "native_ternary_autotuned_dispatches": self.total_native_ternary_autotuned_dispatches,
                "native_ternary_autotune_cache_hits": self.total_native_ternary_autotune_cache_hits,
            },
            "cost_trace": asdict(self.cost_trace),
            "packed_weight_bytes_read": self.total_packed_weight_bytes,
            "native_ternary_kernel_variants": native_variants,
        }


def quantize_activation_values(values: Iterable[float], bits: int = 4) -> ActivationQuantization:
    vals = tuple(float(value) for value in values)
    if bits < 2:
        raise ValueError("activation quantization requires at least 2 bits")
    if not vals:
        return ActivationQuantization((), (), (), bits, 1.0, 0)
    qmax = (2 ** (bits - 1)) - 1
    max_abs = max(abs(value) for value in vals)
    if max_abs == 0:
        return ActivationQuantization(vals, tuple(0 for _ in vals), tuple(0.0 for _ in vals), bits, 1.0, 0)
    scale = max_abs / qmax
    quantized: list[int] = []
    saturated = 0
    for value in vals:
        q = int(round(value / scale))
        if q > qmax:
            q = qmax
            saturated += 1
        elif q < -qmax:
            q = -qmax
            saturated += 1
        quantized.append(q)
    dequantized = tuple(q * scale for q in quantized)
    return ActivationQuantization(vals, tuple(quantized), dequantized, bits, scale, saturated)


@dataclass
class ResidualSynapseBuffer:
    residual_threshold: float = 0.0
    blocks: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def store(self, block_id: str, original: Sequence[float], ternary_block: TernaryBlock) -> tuple[float, ...]:
        dequantized = ternary_block.dequantize()
        residuals = []
        for value, quantized in zip(original, dequantized):
            residual = float(value) - quantized
            residuals.append(residual if abs(residual) > self.residual_threshold else 0.0)
        stored = tuple(residuals)
        self.blocks[block_id] = stored
        return stored

    def restore(self, block_id: str, ternary_block: TernaryBlock) -> tuple[float, ...]:
        residuals = self.blocks.get(block_id, tuple(0.0 for _ in ternary_block.q))
        return tuple(value + residual for value, residual in zip(ternary_block.dequantize(), residuals))

    def l1(self, block_id: str) -> float:
        return fsum(abs(value) for value in self.blocks.get(block_id, ()))


def make_compression_decision(
    block_id: str,
    values: Sequence[float],
    *,
    source: str = "weights",
    threshold: float | None = None,
    residual_buffer: ResidualSynapseBuffer | None = None,
    certify_zeros: bool = False,
    note: str = "",
) -> tuple[TernaryBlock, CompressionDecision]:
    block = ternarize_values(values, threshold=threshold)
    if certify_zeros:
        block = block.certify_zeros()
    residual_l1 = 0.0
    if residual_buffer is not None:
        residual_buffer.store(block_id, tuple(float(value) for value in values), block)
        residual_l1 = residual_buffer.l1(block_id)
    provisional = sum(1 for state in block.zero_states if state == ZeroState.ZERO_PROVISIONAL)
    certified = sum(1 for state in block.zero_states if state == ZeroState.ZERO_CERTIFIED)
    effective_threshold = threshold if threshold is not None else 0.5 * block.scale
    decision = CompressionDecision(
        block_id=block_id,
        source=source,
        original_count=len(block.q),
        active_count=sum(block.mask),
        provisional_zero_count=provisional,
        certified_zero_count=certified,
        scale=block.scale,
        threshold=effective_threshold,
        estimated_bits=block.estimated_bits(),
        residual_l1=residual_l1,
        note=note,
    )
    return block, decision


def torch_available() -> bool:
    return True


@dataclass(frozen=True)
class BitLinearConfig:
    in_features: int
    out_features: int
    bias: bool = True
    activation_bits: int = 4
    threshold: float | None = None
    residual_threshold: float = 0.0
    residual_runtime: bool = False
    shared_scale: bool = True
    log_prefix: str = "bitlinear"
    use_packed_ternary_runtime: bool = True
    use_native_cuda_kernel: bool = True
    require_native_cuda_kernel: bool = False
    native_cuda_kernel_variant: str = "auto"
    native_cuda_autotune: bool = True
    native_cuda_autotune_warmup: int = 1
    native_cuda_autotune_repeat: int = 3
    native_cuda_autotune_cache_path: str | None = None
    native_cuda_autotune_cache_write: bool = True
    use_fast_ste_autograd: bool = True


class BitLinear(nn.Module):
    def __init__(self, config: BitLinearConfig, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        self.config = config
        self.ledger = ledger or CompressionTraceLedger()
        self.float_weight = nn.Parameter(torch.empty(config.out_features, config.in_features))
        self.bias = nn.Parameter(torch.zeros(config.out_features)) if config.bias else None
        self.register_buffer("signs", torch.ones(config.out_features, config.in_features))
        self.register_buffer("mask", torch.ones(config.out_features, config.in_features))
        self.register_buffer("scales", torch.ones(config.out_features, 1))
        self.register_buffer("residual_weight", torch.zeros(config.out_features, config.in_features))
        self.register_buffer(
            "packed_codes",
            torch.zeros(config.out_features, (config.in_features + 3) // 4, dtype=torch.uint8),
        )
        self._last_active_weights = int(config.out_features * config.in_features)
        self._last_total_weights = int(config.out_features * config.in_features)
        self._last_estimated_bits = float(config.out_features * config.in_features)
        self._last_native_kernel_variant = ""
        self._last_native_kernel_family = ""
        self._last_native_autotuned = False
        self._last_native_autotune_cache_hit = False
        self._last_native_autotune_candidate_ms: tuple[tuple[str, float], ...] = ()
        self._native_cuda_instance_variant_cache: dict[tuple[Any, ...], tuple[str, tuple[tuple[str, float], ...]]] = {}
        self._packed_weight_version = -1
        nn.init.kaiming_uniform_(self.float_weight, a=5 ** 0.5)
        self.requantize()

    @classmethod
    def from_linear(cls, linear: Any, *, activation_bits: int = 4, threshold: float | None = None, residual_threshold: float = 0.0, ledger: CompressionTraceLedger | None = None) -> "BitLinear":
        config = BitLinearConfig(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            activation_bits,
            threshold,
            residual_threshold,
            residual_runtime=True,
        )
        module = cls(config, ledger=ledger)
        with torch.no_grad():
            module.float_weight.copy_(linear.weight)
            if linear.bias is not None and module.bias is not None:
                module.bias.copy_(linear.bias)
        module.requantize()
        return module

    @staticmethod
    def _codes_from_sign_mask(signs: Any, mask: Any) -> Any:
        positive = torch.full_like(mask, 2, dtype=torch.uint8)
        negative = torch.ones_like(mask, dtype=torch.uint8)
        zeros = torch.zeros_like(mask, dtype=torch.uint8)
        return torch.where(mask > 0, torch.where(signs > 0, positive, negative), zeros)

    @staticmethod
    def _pack_codes(codes: Any) -> Any:
        out_features, in_features = codes.shape
        pad = (-in_features) % 4
        if pad:
            codes = F.pad(codes, (0, pad), value=0)
        grouped = codes.view(out_features, -1, 4).to(torch.int64)
        packed = (
            grouped[:, :, 0]
            | (grouped[:, :, 1] << 2)
            | (grouped[:, :, 2] << 4)
            | (grouped[:, :, 3] << 6)
        )
        return packed.to(torch.uint8)

    @staticmethod
    def _unpack_codes(packed: Any, in_features: int, *, dtype: Any, device: Any) -> Any:
        scales = torch.ones((packed.shape[0], 1), dtype=dtype, device=device)
        residual = torch.empty((0,), dtype=dtype, device=device)
        return _unpack_packed_ternary_weight(
            packed,
            scales,
            residual,
            in_features,
            dtype=dtype,
            device=device,
            residual_runtime=False,
        )

    def _current_weight_version(self) -> int:
        return int(getattr(self.float_weight, "_version", -1))

    def _sync_quantized_buffers_from_weight(self, *, certify_zeros: bool = False, record_decision: bool = False) -> None:
        with torch.no_grad():
            values = self.float_weight.detach()
            scale = values.abs().mean(dim=1, keepdim=True).clamp_min(1e-12) if self.config.shared_scale else values.abs().mean().clamp_min(1e-12)
            threshold = self.config.threshold if self.config.threshold is not None else 0.5 * scale
            signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
            mask = (values.abs() >= threshold).to(values.dtype)
            quantized = signs * mask * scale
            residual = values - quantized
            if self.config.residual_threshold > 0:
                residual = torch.where(residual.abs() > self.config.residual_threshold, residual, torch.zeros_like(residual))
            self.signs.copy_(signs)
            self.mask.copy_(mask)
            self.scales.copy_(scale if self.config.shared_scale else torch.ones_like(self.scales) * scale)
            self.residual_weight.copy_(residual)
            self.packed_codes.copy_(self._pack_codes(self._codes_from_sign_mask(signs, mask)).to(self.packed_codes.device))

            active_count = int(mask.sum().item())
            total_count = int(mask.numel())
            zero_count = total_count - active_count
            scale_count = values.shape[0] if self.config.shared_scale else 1
            estimated_bits = float(
                total_count
                + active_count
                + scale_count * 16
                + ((self.bias.numel() * 16) if self.bias is not None else 0)
            )
            self._last_active_weights = active_count
            self._last_total_weights = total_count
            self._last_estimated_bits = estimated_bits
            self._packed_weight_version = self._current_weight_version()
            if not record_decision:
                return
            threshold_value = threshold.detach().mean() if isinstance(threshold, torch.Tensor) else torch.as_tensor(threshold)
            decision = CompressionDecision(
                block_id=self.config.log_prefix,
                source="weights",
                original_count=total_count,
                active_count=active_count,
                provisional_zero_count=0 if certify_zeros else zero_count,
                certified_zero_count=zero_count if certify_zeros else 0,
                scale=float(scale.detach().mean().item()),
                threshold=float(threshold_value.item()),
                estimated_bits=estimated_bits,
                residual_l1=float(residual.detach().abs().sum().item()),
                note="packed int2 ternary requantize tensor-stats",
            )
            self.ledger.record_compression(decision)

    def _ensure_quantized_buffers_current(self) -> None:
        current_version = self._current_weight_version()
        if current_version < 0 or current_version != self._packed_weight_version:
            self._sync_quantized_buffers_from_weight(record_decision=False)

    def requantize(self, *, certify_zeros: bool = False) -> None:
        self._sync_quantized_buffers_from_weight(certify_zeros=certify_zeros, record_decision=True)

    def _runtime_weight_ste(self) -> Any:
        values = self.float_weight
        scale = values.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-12) if self.config.shared_scale else values.detach().abs().mean().clamp_min(1e-12)
        threshold = self.config.threshold if self.config.threshold is not None else 0.5 * scale
        signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
        mask = (values.detach().abs() >= threshold).to(values.dtype)
        quantized = signs * mask * scale
        residual = values - quantized
        if self.config.residual_runtime:
            if self.config.residual_threshold > 0:
                residual = torch.where(residual.detach().abs() > self.config.residual_threshold, residual, torch.zeros_like(residual))
            runtime = quantized + residual
        else:
            runtime = quantized
        weight = values + (runtime - values).detach()
        return weight

    def _packed_runtime_weight(self, *, dtype: Any, device: Any) -> Any:
        codes = self._unpack_codes(self.packed_codes, self.config.in_features, dtype=dtype, device=device)
        scales = self.scales.to(device=device, dtype=dtype)
        weight = codes * scales
        if self.config.residual_runtime:
            weight = weight + self.residual_weight.to(device=device, dtype=dtype)
        return weight

    @staticmethod
    def _native_cuda_variant_label(variant: str) -> str:
        return f"{variant}_shared_memory_int2" if variant == "tiled" else "warp_reduction_int2"

    def _heuristic_native_cuda_variant(self, x: Any) -> str:
        flattened_rows = int(x.numel() // self.config.in_features)
        return "warp" if self.config.in_features >= 384 and flattened_rows * self.config.out_features >= 16384 else "tiled"

    def _native_cuda_autotune_key(self, x: Any) -> tuple[Any, ...]:
        device_index = int(x.get_device())
        props = torch.cuda.get_device_properties(device_index)
        rows = int(x.numel() // self.config.in_features)
        return (
            str(x.dtype),
            int(rows),
            int(self.config.in_features),
            int(self.config.out_features),
            int(bool(self.config.residual_runtime)),
            int(self.bias is not None),
            int(device_index),
            str(props.name),
            int(props.major),
            int(props.minor),
        )

    def _native_cuda_instance_autotune_key(self, x: Any) -> tuple[Any, ...]:
        rows = int(x.numel() // self.config.in_features)
        return (
            str(x.dtype),
            int(rows),
            int(self.config.in_features),
            int(self.config.out_features),
            int(bool(self.config.residual_runtime)),
            int(self.bias is not None),
            int(x.get_device()),
        )

    def _native_cuda_autotune_cache_path(self) -> Path | None:
        if not self.config.native_cuda_autotune_cache_path:
            return None
        return Path(self.config.native_cuda_autotune_cache_path).expanduser()

    def _ensure_native_cuda_autotune_profile_loaded(self) -> None:
        path = self._native_cuda_autotune_cache_path()
        if path is None:
            return
        resolved = str(path.resolve(strict=False))
        if resolved in _NATIVE_TERNARY_AUTOTUNE_LOADED_PATHS:
            return
        load_native_ternary_autotune_cache(path, merge=True)
        _NATIVE_TERNARY_AUTOTUNE_LOADED_PATHS.add(resolved)

    def _persist_native_cuda_autotune_profile(self) -> None:
        path = self._native_cuda_autotune_cache_path()
        if path is None or not self.config.native_cuda_autotune_cache_write:
            return
        save_native_ternary_autotune_cache(path)

    def _launch_native_cuda_kernel(self, x: Any, variant: str) -> Any:
        cp, kernel = _cupy_ternary_kernel(x.dtype, variant)
        x_flat = x.detach().contiguous().view(-1, self.config.in_features)
        output = torch.empty(
            (*x.shape[:-1], self.config.out_features),
            device=x.device,
            dtype=x.dtype,
        )
        output_flat = output.view(-1, self.config.out_features)
        packed = self.packed_codes.contiguous()
        scales = self.scales.detach().to(device=x.device, dtype=torch.float32).contiguous().view(-1)
        residual = (
            self.residual_weight.detach().to(device=x.device, dtype=torch.float32).contiguous()
            if self.config.residual_runtime
            else torch.empty((0,), device=x.device, dtype=torch.float32)
        )
        bias = (
            self.bias.detach().to(device=x.device, dtype=torch.float32).contiguous()
            if self.bias is not None
            else torch.empty((0,), device=x.device, dtype=torch.float32)
        )
        if variant == "warp":
            warps_per_block = 8
            total_outputs = int(x_flat.shape[0]) * int(self.config.out_features)
            blocks = ((total_outputs + warps_per_block - 1) // warps_per_block,)
            threads = (warps_per_block * 32,)
        else:
            block_m = 16
            block_n = 16
            blocks = (
                (int(self.config.out_features) + block_n - 1) // block_n,
                (int(x_flat.shape[0]) + block_m - 1) // block_m,
            )
            threads = (block_n, block_m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="ExternalStream is deprecated.*")
            stream = cp.cuda.ExternalStream(torch.cuda.current_stream(x.device).cuda_stream)
        with stream:
            kernel(
                blocks,
                threads,
                (
                    cp.from_dlpack(x_flat),
                    cp.from_dlpack(packed),
                    cp.from_dlpack(scales),
                    cp.from_dlpack(residual),
                    cp.from_dlpack(bias),
                    cp.from_dlpack(output_flat),
                    int(x_flat.shape[0]),
                    int(self.config.out_features),
                    int(self.config.in_features),
                    int(packed.shape[1]),
                    int(bool(self.config.residual_runtime)),
                    int(self.bias is not None),
                ),
            )
        return output

    def _time_native_cuda_variant(self, x: Any, variant: str) -> float:
        warmup = max(0, int(self.config.native_cuda_autotune_warmup))
        repeat = max(1, int(self.config.native_cuda_autotune_repeat))
        for _ in range(warmup):
            self._launch_native_cuda_kernel(x, variant)
        torch.cuda.synchronize(x.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            self._launch_native_cuda_kernel(x, variant)
        end.record()
        torch.cuda.synchronize(x.device)
        return float(start.elapsed_time(end) / repeat)

    def _select_native_cuda_variant(self, x: Any) -> str:
        variant = self.config.native_cuda_kernel_variant
        self._last_native_autotuned = False
        self._last_native_autotune_cache_hit = False
        self._last_native_autotune_candidate_ms = ()
        if variant != "auto":
            if variant not in {"tiled", "warp"}:
                raise RuntimeError(f"unsupported native_cuda_kernel_variant={self.config.native_cuda_kernel_variant!r}")
            return variant
        if not self.config.native_cuda_autotune:
            return self._heuristic_native_cuda_variant(x)
        instance_key = self._native_cuda_instance_autotune_key(x)
        instance_cached = self._native_cuda_instance_variant_cache.get(instance_key)
        if instance_cached is not None:
            selected, candidates = instance_cached
            self._last_native_autotuned = True
            self._last_native_autotune_cache_hit = True
            self._last_native_autotune_candidate_ms = candidates
            return selected
        self._ensure_native_cuda_autotune_profile_loaded()
        key = self._native_cuda_autotune_key(x)
        cached = _NATIVE_TERNARY_AUTOTUNE_CACHE.get(key)
        if cached is not None:
            selected, candidates = cached
            self._native_cuda_instance_variant_cache[instance_key] = (selected, candidates)
            self._last_native_autotuned = True
            self._last_native_autotune_cache_hit = True
            self._last_native_autotune_candidate_ms = candidates
            return selected
        prewarm_failures: list[str] = []
        for candidate in ("tiled", "warp"):
            try:
                self._launch_native_cuda_kernel(x, candidate)
            except Exception as exc:
                prewarm_failures.append(f"{candidate}:{type(exc).__name__}:{exc}")
        torch.cuda.synchronize(x.device)
        candidates: list[tuple[str, float]] = []
        failures: list[str] = list(prewarm_failures)
        for candidate in ("tiled", "warp"):
            try:
                candidates.append((candidate, self._time_native_cuda_variant(x, candidate)))
            except Exception as exc:
                failures.append(f"{candidate}:{type(exc).__name__}:{exc}")
        if not candidates:
            raise RuntimeError("native CUDA autotune failed for all variants: " + "; ".join(failures))
        selected = min(candidates, key=lambda item: item[1])[0]
        measured = tuple((name, float(ms)) for name, ms in candidates)
        _NATIVE_TERNARY_AUTOTUNE_CACHE[key] = (selected, measured)
        self._native_cuda_instance_variant_cache[instance_key] = (selected, measured)
        self._persist_native_cuda_autotune_profile()
        self._last_native_autotuned = True
        self._last_native_autotune_candidate_ms = measured
        return selected

    def _native_cuda_packed_output(self, x: Any) -> Any:
        if not x.is_cuda:
            raise RuntimeError("native ternary kernel requires CUDA input")
        if x.dtype not in _CUPY_TERNARY_KERNEL_NAMES["tiled"]:
            raise RuntimeError(f"native ternary kernel does not support dtype {x.dtype}")
        if not self.config.shared_scale:
            raise RuntimeError("native ternary kernel currently requires per-output shared scales")
        variant = self._select_native_cuda_variant(x)
        self._last_native_kernel_family = variant
        self._last_native_kernel_variant = self._native_cuda_variant_label(variant)
        return self._launch_native_cuda_kernel(x, variant)

    def _quantize_input(self, x: Any) -> Any:
        if self.config.activation_bits <= 0:
            return x
        qmax = (2 ** (self.config.activation_bits - 1)) - 1
        max_abs = x.detach().abs().amax().clamp_min(1e-12)
        scale = max_abs / qmax
        q = torch.clamp(torch.round(x / scale), -qmax, qmax)
        quantized = q * scale
        dq = x + (quantized - x).detach()
        sample_limit = min(16, int(q.numel()))
        self.ledger.record_activation(ActivationQuantization(
            original=tuple(float(value) for value in x.detach().flatten()[:sample_limit].cpu().tolist()),
            quantized=tuple(int(value) for value in q.detach().flatten()[:sample_limit].cpu().tolist()),
            dequantized=tuple(float(value) for value in quantized.detach().flatten()[:sample_limit].cpu().tolist()),
            bits=self.config.activation_bits,
            scale=float(scale),
            saturated=int((q.abs() >= qmax).sum().item()),
            total_values=int(q.numel()),
        ))
        return dq

    def forward(self, x: Any) -> Any:
        xq = self._quantize_input(x)
        output_note = "BitLinear packed int2 ternary forward"
        if self.config.use_packed_ternary_runtime:
            self._ensure_quantized_buffers_current()
            native_kernel = False
            backend = "packed_int2_cuda" if xq.is_cuda else "packed_int2_torch"
            native_note = ""
            if xq.is_cuda and self.config.use_native_cuda_kernel:
                try:
                    packed_output = self._native_cuda_packed_output(xq)
                    native_kernel = True
                    kernel_variant = getattr(self, "_last_native_kernel_variant", "native_int2")
                    backend = f"native_int2_cupy_cuda_{kernel_variant}"
                    if getattr(self, "_last_native_autotuned", False):
                        cache_note = "cache-hit" if getattr(self, "_last_native_autotune_cache_hit", False) else "measured"
                        native_note = f"native CuPy RawKernel {kernel_variant} kernel autotuned {cache_note}"
                    else:
                        native_note = f"native CuPy RawKernel {kernel_variant} kernel"
                except Exception as exc:
                    if self.config.require_native_cuda_kernel:
                        raise
                    with torch.no_grad():
                        packed_weight = self._packed_runtime_weight(dtype=xq.dtype, device=xq.device)
                        packed_output = F.linear(xq, packed_weight, self.bias.to(xq.dtype) if self.bias is not None and self.bias.dtype != xq.dtype else self.bias)
                    native_note = f"native kernel unavailable: {type(exc).__name__}: {exc}"
            else:
                with torch.no_grad():
                    packed_weight = self._packed_runtime_weight(dtype=xq.dtype, device=xq.device)
                    packed_output = F.linear(xq, packed_weight, self.bias.to(xq.dtype) if self.bias is not None and self.bias.dtype != xq.dtype else self.bias)
            if self.config.use_fast_ste_autograd:
                bias_for_backward = self.bias if self.bias is not None else self.float_weight.new_empty(0)
                output = _PackedTernarySTEFunction.apply(
                    xq,
                    self.float_weight,
                    bias_for_backward,
                    packed_output,
                    self.packed_codes,
                    self.scales,
                    self.residual_weight,
                    self.bias is not None,
                    self.config.in_features,
                    self.config.residual_runtime,
                )
                max_abs_error = 0.0
                ste_note = "custom autograd STE backward skips dense STE forward"
            else:
                ste_weight = self._runtime_weight_ste()
                ste_linear_weight = ste_weight
                ste_linear_bias = self.bias
                if xq.is_cuda and xq.dtype in {torch.float16, torch.bfloat16}:
                    ste_linear_weight = ste_weight.to(xq.dtype)
                    ste_linear_bias = self.bias.to(xq.dtype) if self.bias is not None else None
                ste_output = F.linear(xq, ste_linear_weight, ste_linear_bias)
                output = ste_output + (packed_output - ste_output).detach()
                max_abs_error = float((packed_output.detach() - ste_output.detach()).abs().max().cpu()) if packed_output.numel() else 0.0
                ste_note = "dense STE forward compatibility path"
            self.ledger.record_packed_ternary_dispatch(PackedTernaryDispatch(
                layer_id=self.config.log_prefix,
                backend=backend,
                device=str(xq.device),
                packed_weight_bytes=int(self.packed_codes.numel()),
                active_weights=self._last_active_weights,
                total_weights=self._last_total_weights,
                max_abs_error_vs_ste=max_abs_error,
                used_residual=bool(self.config.residual_runtime),
                note=(
                    "forward value read from packed 2-bit ternary weight buffer; gradient uses STE"
                    + (f"; {native_note}" if native_note else "")
                    + f"; {ste_note}"
                ),
                native_kernel=native_kernel,
                kernel_variant=getattr(self, "_last_native_kernel_variant", "") if native_kernel else "torch_unpack_linear",
                autotuned=bool(getattr(self, "_last_native_autotuned", False)) if native_kernel else False,
                autotune_cache_hit=bool(getattr(self, "_last_native_autotune_cache_hit", False)) if native_kernel else False,
                autotune_candidate_ms=(
                    tuple(getattr(self, "_last_native_autotune_candidate_ms", ()))
                    if native_kernel
                    else ()
                ),
            ))
        else:
            ste_weight = self._runtime_weight_ste()
            output = F.linear(xq, ste_weight, self.bias)
            max_abs_error = 0.0
            output_note = "BitLinear STE ternary forward"
        self.ledger.record_layer_forward(LayerForwardEvent(
            layer_id=self.config.log_prefix,
            input_shape=tuple(int(dim) for dim in x.shape),
            output_shape=tuple(int(dim) for dim in output.shape),
            active_weights=self._last_active_weights,
            total_weights=self._last_total_weights,
            estimated_weight_bits=self._last_estimated_bits,
            activation_bits=float(x.numel() * max(self.config.activation_bits, 0)),
            note=f"{output_note}; max_abs_error_vs_ste={max_abs_error:.6g}",
        ))
        return output

    def instrumentation_summary(self) -> Mapping[str, Any]:
        return self.ledger.to_dict()
