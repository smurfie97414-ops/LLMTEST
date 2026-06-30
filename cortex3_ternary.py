from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import fsum
from typing import Any, Iterable, Mapping, Sequence

from cortex3 import CostTrace, TernaryBlock, ZeroState, ternarize_values


import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass
class CompressionTraceLedger:
    compression_decisions: list[CompressionDecision] = field(default_factory=list)
    activation_quantizations: list[ActivationQuantization] = field(default_factory=list)
    expert_activations: list[ExpertActivation] = field(default_factory=list)
    kv_events: list[KVModeEvent] = field(default_factory=list)
    mtp_fsp_events: list[MTPFSPEvent] = field(default_factory=list)
    layer_forward_events: list[LayerForwardEvent] = field(default_factory=list)
    retention_limit: int | None = None
    total_compression_decisions: int = 0
    total_activation_quantizations: int = 0
    total_expert_activations: int = 0
    total_kv_events: int = 0
    total_mtp_fsp_events: int = 0
    total_layer_forward_events: int = 0
    total_weight_bits_read: float = 0.0
    total_activation_bits: float = 0.0
    total_kv_bytes: float = 0.0

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
        return {
            "compression_decisions": [asdict(item) for item in self.compression_decisions],
            "activation_quantizations": [asdict(item) for item in self.activation_quantizations],
            "expert_activations": [asdict(item) for item in self.expert_activations],
            "kv_events": [asdict(item) for item in self.kv_events],
            "mtp_fsp_events": [asdict(item) for item in self.mtp_fsp_events],
            "layer_forward_events": [asdict(item) for item in self.layer_forward_events],
            "retention_limit": self.retention_limit,
            "retained_event_counts": {
                "compression_decisions": len(self.compression_decisions),
                "activation_quantizations": len(self.activation_quantizations),
                "expert_activations": len(self.expert_activations),
                "kv_events": len(self.kv_events),
                "mtp_fsp_events": len(self.mtp_fsp_events),
                "layer_forward_events": len(self.layer_forward_events),
            },
            "total_event_counts": {
                "compression_decisions": self.total_compression_decisions,
                "activation_quantizations": self.total_activation_quantizations,
                "expert_activations": self.total_expert_activations,
                "kv_events": self.total_kv_events,
                "mtp_fsp_events": self.total_mtp_fsp_events,
                "layer_forward_events": self.total_layer_forward_events,
            },
            "cost_trace": asdict(self.cost_trace),
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
    shared_scale: bool = True
    log_prefix: str = "bitlinear"


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
        self._last_active_weights = int(config.out_features * config.in_features)
        self._last_total_weights = int(config.out_features * config.in_features)
        self._last_estimated_bits = float(config.out_features * config.in_features)
        nn.init.kaiming_uniform_(self.float_weight, a=5 ** 0.5)
        self.requantize()

    @classmethod
    def from_linear(cls, linear: Any, *, activation_bits: int = 4, threshold: float | None = None, residual_threshold: float = 0.0, ledger: CompressionTraceLedger | None = None) -> "BitLinear":
        config = BitLinearConfig(linear.in_features, linear.out_features, linear.bias is not None, activation_bits, threshold, residual_threshold)
        module = cls(config, ledger=ledger)
        with torch.no_grad():
            module.float_weight.copy_(linear.weight)
            if linear.bias is not None and module.bias is not None:
                module.bias.copy_(linear.bias)
        module.requantize()
        return module

    def requantize(self, *, certify_zeros: bool = False) -> None:
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
                note="torch BitLinear requantize tensor-stats",
            )
            self.ledger.record_compression(decision)

    def _runtime_weight_ste(self) -> Any:
        values = self.float_weight
        scale = values.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-12) if self.config.shared_scale else values.detach().abs().mean().clamp_min(1e-12)
        threshold = self.config.threshold if self.config.threshold is not None else 0.5 * scale
        signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
        mask = (values.detach().abs() >= threshold).to(values.dtype)
        quantized = signs * mask * scale
        residual = values - quantized
        if self.config.residual_threshold > 0:
            residual = torch.where(residual.detach().abs() > self.config.residual_threshold, residual, torch.zeros_like(residual))
        runtime = quantized + residual
        weight = values + (runtime - values).detach()
        return weight

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
        weight = self._runtime_weight_ste()
        output = F.linear(xq, weight, self.bias)
        self.ledger.record_layer_forward(LayerForwardEvent(
            layer_id=self.config.log_prefix,
            input_shape=tuple(int(dim) for dim in x.shape),
            output_shape=tuple(int(dim) for dim in output.shape),
            active_weights=self._last_active_weights,
            total_weights=self._last_total_weights,
            estimated_weight_bits=self._last_estimated_bits,
            activation_bits=float(x.numel() * max(self.config.activation_bits, 0)),
            note="BitLinear real forward",
        ))
        return output

    def instrumentation_summary(self) -> Mapping[str, Any]:
        return self.ledger.to_dict()
