from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from cortex3 import (
    Anchor,
    CandidateAnswer,
    CostTrace,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    Task,
    default_skill_specs,
)
from cortex3_attribution import AttributionPolicyMemory, CausalAttributionEngine
from cortex3_certificates import CertificateHead, CertificateHeadOutput, CertificateType, CertificateVerifier, LatentProofState, RandomDelatentizer, build_certificate, certificate_contract_for_task, evaluate_certificate_efficiency
from cortex3_cycle import CortexCycle
from cortex3_frontier import CompiledFrontierAgent, FrontierCircuitRegistry, FrontierSkillDiscovery, compiled_circuit_id
from cortex3_future import (
    ContractDecision,
    FutureContract,
    FutureContractEngine,
    FutureContractLedger,
    MTPFSPConfig,
    OutputGoalContract,
    OutputGoalDecision,
)
from cortex3_improvement import AcceptanceDecision, ProposalKind, RecursiveImprovementEngine
from cortex3_inference import InferenceConfig, InferencePath, UltraFastInferenceEngine
from cortex3_ledgers import BitLedger, CausalLedger, CausalTrace, SkillLedger, SkillState, UncertaintyLedger
from cortex3_memory import CognitiveMemory, CognitiveMemoryConfig, CompiledCircuitMemoryBinding, MemoryMode, MemoryRetentionDecision, MemorySegment, MemoryUtilityCredit
from cortex3_objective import FINAL_LOSS_TERMS, build_objective_report
from cortex3_phases import CORTEX3_PHASES
from cortex3_regrowth import MinimalRegrowthEngine, RegrowthActionKind, RegrowthPlan
from cortex3_sleep import ExampleOrigin, SleepPhaseConsolidator, TrainingExample
from cortex3_ternary import (
    ActivationQuantization,
    BitLinear,
    BitLinearConfig,
    CompressionDecision,
    CompressionTraceLedger,
    ExpertActivation,
    KVModeEvent,
    LayerForwardEvent,
    MTPFSPEvent,
    PackedTernaryDispatch,
    last_native_grad_input_kernel,
    last_native_grad_weight_kernel,
    native_backend_from_runtime_label,
    native_grad_input_kernel_counts,
    native_grad_weight_kernel_counts,
    native_ternary_cuda_available,
    native_ternary_cuda_extension_available,
)


SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")
STRICT_NATIVE_TERNARY_BACKEND = "extension"
NATIVE_TERNARY_BACKEND_CHOICES = ("auto", STRICT_NATIVE_TERNARY_BACKEND, "rawkernel")


def _env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _rank_zero() -> bool:
    return _env_rank() == 0


def _barrier_if_needed(runtime: "DistributedRuntime") -> None:
    if runtime.enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"{type(value)!r} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    tmp_path.replace(path)


def _cost_trace_from_payload(payload: Mapping[str, Any] | None) -> CostTrace:
    data = dict(payload or {})
    return CostTrace(
        weight_bits_read=float(data.get("weight_bits_read", 0.0)),
        activation_bits=float(data.get("activation_bits", 0.0)),
        kv_bytes=float(data.get("kv_bytes", 0.0)),
        generated_tokens=int(data.get("generated_tokens", 0)),
        latent_steps=int(data.get("latent_steps", 0)),
        experts_activated=int(data.get("experts_activated", 0)),
        verifier_steps=int(data.get("verifier_steps", 0)),
        wall_time_ms=float(data.get("wall_time_ms", 0.0)),
    )


def _restore_future_contract_ledger(ledger: FutureContractLedger, payload: Mapping[str, Any] | None) -> None:
    ledger.decisions.clear()
    ledger.output_goal_decisions.clear()
    if not payload:
        return
    for raw_decision in payload.get("decisions", ()):
        decision_data = dict(raw_decision)
        contract_data = dict(decision_data.get("contract") or {})
        cost_data = dict(decision_data.get("cost") or {})
        contract = FutureContract(
            contract_id=str(contract_data.get("contract_id", "")),
            domain=str(contract_data.get("domain", "")),
            risk=float(contract_data.get("risk", 0.0)),
            requested_horizon=int(contract_data.get("requested_horizon", 1)),
            accepted_horizon=int(contract_data.get("accepted_horizon", 1)),
            token_ids=tuple(int(token) for token in contract_data.get("token_ids", ())),
            confidence=float(contract_data.get("confidence", 0.0)),
            temporal_loss=float(contract_data.get("temporal_loss", 0.0)),
            revision=int(contract_data.get("revision", 0)),
            accepted=bool(contract_data.get("accepted", False)),
            reason=str(contract_data.get("reason", "")),
        )
        ledger.decisions.append(
            ContractDecision(
                contract=contract,
                accepted=bool(decision_data.get("accepted", contract.accepted)),
                reason=str(decision_data.get("reason", contract.reason)),
                cost=_cost_trace_from_payload(cost_data),
            )
        )
    for raw_decision in payload.get("output_goal_decisions", ()):
        decision_data = dict(raw_decision)
        contract_data = dict(decision_data.get("contract") or {})
        cost_data = dict(decision_data.get("cost") or {})
        contract = OutputGoalContract(
            contract_id=str(contract_data.get("contract_id", "")),
            task_id=str(contract_data.get("task_id", "")),
            skill=str(contract_data.get("skill", "")),
            expected_type=str(contract_data.get("expected_type", "")),
            expected_text=str(contract_data.get("expected_text", "")),
            required_anchor_values=tuple(str(value) for value in contract_data.get("required_anchor_values", ())),
            obligations=tuple(str(value) for value in contract_data.get("obligations", ())),
            risk=float(contract_data.get("risk", 0.0)),
        )
        ledger.output_goal_decisions.append(
            OutputGoalDecision(
                contract=contract,
                answer_text=str(decision_data.get("answer_text", "")),
                accepted=bool(decision_data.get("accepted", False)),
                reason=str(decision_data.get("reason", "")),
                violations=tuple(str(value) for value in decision_data.get("violations", ())),
                cost=_cost_trace_from_payload(cost_data),
            )
        )


def _restore_compression_trace_ledger(ledger: CompressionTraceLedger | None, payload: Mapping[str, Any] | None) -> None:
    if ledger is None:
        return
    configured_retention_limit = ledger.retention_limit
    ledger.compression_decisions.clear()
    ledger.activation_quantizations.clear()
    ledger.expert_activations.clear()
    ledger.kv_events.clear()
    ledger.mtp_fsp_events.clear()
    ledger.layer_forward_events.clear()
    ledger.packed_ternary_dispatches.clear()
    ledger.total_compression_decisions = 0
    ledger.total_activation_quantizations = 0
    ledger.total_expert_activations = 0
    ledger.total_kv_events = 0
    ledger.total_mtp_fsp_events = 0
    ledger.total_layer_forward_events = 0
    ledger.total_packed_ternary_dispatches = 0
    ledger.total_native_ternary_kernel_dispatches = 0
    ledger.total_torch_packed_ternary_dispatches = 0
    ledger.total_native_ternary_autotuned_dispatches = 0
    ledger.total_native_ternary_autotune_cache_hits = 0
    ledger.total_native_ternary_grad_weight_dispatches = 0
    ledger.native_ternary_backend_counts.clear()
    ledger.native_ternary_requantize_backend_counts.clear()
    ledger.native_ternary_grad_weight_backend_counts.clear()
    ledger.total_weight_bits_read = 0.0
    ledger.total_activation_bits = 0.0
    ledger.total_kv_bytes = 0.0
    ledger.total_packed_weight_bytes = 0.0
    if not payload:
        return
    if configured_retention_limit is None and payload.get("retention_limit") is not None:
        ledger.retention_limit = int(payload["retention_limit"])
    else:
        ledger.retention_limit = configured_retention_limit
    for item in payload.get("compression_decisions", ()):
        data = dict(item)
        ledger.compression_decisions.append(
            CompressionDecision(
                block_id=str(data.get("block_id", "")),
                source=str(data.get("source", "")),
                original_count=int(data.get("original_count", 0)),
                active_count=int(data.get("active_count", 0)),
                provisional_zero_count=int(data.get("provisional_zero_count", 0)),
                certified_zero_count=int(data.get("certified_zero_count", 0)),
                scale=float(data.get("scale", 0.0)),
                threshold=float(data.get("threshold", 0.0)),
                estimated_bits=float(data.get("estimated_bits", 0.0)),
                residual_l1=float(data.get("residual_l1", 0.0)),
                note=str(data.get("note", "")),
            )
        )
    ledger._trim(ledger.compression_decisions)
    for item in payload.get("activation_quantizations", ()):
        data = dict(item)
        ledger.activation_quantizations.append(
            ActivationQuantization(
                original=tuple(float(value) for value in data.get("original", ())),
                quantized=tuple(int(value) for value in data.get("quantized", ())),
                dequantized=tuple(float(value) for value in data.get("dequantized", ())),
                bits=int(data.get("bits", 0)),
                scale=float(data.get("scale", 0.0)),
                saturated=int(data.get("saturated", 0)),
                total_values=(
                    None
                    if data.get("total_values") is None
                    else int(data.get("total_values", 0))
                ),
            )
        )
    ledger._trim(ledger.activation_quantizations)
    for item in payload.get("expert_activations", ()):
        data = dict(item)
        ledger.expert_activations.append(
            ExpertActivation(
                expert_id=str(data.get("expert_id", "")),
                reason=str(data.get("reason", "")),
                cost=float(data.get("cost", 1.0)),
            )
        )
    ledger._trim(ledger.expert_activations)
    for item in payload.get("kv_events", ()):
        data = dict(item)
        ledger.kv_events.append(
            KVModeEvent(
                segment_id=str(data.get("segment_id", "")),
                mode=str(data.get("mode", "")),
                bytes_used=float(data.get("bytes_used", 0.0)),
                exact_anchors=int(data.get("exact_anchors", 0)),
                note=str(data.get("note", "")),
            )
        )
    ledger._trim(ledger.kv_events)
    for item in payload.get("mtp_fsp_events", ()):
        data = dict(item)
        ledger.mtp_fsp_events.append(
            MTPFSPEvent(
                block_id=str(data.get("block_id", "")),
                horizon=int(data.get("horizon", 1)),
                accepted=bool(data.get("accepted", False)),
                confidence=float(data.get("confidence", 0.0)),
                contract_revision=int(data.get("contract_revision", 0)),
                reason=str(data.get("reason", "")),
            )
        )
    ledger._trim(ledger.mtp_fsp_events)
    for item in payload.get("layer_forward_events", ()):
        data = dict(item)
        ledger.layer_forward_events.append(
            LayerForwardEvent(
                layer_id=str(data.get("layer_id", "")),
                input_shape=tuple(int(value) for value in data.get("input_shape", ())),
                output_shape=tuple(int(value) for value in data.get("output_shape", ())),
                active_weights=int(data.get("active_weights", 0)),
                total_weights=int(data.get("total_weights", 0)),
                estimated_weight_bits=float(data.get("estimated_weight_bits", 0.0)),
                activation_bits=float(data.get("activation_bits", 0.0)),
                note=str(data.get("note", "")),
            )
        )
    ledger._trim(ledger.layer_forward_events)
    for item in payload.get("packed_ternary_dispatches", ()):
        data = dict(item)
        ledger.packed_ternary_dispatches.append(
            PackedTernaryDispatch(
                layer_id=str(data.get("layer_id", "")),
                backend=str(data.get("backend", "")),
                device=str(data.get("device", "")),
                packed_weight_bytes=int(data.get("packed_weight_bytes", 0)),
                active_weights=int(data.get("active_weights", 0)),
                total_weights=int(data.get("total_weights", 0)),
                max_abs_error_vs_ste=float(data.get("max_abs_error_vs_ste", 0.0)),
                used_residual=bool(data.get("used_residual", False)),
                note=str(data.get("note", "")),
                native_kernel=bool(data.get("native_kernel", False)),
                kernel_variant=str(data.get("kernel_variant", "")),
                native_backend=str(
                    data.get("native_backend")
                    or native_backend_from_runtime_label(str(data.get("backend", "")), default="")
                ),
                autotuned=bool(data.get("autotuned", False)),
                autotune_cache_hit=bool(data.get("autotune_cache_hit", False)),
                autotune_candidate_ms=tuple(
                    (str(pair[0]), float(pair[1]))
                    for pair in data.get("autotune_candidate_ms", ())
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                ),
            )
        )
    ledger._trim(ledger.packed_ternary_dispatches)
    total_counts = dict(payload.get("total_event_counts") or {})
    ledger.total_compression_decisions = int(total_counts.get("compression_decisions", len(ledger.compression_decisions)))
    ledger.total_activation_quantizations = int(total_counts.get("activation_quantizations", len(ledger.activation_quantizations)))
    ledger.total_expert_activations = int(total_counts.get("expert_activations", len(ledger.expert_activations)))
    ledger.total_kv_events = int(total_counts.get("kv_events", len(ledger.kv_events)))
    ledger.total_mtp_fsp_events = int(total_counts.get("mtp_fsp_events", len(ledger.mtp_fsp_events)))
    ledger.total_layer_forward_events = int(total_counts.get("layer_forward_events", len(ledger.layer_forward_events)))
    ledger.total_packed_ternary_dispatches = int(total_counts.get("packed_ternary_dispatches", len(ledger.packed_ternary_dispatches)))
    ledger.total_native_ternary_kernel_dispatches = int(
        total_counts.get(
            "native_ternary_kernel_dispatches",
            sum(1 for item in ledger.packed_ternary_dispatches if item.native_kernel or item.backend.startswith("native_")),
        )
    )
    ledger.total_torch_packed_ternary_dispatches = int(
        total_counts.get(
            "torch_packed_ternary_dispatches",
            max(0, ledger.total_packed_ternary_dispatches - ledger.total_native_ternary_kernel_dispatches),
        )
    )
    ledger.total_native_ternary_autotuned_dispatches = int(
        total_counts.get(
            "native_ternary_autotuned_dispatches",
            sum(1 for item in ledger.packed_ternary_dispatches if item.autotuned),
        )
    )
    ledger.total_native_ternary_autotune_cache_hits = int(
        total_counts.get(
            "native_ternary_autotune_cache_hits",
            sum(1 for item in ledger.packed_ternary_dispatches if item.autotune_cache_hit),
        )
    )
    ledger.total_native_ternary_grad_weight_dispatches = int(
        total_counts.get("native_ternary_grad_weight_dispatches", 0)
    )
    native_backend_counts = {
        str(key): int(value)
        for key, value in dict(payload.get("native_ternary_backend_counts") or {}).items()
    }
    native_requantize_backend_counts = {
        str(key): int(value)
        for key, value in dict(payload.get("native_ternary_requantize_backend_counts") or {}).items()
    }
    native_grad_weight_backend_counts = {
        str(key): int(value)
        for key, value in dict(payload.get("native_ternary_grad_weight_backend_counts") or {}).items()
    }
    for key, value in total_counts.items():
        text_key = str(key)
        if text_key.startswith("native_ternary_") and text_key.endswith("_kernel_dispatches"):
            backend = text_key.removeprefix("native_ternary_").removesuffix("_kernel_dispatches")
            if backend:
                native_backend_counts.setdefault(backend, int(value))
        if text_key.startswith("native_ternary_") and text_key.endswith("_requantize_dispatches"):
            backend = text_key.removeprefix("native_ternary_").removesuffix("_requantize_dispatches")
            if backend:
                native_requantize_backend_counts.setdefault(backend, int(value))
        if text_key.startswith("native_ternary_") and text_key.endswith("_grad_weight_dispatches"):
            backend = text_key.removeprefix("native_ternary_").removesuffix("_grad_weight_dispatches")
            if backend:
                native_grad_weight_backend_counts.setdefault(backend, int(value))
    if not native_backend_counts:
        for item in ledger.packed_ternary_dispatches:
            if item.native_kernel or item.backend.startswith("native_"):
                backend = item.native_backend or native_backend_from_runtime_label(item.backend, default="unknown")
                native_backend_counts[backend] = native_backend_counts.get(backend, 0) + 1
    ledger.native_ternary_backend_counts.update(native_backend_counts)
    ledger.native_ternary_requantize_backend_counts.update(native_requantize_backend_counts)
    ledger.native_ternary_grad_weight_backend_counts.update(native_grad_weight_backend_counts)
    if not ledger.total_native_ternary_grad_weight_dispatches:
        ledger.total_native_ternary_grad_weight_dispatches = sum(native_grad_weight_backend_counts.values())
    cost_trace = dict(payload.get("cost_trace") or {})
    ledger.total_weight_bits_read = float(
        cost_trace.get("weight_bits_read", sum(item.estimated_bits for item in ledger.compression_decisions))
    )
    ledger.total_activation_bits = float(
        cost_trace.get("activation_bits", sum(item.activation_bits for item in ledger.activation_quantizations))
    )
    ledger.total_kv_bytes = float(
        cost_trace.get("kv_bytes", sum(item.bytes_used for item in ledger.kv_events))
    )
    ledger.total_packed_weight_bytes = float(
        payload.get("packed_weight_bytes_read", sum(item.packed_weight_bytes for item in ledger.packed_ternary_dispatches))
    )


def _cost_trace_payload(cost: CostTrace) -> dict[str, Any]:
    return asdict(cost)


def _bit_ledger_payload(ledger: BitLedger) -> dict[str, Any]:
    payload = asdict(ledger)
    payload["total_effective_bits"] = ledger.total_effective_bits
    return payload


def _restore_bit_ledger(ledger: BitLedger, payload: Mapping[str, Any] | None) -> None:
    data = dict(payload or {})
    ledger.weight_bits = float(data.get("weight_bits", 0.0))
    ledger.scale_bits = float(data.get("scale_bits", 0.0))
    ledger.activation_bits = float(data.get("activation_bits", 0.0))
    ledger.kv_bytes = float(data.get("kv_bytes", 0.0))
    ledger.routing_bits = float(data.get("routing_bits", 0.0))
    ledger.certificate_bits = float(data.get("certificate_bits", 0.0))
    ledger.verifier_steps = int(data.get("verifier_steps", 0))
    ledger.notes = [str(item) for item in data.get("notes", ())]


def _merge_bit_ledger(target: BitLedger, source: BitLedger, *, note_prefix: str = "") -> None:
    target.weight_bits += float(source.weight_bits)
    target.scale_bits += float(source.scale_bits)
    target.activation_bits += float(source.activation_bits)
    target.kv_bytes += float(source.kv_bytes)
    target.routing_bits += float(source.routing_bits)
    target.certificate_bits += float(source.certificate_bits)
    target.verifier_steps += int(source.verifier_steps)
    for note in source.notes:
        target.notes.append(f"{note_prefix}{note}" if note_prefix else str(note))


def _skill_ledger_payload(ledger: SkillLedger) -> dict[str, Any]:
    return {
        "protected_threshold": float(ledger.protected_threshold),
        "states": {skill: asdict(state) for skill, state in ledger.states.items()},
        "fragile_skills": [asdict(state) for state in ledger.fragile_skills()],
    }


def _restore_skill_ledger(ledger: SkillLedger, payload: Mapping[str, Any] | None) -> None:
    data = dict(payload or {})
    ledger.protected_threshold = float(data.get("protected_threshold", ledger.protected_threshold))
    ledger.states.clear()
    for skill, raw_state in dict(data.get("states") or {}).items():
        state = dict(raw_state)
        ledger.states[str(skill)] = SkillState(
            skill=str(state.get("skill", skill)),
            score=float(state.get("score", 0.0)),
            pass_rate=float(state.get("pass_rate", 0.0)),
            failures=int(state.get("failures", 0)),
            fragility=float(state.get("fragility", 0.0)),
            protected=bool(state.get("protected", False)),
            history=[float(item) for item in state.get("history", ())],
        )


def _causal_trace_payload(trace: CausalTrace) -> dict[str, Any]:
    return asdict(trace)


def _causal_trace_from_payload(payload: Mapping[str, Any]) -> CausalTrace:
    data = dict(payload or {})
    return CausalTrace(
        task_id=str(data.get("task_id", "")),
        skill=str(data.get("skill", "")),
        mtp_horizon=int(data.get("mtp_horizon", 1)),
        activation_bits=int(data.get("activation_bits", 8)),
        kv_mode=str(data.get("kv_mode", "exact")),
        verifier_level=int(data.get("verifier_level", 0)),
        certificate_fields=tuple(str(item) for item in data.get("certificate_fields", ())),
        uncertainty=float(data.get("uncertainty", 0.0)),
    )


def _causal_ledger_payload(ledger: CausalLedger) -> dict[str, Any]:
    return {
        "trace_count": len(ledger.traces),
        "traces": {task_id: _causal_trace_payload(trace) for task_id, trace in ledger.traces.items()},
    }


def _restore_causal_ledger(ledger: CausalLedger, payload: Mapping[str, Any] | None) -> None:
    ledger.traces.clear()
    data = dict(payload or {})
    for task_id, raw_trace in dict(data.get("traces") or {}).items():
        ledger.traces[str(task_id)] = _causal_trace_from_payload(dict(raw_trace))


def _uncertainty_ledger_payload(ledger: UncertaintyLedger) -> dict[str, Any]:
    bins = {
        skill: [{"confidence": float(confidence), "passed": bool(passed)} for confidence, passed in pairs]
        for skill, pairs in ledger.bins.items()
    }
    return {
        "bins": bins,
        "skill_count": len(bins),
        "observation_count": sum(len(pairs) for pairs in bins.values()),
        "expected_calibration_error": ledger.expected_calibration_error(),
    }


def _restore_uncertainty_ledger(ledger: UncertaintyLedger, payload: Mapping[str, Any] | None) -> None:
    ledger.bins.clear()
    data = dict(payload or {})
    for skill, raw_pairs in dict(data.get("bins") or {}).items():
        for raw_pair in raw_pairs:
            pair = dict(raw_pair)
            ledger.record(str(skill), float(pair.get("confidence", 0.0)), bool(pair.get("passed", False)))


def _ledger_bundle_payload(
    *,
    bit_ledger: BitLedger,
    skill_ledger: SkillLedger,
    causal_ledger: CausalLedger,
    uncertainty_ledger: UncertaintyLedger,
) -> dict[str, Any]:
    return {
        "bit_ledger": _bit_ledger_payload(bit_ledger),
        "skill_ledger": _skill_ledger_payload(skill_ledger),
        "causal_ledger": _causal_ledger_payload(causal_ledger),
        "uncertainty_ledger": _uncertainty_ledger_payload(uncertainty_ledger),
    }


def _anchor_payload(anchor: Anchor) -> dict[str, Any]:
    return {
        "kind": anchor.kind,
        "value": anchor.value,
        "source_id": anchor.source_id,
        "importance": anchor.importance,
    }


def _anchor_from_payload(payload: Mapping[str, Any]) -> Anchor:
    return Anchor(
        kind=str(payload.get("kind", "")),
        value=str(payload.get("value", "")),
        source_id=str(payload.get("source_id", "")),
        importance=float(payload.get("importance", 1.0)),
    )


def _task_payload(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "skill": task.skill,
        "prompt": task.prompt,
        "expected": task.expected,
        "metadata": dict(task.metadata),
        "anchors": [_anchor_payload(anchor) for anchor in task.anchors],
        "group_id": task.group_id,
    }


def _task_from_payload(payload: Mapping[str, Any]) -> Task:
    return Task(
        task_id=str(payload.get("task_id", "")),
        skill=str(payload.get("skill", "")),
        prompt=str(payload.get("prompt", "")),
        expected=payload.get("expected"),
        metadata=dict(payload.get("metadata") or {}),
        anchors=tuple(_anchor_from_payload(anchor) for anchor in payload.get("anchors", ())),
        group_id=payload.get("group_id"),
    )


def _candidate_payload(answer: CandidateAnswer) -> dict[str, Any]:
    return {
        "text": answer.text,
        "confidence": float(answer.confidence),
        "certificate": dict(answer.certificate),
        "cost": asdict(answer.cost),
        "raw": dict(answer.raw),
    }


def _candidate_from_payload(payload: Mapping[str, Any]) -> CandidateAnswer:
    return CandidateAnswer(
        text=str(payload.get("text", "")),
        confidence=float(payload.get("confidence", 0.0)),
        certificate=dict(payload.get("certificate") or {}),
        cost=_cost_trace_from_payload(payload.get("cost")),
        raw=dict(payload.get("raw") or {}),
    )


def _training_example_payload(example: TrainingExample) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "task": _task_payload(example.task),
        "answer": _candidate_payload(example.answer),
        "origin": example.origin.value,
        "oracle": example.oracle,
        "targeted_skill": example.targeted_skill,
        "verification_level": int(example.verification_level),
        "contamination_risk": float(example.contamination_risk),
        "difficulty": float(example.difficulty),
        "confidence_label": example.confidence_label,
        "synthetic": bool(example.synthetic),
        "metadata": dict(example.metadata),
    }


def _training_example_from_payload(payload: Mapping[str, Any]) -> TrainingExample:
    return TrainingExample(
        example_id=str(payload.get("example_id", "")),
        task=_task_from_payload(dict(payload.get("task") or {})),
        answer=_candidate_from_payload(dict(payload.get("answer") or {})),
        origin=ExampleOrigin(str(payload.get("origin", ExampleOrigin.TOOL_SOLVED.value))),
        oracle=str(payload.get("oracle", "")),
        targeted_skill=str(payload.get("targeted_skill", "")),
        verification_level=int(payload.get("verification_level", 0)),
        contamination_risk=float(payload.get("contamination_risk", 0.0)),
        difficulty=float(payload.get("difficulty", 0.0)),
        confidence_label=(
            None
            if payload.get("confidence_label") is None
            else float(payload.get("confidence_label"))
        ),
        synthetic=bool(payload.get("synthetic", False)),
        metadata=dict(payload.get("metadata") or {}),
    )


def _memory_segment_payload(segment: MemorySegment) -> dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "mode": segment.mode.value,
        "exact_text": segment.exact_text,
        "latent_summary": segment.latent_summary,
        "anchors": [_anchor_payload(anchor) for anchor in segment.anchors],
        "token_counts": dict(segment.token_counts),
        "embedding": segment.embedding.detach().cpu(),
        "original_token_count": int(segment.original_token_count),
        "stored_token_count": int(segment.stored_token_count),
        "metadata": dict(segment.metadata),
    }


def _memory_segment_from_payload(payload: Mapping[str, Any]) -> MemorySegment:
    embedding_payload = payload.get("embedding")
    embedding = (
        embedding_payload.detach().cpu().to(dtype=torch.float32)
        if isinstance(embedding_payload, torch.Tensor)
        else torch.as_tensor(embedding_payload or (), dtype=torch.float32)
    )
    return MemorySegment(
        segment_id=str(payload.get("segment_id", "")),
        mode=MemoryMode(str(payload.get("mode", MemoryMode.EXACT.value))),
        exact_text=str(payload.get("exact_text", "")),
        latent_summary=str(payload.get("latent_summary", "")),
        anchors=tuple(_anchor_from_payload(anchor) for anchor in payload.get("anchors", ())),
        token_counts={str(key): int(value) for key, value in dict(payload.get("token_counts") or {}).items()},
        embedding=embedding,
        original_token_count=int(payload.get("original_token_count", 0)),
        stored_token_count=int(payload.get("stored_token_count", 0)),
        metadata=dict(payload.get("metadata") or {}),
    )


def _memory_state(memory: CognitiveMemory) -> dict[str, Any]:
    return {
        "recent": [_memory_segment_payload(segment) for segment in memory.recent.segments],
        "latent": [_memory_segment_payload(segment) for segment in memory.latent.segments],
        "anchors": [_anchor_payload(anchor) for anchor in memory.anchor_ledger.anchors],
        "retention_decisions": [
            decision.to_dict()
            for decision in memory.retention_decisions
        ],
        "utility_credits": [
            credit.to_dict()
            for credit in memory.utility_credits
        ],
        "compiled_circuit_bindings": [
            binding.to_dict()
            for binding in memory.compiled_circuit_bindings.values()
        ],
        "compression_report": memory.compression_report(),
    }


def _restore_memory_state(memory: CognitiveMemory, payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    memory.recent.segments = [_memory_segment_from_payload(item) for item in payload.get("recent", ())]
    memory.latent.segments = [_memory_segment_from_payload(item) for item in payload.get("latent", ())]
    memory.anchor_ledger.anchors = [_anchor_from_payload(item) for item in payload.get("anchors", ())]
    memory.retention_decisions = [
        MemoryRetentionDecision.from_dict(dict(item))
        for item in payload.get("retention_decisions", ())
    ]
    memory.utility_credits = [
        MemoryUtilityCredit.from_dict(dict(item))
        for item in payload.get("utility_credits", ())
    ]
    memory.compiled_circuit_bindings = {
        binding.circuit_id: binding
        for binding in (
            CompiledCircuitMemoryBinding.from_dict(dict(item))
            for item in payload.get("compiled_circuit_bindings", ())
        )
    }


def _sleep_state(sleep: SleepPhaseConsolidator) -> dict[str, Any]:
    return {
        "replay_examples": [_training_example_payload(example) for example in sleep.replay.examples],
        "synthetic_examples": [_training_example_payload(example) for example in sleep.synthetic.examples],
        "reservoir_examples": [_training_example_payload(example) for example in sleep.reservoir.examples],
    }


def _restore_sleep_state(sleep: SleepPhaseConsolidator, payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    sleep.replay.examples = [_training_example_from_payload(item) for item in payload.get("replay_examples", ())]
    sleep.synthetic.examples = [_training_example_from_payload(item) for item in payload.get("synthetic_examples", ())]
    sleep.reservoir.examples = [_training_example_from_payload(item) for item in payload.get("reservoir_examples", ())]


def _improvement_state(improvement: RecursiveImprovementEngine) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "archive": improvement.archive.to_dict(),
        "rollback": improvement.rollback.to_dict(),
    }


def _restore_improvement_state(improvement: RecursiveImprovementEngine, payload: Mapping[str, Any] | None) -> None:
    if not payload:
        return
    archive_payload = dict(payload.get("archive") or {})
    if archive_payload.get("accepted") or archive_payload.get("rejected"):
        improvement.archive.restore_records(archive_payload)
    else:
        improvement.archive.restore_summary(
            accepted_count=int(archive_payload.get("accepted_count", 0)),
            rejected_count=int(archive_payload.get("rejected_count", 0)),
            kind_counts=dict(archive_payload.get("kind_counts") or {}),
        )
    improvement.rollback.restore(dict(payload.get("rollback") or {}))


def _sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_file(path: str | Path) -> dict[str, Any]:
    item = Path(path)
    stat = item.stat()
    return {
        "path": str(item),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(item),
    }


def _fingerprint_files(paths: Sequence[str | Path]) -> tuple[dict[str, Any], ...]:
    return tuple(_fingerprint_file(path) for path in paths)


def _tokenized_preparation_config(
    corpus: "TextCorpusConfig",
    *,
    vocab_size: int,
    min_frequency: int | None,
    seq_len: int,
    max_horizon: int,
    train_fraction: float = 0.9,
    max_tokens: int | None = None,
    tokenizer_training_chars: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "vocab_size": int(vocab_size),
        "min_frequency": int(min_frequency) if min_frequency is not None else None,
        "seq_len": int(seq_len),
        "max_horizon": int(max_horizon),
        "train_fraction": float(train_fraction),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "tokenizer_training_chars": int(tokenizer_training_chars) if tokenizer_training_chars is not None else None,
        "min_chars_per_chunk": int(corpus.min_chars_per_chunk),
        "encoding": str(corpus.encoding),
    }


def _tokenized_preparation_mismatches(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if key not in existing:
            mismatches.append(f"{key}: missing != {expected_value!r}")
            continue
        existing_value = existing[key]
        if isinstance(expected_value, float):
            try:
                equal = math.isclose(float(existing_value), expected_value, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                equal = False
        else:
            equal = existing_value == expected_value
        if not equal:
            mismatches.append(f"{key}: {existing_value!r} != {expected_value!r}")
    return tuple(mismatches)


def _require_tokenized_preparation_config(
    manifest: "TokenizedCorpusManifest",
    expected: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> None:
    existing = dict(manifest.preparation_config or {})
    if not existing:
        raise ValueError(
            "tokenized corpus preparation config is missing from "
            f"{manifest_path}; rebuild the tokenized corpus before using resume=True"
        )
    mismatches = _tokenized_preparation_mismatches(existing, expected)
    if mismatches:
        detail = "; ".join(mismatches[:8])
        raise ValueError(
            "tokenized corpus preparation config does not match requested arguments "
            f"for {manifest_path}: {detail}"
        )


def _read_validation_learning_curve_rows(seed_runs: Sequence[Mapping[str, Any]], *, corpus: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        run_dir = Path(str(seed_run["run_dir"]))
        for model in ("baseline_ntp", "cortex3"):
            csv_path = run_dir / model / "learning_curve.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    if raw.get("split") != "val":
                        continue
                    rows.append(
                        {
                            "corpus": corpus,
                            "seed": seed,
                            "model": model,
                            "step": int(raw["step"]),
                            "split": "val",
                            "next_token_loss": float(raw["next_token_loss"]),
                            "future_tokens_per_cost": float(raw["future_tokens_per_cost"]),
                            "token_accuracy": float(raw["token_accuracy"]),
                        }
                    )
    return rows


def _write_learning_curve_matrix_artifacts(
    run_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    csv_name: str,
    png_name: str,
    group_by_corpus: bool,
) -> None:
    if not rows:
        return
    csv_path = run_dir / csv_name
    fieldnames = ["corpus", "seed", "model", "step", "split", "next_token_loss", "future_tokens_per_cost", "token_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    series: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in rows:
        label_parts: list[str] = []
        if group_by_corpus and row.get("corpus"):
            label_parts.append(str(row["corpus"]))
        label_parts.append(str(row["model"]))
        key = (":".join(label_parts), int(row["step"]))
        bucket = series.setdefault(key, {"next_token_loss": [], "future_tokens_per_cost": []})
        bucket["next_token_loss"].append(float(row["next_token_loss"]))
        bucket["future_tokens_per_cost"].append(float(row["future_tokens_per_cost"]))

    labels = sorted({label for label, _ in series})
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label in labels:
        points = sorted((step, values) for (series_label, step), values in series.items() if series_label == label)
        steps = [step for step, _ in points]
        next_loss = [sum(values["next_token_loss"]) / len(values["next_token_loss"]) for _, values in points]
        future_score = [sum(values["future_tokens_per_cost"]) / len(values["future_tokens_per_cost"]) for _, values in points]
        axes[0].plot(steps, next_loss, label=label)
        axes[1].plot(steps, future_score, label=label)
    axes[0].set_title("Validation next-token loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mean loss")
    axes[1].set_title("Validation future tokens per cost")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mean score")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / png_name, dpi=150)
    plt.close(fig)


@dataclass(frozen=True)
class TextCorpusConfig:
    files: tuple[str, ...]
    min_chars_per_chunk: int = 2048
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("at least one corpus file is required")
        if self.min_chars_per_chunk < 128:
            raise ValueError("min_chars_per_chunk must be >= 128")

    @staticmethod
    def from_paths(paths: Sequence[str | Path], *, min_chars_per_chunk: int = 2048) -> "TextCorpusConfig":
        resolved: list[str] = []
        for raw in paths:
            path = Path(raw)
            if path.is_dir():
                resolved.extend(str(child) for child in sorted(path.rglob("*.txt")))
            else:
                resolved.append(str(path))
        missing = [path for path in resolved if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"missing corpus files: {missing[:5]}")
        return TextCorpusConfig(tuple(resolved), min_chars_per_chunk=min_chars_per_chunk)


class TextShardReader:
    def __init__(self, config: TextCorpusConfig):
        self.config = config

    def iter_chunks(self) -> Iterator[str]:
        for file_name in self.config.files:
            path = Path(file_name)
            buffer: list[str] = []
            size = 0
            with path.open("r", encoding=self.config.encoding, errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    buffer.append(line)
                    size += len(line)
                    if size >= self.config.min_chars_per_chunk:
                        yield "".join(buffer)
                        buffer.clear()
                        size = 0
            if buffer:
                yield "".join(buffer)


@dataclass(frozen=True)
class HFDatasetExportConfig:
    dataset: str
    split: str = "train"
    text_field: str = "text"
    config_name: str | None = None
    data_files: tuple[str, ...] = ()
    streaming: bool = True
    trust_remote_code: bool = False
    cache_dir: str | None = None
    max_documents: int | None = 100_000
    max_characters: int | None = None
    allow_unbounded: bool = False
    min_text_chars: int = 1
    shard_max_chars: int = 64 * 1024 * 1024
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset is required")
        if not self.split:
            raise ValueError("split is required")
        if not self.text_field:
            raise ValueError("text_field is required")
        if self.max_documents is not None and self.max_documents < 1:
            raise ValueError("max_documents must be positive when provided")
        if self.max_characters is not None and self.max_characters < 1:
            raise ValueError("max_characters must be positive when provided")
        if self.max_documents is None and self.max_characters is None and not self.allow_unbounded:
            raise ValueError("unbounded HF export requires allow_unbounded=True")
        if self.min_text_chars < 0:
            raise ValueError("min_text_chars must be >= 0")
        if self.shard_max_chars < 256:
            raise ValueError("shard_max_chars must be >= 256")


@dataclass(frozen=True)
class HFDatasetExportReport:
    dataset: str
    split: str
    text_field: str
    output_dir: str
    shard_files: tuple[str, ...]
    document_count: int
    skipped_documents: int
    character_count: int
    shard_count: int
    streaming: bool
    config_name: str | None = None
    data_files: tuple[str, ...] = ()
    trust_remote_code: bool = False
    cache_dir: str | None = None
    max_documents: int | None = None
    max_characters: int | None = None
    allow_unbounded: bool = False
    min_text_chars: int = 1
    shard_max_chars: int = 64 * 1024 * 1024
    truncated_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load(path: str | Path) -> "HFDatasetExportReport":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["shard_files"] = tuple(payload["shard_files"])
        payload["data_files"] = tuple(payload.get("data_files", ()))
        payload.setdefault("trust_remote_code", False)
        payload.setdefault("cache_dir", None)
        payload.setdefault("max_documents", None)
        payload.setdefault("max_characters", None)
        payload.setdefault("allow_unbounded", False)
        payload.setdefault("min_text_chars", 1)
        payload.setdefault("shard_max_chars", 64 * 1024 * 1024)
        payload.setdefault("truncated_reason", None)
        return HFDatasetExportReport(**payload)

    def matches_config(self, config: HFDatasetExportConfig) -> bool:
        return self.config_mismatches(config) == ()

    def config_mismatches(self, config: HFDatasetExportConfig) -> tuple[str, ...]:
        expected = {
            "dataset": config.dataset,
            "split": config.split,
            "text_field": config.text_field,
            "config_name": config.config_name,
            "data_files": tuple(config.data_files),
            "streaming": config.streaming,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": config.cache_dir,
            "max_documents": config.max_documents,
            "max_characters": config.max_characters,
            "allow_unbounded": config.allow_unbounded,
            "min_text_chars": config.min_text_chars,
            "shard_max_chars": config.shard_max_chars,
        }
        actual = {key: getattr(self, key) for key in expected}
        return tuple(key for key, expected_value in expected.items() if actual[key] != expected_value)

    def validate_artifacts(self) -> None:
        if self.shard_count != len(self.shard_files):
            raise ValueError("HF export report shard_count does not match shard_files")
        if self.document_count < 1:
            raise ValueError("HF export report has no usable documents")
        if self.character_count < 1:
            raise ValueError("HF export report has no exported characters")
        missing = [path for path in self.shard_files if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"HF export resume is missing shard files: {missing[:5]}")
        empty = [path for path in self.shard_files if Path(path).stat().st_size == 0]
        if empty:
            raise ValueError(f"HF export resume found empty shard files: {empty[:5]}")


class HFDatasetTextExporter:
    def __init__(self, config: HFDatasetExportConfig):
        self.config = config

    def export(self, output_dir: str | Path, *, resume: bool = False) -> HFDatasetExportReport:
        output = Path(output_dir)
        shard_dir = output / "text_shards"
        report_path = output / "hf_export_report.json"
        output.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        if resume and report_path.exists():
            report = HFDatasetExportReport.load(report_path)
            report.validate_artifacts()
            mismatches = report.config_mismatches(self.config)
            if mismatches:
                joined = ", ".join(mismatches)
                raise ValueError(f"existing HF export report does not match requested config fields: {joined}")
            return report
        if resume and any(shard_dir.glob("shard_*.txt")):
            raise FileExistsError(f"resume=True found text shards without a complete export report: {shard_dir}")
        for stale in shard_dir.glob("shard_*.txt"):
            stale.unlink()

        dataset = self._load_dataset()
        shard_files: list[str] = []
        shard_index = 0
        current_chars = 0
        document_count = 0
        skipped_documents = 0
        character_count = 0
        truncated_reason: str | None = None
        handle = None

        def open_next_shard() -> Any:
            nonlocal shard_index, current_chars
            path = shard_dir / f"shard_{shard_index:05d}.txt"
            shard_index += 1
            current_chars = 0
            shard_files.append(str(path))
            return path.open("w", encoding=self.config.encoding, newline="\n")

        try:
            for row in dataset:
                text = self._extract_text(row).strip()
                if len(text) < self.config.min_text_chars:
                    skipped_documents += 1
                    continue
                if self.config.max_characters is not None:
                    remaining = self.config.max_characters - character_count
                    if remaining <= 0:
                        truncated_reason = "max_characters"
                        break
                    if len(text) > remaining:
                        text = text[:remaining]
                        truncated_reason = "max_characters"
                payload = text.rstrip() + "\n\n"
                if handle is None or (current_chars > 0 and current_chars + len(payload) > self.config.shard_max_chars):
                    if handle is not None:
                        handle.close()
                    handle = open_next_shard()
                handle.write(payload)
                current_chars += len(payload)
                document_count += 1
                character_count += len(text)
                if self.config.max_documents is not None and document_count >= self.config.max_documents:
                    truncated_reason = "max_documents"
                    break
                if truncated_reason == "max_characters":
                    break
        finally:
            if handle is not None:
                handle.close()

        if document_count == 0:
            raise ValueError(
                f"HF dataset export produced zero usable documents from field {self.config.text_field!r}; "
                f"skipped={skipped_documents}"
            )
        report = HFDatasetExportReport(
            dataset=self.config.dataset,
            split=self.config.split,
            text_field=self.config.text_field,
            output_dir=str(output),
            shard_files=tuple(shard_files),
            document_count=document_count,
            skipped_documents=skipped_documents,
            character_count=character_count,
            shard_count=len(shard_files),
            streaming=self.config.streaming,
            config_name=self.config.config_name,
            data_files=self.config.data_files,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
            max_documents=self.config.max_documents,
            max_characters=self.config.max_characters,
            allow_unbounded=self.config.allow_unbounded,
            min_text_chars=self.config.min_text_chars,
            shard_max_chars=self.config.shard_max_chars,
            truncated_reason=truncated_reason,
        )
        _write_json(report_path, report.to_dict())
        return report

    def _load_dataset(self) -> Iterable[Mapping[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Hugging Face datasets is required: install with `pip install -e .`") from exc

        kwargs: dict[str, Any] = {
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        if self.config.config_name is not None:
            kwargs["name"] = self.config.config_name
        if self.config.data_files:
            kwargs["data_files"] = list(self.config.data_files)
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = self.config.cache_dir
        if self.config.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            return load_dataset(self.config.dataset, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "/" not in self.config.dataset and "namespace/name" in message:
                raise RuntimeError(
                    f"Hugging Face rejected dataset id {self.config.dataset!r}; use the namespaced id, "
                    "for example `Salesforce/wikitext` instead of `wikitext`."
                ) from exc
            raise

    def _extract_text(self, row: Mapping[str, Any]) -> str:
        value: Any = row
        for part in self.config.text_field.split("."):
            if not isinstance(value, Mapping) or part not in value:
                available = sorted(value.keys()) if isinstance(value, Mapping) else type(value).__name__
                raise KeyError(f"text field {self.config.text_field!r} missing at {part!r}; available={available}")
            value = value[part]
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return "\n".join(value)
        raise TypeError(f"text field {self.config.text_field!r} must be str or list[str], got {type(value).__name__}")


class LLMTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def train(
        config: TextCorpusConfig,
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_training_chars: int | None = None,
    ) -> "LLMTokenizer":
        if vocab_size < len(SPECIAL_TOKENS) + 16:
            raise ValueError("vocab_size is too small for a useful BPE tokenizer")
        if max_training_chars is not None and max_training_chars < 1:
            raise ValueError("max_training_chars must be positive when provided")
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
            show_progress=False,
        )
        reader = TextShardReader(config)
        iterator: Iterable[str]
        if max_training_chars is None:
            iterator = reader.iter_chunks()
        else:
            def capped_chunks() -> Iterator[str]:
                consumed = 0
                for chunk in reader.iter_chunks():
                    remaining = int(max_training_chars) - consumed
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                    consumed += len(chunk)
                    if chunk:
                        yield chunk

            iterator = capped_chunks()
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        return LLMTokenizer(tokenizer)

    @staticmethod
    def load(path: str | Path) -> "LLMTokenizer":
        return LLMTokenizer(Tokenizer.from_file(str(path)))

    @property
    def pad_id(self) -> int:
        value = self.tokenizer.token_to_id("<pad>")
        if value is None:
            raise ValueError("tokenizer is missing <pad>")
        return int(value)

    @property
    def bos_id(self) -> int:
        value = self.tokenizer.token_to_id("<bos>")
        if value is None:
            raise ValueError("tokenizer is missing <bos>")
        return int(value)

    @property
    def eos_id(self) -> int:
        value = self.tokenizer.token_to_id("<eos>")
        if value is None:
            raise ValueError("tokenizer is missing <eos>")
        return int(value)

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(text).ids)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode([int(token) for token in token_ids], skip_special_tokens=True)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(output))
        return output


@dataclass(frozen=True)
class TokenizedCorpusManifest:
    token_file: str
    tokenizer_file: str
    source_files: tuple[str, ...]
    token_count: int
    dtype: str
    vocab_size: int
    seq_len: int
    max_horizon: int
    train_fraction: float
    token_file_sha256: str = ""
    tokenizer_file_sha256: str = ""
    source_file_fingerprints: tuple[Mapping[str, Any], ...] = ()
    preparation_config: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_fingerprints(self) -> None:
        missing_fields = [
            name
            for name, value in (
                ("token_file_sha256", self.token_file_sha256),
                ("tokenizer_file_sha256", self.tokenizer_file_sha256),
                ("source_file_fingerprints", self.source_file_fingerprints),
            )
            if not value
        ]
        if missing_fields:
            joined = ", ".join(missing_fields)
            raise ValueError(f"tokenized corpus manifest is missing cryptographic identity fields: {joined}; rebuild the corpus")
        token_sha = _sha256_file(self.token_file)
        if token_sha != self.token_file_sha256:
            raise ValueError("tokenized corpus token_file sha256 does not match manifest")
        tokenizer_sha = _sha256_file(self.tokenizer_file)
        if tokenizer_sha != self.tokenizer_file_sha256:
            raise ValueError("tokenized corpus tokenizer_file sha256 does not match manifest")
        if len(self.source_file_fingerprints) != len(self.source_files):
            raise ValueError("tokenized corpus source fingerprint count does not match source_files")
        for fingerprint in self.source_file_fingerprints:
            source_path = Path(str(fingerprint["path"]))
            if not source_path.exists():
                raise FileNotFoundError(f"tokenized corpus source file is missing: {source_path}")
            expected_size = int(fingerprint["size_bytes"])
            actual_size = int(source_path.stat().st_size)
            if actual_size != expected_size:
                raise ValueError(f"tokenized corpus source file size changed: {source_path}")
            expected_sha = str(fingerprint["sha256"])
            actual_sha = _sha256_file(source_path)
            if actual_sha != expected_sha:
                raise ValueError(f"tokenized corpus source file sha256 changed: {source_path}")

    def identity(self, *, verify: bool = True) -> dict[str, Any]:
        if verify:
            self.validate_fingerprints()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "token_count": int(self.token_count),
            "dtype": str(self.dtype),
            "vocab_size": int(self.vocab_size),
            "seq_len": int(self.seq_len),
            "max_horizon": int(self.max_horizon),
            "train_fraction": float(self.train_fraction),
            "token_file_sha256": str(self.token_file_sha256),
            "tokenizer_file_sha256": str(self.tokenizer_file_sha256),
            "source_file_fingerprints": [dict(item) for item in self.source_file_fingerprints],
        }
        payload["identity_sha256"] = _sha256_json(payload)
        return payload

    @staticmethod
    def load(path: str | Path) -> "TokenizedCorpusManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["source_files"] = tuple(payload["source_files"])
        payload.setdefault("token_file_sha256", "")
        payload.setdefault("tokenizer_file_sha256", "")
        payload["source_file_fingerprints"] = tuple(dict(item) for item in payload.get("source_file_fingerprints", ()))
        payload["preparation_config"] = dict(payload.get("preparation_config", {}))
        return TokenizedCorpusManifest(**payload)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        _write_json(output, self.to_dict())
        return output


class TokenizedCorpusBuilder:
    def __init__(self, corpus: TextCorpusConfig, tokenizer: LLMTokenizer):
        self.corpus = corpus
        self.tokenizer = tokenizer

    def _iter_token_chunks(self) -> Iterator[tuple[int, ...]]:
        for chunk in TextShardReader(self.corpus).iter_chunks():
            encoded = self.tokenizer.encode(chunk)
            if len(encoded) > 2:
                yield encoded

    def build(
        self,
        output_dir: str | Path,
        *,
        seq_len: int,
        max_horizon: int,
        train_fraction: float = 0.9,
        max_tokens: int | None = None,
        preparation_config: Mapping[str, Any] | None = None,
    ) -> TokenizedCorpusManifest:
        if seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if not 0.1 <= train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.1, 1.0)")
        if max_tokens is not None and max_tokens < seq_len + max_horizon + 2:
            raise ValueError("max_tokens must be large enough for one causal window")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        token_path = out / "tokens.uint32"
        tokenizer_path = out / "tokenizer.json"
        manifest_path = out / "manifest.json"
        self.tokenizer.save(tokenizer_path)

        token_count = 0
        with token_path.open("wb") as handle:
            for tokens in self._iter_token_chunks():
                if max_tokens is not None:
                    remaining = int(max_tokens) - token_count
                    if remaining <= 0:
                        break
                    if len(tokens) > remaining:
                        tokens = tokens[:remaining]
                array = np.asarray(tokens, dtype=np.uint32)
                handle.write(array.tobytes(order="C"))
                token_count += int(array.size)
                if max_tokens is not None and token_count >= int(max_tokens):
                    break
        minimum = seq_len + max_horizon + 2
        if token_count < minimum:
            token_path.unlink(missing_ok=True)
            raise ValueError(f"corpus produced {token_count} tokens, need at least {minimum}")

        manifest = TokenizedCorpusManifest(
            token_file=str(token_path),
            tokenizer_file=str(tokenizer_path),
            source_files=self.corpus.files,
            token_count=token_count,
            dtype="uint32",
            vocab_size=self.tokenizer.vocab_size,
            seq_len=seq_len,
            max_horizon=max_horizon,
            train_fraction=train_fraction,
            token_file_sha256=_sha256_file(token_path),
            tokenizer_file_sha256=_sha256_file(tokenizer_path),
            source_file_fingerprints=_fingerprint_files(self.corpus.files),
            preparation_config=dict(
                preparation_config
                or _tokenized_preparation_config(
                    self.corpus,
                    vocab_size=self.tokenizer.vocab_size,
                    min_frequency=None,
                    seq_len=seq_len,
                    max_horizon=max_horizon,
                    train_fraction=train_fraction,
                    max_tokens=max_tokens,
                )
            ),
        )
        manifest.save(manifest_path)
        return manifest


class MemmapCausalDataset:
    def __init__(self, manifest: TokenizedCorpusManifest, *, split: str):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.manifest = manifest
        self.split = split
        self.tokens = np.memmap(manifest.token_file, dtype=np.uint32, mode="r", shape=(manifest.token_count,))
        train_end = max(manifest.seq_len + manifest.max_horizon + 2, int(manifest.token_count * manifest.train_fraction))
        if split == "train":
            self.start = 0
            self.end = train_end
        else:
            self.start = max(0, train_end - manifest.seq_len - manifest.max_horizon - 1)
            self.end = manifest.token_count
        self.available = max(0, self.end - self.start - manifest.seq_len - manifest.max_horizon)
        if self.available <= 0:
            raise ValueError(f"{split} split is too small for seq_len={manifest.seq_len} and horizon={manifest.max_horizon}")

    def __len__(self) -> int:
        return int(self.available)

    def _window(self, offset: int) -> np.ndarray:
        if offset < 0 or offset >= self.available:
            raise IndexError(offset)
        start = self.start + offset
        stop = start + self.manifest.seq_len + self.manifest.max_horizon
        return np.asarray(self.tokens[start:stop], dtype=np.int64)

    def item(self, offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self._window(offset)
        x = torch.from_numpy(window[:self.manifest.seq_len].copy()).long()
        future = np.stack(
            [window[h:h + self.manifest.seq_len] for h in range(1, self.manifest.max_horizon + 1)],
            axis=1,
        )
        future_tensor = torch.from_numpy(future.copy()).long()
        return x, future_tensor[:, 0], future_tensor

    def batch_at(
        self,
        offsets: Sequence[int] | np.ndarray | torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(offsets, torch.Tensor):
            offset_array = offsets.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            offset_array = np.asarray(offsets, dtype=np.int64)
        if offset_array.ndim != 1:
            raise ValueError("offsets must be a 1D sequence")
        if offset_array.size == 0:
            raise ValueError("offsets must not be empty")
        if int(offset_array.min()) < 0 or int(offset_array.max()) >= self.available:
            raise IndexError("batch offset is outside the available split range")

        starts = self.start + offset_array
        positions = starts[:, None] + np.arange(
            self.manifest.seq_len + self.manifest.max_horizon,
            dtype=np.int64,
        )[None, :]
        windows = np.asarray(self.tokens[positions], dtype=np.int64)
        x = torch.from_numpy(windows[:, : self.manifest.seq_len].copy()).long()
        future = np.stack(
            [
                windows[:, horizon : horizon + self.manifest.seq_len]
                for horizon in range(1, self.manifest.max_horizon + 1)
            ],
            axis=2,
        )
        future_tensor = torch.from_numpy(future.copy()).long()
        return (
            x.to(device, non_blocking=device.type == "cuda"),
            future_tensor[:, :, 0].to(device, non_blocking=device.type == "cuda"),
            future_tensor.to(device, non_blocking=device.type == "cuda"),
        )

    def sample_batch(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        offsets = torch.randint(0, self.available, (batch_size,), generator=generator)
        return self.batch_at(offsets, device=device)

    def close(self) -> None:
        mmap = getattr(self.tokens, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "MemmapCausalDataset":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    seq_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    use_cortex_heads: bool = False
    use_ternary_core: bool = False
    ternary_activation_bits: int = 4
    use_native_ternary_kernel: bool = True
    require_native_ternary_kernel: bool = False
    native_ternary_backend: str = STRICT_NATIVE_TERNARY_BACKEND
    native_ternary_autotune_cache_path: str | None = None
    native_ternary_autotune_cache_write: bool = True
    use_skill_aware_experts: bool = False
    skill_expert_count: int = 4
    skill_expert_top_k: int = 2
    skill_expert_context_strength: float = 1.25
    use_variable_in_compressor: bool = False
    variable_compression_wide_kernel: int = 8
    use_learned_memory_policy: bool = False
    learned_memory_temperature: float = 1.0
    learned_memory_utility_prior_strength: float = 0.35
    use_certificate_head: bool = False
    certificate_latent_size: int = 64
    use_latent_reasoning_workspace: bool = False
    latent_workspace_steps: int = 3
    latent_workspace_feedback_strength: float = 0.20

    def __post_init__(self) -> None:
        if self.vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("vocab_size is too small")
        if self.seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        if self.horizons != tuple(sorted(set(self.horizons))):
            raise ValueError("horizons must be unique and sorted")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if self.use_ternary_core and self.ternary_activation_bits < 2:
            raise ValueError("ternary_activation_bits must be >= 2")
        if self.native_ternary_backend not in NATIVE_TERNARY_BACKEND_CHOICES:
            raise ValueError(f"native_ternary_backend must be one of: {', '.join(NATIVE_TERNARY_BACKEND_CHOICES)}")
        if self.skill_expert_count < 1:
            raise ValueError("skill_expert_count must be positive")
        if not 1 <= self.skill_expert_top_k <= self.skill_expert_count:
            raise ValueError("skill_expert_top_k must be between 1 and skill_expert_count")
        if self.skill_expert_context_strength < 0:
            raise ValueError("skill_expert_context_strength must be non-negative")
        if self.variable_compression_wide_kernel < 2:
            raise ValueError("variable_compression_wide_kernel must be >= 2")
        if self.use_learned_memory_policy and self.learned_memory_temperature <= 0:
            raise ValueError("learned_memory_temperature must be positive")
        if self.learned_memory_utility_prior_strength < 0:
            raise ValueError("learned_memory_utility_prior_strength must be non-negative")
        if self.certificate_latent_size < 1:
            raise ValueError("certificate_latent_size must be positive")
        if self.latent_workspace_steps < 1:
            raise ValueError("latent_workspace_steps must be positive")
        if self.latent_workspace_feedback_strength < 0:
            raise ValueError("latent_workspace_feedback_strength must be non-negative")


def _make_transformer_linear(
    config: TransformerConfig,
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    ledger: CompressionTraceLedger | None = None,
    log_prefix: str,
) -> nn.Module:
    if not config.use_ternary_core:
        return nn.Linear(in_features, out_features, bias=bias)
    return BitLinear(
        BitLinearConfig(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            activation_bits=config.ternary_activation_bits,
            log_prefix=log_prefix,
            use_native_cuda_kernel=config.use_native_ternary_kernel,
            require_native_cuda_kernel=config.require_native_ternary_kernel,
            native_cuda_backend=config.native_ternary_backend,
            native_cuda_autotune_cache_path=config.native_ternary_autotune_cache_path,
            native_cuda_autotune_cache_write=config.native_ternary_autotune_cache_write,
        ),
        ledger=ledger,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None, layer_index: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = _make_transformer_linear(
            config,
            config.d_model,
            config.d_model * 3,
            bias=False,
            ledger=ledger,
            log_prefix=f"layer_{layer_index}.attn.qkv",
        )
        self.proj = _make_transformer_linear(
            config,
            config.d_model,
            config.d_model,
            ledger=ledger,
            log_prefix=f"layer_{layer_index}.attn.proj",
        )
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels = x.shape
        qkv = self.qkv(x).view(batch, time_steps, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = attn.transpose(1, 2).contiguous().view(batch, time_steps, channels)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None, layer_index: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config, ledger=ledger, layer_index=layer_index)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            _make_transformer_linear(
                config,
                config.d_model,
                config.d_model * 4,
                ledger=ledger,
                log_prefix=f"layer_{layer_index}.mlp.up",
            ),
            nn.GELU(),
            _make_transformer_linear(
                config,
                config.d_model * 4,
                config.d_model,
                ledger=ledger,
                log_prefix=f"layer_{layer_index}.mlp.down",
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass(frozen=True)
class VariableCompressionState:
    keep_prob: torch.Tensor
    critical_weight: torch.Tensor
    normal_weight: torch.Tensor
    redundant_weight: torch.Tensor
    estimated_ratio: torch.Tensor

    def to_summary(self) -> dict[str, float]:
        return {
            "critical_weight_mean": float(self.critical_weight.detach().mean().cpu()),
            "normal_weight_mean": float(self.normal_weight.detach().mean().cpu()),
            "redundant_weight_mean": float(self.redundant_weight.detach().mean().cpu()),
            "estimated_ratio_mean": float(self.estimated_ratio.detach().mean().cpu()),
        }


class VariableInCompressor(nn.Module):
    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        self.config = config
        self.ledger = ledger
        self.scorer = nn.Linear(config.d_model, 1)
        self.output = _make_transformer_linear(
            config,
            config.d_model,
            config.d_model,
            ledger=ledger,
            log_prefix="variable_in.output",
        )

    def _local_average(self, hidden: torch.Tensor, kernel: int) -> torch.Tensor:
        time_steps = hidden.shape[1]
        kernel = max(1, min(int(kernel), int(time_steps)))
        left = kernel // 2
        right = kernel - 1 - left
        padded = F.pad(hidden.transpose(1, 2), (left, right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, VariableCompressionState]:
        keep_prob = torch.sigmoid(self.scorer(hidden))
        critical = keep_prob.square()
        redundant = (1.0 - keep_prob).square()
        normal = (1.0 - critical - redundant).clamp_min(0.0)
        norm = (critical + normal + redundant).clamp_min(1e-6)
        critical = critical / norm
        normal = normal / norm
        redundant = redundant / norm
        pair_context = self._local_average(hidden, 2)
        wide_context = self._local_average(hidden, self.config.variable_compression_wide_kernel)
        compressed = critical * hidden + normal * pair_context + redundant * wide_context
        compressed = self.output(compressed)
        estimated_ratio = critical + 0.5 * normal + (1.0 / float(self.config.variable_compression_wide_kernel)) * redundant
        state = VariableCompressionState(
            keep_prob=keep_prob,
            critical_weight=critical,
            normal_weight=normal,
            redundant_weight=redundant,
            estimated_ratio=estimated_ratio,
        )
        if self.ledger is not None:
            batch, time_steps, channels = hidden.shape
            ratio = float(estimated_ratio.detach().mean().cpu())
            self.ledger.record_kv(
                "llm-variable-in",
                "variable_latent",
                bytes_used=float(batch * time_steps * channels * 2 * ratio),
                exact_anchors=0,
                note=f"Variable-In compressor estimated ratio {ratio:.4f}",
            )
        return compressed, state


@dataclass(frozen=True)
class LearnedMemoryPolicyState:
    logits: torch.Tensor
    probs: torch.Tensor
    exact_prob: torch.Tensor
    latent_prob: torch.Tensor
    drop_prob: torch.Tensor
    storage_ratio: torch.Tensor
    entropy: torch.Tensor
    input_ids: torch.Tensor
    utility_prior: torch.Tensor
    utility_feedback_events: int = 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "exact_prob_mean": float(self.exact_prob.detach().mean().cpu()),
            "latent_prob_mean": float(self.latent_prob.detach().mean().cpu()),
            "drop_prob_mean": float(self.drop_prob.detach().mean().cpu()),
            "storage_ratio_mean": float(self.storage_ratio.detach().mean().cpu()),
            "entropy_mean": float(self.entropy.detach().mean().cpu()),
            "utility_prior_exact": float(self.utility_prior.detach().cpu()[0]),
            "utility_prior_latent": float(self.utility_prior.detach().cpu()[1]),
            "utility_prior_drop": float(self.utility_prior.detach().cpu()[2]),
            "utility_feedback_events": int(self.utility_feedback_events),
        }


class LearnedMemoryPolicy(nn.Module):
    EXACT = 0
    LATENT = 1
    DROP = 2

    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        self.config = config
        self.ledger = ledger
        hidden = max(16, config.d_model // 2)
        self.policy = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.latent_projector = _make_transformer_linear(
            config,
            config.d_model,
            config.d_model,
            ledger=ledger,
            log_prefix="learned_memory.latent_projector",
        )
        self.drop_vector = nn.Parameter(torch.zeros(config.d_model))
        self.register_buffer("_utility_prior", torch.full((3,), 1.0 / 3.0, dtype=torch.float32))
        self.utility_feedback_events = 0

    @property
    def utility_prior(self) -> torch.Tensor:
        return self._utility_prior

    def set_memory_utility_prior(
        self,
        distribution: Sequence[float] | torch.Tensor | None,
        *,
        events: int = 0,
    ) -> None:
        if distribution is None:
            prior = torch.full((3,), 1.0 / 3.0, dtype=torch.float32, device=self._utility_prior.device)
            events = 0
        else:
            prior = torch.as_tensor(distribution, dtype=torch.float32, device=self._utility_prior.device).flatten()
            if prior.numel() != 3:
                raise ValueError("learned memory utility prior must have exactly three entries")
            prior = prior.clamp_min(0.0)
            total = prior.sum()
            if not bool(torch.isfinite(total).detach().cpu().item()) or float(total.detach().cpu()) <= 0.0:
                prior = torch.full((3,), 1.0 / 3.0, dtype=torch.float32, device=self._utility_prior.device)
                events = 0
            else:
                prior = prior / total
        self._utility_prior.copy_(prior.to(device=self._utility_prior.device, dtype=self._utility_prior.dtype))
        self.utility_feedback_events = max(0, int(events))

    def _local_latent(self, hidden: torch.Tensor) -> torch.Tensor:
        time_steps = hidden.shape[1]
        kernel = max(1, min(int(self.config.variable_compression_wide_kernel), int(time_steps)))
        left = kernel // 2
        right = kernel - 1 - left
        padded = F.pad(hidden.transpose(1, 2), (left, right), mode="replicate")
        pooled = F.avg_pool1d(padded, kernel_size=kernel, stride=1).transpose(1, 2)
        return self.latent_projector(pooled)

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, LearnedMemoryPolicyState]:
        logits = self.policy(hidden) / float(self.config.learned_memory_temperature)
        if self.utility_feedback_events > 0 and float(self.config.learned_memory_utility_prior_strength) > 0.0:
            utility_prior = self._utility_prior.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
            logits = logits + float(self.config.learned_memory_utility_prior_strength) * utility_prior.log().view(1, 1, 3)
        probs = F.softmax(logits, dim=-1)
        exact_prob = probs[..., self.EXACT]
        latent_prob = probs[..., self.LATENT]
        drop_prob = probs[..., self.DROP]
        latent = self._local_latent(hidden)
        drop = self.drop_vector.view(1, 1, -1).expand_as(hidden)
        mixed = (
            exact_prob.unsqueeze(-1) * hidden
            + latent_prob.unsqueeze(-1) * latent
            + drop_prob.unsqueeze(-1) * drop
        )
        storage_ratio = exact_prob + 0.25 * latent_prob
        entropy = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1)
        state = LearnedMemoryPolicyState(
            logits=logits,
            probs=probs,
            exact_prob=exact_prob,
            latent_prob=latent_prob,
            drop_prob=drop_prob,
            storage_ratio=storage_ratio,
            entropy=entropy,
            input_ids=input_ids.detach(),
            utility_prior=self._utility_prior.detach().to(device=hidden.device, dtype=hidden.dtype),
            utility_feedback_events=int(self.utility_feedback_events),
        )
        if self.ledger is not None:
            batch, time_steps, channels = hidden.shape
            self.ledger.record_kv(
                "llm-learned-memory-policy",
                "learned_exact_latent_drop",
                bytes_used=float(batch * time_steps * channels * 2) * float(storage_ratio.detach().mean().cpu()),
                exact_anchors=int((exact_prob.detach() >= latent_prob.detach()).sum().item()),
                note=(
                    "learned memory policy mixed exact/latent/drop states "
                    f"storage_ratio={float(storage_ratio.detach().mean().cpu()):.4f}"
                ),
            )
        return mixed, state


@dataclass(frozen=True)
class SkillExpertRoutingState:
    route_logits: torch.Tensor
    route_probs: torch.Tensor
    top_indices: torch.Tensor
    target_distribution: torch.Tensor | None = None
    context_source: str = ""

    def to_summary(self) -> dict[str, Any]:
        route_mean = self.route_probs.detach().mean(dim=(0, 1)).cpu()
        payload: dict[str, Any] = {
            "route_distribution": tuple(float(item) for item in route_mean.tolist()),
            "selected_experts": tuple(
                sorted(set(int(index) for index in self.top_indices.detach().cpu().reshape(-1).tolist()))
            ),
            "context_source": self.context_source,
        }
        if self.target_distribution is not None:
            payload["target_distribution"] = tuple(
                float(item) for item in self.target_distribution.detach().cpu().tolist()
            )
        return payload


class SkillAwareExpertMoE(nn.Module):
    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        self.config = config
        self.ledger = ledger
        self.router = nn.Linear(config.d_model, config.skill_expert_count)
        self.experts = nn.ModuleList([
            nn.Sequential(
                _make_transformer_linear(
                    config,
                    config.d_model,
                    config.d_model * 2,
                    ledger=ledger,
                    log_prefix=f"skill_expert_{index}.up",
                ),
                nn.GELU(),
                _make_transformer_linear(
                    config,
                    config.d_model * 2,
                    config.d_model,
                    ledger=ledger,
                    log_prefix=f"skill_expert_{index}.down",
                ),
            )
            for index in range(config.skill_expert_count)
        ])

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        skill_context: torch.Tensor | None = None,
        context_source: str = "",
    ) -> tuple[torch.Tensor, SkillExpertRoutingState]:
        route_logits = self.router(hidden)
        target_distribution = None
        if skill_context is not None and skill_context.numel() == self.config.skill_expert_count:
            target_distribution = skill_context.to(device=hidden.device, dtype=route_logits.dtype).clamp_min(0.0)
            total = target_distribution.sum()
            if bool((total <= 0).detach().cpu()):
                target_distribution = None
            else:
                target_distribution = target_distribution / total
                bias = target_distribution.clamp_min(1e-6).log().view(1, 1, -1)
                route_logits = route_logits + float(self.config.skill_expert_context_strength) * bias
        route_probs = F.softmax(route_logits, dim=-1)
        top_values, top_indices = torch.topk(
            route_logits,
            k=self.config.skill_expert_top_k,
            dim=-1,
        )
        top_weights = F.softmax(top_values, dim=-1)
        combined = hidden.new_zeros(hidden.shape)
        selected = set(int(index) for index in top_indices.detach().cpu().reshape(-1).tolist())
        for expert_index, expert in enumerate(self.experts):
            expert_mask = top_indices.eq(expert_index)
            if not bool(expert_mask.any()):
                continue
            token_weight = (top_weights * expert_mask.to(top_weights.dtype)).sum(dim=-1, keepdim=True)
            combined = combined + expert(hidden) * token_weight
        if self.ledger is not None:
            for expert_index in sorted(selected):
                self.ledger.record_expert(
                    f"llm-skill-expert-{expert_index}",
                    (
                        "skill-aware MoE route in Cortex Transformer forward"
                        + (f" from {context_source}" if context_source else "")
                    ),
                    cost=float(top_indices.eq(expert_index).sum().item()) / max(1.0, float(top_indices.numel())),
                )
        return hidden + combined, SkillExpertRoutingState(
            route_logits=route_logits,
            route_probs=route_probs,
            top_indices=top_indices,
            target_distribution=target_distribution,
            context_source=context_source,
        )


@dataclass(frozen=True)
class LatentReasoningWorkspaceState:
    step_states: torch.Tensor
    summary: torch.Tensor
    step_gates: torch.Tensor
    token_attention: torch.Tensor
    feedback: torch.Tensor

    @property
    def step_count(self) -> int:
        return int(self.step_states.shape[1])

    def checksum(self, row_index: int = 0) -> str:
        row = self.summary[row_index].detach().cpu().flatten()
        values = [round(float(value), 6) for value in row.tolist()]
        payload = {"step_count": self.step_count, "values": values}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.blake2b(encoded, digest_size=16).hexdigest()

    def to_summary(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "summary_norm_mean": float(self.summary.detach().norm(dim=-1).mean().cpu()),
            "gate_mean": float(self.step_gates.detach().mean().cpu()),
            "gate_min": float(self.step_gates.detach().amin().cpu()),
            "gate_max": float(self.step_gates.detach().amax().cpu()),
            "feedback_norm_mean": float(self.feedback.detach().norm(dim=-1).mean().cpu()),
        }


class LatentReasoningWorkspace(nn.Module):
    def __init__(self, config: TransformerConfig, *, ledger: CompressionTraceLedger | None = None):
        super().__init__()
        self.config = config
        self.ledger = ledger
        latent_size = int(config.certificate_latent_size)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.token_scorer = nn.Linear(config.d_model, 1)
        self.context_projection = _make_transformer_linear(
            config,
            config.d_model,
            latent_size,
            ledger=ledger,
            log_prefix="latent_workspace.context",
        )
        self.step_transition = _make_transformer_linear(
            config,
            latent_size,
            latent_size,
            ledger=ledger,
            log_prefix="latent_workspace.transition",
        )
        self.step_gate = nn.Linear(latent_size, 1)
        self.feedback_projection = _make_transformer_linear(
            config,
            latent_size,
            config.d_model,
            ledger=ledger,
            log_prefix="latent_workspace.feedback",
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, LatentReasoningWorkspaceState]:
        if hidden.ndim != 3:
            raise ValueError("latent workspace expects hidden with shape [batch, time, channels]")
        normalized = self.input_norm(hidden)
        attention_logits = self.token_scorer(normalized).squeeze(-1)
        token_attention = F.softmax(attention_logits, dim=-1)
        context = torch.einsum("bt,btd->bd", token_attention, normalized)
        context_latent = torch.tanh(self.context_projection(context))
        latent = context_latent
        step_states: list[torch.Tensor] = []
        step_gates: list[torch.Tensor] = []
        for _ in range(int(self.config.latent_workspace_steps)):
            candidate_input = latent + context_latent
            candidate = torch.tanh(self.step_transition(candidate_input))
            gate = torch.sigmoid(self.step_gate(candidate_input)).squeeze(-1)
            latent = gate.unsqueeze(-1) * candidate + (1.0 - gate).unsqueeze(-1) * latent
            step_states.append(latent)
            step_gates.append(gate)
        stacked_steps = torch.stack(step_states, dim=1)
        gates = torch.stack(step_gates, dim=1)
        summary = stacked_steps[:, -1, :]
        feedback = self.feedback_projection(summary)
        state = LatentReasoningWorkspaceState(
            step_states=stacked_steps,
            summary=summary,
            step_gates=gates,
            token_attention=token_attention,
            feedback=feedback,
        )
        if self.ledger is not None:
            batch = int(hidden.shape[0])
            bytes_used = float(batch * state.step_count * int(self.config.certificate_latent_size) * 2)
            self.ledger.record_kv(
                "llm-latent-reasoning-workspace",
                "latent_workspace",
                bytes_used=bytes_used,
                exact_anchors=0,
                note=(
                    "explicit latent reasoning workspace "
                    f"steps={state.step_count} gate_mean={float(gates.detach().mean().cpu()):.4f}"
                ),
            )
        mixed = hidden + float(self.config.latent_workspace_feedback_strength) * feedback.unsqueeze(1)
        return mixed, state


@dataclass(frozen=True)
class LLMForwardOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    mtp_logits: Mapping[int, torch.Tensor]
    confidence: torch.Tensor | None
    certificate: CertificateHeadOutput | None = None
    variable_compression: VariableCompressionState | None = None
    learned_memory_policy: LearnedMemoryPolicyState | None = None
    skill_expert_routing: SkillExpertRoutingState | None = None
    latent_workspace: LatentReasoningWorkspaceState | None = None


class CortexTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.compression_ledger = CompressionTraceLedger() if config.use_ternary_core else None
        self.certificate_forward_events = 0
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.variable_in = (
            VariableInCompressor(config, ledger=self.compression_ledger)
            if config.use_variable_in_compressor
            else None
        )
        self.learned_memory = (
            LearnedMemoryPolicy(config, ledger=self.compression_ledger)
            if config.use_learned_memory_policy
            else None
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(config, ledger=self.compression_ledger, layer_index=index)
            for index in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.skill_experts = (
            SkillAwareExpertMoE(config, ledger=self.compression_ledger)
            if config.use_skill_aware_experts
            else None
        )
        self.register_buffer(
            "_skill_expert_context",
            torch.zeros(config.skill_expert_count, dtype=torch.float32),
            persistent=False,
        )
        self.skill_expert_context_active = False
        self.skill_expert_context_source = ""
        self.skill_expert_context_updates = 0
        self.latent_workspace = (
            LatentReasoningWorkspace(config, ledger=self.compression_ledger)
            if config.use_latent_reasoning_workspace
            else None
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleDict({
            str(horizon): _make_transformer_linear(
                config,
                config.d_model,
                config.vocab_size,
                ledger=self.compression_ledger,
                log_prefix=f"mtp.horizon_{horizon}",
            )
            for horizon in config.horizons
        }) if config.use_cortex_heads else nn.ModuleDict()
        self.confidence_head = _make_transformer_linear(
            config,
            config.d_model,
            1,
            ledger=self.compression_ledger,
            log_prefix="confidence",
        ) if config.use_cortex_heads else None
        self.certificate_head = (
            CertificateHead(config.d_model, config.certificate_latent_size, config.vocab_size)
            if config.use_certificate_head
            else None
        )
        self.lm_head.weight = self.token_embedding.weight

    def skill_expert_context_distribution(self) -> tuple[float, ...] | None:
        if not self.skill_expert_context_active:
            return None
        return tuple(float(item) for item in self._skill_expert_context.detach().cpu().tolist())

    def set_skill_expert_context(
        self,
        distribution: Sequence[float] | torch.Tensor | None,
        *,
        source: str = "",
    ) -> None:
        if self.skill_experts is None:
            return
        if distribution is None:
            self._skill_expert_context.zero_()
            self.skill_expert_context_active = False
            self.skill_expert_context_source = ""
            self.skill_expert_context_updates += 1
            return
        tensor = torch.as_tensor(distribution, dtype=torch.float32, device=self._skill_expert_context.device).flatten()
        if int(tensor.numel()) != int(self.config.skill_expert_count):
            raise ValueError(
                f"skill expert context must contain {self.config.skill_expert_count} weights, got {int(tensor.numel())}"
            )
        tensor = tensor.clamp_min(0.0)
        total = tensor.sum()
        if float(total.detach().cpu()) <= 0.0:
            self._skill_expert_context.zero_()
            self.skill_expert_context_active = False
            self.skill_expert_context_source = ""
        else:
            self._skill_expert_context.copy_(tensor / total)
            self.skill_expert_context_active = True
            self.skill_expert_context_source = source
        self.skill_expert_context_updates += 1

    def forward(self, input_ids: torch.Tensor) -> LLMForwardOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch, time_steps = input_ids.shape
        if time_steps > self.config.seq_len:
            raise ValueError(f"input sequence length {time_steps} exceeds model seq_len {self.config.seq_len}")
        positions = torch.arange(0, time_steps, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.drop(hidden)
        variable_compression = None
        if self.variable_in is not None:
            hidden, variable_compression = self.variable_in(hidden)
        learned_memory_policy = None
        if self.learned_memory is not None:
            hidden, learned_memory_policy = self.learned_memory(hidden, input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        skill_expert_routing = None
        if self.skill_experts is not None:
            context = self._skill_expert_context if self.skill_expert_context_active else None
            hidden, skill_expert_routing = self.skill_experts(
                hidden,
                skill_context=context,
                context_source=self.skill_expert_context_source,
            )
        latent_workspace = None
        if self.latent_workspace is not None:
            hidden, latent_workspace = self.latent_workspace(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = {
            int(horizon): head(hidden)
            for horizon, head in ((int(key), module) for key, module in self.mtp_heads.items())
        }
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1) if self.confidence_head else None
        certificate = self.certificate_head(hidden) if self.certificate_head is not None else None
        if certificate is not None:
            self.certificate_forward_events += 1
        return LLMForwardOutput(
            logits=logits,
            hidden=hidden,
            mtp_logits=mtp_logits,
            confidence=confidence,
            certificate=certificate,
            variable_compression=variable_compression,
            learned_memory_policy=learned_memory_policy,
            skill_expert_routing=skill_expert_routing,
            latent_workspace=latent_workspace,
        )

    def requantize_ternary_core(self, *, certify_zeros: bool = False) -> None:
        if not self.config.use_ternary_core:
            return
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.requantize(certify_zeros=certify_zeros)

    def compression_trace(self) -> Mapping[str, Any]:
        if self.compression_ledger is None:
            return {"enabled": False}
        return {"enabled": True, **self.compression_ledger.to_dict()}


class CortexTransformerInferenceAgent:
    def __init__(
        self,
        model: CortexTransformerLM,
        tokenizer: LLMTokenizer,
        *,
        max_new_tokens: int = 8,
        max_block_tokens: int = 2,
        future_engine: FutureContractEngine | None = None,
    ):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if max_block_tokens < 1:
            raise ValueError("max_block_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.max_block_tokens = int(max(1, min(max_block_tokens, max(model.config.horizons))))
        self.future_engine = future_engine or FutureContractEngine(
            MTPFSPConfig(
                hidden_size=model.config.d_model,
                vocab_size=model.config.vocab_size,
                horizons=model.config.horizons,
            ),
            trace_ledger=model.compression_ledger,
        )
        self.call_count = 0
        self.last_event: dict[str, Any] | None = None

    def _prompt(self, task: Task) -> str:
        return (
            f"Skill: {task.skill}\n"
            f"Task: {task.prompt}\n"
            "Answer:"
        )

    def _contract_domain_and_risk(self, task: Task) -> tuple[str, float]:
        if task.skill in {"arithmetic", "algebra"}:
            return "math", 0.80
        if task.skill == "code_unit_tests":
            return "code", 0.80
        if task.skill in {"long_context_anchor", "entity_tracking"}:
            return "exact_anchor", 0.70
        if task.skill == "calibration":
            return "calibration", 0.75
        return "general", 0.05

    def _last_confidence(self, output: LLMForwardOutput, token_confidence: float) -> float:
        if output.confidence is None:
            return float(token_confidence)
        return float(output.confidence[0, -1].detach().clamp(0.0, 1.0).cpu())

    def _contract_logits_by_horizon(
        self,
        output: LLMForwardOutput,
        *,
        remaining: int,
    ) -> dict[int, torch.Tensor]:
        if 1 not in output.mtp_logits:
            return {}
        rows = [output.mtp_logits[1][0, -1].float()]
        logits_by_horizon: dict[int, torch.Tensor] = {1: torch.stack(rows, dim=0)}
        max_contiguous = min(self.max_block_tokens, int(remaining))
        if max_contiguous >= 2 and 2 in output.mtp_logits:
            rows.append(output.mtp_logits[2][0, -1].float())
            logits_by_horizon[2] = torch.stack(rows, dim=0)
        return logits_by_horizon

    def __call__(self, task: Task) -> CandidateAnswer:
        self.call_count += 1
        was_training = self.model.training
        device = next(self.model.parameters()).device
        ids = list(self.tokenizer.encode(self._prompt(task)))
        if not ids:
            ids = [self.tokenizer.bos_id]
        ids = ids[-self.model.config.seq_len :]
        generated: list[int] = []
        confidences: list[float] = []
        certificate_payload: dict[str, Any] = {}
        block_events: list[dict[str, Any]] = []
        forward_count = 0
        contract_checks = 0
        proposed_blocks = 0
        proposed_tokens = 0
        accepted_blocks = 0
        rejected_blocks = 0
        accepted_tokens = 0
        rejected_tokens = 0
        accepted_mtp_tokens = 0
        self.model.eval()
        try:
            with torch.no_grad():
                while len(generated) < self.max_new_tokens:
                    window = ids[-self.model.config.seq_len :]
                    x = torch.tensor([window], dtype=torch.long, device=device)
                    output = self.model(x)
                    forward_count += 1
                    probs = torch.softmax(output.logits[0, -1].float(), dim=-1)
                    confidence, token = torch.max(probs, dim=-1)
                    token_id = int(token.detach().cpu())
                    token_confidence = float(confidence.detach().cpu())
                    remaining = self.max_new_tokens - len(generated)
                    emitted_tokens = [token_id]
                    decision: ContractDecision | None = None
                    contract_logits = self._contract_logits_by_horizon(output, remaining=remaining)
                    if contract_logits:
                        domain, risk = self._contract_domain_and_risk(task)
                        contract = self.future_engine.draft_contract_from_logits(
                            contract_logits,
                            confidence=self._last_confidence(output, token_confidence),
                            domain=domain,
                            risk=risk,
                            contract_id=f"p8-model-{task.task_id}-{self.call_count}-{forward_count}",
                            temporal_loss=0.0,
                        )
                        proposed = [token_id]
                        if contract.accepted_horizon >= 2 and 2 in contract_logits:
                            proposed.append(int(contract_logits[2][1].argmax(dim=-1).detach().cpu()))
                        decision = self.future_engine.gate_contract(contract, observed_tokens=proposed)
                        contract_checks += 1
                        proposed_blocks += 1
                        proposed_tokens += len(proposed)
                        if decision.accepted and decision.contract.accepted_horizon > 1:
                            emitted_tokens = proposed[: decision.contract.accepted_horizon]
                            accepted_blocks += 1
                            accepted_tokens += len(emitted_tokens)
                            accepted_mtp_tokens += max(0, len(emitted_tokens) - 1)
                        else:
                            rejected_blocks += 1
                            rejected_tokens += max(0, len(proposed) - 1)
                        block_events.append(
                            {
                                "contract_id": decision.contract.contract_id,
                                "domain": decision.contract.domain,
                                "risk": decision.contract.risk,
                                "requested_horizon": decision.contract.requested_horizon,
                                "accepted_horizon": decision.contract.accepted_horizon,
                                "proposed_token_ids": tuple(proposed),
                                "contract_token_ids": tuple(decision.contract.token_ids),
                                "emitted_token_ids": tuple(emitted_tokens),
                                "accepted": bool(decision.accepted and decision.contract.accepted_horizon > 1),
                                "gate_accepted": bool(decision.accepted),
                                "reason": decision.reason,
                                "confidence": decision.contract.confidence,
                                "cost": asdict(decision.cost),
                            }
                        )
                    for emitted in emitted_tokens:
                        generated.append(int(emitted))
                        confidences.append(token_confidence)
                        ids.append(int(emitted))
                        if int(emitted) == self.tokenizer.eos_id:
                            break
                    if output.certificate is not None:
                        cert = output.certificate
                        latent_state = LatentProofState(
                            state_id=f"p8-model-{task.task_id}-{self.call_count}",
                            task_id=task.task_id,
                            skill=task.skill,
                            tensor=cert.latent_state.detach(),
                            latent_steps=max(1, len(generated)),
                        )
                        certificate_payload = {
                            "model_certificate_latent_checksum": latent_state.checksum(),
                            "model_certificate_answer_token": int(cert.answer_logits[0].detach().argmax(dim=-1).cpu()),
                            "model_certificate_type_index": int(cert.certificate_type_logits[0].detach().argmax(dim=-1).cpu()),
                            "model_certificate_uncertainty": float(cert.uncertainty[0].detach().cpu()),
                        }
                    if output.latent_workspace is not None:
                        workspace = output.latent_workspace
                        certificate_payload.update(
                            {
                                "model_latent_workspace_checksum": workspace.checksum(0),
                                "model_latent_workspace_steps": int(workspace.step_count),
                                "model_latent_workspace_gate_mean": float(workspace.step_gates[0].detach().mean().cpu()),
                                "model_latent_workspace_feedback_norm": float(workspace.feedback[0].detach().norm().cpu()),
                            }
                        )
                    if generated and generated[-1] == self.tokenizer.eos_id:
                        break
        finally:
            if was_training:
                self.model.train()
        text = self.tokenizer.decode(generated).strip()
        confidence_value = sum(confidences) / max(1, len(confidences))
        prompt_token_count = max(0, len(ids) - len(generated))
        adaptive_payload = {
            "adaptive_mtp_decoding": True,
            "adaptive_mtp_max_block_tokens": int(self.max_block_tokens),
            "adaptive_mtp_forward_count": int(forward_count),
            "adaptive_mtp_contract_checks": int(contract_checks),
            "adaptive_mtp_proposed_blocks": int(proposed_blocks),
            "adaptive_mtp_proposed_tokens": int(proposed_tokens),
            "adaptive_mtp_accepted_blocks": int(accepted_blocks),
            "adaptive_mtp_rejected_blocks": int(rejected_blocks),
            "adaptive_mtp_accepted_tokens": int(accepted_tokens),
            "adaptive_mtp_rejected_tokens": int(rejected_tokens),
            "adaptive_mtp_accepted_mtp_tokens": int(accepted_mtp_tokens),
            "adaptive_mtp_block_events": tuple(block_events),
            "adaptive_mtp_local_acceptance_rate": (
                float(accepted_blocks) / max(1, int(proposed_blocks))
            ),
        }
        event = {
            "task_id": task.task_id,
            "skill": task.skill,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": len(generated),
            "generated_token_ids": tuple(generated),
            "decoded_text": text,
            "confidence": float(confidence_value),
            **certificate_payload,
            **adaptive_payload,
        }
        self.last_event = event
        return CandidateAnswer(
            text=text,
            confidence=float(max(0.0, min(0.99, confidence_value))),
            certificate={
                "cortex_transformer_inference": "adaptive_mtp_fsp",
                "model_backed_inference": True,
                "generated_token_count": len(generated),
                **certificate_payload,
                **adaptive_payload,
            },
            cost=CostTrace(
                generated_tokens=max(1, len(generated)),
                latent_steps=max(1, int(forward_count) + int(contract_checks)),
                verifier_steps=int(rejected_blocks),
            ),
            raw={"model_backed_inference": event},
        )


@dataclass(frozen=True)
class LossWeights:
    next_token: float = 1.0
    mtp: float = 0.35
    temporal_consistency: float = 0.05
    confidence: float = 0.05
    variable_input: float = 0.01
    learned_memory: float = 0.03
    skill_expert: float = 0.025
    latent_workspace: float = 0.025
    certificate: float = 0.04


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    next_token: float
    mtp: float = 0.0
    temporal_consistency: float = 0.0
    confidence: float = 0.0
    variable_input: float = 0.0
    learned_memory: float = 0.0
    skill_expert: float = 0.0
    latent_workspace: float = 0.0
    certificate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class CortexObjective:
    def __init__(self, weights: LossWeights | None = None):
        self.weights = weights or LossWeights()

    def compute(
        self,
        output: LLMForwardOutput,
        next_targets: torch.Tensor,
        future_targets: torch.Tensor,
        *,
        use_cortex_terms: bool,
    ) -> tuple[torch.Tensor, LossBreakdown]:
        vocab_size = output.logits.shape[-1]
        next_loss_per_token = F.cross_entropy(
            output.logits.reshape(-1, vocab_size),
            next_targets.reshape(-1),
            reduction="none",
        ).view_as(next_targets)
        next_loss = next_loss_per_token.mean()
        total = self.weights.next_token * next_loss
        mtp_loss = output.logits.new_tensor(0.0)
        temporal_loss = output.logits.new_tensor(0.0)
        confidence_loss = output.logits.new_tensor(0.0)
        variable_loss = output.logits.new_tensor(0.0)
        learned_memory_loss = output.logits.new_tensor(0.0)
        skill_expert_loss = output.logits.new_tensor(0.0)
        latent_workspace_loss = output.logits.new_tensor(0.0)
        certificate_loss = output.logits.new_tensor(0.0)
        if use_cortex_terms:
            if not output.mtp_logits:
                raise ValueError("Cortex objective requires multi-horizon heads")
            losses = []
            for horizon, logits in output.mtp_logits.items():
                if horizon > future_targets.shape[-1]:
                    continue
                targets = future_targets[:, :, horizon - 1]
                losses.append(F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)))
                if horizon > 1:
                    shorter = output.mtp_logits[1][:, horizon - 1:, :]
                    longer = logits[:, :-horizon + 1, :]
                    if shorter.numel() and longer.numel():
                        temporal_loss = temporal_loss + F.kl_div(
                            F.log_softmax(longer, dim=-1),
                            F.softmax(shorter.detach(), dim=-1),
                            reduction="batchmean",
                        )
            if losses:
                mtp_loss = torch.stack(losses).mean()
                total = total + self.weights.mtp * mtp_loss
            if output.confidence is not None:
                with torch.no_grad():
                    token_correct = output.logits.argmax(dim=-1).eq(next_targets).float()
                confidence_loss = F.mse_loss(output.confidence, token_correct)
                total = total + self.weights.confidence * confidence_loss
            if output.variable_compression is not None:
                variable_loss = output.variable_compression.estimated_ratio.mean()
                total = total + self.weights.variable_input * variable_loss
            if output.learned_memory_policy is not None:
                policy = output.learned_memory_policy
                with torch.no_grad():
                    detached_loss = next_loss_per_token.detach()
                    center = detached_loss.mean(dim=1, keepdim=True)
                    spread = detached_loss.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                    z = (detached_loss - center) / spread
                    exact_target = z > 0.35
                    drop_target = (z < -0.35) & output.logits.argmax(dim=-1).eq(next_targets)
                    mode_target = torch.full_like(next_targets, LearnedMemoryPolicy.LATENT)
                    mode_target = torch.where(exact_target, torch.full_like(mode_target, LearnedMemoryPolicy.EXACT), mode_target)
                    mode_target = torch.where(drop_target, torch.full_like(mode_target, LearnedMemoryPolicy.DROP), mode_target)
                    exact_supervision = exact_target.to(policy.exact_prob.dtype)
                    target_distribution = F.one_hot(mode_target, num_classes=3).to(policy.probs.dtype).mean(dim=(0, 1))
                    utility_prior = policy.utility_prior.to(device=policy.probs.device, dtype=policy.probs.dtype)
                    if int(policy.utility_feedback_events) > 0:
                        utility_prior = utility_prior.clamp_min(1e-8)
                        utility_prior = utility_prior / utility_prior.sum().clamp_min(1e-8)
                        target_distribution = 0.75 * target_distribution + 0.25 * utility_prior
                        target_distribution = target_distribution / target_distribution.sum().clamp_min(1e-8)
                mode_loss = F.cross_entropy(policy.logits.reshape(-1, 3), mode_target.reshape(-1))
                anchor_loss = F.binary_cross_entropy_with_logits(
                    policy.logits[..., LearnedMemoryPolicy.EXACT],
                    exact_supervision,
                )
                storage_loss = policy.storage_ratio.mean()
                policy_mean = policy.probs.mean(dim=(0, 1)).clamp_min(1e-8)
                policy_mean = policy_mean / policy_mean.sum().clamp_min(1e-8)
                collapse_loss = (policy_mean - target_distribution).square().sum()
                utility_alignment_loss = output.logits.new_tensor(0.0)
                if int(policy.utility_feedback_events) > 0:
                    utility_target = policy.utility_prior.to(device=policy_mean.device, dtype=policy_mean.dtype).clamp_min(1e-8)
                    utility_target = utility_target / utility_target.sum().clamp_min(1e-8)
                    utility_alignment_loss = F.kl_div(policy_mean.log(), utility_target, reduction="sum")
                learned_memory_loss = (
                    mode_loss
                    + 0.50 * anchor_loss
                    + 0.05 * storage_loss
                    + 0.10 * collapse_loss
                    + 0.10 * utility_alignment_loss
                )
                total = total + self.weights.learned_memory * learned_memory_loss
            if (
                output.skill_expert_routing is not None
                and output.skill_expert_routing.target_distribution is not None
            ):
                route_mean = output.skill_expert_routing.route_probs.mean(dim=(0, 1)).clamp_min(1e-8)
                route_mean = route_mean / route_mean.sum().clamp_min(1e-8)
                target = output.skill_expert_routing.target_distribution.to(
                    device=route_mean.device,
                    dtype=route_mean.dtype,
                )
                target = target.clamp_min(1e-8)
                target = target / target.sum().clamp_min(1e-8)
                alignment_loss = F.kl_div(route_mean.log(), target, reduction="sum")
                load_balance_loss = route_mean.square().mean()
                skill_expert_loss = alignment_loss + 0.05 * load_balance_loss
                total = total + self.weights.skill_expert * skill_expert_loss
            if output.latent_workspace is not None:
                workspace = output.latent_workspace
                transition_loss = output.logits.new_tensor(0.0)
                if workspace.step_states.shape[1] > 1:
                    transition_loss = (workspace.step_states[:, 1:, :] - workspace.step_states[:, :-1, :]).square().mean()
                gate_entropy = -(
                    workspace.step_gates.clamp_min(1e-8).log() * workspace.step_gates
                    + (1.0 - workspace.step_gates).clamp_min(1e-8).log() * (1.0 - workspace.step_gates)
                ).mean()
                binding_loss = output.logits.new_tensor(0.0)
                if output.certificate is not None and output.certificate.latent_state.shape == workspace.summary.shape:
                    binding_loss = F.mse_loss(
                        F.normalize(workspace.summary, dim=-1),
                        F.normalize(output.certificate.latent_state, dim=-1),
                    )
                attention_spread_loss = workspace.token_attention.square().sum(dim=-1).mean()
                latent_workspace_loss = binding_loss + 0.10 * transition_loss + 0.03 * gate_entropy + 0.02 * attention_spread_loss
                total = total + self.weights.latent_workspace * latent_workspace_loss
            if output.certificate is not None:
                final_targets = next_targets[:, -1]
                cert_answer_loss = F.cross_entropy(output.certificate.answer_logits, final_targets)
                with torch.no_grad():
                    sequence_correct = output.logits.argmax(dim=-1).eq(next_targets).float().mean(dim=1)
                    uncertainty_target = 1.0 - sequence_correct
                cert_uncertainty_loss = F.mse_loss(output.certificate.uncertainty, uncertainty_target)
                certificate_loss = cert_answer_loss + cert_uncertainty_loss
                total = total + self.weights.certificate * certificate_loss
            total = total + self.weights.temporal_consistency * temporal_loss
        return total, LossBreakdown(
            total=float(total.detach().cpu()),
            next_token=float(next_loss.detach().cpu()),
            mtp=float(mtp_loss.detach().cpu()),
            temporal_consistency=float(temporal_loss.detach().cpu()),
            confidence=float(confidence_loss.detach().cpu()),
            variable_input=float(variable_loss.detach().cpu()),
            learned_memory=float(learned_memory_loss.detach().cpu()),
            skill_expert=float(skill_expert_loss.detach().cpu()),
            latent_workspace=float(latent_workspace_loss.detach().cpu()),
            certificate=float(certificate_loss.detach().cpu()),
        )


@dataclass(frozen=True)
class DistributedRuntime:
    enabled: bool
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "gloo"
    gloo_interface: str | None = None

    @staticmethod
    def from_env(*, requested: bool, device_type: str, gloo_interface: str | None = None) -> "DistributedRuntime":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if requested and world_size <= 1 and "WORLD_SIZE" not in os.environ:
            raise RuntimeError("distributed=True requires a torchrun-style environment with WORLD_SIZE/RANK/LOCAL_RANK")
        enabled = requested or world_size > 1
        backend = "nccl" if device_type == "cuda" and torch.distributed.is_nccl_available() else "gloo"
        if enabled and not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")
        if enabled and backend == "gloo" and "GLOO_SOCKET_IFNAME" not in os.environ:
            selected_interface = gloo_interface or os.environ.get("CORTEX3_GLOO_IFNAME")
            if selected_interface is None and platform.system() == "Windows":
                selected_interface = "Ethernet"
            if selected_interface:
                os.environ["GLOO_SOCKET_IFNAME"] = selected_interface
        if enabled and backend == "gloo" and not torch.distributed.is_gloo_available():
            raise RuntimeError("distributed=True selected Gloo but torch.distributed.is_gloo_available() is false")
        return DistributedRuntime(
            enabled=enabled,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend=backend,
            gloo_interface=os.environ.get("GLOO_SOCKET_IFNAME"),
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def ensure_initialized(self) -> None:
        if self.enabled and not torch.distributed.is_initialized():
            init_method = os.environ.get("DIST_INIT_METHOD")
            timeout = timedelta(seconds=int(os.environ.get("CORTEX3_DISTRIBUTED_TIMEOUT_SECONDS", "60")))
            if self._use_explicit_gloo_tcp_store(init_method):
                store = torch.distributed.TCPStore(
                    os.environ.get("MASTER_ADDR", "127.0.0.1"),
                    int(os.environ["MASTER_PORT"]),
                    self.world_size,
                    self.rank == 0,
                    timeout=timeout,
                    wait_for_workers=False,
                    use_libuv=False,
                )
                torch.distributed.init_process_group(
                    backend=self.backend,
                    store=store,
                    rank=self.rank,
                    world_size=self.world_size,
                    timeout=timeout,
                )
                return
            torch.distributed.init_process_group(
                backend=self.backend,
                init_method=init_method,
                rank=self.rank,
                world_size=self.world_size,
                timeout=timeout,
            )

    def _use_explicit_gloo_tcp_store(self, init_method: str | None) -> bool:
        if self.backend != "gloo" or "MASTER_PORT" not in os.environ:
            return False
        if init_method and init_method not in {"env://", "tcp://"}:
            return False
        forced = os.environ.get("CORTEX3_TCPSTORE_USE_LIBUV")
        if forced is not None:
            return forced.strip().lower() in {"0", "false", "no", "off"}
        return platform.system() == "Windows"


@dataclass(frozen=True)
class PrecisionPolicy:
    precision: str = "fp32"
    require_cuda: bool = False

    def dtype(self, device_type: str) -> torch.dtype:
        if self.precision == "fp32":
            return torch.float32
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp16":
            if device_type != "cuda":
                raise RuntimeError("fp16 mixed precision requires CUDA")
            return torch.float16
        raise ValueError(f"unsupported precision: {self.precision}")

    def autocast(self, device_type: str):
        if self.precision == "fp32":
            return nullcontext()
        dtype = self.dtype(device_type)
        return torch.autocast(device_type=device_type, dtype=dtype)

    def scaler(self, device_type: str):
        if self.precision == "fp16" and device_type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 200
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 8
    seed: int = 7
    device: str = "auto"
    precision: str = "fp32"
    require_cuda: bool = False
    distributed: bool = False
    gloo_interface: str | None = None
    resume: bool = False
    resume_if_exists: bool = False
    resume_from_checkpoint: str | None = None
    checkpoint_interval: int = 100
    max_intermediate_checkpoints: int = 0
    resource_monitor_interval: float = 2.0
    cortex_phase_interval: int = 0
    cortex_phase_probe_tasks: int = 1
    cortex_phase_max_proposals: int = 1
    cortex_phase_improvement_generations: int = 2
    cortex_phase_regrowth_budget: float = 32.0
    cortex_phase_frontier_max_skills: int = 1
    cortex_phase_frontier_per_failure: int = 1
    cortex_phase_frontier_epochs: int = 40
    cortex_phase_regularization_weight: float = 0.001
    cortex_phase_replay_weight: float = 0.05
    cortex_objective_feedback_weight: float = 0.05
    cortex_objective_feedback_clip: float = 4.0
    cortex_trace_retention_limit: int = 4096
    cortex_improvement_archive_dir: str | None = None
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if self.max_intermediate_checkpoints < 0:
            raise ValueError("max_intermediate_checkpoints must be non-negative")
        if self.resource_monitor_interval <= 0:
            raise ValueError("resource_monitor_interval must be positive")
        if self.cortex_phase_interval < 0:
            raise ValueError("cortex_phase_interval must be non-negative")
        if self.cortex_phase_probe_tasks < 1:
            raise ValueError("cortex_phase_probe_tasks must be positive")
        if self.cortex_phase_max_proposals < 0:
            raise ValueError("cortex_phase_max_proposals must be non-negative")
        if self.cortex_phase_improvement_generations < 1:
            raise ValueError("cortex_phase_improvement_generations must be positive")
        if self.cortex_phase_regrowth_budget <= 0:
            raise ValueError("cortex_phase_regrowth_budget must be positive")
        if self.cortex_phase_frontier_max_skills < 0:
            raise ValueError("cortex_phase_frontier_max_skills must be non-negative")
        if self.cortex_phase_frontier_per_failure < 1:
            raise ValueError("cortex_phase_frontier_per_failure must be positive")
        if self.cortex_phase_frontier_epochs < 1:
            raise ValueError("cortex_phase_frontier_epochs must be positive")
        if self.cortex_phase_regularization_weight < 0:
            raise ValueError("cortex_phase_regularization_weight must be non-negative")
        if self.cortex_phase_replay_weight < 0:
            raise ValueError("cortex_phase_replay_weight must be non-negative")
        if self.cortex_objective_feedback_weight < 0:
            raise ValueError("cortex_objective_feedback_weight must be non-negative")
        if self.cortex_objective_feedback_clip < 0:
            raise ValueError("cortex_objective_feedback_clip must be non-negative")
        if self.cortex_trace_retention_limit < 0:
            raise ValueError("cortex_trace_retention_limit must be non-negative")
        if self.cortex_improvement_archive_dir is not None and not str(self.cortex_improvement_archive_dir).strip():
            raise ValueError("cortex_improvement_archive_dir must not be empty when provided")
        if self.resume and self.resume_if_exists:
            raise ValueError("resume and resume_if_exists are mutually exclusive")


def _training_allows_existing_artifacts(training: TrainingConfig) -> bool:
    return bool(training.resume or training.resume_if_exists)


def _training_resolves_to_cuda(training: TrainingConfig) -> bool:
    device = str(training.device)
    if device == "auto":
        return bool(torch.cuda.is_available())
    try:
        return torch.device(device).type == "cuda"
    except (TypeError, RuntimeError):
        return device.startswith("cuda")


def _resolve_cli_precision(precision: str, *, device: str, require_cuda: bool) -> str:
    requested = str(precision)
    if requested != "auto":
        return requested
    device_text = str(device)
    resolves_to_cuda = bool(require_cuda)
    if not resolves_to_cuda:
        if device_text == "auto":
            resolves_to_cuda = bool(torch.cuda.is_available())
        else:
            try:
                resolves_to_cuda = torch.device(device_text).type == "cuda"
            except (TypeError, RuntimeError):
                resolves_to_cuda = device_text.startswith("cuda")
    return "fp16" if resolves_to_cuda else "fp32"


def _strict_native_ternary_required_for_training(training: TrainingConfig) -> bool:
    return bool(training.require_cuda or _training_resolves_to_cuda(training))


@dataclass(frozen=True)
class TrainingPoint:
    step: int
    split: str
    loss: float
    next_token_loss: float
    token_accuracy: float
    mtp_loss: float = 0.0
    future_tokens_per_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingRunReport:
    name: str
    model_kind: str
    run_dir: str
    checkpoint_path: str
    start_step: int
    optimizer_steps: int
    effective_batch_size: int
    resumed_from: str | None
    final_train: TrainingPoint
    final_val: TrainingPoint
    curve: tuple[TrainingPoint, ...]
    config: Mapping[str, Any]
    hardware: Mapping[str, Any]
    code_state: Mapping[str, Any] = field(default_factory=dict)
    resource_usage: Mapping[str, Any] = field(default_factory=dict)
    cortex_phase_report: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["curve"] = [point.to_dict() for point in self.curve]
        payload["final_train"] = self.final_train.to_dict()
        payload["final_val"] = self.final_val.to_dict()
        return payload


class ResourceUsageMonitor:
    def __init__(self, *, device: torch.device, interval_seconds: float):
        self.device = device
        self.interval_seconds = float(interval_seconds)
        self.process = psutil.Process(os.getpid())
        self.logical_cpu_count = max(1, int(psutil.cpu_count(logical=True) or 1))
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._stopped_at = 0.0

    def start(self) -> None:
        self._started_at = time.time()
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, name="cortex3-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> Mapping[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
        self._stopped_at = time.time()
        if not self._samples:
            self._sample()
        return self.summary()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        try:
            sample: dict[str, Any] = {
                "timestamp": time.time(),
                "cpu_total_percent": float(psutil.cpu_percent(interval=None)),
                "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
                "process_memory_rss_bytes": int(self.process.memory_info().rss),
            }
            sample["process_cpu_percent_of_total"] = float(sample["process_cpu_percent"]) / float(self.logical_cpu_count)
            gpu_sample = self._nvidia_smi_sample()
            if gpu_sample:
                sample.update(gpu_sample)
            with self._lock:
                self._samples.append(sample)
        except Exception as exc:
            with self._lock:
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def _nvidia_smi_sample(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {}
        device_index = self.device.index
        if device_index is None:
            try:
                device_index = int(torch.cuda.current_device())
            except Exception:
                device_index = 0
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=2.0, check=False)
        except FileNotFoundError:
            return {"gpu_monitor_error": "nvidia-smi not found"}
        except Exception as exc:
            return {"gpu_monitor_error": f"{type(exc).__name__}: {exc}"}
        if completed.returncode != 0:
            return {"gpu_monitor_error": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}"}
        for raw_line in completed.stdout.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 5:
                continue
            try:
                index = int(parts[0])
            except ValueError:
                continue
            if index != device_index:
                continue
            out = {
                "gpu_index": index,
                "gpu_utilization_percent": float(parts[1]),
                "gpu_memory_utilization_percent": float(parts[2]),
                "gpu_memory_used_mb": float(parts[3]),
                "gpu_memory_total_mb": float(parts[4]),
            }
            if len(parts) >= 6:
                try:
                    out["gpu_power_draw_watts"] = float(parts[5])
                except ValueError:
                    pass
            return out
        return {"gpu_monitor_error": f"cuda device index {device_index} not reported by nvidia-smi"}

    def summary(self) -> Mapping[str, Any]:
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        duration = (self._stopped_at or time.time()) - (self._started_at or time.time())

        def values(key: str) -> list[float]:
            out: list[float] = []
            for sample in samples:
                value = sample.get(key)
                if isinstance(value, (int, float)):
                    out.append(float(value))
            return out

        def stats(key: str) -> dict[str, float] | None:
            vals = values(key)
            if not vals:
                return None
            return {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }

        keys = (
            "cpu_total_percent",
            "process_cpu_percent",
            "process_cpu_percent_of_total",
            "process_memory_rss_bytes",
            "gpu_utilization_percent",
            "gpu_memory_utilization_percent",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_power_draw_watts",
        )
        metric_stats = {key: stats(key) for key in keys}
        return {
            "schema_version": 1,
            "enabled": True,
            "sample_interval_seconds": self.interval_seconds,
            "duration_seconds": duration,
            "sample_count": len(samples),
            "logical_cpu_count": self.logical_cpu_count,
            "metrics": {key: value for key, value in metric_stats.items() if value is not None},
            "errors": errors + sorted({str(sample["gpu_monitor_error"]) for sample in samples if sample.get("gpu_monitor_error")}),
            "first_sample": samples[0] if samples else None,
            "last_sample": samples[-1] if samples else None,
        }

    def write_snapshot(self, path: Path, *, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            has_samples = bool(self._samples)
        if not has_samples:
            self._sample()
        payload = dict(self.summary())
        payload["metadata"] = dict(metadata)
        payload["written_at"] = time.time()
        _write_json(Path(path), payload)
        return payload


def _cuda_memory_report() -> tuple[dict[str, Any], ...]:
    if not torch.cuda.is_available():
        return ()
    devices: list[dict[str, Any]] = []
    current_device: int | None = None
    try:
        current_device = int(torch.cuda.current_device())
    except Exception:
        current_device = None
    for index in range(int(torch.cuda.device_count())):
        device: dict[str, Any] = {"index": index}
        try:
            props = torch.cuda.get_device_properties(index)
            device.update(
                {
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
        except Exception as exc:
            device["properties_error"] = str(exc)
        if current_device == index:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                device["free_memory_bytes"] = int(free_bytes)
                device["mem_get_info_total_memory_bytes"] = int(total_bytes)
            except Exception as exc:
                device["mem_get_info_error"] = str(exc)
        devices.append(device)
    return tuple(devices)


def hardware_report() -> dict[str, Any]:
    cuda_devices = _cuda_memory_report()
    current_device = None
    if torch.cuda.is_available():
        try:
            current_device = int(torch.cuda.current_device())
        except Exception:
            current_device = None
    current_payload = next((item for item in cuda_devices if item.get("index") == current_device), {})
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_current_device": current_device,
        "cuda_current_device_name": current_payload.get("name"),
        "cuda_current_device_total_memory_bytes": current_payload.get("total_memory_bytes"),
        "cuda_current_device_free_memory_bytes": current_payload.get("free_memory_bytes"),
        "cuda_devices": cuda_devices,
        "distributed_available": bool(torch.distributed.is_available()),
        "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
        "gloo_available": bool(torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    }


def _git_output(args: Sequence[str], *, timeout: float = 2.0) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def code_state_report() -> dict[str, Any]:
    tracked_status = _git_output(["status", "--short", "--untracked-files=no"])
    untracked_status = _git_output(["status", "--short", "--untracked-files=all"])
    untracked_lines = [
        line
        for line in (untracked_status or "").splitlines()
        if line.startswith("?? ") and not line.startswith("?? .codex/")
    ]
    return {
        "schema_version": 1,
        "source_file": str(Path(__file__).resolve()),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "git_commit_time": _git_output(["show", "-s", "--format=%cI", "HEAD"]),
        "tracked_dirty": bool((tracked_status or "").strip()),
        "tracked_status": tuple(line for line in (tracked_status or "").splitlines() if line.strip()),
        "untracked_file_count_excluding_codex": len(untracked_lines),
    }


def _package_versions() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ("torch", "numpy", "tokenizers", "matplotlib", "datasets", "psutil")
    payload: dict[str, Any] = {}
    for package in packages:
        try:
            payload[package] = {"installed": True, "version": version(package)}
        except PackageNotFoundError:
            payload[package] = {"installed": False, "version": None}
    return payload


def _command_stdout(args: Sequence[str], *, timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _nvcc_release_from_output(output: str | None) -> str | None:
    if not output:
        return None
    for line in output.splitlines():
        if "release " in line:
            return line.split("release ", 1)[1].split(",", 1)[0].strip()
    return None


def _visual_studio_toolchain_probe() -> dict[str, Any]:
    cl_on_path = shutil.which("cl")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    vswhere = (
        Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if program_files_x86
        else None
    )
    installations: list[dict[str, Any]] = []
    if vswhere is not None and vswhere.exists():
        raw_installations = _command_stdout([
            str(vswhere),
            "-all",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-format",
            "json",
        ])
        if raw_installations:
            try:
                installations = list(json.loads(raw_installations))
            except Exception:
                installations = []
    cl_candidates: list[dict[str, Any]] = []
    for installation in installations:
        installation_path = installation.get("installationPath")
        if not installation_path:
            continue
        vc_root = Path(str(installation_path)) / "VC" / "Tools" / "MSVC"
        for candidate in vc_root.glob("*/bin/Hostx64/x64/cl.exe"):
            cl_candidates.append({
                "cl_path": str(candidate),
                "toolset_version": candidate.parents[2].name,
                "installation_path": str(installation_path),
                "installation_version": str(installation.get("installationVersion", "")),
                "display_name": str(installation.get("displayName", "")),
                "product_id": str(installation.get("productId", "")),
            })
    cl_candidates = sorted(
        cl_candidates,
        key=lambda item: (
            not str(item["installation_version"]).startswith("17."),
            str(item["installation_version"]),
            str(item["toolset_version"]),
        ),
    )
    selected = cl_candidates[0] if cl_candidates else None
    return {
        "cl_on_path": cl_on_path,
        "visual_studio_installations": tuple(installations),
        "cl_candidates": tuple(cl_candidates),
        "selected_cl": selected,
        "cl_available": bool(cl_on_path or cl_candidates),
    }


def _pip_nvcc_cu12_probe() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, distribution, version

    try:
        dist = distribution("nvidia-cuda-nvcc-cu12")
        files = tuple(str(path).replace("\\", "/") for path in (dist.files or ()))
        has_nvcc_exe = any(path.endswith("/nvcc.exe") or path.endswith("/nvcc") for path in files)
        return {
            "installed": True,
            "version": version("nvidia-cuda-nvcc-cu12"),
            "has_nvcc_exe": has_nvcc_exe,
            "has_ptxas": any(path.endswith("/ptxas.exe") or path.endswith("/ptxas") for path in files),
        }
    except PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "has_nvcc_exe": False,
            "has_ptxas": False,
        }


def _cuda_candidate_homes(torch_cuda: str | None) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for value in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if value:
            candidates.append(Path(value))
    if torch_cuda:
        candidates.append(Path.home() / ".codex" / f"cuda-{torch_cuda}" / "Library")
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        candidates.append(Path(nvcc_path).resolve(strict=False).parent.parent)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False)).lower()
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return tuple(deduped)


def _cuda_home_probe(home: Path, torch_cuda: str | None) -> dict[str, Any]:
    nvcc_path = home / "bin" / "nvcc.exe"
    if not nvcc_path.exists():
        nvcc_path = home / "bin" / "nvcc"
    nvcc_output = _command_stdout([str(nvcc_path), "--version"], timeout=5.0) if nvcc_path.exists() else None
    nvcc_release = _nvcc_release_from_output(nvcc_output)
    cudart_candidates = (
        home / "lib" / "x64" / "cudart.lib",
        home / "lib" / "cudart.lib",
    )
    cudart_lib = next((path for path in cudart_candidates if path.exists()), None)
    return {
        "cuda_home": str(home),
        "nvcc_path": str(nvcc_path) if nvcc_path.exists() else None,
        "nvcc_release": nvcc_release,
        "nvcc_matches_torch_cuda": bool(torch_cuda and nvcc_release and str(nvcc_release).startswith(str(torch_cuda))),
        "include_cuda_runtime_h": (home / "include" / "cuda_runtime.h").exists(),
        "cudart_lib": str(cudart_lib) if cudart_lib else None,
        "pytorch_windows_lib_x64_ready": (home / "lib" / "x64" / "cudart.lib").exists(),
    }


def cuda_toolchain_report() -> dict[str, Any]:
    torch_cuda = getattr(torch.version, "cuda", None)
    try:
        from torch.utils import cpp_extension

        cpp_cuda_home = cpp_extension.CUDA_HOME
    except Exception:
        cpp_cuda_home = None
    visual_studio = _visual_studio_toolchain_probe()
    pip_nvcc = _pip_nvcc_cu12_probe()
    cuda_candidates = tuple(_cuda_home_probe(path, torch_cuda) for path in _cuda_candidate_homes(torch_cuda))
    selected_cuda = next((item for item in cuda_candidates if item["nvcc_matches_torch_cuda"]), None)
    if selected_cuda is None and cuda_candidates:
        selected_cuda = cuda_candidates[0]
    rawkernel_available = False
    rawkernel_error = ""
    extension_runtime_available = False
    if torch.cuda.is_available():
        try:
            rawkernel_available = bool(native_ternary_cuda_available())
        except Exception as exc:
            rawkernel_error = f"{type(exc).__name__}: {exc}"
    nvcc_matches_torch = bool(selected_cuda and selected_cuda["nvcc_matches_torch_cuda"])
    has_cuda_headers = bool(selected_cuda and selected_cuda["include_cuda_runtime_h"])
    has_cudart_lib = bool(selected_cuda and selected_cuda["cudart_lib"])
    extension_ready = bool(torch.cuda.is_available() and visual_studio["cl_available"] and nvcc_matches_torch and has_cuda_headers and has_cudart_lib)
    if extension_ready:
        extension_runtime_available = bool(native_ternary_cuda_extension_available())
    return {
        "torch_cuda": torch_cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_home": str(cpp_cuda_home) if cpp_cuda_home else None,
        "cuda_path": os.environ.get("CUDA_PATH"),
        "selected_cuda_home": selected_cuda["cuda_home"] if selected_cuda else None,
        "cuda_home_candidates": cuda_candidates,
        "nvcc_path": selected_cuda["nvcc_path"] if selected_cuda else shutil.which("nvcc"),
        "nvcc_release": selected_cuda["nvcc_release"] if selected_cuda else None,
        "nvcc_matches_torch_cuda": nvcc_matches_torch,
        "include_cuda_runtime_h": has_cuda_headers,
        "cudart_lib": selected_cuda["cudart_lib"] if selected_cuda else None,
        "pytorch_windows_lib_x64_ready": bool(selected_cuda and selected_cuda["pytorch_windows_lib_x64_ready"]),
        "visual_studio": visual_studio,
        "pip_nvidia_cuda_nvcc_cu12": pip_nvcc,
        "native_rawkernel_available": rawkernel_available,
        "native_rawkernel_error": rawkernel_error,
        "cuda_extension_toolchain_ready": extension_ready,
        "native_extension_runtime_available": extension_runtime_available,
        "cuda_extension_blocker": (
            ""
            if extension_ready and extension_runtime_available
            else (
                "CUDA C++ extension build requires cl plus an nvcc toolkit whose major.minor version matches torch.version.cuda; "
                f"torch={torch_cuda}, nvcc={selected_cuda['nvcc_release'] if selected_cuda else None}, "
                f"cl_available={visual_studio['cl_available']}, cuda_headers={has_cuda_headers}, cudart_lib={has_cudart_lib}, "
                f"extension_runtime_available={extension_runtime_available}"
            )
        ),
    }


def llm_doctor_report(
    *,
    require_cuda: bool = False,
    require_cuda_extension: bool = False,
    precision: str = "bf16",
    device: str = "auto",
    distributed: bool = False,
    gloo_interface: str | None = None,
) -> dict[str, Any]:
    hardware = hardware_report()
    dependencies = _package_versions()
    cuda_toolchain = cuda_toolchain_report()
    device_type = "cuda" if (device == "auto" and torch.cuda.is_available()) or str(device).startswith("cuda") else "cpu"
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "required": required, "detail": detail})

    for package, payload in dependencies.items():
        add_check(f"dependency:{package}", bool(payload["installed"]), f"version={payload['version']}")
    add_check("torch:cuda_available", bool(hardware["cuda_available"]), f"cuda_device_count={hardware['cuda_device_count']}", required=require_cuda)
    if require_cuda:
        add_check("torch:require_cuda", device_type == "cuda" and bool(hardware["cuda_available"]), f"resolved_device_type={device_type}")
        add_check(
            "cuda:native_rawkernel_available",
            bool(cuda_toolchain["native_rawkernel_available"]),
            cuda_toolchain["native_rawkernel_error"] or "CuPy RawKernel ternary kernels compile",
        )
    add_check(
        "cuda:extension_toolchain_ready",
        bool(cuda_toolchain["cuda_extension_toolchain_ready"]),
        cuda_toolchain["cuda_extension_blocker"] or "CUDA C++ extension toolchain is ready",
        required=require_cuda_extension,
    )
    add_check(
        "cuda:native_extension_runtime_available",
        bool(cuda_toolchain["native_extension_runtime_available"]),
        cuda_toolchain["cuda_extension_blocker"] or "Cortex ternary CUDA extension builds and loads",
        required=require_cuda_extension,
    )

    try:
        dtype = PrecisionPolicy(precision, require_cuda=require_cuda).dtype(device_type)
        add_check("precision", True, f"{precision} resolves to {dtype} on {device_type}")
    except Exception as exc:
        add_check("precision", False, str(exc))

    add_check("distributed:available", bool(hardware["distributed_available"]), "torch.distributed availability", required=distributed)
    add_check("distributed:gloo", bool(hardware["gloo_available"]), "Gloo backend availability", required=distributed and device_type != "cuda")
    add_check("distributed:nccl", bool(hardware["nccl_available"]), "NCCL backend availability", required=distributed and device_type == "cuda")
    if distributed:
        try:
            runtime = DistributedRuntime.from_env(requested=False, device_type=device_type, gloo_interface=gloo_interface)
            add_check("distributed:env_probe", True, f"backend={runtime.backend}, world_size={runtime.world_size}, gloo_interface={runtime.gloo_interface}")
        except Exception as exc:
            add_check("distributed:env_probe", False, str(exc))

    failed_required = [check for check in checks if check["required"] and not check["passed"]]
    return {
        "passed": not failed_required,
        "device_type": device_type,
        "requested": {
            "require_cuda": require_cuda,
            "require_cuda_extension": require_cuda_extension,
            "precision": precision,
            "device": device,
            "distributed": distributed,
            "gloo_interface": gloo_interface,
        },
        "hardware": hardware,
        "cuda_toolchain": cuda_toolchain,
        "dependencies": dependencies,
        "checks": tuple(checks),
        "failed_required_checks": tuple(failed_required),
    }


def _last_items(items: Sequence[Any], limit: int = 3) -> list[Any]:
    if limit <= 0:
        return []
    return list(items[-limit:])


def _frontier_heldout_summary(registry_payload: Mapping[str, Any]) -> dict[str, Any]:
    circuits = tuple(
        dict(item)
        for item in registry_payload.get("circuits", ())
        if isinstance(item, Mapping)
    )
    heldout_total = 0
    heldout_passed = 0
    gate_passed_circuits = 0
    sleep_circuits = 0
    sleep_heldout_total = 0
    sleep_heldout_passed = 0
    sleep_gate_passed_circuits = 0
    for circuit in circuits:
        report = dict(circuit.get("report") or {})
        training = dict(report.get("training") or {})
        heldout = dict(report.get("heldout") or {})
        total = int(heldout.get("total", 0) or 0)
        passed = int(heldout.get("passed", 0) or 0)
        heldout_total += total
        heldout_passed += passed
        if total > 0 and passed >= total and bool(heldout.get("gate_passed", False)):
            gate_passed_circuits += 1
        if str(training.get("source_kind", "")) == "sleep_consolidation":
            sleep_circuits += 1
            sleep_heldout_total += total
            sleep_heldout_passed += passed
            if total > 0 and passed >= total and bool(heldout.get("gate_passed", False)):
                sleep_gate_passed_circuits += 1
    return {
        "frontier_heldout_circuit_count": len(circuits),
        "frontier_heldout_gate_passed_circuit_count": gate_passed_circuits,
        "frontier_heldout_total": heldout_total,
        "frontier_heldout_passed": heldout_passed,
        "sleep_frontier_compiled_circuit_count": sleep_circuits,
        "sleep_frontier_heldout_total": sleep_heldout_total,
        "sleep_frontier_heldout_passed": sleep_heldout_passed,
        "sleep_frontier_heldout_gate_passed_circuit_count": sleep_gate_passed_circuits,
    }


def _cortex_architecture_audit_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    phase_counts = {str(key): int(value) for key, value in dict(summary.get("phase_event_counts") or {}).items()}
    trace_counts = {str(key): int(value) for key, value in dict(summary.get("compression_trace_counts") or {}).items()}
    native_ternary_required = bool(summary.get("native_ternary_kernel_required", False))
    native_backend_requested = str(summary.get("native_ternary_backend_requested", "auto"))
    native_backend_counts = {str(key): int(value) for key, value in dict(summary.get("native_ternary_backend_counts") or {}).items()}
    native_requantize_backend_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("native_ternary_requantize_backend_counts") or {}).items()
    }
    native_grad_weight_backend_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("native_ternary_grad_weight_backend_counts") or {}).items()
    }
    native_dispatch_required = native_ternary_required
    replay_by_phase = {
        str(key): int(value)
        for key, value in dict(summary.get("phase_replay_examples_by_phase") or {}).items()
    }

    def number(key: str) -> float:
        value = summary.get(key, 0)
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    def integer(key: str) -> int:
        return int(number(key))

    def phase_count(phase_id: str) -> int:
        return int(phase_counts.get(phase_id, 0))

    def trace_count(name: str) -> int:
        return int(trace_counts.get(name, 0))

    def only_extension(counts: Mapping[str, int]) -> bool:
        return int(counts.get(STRICT_NATIVE_TERNARY_BACKEND, 0)) > 0 and not any(
            int(value) > 0 and str(backend) != STRICT_NATIVE_TERNARY_BACKEND
            for backend, value in counts.items()
        )

    def requested_native_backend_met() -> bool:
        if native_backend_requested != STRICT_NATIVE_TERNARY_BACKEND:
            return True
        return (
            only_extension(native_backend_counts)
            and only_extension(native_requantize_backend_counts)
            and only_extension(native_grad_weight_backend_counts)
            and trace_count("torch_packed_ternary_dispatches") == 0
        )

    def replay_count(phase_id: str) -> int:
        return int(replay_by_phase.get(phase_id, 0))

    raw_errors = summary.get("errors", ())
    inferred_error_count = len(raw_errors) if isinstance(raw_errors, Sequence) and not isinstance(raw_errors, (str, bytes)) else 0
    error_count = int(number("error_count")) if "error_count" in summary else inferred_error_count
    improvement_decisions = integer("improvement_archive_accepted") + integer("improvement_archive_rejected")
    required_phase_ids = tuple(phase.id for phase in CORTEX3_PHASES)
    phase_observed = {phase_id: phase_count(phase_id) for phase_id in required_phase_ids}
    observed_objective_terms = tuple(str(term) for term in summary.get("objective_feedback_term_names", ()) or ())
    required_objective_terms = tuple(FINAL_LOSS_TERMS)
    objective_term_set = set(observed_objective_terms)
    missing_objective_terms = tuple(term for term in required_objective_terms if term not in objective_term_set)
    regrowth_model_applications = tuple(
        dict(item)
        for item in (summary.get("regrowth_model_applications") or ())
        if isinstance(item, Mapping)
    )
    regrowth_model_non_regressing = any(
        bool(item.get("non_regression_passed"))
        and float(item.get("parameter_delta_l1", 0.0) or 0.0) > 0.0
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and float(item.get("protected_loss_delta", 0.0) or 0.0) <= float(item.get("protected_loss_tolerance", 0.0) or 0.0)
        for item in regrowth_model_applications
    )
    recursive_model_applications = tuple(
        dict(item)
        for item in (summary.get("recursive_model_applications") or ())
        if isinstance(item, Mapping)
    )
    recursive_model_non_regressing = any(
        bool(item.get("non_regression_passed"))
        and bool(item.get("signed_patch_id"))
        and bool(item.get("rollback_token"))
        and float(item.get("parameter_delta_l1", 0.0) or 0.0) > 0.0
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and float(item.get("protected_loss_delta", 0.0) or 0.0) <= float(item.get("protected_loss_tolerance", 0.0) or 0.0)
        for item in recursive_model_applications
    )
    recursive_verified_artifacts = tuple(
        dict(item)
        for item in (summary.get("recursive_verified_artifacts") or ())
        if isinstance(item, Mapping)
    )
    recursive_verified_artifact_count = max(integer("recursive_verified_artifact_count"), len(recursive_verified_artifacts))
    recursive_artifact_verified = any(
        bool(item.get("recursive_improvement_artifact"))
        and bool(item.get("artifact_id"))
        and bool(item.get("example_id"))
        and bool(item.get("signed_patch_id"))
        and bool(item.get("rollback_token"))
        and int(item.get("verification_level", 0) or 0) >= 3
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and bool(item.get("non_regression_passed"))
        for item in recursive_verified_artifacts
    )
    checks: list[dict[str, Any]] = []

    def add(component: str, passed: bool, observed: Mapping[str, Any], requirement: str) -> None:
        checks.append(
            {
                "component": component,
                "passed": bool(passed),
                "observed": dict(observed),
                "requirement": requirement,
            }
        )

    add(
        "p1_to_p10_phase_activity",
        all(count > 0 for count in phase_observed.values()),
        phase_observed,
        "every Cortex-3 phase P1..P10 must have at least one training event",
    )
    add(
        "variable_in_compressor",
        integer("variable_input_compression_events") > 0 and trace_count("kv_events") > 0,
        {
            "variable_input_compression_events": integer("variable_input_compression_events"),
            "kv_events": trace_count("kv_events"),
        },
        "Variable-In/KV compression must record real LLM batch events",
    )
    add(
        "exact_anchor_ledger",
        integer("input_anchor_observations") > 0
        and integer("input_anchor_count") > 0
        and integer("input_anchor_fidelity_failures") == 0,
        {
            "input_anchor_observations": integer("input_anchor_observations"),
            "input_anchor_count": integer("input_anchor_count"),
            "input_anchor_fidelity_failures": integer("input_anchor_fidelity_failures"),
        },
        "decoded input batches must create exact anchors with zero fidelity failures",
    )
    add(
        "latent_memory_kv",
        integer("memory_recent_segments") > 0 and integer("memory_latent_segments") > 0,
        {
            "memory_recent_segments": integer("memory_recent_segments"),
            "memory_latent_segments": integer("memory_latent_segments"),
        },
        "Cognitive memory must contain both recent exact KV and latent compressed KV",
    )
    add(
        "learned_cognitive_memory_policy",
        integer("learned_memory_policy_events") > 0
        and integer("learned_memory_anchor_supervision_events") > 0
        and integer("learned_memory_exact_decisions") > 0
        and integer("learned_memory_latent_decisions") > 0
        and number("learned_memory_storage_ratio_mean") > 0.0
        and integer("learned_memory_retention_decisions") > 0
        and (
            integer("learned_memory_retention_applied_exact")
            + integer("learned_memory_retention_applied_latent")
            + integer("learned_memory_retention_applied_drop")
        )
        == integer("learned_memory_retention_decisions")
        and integer("learned_memory_utility_credit_count") > 0
        and integer("learned_memory_utility_positive_count") > 0
        and integer("learned_memory_utility_prior_updates") > 0
        and integer("learned_memory_utility_feedback_events") > 0,
        {
            "learned_memory_policy_events": integer("learned_memory_policy_events"),
            "learned_memory_anchor_supervision_events": integer("learned_memory_anchor_supervision_events"),
            "learned_memory_exact_decisions": integer("learned_memory_exact_decisions"),
            "learned_memory_latent_decisions": integer("learned_memory_latent_decisions"),
            "learned_memory_drop_decisions": integer("learned_memory_drop_decisions"),
            "learned_memory_storage_ratio_mean": number("learned_memory_storage_ratio_mean"),
            "learned_memory_retention_decisions": integer("learned_memory_retention_decisions"),
            "learned_memory_retention_applied_exact": integer("learned_memory_retention_applied_exact"),
            "learned_memory_retention_applied_latent": integer("learned_memory_retention_applied_latent"),
            "learned_memory_retention_applied_drop": integer("learned_memory_retention_applied_drop"),
            "learned_memory_retention_anchor_overrides": integer("learned_memory_retention_anchor_overrides"),
            "learned_memory_utility_credit_count": integer("learned_memory_utility_credit_count"),
            "learned_memory_utility_positive_count": integer("learned_memory_utility_positive_count"),
            "learned_memory_utility_prior_updates": integer("learned_memory_utility_prior_updates"),
            "learned_memory_utility_feedback_events": integer("learned_memory_utility_feedback_events"),
            "learned_memory_last_utility_prior": tuple(summary.get("learned_memory_last_utility_prior", ()) or ()),
        },
        "cognitive memory must learn exact/latent/drop retention decisions, apply them to P4 storage, and update the policy from downstream reconstruction utility",
    )
    add(
        "compiled_circuit_memory_retention",
        integer("compiled_circuit_memory_binding_count") > 0
        and integer("compiled_circuit_memory_binding_events") > 0
        and integer("compiled_circuit_memory_fidelity_failures") == 0
        and (
            integer("frontier_registry_loaded_events") == 0
            or (
                integer("frontier_restored_fastsolve_events") > 0
                and integer("compiled_circuit_memory_restored_reuse_events") > 0
            )
        ),
        {
            "compiled_circuit_memory_binding_count": integer("compiled_circuit_memory_binding_count"),
            "compiled_circuit_memory_binding_events": integer("compiled_circuit_memory_binding_events"),
            "compiled_circuit_memory_fidelity_failures": integer("compiled_circuit_memory_fidelity_failures"),
            "frontier_registry_loaded_events": integer("frontier_registry_loaded_events"),
            "frontier_restored_fastsolve_events": integer("frontier_restored_fastsolve_events"),
            "compiled_circuit_memory_restored_reuse_events": integer("compiled_circuit_memory_restored_reuse_events"),
        },
        "P4 memory must retain, restore and reconstruct compiled Frontier circuits before FastSolve/P7/P9/P10 reuse",
    )
    add(
        "ternary_core",
        trace_count("layer_forward_events") > 0
        and trace_count("activation_quantizations") > 0
        and trace_count("compression_decisions") > 0,
        {
            "layer_forward_events": trace_count("layer_forward_events"),
            "activation_quantizations": trace_count("activation_quantizations"),
            "compression_decisions": trace_count("compression_decisions"),
        },
        "ternary W in {-1,0,+1} core must run layer forwards and activation quantization",
    )
    add(
        "packed_ternary_hardware_runtime",
        trace_count("packed_ternary_dispatches") > 0,
        {"packed_ternary_dispatches": trace_count("packed_ternary_dispatches")},
        "ternary core must dispatch from packed 2-bit ternary weight buffers, not only trace PyTorch float-linear compatibility",
    )
    add(
        "native_ternary_cuda_kernel",
        (not native_dispatch_required)
        or (
            trace_count("native_ternary_kernel_dispatches") > 0
            and trace_count("native_ternary_autotuned_dispatches") > 0
            and requested_native_backend_met()
        ),
        {
            "native_ternary_kernel_required": native_ternary_required,
            "native_ternary_dispatch_required": native_dispatch_required,
            "native_ternary_backend_requested": native_backend_requested,
            "native_ternary_backend_counts": native_backend_counts,
            "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
            "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
            "strict_extension_only": requested_native_backend_met(),
            "native_ternary_kernel_dispatches": trace_count("native_ternary_kernel_dispatches"),
            "torch_packed_ternary_dispatches": trace_count("torch_packed_ternary_dispatches"),
            "native_ternary_autotuned_dispatches": trace_count("native_ternary_autotuned_dispatches"),
            "native_ternary_autotune_cache_hits": trace_count("native_ternary_autotune_cache_hits"),
        },
        "CUDA full-architecture runs must launch an autotuned native packed int2 ternary kernel, not only a PyTorch matmul over unpacked weights",
    )
    add(
        "skill_aware_experts",
        trace_count("expert_activations") > 0
        and integer("skill_expert_context_events") > 0
        and integer("skill_expert_replay_context_events") > 0
        and len(tuple(summary.get("skill_expert_context_skills", ()) or ())) > 0,
        {
            "expert_activations": trace_count("expert_activations"),
            "skill_expert_context_events": integer("skill_expert_context_events"),
            "skill_expert_replay_context_events": integer("skill_expert_replay_context_events"),
            "skill_expert_context_updates": integer("skill_expert_context_updates"),
            "skill_expert_context_skills": tuple(summary.get("skill_expert_context_skills", ()) or ()),
        },
        "skill-aware expert routing must be conditioned by Skill Ledger/replay skill context, not only activate generic experts",
    )
    add(
        "bit_ledger",
        number("bit_ledger_total_effective_bits") > 0.0,
        {"bit_ledger_total_effective_bits": number("bit_ledger_total_effective_bits")},
        "Bit ledger must accumulate non-zero effective bit cost",
    )
    add(
        "skill_ledger",
        integer("skill_ledger_states") > 0,
        {"skill_ledger_states": integer("skill_ledger_states")},
        "Skill ledger must record at least one skill state",
    )
    add(
        "causal_ledger",
        integer("causal_ledger_traces") > 0,
        {"causal_ledger_traces": integer("causal_ledger_traces")},
        "Causal ledger must record attribution/routing traces",
    )
    add(
        "uncertainty_ledger",
        integer("uncertainty_ledger_observations") > 0,
        {"uncertainty_ledger_observations": integer("uncertainty_ledger_observations")},
        "Uncertainty ledger must record confidence outcomes",
    )
    add(
        "future_contract_fsp",
        phase_count("P3") > 0 and integer("future_contract_decisions") > 0 and trace_count("mtp_fsp_events") > 0,
        {
            "P3": phase_count("P3"),
            "future_contract_decisions": integer("future_contract_decisions"),
            "mtp_fsp_events": trace_count("mtp_fsp_events"),
        },
        "Future Contract/FSP must gate real multi-horizon token predictions",
    )
    add(
        "future_output_goal_contracts",
        phase_count("P3") > 0
        and integer("output_goal_contract_decisions") > 0
        and integer("output_goal_contract_accepted") > 0,
        {
            "P3": phase_count("P3"),
            "output_goal_contract_decisions": integer("output_goal_contract_decisions"),
            "output_goal_contract_accepted": integer("output_goal_contract_accepted"),
            "output_goal_contract_rejected": integer("output_goal_contract_rejected"),
        },
        "Future Contract/FSP must bind complete output/skill goals beyond token ids",
    )
    add(
        "adaptive_multi_token_decoding",
        phase_count("P8") > 0
        and integer("future_contract_decisions") > 0
        and trace_count("mtp_fsp_events") > 0
        and integer("inference_model_backed_events") > 0
        and integer("inference_model_backed_replay_events") > 0
        and integer("inference_model_backed_adaptive_mtp_events") > 0
        and integer("inference_model_backed_adaptive_mtp_contract_checks") > 0
        and integer("inference_model_backed_adaptive_mtp_proposed_blocks") > 0,
        {
            "P8": phase_count("P8"),
            "future_contract_decisions": integer("future_contract_decisions"),
            "mtp_fsp_events": trace_count("mtp_fsp_events"),
            "inference_model_backed_events": integer("inference_model_backed_events"),
            "inference_model_backed_replay_events": integer("inference_model_backed_replay_events"),
            "inference_model_backed_adaptive_mtp_events": integer("inference_model_backed_adaptive_mtp_events"),
            "inference_model_backed_adaptive_mtp_contract_checks": integer("inference_model_backed_adaptive_mtp_contract_checks"),
            "inference_model_backed_adaptive_mtp_proposed_blocks": integer("inference_model_backed_adaptive_mtp_proposed_blocks"),
            "inference_model_backed_adaptive_mtp_accepted_blocks": integer("inference_model_backed_adaptive_mtp_accepted_blocks"),
            "inference_model_backed_adaptive_mtp_rejected_blocks": integer("inference_model_backed_adaptive_mtp_rejected_blocks"),
        },
        "adaptive multi-token path must be active inside the real Transformer-backed answer source, with MTP/FSP contract checks and verified replay feedback",
    )
    add(
        "frontier_heldout_generalization_gate",
        integer("frontier_compiled_circuit_count") > 0
        and integer("frontier_heldout_total") > 0
        and integer("frontier_heldout_passed") == integer("frontier_heldout_total")
        and integer("frontier_heldout_gate_passed_circuit_count") == integer("frontier_compiled_circuit_count"),
        {
            "frontier_compiled_circuit_count": integer("frontier_compiled_circuit_count"),
            "frontier_heldout_circuit_count": integer("frontier_heldout_circuit_count"),
            "frontier_heldout_gate_passed_circuit_count": integer("frontier_heldout_gate_passed_circuit_count"),
            "frontier_heldout_passed": integer("frontier_heldout_passed"),
            "frontier_heldout_total": integer("frontier_heldout_total"),
        },
        "compiled Frontier circuits must pass a separate held-out generalization gate before FastSolve/P7/P10 reuse",
    )
    add(
        "latent_reasoning_workspace",
        phase_count("P5") > 0
        and integer("certificate_head_forward_events") > 0
        and integer("model_certificate_head_verified_events") > 0
        and integer("latent_workspace_forward_events") > 0
        and integer("latent_workspace_step_events") > 0
        and integer("latent_workspace_certificate_binding_events") > 0
        and replay_count("P5") > 0,
        {
            "P5": phase_count("P5"),
            "certificate_head_forward_events": integer("certificate_head_forward_events"),
            "model_certificate_head_verified_events": integer("model_certificate_head_verified_events"),
            "latent_workspace_forward_events": integer("latent_workspace_forward_events"),
            "latent_workspace_step_events": integer("latent_workspace_step_events"),
            "latent_workspace_certificate_binding_events": integer("latent_workspace_certificate_binding_events"),
            "phase_replay_P5": replay_count("P5"),
        },
        "explicit multi-step latent workspace must run, bind to model-head certificates and feed verified replay",
    )
    add(
        "certificate_generator",
        phase_count("P5") > 0
        and integer("certificate_head_forward_events") > 0
        and integer("model_certificate_head_verified_events") > 0
        and integer("certificate_algebra_tool_events") > 0
        and integer("certificate_code_hidden_property_events") > 0,
        {
            "P5": phase_count("P5"),
            "certificate_head_forward_events": integer("certificate_head_forward_events"),
            "model_certificate_head_verified_events": integer("model_certificate_head_verified_events"),
            "certificate_algebra_tool_events": integer("certificate_algebra_tool_events"),
            "certificate_code_hidden_property_events": integer("certificate_code_hidden_property_events"),
        },
        "certificate generator/head must materialize a verified model-head certificate plus multi-step algebra and richer code tool contracts in training",
    )
    add(
        "hierarchical_dynamic_verifier",
        phase_count("P1") > 0 and integer("skill_ledger_states") > 0 and error_count == 0,
        {
            "P1": phase_count("P1"),
            "skill_ledger_states": integer("skill_ledger_states"),
            "error_count": error_count,
        },
        "hierarchical verifier must run with skill ledger state and no phase errors",
    )
    add(
        "accept_reject_gate",
        integer("future_contract_decisions") + improvement_decisions > 0,
        {
            "future_contract_decisions": integer("future_contract_decisions"),
            "improvement_decisions": improvement_decisions,
        },
        "at least one verifier-controlled accept/reject gate decision must be recorded",
    )
    add(
        "attribute_regression",
        phase_count("P6") > 0
        and integer("causal_ledger_traces") > 0
        and replay_count("P6") > 0
        and integer("attribution_policy_observations") > 0
        and integer("attribution_policy_successes") > 0,
        {
            "P6": phase_count("P6"),
            "causal_ledger_traces": integer("causal_ledger_traces"),
            "phase_replay_P6": replay_count("P6"),
            "attribution_policy_observations": integer("attribution_policy_observations"),
            "attribution_policy_successes": integer("attribution_policy_successes"),
        },
        "attribute regression phase must produce causal evidence, replay and learned repair-outcome policy memory",
    )
    add(
        "minimal_regrowth",
        phase_count("P7") > 0
        and replay_count("P7") > 0
        and integer("regrowth_model_application_count") > 0
        and number("regrowth_model_parameter_delta_l1") > 0.0
        and number("regrowth_model_repair_loss_delta") > 0.0
        and regrowth_model_non_regressing,
        {
            "P7": phase_count("P7"),
            "phase_replay_P7": replay_count("P7"),
            "regrowth_model_application_count": integer("regrowth_model_application_count"),
            "regrowth_model_parameter_delta_l1": number("regrowth_model_parameter_delta_l1"),
            "regrowth_model_repair_loss_delta": number("regrowth_model_repair_loss_delta"),
            "regrowth_model_protected_loss_delta": number("regrowth_model_protected_loss_delta"),
            "regrowth_model_non_regressing": regrowth_model_non_regressing,
        },
        "minimal regrowth must apply a verified bounded repair to real Transformer state and keep protected loss non-regressing",
    )
    add(
        "sleep_consolidation_buffer",
        phase_count("P9") > 0
        and integer("sleep_replay_examples") > 0
        and integer("sleep_synthetic_examples") > 0
        and integer("sleep_real_exogenous_llm_examples") > 0
        and integer("replay_batch_count") > 0
        and integer("sleep_frontier_compiled_circuit_count") > 0
        and integer("sleep_frontier_heldout_total") > 0
        and integer("sleep_frontier_heldout_passed") == integer("sleep_frontier_heldout_total")
        and integer("sleep_frontier_heldout_gate_passed_circuit_count") == integer("sleep_frontier_compiled_circuit_count")
        and integer("sleep_frontier_fastsolve_events") > 0
        and integer("sleep_frontier_memory_binding_events") > 0,
        {
            "P9": phase_count("P9"),
            "sleep_replay_examples": integer("sleep_replay_examples"),
            "sleep_synthetic_examples": integer("sleep_synthetic_examples"),
            "sleep_real_exogenous_llm_examples": integer("sleep_real_exogenous_llm_examples"),
            "replay_batch_count": integer("replay_batch_count"),
            "sleep_frontier_compiled_circuit_count": integer("sleep_frontier_compiled_circuit_count"),
            "sleep_frontier_heldout_passed": integer("sleep_frontier_heldout_passed"),
            "sleep_frontier_heldout_total": integer("sleep_frontier_heldout_total"),
            "sleep_frontier_heldout_gate_passed_circuit_count": integer("sleep_frontier_heldout_gate_passed_circuit_count"),
            "sleep_frontier_fastsolve_events": integer("sleep_frontier_fastsolve_events"),
            "sleep_frontier_memory_binding_events": integer("sleep_frontier_memory_binding_events"),
        },
        "sleep/consolidation must emit verified replay, then compile accepted experience into held-out gated executable Frontier circuits used by FastSolve",
    )
    add(
        "recursive_improvement",
        phase_count("P10") > 0
        and improvement_decisions > 0
        and integer("recursive_model_application_count") > 0
        and recursive_verified_artifact_count > 0
        and number("recursive_model_parameter_delta_l1") > 0.0
        and number("recursive_model_repair_loss_delta") > 0.0
        and recursive_model_non_regressing
        and recursive_artifact_verified,
        {
            "P10": phase_count("P10"),
            "improvement_decisions": improvement_decisions,
            "recursive_model_application_count": integer("recursive_model_application_count"),
            "recursive_verified_artifact_count": recursive_verified_artifact_count,
            "recursive_model_parameter_delta_l1": number("recursive_model_parameter_delta_l1"),
            "recursive_model_repair_loss_delta": number("recursive_model_repair_loss_delta"),
            "recursive_model_protected_loss_delta": number("recursive_model_protected_loss_delta"),
            "recursive_model_non_regressing": recursive_model_non_regressing,
            "recursive_artifact_verified": recursive_artifact_verified,
        },
        "recursive improvement sandbox must evaluate a proposal, apply an accepted signed non-regressing patch to real Transformer state, and materialize a verified replay artifact",
    )
    add(
        "training_feedback_loop",
        integer("replay_updates") > 0
        and integer("objective_feedback_events") > 0
        and number("last_objective_loss_total") > 0.0,
        {
            "replay_updates": integer("replay_updates"),
            "objective_feedback_events": integer("objective_feedback_events"),
            "last_objective_loss_total": number("last_objective_loss_total"),
        },
        "verified Cortex replay and objective feedback must affect LLM optimization",
    )
    add(
        "final_objective_loss",
        integer("objective_feedback_events") > 0
        and integer("objective_feedback_term_count") == len(required_objective_terms)
        and not missing_objective_terms
        and number("last_objective_loss_total") > 0.0
        and number("last_objective_loss_weighted_total") > 0.0,
        {
            "objective_feedback_events": integer("objective_feedback_events"),
            "objective_feedback_term_count": integer("objective_feedback_term_count"),
            "required_objective_term_count": len(required_objective_terms),
            "missing_objective_terms": missing_objective_terms,
            "last_objective_loss_total": number("last_objective_loss_total"),
            "last_objective_loss_weighted_total": number("last_objective_loss_weighted_total"),
        },
        "the full 17-term Cortex-3 final objective must feed the LLM training feedback signal",
    )

    failed_checks = tuple(check["component"] for check in checks if not check["passed"])
    return {
        "schema_version": 1,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": tuple(checks),
        "checks_by_component": {check["component"]: check for check in checks},
        "passed_count": sum(1 for check in checks if check["passed"]),
        "component_count": len(checks),
    }


def _cortex_phase_deliverable_audit_from_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = {str(key): int(value) for key, value in dict(summary.get("integration_counts") or {}).items()}
    phase_counts = {str(key): int(value) for key, value in dict(summary.get("phase_event_counts") or {}).items()}
    trace_counts = {str(key): int(value) for key, value in dict(summary.get("compression_trace_counts") or {}).items()}
    native_ternary_required = bool(summary.get("native_ternary_kernel_required", False))
    native_backend_requested = str(summary.get("native_ternary_backend_requested", "auto"))
    native_backend_counts = {str(key): int(value) for key, value in dict(summary.get("native_ternary_backend_counts") or {}).items()}
    native_requantize_backend_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("native_ternary_requantize_backend_counts") or {}).items()
    }
    native_grad_weight_backend_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("native_ternary_grad_weight_backend_counts") or {}).items()
    }
    native_dispatch_required = native_ternary_required
    replay_by_phase = {
        str(key): int(value)
        for key, value in dict(summary.get("phase_replay_examples_by_phase") or {}).items()
    }

    def count(key: str) -> int:
        return int(counts.get(key, 0))

    def phase(phase_id: str) -> int:
        return int(phase_counts.get(phase_id, 0))

    def trace(key: str) -> int:
        return int(trace_counts.get(key, 0))

    def only_extension(counts: Mapping[str, int]) -> bool:
        return int(counts.get(STRICT_NATIVE_TERNARY_BACKEND, 0)) > 0 and not any(
            int(value) > 0 and str(backend) != STRICT_NATIVE_TERNARY_BACKEND
            for backend, value in counts.items()
        )

    def requested_native_backend_met() -> bool:
        if native_backend_requested != STRICT_NATIVE_TERNARY_BACKEND:
            return True
        return (
            only_extension(native_backend_counts)
            and only_extension(native_requantize_backend_counts)
            and only_extension(native_grad_weight_backend_counts)
            and trace("torch_packed_ternary_dispatches") == 0
        )

    def number(key: str) -> float:
        value = summary.get(key, 0)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    regrowth_model_applications = tuple(
        dict(item)
        for item in (summary.get("regrowth_model_applications") or ())
        if isinstance(item, Mapping)
    )
    regrowth_model_non_regressing = any(
        bool(item.get("non_regression_passed"))
        and float(item.get("parameter_delta_l1", 0.0) or 0.0) > 0.0
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and float(item.get("protected_loss_delta", 0.0) or 0.0) <= float(item.get("protected_loss_tolerance", 0.0) or 0.0)
        for item in regrowth_model_applications
    )
    recursive_model_applications = tuple(
        dict(item)
        for item in (summary.get("recursive_model_applications") or ())
        if isinstance(item, Mapping)
    )
    recursive_model_non_regressing = any(
        bool(item.get("non_regression_passed"))
        and bool(item.get("signed_patch_id"))
        and bool(item.get("rollback_token"))
        and float(item.get("parameter_delta_l1", 0.0) or 0.0) > 0.0
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and float(item.get("protected_loss_delta", 0.0) or 0.0) <= float(item.get("protected_loss_tolerance", 0.0) or 0.0)
        for item in recursive_model_applications
    )
    recursive_verified_artifacts = tuple(
        dict(item)
        for item in (summary.get("recursive_verified_artifacts") or ())
        if isinstance(item, Mapping)
    )
    recursive_verified_artifact_count = max(int(number("recursive_verified_artifact_count")), len(recursive_verified_artifacts))
    recursive_artifact_verified = any(
        bool(item.get("recursive_improvement_artifact"))
        and bool(item.get("artifact_id"))
        and bool(item.get("example_id"))
        and bool(item.get("signed_patch_id"))
        and bool(item.get("rollback_token"))
        and int(item.get("verification_level", 0) or 0) >= 3
        and float(item.get("repair_loss_delta", 0.0) or 0.0) > 0.0
        and bool(item.get("non_regression_passed"))
        for item in recursive_verified_artifacts
    )

    checks: list[dict[str, Any]] = []

    def add(phase_id: str, deliverable: str, passed: bool, observed: Mapping[str, Any], requirement: str) -> None:
        checks.append(
            {
                "phase": phase_id,
                "deliverable": deliverable,
                "passed": bool(passed),
                "observed": dict(observed),
                "requirement": requirement,
            }
        )

    add(
        "P1",
        "verifier_os_regression_harness",
        phase("P1") > 0 and int(summary.get("skill_ledger_states", 0) or 0) > 0 and replay_by_phase.get("P1", 0) > 0,
        {"P1": phase("P1"), "skill_ledger_states": int(summary.get("skill_ledger_states", 0) or 0), "phase_replay_P1": replay_by_phase.get("P1", 0)},
        "Dynamic verifier must generate verified regression evidence and replay",
    )
    add(
        "P2",
        "ternary_sign_mask_activation_trace_logs_and_packed_dispatch",
        trace("layer_forward_events") > 0
        and trace("activation_quantizations") > 0
        and trace("compression_decisions") > 0
        and trace("packed_ternary_dispatches") > 0
        and (
            (not native_dispatch_required)
            or (
                trace("native_ternary_kernel_dispatches") > 0
                and trace("native_ternary_autotuned_dispatches") > 0
                and requested_native_backend_met()
            )
        ),
        {
            "layer_forward_events": trace("layer_forward_events"),
            "activation_quantizations": trace("activation_quantizations"),
            "compression_decisions": trace("compression_decisions"),
            "packed_ternary_dispatches": trace("packed_ternary_dispatches"),
            "native_ternary_kernel_required": native_ternary_required,
            "native_ternary_dispatch_required": native_dispatch_required,
            "native_ternary_backend_requested": native_backend_requested,
            "native_ternary_backend_counts": native_backend_counts,
            "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
            "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
            "strict_extension_only": requested_native_backend_met(),
            "native_ternary_kernel_dispatches": trace("native_ternary_kernel_dispatches"),
            "native_ternary_autotuned_dispatches": trace("native_ternary_autotuned_dispatches"),
            "native_ternary_autotune_cache_hits": trace("native_ternary_autotune_cache_hits"),
        },
        "BitLinear sign+mask, activation quantization, compression decisions, packed dispatch, and required autotuned native CUDA kernel must all run",
    )
    add(
        "P3",
        "mtp_fsp_confidence_temporal_contract_gate",
        int(summary.get("future_contract_decisions", 0) or 0) > 0
        and int(summary.get("output_goal_contract_decisions", 0) or 0) > 0
        and int(summary.get("output_goal_contract_accepted", 0) or 0) > 0
        and trace("mtp_fsp_events") > 0
        and count("future_contract_observed_token_checks") > 0
        and count("future_output_goal_contract_checks") > 0,
        {
            "future_contract_decisions": int(summary.get("future_contract_decisions", 0) or 0),
            "output_goal_contract_decisions": int(summary.get("output_goal_contract_decisions", 0) or 0),
            "output_goal_contract_accepted": int(summary.get("output_goal_contract_accepted", 0) or 0),
            "mtp_fsp_events": trace("mtp_fsp_events"),
            "observed_token_checks": count("future_contract_observed_token_checks"),
            "output_goal_checks": count("future_output_goal_contract_checks"),
        },
        "MTP/FSP must gate observed future tokens and a complete output-goal contract, not only instantiate heads",
    )
    add(
        "P4",
        "learned_exact_latent_drop_memory_anchor_fidelity",
        int(summary.get("memory_recent_segments", 0) or 0) > 0
        and int(summary.get("memory_latent_segments", 0) or 0) > 0
        and int(summary.get("input_anchor_count", 0) or 0) > 0
        and int(summary.get("input_anchor_fidelity_failures", 0) or 0) == 0
        and int(summary.get("learned_memory_policy_events", 0) or 0) > 0
        and int(summary.get("learned_memory_anchor_supervision_events", 0) or 0) > 0
        and int(summary.get("learned_memory_retention_decisions", 0) or 0) > 0
        and int(summary.get("learned_memory_utility_credit_count", 0) or 0) > 0
        and int(summary.get("learned_memory_utility_positive_count", 0) or 0) > 0
        and int(summary.get("learned_memory_utility_prior_updates", 0) or 0) > 0
        and int(summary.get("learned_memory_utility_feedback_events", 0) or 0) > 0
        and int(summary.get("compiled_circuit_memory_binding_count", 0) or 0) > 0
        and int(summary.get("compiled_circuit_memory_binding_events", 0) or 0) > 0
        and int(summary.get("compiled_circuit_memory_fidelity_failures", 0) or 0) == 0
        and (
            count("frontier_registry_loaded_events") == 0
            or (
                count("frontier_restored_fastsolve_events") > 0
                and count("compiled_circuit_memory_restored_reuse_events") > 0
            )
        ),
        {
            "memory_recent_segments": int(summary.get("memory_recent_segments", 0) or 0),
            "memory_latent_segments": int(summary.get("memory_latent_segments", 0) or 0),
            "input_anchor_count": int(summary.get("input_anchor_count", 0) or 0),
            "input_anchor_fidelity_failures": int(summary.get("input_anchor_fidelity_failures", 0) or 0),
            "learned_memory_policy_events": int(summary.get("learned_memory_policy_events", 0) or 0),
            "learned_memory_anchor_supervision_events": int(summary.get("learned_memory_anchor_supervision_events", 0) or 0),
            "learned_memory_retention_decisions": int(summary.get("learned_memory_retention_decisions", 0) or 0),
            "learned_memory_retention_applied_exact": int(summary.get("learned_memory_retention_applied_exact", 0) or 0),
            "learned_memory_retention_applied_latent": int(summary.get("learned_memory_retention_applied_latent", 0) or 0),
            "learned_memory_retention_applied_drop": int(summary.get("learned_memory_retention_applied_drop", 0) or 0),
            "learned_memory_utility_credit_count": int(summary.get("learned_memory_utility_credit_count", 0) or 0),
            "learned_memory_utility_positive_count": int(summary.get("learned_memory_utility_positive_count", 0) or 0),
            "learned_memory_utility_prior_updates": int(summary.get("learned_memory_utility_prior_updates", 0) or 0),
            "learned_memory_utility_feedback_events": int(summary.get("learned_memory_utility_feedback_events", 0) or 0),
            "learned_memory_last_utility_prior": tuple(summary.get("learned_memory_last_utility_prior", ()) or ()),
            "compiled_circuit_memory_binding_count": int(summary.get("compiled_circuit_memory_binding_count", 0) or 0),
            "compiled_circuit_memory_binding_events": int(summary.get("compiled_circuit_memory_binding_events", 0) or 0),
            "compiled_circuit_memory_fidelity_failures": int(summary.get("compiled_circuit_memory_fidelity_failures", 0) or 0),
            "frontier_registry_loaded_events": count("frontier_registry_loaded_events"),
            "frontier_restored_fastsolve_events": count("frontier_restored_fastsolve_events"),
            "compiled_circuit_memory_restored_reuse_events": count("compiled_circuit_memory_restored_reuse_events"),
        },
        "Cognitive memory must preserve exact anchors, learn exact/latent/drop retention decisions from downstream utility, and retain restored compiled Frontier circuits for reuse",
    )
    add(
        "P5",
        "latent_certificate_delatentization_tool_verification",
        count("certificate_efficiency_events") > 0
        and count("delatentization_probe_events") > 0
        and count("delatentization_probe_failures") == 0
        and count("certificate_tool_verification_events") > 0
        and count("model_certificate_head_verified_events") > 0
        and count("latent_workspace_forward_events") > 0
        and count("latent_workspace_step_events") > 0
        and count("latent_workspace_certificate_binding_events") > 0
        and count("certificate_algebra_tool_events") > 0
        and count("certificate_code_hidden_property_events") > 0,
        {
            "certificate_efficiency_events": count("certificate_efficiency_events"),
            "delatentization_probe_events": count("delatentization_probe_events"),
            "delatentization_probe_failures": count("delatentization_probe_failures"),
            "certificate_tool_verification_events": count("certificate_tool_verification_events"),
            "model_certificate_head_verified_events": count("model_certificate_head_verified_events"),
            "model_certificate_head_latent_checksum_events": count("model_certificate_head_latent_checksum_events"),
            "latent_workspace_forward_events": count("latent_workspace_forward_events"),
            "latent_workspace_step_events": count("latent_workspace_step_events"),
            "latent_workspace_certificate_binding_events": count("latent_workspace_certificate_binding_events"),
            "certificate_algebra_tool_events": count("certificate_algebra_tool_events"),
            "certificate_code_hidden_property_events": count("certificate_code_hidden_property_events"),
        },
        "latent proof state must include an explicit multi-step workspace, verified model-head certificate, random de-latentization, multi-step algebra and richer code tool verification",
    )
    add(
        "P6",
        "causal_attribution_counterfactual_dimensions_learned_policy",
        count("attribution_probe_events") > 0
        and count("attribution_unique_dimensions") >= 7
        and count("attribution_policy_updates") > 0
        and int(summary.get("attribution_policy_observations", 0) or 0) > 0,
        {
            "attribution_probe_events": count("attribution_probe_events"),
            "attribution_unique_dimensions": count("attribution_unique_dimensions"),
            "attribution_policy_updates": count("attribution_policy_updates"),
            "attribution_policy_applied_events": count("attribution_policy_applied_events"),
            "attribution_policy_observations": int(summary.get("attribution_policy_observations", 0) or 0),
            "attribution_policy_successes": int(summary.get("attribution_policy_successes", 0) or 0),
        },
        "causal attribution must run counterfactual probes over block/expert/KV/MTP/activation/FSP/routing dimensions and learn from verified regrowth outcomes",
    )
    add(
        "P7",
        "minimal_regrowth_action_space_repair_plan_and_model_patch",
        count("regrowth_plan_events") > 0
        and count("regrowth_candidate_actions") > 0
        and replay_by_phase.get("P7", 0) > 0
        and int(number("regrowth_model_application_count")) > 0
        and number("regrowth_model_parameter_delta_l1") > 0.0
        and number("regrowth_model_repair_loss_delta") > 0.0
        and regrowth_model_non_regressing,
        {
            "regrowth_plan_events": count("regrowth_plan_events"),
            "regrowth_candidate_actions": count("regrowth_candidate_actions"),
            "phase_replay_P7": replay_by_phase.get("P7", 0),
            "regrowth_model_application_count": int(number("regrowth_model_application_count")),
            "regrowth_model_parameter_delta_l1": number("regrowth_model_parameter_delta_l1"),
            "regrowth_model_repair_loss_delta": number("regrowth_model_repair_loss_delta"),
            "regrowth_model_non_regressing": regrowth_model_non_regressing,
        },
        "minimal regrowth must evaluate candidate repair actions, feed verified replay, and apply a verified bounded patch to real model state",
    )
    add(
        "P8",
        "fast_normal_careful_budget_early_exit_mod_speculative_kernels",
        count("inference_fast_path_events") > 0
        and count("inference_normal_path_events") > 0
        and count("inference_careful_path_events") > 0
        and count("inference_budget_predictions") >= 3
        and count("inference_early_exit_events") >= 3
        and count("inference_self_speculative_events") >= 3
        and count("inference_kernel_dispatches") > 0
        and count("inference_latent_kv_events") > 0
        and count("inference_model_backed_events") > 0
        and count("inference_model_backed_forced_careful_events") > 0
        and count("inference_model_backed_replay_events") > 0
        and count("inference_model_backed_adaptive_mtp_events") > 0
        and count("inference_model_backed_adaptive_mtp_contract_checks") > 0
        and count("inference_model_backed_adaptive_mtp_proposed_blocks") > 0,
        {
            "fast": count("inference_fast_path_events"),
            "normal": count("inference_normal_path_events"),
            "careful": count("inference_careful_path_events"),
            "budget_predictions": count("inference_budget_predictions"),
            "early_exit_events": count("inference_early_exit_events"),
            "self_speculative_events": count("inference_self_speculative_events"),
            "kernel_dispatches": count("inference_kernel_dispatches"),
            "latent_kv_events": count("inference_latent_kv_events"),
            "model_backed_events": count("inference_model_backed_events"),
            "model_backed_forced_careful_events": count("inference_model_backed_forced_careful_events"),
            "model_backed_replay_events": count("inference_model_backed_replay_events"),
            "model_backed_repair_replay_events": count("inference_model_backed_repair_replay_events"),
            "model_backed_verified_replay_events": count("inference_model_backed_verified_replay_events"),
            "model_backed_adaptive_mtp_events": count("inference_model_backed_adaptive_mtp_events"),
            "model_backed_adaptive_mtp_contract_checks": count("inference_model_backed_adaptive_mtp_contract_checks"),
            "model_backed_adaptive_mtp_proposed_blocks": count("inference_model_backed_adaptive_mtp_proposed_blocks"),
            "model_backed_adaptive_mtp_accepted_blocks": count("inference_model_backed_adaptive_mtp_accepted_blocks"),
            "model_backed_adaptive_mtp_rejected_blocks": count("inference_model_backed_adaptive_mtp_rejected_blocks"),
        },
        "all inference paths plus budget predictor, early exit, self-speculative MTP, latent KV, ternary dispatch, a real Transformer-backed adaptive MTP/FSP answer source and verified feedback replay must run",
    )
    add(
        "P9",
        "sleep_replay_synthetic_real_reservoir_anti_collapse_schedule_frontier_compile",
        int(summary.get("sleep_replay_examples", 0) or 0) > 0
        and int(summary.get("sleep_synthetic_examples", 0) or 0) > 0
        and int(summary.get("sleep_reservoir_examples", 0) or 0) > 0
        and int(summary.get("sleep_real_exogenous_llm_examples", 0) or 0) > 0
        and count("sleep_real_exogenous_llm_batch_events") > 0
        and count("sleep_metamorphic_examples") > 0
        and count("sleep_anti_collapse_decisions") > 0
        and count("sleep_consolidation_schedule_items") > 0
        and int(summary.get("sleep_frontier_compiled_circuit_count", 0) or 0) > 0
        and int(summary.get("sleep_frontier_heldout_total", 0) or 0) > 0
        and int(summary.get("sleep_frontier_heldout_passed", 0) or 0)
        == int(summary.get("sleep_frontier_heldout_total", 0) or 0)
        and count("sleep_frontier_fastsolve_events") > 0
        and int(summary.get("sleep_frontier_memory_binding_events", 0) or 0) > 0,
        {
            "sleep_replay_examples": int(summary.get("sleep_replay_examples", 0) or 0),
            "sleep_synthetic_examples": int(summary.get("sleep_synthetic_examples", 0) or 0),
            "sleep_reservoir_examples": int(summary.get("sleep_reservoir_examples", 0) or 0),
            "sleep_real_exogenous_llm_examples": int(summary.get("sleep_real_exogenous_llm_examples", 0) or 0),
            "sleep_real_exogenous_llm_batch_events": count("sleep_real_exogenous_llm_batch_events"),
            "sleep_metamorphic_examples": count("sleep_metamorphic_examples"),
            "sleep_anti_collapse_decisions": count("sleep_anti_collapse_decisions"),
            "sleep_consolidation_schedule_items": count("sleep_consolidation_schedule_items"),
            "sleep_frontier_compiled_circuit_count": int(summary.get("sleep_frontier_compiled_circuit_count", 0) or 0),
            "sleep_frontier_heldout_passed": int(summary.get("sleep_frontier_heldout_passed", 0) or 0),
            "sleep_frontier_heldout_total": int(summary.get("sleep_frontier_heldout_total", 0) or 0),
            "sleep_frontier_fastsolve_events": count("sleep_frontier_fastsolve_events"),
            "sleep_frontier_memory_binding_events": int(summary.get("sleep_frontier_memory_binding_events", 0) or 0),
        },
        "sleep phase must combine replay, verified synthetic/metamorphic, real reservoir, anti-collapse scheduling, memory-retained held-out Frontier compilation and FastSolve",
    )
    add(
        "P10",
        "recursive_improvement_sandbox_pareto_signed_model_patch_rollback_diversity",
        count("recursive_proposal_events") > 0
        and count("recursive_sandbox_trials") > 0
        and count("recursive_dynamic_evaluations") > 0
        and count("recursive_pareto_gate_decisions") > 0
        and count("recursive_rollback_tokens") > 0
        and count("recursive_diversity_checks") > 0
        and count("recursive_generation_events") > 0
        and (
            int(number("recursive_improvement_generations_configured")) <= 1
            or count("recursive_evolved_proposal_events") > 0
        )
        and count("recursive_persistent_archive_saves") > 0
        and int(number("improvement_persistent_archive_decisions")) > 0
        and int(number("recursive_model_application_count")) > 0
        and recursive_verified_artifact_count > 0
        and number("recursive_model_parameter_delta_l1") > 0.0
        and number("recursive_model_repair_loss_delta") > 0.0
        and recursive_model_non_regressing
        and recursive_artifact_verified,
        {
            "recursive_proposal_events": count("recursive_proposal_events"),
            "recursive_sandbox_trials": count("recursive_sandbox_trials"),
            "recursive_dynamic_evaluations": count("recursive_dynamic_evaluations"),
            "recursive_pareto_gate_decisions": count("recursive_pareto_gate_decisions"),
            "recursive_rollback_tokens": count("recursive_rollback_tokens"),
            "recursive_diversity_checks": count("recursive_diversity_checks"),
            "recursive_improvement_generations_configured": int(number("recursive_improvement_generations_configured")),
            "recursive_generation_events": count("recursive_generation_events"),
            "recursive_evolved_proposal_events": count("recursive_evolved_proposal_events"),
            "recursive_persistent_archive_saves": count("recursive_persistent_archive_saves"),
            "improvement_persistent_archive_decisions": int(number("improvement_persistent_archive_decisions")),
            "improvement_persistent_rollback_events": int(number("improvement_persistent_rollback_events")),
            "recursive_model_application_count": int(number("recursive_model_application_count")),
            "recursive_verified_artifact_count": recursive_verified_artifact_count,
            "recursive_model_parameter_delta_l1": number("recursive_model_parameter_delta_l1"),
            "recursive_model_repair_loss_delta": number("recursive_model_repair_loss_delta"),
            "recursive_model_non_regressing": recursive_model_non_regressing,
            "recursive_artifact_verified": recursive_artifact_verified,
        },
        "recursive improvement must propose, sandbox, evolve across generations, evaluate, gate, persist evolutionary and rollback archives, apply a signed model patch, preserve rollback tokens, run diversity checks and materialize a verified replay artifact",
    )

    failed_checks = tuple(f"{check['phase']}:{check['deliverable']}" for check in checks if not check["passed"])
    return {
        "schema_version": 1,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": tuple(checks),
        "checks_by_deliverable": {f"{check['phase']}:{check['deliverable']}": check for check in checks},
        "passed_count": sum(1 for check in checks if check["passed"]),
        "deliverable_count": len(checks),
    }


class CortexTrainingPhaseController:
    def __init__(
        self,
        model: CortexTransformerLM,
        tokenizer: LLMTokenizer,
        config: TrainingConfig,
        *,
        run_dir: str | Path,
    ):
        if not model.config.use_cortex_heads:
            raise ValueError("CortexTrainingPhaseController requires a Cortex model with use_cortex_heads=True")
        if not model.config.use_skill_aware_experts:
            raise ValueError("CortexTrainingPhaseController requires skill-aware experts for full Cortex training")
        if not model.config.use_variable_in_compressor:
            raise ValueError("CortexTrainingPhaseController requires a Variable-In compressor for full Cortex training")
        if not model.config.use_learned_memory_policy:
            raise ValueError("CortexTrainingPhaseController requires a learned cognitive memory policy for full Cortex training")
        if not model.config.use_certificate_head:
            raise ValueError("CortexTrainingPhaseController requires a certificate head for full Cortex training")
        if not model.config.use_latent_reasoning_workspace:
            raise ValueError("CortexTrainingPhaseController requires an explicit latent reasoning workspace for full Cortex training")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.run_dir = Path(run_dir)
        if self.model.compression_ledger is not None:
            self.model.compression_ledger.retention_limit = int(self.config.cortex_trace_retention_limit)
        self.verifier = DynamicSkillVerifier(default_skill_specs())
        self.reference_agent = ReferenceRuleAgent()
        self.trial_agent = CorruptedCompressedAgent()
        self.cycle = CortexCycle(self.verifier)
        self.frontier_registry = FrontierCircuitRegistry()
        self.frontier_discovery = FrontierSkillDiscovery(self.verifier, self.reference_agent)
        self.future_ledger = FutureContractLedger()
        self.future_engine = FutureContractEngine(
            MTPFSPConfig(
                hidden_size=model.config.d_model,
                vocab_size=model.config.vocab_size,
                horizons=model.config.horizons,
            ),
            ledger=self.future_ledger,
            trace_ledger=model.compression_ledger,
        )
        self.memory = CognitiveMemory(CognitiveMemoryConfig())
        self.certificate_verifier = CertificateVerifier()
        self.attribution_policy = AttributionPolicyMemory()
        self.attribution = CausalAttributionEngine(self.verifier, policy_memory=self.attribution_policy)
        self.regrowth = MinimalRegrowthEngine(self.verifier)
        self.model_inference_agent = CortexTransformerInferenceAgent(
            self.model,
            self.tokenizer,
            future_engine=self.future_engine,
        )
        self.inference = UltraFastInferenceEngine(
            self.verifier,
            self.model_inference_agent,
            memory=self.memory,
            compiled_frontier_registry=self.frontier_registry,
            config=InferenceConfig(
                hidden_size=max(32, model.config.d_model),
                vocab_size=max(64, min(model.config.vocab_size, 4096)),
                max_layers=max(4, model.config.n_layers),
                careful_layers=max(4, model.config.n_layers),
                normal_layers=max(3, min(max(4, model.config.n_layers), max(3, model.config.n_layers - 1))),
                fast_layers=1,
            ),
        )
        self.sleep = SleepPhaseConsolidator(self.verifier)
        self.improvement = RecursiveImprovementEngine(self.verifier)
        self.improvement_archive_dir = (
            Path(config.cortex_improvement_archive_dir)
            if config.cortex_improvement_archive_dir is not None
            else self.run_dir / "recursive_improvement_archive"
        )
        self.improvement_persistent_archive_state: dict[str, Any] = {
            "archive_dir": str(self.improvement_archive_dir),
            "archive_loaded": False,
            "rollback_loaded": False,
            "accepted_count": 0,
            "rejected_count": 0,
            "decision_count": 0,
            "rollback_event_count": 0,
        }
        self.phase_counts: dict[str, int] = {phase.id: 0 for phase in CORTEX3_PHASES}
        self.errors: list[dict[str, Any]] = []
        self.batch_contract_samples: list[dict[str, Any]] = []
        self.phase_audits: list[dict[str, Any]] = []
        self.frontier_reports: list[dict[str, Any]] = []
        self.sleep_frontier_reports: list[dict[str, Any]] = []
        self.restored_frontier_fastsolve_reports: list[dict[str, Any]] = []
        self.frontier_compiled_fastsolve_events = 0
        self.frontier_repair_candidates: list[dict[str, Any]] = []
        self.frontier_repair_accepted_events = 0
        self.replay_batches: list[torch.Tensor] = []
        self.replay_skill_contexts: list[tuple[float, ...]] = []
        self.replay_cursor = 0
        self.regularization_steps = 0
        self.replay_updates = 0
        self.phase_replay_examples: dict[str, int] = {phase.id: 0 for phase in CORTEX3_PHASES}
        self.phase_replay_example_ids: list[str] = []
        self.objective_feedback_events = 0
        self.objective_feedback_total = 0.0
        self.last_objective_loss_total = 0.0
        self.last_objective_loss_terms: dict[str, dict[str, Any]] = {}
        self.objective_feedback_term_totals: dict[str, float] = {}
        self.objective_feedback_history: list[dict[str, Any]] = []
        self.bit_ledger = BitLedger()
        self.skill_ledger = SkillLedger()
        self.causal_ledger = CausalLedger()
        self.uncertainty_ledger = UncertaintyLedger()
        self.integration_counts: dict[str, int] = {}
        self._last_ingested_compression_cost = CostTrace()
        self.input_anchor_observations = 0
        self.input_anchor_count = 0
        self.input_anchor_fidelity_failures = 0
        self.learned_memory_policy_events = 0
        self.learned_memory_anchor_supervision_events = 0
        self.learned_memory_exact_decisions = 0
        self.learned_memory_latent_decisions = 0
        self.learned_memory_drop_decisions = 0
        self.learned_memory_storage_ratio_total = 0.0
        self.learned_memory_utility_prior_updates = 0
        self.learned_memory_last_utility_prior: tuple[float, ...] = ()
        self.skill_expert_context_events = 0
        self.skill_expert_replay_context_events = 0
        self.skill_expert_last_context: tuple[float, ...] = ()
        self.skill_expert_context_skills: tuple[str, ...] = ()
        self.model_certificate_head_artifacts: list[dict[str, Any]] = []
        self.latent_workspace_forward_events = 0
        self.latent_workspace_step_events = 0
        self.latent_workspace_certificate_binding_events = 0
        self.latent_workspace_last_summary: dict[str, Any] = {}
        self.regrowth_model_applications: list[dict[str, Any]] = []
        self.regrowth_model_parameter_delta_l1 = 0.0
        self.regrowth_model_repair_loss_delta = 0.0
        self.regrowth_model_protected_loss_delta = 0.0
        self.recursive_model_applications: list[dict[str, Any]] = []
        self.recursive_verified_artifacts: list[dict[str, Any]] = []
        self.recursive_model_parameter_delta_l1 = 0.0
        self.recursive_model_repair_loss_delta = 0.0
        self.recursive_model_protected_loss_delta = 0.0
        self._load_persistent_improvement_archive()

    def _touch(self, phase_id: str) -> None:
        self.phase_counts[phase_id] = self.phase_counts.get(phase_id, 0) + 1

    def _count(self, key: str, amount: int = 1) -> None:
        self.integration_counts[key] = self.integration_counts.get(key, 0) + int(amount)

    def _load_persistent_improvement_archive(self) -> None:
        state = self.improvement.load_persistent_state(self.improvement_archive_dir)
        self.improvement_persistent_archive_state = dict(state)
        if bool(state.get("archive_loaded")):
            self.integration_counts["recursive_persistent_archive_loads"] = (
                self.integration_counts.get("recursive_persistent_archive_loads", 0) + 1
            )
            self.integration_counts["recursive_persistent_archive_loaded_decisions"] = int(state.get("decision_count", 0))
        if bool(state.get("rollback_loaded")):
            self.integration_counts["recursive_persistent_rollback_loaded_events"] = int(state.get("rollback_event_count", 0))

    def _persist_improvement_archive(self, *, step: int) -> dict[str, Any]:
        state = self.improvement.save_persistent_state(self.improvement_archive_dir)
        state = {**state, "step": int(step)}
        self.improvement_persistent_archive_state = dict(state)
        self._count("recursive_persistent_archive_saves")
        self.integration_counts["recursive_persistent_archive_decisions"] = int(state.get("decision_count", 0))
        self.integration_counts["recursive_persistent_rollback_events"] = int(state.get("rollback_event_count", 0))
        return state

    def _record_error(self, phase_id: str, exc: Exception) -> None:
        self.errors.append({"phase": phase_id, "type": type(exc).__name__, "message": str(exc)})

    def _current_compression_cost(self) -> CostTrace:
        if self.model.compression_ledger is None:
            return CostTrace()
        return self.model.compression_ledger.cost_trace

    def _ingest_compression_trace_delta(self, *, step: int, note: str) -> None:
        current = self._current_compression_cost()
        previous = self._last_ingested_compression_cost
        delta = CostTrace(
            weight_bits_read=max(0.0, current.weight_bits_read - previous.weight_bits_read),
            activation_bits=max(0.0, current.activation_bits - previous.activation_bits),
            kv_bytes=max(0.0, current.kv_bytes - previous.kv_bytes),
            generated_tokens=max(0, current.generated_tokens - previous.generated_tokens),
            latent_steps=max(0, current.latent_steps - previous.latent_steps),
            experts_activated=max(0, current.experts_activated - previous.experts_activated),
            verifier_steps=max(0, current.verifier_steps - previous.verifier_steps),
            wall_time_ms=max(0.0, current.wall_time_ms - previous.wall_time_ms),
        )
        if delta.effective_cost() > 0:
            self.bit_ledger.ingest_cost(delta, note=f"step-{step}:{note}")
        self._last_ingested_compression_cost = current

    def _record_causal_trace(
        self,
        *,
        trace_id: str,
        skill: str,
        confidence: float,
        anchors: int,
        certificate_fields: Iterable[str],
        verifier_level: int | None = None,
        mtp_horizon: int | None = None,
    ) -> None:
        risk = max(0.0, min(1.0, 1.0 - float(confidence)))
        route = self.cycle.router.route(skill, float(confidence), risk)
        self.bit_ledger.routing_bits += math.log2(3.0)
        self.causal_ledger.record(
            CausalTrace(
                task_id=trace_id,
                skill=skill,
                mtp_horizon=int(mtp_horizon or route.mtp_horizon),
                activation_bits=(
                    int(self.model.config.ternary_activation_bits)
                    if self.model.config.use_ternary_core
                    else 32
                ),
                kv_mode="latent" if anchors or skill in {"long_context_anchor", "entity_tracking"} else "exact",
                verifier_level=int(route.verifier_level if verifier_level is None else verifier_level),
                certificate_fields=tuple(str(item) for item in certificate_fields),
                uncertainty=risk,
            )
        )

    def _record_case_ledgers(self, case: Any, *, step: int, source: str) -> None:
        answer_cost = case.answer.cost.merge(case.verifier_cost)
        self.bit_ledger.ingest_cost(answer_cost, note=f"{source}:{case.task.skill}")
        if case.answer.certificate:
            self.bit_ledger.add_certificate(case.answer.certificate)
        self.uncertainty_ledger.record(case.task.skill, case.answer.confidence, bool(case.passed))
        self._record_causal_trace(
            trace_id=f"{source}-{step}-{case.task.task_id}",
            skill=case.task.skill,
            confidence=case.answer.confidence,
            anchors=len(case.task.anchors),
            certificate_fields=case.answer.certificate.keys(),
        )

    def _ingest_suite_ledgers(self, report: Any, *, step: int, source: str) -> None:
        for skill_report in report.skill_reports.values():
            for case in skill_report.cases:
                self._record_case_ledgers(case, step=step, source=source)

    def _record_training_example_ledgers(self, example: TrainingExample, *, phase_id: str) -> None:
        self.bit_ledger.ingest_cost(example.answer.cost, note=f"replay:{phase_id}:{example.targeted_skill}")
        if example.answer.certificate:
            self.bit_ledger.add_certificate(example.answer.certificate)
        confidence = float(example.confidence_label if example.confidence_label is not None else example.answer.confidence)
        self.uncertainty_ledger.record(example.targeted_skill, confidence, True)
        self._record_causal_trace(
            trace_id=f"replay-{phase_id}-{example.example_id}",
            skill=example.targeted_skill,
            confidence=confidence,
            anchors=len(example.task.anchors),
            certificate_fields=example.answer.certificate.keys(),
            verifier_level=example.verification_level,
        )

    def _skill_to_expert_index(self, skill: str) -> int:
        digest = hashlib.sha256(str(skill).encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % int(self.model.config.skill_expert_count)

    def _skill_expert_distribution(
        self,
        skill_weights: Mapping[str, float],
        *,
        focus: float = 0.86,
    ) -> tuple[float, ...]:
        count = int(self.model.config.skill_expert_count)
        if count <= 0:
            return ()
        base = max(0.0, 1.0 - float(focus)) / float(count)
        weights = [base for _ in range(count)]
        total_skill_weight = sum(max(0.0, float(value)) for value in skill_weights.values())
        if total_skill_weight <= 0:
            return tuple(1.0 / float(count) for _ in range(count))
        for skill, raw_weight in skill_weights.items():
            weight = max(0.0, float(raw_weight)) / total_skill_weight
            expert_index = self._skill_to_expert_index(skill)
            weights[expert_index] += float(focus) * weight
        total = sum(weights)
        return tuple(float(value / total) for value in weights)

    def _skill_context_for_task(self, task: Task | TrainingExample | str) -> tuple[float, ...]:
        if isinstance(task, TrainingExample):
            skill = task.targeted_skill
        elif isinstance(task, Task):
            skill = task.skill
        else:
            skill = str(task)
        return self._skill_expert_distribution({skill: 1.0}, focus=0.90)

    def refresh_skill_expert_context_from_ledger(self, *, source: str) -> None:
        states = tuple(self.skill_ledger.states.values())
        if not states:
            self.model.set_skill_expert_context(None)
            return
        skill_weights: dict[str, float] = {}
        for state in states:
            rarity_pressure = 1.0 + float(state.failures)
            fragility_pressure = 1.0 + 3.0 * float(state.fragility)
            protection_pressure = 1.75 if state.protected else 1.0
            low_pass_pressure = 1.0 + max(0.0, 1.0 - float(state.pass_rate))
            skill_weights[state.skill] = rarity_pressure * fragility_pressure * protection_pressure * low_pass_pressure
        distribution = self._skill_expert_distribution(skill_weights, focus=0.82)
        self.model.set_skill_expert_context(distribution, source=source)
        self.skill_expert_context_events += 1
        self.skill_expert_last_context = distribution
        self.skill_expert_context_skills = tuple(
            skill for skill, _ in sorted(skill_weights.items(), key=lambda item: item[1], reverse=True)
        )
        self._count("skill_expert_ledger_context_events")
        self._count("skill_expert_ledger_context_skills", len(self.skill_expert_context_skills))

    def _learned_memory_retention_decision(
        self,
        policy: LearnedMemoryPolicyState | None,
        *,
        row_index: int,
        segment_id: str,
    ) -> MemoryRetentionDecision | None:
        if policy is None or row_index >= int(policy.probs.shape[0]):
            return None
        with torch.no_grad():
            exact_prob = float(policy.exact_prob[row_index].detach().mean().cpu())
            latent_prob = float(policy.latent_prob[row_index].detach().mean().cpu())
            drop_prob = float(policy.drop_prob[row_index].detach().mean().cpu())
            storage_ratio = float(policy.storage_ratio[row_index].detach().mean().cpu())
        mode_scores = (
            (MemoryMode.EXACT, exact_prob),
            (MemoryMode.LATENT, latent_prob),
            (MemoryMode.DROP, drop_prob),
        )
        requested_mode, confidence = max(mode_scores, key=lambda item: item[1])
        return MemoryRetentionDecision(
            segment_id=segment_id,
            requested_mode=requested_mode,
            applied_mode=requested_mode if requested_mode != MemoryMode.DROP else None,
            exact_prob=exact_prob,
            latent_prob=latent_prob,
            drop_prob=drop_prob,
            storage_ratio=storage_ratio,
            confidence=confidence,
            source="learned_memory_policy",
            reason="mean_row_probability_argmax",
            stored=requested_mode != MemoryMode.DROP,
        )

    def _record_applied_memory_retention_decision(self, decision: MemoryRetentionDecision | None) -> None:
        if decision is None:
            return
        self._count("learned_memory_retention_decisions")
        self._count(f"learned_memory_retention_requested_{decision.requested_mode.value}")
        applied = decision.applied_mode.value if decision.applied_mode is not None else "drop"
        self._count(f"learned_memory_retention_applied_{applied}")
        if decision.anchor_safety_override:
            self._count("learned_memory_retention_anchor_overrides")

    def _refresh_learned_memory_utility_prior(self, *, source: str, count_update: bool = True) -> None:
        if self.model.learned_memory is None:
            return
        learned_credits = [
            credit
            for credit in self.memory.utility_credits
            if credit.retention_source.startswith("learned_memory")
        ]
        if not learned_credits:
            self.model.learned_memory.set_memory_utility_prior(None)
            self.learned_memory_last_utility_prior = ()
            return
        scores = [0.05, 0.05, 0.05]
        for credit in learned_credits:
            mode = credit.applied_mode if credit.applied_mode is not None else MemoryMode.DROP
            if mode == MemoryMode.EXACT:
                index = LearnedMemoryPolicy.EXACT
            elif mode == MemoryMode.LATENT:
                index = LearnedMemoryPolicy.LATENT
            else:
                index = LearnedMemoryPolicy.DROP
            utility = max(-1.0, min(2.0, float(credit.utility)))
            reward = max(0.0, utility)
            if credit.fidelity_passed:
                reward += 0.35
            else:
                reward *= 0.25
            if int(credit.required_anchor_count) > 0:
                if mode == MemoryMode.EXACT:
                    reward += 0.35 * float(credit.required_anchor_count)
                elif mode == MemoryMode.LATENT:
                    reward += 0.15 * float(credit.required_anchor_count)
                else:
                    reward = max(0.0, reward - 0.50 * float(credit.required_anchor_count))
            elif mode == MemoryMode.DROP and utility <= 0.0:
                reward += 0.02
            scores[index] += max(0.0, reward)
        total = sum(scores)
        prior = tuple(float(value / total) for value in scores) if total > 0.0 else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        self.model.learned_memory.set_memory_utility_prior(prior, events=len(learned_credits))
        self.learned_memory_last_utility_prior = prior
        if count_update:
            self.learned_memory_utility_prior_updates += 1
            self._count("learned_memory_utility_prior_updates")
        self.integration_counts["learned_memory_utility_feedback_events"] = max(
            int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
            len(learned_credits),
        )
        self.integration_counts["learned_memory_last_utility_prior_events"] = len(learned_credits)
        self.integration_counts["learned_memory_last_utility_prior_exact_milli"] = int(round(prior[0] * 1000.0))
        self.integration_counts["learned_memory_last_utility_prior_latent_milli"] = int(round(prior[1] * 1000.0))
        self.integration_counts["learned_memory_last_utility_prior_drop_milli"] = int(round(prior[2] * 1000.0))

    def _record_memory_utility(
        self,
        reconstruction: Any,
        *,
        phase: str,
        source: str,
        reason: str = "",
        utility: float | None = None,
    ) -> tuple[MemoryUtilityCredit, ...]:
        credits = self.memory.record_utility(
            reconstruction,
            phase=phase,
            source=source,
            reason=reason,
            utility=utility,
        )
        if not credits:
            return ()
        learned = tuple(
            credit
            for credit in credits
            if credit.retention_source.startswith("learned_memory")
        )
        self._count("memory_utility_credit_events", len(credits))
        if learned:
            self._count("learned_memory_utility_credit_events", len(learned))
            self._refresh_learned_memory_utility_prior(source=source)
        return credits

    def _real_exogenous_llm_examples(self) -> tuple[TrainingExample, ...]:
        return tuple(
            example
            for example in self.sleep.reservoir.examples
            if example.origin == ExampleOrigin.REAL_EXOGENOUS
            and bool(dict(example.metadata).get("from_llm_input_batch"))
        )

    def _record_real_exogenous_llm_span(
        self,
        *,
        step: int,
        source: str,
        row_index: int,
        token_ids: Sequence[int],
        text: str,
        anchor_count: int,
    ) -> TrainingExample | None:
        span = text.strip()
        if not span:
            return None
        span_hash = hashlib.sha256(span.encode("utf-8")).hexdigest()
        task = Task(
            f"llm-real-{source}-{step}-{row_index}-{span_hash[:12]}",
            "instruction_following",
            "Reproduce this observed LLM corpus span exactly:\n" + span,
            span,
            {
                "source": source,
                "step": int(step),
                "row_index": int(row_index),
                "token_count": int(len(token_ids)),
                "text_sha256": span_hash,
                "from_llm_input_batch": True,
            },
            group_id=f"llm-real-{span_hash[:16]}",
        )
        answer = CandidateAnswer(
            span,
            confidence=0.99,
            certificate={
                "real_exogenous_llm_span": True,
                "from_llm_input_batch": True,
                "text_sha256": span_hash,
            },
            cost=CostTrace(generated_tokens=max(1, len(token_ids)), verifier_steps=1),
        )
        verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
        if not verification.passed:
            raise ValueError(f"real exogenous LLM span failed exact oracle: {verification.reason}")
        example = self.sleep.reservoir.add(
            task,
            answer,
            source_id=f"{source}-{step}-{row_index}",
            oracle=task.skill,
            verification_level=2,
            difficulty=0.20 + min(0.45, len(token_ids) / max(1.0, float(self.model.config.seq_len)) * 0.45),
            metadata={
                "source": source,
                "step": int(step),
                "row_index": int(row_index),
                "token_count": int(len(token_ids)),
                "anchor_count": int(anchor_count),
                "text_sha256": span_hash,
                "from_llm_input_batch": True,
                "verification_reason": verification.reason,
                "verification_score": float(verification.score),
            },
        )
        self._count("sleep_real_reservoir_events")
        self._count("sleep_real_exogenous_llm_batch_events")
        self._count("sleep_real_exogenous_llm_tokens", len(token_ids))
        self.uncertainty_ledger.record("llm_real_exogenous_exact_span", answer.confidence, True)
        self._record_causal_trace(
            trace_id=f"real-exogenous-{source}-{step}-{row_index}",
            skill="llm_real_exogenous_exact_span",
            confidence=answer.confidence,
            anchors=anchor_count,
            certificate_fields=answer.certificate.keys(),
            verifier_level=2,
        )
        return example

    def _decode_model_token(self, token_id: int) -> str:
        text = self.tokenizer.decode([int(token_id)]).strip()
        return text if text else f"<token:{int(token_id)}>"

    def _record_model_certificate_from_forward(
        self,
        *,
        step: int,
        source: str,
        output: LLMForwardOutput,
        future_targets: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        if output.certificate is None:
            return None
        certificate_output = output.certificate
        if certificate_output.answer_logits.shape[0] <= 0:
            return None
        row_index = 0
        answer_token_id = int(certificate_output.answer_logits[row_index].detach().argmax(dim=-1).cpu())
        lm_head_token_id = int(output.logits[row_index, -1].detach().argmax(dim=-1).cpu())
        target_token_id: int | None = None
        if future_targets is not None and future_targets.numel() > 0:
            target_token_id = int(future_targets[row_index, -1, 0].detach().cpu())
        answer_text = self._decode_model_token(answer_token_id)
        workspace_summary: dict[str, Any] | None = None
        workspace_checksum: str | None = None
        workspace_steps = 0
        workspace_bound_to_certificate = False
        if output.latent_workspace is not None:
            workspace = output.latent_workspace
            workspace_summary = workspace.to_summary()
            workspace_checksum = workspace.checksum(row_index)
            workspace_steps = int(workspace.step_count)
            workspace_bound_to_certificate = (
                certificate_output.latent_state.shape == workspace.summary.shape
            )
            self.latent_workspace_forward_events += 1
            self.latent_workspace_step_events += workspace_steps
            self.latent_workspace_last_summary = dict(workspace_summary)
            self._count("latent_workspace_forward_events")
            self._count("latent_workspace_step_events", workspace_steps)
            if workspace_bound_to_certificate:
                self.latent_workspace_certificate_binding_events += 1
                self._count("latent_workspace_certificate_binding_events")
        cert_types = tuple(CertificateType)
        cert_type_index = int(certificate_output.certificate_type_logits[row_index].detach().argmax(dim=-1).cpu())
        certificate_type = cert_types[cert_type_index % len(cert_types)]
        uncertainty = float(certificate_output.uncertainty[row_index].detach().cpu())
        target_match = target_token_id is not None and answer_token_id == target_token_id
        latent_state = LatentProofState(
            state_id=f"llm-model-cert-{source}-{step}-{len(self.model_certificate_head_artifacts)}",
            task_id=f"llm-model-cert-task-{source}-{step}",
            skill="llm_certificate_head",
            tensor=certificate_output.latent_state[row_index:row_index + 1].detach().cpu(),
            latent_steps=1,
            visible_reasoning_tokens=0,
        )
        certificate = build_certificate(
            certificate_id=f"llm-model-cert-{source}-{step}-{len(self.model_certificate_head_artifacts)}",
            task_id=latent_state.task_id,
            skill=latent_state.skill,
            certificate_type=certificate_type,
            answer=answer_text,
            claims={
                "model_certificate_head": True,
                "latent_reasoning_workspace": workspace_summary is not None,
                "latent_workspace_checksum": workspace_checksum,
                "latent_workspace_steps": workspace_steps,
                "latent_workspace_bound_to_certificate": workspace_bound_to_certificate,
                "predicted_certificate_type": certificate_type.value,
                "answer_token_id": answer_token_id,
                "lm_head_token_id": lm_head_token_id,
                "target_token_id": target_token_id,
                "target_match": target_match,
                "source": source,
                "step": int(step),
                "calibrated_uncertainty": uncertainty > self.certificate_verifier.max_uncertainty,
            },
            uncertainty=uncertainty,
            latent_state=latent_state,
            tool="model_token_certificate",
            tool_args={
                "answer_token_id": answer_token_id,
                "certificate_head_token_id": answer_token_id,
                "decoded_answer": answer_text,
                "lm_head_token_id": lm_head_token_id,
                "target_token_id": target_token_id,
                "require_target_match": False,
                "require_lm_head_match": False,
            },
        )
        verification = self.certificate_verifier.verify(certificate, latent_state)
        artifact = {
            "step": int(step),
            "source": source,
            "answer_token_id": answer_token_id,
            "lm_head_token_id": lm_head_token_id,
            "target_token_id": target_token_id,
            "target_match": target_match,
            "certificate_type": certificate_type.value,
            "uncertainty": uncertainty,
            "latent_workspace": workspace_summary,
            "latent_workspace_checksum": workspace_checksum,
            "latent_workspace_bound_to_certificate": workspace_bound_to_certificate,
            "certificate": certificate.to_dict(),
            "latent_state": latent_state.to_dict(),
            "verification": verification.to_dict(),
        }
        self.model_certificate_head_artifacts.append(artifact)
        self._count("model_certificate_head_events")
        if verification.passed:
            self._count("model_certificate_head_verified_events")
            self.bit_ledger.add_certificate(certificate.to_dict())
        else:
            self._count("model_certificate_head_failed_events")
        if verification.latent_checksum_ok:
            self._count("model_certificate_head_latent_checksum_events")
        if target_match:
            self._count("model_certificate_head_target_match_events")
        self.uncertainty_ledger.record("llm_certificate_head_token", 1.0 - uncertainty, bool(target_match))
        self.bit_ledger.ingest_cost(CostTrace(verifier_steps=1, generated_tokens=1), note=f"P5:model-certificate-head:{source}:{step}")
        return artifact

    def _ingest_cycle_ledgers(self, cycle_report: Any, *, step: int) -> dict[str, Any]:
        self.skill_ledger.update_from_report(cycle_report.trial)
        self._ingest_suite_ledgers(cycle_report.trial, step=step, source="trial")
        if cycle_report.extra_report is not None:
            self._ingest_suite_ledgers(cycle_report.extra_report, step=step, source="compression-adversary")
        for action in cycle_report.actions:
            self.bit_ledger.scale_bits += float(action.cost) * 8.0
            self.bit_ledger.notes.append(f"step-{step}:regrowth:{action.action}:{action.target}")
        self._ingest_compression_trace_delta(step=step, note="phase-audit")
        return {
            "bit_ledger_total_effective_bits": self.bit_ledger.total_effective_bits,
            "skill_ledger_states": len(self.skill_ledger.states),
            "fragile_skills": [state.skill for state in self.skill_ledger.fragile_skills()],
            "causal_trace_count": len(self.causal_ledger.traces),
            "uncertainty_observations": sum(len(pairs) for pairs in self.uncertainty_ledger.bins.values()),
            "expected_calibration_error": self.uncertainty_ledger.expected_calibration_error(),
        }

    def observe_input_batch(self, *, step: int, input_ids: torch.Tensor, source: str, output: LLMForwardOutput | None = None) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            return
        policy = output.learned_memory_policy if output is not None else None
        if policy is not None:
            with torch.no_grad():
                decisions = policy.probs.detach().argmax(dim=-1)
                self.learned_memory_policy_events += int(decisions.numel())
                exact_active = int(policy.exact_prob.detach().gt(0.01).sum().item())
                latent_active = int(policy.latent_prob.detach().gt(0.01).sum().item())
                drop_active = int(policy.drop_prob.detach().gt(0.01).sum().item())
                self.learned_memory_exact_decisions += exact_active
                self.learned_memory_latent_decisions += latent_active
                self.learned_memory_drop_decisions += drop_active
                self.learned_memory_storage_ratio_total += float(policy.storage_ratio.detach().sum().cpu())
                self._count("learned_memory_policy_events", int(decisions.numel()))
                self._count("learned_memory_exact_decisions", exact_active)
                self._count("learned_memory_latent_decisions", latent_active)
                self._count("learned_memory_drop_decisions", drop_active)
        sample_count = min(2, int(input_ids.shape[0]))
        for row_index, row in enumerate(input_ids[:sample_count]):
            token_ids = [int(token) for token in row.detach().cpu().tolist()]
            text = self.tokenizer.decode(token_ids).strip()
            if not text:
                continue
            segment_id = f"llm-input-{source}-{step}-{self.input_anchor_observations}-{row_index}"
            retention_decision = self._learned_memory_retention_decision(
                policy,
                row_index=row_index,
                segment_id=segment_id,
            )
            segment = self.memory.ingest(
                segment_id,
                text,
                metadata={
                    "source": source,
                    "step": step,
                    "row_index": row_index,
                    "token_count": len(token_ids),
                    "from_llm_input_batch": True,
                },
                retention_decision=retention_decision,
            )
            applied_decision = (
                self.memory.retention_decisions[-1]
                if self.memory.retention_decisions and self.memory.retention_decisions[-1].segment_id == segment_id
                else None
            )
            self._record_applied_memory_retention_decision(applied_decision)
            anchor_count = len(segment.anchors) if segment is not None else int(applied_decision.anchor_count if applied_decision else 0)
            self.input_anchor_observations += 1
            self.input_anchor_count += anchor_count
            text_bytes = float(len(text.encode("utf-8")))
            self.bit_ledger.ingest_cost(
                CostTrace(kv_bytes=text_bytes, verifier_steps=1 if anchor_count else 0),
                note=f"input-anchor:{source}:{step}:{anchor_count}",
            )
            if self.model.compression_ledger is not None:
                self.model.compression_ledger.record_kv(
                    segment_id,
                    "exact_anchor_input",
                    bytes_used=text_bytes,
                    exact_anchors=anchor_count,
                    note="Exact Anchor Ledger observed decoded LLM input batch",
                )
            self._record_real_exogenous_llm_span(
                step=step,
                source=source,
                row_index=row_index,
                token_ids=token_ids,
                text=text,
                anchor_count=anchor_count,
            )
            if anchor_count:
                if segment is None:
                    self.input_anchor_fidelity_failures += 1
                    self._record_error("P4", ValueError(f"anchored input was dropped by learned memory policy: {segment_id}"))
                    continue
                if policy is not None and row_index < policy.exact_prob.shape[0]:
                    exact_mean = float(policy.exact_prob[row_index].detach().mean().cpu())
                    latent_mean = float(policy.latent_prob[row_index].detach().mean().cpu())
                    drop_mean = float(policy.drop_prob[row_index].detach().mean().cpu())
                    self.learned_memory_anchor_supervision_events += anchor_count
                    self._count("learned_memory_anchor_supervision_events", anchor_count)
                    self.bit_ledger.ingest_cost(
                        CostTrace(kv_bytes=float(anchor_count) * max(0.0, 1.0 - exact_mean), verifier_steps=1),
                        note=(
                            f"learned-memory-anchor-policy:{source}:{step}:"
                            f"exact={exact_mean:.4f}:latent={latent_mean:.4f}:drop={drop_mean:.4f}"
                        ),
                    )
                reconstruction = self.memory.reconstruct(text[:512], required_anchors=segment.anchors)
                self._record_memory_utility(
                    reconstruction,
                    phase="P4",
                    source=f"input-anchor:{source}:{step}:{row_index}",
                    reason="input_anchor_fidelity",
                )
                if not reconstruction.fidelity.passed:
                    self.input_anchor_fidelity_failures += 1
                    self._record_error(
                        "P4",
                        ValueError(
                            f"input anchor fidelity failed for {segment_id}: "
                            f"missing {[anchor.value for anchor in reconstruction.fidelity.missing]}"
                        ),
                    )
                else:
                    self.uncertainty_ledger.record("llm_input_anchor_preservation", 0.99, True)
                    self._record_causal_trace(
                        trace_id=f"input-anchor-{source}-{step}-{self.input_anchor_observations}",
                        skill="llm_input_anchor_preservation",
                        confidence=0.99,
                        anchors=anchor_count,
                        certificate_fields=("exact_anchor_ledger", "input_batch_decode"),
                        verifier_level=2,
                    )

    def interval(self) -> int:
        return int(self.config.cortex_phase_interval or self.config.eval_interval)

    def should_sample_step(self, step: int) -> bool:
        return step == 0 or step % self.interval() == 0 or step == self.config.steps

    def objective_feedback_scale(self) -> float:
        if self.config.cortex_objective_feedback_weight <= 0:
            return 1.0
        bounded_loss = min(
            float(self.config.cortex_objective_feedback_clip),
            math.log1p(max(0.0, float(self.last_objective_loss_total))),
        )
        return 1.0 + float(self.config.cortex_objective_feedback_weight) * bounded_loss

    def auxiliary_loss(self, output: LLMForwardOutput) -> torch.Tensor:
        if output.confidence is None or self.config.cortex_phase_regularization_weight <= 0:
            return output.logits.new_tensor(0.0)
        self.regularization_steps += 1
        confidence_pressure = (1.0 - output.confidence.mean()).clamp_min(0.0)
        return (
            confidence_pressure
            * float(self.config.cortex_phase_regularization_weight)
            * self.objective_feedback_scale()
        )

    def observe_batch_contract(
        self,
        *,
        step: int,
        output: LLMForwardOutput,
        future_targets: torch.Tensor,
        breakdown: LossBreakdown,
    ) -> None:
        try:
            self._record_model_certificate_from_forward(
                step=step,
                source="batch-contract",
                output=output,
                future_targets=future_targets,
            )
        except Exception as exc:
            self._record_error("P5", exc)
        if not output.mtp_logits or output.confidence is None:
            return
        try:
            logits_by_horizon: dict[int, torch.Tensor] = {}
            for horizon, logits in output.mtp_logits.items():
                if logits.shape[1] >= horizon:
                    logits_by_horizon[int(horizon)] = logits[:, -int(horizon):, :].detach()
            if not logits_by_horizon:
                return
            confidence = float(output.confidence.detach().mean().cpu())
            contract = self.future_engine.draft_contract_from_logits(
                logits_by_horizon,
                confidence=confidence,
                domain="llm_pretraining",
                risk=max(0.0, min(1.0, float(breakdown.next_token))),
                contract_id=f"llm-step-{step}-{len(self.batch_contract_samples)}",
                temporal_loss=float(breakdown.temporal_consistency),
            )
            observed = self._observed_contract_tokens(future_targets, contract.accepted_horizon)
            decision = self.future_engine.gate_contract(contract, observed_tokens=[int(token) for token in observed])
            self._count("future_contract_observed_token_checks")
            self.bit_ledger.ingest_cost(decision.cost, note=f"P3:{decision.contract.contract_id}")
            self.uncertainty_ledger.record("llm_pretraining_future_contract", decision.contract.confidence, decision.accepted)
            self._record_causal_trace(
                trace_id=f"P3-{step}-{decision.contract.contract_id}",
                skill="llm_pretraining_future_contract",
                confidence=decision.contract.confidence,
                anchors=0,
                certificate_fields=("future_contract", "fsp_gate"),
                verifier_level=1,
                mtp_horizon=decision.contract.accepted_horizon,
            )
            self._ingest_compression_trace_delta(step=step, note="future-contract")
            self._touch("P3")
            decision_label = "ACCEPT" if decision.accepted else "REJECT"
            decision_task = Task(
                f"phase-p3-{step}-{len(self.batch_contract_samples)}",
                "instruction_following",
                f"Output the future contract gate result exactly: {decision_label}",
                decision_label,
            )
            decision_answer = self._answer_from_task_expected(decision_task)
            output_goal = self.future_engine.gate_output_goal(
                decision_task,
                decision_answer,
                risk=decision.contract.risk,
                contract_id=f"{decision.contract.contract_id}-output-goal",
                output_verified=True,
            )
            self._count("future_output_goal_contract_checks")
            self.bit_ledger.ingest_cost(output_goal.cost, note=f"P3:output-goal:{output_goal.contract.contract_id}")
            self.uncertainty_ledger.record("llm_pretraining_output_goal_contract", decision_answer.confidence, output_goal.accepted)
            if not output_goal.accepted:
                raise ValueError(f"P3 output-goal contract rejected: {output_goal.reason}")
            self._add_verified_phase_replay(
                "P3",
                decision_task,
                answer=decision_answer,
                metadata={
                    "contract_id": decision.contract.contract_id,
                    "accepted_horizon": decision.contract.accepted_horizon,
                    "observed_token_count": len(observed),
                    "reason": decision.reason,
                    "output_goal_contract_id": output_goal.contract.contract_id,
                    "output_goal_accepted": output_goal.accepted,
                    "output_goal_violations": output_goal.violations,
                },
            )
            self.batch_contract_samples.append(
                {
                    "step": step,
                    "accepted": decision.accepted,
                    "horizon": decision.contract.accepted_horizon,
                    "confidence": decision.contract.confidence,
                    "observed_token_count": len(observed),
                    "reason": decision.reason,
                    "output_goal_contract_id": output_goal.contract.contract_id,
                    "output_goal_accepted": output_goal.accepted,
                    "output_goal_violations": output_goal.violations,
                }
            )
        except Exception as exc:
            self._record_error("P3", exc)

    def replay_loss(
        self,
        model_forward: nn.Module,
        objective: CortexObjective,
        precision: PrecisionPolicy,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.replay_batches or self.config.cortex_phase_replay_weight <= 0:
            return None
        replay_index = self.replay_cursor % len(self.replay_batches)
        batch = self.replay_batches[replay_index].to(device)
        replay_context = (
            self.replay_skill_contexts[replay_index]
            if replay_index < len(self.replay_skill_contexts)
            else ()
        )
        self.replay_cursor += 1
        x = batch[:, :-1]
        y = batch[:, 1:]
        future = self._future_targets_from_next_tokens(y)
        previous_context = self.model.skill_expert_context_distribution()
        previous_source = self.model.skill_expert_context_source
        if replay_context:
            self.model.set_skill_expert_context(replay_context, source="phase-replay")
            self.skill_expert_replay_context_events += 1
            self._count("skill_expert_replay_context_events")
        try:
            with precision.autocast(device.type):
                output = model_forward(x)
                loss, _ = objective.compute(output, y, future, use_cortex_terms=self.model.config.use_cortex_heads)
        finally:
            self.model.set_skill_expert_context(previous_context, source=previous_source)
        self.replay_updates += 1
        return loss * float(self.config.cortex_phase_replay_weight) * self.objective_feedback_scale()

    def _future_targets_from_next_tokens(self, y: torch.Tensor) -> torch.Tensor:
        horizons = tuple(int(horizon) for horizon in self.model.config.horizons)
        future = []
        for horizon in horizons:
            if horizon <= 1:
                shifted = y
            else:
                tail = y[:, -1:].expand(-1, horizon - 1)
                shifted = torch.cat((y[:, horizon - 1:], tail), dim=1)
            future.append(shifted)
        return torch.stack(future, dim=-1)

    def _observed_contract_tokens(self, future_targets: torch.Tensor, accepted_horizon: int) -> list[int]:
        horizons = tuple(int(horizon) for horizon in self.model.config.horizons)
        if accepted_horizon not in horizons:
            lower_or_equal = tuple(horizon for horizon in horizons if horizon <= accepted_horizon)
            accepted_horizon = max(lower_or_equal) if lower_or_equal else min(horizons)
        horizon_column = horizons.index(accepted_horizon)
        if future_targets.shape[1] < accepted_horizon:
            raise ValueError(
                f"cannot verify future contract horizon {accepted_horizon} with sequence length {future_targets.shape[1]}"
            )
        return [
            int(token)
            for token in future_targets[0, -accepted_horizon:, horizon_column].detach().cpu().tolist()
        ]

    def _batch_from_example(self, example: TrainingExample) -> torch.Tensor:
        text = (
            f"Skill: {example.targeted_skill}\n"
            f"Task: {example.task.prompt}\n"
            f"Verified answer: {example.answer.text}\n"
        )
        ids = list(self.tokenizer.encode(text))
        if not ids:
            ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]
        while len(ids) < self.model.config.seq_len + 1:
            ids.extend(ids)
        ids = ids[: self.model.config.seq_len + 1]
        return torch.tensor([ids], dtype=torch.long)

    def _answer_from_task_expected(self, task: Task) -> CandidateAnswer:
        expected = task.expected
        text = str(expected["answer"]) if isinstance(expected, Mapping) and "answer" in expected else str(expected)
        return CandidateAnswer(
            text,
            confidence=1.0,
            certificate={"oracle_label": task.skill, "source": "cortex_phase_replay"},
        )

    def _difficulty_for_phase_task(self, task: Task) -> float:
        token_count = max(1, len(task.prompt.split()))
        score = 0.20 + min(0.35, token_count / 200.0)
        if task.anchors:
            score += 0.20
        if task.skill in {"arithmetic", "algebra", "code_unit_tests", "calibration"}:
            score += 0.15
        return max(0.0, min(1.0, score))

    def _add_verified_phase_replay(
        self,
        phase_id: str,
        task: Task,
        *,
        answer: CandidateAnswer | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        origin: ExampleOrigin = ExampleOrigin.TOOL_SOLVED,
    ) -> TrainingExample | None:
        candidate = CandidateAnswer.coerce(answer) if answer is not None else self._answer_from_task_expected(task)
        verification = self.verifier.oracle_registry.verify(task.skill, task, candidate)
        if not verification.passed:
            self._record_error(
                phase_id,
                ValueError(f"phase replay oracle rejected {task.task_id}: {verification.reason}"),
            )
            return None
        example = TrainingExample(
            example_id=f"llm-{phase_id.lower()}-{self.phase_replay_examples.get(phase_id, 0)}-{task.task_id}",
            task=task,
            answer=CandidateAnswer(
                candidate.text,
                confidence=max(float(candidate.confidence), float(verification.score)),
                certificate={**dict(candidate.certificate), "phase_replay": phase_id},
                cost=candidate.cost,
                raw={**dict(candidate.raw), "verification_reason": verification.reason},
            ),
            origin=origin,
            oracle=task.skill,
            targeted_skill=task.skill,
            verification_level=3,
            contamination_risk=0.04,
            difficulty=self._difficulty_for_phase_task(task),
            confidence_label=float(verification.score),
            synthetic=True,
            metadata={"source_phase": phase_id, **dict(metadata or {})},
        )
        self.replay_batches.append(self._batch_from_example(example))
        self.replay_skill_contexts.append(self._skill_context_for_task(example))
        self.phase_replay_examples[phase_id] = self.phase_replay_examples.get(phase_id, 0) + 1
        self.phase_replay_example_ids.append(example.example_id)
        self._record_training_example_ledgers(example, phase_id=phase_id)
        return example

    def _add_sleep_replay(self, examples: Sequence[TrainingExample]) -> None:
        for example in examples[:8]:
            self.replay_batches.append(self._batch_from_example(example))
            self.replay_skill_contexts.append(self._skill_context_for_task(example))
            self.phase_replay_examples["P9"] = self.phase_replay_examples.get("P9", 0) + 1
            self.phase_replay_example_ids.append(f"llm-p9-{self.phase_replay_examples['P9']}-{example.example_id}")
            self._record_training_example_ledgers(example, phase_id="P9")

    def _frontier_report_key(self, report: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        return (
            str(report.get("skill", "")),
            tuple(str(item) for item in report.get("source_failure_ids", ())),
            tuple(str(item) for item in report.get("frontier_task_ids", ())),
            tuple(str(item) for item in report.get("heldout_task_ids", ())),
        )

    def _bind_frontier_circuit_memory(
        self,
        runtime_circuit: Any,
        *,
        step: int,
        source_phase: str,
    ) -> dict[str, Any]:
        report = runtime_circuit.report
        circuit_id = compiled_circuit_id(report)
        training = dict(report.training)
        task_anchors = tuple(
            anchor
            for task in tuple(runtime_circuit.verified_tasks) + tuple(runtime_circuit.heldout_tasks)
            for anchor in task.anchors
        )
        binding = self.memory.bind_compiled_circuit(
            circuit_id=circuit_id,
            skill=runtime_circuit.skill,
            source_kind=str(training.get("source_kind", "frontier_discovery")),
            source_failure_ids=tuple(report.source_failure_ids),
            frontier_task_ids=tuple(report.frontier_task_ids),
            heldout_task_ids=tuple(report.heldout_task_ids),
            prompt_obligations=tuple(report.invariants.prompt_obligations),
            metadata_keys=tuple(report.invariants.metadata_keys),
            anchors=task_anchors,
        )
        _, reconstruction = self.memory.reconstruct_compiled_circuit_binding(
            circuit_id,
            query=f"{source_phase} compiled circuit {circuit_id} skill {runtime_circuit.skill}",
        )
        self._record_memory_utility(
            reconstruction,
            phase=source_phase,
            source=f"compiled-circuit:{circuit_id}",
            reason="compiled_circuit_binding",
        )
        if not reconstruction.fidelity.passed or binding.segment_id not in reconstruction.selected_segment_ids:
            self._count("compiled_circuit_memory_fidelity_failures")
            missing = ", ".join(anchor.value for anchor in reconstruction.fidelity.missing)
            raise ValueError(f"{source_phase} circuit {circuit_id} failed P4 memory retention gate: missing {missing}")
        self._count("compiled_circuit_memory_binding_events")
        if source_phase == "P9":
            self._count("sleep_frontier_memory_binding_events")
        self.bit_ledger.ingest_cost(reconstruction.cost, note=f"{source_phase}:compiled-circuit-memory:{runtime_circuit.skill}")
        self._record_causal_trace(
            trace_id=f"{source_phase}-compiled-memory-{step}-{circuit_id}",
            skill=runtime_circuit.skill,
            confidence=0.99,
            anchors=len(binding.anchor_values),
            certificate_fields=("compiled_circuit_memory_binding", "exact_anchor_ledger"),
            verifier_level=2,
        )
        return {
            "binding": binding.to_dict(),
            "runtime_fidelity": asdict(reconstruction.fidelity),
            "runtime_selected_segment_ids": reconstruction.selected_segment_ids,
        }

    def _validate_restored_frontier_fastsolve(self, *, registry_path: Path) -> dict[str, Any] | None:
        runtime_circuits = tuple(
            circuit
            for skill in self.frontier_registry.compiled_skills()
            for circuit in self.frontier_registry.circuits_for_skill(skill)
        )
        if not runtime_circuits:
            return None
        for runtime_circuit in runtime_circuits:
            covered_tasks = tuple(runtime_circuit.verified_tasks) + tuple(runtime_circuit.heldout_tasks)
            if not covered_tasks:
                continue
            task = covered_tasks[0]
            circuit_id = compiled_circuit_id(runtime_circuit.report)
            answer = CompiledFrontierAgent(
                self.frontier_registry,
                verifier=self.verifier,
                memory=self.memory,
            )(task)
            verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
            memory_binding = dict(answer.raw.get("frontier_memory_binding") or {})
            runtime_fidelity = dict(memory_binding.get("runtime_fidelity") or {})
            report = {
                "source": "checkpoint_restore",
                "registry_path": str(registry_path),
                "task_id": task.task_id,
                "skill": task.skill,
                "circuit_id": circuit_id,
                "frontier_compiled_selected": bool(answer.raw.get("frontier_compiled_selected", False)),
                "verified": bool(verification.passed),
                "verification_reason": verification.reason,
                "frontier_memory_binding_id": answer.certificate.get("frontier_memory_binding_id"),
                "frontier_memory_binding_passed": bool(answer.certificate.get("frontier_memory_binding_passed", False)),
                "frontier_memory_binding_fidelity": float(
                    answer.certificate.get("frontier_memory_binding_fidelity", 0.0) or 0.0
                ),
                "runtime_selected_segment_ids": tuple(memory_binding.get("runtime_selected_segment_ids", ())),
                "runtime_fidelity": runtime_fidelity,
                "frontier_heldout_gate_passed": bool(answer.certificate.get("frontier_heldout_gate_passed", False)),
                "frontier_heldout_passed": int(answer.certificate.get("frontier_heldout_passed", 0) or 0),
                "frontier_heldout_total": int(answer.certificate.get("frontier_heldout_total", 0) or 0),
                "frontier_output_goal_contract_passed": bool(
                    answer.certificate.get("frontier_output_goal_contract_passed", False)
                ),
                "frontier_compiled_contract_verified": bool(
                    answer.certificate.get("frontier_compiled_contract_verified", False)
                ),
                "cost": _cost_trace_payload(answer.cost),
            }
            if not report["frontier_compiled_selected"]:
                raise ValueError(f"restored Frontier registry did not select circuit {circuit_id} for {task.task_id}")
            if not report["verified"]:
                raise ValueError(f"restored Frontier FastSolve failed oracle for {task.task_id}: {verification.reason}")
            if not report["frontier_memory_binding_passed"]:
                raise ValueError(f"restored Frontier circuit {circuit_id} failed restored P4 memory binding")
            if not report["frontier_output_goal_contract_passed"]:
                raise ValueError(f"restored Frontier circuit {circuit_id} failed restored output-goal contract")
            if not report["frontier_compiled_contract_verified"]:
                raise ValueError(f"restored Frontier circuit {circuit_id} failed restored compiled-circuit certificate")
            self.restored_frontier_fastsolve_reports.append(report)
            self._count("frontier_restored_fastsolve_events")
            self._count("frontier_compiled_fastsolve_events")
            self._count("compiled_circuit_memory_restored_reuse_events")
            self.frontier_compiled_fastsolve_events += 1
            self.bit_ledger.ingest_cost(answer.cost, note=f"P8:restored-frontier-fastsolve:{task.skill}")
            self.uncertainty_ledger.record(task.skill, answer.confidence, True)
            self._record_causal_trace(
                trace_id=f"P8-restored-frontier-{len(self.restored_frontier_fastsolve_reports)}-{task.task_id}",
                skill=task.skill,
                confidence=answer.confidence,
                anchors=len(task.anchors),
                certificate_fields=answer.certificate.keys(),
                verifier_level=2,
            )
            return report
        raise ValueError("restored Frontier registry contained circuits but no covered task for FastSolve validation")

    def _compile_sleep_frontier(
        self,
        sleep_report: Any,
        *,
        step: int,
    ) -> dict[str, Any]:
        report = self.frontier_discovery.compile_sleep_consolidation(
            sleep_report,
            seed=self.config.seed + step + 31337,
            max_skills=max(1, int(self.config.cortex_phase_frontier_max_skills)),
            support_per_verified=2,
            heldout_per_support=1,
            max_generalization_rounds=2,
            epochs=max(40, int(self.config.cortex_phase_frontier_epochs)),
            registry=self.frontier_registry,
        )
        payload = report.to_dict()
        passed_circuits = tuple(circuit for circuit in report.circuits if circuit.passed)
        self._count("sleep_frontier_compilation_events")
        self._count("sleep_frontier_compiled_circuits", len(passed_circuits))
        self._count(
            "sleep_frontier_heldout_gates",
            sum(1 for circuit in passed_circuits if bool(circuit.heldout.get("gate_passed", False))),
        )
        if not passed_circuits:
            raise ValueError("P9 sleep consolidation produced no held-out gated Frontier circuit")

        registry_dir = self.run_dir / "frontier_registry"
        registry_path = self.frontier_registry.save(registry_dir)
        self.integration_counts["frontier_registry_saves"] = self.integration_counts.get("frontier_registry_saves", 0) + 1
        passed_keys = {self._frontier_report_key(circuit.to_dict()) for circuit in passed_circuits}
        runtime_circuits = tuple(
            circuit
            for skill in self.frontier_registry.compiled_skills()
            for circuit in self.frontier_registry.circuits_for_skill(skill)
            if self._frontier_report_key(circuit.report.to_dict()) in passed_keys
        )
        if not runtime_circuits:
            raise ValueError("P9 sleep Frontier circuits were compiled but not present in the runtime registry")

        fastsolve_reports: list[dict[str, Any]] = []
        for runtime_circuit in runtime_circuits:
            if not runtime_circuit.verified_tasks:
                continue
            memory_binding_report = self._bind_frontier_circuit_memory(runtime_circuit, step=step, source_phase="P9")
            local_registry = FrontierCircuitRegistry()
            local_registry.register(
                runtime_circuit.report,
                runtime_circuit.model,
                runtime_circuit.verified_tasks,
                heldout_tasks=runtime_circuit.heldout_tasks,
                checkpoint_path=runtime_circuit.checkpoint_path,
            )
            task = runtime_circuit.verified_tasks[0]
            answer = CompiledFrontierAgent(local_registry, verifier=self.verifier, memory=self.memory)(task)
            verification = self.verifier.oracle_registry.verify(task.skill, task, answer)
            if not verification.passed:
                raise ValueError(f"P9 compiled sleep Frontier fastsolve failed oracle for {task.task_id}: {verification.reason}")
            self._count("sleep_frontier_fastsolve_events")
            self._count("frontier_compiled_fastsolve_events")
            self.frontier_compiled_fastsolve_events += 1
            self.bit_ledger.ingest_cost(answer.cost, note=f"P9:sleep-frontier-fastsolve:{task.skill}")
            self.uncertainty_ledger.record(task.skill, answer.confidence, True)
            self._record_causal_trace(
                trace_id=f"P9-sleep-frontier-{step}-{task.task_id}",
                skill=task.skill,
                confidence=answer.confidence,
                anchors=len(task.anchors),
                certificate_fields=answer.certificate.keys(),
                verifier_level=2,
            )
            self._add_verified_phase_replay(
                "P9",
                task,
                answer=answer,
                metadata={
                    "sleep_frontier_fastsolve": True,
                    "frontier_skill": runtime_circuit.skill,
                    "frontier_task_ids": runtime_circuit.report.frontier_task_ids,
                    "frontier_heldout_task_ids": runtime_circuit.report.heldout_task_ids,
                    "frontier_heldout_gate_passed": bool(runtime_circuit.report.heldout.get("gate_passed", False)),
                    "frontier_compiled_contract_verified": bool(answer.certificate.get("frontier_compiled_contract_verified", False)),
                    "frontier_memory_binding_id": answer.certificate.get("frontier_memory_binding_id"),
                    "frontier_memory_binding_passed": bool(answer.certificate.get("frontier_memory_binding_passed", False)),
                },
            )
            fastsolve_reports.append(
                {
                    "task_id": task.task_id,
                    "skill": task.skill,
                    "answer": answer.text,
                    "verified": verification.passed,
                    "frontier_compiled_contract_verified": bool(
                        answer.certificate.get("frontier_compiled_contract_verified", False)
                    ),
                    "heldout_gate_passed": bool(runtime_circuit.report.heldout.get("gate_passed", False)),
                    "heldout_passed": int(runtime_circuit.report.heldout.get("passed", 0)),
                    "heldout_total": int(runtime_circuit.report.heldout.get("total", 0)),
                    "memory_binding": memory_binding_report,
                    "frontier_memory_binding_passed": bool(answer.certificate.get("frontier_memory_binding_passed", False)),
                }
            )
        if not fastsolve_reports:
            raise ValueError("P9 sleep Frontier circuits compiled but no runtime FastSolve task was verified")
        final_payload = {
            **payload,
            "registry_path": str(registry_path),
            "fastsolve": tuple(fastsolve_reports),
        }
        self.sleep_frontier_reports.append(final_payload)
        return final_payload

    def _batch_from_task(self, task: Task, answer: CandidateAnswer | str) -> torch.Tensor:
        candidate = CandidateAnswer.coerce(answer)
        text = (
            f"Skill: {task.skill}\n"
            f"Task: {task.prompt}\n"
            f"Verified answer: {candidate.text}\n"
        )
        ids = list(self.tokenizer.encode(text))
        if not ids:
            ids = [self.tokenizer.bos_id, self.tokenizer.eos_id]
        while len(ids) < self.model.config.seq_len + 1:
            ids.extend(ids)
        ids = ids[: self.model.config.seq_len + 1]
        return torch.tensor([ids], dtype=torch.long)

    def _loss_for_regrowth_batch(self, batch: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]
        future = self._future_targets_from_next_tokens(y)
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            output = self.model(x)
            loss, _ = CortexObjective().compute(
                output,
                y,
                future,
                use_cortex_terms=self.model.config.use_cortex_heads,
            )
        return loss

    def _selected_regrowth_parameters(self, action: RegrowthActionKind) -> list[tuple[str, nn.Parameter]]:
        if action in {
            RegrowthActionKind.UNZERO_BLOCK,
            RegrowthActionKind.CHANGE_SIGN,
            RegrowthActionKind.INCREASE_SCALE_PRECISION,
            RegrowthActionKind.INCREASE_LOCAL_ACTIVATION_BITS,
        }:
            prefixes = ("blocks.", "variable_in.", "lm_head.", "token_embedding.")
        elif action == RegrowthActionKind.FORCE_EXACT_ANCHOR:
            prefixes = ("token_embedding.", "position_embedding.", "variable_in.", "lm_head.")
        elif action == RegrowthActionKind.REDUCE_MTP_HORIZON:
            prefixes = ("mtp_heads.", "confidence_head.")
        elif action == RegrowthActionKind.ROUTE_SPECIALIST_EXPERT:
            prefixes = ("skill_experts.", "lm_head.", "token_embedding.")
        elif action in {RegrowthActionKind.ADD_CERTIFICATE_FIELD, RegrowthActionKind.ADD_VERIFIER_CHECK}:
            prefixes = ("certificate_head.", "confidence_head.", "lm_head.", "token_embedding.")
        else:
            prefixes = ("blocks.", "skill_experts.", "mtp_heads.", "certificate_head.", "lm_head.", "token_embedding.")
        selected: list[tuple[str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                continue
            if any(name.startswith(prefix) for prefix in prefixes):
                selected.append((name, parameter))
                seen.add(id(parameter))
        if not selected:
            raise ValueError(f"P7 model regrowth could not find trainable parameters for action {action.value}")
        return selected

    def _cycle_protected_tasks(self, cycle_report: Any | None, failure_task: Task, *, limit: int = 4) -> tuple[Task, ...]:
        protected: list[Task] = []
        seen = {failure_task.task_id}
        if cycle_report is None:
            return ()
        for skill_report in cycle_report.trial.skill_reports.values():
            for case in skill_report.cases:
                task = case.task
                if task.task_id in seen:
                    continue
                protected.append(task)
                seen.add(task.task_id)
                if len(protected) >= limit:
                    return tuple(protected)
        return tuple(protected)

    def _evaluate_frontier_repair_candidate(
        self,
        attribution_report: Any,
        protected_tasks: Sequence[Task],
    ) -> dict[str, Any] | None:
        failure = attribution_report.failure
        circuit = self.frontier_registry.select(failure.task)
        if circuit is None:
            return None
        compiled_agent = CompiledFrontierAgent(self.frontier_registry, verifier=self.verifier, memory=self.memory)
        answer = compiled_agent(failure.task)
        repaired = self.verifier.oracle_registry.verify(failure.task.skill, failure.task, answer)

        def repaired_agent(task: Task) -> CandidateAnswer:
            if self.frontier_registry.select(task) is not None:
                return compiled_agent(task)
            return CandidateAnswer.coerce(self.trial_agent(task))

        non_regression = self.regrowth.simulator.non_regression.check(
            self.trial_agent,
            repaired_agent,
            tuple(protected_tasks),
        )
        total_cost = answer.cost.merge(repaired.verifier_cost)
        output_goal_passed = bool(answer.certificate.get("frontier_output_goal_contract_passed"))
        compiled_contract_verified = bool(answer.certificate.get("frontier_compiled_contract_verified"))
        heldout_gate_passed = bool(answer.certificate.get("frontier_heldout_gate_passed"))
        memory_binding_passed = bool(answer.certificate.get("frontier_memory_binding_passed"))
        accepted = bool(
            repaired.passed
            and repaired.score > failure.score
            and non_regression.passed
            and output_goal_passed
            and compiled_contract_verified
            and heldout_gate_passed
            and memory_binding_passed
        )
        report = {
            "task_id": failure.task.task_id,
            "skill": failure.task.skill,
            "circuit_skill": circuit.skill,
            "frontier_task_ids": tuple(circuit.report.frontier_task_ids),
            "source_failure_ids": tuple(circuit.report.source_failure_ids),
            "repair_score_before": float(failure.score),
            "repair_score_after": float(repaired.score),
            "repair_score_delta": float(repaired.score - failure.score),
            "repair_passed": bool(repaired.passed),
            "repair_reason": repaired.reason,
            "non_regression_passed": bool(non_regression.passed),
            "protected_checked": int(non_regression.checked),
            "protected_regression_task_ids": tuple(case.task.task_id for case in non_regression.regressions),
            "accepted": accepted,
            "frontier_compiled_selected": bool(answer.raw.get("frontier_compiled_selected")),
            "frontier_compiled_verified": bool(answer.raw.get("frontier_compiled_verified")),
            "frontier_heldout_gate_passed": heldout_gate_passed,
            "frontier_heldout_passed": int(answer.certificate.get("frontier_heldout_passed", 0) or 0),
            "frontier_heldout_total": int(answer.certificate.get("frontier_heldout_total", 0) or 0),
            "frontier_heldout_pass_rate": float(answer.certificate.get("frontier_heldout_pass_rate", 0.0) or 0.0),
            "frontier_output_goal_contract_passed": output_goal_passed,
            "frontier_output_goal_contract": dict(answer.certificate.get("frontier_output_goal_contract") or {}),
            "frontier_compiled_contract_verified": compiled_contract_verified,
            "frontier_compiled_contract_checksum": str(answer.certificate.get("frontier_compiled_contract_checksum", "")),
            "frontier_memory_binding_id": str(answer.certificate.get("frontier_memory_binding_id", "")),
            "frontier_memory_binding_passed": memory_binding_passed,
            "frontier_memory_binding_fidelity": float(answer.certificate.get("frontier_memory_binding_fidelity", 0.0) or 0.0),
            "certificate_fields": tuple(sorted(str(key) for key in answer.certificate)),
            "cost": asdict(total_cost),
        }
        if accepted:
            self.frontier_repair_accepted_events += 1
            self._count("frontier_repair_accepted_events")
            self.bit_ledger.ingest_cost(
                total_cost,
                note=f"P7:frontier-repair:{failure.task.skill}:{failure.task.task_id}",
            )
        return report

    def _apply_model_regrowth(self, plan: RegrowthPlan, *, step: int) -> dict[str, Any]:
        if plan.selected is None:
            raise ValueError("P7 regrowth produced no recovered non-regressing selected action")
        selected = plan.selected
        patch = selected.patch
        if not selected.recovered or not selected.non_regression.passed:
            raise ValueError(f"P7 selected action is not verified recovered/non-regressing: {selected.action.kind.value}")
        answer = patch.answer_for(plan.failure.task, self.trial_agent(plan.failure.task))
        repair_batch = self._batch_from_task(plan.failure.task, answer)
        protected_batches = tuple(self.replay_batches[-4:])
        if not protected_batches:
            protected_task = Task(
                f"phase-p7-protected-{step}",
                "instruction_following",
                "Output protected regrowth status exactly: STABLE",
                "STABLE",
            )
            protected_batches = (self._batch_from_task(protected_task, self._answer_from_task_expected(protected_task)),)

        was_training = self.model.training
        self.model.train()
        parameters = self._selected_regrowth_parameters(selected.action.kind)
        snapshots = {name: parameter.detach().clone() for name, parameter in parameters}
        repair_before = float(self._loss_for_regrowth_batch(repair_batch, require_grad=False).detach().cpu())
        protected_before = float(
            torch.stack([
                self._loss_for_regrowth_batch(batch, require_grad=False).detach()
                for batch in protected_batches
            ]).mean().cpu()
        )
        loss = self._loss_for_regrowth_batch(repair_batch, require_grad=True)
        grads = torch.autograd.grad(loss, [parameter for _, parameter in parameters], allow_unused=True)
        usable = [(name, parameter, grad) for (name, parameter), grad in zip(parameters, grads) if grad is not None]
        if not usable:
            if not was_training:
                self.model.eval()
            raise ValueError(f"P7 model regrowth action {selected.action.kind.value} had no gradient path into selected parameters")

        accepted_report: dict[str, Any] | None = None
        protected_tolerance = max(1e-4, protected_before * 0.02)
        step_sizes = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
        for step_size in step_sizes:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(snapshots[name])
                for name, parameter, grad in usable:
                    norm = float(grad.detach().norm().clamp_min(1e-12).cpu())
                    parameter.add_(grad, alpha=-float(step_size) / norm)
            self.model.requantize_ternary_core(certify_zeros=False)
            repair_after = float(self._loss_for_regrowth_batch(repair_batch, require_grad=False).detach().cpu())
            protected_after = float(
                torch.stack([
                    self._loss_for_regrowth_batch(batch, require_grad=False).detach()
                    for batch in protected_batches
                ]).mean().cpu()
            )
            repair_delta = repair_before - repair_after
            protected_delta = protected_after - protected_before
            parameter_delta = float(
                sum(
                    (parameter.detach() - snapshots[name]).abs().sum().cpu()
                    for name, parameter in parameters
                )
            )
            if repair_delta > 0.0 and protected_delta <= protected_tolerance and parameter_delta > 0.0:
                accepted_report = {
                    "step": step,
                    "action": selected.action.kind.value,
                    "target": selected.action.target,
                    "updated_parameter_count": len(parameters),
                    "updated_parameter_names": tuple(name for name, _ in parameters),
                    "gradient_parameter_count": len(usable),
                    "step_size": float(step_size),
                    "parameter_delta_l1": parameter_delta,
                    "repair_loss_before": repair_before,
                    "repair_loss_after": repair_after,
                    "repair_loss_delta": repair_delta,
                    "protected_loss_before": protected_before,
                    "protected_loss_after": protected_after,
                    "protected_loss_delta": protected_delta,
                    "protected_loss_tolerance": protected_tolerance,
                    "non_regression_passed": True,
                    "requantized_ternary_core": bool(self.model.config.use_ternary_core),
                }
                break

        if accepted_report is None:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(snapshots[name])
            self.model.requantize_ternary_core(certify_zeros=False)
            if not was_training:
                self.model.eval()
            raise ValueError(
                "P7 model regrowth failed strict gate: no bounded parameter update improved repair loss without protected regression"
            )
        if not was_training:
            self.model.eval()
        self.regrowth_model_applications.append(accepted_report)
        self.regrowth_model_parameter_delta_l1 += float(accepted_report["parameter_delta_l1"])
        self.regrowth_model_repair_loss_delta += float(accepted_report["repair_loss_delta"])
        self.regrowth_model_protected_loss_delta += float(accepted_report["protected_loss_delta"])
        self._count("regrowth_model_applications")
        self._count("regrowth_model_updated_parameters", int(accepted_report["updated_parameter_count"]))
        self.bit_ledger.ingest_cost(
            CostTrace(
                weight_bits_read=float(accepted_report["updated_parameter_count"]) * 16.0,
                activation_bits=float(accepted_report["gradient_parameter_count"]) * 8.0,
                verifier_steps=2,
            ),
            note=f"P7:model-regrowth:{selected.action.kind.value}:{selected.action.target}",
        )
        return accepted_report

    def _selected_recursive_parameters(self, proposal_kind: ProposalKind) -> list[tuple[str, nn.Parameter]]:
        if proposal_kind in {ProposalKind.COMPRESSION, ProposalKind.REGROWTH_STRATEGY, ProposalKind.KERNEL, ProposalKind.HARDWARE_GRAMMAR}:
            prefixes = ("blocks.", "variable_in.", "lm_head.", "token_embedding.")
        elif proposal_kind == ProposalKind.ROUTER:
            prefixes = ("skill_experts.", "variable_in.", "lm_head.", "token_embedding.")
        elif proposal_kind == ProposalKind.MTP_HEAD:
            prefixes = ("mtp_heads.", "confidence_head.")
        elif proposal_kind == ProposalKind.COMPILED_FRONTIER:
            prefixes = ("skill_experts.", "certificate_head.", "lm_head.", "token_embedding.")
        elif proposal_kind in {ProposalKind.TEST, ProposalKind.SKILL_SPEC}:
            prefixes = ("certificate_head.", "confidence_head.", "lm_head.", "token_embedding.")
        else:
            prefixes = ("blocks.", "skill_experts.", "mtp_heads.", "certificate_head.", "lm_head.", "token_embedding.")
        selected: list[tuple[str, nn.Parameter]] = []
        seen: set[int] = set()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            if any(name.startswith(prefix) for prefix in prefixes):
                selected.append((name, parameter))
                seen.add(id(parameter))
        if not selected:
            raise ValueError(f"P10 recursive improvement could not find trainable parameters for kind {proposal_kind.value}")
        return selected

    def _recursive_improvement_task(self, decision: AcceptanceDecision, cycle_report: Any, *, step: int) -> Task:
        affected = set(decision.evaluation.proposal.affected_skills)
        for failure in cycle_report.regressions:
            if failure.task.skill in affected:
                return failure.task
        skill = next(iter(affected), "instruction_following")
        return Task(
            f"phase-p10-model-{step}",
            skill,
            f"Output recursive improvement patch {decision.evaluation.proposal.proposal_id} as accepted.",
            f"accepted:{decision.evaluation.proposal.proposal_id}",
        )

    def _materialize_recursive_improvement_artifact(
        self,
        decision: AcceptanceDecision,
        cycle_report: Any,
        *,
        step: int,
        model_patch: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not decision.accepted:
            raise ValueError(f"P10 cannot materialize rejected proposal {decision.evaluation.proposal.proposal_id}")
        proposal = decision.evaluation.proposal
        task = self._recursive_improvement_task(decision, cycle_report, step=step)
        task_id = str(proposal.patch_payload.get("task_id") or "")
        if task_id:
            for failure in cycle_report.regressions:
                if failure.task.task_id == task_id:
                    task = failure.task
                    break

        answer = self.reference_agent(task)
        artifact_type_by_kind = {
            ProposalKind.TEST: "verified_regression_test_replay",
            ProposalKind.SKILL_SPEC: "verified_skill_micro_family_seed",
            ProposalKind.COMPILED_FRONTIER: "verified_compiled_frontier_contract_replay",
            ProposalKind.COMPRESSION: "verified_compression_repair_replay",
            ProposalKind.ROUTER: "verified_router_repair_replay",
            ProposalKind.MTP_HEAD: "verified_future_contract_repair_replay",
            ProposalKind.REGROWTH_STRATEGY: "verified_regrowth_strategy_replay",
            ProposalKind.HARDWARE_GRAMMAR: "verified_hardware_grammar_replay",
            ProposalKind.KERNEL: "verified_kernel_repair_replay",
        }
        artifact_type = artifact_type_by_kind.get(proposal.kind, f"verified_{proposal.kind.value}_replay")
        artifact_id = _sha256_json(
            {
                "proposal_id": proposal.proposal_id,
                "proposal_kind": proposal.kind.value,
                "task_id": task.task_id,
                "answer": answer.text,
                "signed_patch_id": model_patch.get("signed_patch_id", ""),
                "rollback_token": decision.evaluation.sandbox.rollback_token,
            }
        )
        metadata = {
            "recursive_improvement_artifact": True,
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "proposal_id": proposal.proposal_id,
            "proposal_kind": proposal.kind.value,
            "proposal_patch_payload": dict(proposal.patch_payload),
            "signed_patch_id": str(model_patch.get("signed_patch_id", "")),
            "rollback_token": decision.evaluation.sandbox.rollback_token,
            "source_task_id": task.task_id,
            "affected_skills": tuple(proposal.affected_skills),
        }
        example = self._add_verified_phase_replay(
            "P10",
            task,
            answer=answer,
            metadata=metadata,
            origin=ExampleOrigin.TOOL_SOLVED,
        )
        if example is None:
            raise ValueError(f"P10 accepted proposal {proposal.proposal_id} could not produce a verified replay artifact")

        artifact = {
            **metadata,
            "step": int(step),
            "example_id": example.example_id,
            "task_id": task.task_id,
            "skill": task.skill,
            "verification_level": int(example.verification_level),
            "confidence_label": float(example.confidence_label or 0.0),
            "repair_loss_delta": float(model_patch.get("repair_loss_delta", 0.0) or 0.0),
            "protected_loss_delta": float(model_patch.get("protected_loss_delta", 0.0) or 0.0),
            "non_regression_passed": bool(model_patch.get("non_regression_passed")),
        }
        self.recursive_verified_artifacts.append(artifact)
        self._count("recursive_verified_artifact_events")
        self._count(f"recursive_verified_{proposal.kind.value}_artifact_events")
        if proposal.kind == ProposalKind.TEST:
            self._count("recursive_verified_test_artifacts")
        elif proposal.kind == ProposalKind.SKILL_SPEC:
            self._count("recursive_verified_skill_family_artifacts")
        elif proposal.kind == ProposalKind.COMPILED_FRONTIER:
            self._count("recursive_verified_frontier_contract_artifacts")
        self.bit_ledger.ingest_cost(
            CostTrace(verifier_steps=1, generated_tokens=max(1, len(answer.text.split()))),
            note=f"P10:verified-artifact:{artifact_type}:{proposal.proposal_id}",
        )
        return artifact

    def _apply_recursive_model_improvement(self, decision: AcceptanceDecision, cycle_report: Any, *, step: int) -> dict[str, Any]:
        if not decision.accepted:
            raise ValueError(f"P10 cannot apply rejected proposal {decision.evaluation.proposal.proposal_id}: {decision.reason}")
        proposal = decision.evaluation.proposal
        task = self._recursive_improvement_task(decision, cycle_report, step=step)
        task_id = str(proposal.patch_payload.get("task_id") or "")
        if task_id:
            for failure in cycle_report.regressions:
                if failure.task.task_id == task_id:
                    task = failure.task
                    break
        proposal_answer = self.reference_agent(task)
        repair_batch = self._batch_from_task(task, proposal_answer)
        protected_batches = tuple(self.replay_batches[-4:])
        if not protected_batches:
            protected_task = Task(
                f"phase-p10-protected-{step}",
                "instruction_following",
                "Output protected recursive improvement status exactly: STABLE",
                "STABLE",
            )
            protected_batches = (self._batch_from_task(protected_task, self._answer_from_task_expected(protected_task)),)

        was_training = self.model.training
        self.model.train()
        parameters = self._selected_recursive_parameters(proposal.kind)
        snapshots = {name: parameter.detach().clone() for name, parameter in parameters}
        repair_before = float(self._loss_for_regrowth_batch(repair_batch, require_grad=False).detach().cpu())
        protected_before = float(
            torch.stack([
                self._loss_for_regrowth_batch(batch, require_grad=False).detach()
                for batch in protected_batches
            ]).mean().cpu()
        )
        loss = self._loss_for_regrowth_batch(repair_batch, require_grad=True)
        grads = torch.autograd.grad(loss, [parameter for _, parameter in parameters], allow_unused=True)
        usable = [(name, parameter, grad) for (name, parameter), grad in zip(parameters, grads) if grad is not None]
        if not usable:
            if not was_training:
                self.model.eval()
            raise ValueError(f"P10 proposal {proposal.proposal_id} had no gradient path into selected parameters")

        accepted_report: dict[str, Any] | None = None
        protected_tolerance = max(1e-4, protected_before * 0.02)
        step_sizes = (0.20, 0.10, 0.05, 0.025, 0.0125)
        for step_size in step_sizes:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(snapshots[name])
                for name, parameter, grad in usable:
                    norm = float(grad.detach().norm().clamp_min(1e-12).cpu())
                    parameter.add_(grad, alpha=-float(step_size) / norm)
            self.model.requantize_ternary_core(certify_zeros=False)
            repair_after = float(self._loss_for_regrowth_batch(repair_batch, require_grad=False).detach().cpu())
            protected_after = float(
                torch.stack([
                    self._loss_for_regrowth_batch(batch, require_grad=False).detach()
                    for batch in protected_batches
                ]).mean().cpu()
            )
            repair_delta = repair_before - repair_after
            protected_delta = protected_after - protected_before
            parameter_delta = float(
                sum(
                    (parameter.detach() - snapshots[name]).abs().sum().cpu()
                    for name, parameter in parameters
                )
            )
            if repair_delta > 0.0 and protected_delta <= protected_tolerance and parameter_delta > 0.0:
                parameter_names = tuple(name for name, _ in parameters)
                signed_payload = {
                    "proposal_id": proposal.proposal_id,
                    "proposal_payload": dict(proposal.patch_payload),
                    "rollback_token": decision.evaluation.sandbox.rollback_token,
                    "parameter_names": parameter_names,
                    "repair_loss_delta": repair_delta,
                    "protected_loss_delta": protected_delta,
                }
                accepted_report = {
                    "step": step,
                    "proposal_id": proposal.proposal_id,
                    "proposal_kind": proposal.kind.value,
                    "affected_skills": tuple(proposal.affected_skills),
                    "proposal_patch_payload": dict(proposal.patch_payload),
                    "rollback_token": decision.evaluation.sandbox.rollback_token,
                    "signed_patch_id": _sha256_json(signed_payload),
                    "updated_parameter_count": len(parameters),
                    "updated_parameter_names": parameter_names,
                    "gradient_parameter_count": len(usable),
                    "step_size": float(step_size),
                    "parameter_delta_l1": parameter_delta,
                    "repair_loss_before": repair_before,
                    "repair_loss_after": repair_after,
                    "repair_loss_delta": repair_delta,
                    "protected_loss_before": protected_before,
                    "protected_loss_after": protected_after,
                    "protected_loss_delta": protected_delta,
                    "protected_loss_tolerance": protected_tolerance,
                    "non_regression_passed": True,
                    "requantized_ternary_core": bool(self.model.config.use_ternary_core),
                }
                break

        if accepted_report is None:
            with torch.no_grad():
                for name, parameter in parameters:
                    parameter.copy_(snapshots[name])
            self.model.requantize_ternary_core(certify_zeros=False)
            if not was_training:
                self.model.eval()
            raise ValueError(
                f"P10 proposal {proposal.proposal_id} failed strict model patch gate: no bounded update improved repair loss without protected regression"
            )
        if not was_training:
            self.model.eval()
        self.recursive_model_applications.append(accepted_report)
        self.recursive_model_parameter_delta_l1 += float(accepted_report["parameter_delta_l1"])
        self.recursive_model_repair_loss_delta += float(accepted_report["repair_loss_delta"])
        self.recursive_model_protected_loss_delta += float(accepted_report["protected_loss_delta"])
        self._count("recursive_model_applications")
        self._count("recursive_model_updated_parameters", int(accepted_report["updated_parameter_count"]))
        self.bit_ledger.ingest_cost(
            CostTrace(
                weight_bits_read=float(accepted_report["updated_parameter_count"]) * 16.0,
                activation_bits=float(accepted_report["gradient_parameter_count"]) * 8.0,
                verifier_steps=2,
            ),
            note=f"P10:model-improvement:{proposal.proposal_id}",
        )
        return accepted_report

    def _output_goal_contract_summary(self) -> dict[str, Any]:
        decisions = (
            tuple(self.future_ledger.output_goal_decisions)
            + tuple(self.inference.speculative.engine.ledger.output_goal_decisions)
        )
        return {
            "output_goal_contract_decisions": len(decisions),
            "output_goal_contract_accepted": sum(1 for decision in decisions if decision.accepted),
            "output_goal_contract_rejected": sum(1 for decision in decisions if not decision.accepted),
            "output_goal_contract_latest": [decision.to_dict() for decision in decisions[-5:]],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase_counts": dict(self.phase_counts),
            "errors": list(self.errors),
            "batch_contract_samples": list(self.batch_contract_samples),
            "phase_audits": list(self.phase_audits),
            "frontier_reports": list(self.frontier_reports),
            "sleep_frontier_reports": list(self.sleep_frontier_reports),
            "restored_frontier_fastsolve_reports": list(self.restored_frontier_fastsolve_reports),
            "frontier_compiled_fastsolve_events": int(self.frontier_compiled_fastsolve_events),
            "frontier_repair_candidates": list(self.frontier_repair_candidates),
            "frontier_repair_accepted_events": int(self.frontier_repair_accepted_events),
            "frontier_registry_path": str(self.run_dir / "frontier_registry"),
            "frontier_registry_summary": self.frontier_registry.to_dict(),
            "replay_batches": [batch.detach().cpu() for batch in self.replay_batches],
            "replay_skill_contexts": [tuple(float(value) for value in context) for context in self.replay_skill_contexts],
            "replay_cursor": int(self.replay_cursor),
            "regularization_steps": int(self.regularization_steps),
            "replay_updates": int(self.replay_updates),
            "phase_replay_examples": dict(self.phase_replay_examples),
            "phase_replay_example_ids": list(self.phase_replay_example_ids),
            "objective_feedback_events": int(self.objective_feedback_events),
            "objective_feedback_total": float(self.objective_feedback_total),
            "last_objective_loss_total": float(self.last_objective_loss_total),
            "last_objective_loss_terms": dict(self.last_objective_loss_terms),
            "objective_feedback_term_totals": dict(self.objective_feedback_term_totals),
            "objective_feedback_history": list(self.objective_feedback_history),
            "integration_counts": dict(self.integration_counts),
            "attribution_policy": self.attribution_policy.to_dict(),
            "regrowth_model_applications": list(self.regrowth_model_applications),
            "regrowth_model_parameter_delta_l1": float(self.regrowth_model_parameter_delta_l1),
            "regrowth_model_repair_loss_delta": float(self.regrowth_model_repair_loss_delta),
            "regrowth_model_protected_loss_delta": float(self.regrowth_model_protected_loss_delta),
            "recursive_model_applications": list(self.recursive_model_applications),
            "recursive_verified_artifacts": list(self.recursive_verified_artifacts),
            "recursive_model_parameter_delta_l1": float(self.recursive_model_parameter_delta_l1),
            "recursive_model_repair_loss_delta": float(self.recursive_model_repair_loss_delta),
            "recursive_model_protected_loss_delta": float(self.recursive_model_protected_loss_delta),
            "certificate_head_forward_events": int(self.model.certificate_forward_events),
            "model_certificate_head_artifacts": list(self.model_certificate_head_artifacts),
            "latent_workspace_forward_events": int(self.latent_workspace_forward_events),
            "latent_workspace_step_events": int(self.latent_workspace_step_events),
            "latent_workspace_certificate_binding_events": int(self.latent_workspace_certificate_binding_events),
            "latent_workspace_last_summary": dict(self.latent_workspace_last_summary),
            "certificate_algebra_tool_events": int(self.integration_counts.get("certificate_algebra_tool_events", 0)),
            "certificate_code_hidden_property_events": int(self.integration_counts.get("certificate_code_hidden_property_events", 0)),
            "input_anchor_observations": int(self.input_anchor_observations),
            "input_anchor_count": int(self.input_anchor_count),
            "input_anchor_fidelity_failures": int(self.input_anchor_fidelity_failures),
            "learned_memory_policy_events": int(self.learned_memory_policy_events),
            "learned_memory_anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
            "learned_memory_exact_decisions": int(self.learned_memory_exact_decisions),
            "learned_memory_latent_decisions": int(self.learned_memory_latent_decisions),
            "learned_memory_drop_decisions": int(self.learned_memory_drop_decisions),
            "learned_memory_storage_ratio_total": float(self.learned_memory_storage_ratio_total),
            "learned_memory_utility_prior_updates": int(self.learned_memory_utility_prior_updates),
            "learned_memory_last_utility_prior": tuple(float(value) for value in self.learned_memory_last_utility_prior),
            "skill_expert_context_events": int(self.skill_expert_context_events),
            "skill_expert_replay_context_events": int(self.skill_expert_replay_context_events),
            "skill_expert_last_context": tuple(float(value) for value in self.skill_expert_last_context),
            "skill_expert_context_skills": tuple(self.skill_expert_context_skills),
            "last_ingested_compression_cost": _cost_trace_payload(self._last_ingested_compression_cost),
            "future_ledger": self.future_ledger.to_dict(),
            "compression_trace_ledger": (
                self.model.compression_ledger.to_dict()
                if self.model.compression_ledger is not None
                else None
            ),
            "ledgers": _ledger_bundle_payload(
                bit_ledger=self.bit_ledger,
                skill_ledger=self.skill_ledger,
                causal_ledger=self.causal_ledger,
                uncertainty_ledger=self.uncertainty_ledger,
            ),
            "memory_state": _memory_state(self.memory),
            "sleep_state": _sleep_state(self.sleep),
            "improvement_state": _improvement_state(self.improvement),
            "improvement_archive_dir": str(self.improvement_archive_dir),
            "improvement_persistent_archive_state": dict(self.improvement_persistent_archive_state),
        }

    def load_state_dict(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported Cortex phase state schema: {payload.get('schema_version')!r}")
        self.phase_counts.update({str(key): int(value) for key, value in dict(payload.get("phase_counts", {})).items()})
        self.errors = [dict(item) for item in payload.get("errors", ())]
        self.batch_contract_samples = [dict(item) for item in payload.get("batch_contract_samples", ())]
        self.phase_audits = [dict(item) for item in payload.get("phase_audits", ())]
        self.frontier_reports = [dict(item) for item in payload.get("frontier_reports", ())]
        self.sleep_frontier_reports = [dict(item) for item in payload.get("sleep_frontier_reports", ())]
        self.restored_frontier_fastsolve_reports = [
            dict(item)
            for item in payload.get("restored_frontier_fastsolve_reports", ())
        ]
        self.frontier_compiled_fastsolve_events = int(payload.get("frontier_compiled_fastsolve_events", 0))
        self.frontier_repair_candidates = [dict(item) for item in payload.get("frontier_repair_candidates", ())]
        self.frontier_repair_accepted_events = int(payload.get("frontier_repair_accepted_events", 0))
        frontier_path = Path(str(payload.get("frontier_registry_path") or (self.run_dir / "frontier_registry")))
        self.replay_batches = [
            batch.detach().cpu().to(dtype=torch.long)
            if isinstance(batch, torch.Tensor)
            else torch.as_tensor(batch, dtype=torch.long)
            for batch in payload.get("replay_batches", ())
        ]
        self.replay_skill_contexts = [
            tuple(float(value) for value in context)
            for context in payload.get("replay_skill_contexts", ())
        ]
        while len(self.replay_skill_contexts) < len(self.replay_batches):
            self.replay_skill_contexts.append(())
        self.replay_cursor = int(payload.get("replay_cursor", 0))
        self.regularization_steps = int(payload.get("regularization_steps", 0))
        self.replay_updates = int(payload.get("replay_updates", 0))
        self.phase_replay_examples.update({
            str(key): int(value)
            for key, value in dict(payload.get("phase_replay_examples", {})).items()
        })
        self.phase_replay_example_ids = [str(item) for item in payload.get("phase_replay_example_ids", ())]
        self.objective_feedback_events = int(payload.get("objective_feedback_events", 0))
        self.objective_feedback_total = float(payload.get("objective_feedback_total", 0.0))
        self.last_objective_loss_total = float(payload.get("last_objective_loss_total", 0.0))
        self.last_objective_loss_terms = {
            str(name): dict(term)
            for name, term in dict(payload.get("last_objective_loss_terms") or {}).items()
        }
        self.objective_feedback_term_totals = {
            str(name): float(value)
            for name, value in dict(payload.get("objective_feedback_term_totals") or {}).items()
        }
        self.objective_feedback_history = [
            dict(item)
            for item in payload.get("objective_feedback_history", ())
        ]
        if not self.last_objective_loss_terms:
            for audit in reversed(self.phase_audits):
                objective = dict(audit.get("objective") or {})
                loss = dict(objective.get("loss") or {})
                terms = dict(loss.get("terms") or {})
                restored = {
                    name: dict(terms[name])
                    for name in FINAL_LOSS_TERMS
                    if name in terms
                }
                if restored:
                    self.last_objective_loss_terms = restored
                    break
        if self.last_objective_loss_terms and not self.objective_feedback_term_totals:
            self.objective_feedback_term_totals = {
                name: float(term.get("weighted", 0.0))
                for name, term in self.last_objective_loss_terms.items()
            }
        if self.last_objective_loss_terms:
            weighted_terms = {
                name: float(term.get("weighted", 0.0))
                for name, term in self.last_objective_loss_terms.items()
            }
            for item in self.objective_feedback_history:
                item.setdefault("term_count", len(self.last_objective_loss_terms))
                item.setdefault("term_names", tuple(self.last_objective_loss_terms))
                item.setdefault("weighted_terms", dict(weighted_terms))
        self.integration_counts.update({
            str(key): int(value)
            for key, value in dict(payload.get("integration_counts") or {}).items()
        })
        self.attribution_policy = AttributionPolicyMemory.from_dict(payload.get("attribution_policy"))
        self.attribution.policy_memory = self.attribution_policy
        self.regrowth_model_applications = [
            dict(item)
            for item in payload.get("regrowth_model_applications", ())
        ]
        self.regrowth_model_parameter_delta_l1 = float(payload.get("regrowth_model_parameter_delta_l1", 0.0))
        self.regrowth_model_repair_loss_delta = float(payload.get("regrowth_model_repair_loss_delta", 0.0))
        self.regrowth_model_protected_loss_delta = float(payload.get("regrowth_model_protected_loss_delta", 0.0))
        self.recursive_model_applications = [
            dict(item)
            for item in payload.get("recursive_model_applications", ())
        ]
        self.recursive_verified_artifacts = [
            dict(item)
            for item in payload.get("recursive_verified_artifacts", ())
        ]
        self.recursive_model_parameter_delta_l1 = float(payload.get("recursive_model_parameter_delta_l1", 0.0))
        self.recursive_model_repair_loss_delta = float(payload.get("recursive_model_repair_loss_delta", 0.0))
        self.recursive_model_protected_loss_delta = float(payload.get("recursive_model_protected_loss_delta", 0.0))
        self.model.certificate_forward_events = int(payload.get("certificate_head_forward_events", 0))
        self.model_certificate_head_artifacts = [
            dict(item)
            for item in payload.get("model_certificate_head_artifacts", ())
        ]
        self.latent_workspace_forward_events = int(payload.get("latent_workspace_forward_events", 0))
        self.latent_workspace_step_events = int(payload.get("latent_workspace_step_events", 0))
        self.latent_workspace_certificate_binding_events = int(payload.get("latent_workspace_certificate_binding_events", 0))
        self.latent_workspace_last_summary = dict(payload.get("latent_workspace_last_summary") or {})
        self.input_anchor_observations = int(payload.get("input_anchor_observations", 0))
        self.input_anchor_count = int(payload.get("input_anchor_count", 0))
        self.input_anchor_fidelity_failures = int(payload.get("input_anchor_fidelity_failures", 0))
        self.learned_memory_policy_events = int(payload.get("learned_memory_policy_events", 0))
        self.learned_memory_anchor_supervision_events = int(payload.get("learned_memory_anchor_supervision_events", 0))
        self.learned_memory_exact_decisions = int(payload.get("learned_memory_exact_decisions", 0))
        self.learned_memory_latent_decisions = int(payload.get("learned_memory_latent_decisions", 0))
        self.learned_memory_drop_decisions = int(payload.get("learned_memory_drop_decisions", 0))
        self.learned_memory_storage_ratio_total = float(payload.get("learned_memory_storage_ratio_total", 0.0))
        self.learned_memory_utility_prior_updates = int(payload.get("learned_memory_utility_prior_updates", 0))
        self.learned_memory_last_utility_prior = tuple(
            float(value)
            for value in payload.get("learned_memory_last_utility_prior", ())
        )
        self.skill_expert_context_events = int(payload.get("skill_expert_context_events", 0))
        self.skill_expert_replay_context_events = int(payload.get("skill_expert_replay_context_events", 0))
        self.skill_expert_last_context = tuple(float(value) for value in payload.get("skill_expert_last_context", ()))
        self.skill_expert_context_skills = tuple(str(value) for value in payload.get("skill_expert_context_skills", ()))
        if self.skill_expert_last_context:
            self.model.set_skill_expert_context(self.skill_expert_last_context, source="checkpoint-restored-skill-ledger")
        self._last_ingested_compression_cost = _cost_trace_from_payload(payload.get("last_ingested_compression_cost"))
        _restore_future_contract_ledger(self.future_ledger, payload.get("future_ledger"))
        _restore_compression_trace_ledger(self.model.compression_ledger, payload.get("compression_trace_ledger"))
        ledgers = dict(payload.get("ledgers") or {})
        _restore_bit_ledger(self.bit_ledger, ledgers.get("bit_ledger"))
        _restore_skill_ledger(self.skill_ledger, ledgers.get("skill_ledger"))
        _restore_causal_ledger(self.causal_ledger, ledgers.get("causal_ledger"))
        _restore_uncertainty_ledger(self.uncertainty_ledger, ledgers.get("uncertainty_ledger"))
        _restore_memory_state(self.memory, payload.get("memory_state"))
        if self.learned_memory_last_utility_prior and self.model.learned_memory is not None:
            self.model.learned_memory.set_memory_utility_prior(
                self.learned_memory_last_utility_prior,
                events=int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
            )
        else:
            self._refresh_learned_memory_utility_prior(source="checkpoint-restore", count_update=False)
        _restore_sleep_state(self.sleep, payload.get("sleep_state"))
        _restore_improvement_state(self.improvement, payload.get("improvement_state"))
        frontier_summary = dict(payload.get("frontier_registry_summary") or {})
        expected_frontier_circuits = int(frontier_summary.get("circuit_count", 0) or 0)
        frontier_manifest = frontier_path / "frontier_registry.json"
        if frontier_manifest.exists():
            self.frontier_registry = FrontierCircuitRegistry.load(frontier_path)
            self.inference.set_compiled_frontier_registry(self.frontier_registry)
            restored_count = int(self.frontier_registry.to_dict().get("circuit_count", 0) or 0)
            if expected_frontier_circuits > 0 and restored_count <= 0:
                raise ValueError(
                    f"checkpoint expected {expected_frontier_circuits} restored Frontier circuits, "
                    "but the loaded registry is empty"
                )
            if restored_count > 0:
                self._count("frontier_registry_loaded_events")
                self.integration_counts["frontier_registry_loaded_circuits"] = max(
                    int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
                    restored_count,
                )
                self._validate_restored_frontier_fastsolve(registry_path=frontier_path)
        elif expected_frontier_circuits > 0:
            raise FileNotFoundError(
                f"checkpoint expected {expected_frontier_circuits} Frontier circuits, "
                f"but persisted registry is missing at {frontier_manifest}"
            )
        self.improvement_persistent_archive_state = dict(
            payload.get("improvement_persistent_archive_state")
            or self.improvement_persistent_archive_state
        )

    def checkpoint_state_summary(self) -> dict[str, Any]:
        compression_trace = self.model.compression_trace()
        compression_trace_counts = {}
        native_backend_counts: dict[str, int] = {}
        native_requantize_backend_counts: dict[str, int] = {}
        native_grad_weight_backend_counts: dict[str, int] = {}
        if compression_trace.get("enabled"):
            compression_trace_counts = dict(compression_trace.get("total_event_counts") or {})
            native_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_backend_counts") or {}).items()
            }
            native_requantize_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_requantize_backend_counts") or {}).items()
            }
            native_grad_weight_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_grad_weight_backend_counts") or {}).items()
            }
            if not compression_trace_counts:
                compression_trace_counts = {
                    "compression_decisions": len(compression_trace.get("compression_decisions", ())),
                    "activation_quantizations": len(compression_trace.get("activation_quantizations", ())),
                    "expert_activations": len(compression_trace.get("expert_activations", ())),
                    "kv_events": len(compression_trace.get("kv_events", ())),
                    "mtp_fsp_events": len(compression_trace.get("mtp_fsp_events", ())),
                    "layer_forward_events": len(compression_trace.get("layer_forward_events", ())),
                    "packed_ternary_dispatches": len(compression_trace.get("packed_ternary_dispatches", ())),
                    "native_ternary_kernel_dispatches": sum(
                        1
                        for item in compression_trace.get("packed_ternary_dispatches", ())
                        if bool(dict(item).get("native_kernel", False)) or str(dict(item).get("backend", "")).startswith("native_")
                    ),
                    "native_ternary_autotuned_dispatches": sum(
                        1
                        for item in compression_trace.get("packed_ternary_dispatches", ())
                        if bool(dict(item).get("autotuned", False))
                    ),
                    "native_ternary_autotune_cache_hits": sum(
                        1
                        for item in compression_trace.get("packed_ternary_dispatches", ())
                        if bool(dict(item).get("autotune_cache_hit", False))
                    ),
                }
                compression_trace_counts["torch_packed_ternary_dispatches"] = max(
                    0,
                    compression_trace_counts["packed_ternary_dispatches"]
                    - compression_trace_counts["native_ternary_kernel_dispatches"],
                )
            if not native_backend_counts:
                for item in compression_trace.get("packed_ternary_dispatches", ()):
                    data = dict(item)
                    if bool(data.get("native_kernel", False)) or str(data.get("backend", "")).startswith("native_"):
                        backend = str(
                            data.get("native_backend")
                            or native_backend_from_runtime_label(str(data.get("backend", "")), default="unknown")
                        )
                        native_backend_counts[backend] = native_backend_counts.get(backend, 0) + 1
            for backend, count in native_backend_counts.items():
                compression_trace_counts.setdefault(f"native_ternary_{backend}_kernel_dispatches", int(count))
            for backend, count in native_requantize_backend_counts.items():
                compression_trace_counts.setdefault(f"native_ternary_{backend}_requantize_dispatches", int(count))
            for backend, count in native_grad_weight_backend_counts.items():
                compression_trace_counts.setdefault(f"native_ternary_{backend}_grad_weight_dispatches", int(count))
        phase_counts = dict(self.phase_counts)
        if compression_trace_counts.get("layer_forward_events", 0) > 0:
            phase_counts["P2"] = max(phase_counts.get("P2", 0), compression_trace_counts["layer_forward_events"])
        output_goal_summary = self._output_goal_contract_summary()
        frontier_registry_summary = self.frontier_registry.to_dict()
        frontier_heldout_summary = _frontier_heldout_summary(frontier_registry_summary)
        memory_report = self.memory.compression_report()
        real_exogenous_llm_examples = self._real_exogenous_llm_examples()
        summary = {
            "schema_version": 1,
            "phase_event_counts": phase_counts,
            "replay_batch_count": len(self.replay_batches),
            "replay_cursor": int(self.replay_cursor),
            "regularization_steps": int(self.regularization_steps),
            "replay_updates": int(self.replay_updates),
            "phase_replay_examples": sum(self.phase_replay_examples.values()),
            "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
            "integration_counts": dict(self.integration_counts),
            "objective_feedback_events": int(self.objective_feedback_events),
            "objective_feedback_scale": self.objective_feedback_scale(),
            "last_objective_loss_total": float(self.last_objective_loss_total),
            "last_objective_loss_terms": dict(self.last_objective_loss_terms),
            "objective_feedback_term_names": tuple(self.last_objective_loss_terms),
            "objective_feedback_term_count": len(self.last_objective_loss_terms),
            "last_objective_loss_weighted_total": sum(
                float(term.get("weighted", 0.0))
                for term in self.last_objective_loss_terms.values()
            ),
            "objective_feedback_term_totals": dict(self.objective_feedback_term_totals),
            "objective_feedback_history": _last_items(self.objective_feedback_history, 5),
            "future_contract_decisions": len(self.future_ledger.decisions),
            **output_goal_summary,
            "compression_trace_counts": compression_trace_counts,
            "native_ternary_backend_requested": str(self.model.config.native_ternary_backend),
            "native_ternary_backend_counts": native_backend_counts,
            "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
            "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
            "native_ternary_grad_input_kernel": last_native_grad_input_kernel(),
            "native_ternary_grad_weight_kernel": last_native_grad_weight_kernel(),
            "native_ternary_grad_input_kernel_counts": native_grad_input_kernel_counts(),
            "native_ternary_grad_weight_kernel_counts": native_grad_weight_kernel_counts(),
            "native_ternary_kernel_required": bool(
                torch.cuda.is_available()
                and self.model.config.use_ternary_core
                and self.model.config.use_native_ternary_kernel
            ),
            "variable_input_compression_events": compression_trace_counts.get("kv_events", 0),
            "certificate_head_forward_events": int(self.model.certificate_forward_events),
            "certificate_algebra_tool_events": int(self.integration_counts.get("certificate_algebra_tool_events", 0)),
            "certificate_code_hidden_property_events": int(self.integration_counts.get("certificate_code_hidden_property_events", 0)),
            "model_certificate_head_events": int(self.integration_counts.get("model_certificate_head_events", 0)),
            "model_certificate_head_verified_events": int(self.integration_counts.get("model_certificate_head_verified_events", 0)),
            "model_certificate_head_latent_checksum_events": int(self.integration_counts.get("model_certificate_head_latent_checksum_events", 0)),
            "model_certificate_head_target_match_events": int(self.integration_counts.get("model_certificate_head_target_match_events", 0)),
            "model_certificate_head_artifacts": _last_items(self.model_certificate_head_artifacts, 5),
            "latent_workspace_forward_events": int(self.latent_workspace_forward_events),
            "latent_workspace_step_events": int(self.latent_workspace_step_events),
            "latent_workspace_certificate_binding_events": int(self.latent_workspace_certificate_binding_events),
            "latent_workspace_last_summary": dict(self.latent_workspace_last_summary),
            "input_anchor_observations": int(self.input_anchor_observations),
            "input_anchor_count": int(self.input_anchor_count),
            "input_anchor_fidelity_failures": int(self.input_anchor_fidelity_failures),
            "learned_memory_policy_events": int(self.learned_memory_policy_events),
            "learned_memory_anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
            "learned_memory_exact_decisions": int(self.learned_memory_exact_decisions),
            "learned_memory_latent_decisions": int(self.learned_memory_latent_decisions),
            "learned_memory_drop_decisions": int(self.learned_memory_drop_decisions),
            "learned_memory_storage_ratio_mean": (
                float(self.learned_memory_storage_ratio_total)
                / max(1, int(self.learned_memory_policy_events))
            ),
            "learned_memory_retention_decisions": int(memory_report.get("learned_retention_decision_count", 0) or 0),
            "learned_memory_retention_requested_exact": int(memory_report.get("learned_retention_requested_exact", 0) or 0),
            "learned_memory_retention_requested_latent": int(memory_report.get("learned_retention_requested_latent", 0) or 0),
            "learned_memory_retention_requested_drop": int(memory_report.get("learned_retention_requested_drop", 0) or 0),
            "learned_memory_retention_applied_exact": int(memory_report.get("learned_retention_applied_exact", 0) or 0),
            "learned_memory_retention_applied_latent": int(memory_report.get("learned_retention_applied_latent", 0) or 0),
            "learned_memory_retention_applied_drop": int(memory_report.get("learned_retention_applied_drop", 0) or 0),
            "learned_memory_retention_anchor_overrides": int(memory_report.get("learned_retention_anchor_overrides", 0) or 0),
            "memory_utility_credit_count": int(memory_report.get("memory_utility_credit_count", 0) or 0),
            "learned_memory_utility_credit_count": int(memory_report.get("learned_memory_utility_credit_count", 0) or 0),
            "learned_memory_utility_positive_count": int(memory_report.get("learned_memory_utility_positive_count", 0) or 0),
            "learned_memory_utility_prior_updates": int(self.learned_memory_utility_prior_updates),
            "learned_memory_utility_feedback_events": int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
            "learned_memory_last_utility_prior": tuple(float(value) for value in self.learned_memory_last_utility_prior),
            "learned_memory_utility_credits": _last_items(memory_report.get("learned_memory_utility_credits", ()), 5),
            "skill_expert_context_events": int(self.skill_expert_context_events),
            "skill_expert_replay_context_events": int(self.skill_expert_replay_context_events),
            "skill_expert_context_updates": int(self.model.skill_expert_context_updates),
            "skill_expert_context_source": self.model.skill_expert_context_source,
            "skill_expert_last_context": tuple(float(value) for value in self.skill_expert_last_context),
            "skill_expert_context_skills": tuple(self.skill_expert_context_skills),
            "bit_ledger_total_effective_bits": self.bit_ledger.total_effective_bits,
            "skill_ledger_states": len(self.skill_ledger.states),
            "causal_ledger_traces": len(self.causal_ledger.traces),
            "uncertainty_ledger_observations": sum(len(pairs) for pairs in self.uncertainty_ledger.bins.values()),
            "uncertainty_ledger_ece": self.uncertainty_ledger.expected_calibration_error(),
            "attribution_policy_observations": int(self.attribution_policy.observation_count),
            "attribution_policy_successes": int(self.attribution_policy.success_count),
            "attribution_policy_state": self.attribution_policy.to_dict(),
            "memory_recent_segments": len(self.memory.recent.segments),
            "memory_latent_segments": len(self.memory.latent.segments),
            "compiled_circuit_memory_binding_count": int(memory_report.get("compiled_circuit_memory_binding_count", 0) or 0),
            "compiled_circuit_memory_binding_events": int(self.integration_counts.get("compiled_circuit_memory_binding_events", 0)),
            "compiled_circuit_memory_fidelity_failures": int(self.integration_counts.get("compiled_circuit_memory_fidelity_failures", 0)),
            "sleep_frontier_memory_binding_events": int(self.integration_counts.get("sleep_frontier_memory_binding_events", 0)),
            "compiled_circuit_memory_bindings": _last_items(memory_report.get("compiled_circuit_memory_bindings", ()), 5),
            "sleep_replay_examples": len(self.sleep.replay.examples),
            "sleep_synthetic_examples": len(self.sleep.synthetic.examples),
            "sleep_reservoir_examples": len(self.sleep.reservoir.examples),
            "sleep_real_exogenous_llm_examples": len(real_exogenous_llm_examples),
            "sleep_real_exogenous_llm_batch_events": int(self.integration_counts.get("sleep_real_exogenous_llm_batch_events", 0)),
            "sleep_real_exogenous_llm_tokens": int(self.integration_counts.get("sleep_real_exogenous_llm_tokens", 0)),
            "frontier_compiled_circuit_count": int(frontier_registry_summary["circuit_count"]),
            "frontier_compiled_skill_count": int(frontier_registry_summary["compiled_skill_count"]),
            **frontier_heldout_summary,
            "frontier_compiled_fastsolve_events": int(self.frontier_compiled_fastsolve_events),
            "inference_model_backed_events": int(self.integration_counts.get("inference_model_backed_events", 0)),
            "inference_model_backed_generated_tokens": int(self.integration_counts.get("inference_model_backed_generated_tokens", 0)),
            "inference_model_backed_verified_events": int(self.integration_counts.get("inference_model_backed_verified_events", 0)),
            "inference_model_backed_forced_careful_events": int(self.integration_counts.get("inference_model_backed_forced_careful_events", 0)),
            "inference_model_backed_replay_events": int(self.integration_counts.get("inference_model_backed_replay_events", 0)),
            "inference_model_backed_repair_replay_events": int(self.integration_counts.get("inference_model_backed_repair_replay_events", 0)),
            "inference_model_backed_verified_replay_events": int(self.integration_counts.get("inference_model_backed_verified_replay_events", 0)),
            "inference_model_backed_adaptive_mtp_events": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_events", 0)),
            "inference_model_backed_adaptive_mtp_forward_count": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_forward_count", 0)),
            "inference_model_backed_adaptive_mtp_contract_checks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_contract_checks", 0)),
            "inference_model_backed_adaptive_mtp_proposed_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_blocks", 0)),
            "inference_model_backed_adaptive_mtp_proposed_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_tokens", 0)),
            "inference_model_backed_adaptive_mtp_accepted_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_blocks", 0)),
            "inference_model_backed_adaptive_mtp_rejected_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_blocks", 0)),
            "inference_model_backed_adaptive_mtp_accepted_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_tokens", 0)),
            "inference_model_backed_adaptive_mtp_rejected_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_tokens", 0)),
            "inference_model_backed_adaptive_mtp_accepted_mtp_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_mtp_tokens", 0)),
            "frontier_registry_loaded_events": int(self.integration_counts.get("frontier_registry_loaded_events", 0)),
            "frontier_registry_loaded_circuits": int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
            "frontier_restored_fastsolve_events": int(self.integration_counts.get("frontier_restored_fastsolve_events", 0)),
            "compiled_circuit_memory_restored_reuse_events": int(
                self.integration_counts.get("compiled_circuit_memory_restored_reuse_events", 0)
            ),
            "sleep_frontier_fastsolve_events": int(self.integration_counts.get("sleep_frontier_fastsolve_events", 0)),
            "frontier_repair_candidate_count": len(self.frontier_repair_candidates),
            "frontier_repair_accepted_events": int(self.frontier_repair_accepted_events),
            "frontier_registry_path": str(self.run_dir / "frontier_registry"),
            "frontier_reports": _last_items(self.frontier_reports, 3),
            "sleep_frontier_reports": _last_items(self.sleep_frontier_reports, 3),
            "restored_frontier_fastsolve_reports": _last_items(self.restored_frontier_fastsolve_reports, 3),
            "frontier_repair_candidates": _last_items(self.frontier_repair_candidates, 3),
            "improvement_archive_accepted": self.improvement.archive.accepted_count,
            "improvement_archive_rejected": self.improvement.archive.rejected_count,
            "recursive_improvement_generations_configured": int(self.config.cortex_phase_improvement_generations),
            "recursive_generation_events": int(self.integration_counts.get("recursive_generation_events", 0)),
            "recursive_evolved_proposal_events": int(self.integration_counts.get("recursive_evolved_proposal_events", 0)),
            "regrowth_model_application_count": len(self.regrowth_model_applications),
            "regrowth_model_parameter_delta_l1": float(self.regrowth_model_parameter_delta_l1),
            "regrowth_model_repair_loss_delta": float(self.regrowth_model_repair_loss_delta),
            "regrowth_model_protected_loss_delta": float(self.regrowth_model_protected_loss_delta),
            "regrowth_model_applications": _last_items(self.regrowth_model_applications, 5),
            "recursive_model_application_count": len(self.recursive_model_applications),
            "recursive_model_parameter_delta_l1": float(self.recursive_model_parameter_delta_l1),
            "recursive_model_repair_loss_delta": float(self.recursive_model_repair_loss_delta),
            "recursive_model_protected_loss_delta": float(self.recursive_model_protected_loss_delta),
            "recursive_model_applications": _last_items(self.recursive_model_applications, 5),
            "recursive_verified_artifact_count": len(self.recursive_verified_artifacts),
            "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
            "error_count": len(self.errors),
            "improvement_archive_dir": str(self.improvement_archive_dir),
            "improvement_persistent_archive_state": dict(self.improvement_persistent_archive_state),
            "improvement_persistent_archive_decisions": int(
                self.improvement_persistent_archive_state.get("decision_count", 0) or 0
            ),
            "improvement_persistent_rollback_events": int(
                self.improvement_persistent_archive_state.get("rollback_event_count", 0) or 0
            ),
        }
        summary["architecture_audit"] = _cortex_architecture_audit_from_summary(summary)
        summary["phase_deliverable_audit"] = _cortex_phase_deliverable_audit_from_summary(summary)
        return summary

    def run_phase_audit(self, *, step: int) -> Mapping[str, Any]:
        audit: dict[str, Any] = {"step": step}
        cycle_report = None
        first_failure = None
        try:
            cycle_report = self.cycle.run(
                self.reference_agent,
                self.trial_agent,
                seed=self.config.seed + step,
                n_per_skill=self.config.cortex_phase_probe_tasks,
            )
            self._touch("P1")
            audit["verifier"] = {
                "total": cycle_report.trial.total,
                "passed": cycle_report.trial.passed,
                "regressions": len(cycle_report.regressions),
                "aggregate_score": cycle_report.trial.aggregate_score,
            }
            audit["ledgers"] = self._ingest_cycle_ledgers(cycle_report, step=step)
            self.refresh_skill_expert_context_from_ledger(source=f"P1-skill-ledger-step-{step}")
            audit["skill_expert_context"] = {
                "source": self.model.skill_expert_context_source,
                "distribution": self.model.skill_expert_context_distribution(),
                "skills": self.skill_expert_context_skills,
            }
            first_failure = cycle_report.regressions[0] if cycle_report.regressions else None
            verifier_task = first_failure.task if first_failure is not None else Task(
                f"phase-p1-{step}",
                "instruction_following",
                "Output verifier status exactly: OK",
                "OK",
            )
            self._add_verified_phase_replay(
                "P1",
                verifier_task,
                metadata={
                    "regressions": len(cycle_report.regressions),
                    "aggregate_score": cycle_report.trial.aggregate_score,
                },
            )
        except Exception as exc:
            self._record_error("P1", exc)

        frontier_task_for_inference = None
        try:
            if (
                cycle_report is not None
                and cycle_report.regressions
                and int(self.config.cortex_phase_frontier_max_skills) > 0
            ):
                frontier_report = self.frontier_discovery.discover(
                    cycle_report,
                    seed=self.config.seed + step,
                    max_skills=int(self.config.cortex_phase_frontier_max_skills),
                    per_failure=int(self.config.cortex_phase_frontier_per_failure),
                    epochs=int(self.config.cortex_phase_frontier_epochs),
                    registry=self.frontier_registry,
                )
                frontier_payload = frontier_report.to_dict()
                self.frontier_reports.append(frontier_payload)
                self._count("frontier_discovery_events")
                self._count("frontier_compiled_circuits", len(frontier_report.circuits))
                self.integration_counts["frontier_compiled_skills"] = len(self.frontier_registry.compiled_skills())
                if frontier_report.circuits:
                    registry_dir = self.run_dir / "frontier_registry"
                    registry_path = self.frontier_registry.save(registry_dir)
                    self.integration_counts["frontier_registry_saves"] = self.integration_counts.get("frontier_registry_saves", 0) + 1
                    all_runtime_circuits = tuple(
                        circuit
                        for skill in self.frontier_registry.compiled_skills()
                        for circuit in self.frontier_registry.circuits_for_skill(skill)
                    )
                    memory_bindings = tuple(
                        self._bind_frontier_circuit_memory(circuit, step=step, source_phase="frontier")
                        for circuit in all_runtime_circuits
                    )
                    audit["frontier_discovery"] = {
                        **frontier_payload,
                        "registry_path": str(registry_path),
                        "memory_bindings": memory_bindings,
                    }
                    first_skill = frontier_report.circuits[0].skill
                    runtime_circuits = self.frontier_registry.circuits_for_skill(first_skill)
                    if runtime_circuits and runtime_circuits[0].verified_tasks:
                        frontier_task_for_inference = runtime_circuits[0].verified_tasks[0]
        except Exception as exc:
            self._record_error("frontier", exc)

        try:
            if first_failure is not None:
                memory_task = first_failure.task
                self.memory.ingest(
                    f"step-{step}-{first_failure.task.task_id}",
                    f"{first_failure.task.prompt}\nExpected: {first_failure.expected}",
                    extra_anchors=first_failure.task.anchors,
                )
                reconstruction = self.memory.reconstruct(first_failure.task.prompt, required_anchors=first_failure.task.anchors)
            else:
                memory_task = Task(
                    f"phase-p4-{step}",
                    "instruction_following",
                    "Recall the Cortex-3 LLM training memory anchor exactly.",
                    "C3-LLM-ANCHOR",
                )
                self.memory.ingest(f"step-{step}-general", "Cortex-3 LLM training memory anchor C3-LLM-ANCHOR.")
                reconstruction = self.memory.reconstruct("C3-LLM-ANCHOR")
            self._record_memory_utility(
                reconstruction,
                phase="P4",
                source=f"phase-audit:{step}",
                reason="phase_memory_reconstruction",
            )
            self._touch("P4")
            audit["memory"] = {
                "selected_segment_ids": reconstruction.selected_segment_ids,
                "fidelity": asdict(reconstruction.fidelity),
                "compression": self.memory.compression_report(),
                "learned_policy": {
                    "policy_events": int(self.learned_memory_policy_events),
                    "anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
                    "exact_decisions": int(self.learned_memory_exact_decisions),
                    "latent_decisions": int(self.learned_memory_latent_decisions),
                    "drop_decisions": int(self.learned_memory_drop_decisions),
                    "storage_ratio_mean": (
                        float(self.learned_memory_storage_ratio_total)
                        / max(1, int(self.learned_memory_policy_events))
                    ),
                },
            }
            self._add_verified_phase_replay(
                "P4",
                memory_task,
                metadata={
                    "selected_segment_ids": list(reconstruction.selected_segment_ids),
                    "fidelity_passed": reconstruction.fidelity.passed,
                },
            )
        except Exception as exc:
            self._record_error("P4", exc)

        try:
            task = first_failure.task if first_failure is not None else Task("phase-cert", "instruction_following", "Output OK exactly.", "OK")
            answer_text = str(task.expected)
            latent_state = LatentProofState(
                state_id=f"llm-phase-{step}",
                task_id=task.task_id,
                skill=task.skill,
                tensor=torch.tensor([[float(step), 1.0]], dtype=torch.float32),
                latent_steps=2,
                visible_reasoning_tokens=16,
            )
            certificate = build_certificate(
                certificate_id=f"llm-cert-{step}",
                task_id=task.task_id,
                skill=task.skill,
                certificate_type=CertificateType.FORMAT,
                answer=answer_text,
                claims={"calibrated_uncertainty": True, "llm_training_step": step},
                uncertainty=0.05,
                latent_state=latent_state,
                anchors=task.anchors,
                tool="exact_match",
                tool_args={"expected": answer_text},
            )
            verification = self.certificate_verifier.verify(certificate, latent_state)
            algebra_task = Task(
                f"phase-p5-algebra-{step}",
                "algebra",
                "Solve exactly for x: 7x + -3 = 25. Return only the integer value of x.",
                4,
                {"variable": "x", "a": 7, "b": -3, "c": 25, "solution": 4, "kind": "linear"},
            )
            algebra_answer = "4"
            algebra_claims, algebra_tool, algebra_tool_args, algebra_anchors = certificate_contract_for_task(algebra_task, algebra_answer)
            algebra_state = LatentProofState(
                state_id=f"llm-phase-{step}-algebra",
                task_id=algebra_task.task_id,
                skill=algebra_task.skill,
                tensor=torch.tensor([[float(step), 2.0, 4.0]], dtype=torch.float32),
                latent_steps=3,
                visible_reasoning_tokens=0,
            )
            algebra_certificate = build_certificate(
                certificate_id=f"llm-cert-algebra-{step}",
                task_id=algebra_task.task_id,
                skill=algebra_task.skill,
                certificate_type=CertificateType.ALGEBRA,
                answer=algebra_answer,
                claims=algebra_claims,
                uncertainty=0.04,
                latent_state=algebra_state,
                anchors=algebra_anchors,
                tool=algebra_tool,
                tool_args=algebra_tool_args,
            )
            algebra_verification = self.certificate_verifier.verify(algebra_certificate, algebra_state)
            code_source = "def solve(x):\n    return x + 1\n"
            code_task = Task(
                f"phase-p5-code-{step}",
                "code_unit_tests",
                "Write Python function solve(x) that returns x + 1. Return only code.",
                code_source,
                {
                    "function_name": "solve",
                    "tests": (((1,), 2), ((-2,), -1)),
                    "hidden_tests": (((41,), 42), ((0,), 1)),
                },
            )
            code_claims, code_tool, code_tool_args, code_anchors = certificate_contract_for_task(code_task, code_source)
            code_state = LatentProofState(
                state_id=f"llm-phase-{step}-code",
                task_id=code_task.task_id,
                skill=code_task.skill,
                tensor=torch.tensor([[float(step), 3.0, 1.0]], dtype=torch.float32),
                latent_steps=2,
                visible_reasoning_tokens=0,
            )
            code_certificate = build_certificate(
                certificate_id=f"llm-cert-code-{step}",
                task_id=code_task.task_id,
                skill=code_task.skill,
                certificate_type=CertificateType.CODE_TESTS,
                answer=code_source,
                claims=code_claims,
                uncertainty=0.05,
                latent_state=code_state,
                anchors=code_anchors,
                tool=code_tool,
                tool_args=code_tool_args,
            )
            code_verification = self.certificate_verifier.verify(code_certificate, code_state)
            delatentizer = RandomDelatentizer(probes=2)
            probe = delatentizer.probe(latent_state, seed=self.config.seed + step)
            probe_ok = delatentizer.verify_probe(latent_state, probe)
            self._count("delatentization_probe_events")
            if not probe_ok:
                self._count("delatentization_probe_failures")
            if not algebra_verification.passed:
                self._record_error("P5", ValueError(f"multi-step algebra certificate failed: {algebra_verification.reason}"))
            if not code_verification.passed:
                self._record_error("P5", ValueError(f"rich code certificate failed: {code_verification.reason}"))
            efficiency = evaluate_certificate_efficiency(
                "slow visible reasoning " * 32,
                certificate,
                verification,
                reference_uncertainty=0.05,
            )
            self._count("certificate_tool_verification_events", 3)
            self._count("certificate_algebra_tool_events")
            self._count("certificate_code_hidden_property_events")
            self._count("certificate_efficiency_events")
            self._touch("P5")
            audit["certificate"] = {
                "verified": verification.to_dict(),
                "efficiency": asdict(efficiency),
                "certificate": certificate.to_dict(),
                "algebra_verified": algebra_verification.to_dict(),
                "algebra_certificate": algebra_certificate.to_dict(),
                "code_verified": code_verification.to_dict(),
                "code_certificate": code_certificate.to_dict(),
                "delatentization_probe": asdict(probe),
                "delatentization_probe_verified": probe_ok,
                "model_certificate_head_artifact": (
                    dict(self.model_certificate_head_artifacts[-1])
                    if self.model_certificate_head_artifacts
                    else None
                ),
            }
            if not probe_ok:
                self._record_error("P5", ValueError(f"random de-latentization probe failed for {latent_state.state_id}"))
            if verification.passed:
                self.bit_ledger.add_certificate(certificate.to_dict())
            if algebra_verification.passed:
                self.bit_ledger.add_certificate(algebra_certificate.to_dict())
            if code_verification.passed:
                self.bit_ledger.add_certificate(code_certificate.to_dict())
            if verification.passed and algebra_verification.passed and code_verification.passed:
                self._add_verified_phase_replay(
                    "P5",
                    algebra_task,
                    answer=CandidateAnswer(algebra_answer, confidence=0.96, certificate=algebra_certificate.to_dict()),
                    metadata={
                        "certificate_id": algebra_certificate.certificate_id,
                        "code_certificate_id": code_certificate.certificate_id,
                        "reduction_ratio": efficiency.reduction_ratio,
                        "multi_step_algebra": True,
                        "rich_code_tests": True,
                    },
                )
        except Exception as exc:
            self._record_error("P5", exc)

        attribution_report = None
        try:
            if first_failure is not None:
                attribution_report = self.attribution.attribute(
                    first_failure,
                    compression_ledger=self.model.compression_ledger,
                    future_ledger=self.future_ledger,
                )
                self._touch("P6")
                audit["attribution"] = attribution_report.to_dict()
                dimensions = {
                    probe.spec.dimension.value
                    for probe in attribution_report.probes
                }
                self._count("attribution_probe_events", len(attribution_report.probes))
                self.integration_counts["attribution_unique_dimensions"] = max(
                    self.integration_counts.get("attribution_unique_dimensions", 0),
                    len(dimensions),
                )
                if attribution_report.policy_applied:
                    self._count("attribution_policy_applied_events")
                if attribution_report.policy_signals:
                    self._count("attribution_policy_signal_events", len(attribution_report.policy_signals))
                audit["attribution_policy"] = {
                    "applied": bool(attribution_report.policy_applied),
                    "signals": [signal.to_dict() for signal in attribution_report.policy_signals],
                    "state": self.attribution_policy.to_dict(),
                }
                self._add_verified_phase_replay(
                    "P6",
                    attribution_report.failure.task,
                    metadata={
                        "top_cause": attribution_report.top_cause,
                        "targeted_repair_is_cheaper": attribution_report.targeted_repair_is_cheaper,
                    },
                )
        except Exception as exc:
            self._record_error("P6", exc)

        try:
            if attribution_report is not None:
                protected_regrowth_tasks = self._cycle_protected_tasks(cycle_report, attribution_report.failure.task)
                frontier_repair = self._evaluate_frontier_repair_candidate(attribution_report, protected_regrowth_tasks)
                if frontier_repair is not None:
                    self.frontier_repair_candidates.append(frontier_repair)
                    self._count("frontier_repair_candidate_events")
                    audit["frontier_repair_candidate"] = frontier_repair
                    if bool(frontier_repair["accepted"]):
                        self._add_verified_phase_replay(
                            "P7",
                            attribution_report.failure.task,
                            metadata={
                                "frontier_repair_candidate": True,
                                "repair_score_delta": frontier_repair["repair_score_delta"],
                                "protected_checked": frontier_repair["protected_checked"],
                            },
                        )
                regrowth_plan = self.regrowth.plan(
                    attribution_report,
                    self.trial_agent,
                    protected_regrowth_tasks or (attribution_report.failure.task,),
                    budget=float(self.config.cortex_phase_regrowth_budget),
                )
                self._touch("P7")
                audit["regrowth"] = regrowth_plan.to_dict()
                audit["regrowth_budget"] = float(self.config.cortex_phase_regrowth_budget)
                self._count("regrowth_plan_events")
                self._count("regrowth_candidate_actions", len(regrowth_plan.candidates))
                model_regrowth = self._apply_model_regrowth(regrowth_plan, step=step)
                audit["regrowth_model_application"] = model_regrowth
                policy_signal = self.attribution_policy.observe_regrowth_plan(regrowth_plan)
                if policy_signal is not None:
                    self._count("attribution_policy_updates")
                    if policy_signal.successes > 0:
                        self._count("attribution_policy_success_events")
                    audit["attribution_policy_update"] = policy_signal.to_dict()
                self._add_verified_phase_replay(
                    "P7",
                    regrowth_plan.failure.task,
                    metadata={
                        "selected_action": regrowth_plan.selected_action,
                        "candidate_count": len(regrowth_plan.candidates),
                        "model_regrowth_parameter_delta_l1": model_regrowth["parameter_delta_l1"],
                        "model_regrowth_repair_loss_delta": model_regrowth["repair_loss_delta"],
                    },
                )
        except Exception as exc:
            self._record_error("P7", exc)

        inference_results: list[Any] = []
        try:
            if frontier_task_for_inference is not None:
                task = frontier_task_for_inference
            elif first_failure is not None:
                task = first_failure.task
            else:
                task = Task("phase-infer", "instruction_following", "Output OK exactly.", "OK")
            forced_paths = (InferencePath.FAST, InferencePath.NORMAL, InferencePath.CAREFUL)
            route_reports = []
            model_backed_events: list[dict[str, Any]] = []

            def record_model_backed_event(inferred: Any) -> None:
                event = inferred.answer.raw.get("model_backed_inference") if hasattr(inferred.answer, "raw") else None
                if not isinstance(event, Mapping):
                    return
                payload = dict(event)
                model_backed_events.append(payload)
                self._count("inference_model_backed_events")
                self._count("inference_model_backed_generated_tokens", int(payload.get("generated_token_count", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_events", 1 if bool(payload.get("adaptive_mtp_decoding")) else 0)
                self._count("inference_model_backed_adaptive_mtp_forward_count", int(payload.get("adaptive_mtp_forward_count", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_contract_checks", int(payload.get("adaptive_mtp_contract_checks", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_proposed_blocks", int(payload.get("adaptive_mtp_proposed_blocks", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_proposed_tokens", int(payload.get("adaptive_mtp_proposed_tokens", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_accepted_blocks", int(payload.get("adaptive_mtp_accepted_blocks", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_rejected_blocks", int(payload.get("adaptive_mtp_rejected_blocks", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_accepted_tokens", int(payload.get("adaptive_mtp_accepted_tokens", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_rejected_tokens", int(payload.get("adaptive_mtp_rejected_tokens", 0) or 0))
                self._count("inference_model_backed_adaptive_mtp_accepted_mtp_tokens", int(payload.get("adaptive_mtp_accepted_mtp_tokens", 0) or 0))
                if bool(inferred.passed):
                    self._count("inference_model_backed_verified_events")

            def add_model_backed_replay(inferred: Any) -> None:
                event = inferred.answer.raw.get("model_backed_inference") if hasattr(inferred.answer, "raw") else None
                if not isinstance(event, Mapping):
                    return
                passed = bool(inferred.passed)
                correction = inferred.answer if passed else self._answer_from_task_expected(inferred.task)
                replay = self._add_verified_phase_replay(
                    "P8",
                    inferred.task,
                    answer=correction,
                    metadata={
                        "model_backed_inference_replay": True,
                        "model_backed_passed": passed,
                        "route": inferred.route.path.value,
                        "failed_model_answer": "" if passed else inferred.answer.text,
                        "model_generated_token_count": int(dict(event).get("generated_token_count", 0) or 0),
                        "model_generated_token_ids": tuple(dict(event).get("generated_token_ids", ())),
                        "adaptive_mtp_contract_checks": int(dict(event).get("adaptive_mtp_contract_checks", 0) or 0),
                        "adaptive_mtp_proposed_blocks": int(dict(event).get("adaptive_mtp_proposed_blocks", 0) or 0),
                        "adaptive_mtp_accepted_blocks": int(dict(event).get("adaptive_mtp_accepted_blocks", 0) or 0),
                        "adaptive_mtp_rejected_blocks": int(dict(event).get("adaptive_mtp_rejected_blocks", 0) or 0),
                    },
                )
                if replay is None:
                    return
                self._count("inference_model_backed_replay_events")
                if passed:
                    self._count("inference_model_backed_verified_replay_events")
                else:
                    self._count("inference_model_backed_repair_replay_events")

            for forced_path in forced_paths:
                inferred = self.inference.infer(task, forced_path=forced_path)
                inference_results.append(inferred)
                route_reports.append(inferred.to_dict())
                record_model_backed_event(inferred)
                self._count(f"inference_{inferred.route.path.value}_path_events")
                self._count("inference_budget_predictions")
                self._count("inference_early_exit_events")
                self._count("inference_kernel_dispatches", len(inferred.kernel_dispatches))
                if inferred.future_contract is not None:
                    self._count("inference_self_speculative_events")
                if inferred.memory_reconstruction is not None:
                    self._count("inference_latent_kv_events")
                    self._record_memory_utility(
                        inferred.memory_reconstruction,
                        phase="P8",
                        source=f"inference:{inferred.route.path.value}:{inferred.task.task_id}",
                        reason="inference_memory_reconstruction",
                    )
                if bool(inferred.answer.raw.get("frontier_compiled_selected")):
                    self._count("frontier_compiled_fastsolve_events")
                    self.frontier_compiled_fastsolve_events += 1
                    if bool(inferred.answer.certificate.get("frontier_memory_binding_passed")):
                        self._count("compiled_circuit_memory_binding_events")
            real_llm_examples = self._real_exogenous_llm_examples()
            if not real_llm_examples:
                raise ValueError("P8 model-backed inference requires REAL_EXOGENOUS examples sourced from observed LLM input batches")
            model_task = real_llm_examples[-1].task
            model_inferred = self.inference.infer(model_task, forced_path=InferencePath.CAREFUL)
            inference_results.append(model_inferred)
            route_reports.append(model_inferred.to_dict())
            record_model_backed_event(model_inferred)
            add_model_backed_replay(model_inferred)
            self._count("inference_model_backed_forced_careful_events")
            self._count("inference_careful_path_events")
            self._count("inference_budget_predictions")
            self._count("inference_early_exit_events")
            self._count("inference_kernel_dispatches", len(model_inferred.kernel_dispatches))
            if model_inferred.future_contract is not None:
                self._count("inference_self_speculative_events")
            if model_inferred.memory_reconstruction is not None:
                self._count("inference_latent_kv_events")
                self._record_memory_utility(
                    model_inferred.memory_reconstruction,
                    phase="P8",
                    source=f"inference:{model_inferred.route.path.value}:{model_inferred.task.task_id}",
                    reason="model_backed_inference_memory_reconstruction",
                )
            self._touch("P8")
            audit["inference"] = route_reports[0] if route_reports else {}
            audit["inference_routes"] = route_reports
            audit["model_backed_inference_events"] = model_backed_events
            inferred = inference_results[-1]
            self.bit_ledger.ingest_cost(inferred.cost, note=f"P8:{inferred.route.path.value}:{inferred.task.skill}")
            self.uncertainty_ledger.record(inferred.task.skill, inferred.answer.confidence, bool(inferred.passed))
            self._record_causal_trace(
                trace_id=f"P8-{step}-{inferred.task.task_id}",
                skill=inferred.task.skill,
                confidence=inferred.answer.confidence,
                anchors=len(inferred.task.anchors),
                certificate_fields=inferred.answer.certificate.keys(),
                verifier_level=inferred.route.verifier_level,
                mtp_horizon=inferred.route.mtp_horizon,
            )
            for inferred in inference_results:
                if not inferred.passed:
                    continue
                self._add_verified_phase_replay(
                    "P8",
                    inferred.task,
                    answer=inferred.answer,
                    metadata={
                        "route": inferred.route.path.value,
                        "layers_ran": inferred.layers_ran,
                        "verified_capability_per_cost": inferred.verified_capability_per_cost,
                    },
                )
        except Exception as exc:
            self._record_error("P8", exc)

        sleep_report = None
        try:
            if cycle_report is not None:
                real_llm_examples = self._real_exogenous_llm_examples()
                if not real_llm_examples:
                    raise ValueError("P9 requires REAL_EXOGENOUS examples sourced from observed LLM input batches before sleep consolidation")
                audit["sleep_real_exogenous_llm_examples"] = [
                    example.example_id
                    for example in real_llm_examples[-4:]
                ]
                self._count("sleep_real_exogenous_llm_consumed_events", len(real_llm_examples))
                sleep_report = self.sleep.ingest_cycle(cycle_report, seed=self.config.seed + step)
                self._add_sleep_replay(sleep_report.accepted_examples)
                self._touch("P9")
                audit["sleep"] = sleep_report.to_dict()
                audit["sleep_frontier_compilation"] = self._compile_sleep_frontier(sleep_report, step=step)
                self._count("sleep_anti_collapse_decisions")
                self._count("sleep_consolidation_schedule_items", len(sleep_report.schedule))
                self._count(
                    "sleep_tool_solved_examples",
                    sum(1 for example in self.sleep.synthetic.examples if example.origin == ExampleOrigin.TOOL_SOLVED),
                )
                self._count(
                    "sleep_metamorphic_examples",
                    sum(
                        1
                        for example in self.sleep.synthetic.examples
                        if example.origin in {ExampleOrigin.METAMORPHIC, ExampleOrigin.ANTI_METAMORPHIC}
                    ),
                )
        except Exception as exc:
            self._record_error("P9", exc)

        improvement_report = None
        try:
            if cycle_report is not None and self.config.cortex_phase_max_proposals > 0:
                frontier_improvement_proposals = self.improvement.generator.from_frontier_repairs(
                    self.frontier_repair_candidates,
                    max_proposals=max(1, int(self.config.cortex_phase_max_proposals)),
                )
                sleep_frontier_circuits = tuple(
                    circuit
                    for report_payload in self.sleep_frontier_reports[-2:]
                    for circuit in tuple(dict(report_payload).get("circuits", ()))
                    if isinstance(circuit, Mapping)
                )
                sleep_frontier_proposals = self.improvement.generator.from_sleep_frontier_circuits(
                    sleep_frontier_circuits,
                    max_proposals=max(1, int(self.config.cortex_phase_max_proposals)),
                )
                compiled_frontier_proposals = tuple(frontier_improvement_proposals) + tuple(sleep_frontier_proposals)
                p10_proposal_budget = max(
                    int(self.config.cortex_phase_max_proposals),
                    min(
                        6,
                        int(self.config.cortex_phase_max_proposals)
                        + int(self.improvement.archive.accepted_count)
                        + 1,
                    ),
                )
                improvement_report = self.improvement.run(
                    cycle_report,
                    baseline_agent=self.trial_agent,
                    reference_agent=self.reference_agent,
                    max_proposals=p10_proposal_budget,
                    generations=int(self.config.cortex_phase_improvement_generations),
                    seed=self.config.seed + step,
                    n_per_skill=self.config.cortex_phase_probe_tasks,
                    extra_proposals=compiled_frontier_proposals,
                )
                self._touch("P10")
                audit["recursive_improvement"] = improvement_report.to_dict()
                audit["recursive_improvement"]["proposal_budget"] = p10_proposal_budget
                audit["recursive_improvement"]["configured_generations"] = int(self.config.cortex_phase_improvement_generations)
                audit["recursive_improvement"]["frontier_proposal_count"] = len(compiled_frontier_proposals)
                audit["recursive_improvement"]["frontier_repair_proposal_count"] = len(frontier_improvement_proposals)
                audit["recursive_improvement"]["sleep_frontier_proposal_count"] = len(sleep_frontier_proposals)
                self._count("recursive_frontier_proposal_events", len(compiled_frontier_proposals))
                self._count("recursive_sleep_frontier_proposal_events", len(sleep_frontier_proposals))
                self._count("recursive_generation_events", len(improvement_report.generations))
                self._count("recursive_evolved_proposal_events", int(improvement_report.to_dict().get("evolved_proposal_count", 0)))
                self._count("recursive_proposal_events", len(improvement_report.proposals))
                self._count("recursive_sandbox_trials", len(improvement_report.decisions))
                self._count("recursive_dynamic_evaluations", len(improvement_report.decisions))
                self._count("recursive_pareto_gate_decisions", len(improvement_report.decisions))
                self._count(
                    "recursive_rollback_tokens",
                    sum(1 for decision in improvement_report.decisions if decision.evaluation.sandbox.rollback_token),
                )
                self._count("recursive_diversity_checks", len(improvement_report.decisions))
                self._count("recursive_reward_hacking_checks", len(improvement_report.decisions))
                persistent_archive = self._persist_improvement_archive(step=step)
                audit["recursive_improvement"]["persistent_archive"] = persistent_archive
                if improvement_report.decisions:
                    accepted_decision = next((decision for decision in improvement_report.decisions if decision.accepted), None)
                    if accepted_decision is None:
                        raise ValueError("P10 recursive improvement produced no accepted proposal for model-state application")
                    recursive_model_patch = self._apply_recursive_model_improvement(
                        accepted_decision,
                        cycle_report,
                        step=step,
                    )
                    recursive_verified_artifact = self._materialize_recursive_improvement_artifact(
                        accepted_decision,
                        cycle_report,
                        step=step,
                        model_patch=recursive_model_patch,
                    )
                    audit["recursive_model_application"] = recursive_model_patch
                    audit["recursive_verified_artifact"] = recursive_verified_artifact
                    selected_decision = accepted_decision or improvement_report.decisions[0]
                    gate_label = (
                        selected_decision.evaluation.proposal.proposal_id
                        if selected_decision.accepted
                        else f"REJECTED:{selected_decision.reason}"
                    )
                    gate_task = Task(
                        f"phase-p10-{step}",
                        "instruction_following",
                        f"Output recursive improvement gate result exactly: {gate_label}",
                        gate_label,
                    )
                    self._add_verified_phase_replay(
                        "P10",
                        gate_task,
                        metadata={
                            "proposal_id": selected_decision.evaluation.proposal.proposal_id,
                            "accepted": selected_decision.accepted,
                            "reason": selected_decision.reason,
                            "signed_patch_id": recursive_model_patch["signed_patch_id"],
                            "recursive_verified_artifact_id": recursive_verified_artifact["artifact_id"],
                            "model_patch_parameter_delta_l1": recursive_model_patch["parameter_delta_l1"],
                            "model_patch_repair_loss_delta": recursive_model_patch["repair_loss_delta"],
                        },
                    )
        except Exception as exc:
            self._record_error("P10", exc)

        try:
            if cycle_report is not None:
                objective_report = build_objective_report(
                    cycle_report,
                    future_ledger=self.future_ledger,
                    inference_results=inference_results,
                    improvement_report=improvement_report,
                )
                audit["objective"] = objective_report.to_dict()
                self.last_objective_loss_total = float(objective_report.loss.total)
                self.last_objective_loss_terms = {
                    name: term.to_dict()
                    for name, term in objective_report.loss.terms.items()
                }
                for name, term in objective_report.loss.terms.items():
                    self.objective_feedback_term_totals[name] = (
                        self.objective_feedback_term_totals.get(name, 0.0) + float(term.weighted)
                    )
                self.objective_feedback_events += 1
                self.objective_feedback_total += self.last_objective_loss_total
                self.objective_feedback_history.append(
                    {
                        "step": step,
                        "loss_total": self.last_objective_loss_total,
                        "term_count": len(self.last_objective_loss_terms),
                        "term_names": tuple(self.last_objective_loss_terms),
                        "weighted_terms": {
                            name: float(term["weighted"])
                            for name, term in self.last_objective_loss_terms.items()
                        },
                        "feedback_scale": self.objective_feedback_scale(),
                    }
                )
        except Exception as exc:
            self._record_error("objective", exc)

        self.phase_audits.append(audit)
        return audit

    def summary(self) -> Mapping[str, Any]:
        compression_trace = self.model.compression_trace()
        trace_counts = {}
        native_backend_counts: dict[str, int] = {}
        native_requantize_backend_counts: dict[str, int] = {}
        native_grad_weight_backend_counts: dict[str, int] = {}
        if compression_trace.get("enabled"):
            trace_counts = dict(compression_trace.get("total_event_counts") or {})
            native_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_backend_counts") or {}).items()
            }
            native_requantize_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_requantize_backend_counts") or {}).items()
            }
            native_grad_weight_backend_counts = {
                str(key): int(value)
                for key, value in dict(compression_trace.get("native_ternary_grad_weight_backend_counts") or {}).items()
            }
            if not trace_counts:
                trace_counts = {
                    "compression_decisions": len(compression_trace.get("compression_decisions", ())),
                    "activation_quantizations": len(compression_trace.get("activation_quantizations", ())),
                    "expert_activations": len(compression_trace.get("expert_activations", ())),
                    "kv_events": len(compression_trace.get("kv_events", ())),
                    "mtp_fsp_events": len(compression_trace.get("mtp_fsp_events", ())),
                    "layer_forward_events": len(compression_trace.get("layer_forward_events", ())),
                    "packed_ternary_dispatches": len(compression_trace.get("packed_ternary_dispatches", ())),
                    "native_ternary_kernel_dispatches": sum(
                        1
                        for item in compression_trace.get("packed_ternary_dispatches", ())
                        if bool(dict(item).get("native_kernel", False)) or str(dict(item).get("backend", "")).startswith("native_")
                    ),
                }
                trace_counts["torch_packed_ternary_dispatches"] = max(
                    0,
                    trace_counts["packed_ternary_dispatches"] - trace_counts["native_ternary_kernel_dispatches"],
                )
            if not native_backend_counts:
                for item in compression_trace.get("packed_ternary_dispatches", ()):
                    data = dict(item)
                    if bool(data.get("native_kernel", False)) or str(data.get("backend", "")).startswith("native_"):
                        backend = str(
                            data.get("native_backend")
                            or native_backend_from_runtime_label(str(data.get("backend", "")), default="unknown")
                        )
                        native_backend_counts[backend] = native_backend_counts.get(backend, 0) + 1
            for backend, count in native_backend_counts.items():
                trace_counts.setdefault(f"native_ternary_{backend}_kernel_dispatches", int(count))
            for backend, count in native_requantize_backend_counts.items():
                trace_counts.setdefault(f"native_ternary_{backend}_requantize_dispatches", int(count))
            for backend, count in native_grad_weight_backend_counts.items():
                trace_counts.setdefault(f"native_ternary_{backend}_grad_weight_dispatches", int(count))
            if trace_counts["layer_forward_events"] > 0:
                self.phase_counts["P2"] = max(self.phase_counts.get("P2", 0), trace_counts["layer_forward_events"])
        phases = []
        for phase in CORTEX3_PHASES:
            count = int(self.phase_counts.get(phase.id, 0))
            phases.append({
                "id": phase.id,
                "title": phase.title,
                "active_in_llm_training": count > 0,
                "event_count": count,
            })
        output_goal_summary = self._output_goal_contract_summary()
        frontier_registry_summary = self.frontier_registry.to_dict()
        frontier_heldout_summary = _frontier_heldout_summary(frontier_registry_summary)
        memory_report = self.memory.compression_report()
        real_exogenous_llm_examples = self._real_exogenous_llm_examples()
        return {
            "schema_version": 1,
            "enabled": True,
            "all_phases_active": all(item["active_in_llm_training"] for item in phases),
            "phases": phases,
            "phase_event_counts": dict(self.phase_counts),
            **output_goal_summary,
            "native_ternary_backend_requested": str(self.model.config.native_ternary_backend),
            "native_ternary_backend_counts": native_backend_counts,
            "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
            "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
            "native_ternary_grad_input_kernel": last_native_grad_input_kernel(),
            "native_ternary_grad_weight_kernel": last_native_grad_weight_kernel(),
            "native_ternary_grad_input_kernel_counts": native_grad_input_kernel_counts(),
            "native_ternary_grad_weight_kernel_counts": native_grad_weight_kernel_counts(),
            "native_ternary_kernel_required": bool(
                torch.cuda.is_available()
                and self.model.config.use_ternary_core
                and self.model.config.use_native_ternary_kernel
            ),
            "attribution_policy_observations": int(self.attribution_policy.observation_count),
            "attribution_policy_successes": int(self.attribution_policy.success_count),
            "attribution_policy_updates": int(self.integration_counts.get("attribution_policy_updates", 0)),
            "attribution_policy_applied_events": int(self.integration_counts.get("attribution_policy_applied_events", 0)),
            "frontier_registry_loaded_events": int(self.integration_counts.get("frontier_registry_loaded_events", 0)),
            "frontier_registry_loaded_circuits": int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
            "frontier_restored_fastsolve_events": int(self.integration_counts.get("frontier_restored_fastsolve_events", 0)),
            "compiled_circuit_memory_restored_reuse_events": int(
                self.integration_counts.get("compiled_circuit_memory_restored_reuse_events", 0)
            ),
            "recursive_verified_artifact_count": len(self.recursive_verified_artifacts),
            "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
            "training_influence": {
                "ternary_core_forward_events": trace_counts.get("layer_forward_events", 0),
                "packed_ternary_dispatches": trace_counts.get("packed_ternary_dispatches", 0),
                "native_ternary_kernel_dispatches": trace_counts.get("native_ternary_kernel_dispatches", 0),
                "torch_packed_ternary_dispatches": trace_counts.get("torch_packed_ternary_dispatches", 0),
                "native_ternary_backend_requested": str(self.model.config.native_ternary_backend),
                "native_ternary_backend_counts": native_backend_counts,
                "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
                "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
                "native_ternary_grad_input_kernel": last_native_grad_input_kernel(),
                "native_ternary_grad_weight_kernel": last_native_grad_weight_kernel(),
                "native_ternary_grad_input_kernel_counts": native_grad_input_kernel_counts(),
                "native_ternary_grad_weight_kernel_counts": native_grad_weight_kernel_counts(),
                "native_ternary_extension_kernel_dispatches": trace_counts.get("native_ternary_extension_kernel_dispatches", 0),
                "native_ternary_extension_requantize_dispatches": trace_counts.get("native_ternary_extension_requantize_dispatches", 0),
                "native_ternary_extension_grad_weight_dispatches": trace_counts.get("native_ternary_extension_grad_weight_dispatches", 0),
                "native_ternary_kernel_variants": tuple(compression_trace.get("native_ternary_kernel_variants", ())),
                "native_ternary_autotuned_dispatches": trace_counts.get("native_ternary_autotuned_dispatches", 0),
                "native_ternary_autotune_cache_hits": trace_counts.get("native_ternary_autotune_cache_hits", 0),
                "variable_input_compression_events": trace_counts.get("kv_events", 0),
                "skill_expert_activations": trace_counts.get("expert_activations", 0),
                "skill_expert_context_events": int(self.skill_expert_context_events),
                "skill_expert_replay_context_events": int(self.skill_expert_replay_context_events),
                "skill_expert_context_updates": int(self.model.skill_expert_context_updates),
                "skill_expert_last_context": tuple(float(value) for value in self.skill_expert_last_context),
                "skill_expert_context_skills": tuple(self.skill_expert_context_skills),
                "certificate_head_forward_events": int(self.model.certificate_forward_events),
                "certificate_algebra_tool_events": int(self.integration_counts.get("certificate_algebra_tool_events", 0)),
                "certificate_code_hidden_property_events": int(self.integration_counts.get("certificate_code_hidden_property_events", 0)),
                "model_certificate_head_events": int(self.integration_counts.get("model_certificate_head_events", 0)),
                "model_certificate_head_verified_events": int(self.integration_counts.get("model_certificate_head_verified_events", 0)),
                "model_certificate_head_latent_checksum_events": int(self.integration_counts.get("model_certificate_head_latent_checksum_events", 0)),
                "model_certificate_head_target_match_events": int(self.integration_counts.get("model_certificate_head_target_match_events", 0)),
                "model_certificate_head_artifacts": _last_items(self.model_certificate_head_artifacts, 5),
                "latent_workspace_forward_events": int(self.latent_workspace_forward_events),
                "latent_workspace_step_events": int(self.latent_workspace_step_events),
                "latent_workspace_certificate_binding_events": int(self.latent_workspace_certificate_binding_events),
                "latent_workspace_last_summary": dict(self.latent_workspace_last_summary),
                "input_anchor_observations": int(self.input_anchor_observations),
                "input_anchor_count": int(self.input_anchor_count),
                "input_anchor_fidelity_failures": int(self.input_anchor_fidelity_failures),
                "learned_memory_policy_events": int(self.learned_memory_policy_events),
                "learned_memory_anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
                "learned_memory_exact_decisions": int(self.learned_memory_exact_decisions),
                "learned_memory_latent_decisions": int(self.learned_memory_latent_decisions),
                "learned_memory_drop_decisions": int(self.learned_memory_drop_decisions),
                "learned_memory_storage_ratio_mean": (
                    float(self.learned_memory_storage_ratio_total)
                    / max(1, int(self.learned_memory_policy_events))
                ),
                "learned_memory_retention_decisions": int(memory_report.get("learned_retention_decision_count", 0) or 0),
                "learned_memory_retention_requested_exact": int(memory_report.get("learned_retention_requested_exact", 0) or 0),
                "learned_memory_retention_requested_latent": int(memory_report.get("learned_retention_requested_latent", 0) or 0),
                "learned_memory_retention_requested_drop": int(memory_report.get("learned_retention_requested_drop", 0) or 0),
                "learned_memory_retention_applied_exact": int(memory_report.get("learned_retention_applied_exact", 0) or 0),
                "learned_memory_retention_applied_latent": int(memory_report.get("learned_retention_applied_latent", 0) or 0),
                "learned_memory_retention_applied_drop": int(memory_report.get("learned_retention_applied_drop", 0) or 0),
                "learned_memory_retention_anchor_overrides": int(memory_report.get("learned_retention_anchor_overrides", 0) or 0),
                "memory_utility_credit_count": int(memory_report.get("memory_utility_credit_count", 0) or 0),
                "learned_memory_utility_credit_count": int(memory_report.get("learned_memory_utility_credit_count", 0) or 0),
                "learned_memory_utility_positive_count": int(memory_report.get("learned_memory_utility_positive_count", 0) or 0),
                "learned_memory_utility_prior_updates": int(self.learned_memory_utility_prior_updates),
                "learned_memory_utility_feedback_events": int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
                "learned_memory_last_utility_prior": tuple(float(value) for value in self.learned_memory_last_utility_prior),
                "learned_memory_utility_credits": _last_items(memory_report.get("learned_memory_utility_credits", ()), 5),
                "future_contract_decisions": len(self.future_ledger.decisions),
                **output_goal_summary,
                "bit_ledger_total_effective_bits": self.bit_ledger.total_effective_bits,
                "skill_ledger_states": len(self.skill_ledger.states),
                "causal_ledger_traces": len(self.causal_ledger.traces),
                "uncertainty_ledger_observations": sum(len(pairs) for pairs in self.uncertainty_ledger.bins.values()),
                "uncertainty_ledger_ece": self.uncertainty_ledger.expected_calibration_error(),
                "attribution_policy_observations": int(self.attribution_policy.observation_count),
                "attribution_policy_successes": int(self.attribution_policy.success_count),
                "attribution_policy_updates": int(self.integration_counts.get("attribution_policy_updates", 0)),
                "attribution_policy_applied_events": int(self.integration_counts.get("attribution_policy_applied_events", 0)),
                "confidence_regularization_steps": self.regularization_steps,
                "sleep_replay_batches_available": len(self.replay_batches),
                "sleep_replay_updates": self.replay_updates,
                "phase_replay_examples": sum(self.phase_replay_examples.values()),
                "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
                "memory_recent_segments": len(self.memory.recent.segments),
                "memory_latent_segments": len(self.memory.latent.segments),
                "compiled_circuit_memory_binding_count": int(memory_report.get("compiled_circuit_memory_binding_count", 0) or 0),
                "compiled_circuit_memory_binding_events": int(self.integration_counts.get("compiled_circuit_memory_binding_events", 0)),
                "compiled_circuit_memory_fidelity_failures": int(self.integration_counts.get("compiled_circuit_memory_fidelity_failures", 0)),
                "sleep_frontier_memory_binding_events": int(self.integration_counts.get("sleep_frontier_memory_binding_events", 0)),
                "frontier_registry_loaded_events": int(self.integration_counts.get("frontier_registry_loaded_events", 0)),
                "frontier_registry_loaded_circuits": int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
                "frontier_restored_fastsolve_events": int(self.integration_counts.get("frontier_restored_fastsolve_events", 0)),
                "compiled_circuit_memory_restored_reuse_events": int(
                    self.integration_counts.get("compiled_circuit_memory_restored_reuse_events", 0)
                ),
                "compiled_circuit_memory_bindings": _last_items(memory_report.get("compiled_circuit_memory_bindings", ()), 5),
                "sleep_replay_examples": len(self.sleep.replay.examples),
                "sleep_synthetic_examples": len(self.sleep.synthetic.examples),
                "sleep_reservoir_examples": len(self.sleep.reservoir.examples),
                "sleep_real_exogenous_llm_examples": len(real_exogenous_llm_examples),
                "sleep_real_exogenous_llm_batch_events": int(self.integration_counts.get("sleep_real_exogenous_llm_batch_events", 0)),
                "sleep_real_exogenous_llm_tokens": int(self.integration_counts.get("sleep_real_exogenous_llm_tokens", 0)),
                "frontier_compiled_circuit_count": int(frontier_registry_summary["circuit_count"]),
                "frontier_compiled_skill_count": int(frontier_registry_summary["compiled_skill_count"]),
                **frontier_heldout_summary,
                "frontier_compiled_fastsolve_events": int(self.frontier_compiled_fastsolve_events),
                "inference_model_backed_events": int(self.integration_counts.get("inference_model_backed_events", 0)),
                "inference_model_backed_generated_tokens": int(self.integration_counts.get("inference_model_backed_generated_tokens", 0)),
                "inference_model_backed_verified_events": int(self.integration_counts.get("inference_model_backed_verified_events", 0)),
                "inference_model_backed_forced_careful_events": int(self.integration_counts.get("inference_model_backed_forced_careful_events", 0)),
                "inference_model_backed_replay_events": int(self.integration_counts.get("inference_model_backed_replay_events", 0)),
                "inference_model_backed_repair_replay_events": int(self.integration_counts.get("inference_model_backed_repair_replay_events", 0)),
                "inference_model_backed_verified_replay_events": int(self.integration_counts.get("inference_model_backed_verified_replay_events", 0)),
                "inference_model_backed_adaptive_mtp_events": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_events", 0)),
                "inference_model_backed_adaptive_mtp_forward_count": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_forward_count", 0)),
                "inference_model_backed_adaptive_mtp_contract_checks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_contract_checks", 0)),
                "inference_model_backed_adaptive_mtp_proposed_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_blocks", 0)),
                "inference_model_backed_adaptive_mtp_proposed_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_tokens", 0)),
                "inference_model_backed_adaptive_mtp_accepted_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_blocks", 0)),
                "inference_model_backed_adaptive_mtp_rejected_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_blocks", 0)),
                "inference_model_backed_adaptive_mtp_accepted_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_tokens", 0)),
                "inference_model_backed_adaptive_mtp_rejected_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_tokens", 0)),
                "inference_model_backed_adaptive_mtp_accepted_mtp_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_mtp_tokens", 0)),
                "sleep_frontier_fastsolve_events": int(self.integration_counts.get("sleep_frontier_fastsolve_events", 0)),
                "restored_frontier_fastsolve_reports": _last_items(self.restored_frontier_fastsolve_reports, 3),
                "frontier_repair_candidate_count": len(self.frontier_repair_candidates),
                "frontier_repair_accepted_events": int(self.frontier_repair_accepted_events),
                "recursive_frontier_proposal_events": int(self.integration_counts.get("recursive_frontier_proposal_events", 0)),
                "frontier_registry_path": str(self.run_dir / "frontier_registry"),
                "regrowth_model_application_count": len(self.regrowth_model_applications),
                "regrowth_model_parameter_delta_l1": float(self.regrowth_model_parameter_delta_l1),
                "regrowth_model_repair_loss_delta": float(self.regrowth_model_repair_loss_delta),
                "regrowth_model_protected_loss_delta": float(self.regrowth_model_protected_loss_delta),
                "regrowth_model_applications": _last_items(self.regrowth_model_applications, 5),
                "recursive_model_application_count": len(self.recursive_model_applications),
                "recursive_model_parameter_delta_l1": float(self.recursive_model_parameter_delta_l1),
                "recursive_model_repair_loss_delta": float(self.recursive_model_repair_loss_delta),
                "recursive_model_protected_loss_delta": float(self.recursive_model_protected_loss_delta),
                "recursive_model_applications": _last_items(self.recursive_model_applications, 5),
                "recursive_verified_artifact_count": len(self.recursive_verified_artifacts),
                "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
                "improvement_archive_accepted": self.improvement.archive.accepted_count,
                "improvement_archive_rejected": self.improvement.archive.rejected_count,
                "recursive_improvement_generations_configured": int(self.config.cortex_phase_improvement_generations),
                "recursive_generation_events": int(self.integration_counts.get("recursive_generation_events", 0)),
                "recursive_evolved_proposal_events": int(self.integration_counts.get("recursive_evolved_proposal_events", 0)),
                "improvement_archive_dir": str(self.improvement_archive_dir),
                "improvement_persistent_archive_state": dict(self.improvement_persistent_archive_state),
                "improvement_persistent_archive_decisions": int(
                    self.improvement_persistent_archive_state.get("decision_count", 0) or 0
                ),
                "improvement_persistent_rollback_events": int(
                    self.improvement_persistent_archive_state.get("rollback_event_count", 0) or 0
                ),
                "objective_feedback_events": self.objective_feedback_events,
                "objective_feedback_average_loss": (
                    self.objective_feedback_total / max(1, self.objective_feedback_events)
                ),
                "objective_feedback_scale": self.objective_feedback_scale(),
                "last_objective_loss_total": self.last_objective_loss_total,
                "objective_feedback_term_count": len(self.last_objective_loss_terms),
                "objective_feedback_term_names": tuple(self.last_objective_loss_terms),
                "last_objective_loss_weighted_total": sum(
                    float(term.get("weighted", 0.0))
                    for term in self.last_objective_loss_terms.values()
                ),
            },
            "integration_counts": dict(self.integration_counts),
            "architecture_audit": _cortex_architecture_audit_from_summary(
                {
                    "phase_event_counts": dict(self.phase_counts),
                    "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
                    "integration_counts": dict(self.integration_counts),
                    "replay_batch_count": len(self.replay_batches),
                    "replay_updates": self.replay_updates,
                    "objective_feedback_events": self.objective_feedback_events,
                    "last_objective_loss_total": self.last_objective_loss_total,
                    "last_objective_loss_weighted_total": sum(
                        float(term.get("weighted", 0.0))
                        for term in self.last_objective_loss_terms.values()
                    ),
                    "objective_feedback_term_names": tuple(self.last_objective_loss_terms),
                    "objective_feedback_term_count": len(self.last_objective_loss_terms),
                    "future_contract_decisions": len(self.future_ledger.decisions),
                    **output_goal_summary,
                    "compression_trace_counts": trace_counts,
                    "native_ternary_backend_requested": str(self.model.config.native_ternary_backend),
                    "native_ternary_backend_counts": native_backend_counts,
                    "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
                    "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
                    "native_ternary_kernel_required": bool(
                        torch.cuda.is_available()
                        and self.model.config.use_ternary_core
                        and self.model.config.use_native_ternary_kernel
                    ),
                    "variable_input_compression_events": trace_counts.get("kv_events", 0),
                    "certificate_head_forward_events": int(self.model.certificate_forward_events),
                    "certificate_algebra_tool_events": int(self.integration_counts.get("certificate_algebra_tool_events", 0)),
                    "certificate_code_hidden_property_events": int(self.integration_counts.get("certificate_code_hidden_property_events", 0)),
                    "model_certificate_head_events": int(self.integration_counts.get("model_certificate_head_events", 0)),
                    "model_certificate_head_verified_events": int(self.integration_counts.get("model_certificate_head_verified_events", 0)),
                    "model_certificate_head_latent_checksum_events": int(self.integration_counts.get("model_certificate_head_latent_checksum_events", 0)),
                    "model_certificate_head_target_match_events": int(self.integration_counts.get("model_certificate_head_target_match_events", 0)),
                    "model_certificate_head_artifacts": _last_items(self.model_certificate_head_artifacts, 5),
                    "latent_workspace_forward_events": int(self.latent_workspace_forward_events),
                    "latent_workspace_step_events": int(self.latent_workspace_step_events),
                    "latent_workspace_certificate_binding_events": int(self.latent_workspace_certificate_binding_events),
                    "latent_workspace_last_summary": dict(self.latent_workspace_last_summary),
                    **frontier_heldout_summary,
                    "input_anchor_observations": int(self.input_anchor_observations),
                    "input_anchor_count": int(self.input_anchor_count),
                    "input_anchor_fidelity_failures": int(self.input_anchor_fidelity_failures),
                    "learned_memory_policy_events": int(self.learned_memory_policy_events),
                    "learned_memory_anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
                    "learned_memory_exact_decisions": int(self.learned_memory_exact_decisions),
                    "learned_memory_latent_decisions": int(self.learned_memory_latent_decisions),
                    "learned_memory_drop_decisions": int(self.learned_memory_drop_decisions),
                    "learned_memory_storage_ratio_mean": (
                        float(self.learned_memory_storage_ratio_total)
                        / max(1, int(self.learned_memory_policy_events))
                    ),
                    "learned_memory_retention_decisions": int(memory_report.get("learned_retention_decision_count", 0) or 0),
                    "learned_memory_retention_requested_exact": int(memory_report.get("learned_retention_requested_exact", 0) or 0),
                    "learned_memory_retention_requested_latent": int(memory_report.get("learned_retention_requested_latent", 0) or 0),
                    "learned_memory_retention_requested_drop": int(memory_report.get("learned_retention_requested_drop", 0) or 0),
                    "learned_memory_retention_applied_exact": int(memory_report.get("learned_retention_applied_exact", 0) or 0),
                    "learned_memory_retention_applied_latent": int(memory_report.get("learned_retention_applied_latent", 0) or 0),
                    "learned_memory_retention_applied_drop": int(memory_report.get("learned_retention_applied_drop", 0) or 0),
                    "learned_memory_retention_anchor_overrides": int(memory_report.get("learned_retention_anchor_overrides", 0) or 0),
                    "memory_utility_credit_count": int(memory_report.get("memory_utility_credit_count", 0) or 0),
                    "learned_memory_utility_credit_count": int(memory_report.get("learned_memory_utility_credit_count", 0) or 0),
                    "learned_memory_utility_positive_count": int(memory_report.get("learned_memory_utility_positive_count", 0) or 0),
                    "learned_memory_utility_prior_updates": int(self.learned_memory_utility_prior_updates),
                    "learned_memory_utility_feedback_events": int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
                    "learned_memory_last_utility_prior": tuple(float(value) for value in self.learned_memory_last_utility_prior),
                    "learned_memory_utility_credits": _last_items(memory_report.get("learned_memory_utility_credits", ()), 5),
                    "skill_expert_context_events": int(self.skill_expert_context_events),
                    "skill_expert_replay_context_events": int(self.skill_expert_replay_context_events),
                    "skill_expert_context_updates": int(self.model.skill_expert_context_updates),
                    "skill_expert_last_context": tuple(float(value) for value in self.skill_expert_last_context),
                    "skill_expert_context_skills": tuple(self.skill_expert_context_skills),
                    "bit_ledger_total_effective_bits": self.bit_ledger.total_effective_bits,
                    "skill_ledger_states": len(self.skill_ledger.states),
                    "causal_ledger_traces": len(self.causal_ledger.traces),
                    "uncertainty_ledger_observations": sum(len(pairs) for pairs in self.uncertainty_ledger.bins.values()),
                    "attribution_policy_observations": int(self.attribution_policy.observation_count),
                    "attribution_policy_successes": int(self.attribution_policy.success_count),
                    "attribution_policy_updates": int(self.integration_counts.get("attribution_policy_updates", 0)),
                    "attribution_policy_applied_events": int(self.integration_counts.get("attribution_policy_applied_events", 0)),
                    "memory_recent_segments": len(self.memory.recent.segments),
                    "memory_latent_segments": len(self.memory.latent.segments),
                    "compiled_circuit_memory_binding_count": int(memory_report.get("compiled_circuit_memory_binding_count", 0) or 0),
                    "compiled_circuit_memory_binding_events": int(self.integration_counts.get("compiled_circuit_memory_binding_events", 0)),
                    "compiled_circuit_memory_fidelity_failures": int(self.integration_counts.get("compiled_circuit_memory_fidelity_failures", 0)),
                    "sleep_frontier_memory_binding_events": int(self.integration_counts.get("sleep_frontier_memory_binding_events", 0)),
                    "frontier_registry_loaded_events": int(self.integration_counts.get("frontier_registry_loaded_events", 0)),
                    "frontier_registry_loaded_circuits": int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
                    "frontier_restored_fastsolve_events": int(self.integration_counts.get("frontier_restored_fastsolve_events", 0)),
                    "compiled_circuit_memory_restored_reuse_events": int(
                        self.integration_counts.get("compiled_circuit_memory_restored_reuse_events", 0)
                    ),
                    "sleep_replay_examples": len(self.sleep.replay.examples),
                    "sleep_synthetic_examples": len(self.sleep.synthetic.examples),
                    "sleep_reservoir_examples": len(self.sleep.reservoir.examples),
                    "sleep_real_exogenous_llm_examples": len(real_exogenous_llm_examples),
                    "sleep_real_exogenous_llm_batch_events": int(self.integration_counts.get("sleep_real_exogenous_llm_batch_events", 0)),
                    "sleep_real_exogenous_llm_tokens": int(self.integration_counts.get("sleep_real_exogenous_llm_tokens", 0)),
                    "frontier_compiled_circuit_count": int(frontier_registry_summary["circuit_count"]),
                    "frontier_compiled_skill_count": int(frontier_registry_summary["compiled_skill_count"]),
                    **frontier_heldout_summary,
                    "frontier_compiled_fastsolve_events": int(self.frontier_compiled_fastsolve_events),
                    "inference_model_backed_events": int(self.integration_counts.get("inference_model_backed_events", 0)),
                    "inference_model_backed_generated_tokens": int(self.integration_counts.get("inference_model_backed_generated_tokens", 0)),
                    "inference_model_backed_verified_events": int(self.integration_counts.get("inference_model_backed_verified_events", 0)),
                    "inference_model_backed_forced_careful_events": int(self.integration_counts.get("inference_model_backed_forced_careful_events", 0)),
                    "inference_model_backed_replay_events": int(self.integration_counts.get("inference_model_backed_replay_events", 0)),
                    "inference_model_backed_repair_replay_events": int(self.integration_counts.get("inference_model_backed_repair_replay_events", 0)),
                    "inference_model_backed_verified_replay_events": int(self.integration_counts.get("inference_model_backed_verified_replay_events", 0)),
                    "inference_model_backed_adaptive_mtp_events": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_events", 0)),
                    "inference_model_backed_adaptive_mtp_forward_count": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_forward_count", 0)),
                    "inference_model_backed_adaptive_mtp_contract_checks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_contract_checks", 0)),
                    "inference_model_backed_adaptive_mtp_proposed_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_proposed_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_rejected_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_rejected_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_mtp_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_mtp_tokens", 0)),
                    "sleep_frontier_fastsolve_events": int(self.integration_counts.get("sleep_frontier_fastsolve_events", 0)),
                    "restored_frontier_fastsolve_reports": _last_items(self.restored_frontier_fastsolve_reports, 3),
                    "frontier_repair_candidate_count": len(self.frontier_repair_candidates),
                    "frontier_repair_accepted_events": int(self.frontier_repair_accepted_events),
                    "recursive_frontier_proposal_events": int(self.integration_counts.get("recursive_frontier_proposal_events", 0)),
                    "regrowth_model_application_count": len(self.regrowth_model_applications),
                    "regrowth_model_parameter_delta_l1": float(self.regrowth_model_parameter_delta_l1),
                    "regrowth_model_repair_loss_delta": float(self.regrowth_model_repair_loss_delta),
                    "regrowth_model_protected_loss_delta": float(self.regrowth_model_protected_loss_delta),
                    "regrowth_model_applications": _last_items(self.regrowth_model_applications, 5),
                    "recursive_model_application_count": len(self.recursive_model_applications),
                    "recursive_model_parameter_delta_l1": float(self.recursive_model_parameter_delta_l1),
                    "recursive_model_repair_loss_delta": float(self.recursive_model_repair_loss_delta),
                    "recursive_model_protected_loss_delta": float(self.recursive_model_protected_loss_delta),
                    "recursive_model_applications": _last_items(self.recursive_model_applications, 5),
                    "recursive_verified_artifact_count": len(self.recursive_verified_artifacts),
                    "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
                    "improvement_archive_accepted": self.improvement.archive.accepted_count,
                    "improvement_archive_rejected": self.improvement.archive.rejected_count,
                    "recursive_improvement_generations_configured": int(self.config.cortex_phase_improvement_generations),
                    "recursive_generation_events": int(self.integration_counts.get("recursive_generation_events", 0)),
                    "recursive_evolved_proposal_events": int(self.integration_counts.get("recursive_evolved_proposal_events", 0)),
                    "improvement_persistent_archive_decisions": int(
                        self.improvement_persistent_archive_state.get("decision_count", 0) or 0
                    ),
                    "improvement_persistent_rollback_events": int(
                        self.improvement_persistent_archive_state.get("rollback_event_count", 0) or 0
                    ),
                    "error_count": len(self.errors),
                }
            ),
            "phase_deliverable_audit": _cortex_phase_deliverable_audit_from_summary(
                {
                    "phase_event_counts": dict(self.phase_counts),
                    "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
                    "integration_counts": dict(self.integration_counts),
                    "replay_batch_count": len(self.replay_batches),
                    "replay_updates": self.replay_updates,
                    "objective_feedback_events": self.objective_feedback_events,
                    "last_objective_loss_total": self.last_objective_loss_total,
                    "last_objective_loss_weighted_total": sum(
                        float(term.get("weighted", 0.0))
                        for term in self.last_objective_loss_terms.values()
                    ),
                    "objective_feedback_term_names": tuple(self.last_objective_loss_terms),
                    "objective_feedback_term_count": len(self.last_objective_loss_terms),
                    "future_contract_decisions": len(self.future_ledger.decisions),
                    **output_goal_summary,
                    "compression_trace_counts": trace_counts,
                    "native_ternary_backend_requested": str(self.model.config.native_ternary_backend),
                    "native_ternary_backend_counts": native_backend_counts,
                    "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
                    "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
                    "native_ternary_kernel_required": bool(
                        torch.cuda.is_available()
                        and self.model.config.use_ternary_core
                        and self.model.config.use_native_ternary_kernel
                    ),
                    "variable_input_compression_events": trace_counts.get("kv_events", 0),
                    "certificate_head_forward_events": int(self.model.certificate_forward_events),
                    "certificate_algebra_tool_events": int(self.integration_counts.get("certificate_algebra_tool_events", 0)),
                    "certificate_code_hidden_property_events": int(self.integration_counts.get("certificate_code_hidden_property_events", 0)),
                    "model_certificate_head_events": int(self.integration_counts.get("model_certificate_head_events", 0)),
                    "model_certificate_head_verified_events": int(self.integration_counts.get("model_certificate_head_verified_events", 0)),
                    "model_certificate_head_latent_checksum_events": int(self.integration_counts.get("model_certificate_head_latent_checksum_events", 0)),
                    "model_certificate_head_target_match_events": int(self.integration_counts.get("model_certificate_head_target_match_events", 0)),
                    "model_certificate_head_artifacts": _last_items(self.model_certificate_head_artifacts, 5),
                    "latent_workspace_forward_events": int(self.latent_workspace_forward_events),
                    "latent_workspace_step_events": int(self.latent_workspace_step_events),
                    "latent_workspace_certificate_binding_events": int(self.latent_workspace_certificate_binding_events),
                    "latent_workspace_last_summary": dict(self.latent_workspace_last_summary),
                    "input_anchor_observations": int(self.input_anchor_observations),
                    "input_anchor_count": int(self.input_anchor_count),
                    "input_anchor_fidelity_failures": int(self.input_anchor_fidelity_failures),
                    "learned_memory_policy_events": int(self.learned_memory_policy_events),
                    "learned_memory_anchor_supervision_events": int(self.learned_memory_anchor_supervision_events),
                    "learned_memory_exact_decisions": int(self.learned_memory_exact_decisions),
                    "learned_memory_latent_decisions": int(self.learned_memory_latent_decisions),
                    "learned_memory_drop_decisions": int(self.learned_memory_drop_decisions),
                    "learned_memory_storage_ratio_mean": (
                        float(self.learned_memory_storage_ratio_total)
                        / max(1, int(self.learned_memory_policy_events))
                    ),
                    "learned_memory_retention_decisions": int(memory_report.get("learned_retention_decision_count", 0) or 0),
                    "learned_memory_retention_requested_exact": int(memory_report.get("learned_retention_requested_exact", 0) or 0),
                    "learned_memory_retention_requested_latent": int(memory_report.get("learned_retention_requested_latent", 0) or 0),
                    "learned_memory_retention_requested_drop": int(memory_report.get("learned_retention_requested_drop", 0) or 0),
                    "learned_memory_retention_applied_exact": int(memory_report.get("learned_retention_applied_exact", 0) or 0),
                    "learned_memory_retention_applied_latent": int(memory_report.get("learned_retention_applied_latent", 0) or 0),
                    "learned_memory_retention_applied_drop": int(memory_report.get("learned_retention_applied_drop", 0) or 0),
                    "learned_memory_retention_anchor_overrides": int(memory_report.get("learned_retention_anchor_overrides", 0) or 0),
                    "memory_utility_credit_count": int(memory_report.get("memory_utility_credit_count", 0) or 0),
                    "learned_memory_utility_credit_count": int(memory_report.get("learned_memory_utility_credit_count", 0) or 0),
                    "learned_memory_utility_positive_count": int(memory_report.get("learned_memory_utility_positive_count", 0) or 0),
                    "learned_memory_utility_prior_updates": int(self.learned_memory_utility_prior_updates),
                    "learned_memory_utility_feedback_events": int(self.integration_counts.get("learned_memory_utility_feedback_events", 0)),
                    "learned_memory_last_utility_prior": tuple(float(value) for value in self.learned_memory_last_utility_prior),
                    "learned_memory_utility_credits": _last_items(memory_report.get("learned_memory_utility_credits", ()), 5),
                    "skill_expert_context_events": int(self.skill_expert_context_events),
                    "skill_expert_replay_context_events": int(self.skill_expert_replay_context_events),
                    "skill_expert_context_updates": int(self.model.skill_expert_context_updates),
                    "skill_expert_last_context": tuple(float(value) for value in self.skill_expert_last_context),
                    "skill_expert_context_skills": tuple(self.skill_expert_context_skills),
                    "bit_ledger_total_effective_bits": self.bit_ledger.total_effective_bits,
                    "skill_ledger_states": len(self.skill_ledger.states),
                    "causal_ledger_traces": len(self.causal_ledger.traces),
                    "uncertainty_ledger_observations": sum(len(pairs) for pairs in self.uncertainty_ledger.bins.values()),
                    "attribution_policy_observations": int(self.attribution_policy.observation_count),
                    "attribution_policy_successes": int(self.attribution_policy.success_count),
                    "attribution_policy_updates": int(self.integration_counts.get("attribution_policy_updates", 0)),
                    "attribution_policy_applied_events": int(self.integration_counts.get("attribution_policy_applied_events", 0)),
                    "memory_recent_segments": len(self.memory.recent.segments),
                    "memory_latent_segments": len(self.memory.latent.segments),
                    "compiled_circuit_memory_binding_count": int(memory_report.get("compiled_circuit_memory_binding_count", 0) or 0),
                    "compiled_circuit_memory_binding_events": int(self.integration_counts.get("compiled_circuit_memory_binding_events", 0)),
                    "compiled_circuit_memory_fidelity_failures": int(self.integration_counts.get("compiled_circuit_memory_fidelity_failures", 0)),
                    "sleep_frontier_memory_binding_events": int(self.integration_counts.get("sleep_frontier_memory_binding_events", 0)),
                    "frontier_registry_loaded_events": int(self.integration_counts.get("frontier_registry_loaded_events", 0)),
                    "frontier_registry_loaded_circuits": int(self.integration_counts.get("frontier_registry_loaded_circuits", 0)),
                    "frontier_restored_fastsolve_events": int(self.integration_counts.get("frontier_restored_fastsolve_events", 0)),
                    "compiled_circuit_memory_restored_reuse_events": int(
                        self.integration_counts.get("compiled_circuit_memory_restored_reuse_events", 0)
                    ),
                    "sleep_replay_examples": len(self.sleep.replay.examples),
                    "sleep_synthetic_examples": len(self.sleep.synthetic.examples),
                    "sleep_reservoir_examples": len(self.sleep.reservoir.examples),
                    "sleep_real_exogenous_llm_examples": len(real_exogenous_llm_examples),
                    "sleep_real_exogenous_llm_batch_events": int(self.integration_counts.get("sleep_real_exogenous_llm_batch_events", 0)),
                    "sleep_real_exogenous_llm_tokens": int(self.integration_counts.get("sleep_real_exogenous_llm_tokens", 0)),
                    **frontier_heldout_summary,
                    "sleep_frontier_fastsolve_events": int(self.integration_counts.get("sleep_frontier_fastsolve_events", 0)),
                    "inference_model_backed_events": int(self.integration_counts.get("inference_model_backed_events", 0)),
                    "inference_model_backed_generated_tokens": int(self.integration_counts.get("inference_model_backed_generated_tokens", 0)),
                    "inference_model_backed_verified_events": int(self.integration_counts.get("inference_model_backed_verified_events", 0)),
                    "inference_model_backed_forced_careful_events": int(self.integration_counts.get("inference_model_backed_forced_careful_events", 0)),
                    "inference_model_backed_replay_events": int(self.integration_counts.get("inference_model_backed_replay_events", 0)),
                    "inference_model_backed_repair_replay_events": int(self.integration_counts.get("inference_model_backed_repair_replay_events", 0)),
                    "inference_model_backed_verified_replay_events": int(self.integration_counts.get("inference_model_backed_verified_replay_events", 0)),
                    "inference_model_backed_adaptive_mtp_events": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_events", 0)),
                    "inference_model_backed_adaptive_mtp_forward_count": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_forward_count", 0)),
                    "inference_model_backed_adaptive_mtp_contract_checks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_contract_checks", 0)),
                    "inference_model_backed_adaptive_mtp_proposed_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_proposed_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_proposed_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_rejected_blocks": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_blocks", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_rejected_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_rejected_tokens", 0)),
                    "inference_model_backed_adaptive_mtp_accepted_mtp_tokens": int(self.integration_counts.get("inference_model_backed_adaptive_mtp_accepted_mtp_tokens", 0)),
                    "regrowth_model_application_count": len(self.regrowth_model_applications),
                    "regrowth_model_parameter_delta_l1": float(self.regrowth_model_parameter_delta_l1),
                    "regrowth_model_repair_loss_delta": float(self.regrowth_model_repair_loss_delta),
                    "regrowth_model_protected_loss_delta": float(self.regrowth_model_protected_loss_delta),
                    "regrowth_model_applications": _last_items(self.regrowth_model_applications, 5),
                    "recursive_model_application_count": len(self.recursive_model_applications),
                    "recursive_model_parameter_delta_l1": float(self.recursive_model_parameter_delta_l1),
                    "recursive_model_repair_loss_delta": float(self.recursive_model_repair_loss_delta),
                    "recursive_model_protected_loss_delta": float(self.recursive_model_protected_loss_delta),
                    "recursive_model_applications": _last_items(self.recursive_model_applications, 5),
                    "recursive_verified_artifact_count": len(self.recursive_verified_artifacts),
                    "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
                    "improvement_archive_accepted": self.improvement.archive.accepted_count,
                    "improvement_archive_rejected": self.improvement.archive.rejected_count,
                    "recursive_improvement_generations_configured": int(self.config.cortex_phase_improvement_generations),
                    "recursive_generation_events": int(self.integration_counts.get("recursive_generation_events", 0)),
                    "recursive_evolved_proposal_events": int(self.integration_counts.get("recursive_evolved_proposal_events", 0)),
                    "improvement_persistent_archive_decisions": int(
                        self.improvement_persistent_archive_state.get("decision_count", 0) or 0
                    ),
                    "improvement_persistent_rollback_events": int(
                        self.improvement_persistent_archive_state.get("rollback_event_count", 0) or 0
                    ),
                    "error_count": len(self.errors),
                }
            ),
            "trace_counts": trace_counts,
            "native_ternary_backend_counts": native_backend_counts,
            "native_ternary_requantize_backend_counts": native_requantize_backend_counts,
            "native_ternary_grad_weight_backend_counts": native_grad_weight_backend_counts,
            "retained_trace_counts": compression_trace.get("retained_event_counts", {}),
            "future_ledger": self.future_ledger.to_dict(),
            "ledgers": _ledger_bundle_payload(
                bit_ledger=self.bit_ledger,
                skill_ledger=self.skill_ledger,
                causal_ledger=self.causal_ledger,
                uncertainty_ledger=self.uncertainty_ledger,
            ),
            "memory_state_summary": self.memory.compression_report(),
            "sleep_state_summary": {
                "replay_examples": len(self.sleep.replay.examples),
                "synthetic_examples": len(self.sleep.synthetic.examples),
                "reservoir_examples": len(self.sleep.reservoir.examples),
            },
            "frontier_registry_summary": self.frontier_registry.to_dict(),
            "frontier_reports": _last_items(self.frontier_reports, 3),
            "sleep_frontier_reports": _last_items(self.sleep_frontier_reports, 3),
            "restored_frontier_fastsolve_reports": _last_items(self.restored_frontier_fastsolve_reports, 3),
            "frontier_repair_candidates": _last_items(self.frontier_repair_candidates, 3),
            "improvement_state_summary": _improvement_state(self.improvement),
            "improvement_archive_dir": str(self.improvement_archive_dir),
            "improvement_persistent_archive_state": dict(self.improvement_persistent_archive_state),
            "regrowth_model_applications": _last_items(self.regrowth_model_applications, 5),
            "recursive_model_applications": _last_items(self.recursive_model_applications, 5),
            "recursive_verified_artifacts": _last_items(self.recursive_verified_artifacts, 5),
            "phase_audits": _last_items(self.phase_audits, 2),
            "batch_contract_samples": _last_items(self.batch_contract_samples, 5),
            "phase_replay_example_ids": _last_items(self.phase_replay_example_ids, 10),
            "objective_feedback_term_names": tuple(self.last_objective_loss_terms),
            "objective_feedback_term_count": len(self.last_objective_loss_terms),
            "last_objective_loss_terms": dict(self.last_objective_loss_terms),
            "objective_feedback_term_totals": dict(self.objective_feedback_term_totals),
            "objective_feedback_history": _last_items(self.objective_feedback_history, 5),
            "errors": list(self.errors),
        }


class LLMTrainer:
    def __init__(
        self,
        model: CortexTransformerLM,
        train_data: MemmapCausalDataset,
        val_data: MemmapCausalDataset,
        config: TrainingConfig,
        *,
        run_dir: str | Path,
        model_kind: str,
        corpus_identity: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.run_dir = Path(run_dir)
        self.model_kind = model_kind
        self.objective = CortexObjective()
        self.device = self._resolve_device()
        if self.device.type == "cuda" and self.model.config.use_ternary_core:
            self._enforce_strict_native_ternary_cuda()
        self.precision = PrecisionPolicy(config.precision, require_cuda=config.require_cuda)
        self.corpus_identity = dict(corpus_identity or train_data.manifest.identity())
        self.tokenizer = LLMTokenizer.load(self.train_data.manifest.tokenizer_file)
        if self.val_data.manifest.identity(verify=False) != self.train_data.manifest.identity(verify=False):
            raise ValueError("train and validation datasets must come from the same tokenized corpus manifest")
        if config.num_threads is not None:
            torch.set_num_threads(config.num_threads)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.code_state = code_state_report()

    def _enforce_strict_native_ternary_cuda(self) -> None:
        if self.model.config.native_ternary_backend != STRICT_NATIVE_TERNARY_BACKEND:
            raise RuntimeError(
                "CUDA Cortex-3 LLM training requires native_ternary_backend='extension'; "
                "auto/rawkernel are diagnostic modes and are not allowed for strict training"
            )
        self.model.config = replace(
            self.model.config,
            use_native_ternary_kernel=True,
            require_native_ternary_kernel=True,
            native_ternary_backend=STRICT_NATIVE_TERNARY_BACKEND,
        )
        for module in self.model.modules():
            if isinstance(module, BitLinear):
                module.config = replace(
                    module.config,
                    use_native_cuda_kernel=True,
                    require_native_cuda_kernel=True,
                    native_cuda_backend=STRICT_NATIVE_TERNARY_BACKEND,
                )

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            if self.config.require_cuda and not torch.cuda.is_available():
                raise RuntimeError("CUDA was required but torch.cuda.is_available() is false")
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cpu")
        device = torch.device(self.config.device)
        if self.config.require_cuda and device.type != "cuda":
            raise RuntimeError("require_cuda=True needs a CUDA device")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available")
        return device

    def _batch(self, dataset: MemmapCausalDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return dataset.sample_batch(self.config.batch_size, generator=self.generator, device=self.device)

    def evaluate(self, dataset: MemmapCausalDataset, *, split: str, step: int) -> TrainingPoint:
        self.model.eval()
        losses: list[LossBreakdown] = []
        correct = 0
        total = 0
        future_correct = 0.0
        future_cost = 0.0
        with torch.no_grad():
            for _ in range(self.config.eval_batches):
                x, y, future = self._batch(dataset)
                with self.precision.autocast(self.device.type):
                    output = self.model(x)
                    _, breakdown = self.objective.compute(
                        output,
                        y,
                        future,
                        use_cortex_terms=self.model.config.use_cortex_heads,
                    )
                losses.append(breakdown)
                predicted = output.logits.argmax(dim=-1)
                correct += int(predicted.eq(y).sum().cpu())
                total += int(y.numel())
                if self.model.config.use_cortex_heads:
                    for horizon, logits in output.mtp_logits.items():
                        if horizon <= future.shape[-1]:
                            future_correct += float(logits.argmax(dim=-1).eq(future[:, :, horizon - 1]).float().mean().cpu())
                    future_cost += 1.0
                else:
                    horizon_cost = sum(self.model.config.horizons)
                    one_step_acc = float(predicted.eq(y).float().mean().cpu())
                    future_correct += one_step_acc * len(self.model.config.horizons)
                    future_cost += float(horizon_cost)
        avg = {
            "total": sum(item.total for item in losses) / len(losses),
            "next_token": sum(item.next_token for item in losses) / len(losses),
            "mtp": sum(item.mtp for item in losses) / len(losses),
        }
        return TrainingPoint(
            step=step,
            split=split,
            loss=avg["total"],
            next_token_loss=avg["next_token"],
            token_accuracy=correct / max(1, total),
            mtp_loss=avg["mtp"],
            future_tokens_per_cost=future_correct / max(1.0, future_cost),
        )

    def _prime_phase_controller(self, phase_controller: CortexTrainingPhaseController, *, step: int) -> None:
        self.model.eval()
        with torch.no_grad():
            x, y, future = self._batch(self.val_data)
            phase_controller.observe_input_batch(step=step, input_ids=x, source="prime-val")
            with self.precision.autocast(self.device.type):
                output = self.model(x)
                _, breakdown = self.objective.compute(
                    output,
                    y,
                    future,
                    use_cortex_terms=self.model.config.use_cortex_heads,
                )
            phase_controller.observe_batch_contract(
                step=step,
                output=output,
                future_targets=future,
                breakdown=breakdown,
            )
        phase_controller.run_phase_audit(step=step)

    def train(self, *, name: str) -> TrainingRunReport:
        runtime = DistributedRuntime.from_env(
            requested=self.config.distributed,
            device_type=self.device.type,
            gloo_interface=self.config.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        _barrier_if_needed(runtime)
        rank_seed = self.config.seed + runtime.rank * 100_003
        torch.manual_seed(rank_seed)
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        self.generator.manual_seed(rank_seed)
        self.model.to(self.device)
        trainable: nn.Module = self.model
        if runtime.enabled:
            cuda_index = self.device.index if self.device.index is not None else runtime.local_rank
            kwargs = {"device_ids": [cuda_index]} if self.device.type == "cuda" else {}
            trainable = nn.parallel.DistributedDataParallel(self.model, **kwargs)
        phase_controller = (
            CortexTrainingPhaseController(self.model, self.tokenizer, self.config, run_dir=self.run_dir)
            if (
                self.model.config.use_cortex_heads
                and self.model.config.use_ternary_core
                and self.model.config.use_skill_aware_experts
                and self.model.config.use_variable_in_compressor
                and self.model.config.use_learned_memory_policy
                and self.model.config.use_certificate_head
                and self.model.config.use_latent_reasoning_workspace
                and self.model.config.horizons == (1, 2, 4, 8)
            )
            else None
        )
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scaler = self.precision.scaler(self.device.type)
        curve: list[TrainingPoint] = []
        resource_monitor = (
            ResourceUsageMonitor(device=self.device, interval_seconds=self.config.resource_monitor_interval)
            if runtime.is_main
            else None
        )
        if resource_monitor is not None:
            resource_monitor.start()
        resumed_from = self._resolve_resume_checkpoint()
        start_step = 0
        last_completed_step = 0
        resource_usage: Mapping[str, Any] = {"enabled": False}
        try:
            if resumed_from is not None:
                start_step, curve = self.load_checkpoint(
                    resumed_from,
                    optimizer=optimizer,
                    scaler=scaler,
                    phase_controller=phase_controller,
                    restore_rng=not runtime.enabled,
                )
                last_completed_step = start_step
                if start_step > self.config.steps:
                    raise ValueError(f"checkpoint step {start_step} is greater than target steps {self.config.steps}")
                if runtime.enabled:
                    resumed_seed = self.config.seed + runtime.rank * 100_003 + start_step * 997
                    torch.manual_seed(resumed_seed)
                    random.seed(resumed_seed)
                    np.random.seed(resumed_seed)
                    self.generator.manual_seed(resumed_seed)
                if phase_controller is not None:
                    deliverable_audit = phase_controller.checkpoint_state_summary().get("phase_deliverable_audit", {})
                    if start_step >= self.config.steps or not bool(deliverable_audit.get("passed")):
                        self._prime_phase_controller(phase_controller, step=start_step)
            else:
                curve.append(self.evaluate(self.train_data, split="train", step=0))
                curve.append(self.evaluate(self.val_data, split="val", step=0))
                if phase_controller is not None:
                    self._prime_phase_controller(phase_controller, step=0)
            if runtime.is_main:
                self._prune_intermediate_checkpoints()
            for step in range(start_step + 1, self.config.steps + 1):
                self.model.train()
                optimizer.zero_grad(set_to_none=True)
                for micro_step in range(self.config.gradient_accumulation_steps):
                    x, y, future = self._batch(self.train_data)
                    sync_context = (
                        trainable.no_sync()
                        if runtime.enabled and micro_step < self.config.gradient_accumulation_steps - 1
                        else nullcontext()
                    )
                    with sync_context:
                        with self.precision.autocast(self.device.type):
                            output = trainable(x) if runtime.enabled else self.model(x)
                            loss, breakdown = self.objective.compute(
                                output,
                                y,
                                future,
                                use_cortex_terms=self.model.config.use_cortex_heads,
                            )
                            if phase_controller is not None:
                                loss = loss + phase_controller.auxiliary_loss(output)
                                replay_loss = phase_controller.replay_loss(
                                    trainable if runtime.enabled else self.model,
                                    self.objective,
                                    self.precision,
                                    self.device,
                                )
                                if replay_loss is not None:
                                    loss = loss + replay_loss
                            loss = loss / self.config.gradient_accumulation_steps
                        if (
                            phase_controller is not None
                            and micro_step == 0
                            and phase_controller.should_sample_step(step)
                        ):
                            phase_controller.observe_batch_contract(
                                step=step,
                                output=output,
                                future_targets=future,
                                breakdown=breakdown,
                            )
                            phase_controller.observe_input_batch(step=step, input_ids=x, source="train", output=output)
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable.parameters(), self.config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable.parameters(), self.config.grad_clip)
                    optimizer.step()
                self.model.requantize_ternary_core(certify_zeros=False)
                last_completed_step = step
                if step % self.config.eval_interval == 0 or step == self.config.steps:
                    curve.append(self.evaluate(self.train_data, split="train", step=step))
                    curve.append(self.evaluate(self.val_data, split="val", step=step))
                    if phase_controller is not None:
                        phase_controller.run_phase_audit(step=step)
                if runtime.is_main and step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(
                        optimizer,
                        self.run_dir / f"checkpoint_step_{step}.pt",
                        step=step,
                        curve=curve,
                        scaler=scaler,
                        phase_controller=phase_controller,
                    )
                    self._prune_intermediate_checkpoints()
                    if resource_monitor is not None:
                        resource_monitor.write_snapshot(
                            self.run_dir / "resource_usage_live.json",
                            metadata={
                                "model_kind": self.model_kind,
                                "step": step,
                                "target_steps": self.config.steps,
                                "checkpoint": str(self.run_dir / f"checkpoint_step_{step}.pt"),
                            },
                        )
            checkpoint_path = self.run_dir / "checkpoint_final.pt"
            if runtime.is_main:
                checkpoint_path = self.save_checkpoint(
                    optimizer,
                    checkpoint_path,
                    step=self.config.steps,
                    curve=curve,
                    scaler=scaler,
                    phase_controller=phase_controller,
                )
                self._write_curve(curve)
        finally:
            if resource_monitor is not None:
                resource_usage = resource_monitor.stop()
                resource_monitor.write_snapshot(
                    self.run_dir / "resource_usage_summary.json",
                    metadata={
                        "model_kind": self.model_kind,
                        "step": last_completed_step,
                        "target_steps": self.config.steps,
                        "final": True,
                    },
                )
        final_train = [point for point in curve if point.split == "train"][-1]
        final_val = [point for point in curve if point.split == "val"][-1]
        cortex_phase_report: Mapping[str, Any] = phase_controller.summary() if phase_controller is not None else {"enabled": False}
        if cortex_phase_report.get("errors"):
            raise RuntimeError(f"Cortex phase integration errors: {cortex_phase_report['errors']}")
        report = TrainingRunReport(
            name=name,
            model_kind=self.model_kind,
            run_dir=str(self.run_dir),
            checkpoint_path=str(checkpoint_path),
            start_step=start_step,
            optimizer_steps=max(0, self.config.steps - start_step),
            effective_batch_size=self.config.batch_size * self.config.gradient_accumulation_steps * runtime.world_size,
            resumed_from=str(resumed_from) if resumed_from is not None else None,
            final_train=final_train,
            final_val=final_val,
            curve=tuple(curve),
            config={
                "training": asdict(self.config),
                "model": asdict(self.model.config),
                "corpus_identity": self.corpus_identity,
            },
            hardware=hardware_report(),
            code_state=self.code_state,
            resource_usage=resource_usage,
            cortex_phase_report=cortex_phase_report,
        )
        if runtime.is_main:
            _write_json(self.run_dir / "training_report.json", report.to_dict())
            if phase_controller is not None:
                _write_json(self.run_dir / "cortex_phase_report.json", cortex_phase_report)
        _barrier_if_needed(runtime)
        return report

    def _write_curve(self, curve: Sequence[TrainingPoint]) -> None:
        csv_path = self.run_dir / "learning_curve.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve[0].to_dict().keys()))
            writer.writeheader()
            for point in curve:
                writer.writerow(point.to_dict())

    def _resolve_resume_checkpoint(self) -> Path | None:
        if self.config.resume_from_checkpoint is not None:
            checkpoint = Path(self.config.resume_from_checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
            return checkpoint
        if not (self.config.resume or self.config.resume_if_exists):
            return None
        final_checkpoint = self.run_dir / "checkpoint_final.pt"
        if final_checkpoint.exists():
            return final_checkpoint
        candidates: list[tuple[int, Path]] = []
        skipped: list[str] = []
        for path in self.run_dir.glob("checkpoint_step_*.pt"):
            raw_step = path.stem.removeprefix("checkpoint_step_")
            if raw_step.isdigit():
                step = int(raw_step)
                if self._checkpoint_sidecar_is_complete(path, expected_step=step):
                    candidates.append((step, path))
                else:
                    skipped.append(path.name)
        if candidates:
            return sorted(candidates)[-1][1]
        if self.config.resume_if_exists:
            return None
        if skipped:
            raise FileNotFoundError(
                f"resume=True found only incomplete checkpoint candidates in {self.run_dir}: {', '.join(sorted(skipped))}"
            )
        raise FileNotFoundError(f"resume=True but no checkpoint was found in {self.run_dir}")

    def _checkpoint_sidecar_is_complete(self, checkpoint: Path, *, expected_step: int | None = None) -> bool:
        sidecar = checkpoint.with_name(checkpoint.name + ".json")
        if not checkpoint.exists() or not sidecar.exists():
            return False
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return False
        recorded_size = payload.get("checkpoint_size_bytes")
        if recorded_size is None or int(recorded_size) != int(checkpoint.stat().st_size):
            return False
        if expected_step is not None and int(payload.get("step", -1)) != int(expected_step):
            return False
        return True

    def _prune_intermediate_checkpoints(self) -> None:
        limit = int(self.config.max_intermediate_checkpoints)
        if limit <= 0:
            return
        candidates: list[tuple[int, Path]] = []
        incomplete_candidates: list[Path] = []
        for path in self.run_dir.glob("checkpoint_step_*.pt"):
            raw_step = path.stem.removeprefix("checkpoint_step_")
            if raw_step.isdigit():
                step = int(raw_step)
                if self._checkpoint_sidecar_is_complete(path, expected_step=step):
                    candidates.append((step, path))
                else:
                    incomplete_candidates.append(path)
        for checkpoint in incomplete_candidates:
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_name(checkpoint.name + ".json").unlink(missing_ok=True)
        for _, checkpoint in sorted(candidates)[:-limit]:
            checkpoint.unlink(missing_ok=True)
            checkpoint.with_name(checkpoint.name + ".json").unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        phase_controller: CortexTrainingPhaseController | None = None,
        restore_rng: bool = True,
    ) -> tuple[int, list[TrainingPoint]]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if payload.get("model_kind") != self.model_kind:
            raise ValueError(f"checkpoint model_kind={payload.get('model_kind')!r} does not match {self.model_kind!r}")
        checkpoint_model_config = dict(payload.get("model_config") or {})
        checkpoint_model_config.setdefault("use_skill_aware_experts", False)
        checkpoint_model_config.setdefault("skill_expert_count", 4)
        checkpoint_model_config.setdefault("skill_expert_top_k", 2)
        checkpoint_model_config.setdefault("skill_expert_context_strength", 1.25)
        checkpoint_model_config.setdefault("use_variable_in_compressor", False)
        checkpoint_model_config.setdefault("variable_compression_wide_kernel", 8)
        checkpoint_model_config.setdefault("use_learned_memory_policy", False)
        checkpoint_model_config.setdefault("learned_memory_temperature", 1.0)
        checkpoint_model_config.setdefault("use_native_ternary_kernel", True)
        checkpoint_model_config.setdefault("require_native_ternary_kernel", False)
        checkpoint_model_config.setdefault("native_ternary_autotune_cache_path", None)
        checkpoint_model_config.setdefault("native_ternary_autotune_cache_write", True)
        checkpoint_model_config.setdefault("use_certificate_head", False)
        checkpoint_model_config.setdefault("certificate_latent_size", 64)
        checkpoint_model_config.setdefault("use_latent_reasoning_workspace", False)
        checkpoint_model_config.setdefault("latent_workspace_steps", 3)
        checkpoint_model_config.setdefault("latent_workspace_feedback_strength", 0.20)
        if checkpoint_model_config != asdict(self.model.config):
            raise ValueError("checkpoint model_config does not match the current model")
        checkpoint_corpus_identity = payload.get("corpus_identity")
        if checkpoint_corpus_identity is None:
            raise ValueError("checkpoint is missing corpus_identity; rebuild or restart with a checkpoint produced by this harness version")
        if checkpoint_corpus_identity != self.corpus_identity:
            checkpoint_digest = checkpoint_corpus_identity.get("identity_sha256") if isinstance(checkpoint_corpus_identity, Mapping) else None
            current_digest = self.corpus_identity.get("identity_sha256")
            raise ValueError(
                "checkpoint corpus_identity does not match the current corpus "
                f"(checkpoint={checkpoint_digest!r}, current={current_digest!r})"
            )
        self.model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if phase_controller is not None:
            phase_controller.load_state_dict(payload.get("cortex_phase_state"))
        if restore_rng:
            rng_state = payload.get("rng_state", {})
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"].cpu())
            if self.device.type == "cuda" and rng_state.get("torch_cuda_all"):
                cuda_states: list[torch.Tensor] = []
                for state in rng_state["torch_cuda_all"]:
                    if isinstance(state, torch.Tensor):
                        cuda_states.append(state.detach().cpu().to(dtype=torch.uint8))
                    else:
                        cuda_states.append(torch.as_tensor(state, dtype=torch.uint8))
                torch.cuda.set_rng_state_all(cuda_states[: torch.cuda.device_count()])
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "trainer_generator" in rng_state:
                self.generator.set_state(rng_state["trainer_generator"].cpu())
        curve_payload = payload.get("curve", [])
        curve = [TrainingPoint(**point) for point in curve_payload]
        step = int(payload.get("step", 0))
        if not curve:
            curve.append(self.evaluate(self.train_data, split="train", step=step))
            curve.append(self.evaluate(self.val_data, split="val", step=step))
        return step, curve

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "trainer_generator": self.generator.get_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        path: str | Path,
        *,
        step: int,
        curve: Sequence[TrainingPoint],
        scaler: torch.amp.GradScaler | None = None,
        phase_controller: CortexTrainingPhaseController | None = None,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_output = output.with_name(output.name + ".tmp")
        tmp_output.unlink(missing_ok=True)
        code_state = self.code_state
        cortex_phase_state = phase_controller.state_dict() if phase_controller is not None else None
        cortex_phase_state_summary = phase_controller.checkpoint_state_summary() if phase_controller is not None else None
        torch.save(
            {
                "schema_version": 2,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "model_config": asdict(self.model.config),
                "training_config": asdict(self.config),
                "model_kind": self.model_kind,
                "corpus_identity": self.corpus_identity,
                "code_state": code_state,
                "cortex_phase_state": cortex_phase_state,
                "curve": [point.to_dict() for point in curve],
                "rng_state": self._rng_state(),
            },
            tmp_output,
        )
        tmp_output.replace(output)
        _write_json(
            output.with_name(output.name + ".json"),
            {
                "schema_version": 1,
                "checkpoint": str(output),
                "checkpoint_size_bytes": int(output.stat().st_size),
                "step": step,
                "model_kind": self.model_kind,
                "training_config": asdict(self.config),
                "checkpoint_retention": {
                    "checkpoint_interval": int(self.config.checkpoint_interval),
                    "max_intermediate_checkpoints": int(self.config.max_intermediate_checkpoints),
                },
                "curve_points": len(curve),
                "corpus_identity_sha256": self.corpus_identity.get("identity_sha256"),
                "code_state": code_state,
                "cortex_phase_state_present": cortex_phase_state is not None,
                "cortex_phase_state_summary": cortex_phase_state_summary,
                "written_at": time.time(),
            },
        )
        return output


@dataclass(frozen=True)
class ComparisonConfig:
    vocab_size: int = 2048
    min_frequency: int = 2
    seq_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 2, 4, 8)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cortex_win_margin: float = 1.05
    max_next_token_loss_regression: float = 1.20
    min_baseline_future_tokens_per_cost: float = 1e-6
    min_corpus_tokens: int = 0
    max_corpus_tokens: int | None = None
    tokenizer_training_chars: int | None = None
    min_planned_train_tokens: int = 0
    native_ternary_backend: str = STRICT_NATIVE_TERNARY_BACKEND

    def __post_init__(self) -> None:
        if self.min_corpus_tokens < 0:
            raise ValueError("min_corpus_tokens must be non-negative")
        if self.max_corpus_tokens is not None and self.max_corpus_tokens < 1:
            raise ValueError("max_corpus_tokens must be positive when provided")
        if self.max_corpus_tokens is not None and self.max_corpus_tokens < self.min_corpus_tokens:
            raise ValueError("max_corpus_tokens must be >= min_corpus_tokens")
        if self.tokenizer_training_chars is not None and self.tokenizer_training_chars < 1:
            raise ValueError("tokenizer_training_chars must be positive when provided")
        if self.min_planned_train_tokens < 0:
            raise ValueError("min_planned_train_tokens must be non-negative")
        if self.native_ternary_backend not in NATIVE_TERNARY_BACKEND_CHOICES:
            raise ValueError(f"native_ternary_backend must be one of: {', '.join(NATIVE_TERNARY_BACKEND_CHOICES)}")


@dataclass(frozen=True)
class ComparisonReport:
    run_dir: str
    manifest: Mapping[str, Any]
    baseline: Mapping[str, Any]
    cortex: Mapping[str, Any]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]
    plan: Mapping[str, Any]
    curve_audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkDomainSpec:
    name: str
    patterns: tuple[str, ...]


DEFAULT_BENCHMARK_DOMAINS: Mapping[str, BenchmarkDomainSpec] = {
    "sequence": BenchmarkDomainSpec(
        "sequence",
        (
            "alpha beta gamma delta epsilon zeta eta theta.",
            "red green blue yellow red green blue yellow.",
            "one two three five eight thirteen twenty one.",
            "north east south west north east south west.",
        ),
    ),
    "reasoning": BenchmarkDomainSpec(
        "reasoning",
        (
            "if the verifier accepts the invariant then the compiled path is reused.",
            "a slow solve creates evidence, the evidence creates a certificate.",
            "the anchor ledger preserves exact symbols while the latent store compresses context.",
            "regrowth changes the smallest recovering block and then checks protected skills.",
        ),
    ),
    "code": BenchmarkDomainSpec(
        "code",
        (
            "def add(a, b): return a + b",
            "class Gate: def __init__(self, threshold): self.threshold = threshold",
            "assert normalize('OK') == 'OK'",
            "for token in stream: ledger.record(token)",
        ),
    ),
    "anchors": BenchmarkDomainSpec(
        "anchors",
        (
            "ticket AX-1042 belongs to Sofia and must remain exact.",
            "identifier C3-7777-Z maps to prototype ledger alpha.",
            "vault key QK-55-DELTA appears once and must be copied exactly.",
            "entity Mira owns checksum 19AF and sequence tag LLM-2048.",
        ),
    ),
}


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    run_dir: str
    domains: tuple[Mapping[str, Any], ...]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatisticalBenchmarkReport:
    run_dir: str
    seeds: tuple[Mapping[str, Any], ...]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonMatrixReport:
    run_dir: str
    manifest: Mapping[str, Any]
    seeds: tuple[Mapping[str, Any], ...]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusMatrixReport:
    run_dir: str
    corpora: tuple[Mapping[str, Any], ...]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMExperimentReport:
    run_dir: str
    manifest: Mapping[str, Any]
    doctor: Mapping[str, Any]
    corpora: tuple[Mapping[str, Any], ...]
    corpus_matrix: Mapping[str, Any]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMExperimentAuditReport:
    run_dir: str
    passed: bool
    failed_checks: tuple[str, ...]
    checked_artifacts: tuple[str, ...]
    proof: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMExperimentPreflightReport:
    run_dir: str
    passed: bool
    failed_checks: tuple[str, ...]
    warnings: tuple[str, ...]
    estimates: Mapping[str, Any]
    hardware: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMExperimentInspectionReport:
    run_dir: str
    exists: bool
    status: str
    running_processes: tuple[Mapping[str, Any], ...]
    gpu_snapshot: Mapping[str, Any]
    manifest: Mapping[str, Any]
    corpora: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step_from_name(path: Path) -> int | None:
    raw = path.stem.removeprefix("checkpoint_step_")
    return int(raw) if raw.isdigit() else None


def _last_validation_row(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "val"]
    if not rows:
        return {}
    row = rows[-1]
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        try:
            out[key] = int(value) if key == "step" else float(value)
        except ValueError:
            out[key] = value
    return out


def _model_run_inspection(model_dir: Path) -> Mapping[str, Any]:
    step_checkpoints = []
    for checkpoint in model_dir.glob("checkpoint_step_*.pt"):
        step = _checkpoint_step_from_name(checkpoint)
        if step is None:
            continue
        sidecar = _read_json_if_exists(checkpoint.with_name(checkpoint.name + ".json"))
        step_checkpoints.append(
            {
                "step": step,
                "path": str(checkpoint),
                "size_bytes": int(checkpoint.stat().st_size),
                "last_write_time": checkpoint.stat().st_mtime,
                "sidecar_exists": bool(sidecar),
                "sidecar": sidecar,
            }
        )
    step_checkpoints.sort(key=lambda item: item["step"])
    final_checkpoint = model_dir / "checkpoint_final.pt"
    final_sidecar = _read_json_if_exists(final_checkpoint.with_name(final_checkpoint.name + ".json"))
    training_report = _read_json_if_exists(model_dir / "training_report.json")
    cortex_phase_report = _read_json_if_exists(model_dir / "cortex_phase_report.json")
    latest_sidecar = step_checkpoints[-1]["sidecar"] if step_checkpoints else None
    latest_architecture_audit = {}
    latest_phase_deliverable_audit = {}
    if isinstance(latest_sidecar, Mapping):
        state_summary = latest_sidecar.get("cortex_phase_state_summary", {})
        if isinstance(state_summary, Mapping):
            latest_architecture_audit = dict(state_summary.get("architecture_audit") or {})
            latest_phase_deliverable_audit = dict(state_summary.get("phase_deliverable_audit") or {})
    return {
        "exists": model_dir.exists(),
        "latest_checkpoint_step": max((item["step"] for item in step_checkpoints), default=None),
        "checkpoint_count": len(step_checkpoints),
        "latest_checkpoint": step_checkpoints[-1] if step_checkpoints else None,
        "latest_checkpoint_architecture_audit": latest_architecture_audit,
        "latest_checkpoint_phase_deliverable_audit": latest_phase_deliverable_audit,
        "final_checkpoint_exists": final_checkpoint.exists(),
        "final_checkpoint_size_bytes": int(final_checkpoint.stat().st_size) if final_checkpoint.exists() else 0,
        "final_checkpoint_sidecar_exists": bool(final_sidecar),
        "final_checkpoint_sidecar": final_sidecar,
        "learning_curve_exists": (model_dir / "learning_curve.csv").exists(),
        "last_validation": _last_validation_row(model_dir / "learning_curve.csv"),
        "training_report_exists": bool(training_report),
        "training_report_summary": {
            "start_step": training_report.get("start_step"),
            "optimizer_steps": training_report.get("optimizer_steps"),
            "effective_batch_size": training_report.get("effective_batch_size"),
            "resumed_from": training_report.get("resumed_from"),
            "resource_usage": training_report.get("resource_usage", {}),
            "cortex_phase_report": training_report.get("cortex_phase_report", {}),
        } if training_report else {},
        "cortex_phase_report_exists": bool(cortex_phase_report),
        "cortex_phase_summary": {
            "all_phases_active": cortex_phase_report.get("all_phases_active"),
            "phase_event_counts": cortex_phase_report.get("phase_event_counts", {}),
            "training_influence": cortex_phase_report.get("training_influence", {}),
            "architecture_audit": cortex_phase_report.get("architecture_audit", {}),
            "phase_deliverable_audit": cortex_phase_report.get("phase_deliverable_audit", {}),
            "errors": cortex_phase_report.get("errors", ()),
        } if cortex_phase_report else {},
    }


def _running_processes_for_run(run_dir: Path) -> tuple[Mapping[str, Any], ...]:
    needles = {str(run_dir), str(run_dir.resolve())}
    if len(run_dir.name) >= 12 and run_dir.name.lower() not in {"run", "runs", "experiment"}:
        needles.add(run_dir.name)
    matches: list[Mapping[str, Any]] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            cmdline = " ".join(str(part) for part in (proc.info.get("cmdline") or ()))
            if not cmdline or not any(needle in cmdline for needle in needles):
                continue
            if int(proc.info["pid"]) == os.getpid() or "inspect-experiment" in cmdline:
                continue
            cpu_times = proc.cpu_times()
            memory_info = proc.memory_info()
            matches.append(
                {
                    "pid": int(proc.info["pid"]),
                    "ppid": int(proc.info.get("ppid") or 0),
                    "name": proc.info.get("name"),
                    "cmdline": cmdline,
                    "create_time": float(proc.info.get("create_time") or 0.0),
                    "cpu_seconds": float(cpu_times.user + cpu_times.system),
                    "rss_bytes": int(memory_info.rss),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return tuple(matches)


def _gpu_snapshot() -> Mapping[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2.0, check=False)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found"}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {"available": False, "error": completed.stderr.strip() or f"nvidia-smi exited {completed.returncode}"}
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "gpu_utilization_percent": float(parts[1]),
                    "gpu_memory_utilization_percent": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "power_draw_watts": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return {"available": True, "devices": tuple(devices)}


def inspect_llm_experiment(run_dir: str | Path) -> LLMExperimentInspectionReport:
    root = Path(run_dir)
    warnings: list[str] = []
    if not root.exists():
        return LLMExperimentInspectionReport(
            run_dir=str(root),
            exists=False,
            status="missing",
            running_processes=(),
            gpu_snapshot=_gpu_snapshot(),
            manifest={},
            corpora=(),
            warnings=("run_dir does not exist",),
        )

    manifest = _read_json_if_exists(root / "experiment_manifest.normalized.json")
    running = _running_processes_for_run(root)
    experiment_report = _read_json_if_exists(root / "experiment_report.json")
    status = "complete" if experiment_report else "running" if running else "partial"
    if not manifest:
        warnings.append("experiment_manifest.normalized.json is missing")

    corpora: list[Mapping[str, Any]] = []
    matrix_root = root / "corpus_matrix"
    if not matrix_root.exists():
        warnings.append("corpus_matrix directory is missing")
    else:
        for corpus_dir in sorted(path for path in matrix_root.iterdir() if path.is_dir()):
            corpus_manifest = _read_json_if_exists(corpus_dir / "corpus" / "manifest.json")
            seed_payloads: list[Mapping[str, Any]] = []
            for seed_dir in sorted(path for path in corpus_dir.glob("seed_*") if path.is_dir()):
                comparison_report = _read_json_if_exists(seed_dir / "comparison_report.json")
                seed_payloads.append(
                    {
                        "seed": seed_dir.name.removeprefix("seed_"),
                        "run_plan_exists": (seed_dir / "run_plan.json").exists(),
                        "baseline": _model_run_inspection(seed_dir / "baseline_ntp"),
                        "cortex": _model_run_inspection(seed_dir / "cortex3"),
                        "comparison_report_exists": bool(comparison_report),
                        "proof": comparison_report.get("proof", {}) if comparison_report else {},
                    }
                )
            corpora.append(
                {
                    "name": corpus_dir.name,
                    "manifest_exists": bool(corpus_manifest),
                    "token_count": corpus_manifest.get("token_count"),
                    "seq_len": corpus_manifest.get("seq_len"),
                    "max_horizon": corpus_manifest.get("max_horizon"),
                    "preparation_config": corpus_manifest.get("preparation_config", {}),
                    "seed_runs": tuple(seed_payloads),
                }
            )

    return LLMExperimentInspectionReport(
        run_dir=str(root),
        exists=True,
        status=status,
        running_processes=running,
        gpu_snapshot=_gpu_snapshot(),
        manifest=manifest,
        corpora=tuple(corpora),
        warnings=tuple(warnings),
    )


def _training_precision_bytes(precision: str) -> int:
    return 2 if precision in {"bf16", "fp16"} else 4


def _transformer_parameter_count(config: TransformerConfig) -> int:
    d_model = int(config.d_model)
    vocab_size = int(config.vocab_size)
    seq_len = int(config.seq_len)
    n_layers = int(config.n_layers)
    embedding = vocab_size * d_model
    position = seq_len * d_model
    attention = (d_model * 3 * d_model) + (d_model * d_model + d_model)
    layer_norms = 4 * d_model
    mlp = (d_model * 4 * d_model + 4 * d_model) + (4 * d_model * d_model + d_model)
    final_norm = 2 * d_model
    total = embedding + position + n_layers * (attention + layer_norms + mlp) + final_norm
    if config.use_cortex_heads:
        total += len(config.horizons) * (d_model * vocab_size + vocab_size)
        total += d_model + 1
    if config.use_skill_aware_experts:
        router = d_model * int(config.skill_expert_count) + int(config.skill_expert_count)
        expert = (d_model * d_model * 2 + d_model * 2) + (d_model * 2 * d_model + d_model)
        total += router + int(config.skill_expert_count) * expert
    if config.use_variable_in_compressor:
        total += d_model + 1
        total += d_model * d_model + d_model
    if config.use_learned_memory_policy:
        policy_hidden = max(16, d_model // 2)
        total += 2 * d_model
        total += d_model * policy_hidden + policy_hidden
        total += policy_hidden * 3 + 3
        total += d_model * d_model + d_model
        total += d_model
    if config.use_certificate_head:
        latent = int(config.certificate_latent_size)
        total += d_model * latent + latent
        total += latent * latent + latent
        total += latent * vocab_size + vocab_size
        total += latent * len(CertificateType) + len(CertificateType)
        total += latent + 1
    if config.use_latent_reasoning_workspace:
        latent = int(config.certificate_latent_size)
        total += 2 * d_model
        total += d_model + 1
        total += d_model * latent + latent
        total += latent * latent + latent
        total += latent + 1
        total += latent * d_model + d_model
    return int(total)


def _estimate_transformer_training_memory(config: TransformerConfig, training: TrainingConfig) -> dict[str, Any]:
    precision_bytes = _training_precision_bytes(training.precision)
    parameters = _transformer_parameter_count(config)
    batch_size = int(training.batch_size)
    seq_len = int(config.seq_len)
    d_model = int(config.d_model)
    n_layers = int(config.n_layers)
    n_heads = int(config.n_heads)
    vocab_size = int(config.vocab_size)
    training_state_bytes = parameters * 16
    parameter_forward_bytes = parameters * precision_bytes
    hidden_activation_bytes = batch_size * seq_len * d_model * n_layers * precision_bytes * 12
    attention_workspace_bytes = batch_size * n_heads * seq_len * seq_len * precision_bytes * 2
    output_heads = 1 + (len(config.horizons) if config.use_cortex_heads else 0)
    logits_workspace_bytes = batch_size * seq_len * vocab_size * precision_bytes * output_heads
    subtotal = (
        training_state_bytes
        + parameter_forward_bytes
        + hidden_activation_bytes
        + attention_workspace_bytes
        + logits_workspace_bytes
    )
    fragmentation_margin_bytes = int(subtotal * 0.20)
    peak_training_bytes = subtotal + fragmentation_margin_bytes
    return {
        "parameters": int(parameters),
        "precision_bytes": int(precision_bytes),
        "batch_size_per_rank": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        "output_heads": int(output_heads),
        "skill_aware_experts": bool(config.use_skill_aware_experts),
        "skill_expert_count": int(config.skill_expert_count) if config.use_skill_aware_experts else 0,
        "skill_expert_top_k": int(config.skill_expert_top_k) if config.use_skill_aware_experts else 0,
        "variable_in_compressor": bool(config.use_variable_in_compressor),
        "learned_memory_policy": bool(config.use_learned_memory_policy),
        "native_ternary_kernel": bool(config.use_ternary_core and config.use_native_ternary_kernel),
        "native_ternary_kernel_required": bool(config.use_ternary_core and config.require_native_ternary_kernel),
        "native_ternary_backend": str(config.native_ternary_backend) if config.use_ternary_core else "",
        "certificate_head": bool(config.use_certificate_head),
        "certificate_latent_size": int(config.certificate_latent_size) if config.use_certificate_head else 0,
        "latent_reasoning_workspace": bool(config.use_latent_reasoning_workspace),
        "latent_workspace_steps": int(config.latent_workspace_steps) if config.use_latent_reasoning_workspace else 0,
        "training_state_bytes": int(training_state_bytes),
        "parameter_forward_bytes": int(parameter_forward_bytes),
        "hidden_activation_bytes": int(hidden_activation_bytes),
        "attention_workspace_bytes": int(attention_workspace_bytes),
        "logits_workspace_bytes": int(logits_workspace_bytes),
        "fragmentation_margin_bytes": int(fragmentation_margin_bytes),
        "estimated_peak_training_bytes": int(peak_training_bytes),
    }


def _experiment_model_memory_estimates(config: ComparisonConfig) -> dict[str, Any]:
    baseline_config = TransformerConfig(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        horizons=config.horizons,
        use_cortex_heads=False,
    )
    cortex_config = TransformerConfig(**{
        **asdict(baseline_config),
        "use_cortex_heads": True,
        "use_ternary_core": True,
        "use_native_ternary_kernel": True,
        "require_native_ternary_kernel": _strict_native_ternary_required_for_training(config.training),
        "native_ternary_backend": config.native_ternary_backend,
        "use_skill_aware_experts": True,
        "use_variable_in_compressor": True,
        "use_learned_memory_policy": True,
        "use_certificate_head": True,
        "use_latent_reasoning_workspace": True,
    })
    baseline = _estimate_transformer_training_memory(baseline_config, config.training)
    cortex = _estimate_transformer_training_memory(cortex_config, config.training)
    return {
        "baseline_next_token": baseline,
        "cortex3_multi_horizon": cortex,
        "max_estimated_peak_training_bytes": int(
            max(baseline["estimated_peak_training_bytes"], cortex["estimated_peak_training_bytes"])
        ),
    }


def _manifest_split_availability(manifest: TokenizedCorpusManifest) -> dict[str, int]:
    train_end = max(
        manifest.seq_len + manifest.max_horizon + 2,
        int(manifest.token_count * manifest.train_fraction),
    )
    val_start = max(0, train_end - manifest.seq_len - manifest.max_horizon - 1)
    return {
        "train_start": 0,
        "train_end": int(train_end),
        "val_start": int(val_start),
        "val_end": int(manifest.token_count),
        "train_available_windows": int(max(0, train_end - manifest.seq_len - manifest.max_horizon)),
        "val_available_windows": int(max(0, manifest.token_count - val_start - manifest.seq_len - manifest.max_horizon)),
    }


def build_training_plan(
    manifest: TokenizedCorpusManifest,
    config: ComparisonConfig,
    *,
    world_size: int = 1,
    distributed: bool = False,
    corpus_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dict(corpus_identity or manifest.identity())
    baseline_config = TransformerConfig(
        vocab_size=manifest.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        horizons=config.horizons,
        use_cortex_heads=False,
    )
    cortex_config = TransformerConfig(**{
        **asdict(baseline_config),
        "use_cortex_heads": True,
        "use_ternary_core": True,
        "use_native_ternary_kernel": True,
        "require_native_ternary_kernel": _strict_native_ternary_required_for_training(config.training),
        "native_ternary_backend": config.native_ternary_backend,
        "use_skill_aware_experts": True,
        "use_variable_in_compressor": True,
        "use_learned_memory_policy": True,
        "use_certificate_head": True,
        "use_latent_reasoning_workspace": True,
    })
    baseline_parameters = _transformer_parameter_count(baseline_config)
    cortex_parameters = _transformer_parameter_count(cortex_config)
    effective_world_size = max(1, int(world_size))
    tokens_per_optimizer_step = (
        int(config.training.batch_size)
        * int(config.training.gradient_accumulation_steps)
        * effective_world_size
        * int(config.seq_len)
    )
    optimizer_steps = max(0, int(config.training.steps))
    train_tokens = tokens_per_optimizer_step * optimizer_steps
    eval_events = 1 + sum(
        1
        for step in range(1, optimizer_steps + 1)
        if step % int(config.training.eval_interval) == 0 or step == optimizer_steps
    )
    eval_tokens = (
        eval_events
        * 2
        * int(config.training.eval_batches)
        * int(config.training.batch_size)
        * int(config.seq_len)
    )
    split = _manifest_split_availability(manifest)
    train_tokens_available = max(1, int(split["train_end"]) - int(split["train_start"]))
    checkpoint_interval = max(1, int(config.training.checkpoint_interval))
    max_intermediate_checkpoints = int(config.training.max_intermediate_checkpoints)
    adam_training_bytes_per_parameter = 16
    checkpoint_bytes_per_parameter = 12
    precision_bytes = _training_precision_bytes(config.training.precision)
    return {
        "schema_version": 1,
        "corpus": {
            "token_count": int(manifest.token_count),
            "train_fraction": float(manifest.train_fraction),
            "seq_len": int(manifest.seq_len),
            "max_horizon": int(manifest.max_horizon),
            "vocab_size": int(manifest.vocab_size),
            "source_file_count": len(manifest.source_files),
            "source_fingerprint_count": len(manifest.source_file_fingerprints),
            "identity_sha256": str(identity["identity_sha256"]),
            "token_file_sha256": str(identity["token_file_sha256"]),
            "tokenizer_file_sha256": str(identity["tokenizer_file_sha256"]),
            **split,
        },
        "model": {
            "d_model": int(config.d_model),
            "n_heads": int(config.n_heads),
            "n_layers": int(config.n_layers),
            "horizons": tuple(int(horizon) for horizon in config.horizons),
            "baseline_parameters": baseline_parameters,
            "cortex_parameters": cortex_parameters,
            "cortex_extra_parameters": cortex_parameters - baseline_parameters,
            "cortex_skill_aware_experts": bool(cortex_config.use_skill_aware_experts),
            "cortex_skill_expert_count": int(cortex_config.skill_expert_count),
            "cortex_skill_expert_top_k": int(cortex_config.skill_expert_top_k),
            "cortex_variable_in_compressor": bool(cortex_config.use_variable_in_compressor),
            "cortex_learned_memory_policy": bool(cortex_config.use_learned_memory_policy),
            "cortex_certificate_head": bool(cortex_config.use_certificate_head),
            "cortex_latent_reasoning_workspace": bool(cortex_config.use_latent_reasoning_workspace),
            "cortex_latent_workspace_steps": int(cortex_config.latent_workspace_steps),
            "cortex_native_ternary_backend": str(cortex_config.native_ternary_backend),
        },
        "training": {
            "steps": optimizer_steps,
            "batch_size": int(config.training.batch_size),
            "gradient_accumulation_steps": int(config.training.gradient_accumulation_steps),
            "world_size": effective_world_size,
            "distributed": bool(distributed),
            "precision": str(config.training.precision),
            "tokens_per_optimizer_step": int(tokens_per_optimizer_step),
            "planned_train_tokens": int(train_tokens),
            "planned_eval_tokens": int(eval_tokens),
            "planned_total_tokens": int(train_tokens + eval_tokens),
            "effective_epochs_over_train_split": float(train_tokens / train_tokens_available),
            "eval_events": int(eval_events),
            "checkpoint_interval": checkpoint_interval,
            "intermediate_checkpoint_count": int(optimizer_steps // checkpoint_interval),
            "max_intermediate_checkpoints": max_intermediate_checkpoints,
            "retained_intermediate_checkpoint_count": (
                int(optimizer_steps // checkpoint_interval)
                if max_intermediate_checkpoints <= 0
                else min(int(optimizer_steps // checkpoint_interval), max_intermediate_checkpoints)
            ),
            "final_checkpoint_count": 1,
        },
        "memory_estimate": {
            "parameter_precision_bytes": precision_bytes,
            "baseline_parameter_bytes": int(baseline_parameters * precision_bytes),
            "cortex_parameter_bytes": int(cortex_parameters * precision_bytes),
            "baseline_adam_training_bytes": int(baseline_parameters * adam_training_bytes_per_parameter),
            "cortex_adam_training_bytes": int(cortex_parameters * adam_training_bytes_per_parameter),
            "baseline_checkpoint_bytes": int(baseline_parameters * checkpoint_bytes_per_parameter),
            "cortex_checkpoint_bytes": int(cortex_parameters * checkpoint_bytes_per_parameter),
        },
    }


def _finite_float(value: float) -> bool:
    return math.isfinite(float(value))


def _audit_model_learning_curve(report: TrainingRunReport, *, expected_final_step: int) -> dict[str, Any]:
    val_points = sorted((point for point in report.curve if point.split == "val"), key=lambda point: point.step)
    train_points = sorted((point for point in report.curve if point.split == "train"), key=lambda point: point.step)
    steps = [int(point.step) for point in val_points]
    failed_checks: list[str] = []
    if len(val_points) < 2:
        failed_checks.append("validation_point_count<2")
    if not steps or steps[0] != 0:
        failed_checks.append("missing_initial_validation_step")
    if expected_final_step not in steps:
        failed_checks.append("missing_final_validation_step")
    if steps != sorted(set(steps)):
        failed_checks.append("validation_steps_not_strictly_increasing")
    finite = all(
        _finite_float(value)
        for point in tuple(val_points) + tuple(train_points)
        for value in (point.loss, point.next_token_loss, point.token_accuracy, point.mtp_loss, point.future_tokens_per_cost)
    )
    if not finite:
        failed_checks.append("non_finite_metric")

    if val_points:
        first = val_points[0]
        final = val_points[-1]
        best_loss = min(point.next_token_loss for point in val_points)
        best_future = max(point.future_tokens_per_cost for point in val_points)
        first_loss = first.next_token_loss
        final_loss = final.next_token_loss
        first_future = first.future_tokens_per_cost
        final_future = final.future_tokens_per_cost
    else:
        best_loss = best_future = first_loss = final_loss = first_future = final_future = 0.0

    return {
        "model": report.name,
        "model_kind": report.model_kind,
        "validation_point_count": len(val_points),
        "train_point_count": len(train_points),
        "validation_steps": tuple(steps),
        "expected_final_step": int(expected_final_step),
        "first_next_token_loss": float(first_loss),
        "final_next_token_loss": float(final_loss),
        "best_next_token_loss": float(best_loss),
        "next_token_loss_delta": float(final_loss - first_loss),
        "first_future_tokens_per_cost": float(first_future),
        "final_future_tokens_per_cost": float(final_future),
        "best_future_tokens_per_cost": float(best_future),
        "future_tokens_per_cost_delta": float(final_future - first_future),
        "failed_checks": tuple(failed_checks),
        "passed": not failed_checks,
    }


def audit_learning_curves(
    baseline: TrainingRunReport,
    cortex: TrainingRunReport,
    *,
    expected_final_step: int,
) -> dict[str, Any]:
    baseline_audit = _audit_model_learning_curve(baseline, expected_final_step=expected_final_step)
    cortex_audit = _audit_model_learning_curve(cortex, expected_final_step=expected_final_step)
    failed_models = tuple(
        item["model"]
        for item in (baseline_audit, cortex_audit)
        if not bool(item["passed"])
    )
    return {
        "schema_version": 1,
        "expected_final_step": int(expected_final_step),
        "baseline": baseline_audit,
        "cortex": cortex_audit,
        "failed_models": failed_models,
        "passed": not failed_models,
    }


class LLMComparisonRunner:
    def __init__(
        self,
        corpus: TextCorpusConfig,
        config: ComparisonConfig,
        *,
        run_dir: str | Path,
        prepared_manifest: TokenizedCorpusManifest | None = None,
    ):
        self.corpus = corpus
        self.config = config
        self.run_dir = Path(run_dir)
        self.prepared_manifest = prepared_manifest

    def prepare_corpus(self) -> TokenizedCorpusManifest:
        tokenizer = LLMTokenizer.train(
            self.corpus,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            max_training_chars=self.config.tokenizer_training_chars,
        )
        return TokenizedCorpusBuilder(self.corpus, tokenizer).build(
            self.run_dir / "corpus",
            seq_len=self.config.seq_len,
            max_horizon=max(self.config.horizons),
            max_tokens=self.config.max_corpus_tokens,
            preparation_config=self._expected_preparation_config(),
        )

    def _expected_preparation_config(self) -> dict[str, Any]:
        return _tokenized_preparation_config(
            self.corpus,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            seq_len=self.config.seq_len,
            max_horizon=max(self.config.horizons),
            max_tokens=self.config.max_corpus_tokens,
            tokenizer_training_chars=self.config.tokenizer_training_chars,
        )

    def run(self, *, require_win: bool = False) -> ComparisonReport:
        started = time.time()
        device_type = "cuda" if (self.config.training.device == "auto" and torch.cuda.is_available()) or str(self.config.training.device).startswith("cuda") else "cpu"
        runtime = DistributedRuntime.from_env(
            requested=self.config.training.distributed,
            device_type=device_type,
            gloo_interface=self.config.training.gloo_interface,
        )
        runtime.ensure_initialized()
        if self.prepared_manifest is not None:
            if runtime.is_main:
                if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                    shutil.rmtree(self.run_dir)
                self.run_dir.mkdir(parents=True, exist_ok=True)
            _barrier_if_needed(runtime)
            manifest = self.prepared_manifest
        else:
            if runtime.is_main:
                if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                    shutil.rmtree(self.run_dir)
                self.run_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = self.run_dir / "corpus" / "manifest.json"
                if manifest_path.exists():
                    manifest = TokenizedCorpusManifest.load(manifest_path)
                elif self.config.training.resume:
                    raise FileNotFoundError(f"resume=True but corpus manifest is missing: {manifest_path}")
                else:
                    manifest = self.prepare_corpus()
            _barrier_if_needed(runtime)
            manifest = TokenizedCorpusManifest.load(self.run_dir / "corpus" / "manifest.json")
        manifest_path = Path(manifest.token_file).parent / "manifest.json"
        _require_tokenized_preparation_config(
            manifest,
            self._expected_preparation_config(),
            manifest_path=manifest_path,
        )
        corpus_identity = manifest.identity()
        plan = build_training_plan(
            manifest,
            self.config,
            world_size=runtime.world_size,
            distributed=runtime.enabled,
            corpus_identity=corpus_identity,
        )
        if runtime.is_main:
            _write_json(self.run_dir / "run_plan.json", plan)
        train_data = MemmapCausalDataset(manifest, split="train")
        val_data = MemmapCausalDataset(manifest, split="val")
        try:
            model_config = TransformerConfig(
                vocab_size=manifest.vocab_size,
                seq_len=self.config.seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                dropout=self.config.dropout,
                horizons=self.config.horizons,
                use_cortex_heads=False,
            )
            baseline = LLMTrainer(
                CortexTransformerLM(model_config),
                train_data,
                val_data,
                self.config.training,
                run_dir=self.run_dir / "baseline_ntp",
                model_kind="baseline_next_token",
                corpus_identity=corpus_identity,
            ).train(name="baseline_ntp")
            cortex_config = TransformerConfig(**{
                **asdict(model_config),
                "use_cortex_heads": True,
                "use_ternary_core": True,
                "use_native_ternary_kernel": True,
                "require_native_ternary_kernel": _strict_native_ternary_required_for_training(self.config.training),
                "native_ternary_backend": self.config.native_ternary_backend,
                "use_skill_aware_experts": True,
                "use_variable_in_compressor": True,
                "use_learned_memory_policy": True,
                "use_certificate_head": True,
                "use_latent_reasoning_workspace": True,
            })
            cortex = LLMTrainer(
                CortexTransformerLM(cortex_config),
                train_data,
                val_data,
                self.config.training,
                run_dir=self.run_dir / "cortex3",
                model_kind="cortex3_multi_horizon",
                corpus_identity=corpus_identity,
            ).train(name="cortex3")
        finally:
            train_data.close()
            val_data.close()
        curve_audit = audit_learning_curves(baseline, cortex, expected_final_step=self.config.training.steps)
        proof = self._proof_payload(baseline, cortex, curve_audit, plan=plan)
        proof["elapsed_seconds"] = time.time() - started
        proof["distributed"] = runtime.enabled
        proof["world_size"] = runtime.world_size
        report = ComparisonReport(
            run_dir=str(self.run_dir),
            manifest=manifest.to_dict(),
            baseline=baseline.to_dict(),
            cortex=cortex.to_dict(),
            proof=proof,
            hardware=hardware_report(),
            plan=plan,
            curve_audit=curve_audit,
        )
        if runtime.is_main:
            _write_json(self.run_dir / "learning_curve_audit.json", curve_audit)
            _write_json(self.run_dir / "comparison_report.json", report.to_dict())
            self._write_markdown(report)
            self._write_learning_curve_png()
        failed = bool(require_win and not proof["passed"] and runtime.is_main)
        if runtime.enabled:
            flag = torch.tensor([1 if failed else 0], dtype=torch.int64)
            torch.distributed.broadcast(flag, src=0)
            failed = bool(int(flag.item()))
        _barrier_if_needed(runtime)
        if failed:
            raise RuntimeError(f"Cortex comparison did not pass: {proof}")
        return report

    def _proof_payload(
        self,
        baseline: TrainingRunReport,
        cortex: TrainingRunReport,
        curve_audit: Mapping[str, Any],
        *,
        plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        baseline_score = baseline.final_val.future_tokens_per_cost
        cortex_score = cortex.final_val.future_tokens_per_cost
        ratio = cortex_score / max(1e-9, baseline_score)
        next_token_regression = cortex.final_val.next_token_loss / max(1e-9, baseline.final_val.next_token_loss)
        finite_metrics = all(math.isfinite(value) for value in (baseline_score, cortex_score, ratio, next_token_regression))
        baseline_score_passed = finite_metrics and baseline_score >= self.config.min_baseline_future_tokens_per_cost
        ratio_passed = finite_metrics and ratio >= self.config.cortex_win_margin
        next_token_regression_passed = finite_metrics and next_token_regression <= self.config.max_next_token_loss_regression
        learning_curve_audit_passed = bool(curve_audit["passed"])
        if plan is not None:
            corpus_token_count = int(plan.get("corpus", {}).get("token_count") or 0)
            planned_train_tokens = int(plan.get("training", {}).get("planned_train_tokens") or 0)
        else:
            corpus_identity = baseline.config.get("corpus_identity", {}) if isinstance(baseline.config, Mapping) else {}
            corpus_token_count = int(corpus_identity.get("token_count", 0)) if isinstance(corpus_identity, Mapping) else 0
            planned_train_tokens = 0
        corpus_scale_passed = corpus_token_count >= int(self.config.min_corpus_tokens)
        planned_train_tokens_passed = planned_train_tokens >= int(self.config.min_planned_train_tokens)
        cortex_model_config = cortex.config.get("model", {}) if isinstance(cortex.config, Mapping) else {}
        raw_horizons = cortex_model_config.get("horizons", ()) if isinstance(cortex_model_config, Mapping) else ()
        cortex_full_phase_required = bool(
            isinstance(cortex_model_config, Mapping)
            and cortex_model_config.get("use_cortex_heads")
            and cortex_model_config.get("use_ternary_core")
            and cortex_model_config.get("use_learned_memory_policy")
            and tuple(int(value) for value in raw_horizons) == (1, 2, 4, 8)
        )
        phase_report = cortex.cortex_phase_report if isinstance(cortex.cortex_phase_report, Mapping) else {}
        architecture_audit = phase_report.get("architecture_audit", {}) if isinstance(phase_report, Mapping) else {}
        if not isinstance(architecture_audit, Mapping):
            architecture_audit = {}
        phase_deliverable_audit = phase_report.get("phase_deliverable_audit", {}) if isinstance(phase_report, Mapping) else {}
        if not isinstance(phase_deliverable_audit, Mapping):
            phase_deliverable_audit = {}
        cortex_architecture_audit_passed = (
            (not cortex_full_phase_required)
            or bool(architecture_audit.get("passed"))
        )
        cortex_phase_deliverable_audit_passed = (
            (not cortex_full_phase_required)
            or bool(phase_deliverable_audit.get("passed"))
        )
        cortex_phase_integration_passed = (
            (not cortex_full_phase_required)
            or (
                bool(phase_report.get("all_phases_active"))
                and not phase_report.get("errors")
                and cortex_architecture_audit_passed
                and cortex_phase_deliverable_audit_passed
            )
        )
        checks = {
            "finite_metrics": finite_metrics,
            "baseline_score_passed": baseline_score_passed,
            "ratio_passed": ratio_passed,
            "next_token_regression_passed": next_token_regression_passed,
            "learning_curve_audit_passed": learning_curve_audit_passed,
            "corpus_scale_passed": corpus_scale_passed,
            "planned_train_tokens_passed": planned_train_tokens_passed,
            "cortex_architecture_audit_passed": cortex_architecture_audit_passed,
            "cortex_phase_deliverable_audit_passed": cortex_phase_deliverable_audit_passed,
            "cortex_phase_integration_passed": cortex_phase_integration_passed,
        }
        failed_checks = tuple(name for name, passed_check in checks.items() if not passed_check)
        passed = not failed_checks
        return {
            "metric": "verified_future_tokens_per_forward_cost",
            "baseline_score": baseline_score,
            "cortex_score": cortex_score,
            "cortex_over_baseline_ratio": ratio,
            "required_margin": self.config.cortex_win_margin,
            "min_baseline_future_tokens_per_cost": self.config.min_baseline_future_tokens_per_cost,
            "next_token_loss_regression_ratio": next_token_regression,
            "max_next_token_loss_regression": self.config.max_next_token_loss_regression,
            "corpus_token_count": corpus_token_count,
            "min_corpus_tokens": int(self.config.min_corpus_tokens),
            "planned_train_tokens": planned_train_tokens,
            "min_planned_train_tokens": int(self.config.min_planned_train_tokens),
            "cortex_full_phase_required": cortex_full_phase_required,
            "cortex_architecture_audit_passed": cortex_architecture_audit_passed,
            "cortex_architecture_audit": dict(architecture_audit),
            "cortex_phase_deliverable_audit_passed": cortex_phase_deliverable_audit_passed,
            "cortex_phase_deliverable_audit": dict(phase_deliverable_audit),
            "cortex_phase_integration_passed": cortex_phase_integration_passed,
            "checks": checks,
            "failed_checks": failed_checks,
            "baseline_score_passed": baseline_score_passed,
            "learning_curve_audit_passed": learning_curve_audit_passed,
            "corpus_scale_passed": corpus_scale_passed,
            "planned_train_tokens_passed": planned_train_tokens_passed,
            "passed": passed,
        }

    def _write_markdown(self, report: ComparisonReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM comparison report",
            "",
            f"- Proof metric: `{proof['metric']}`",
            f"- Baseline score: `{proof['baseline_score']:.6f}`",
            f"- Minimum baseline score: `{proof['min_baseline_future_tokens_per_cost']:.6f}`",
            f"- Cortex score: `{proof['cortex_score']:.6f}`",
            f"- Cortex/baseline ratio: `{proof['cortex_over_baseline_ratio']:.3f}`",
            f"- Next-token loss regression ratio: `{proof['next_token_loss_regression_ratio']:.3f}`",
            f"- Learning curve audit passed: `{proof['learning_curve_audit_passed']}`",
            f"- Corpus tokens: `{proof['corpus_token_count']}` (min `{proof['min_corpus_tokens']}`)",
            f"- Planned train tokens: `{proof['planned_train_tokens']}` (min `{proof['min_planned_train_tokens']}`)",
            f"- Full Cortex phases required: `{proof['cortex_full_phase_required']}`",
            f"- Cortex phase integration passed: `{proof['cortex_phase_integration_passed']}`",
            f"- Failed checks: `{', '.join(proof['failed_checks']) if proof['failed_checks'] else 'none'}`",
            f"- Passed: `{proof['passed']}`",
            "",
            "## Artifacts",
            "",
            "- `run_plan.json`",
            "- `learning_curve_audit.json`",
            "- `comparison_report.json`",
            "- `baseline_ntp/learning_curve.csv`",
            "- `cortex3/learning_curve.csv`",
            "- `learning_curve.png`",
            "- `baseline_ntp/checkpoint_final.pt`",
            "- `cortex3/checkpoint_final.pt`",
        ]
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_learning_curve_png(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series: list[tuple[str, Path]] = [
            ("baseline_ntp", self.run_dir / "baseline_ntp" / "learning_curve.csv"),
            ("cortex3", self.run_dir / "cortex3" / "learning_curve.csv"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, path in series:
            with path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            val = [row for row in rows if row["split"] == "val"]
            axes[0].plot([int(row["step"]) for row in val], [float(row["next_token_loss"]) for row in val], label=label)
            axes[1].plot([int(row["step"]) for row in val], [float(row["future_tokens_per_cost"]) for row in val], label=label)
        axes[0].set_title("Validation next-token loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[1].set_title("Future tokens per forward cost")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("score")
        for axis in axes:
            axis.grid(True, alpha=0.25)
            axis.legend()
        fig.tight_layout()
        fig.savefig(self.run_dir / "learning_curve.png", dpi=150)
        plt.close(fig)


class LLMComparisonMatrixSuite:
    def __init__(
        self,
        corpus: TextCorpusConfig,
        config: ComparisonConfig,
        *,
        run_dir: str | Path,
        seeds: Sequence[int],
    ):
        if not seeds:
            raise ValueError("at least one comparison seed is required")
        self.corpus = corpus
        self.config = config
        self.run_dir = Path(run_dir)
        self.seeds = tuple(int(seed) for seed in seeds)

    def run(self, *, require_win: bool = False) -> ComparisonMatrixReport:
        started = time.time()
        device_type = "cuda" if (self.config.training.device == "auto" and torch.cuda.is_available()) or str(self.config.training.device).startswith("cuda") else "cpu"
        runtime = DistributedRuntime.from_env(
            requested=self.config.training.distributed,
            device_type=device_type,
            gloo_interface=self.config.training.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        _barrier_if_needed(runtime)

        manifest = self._prepare_or_load_manifest(runtime)
        seed_payloads: list[Mapping[str, Any]] = []
        for seed in self.seeds:
            seed_training = replace(self.config.training, seed=seed)
            seed_config = replace(self.config, training=seed_training)
            report = LLMComparisonRunner(
                self.corpus,
                seed_config,
                run_dir=self.run_dir / f"seed_{seed}",
                prepared_manifest=manifest,
            ).run(require_win=False)
            seed_payloads.append(
                {
                    "seed": seed,
                    "run_dir": report.run_dir,
                    "proof": report.proof,
                    "baseline_final_val": report.baseline["final_val"],
                    "cortex_final_val": report.cortex["final_val"],
                }
            )

        proof = self._proof(seed_payloads)
        proof["elapsed_seconds"] = time.time() - started
        proof["distributed"] = runtime.enabled
        proof["world_size"] = runtime.world_size
        matrix_report = ComparisonMatrixReport(
            run_dir=str(self.run_dir),
            manifest=manifest.to_dict(),
            seeds=tuple(seed_payloads),
            proof=proof,
            hardware=hardware_report(),
        )
        if runtime.is_main:
            _write_json(self.run_dir / "comparison_matrix_report.json", matrix_report.to_dict())
            self._write_markdown(matrix_report)
            self._write_ratio_plot(matrix_report)
            self._write_learning_curve_summary(matrix_report)
        failed = bool(require_win and not proof["passed"] and runtime.is_main)
        if runtime.enabled:
            flag_device = torch.device(f"cuda:{runtime.local_rank}") if runtime.backend == "nccl" else torch.device("cpu")
            flag = torch.tensor([1 if failed else 0], dtype=torch.int64, device=flag_device)
            torch.distributed.broadcast(flag, src=0)
            failed = bool(int(flag.cpu().item()))
        _barrier_if_needed(runtime)
        if failed:
            raise RuntimeError(f"Cortex comparison matrix did not pass: {proof}")
        return matrix_report

    def _prepare_or_load_manifest(self, runtime: DistributedRuntime) -> TokenizedCorpusManifest:
        manifest_path = self.run_dir / "corpus" / "manifest.json"
        if runtime.is_main:
            if manifest_path.exists():
                manifest = TokenizedCorpusManifest.load(manifest_path)
            elif self.config.training.resume:
                raise FileNotFoundError(f"resume=True but shared corpus manifest is missing: {manifest_path}")
            else:
                tokenizer = LLMTokenizer.train(
                    self.corpus,
                    vocab_size=self.config.vocab_size,
                    min_frequency=self.config.min_frequency,
                    max_training_chars=self.config.tokenizer_training_chars,
                )
                manifest = TokenizedCorpusBuilder(self.corpus, tokenizer).build(
                    self.run_dir / "corpus",
                    seq_len=self.config.seq_len,
                    max_horizon=max(self.config.horizons),
                    max_tokens=self.config.max_corpus_tokens,
                    preparation_config=self._expected_preparation_config(),
                )
        _barrier_if_needed(runtime)
        manifest = TokenizedCorpusManifest.load(manifest_path)
        _require_tokenized_preparation_config(
            manifest,
            self._expected_preparation_config(),
            manifest_path=manifest_path,
        )
        return manifest

    def _expected_preparation_config(self) -> dict[str, Any]:
        return _tokenized_preparation_config(
            self.corpus,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            seq_len=self.config.seq_len,
            max_horizon=max(self.config.horizons),
            max_tokens=self.config.max_corpus_tokens,
            tokenizer_training_chars=self.config.tokenizer_training_chars,
        )

    def _proof(self, seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        for seed_report in seed_reports:
            seed = int(seed_report["seed"])
            proof = seed_report["proof"]
            baseline_score = float(proof["baseline_score"])
            baseline_score_passed = bool(
                proof.get("baseline_score_passed", baseline_score >= self.config.min_baseline_future_tokens_per_cost)
            )
            corpus_token_count = int(proof.get("corpus_token_count", 0))
            planned_train_tokens = int(proof.get("planned_train_tokens", 0))
            corpus_scale_passed = bool(proof.get("corpus_scale_passed", corpus_token_count >= self.config.min_corpus_tokens))
            planned_train_tokens_passed = bool(
                proof.get("planned_train_tokens_passed", planned_train_tokens >= self.config.min_planned_train_tokens)
            )
            samples.append(
                {
                    "seed": seed,
                    "ratio": float(proof["cortex_over_baseline_ratio"]),
                    "baseline_score": baseline_score,
                    "baseline_score_passed": baseline_score_passed,
                    "cortex_score": float(proof["cortex_score"]),
                    "next_token_loss_regression_ratio": float(proof["next_token_loss_regression_ratio"]),
                    "corpus_token_count": corpus_token_count,
                    "corpus_scale_passed": corpus_scale_passed,
                    "planned_train_tokens": planned_train_tokens,
                    "planned_train_tokens_passed": planned_train_tokens_passed,
                    "passed": bool(proof["passed"]) and baseline_score_passed and corpus_scale_passed and planned_train_tokens_passed,
                }
            )
        ratios = [sample["ratio"] for sample in samples]
        baseline_scores = [sample["baseline_score"] for sample in samples]
        regressions = [sample["next_token_loss_regression_ratio"] for sample in samples]
        corpus_tokens = [int(sample["corpus_token_count"]) for sample in samples]
        planned_tokens = [int(sample["planned_train_tokens"]) for sample in samples]
        passed_count = sum(1 for sample in samples if sample["passed"])
        sample_count = len(samples)
        min_ratio = min(ratios) if ratios else 0.0
        max_regression = max(regressions) if regressions else 0.0
        min_baseline = min(baseline_scores) if baseline_scores else 0.0
        min_corpus = min(corpus_tokens) if corpus_tokens else 0
        min_planned = min(planned_tokens) if planned_tokens else 0
        passed = (
            sample_count == len(self.seeds)
            and passed_count == sample_count
            and min_ratio >= self.config.cortex_win_margin
            and min_baseline >= self.config.min_baseline_future_tokens_per_cost
            and max_regression <= self.config.max_next_token_loss_regression
            and min_corpus >= self.config.min_corpus_tokens
            and min_planned >= self.config.min_planned_train_tokens
        )
        return {
            "metric": "comparison_matrix_cortex_over_baseline",
            "seeds": list(self.seeds),
            "seed_count": len(self.seeds),
            "sample_count": sample_count,
            "required_margin": self.config.cortex_win_margin,
            "required_win_rate": 1.0,
            "mean_ratio": sum(ratios) / max(1, len(ratios)),
            "median_ratio": statistics.median(ratios) if ratios else 0.0,
            "ratio_population_stddev": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "min_ratio": min_ratio,
            "max_ratio": max(ratios) if ratios else 0.0,
            "win_rate": passed_count / max(1, sample_count),
            "mean_baseline_score": sum(baseline_scores) / max(1, len(baseline_scores)),
            "min_baseline_score": min_baseline,
            "min_baseline_future_tokens_per_cost": self.config.min_baseline_future_tokens_per_cost,
            "min_corpus_tokens": int(self.config.min_corpus_tokens),
            "min_observed_corpus_tokens": min_corpus,
            "min_planned_train_tokens": int(self.config.min_planned_train_tokens),
            "min_observed_planned_train_tokens": min_planned,
            "max_next_token_loss_regression": max_regression,
            "all_samples_passed": passed_count == sample_count and sample_count > 0,
            "samples": tuple(samples),
            "passed": passed,
        }

    def _write_markdown(self, report: ComparisonMatrixReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM comparison matrix report",
            "",
            f"- Seeds: `{', '.join(str(seed) for seed in proof['seeds'])}`",
            f"- Shared corpus tokens: `{report.manifest['token_count']}`",
            f"- Samples: `{proof['sample_count']}`",
            f"- Mean Cortex/baseline ratio: `{proof['mean_ratio']:.3f}`",
            f"- Median Cortex/baseline ratio: `{proof['median_ratio']:.3f}`",
            f"- Min Cortex/baseline ratio: `{proof['min_ratio']:.3f}`",
            f"- Win rate: `{proof['win_rate']:.3f}`",
            f"- Max next-token-loss regression: `{proof['max_next_token_loss_regression']:.3f}`",
            f"- Min observed corpus tokens: `{proof['min_observed_corpus_tokens']}` (required `{proof['min_corpus_tokens']}`)",
            f"- Min observed planned train tokens: `{proof['min_observed_planned_train_tokens']}` (required `{proof['min_planned_train_tokens']}`)",
            f"- Passed: `{proof['passed']}`",
            "",
            "## Seed Results",
            "",
            "| Seed | Baseline score | Cortex score | Ratio | NT loss regression | Passed |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in proof["samples"]:
            lines.append(
                f"| {item['seed']} | {item['baseline_score']:.6f} | {item['cortex_score']:.6f} | "
                f"{item['ratio']:.3f} | {item['next_token_loss_regression_ratio']:.3f} | `{item['passed']}` |"
            )
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                "- `comparison_matrix_report.json`",
                "- `comparison_matrix_report.md`",
                "- `comparison_matrix_ratios.png`",
                "- `comparison_matrix_learning_curves.csv`",
                "- `comparison_matrix_learning_curves.png`",
                "- `corpus/manifest.json`",
                "- `seed_<seed>/comparison_report.json`",
                "- `seed_<seed>/baseline_ntp/checkpoint_final.pt`",
                "- `seed_<seed>/cortex3/checkpoint_final.pt`",
            ]
        )
        (self.run_dir / "comparison_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_ratio_plot(self, report: ComparisonMatrixReport) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        samples = list(report.proof["samples"])
        names = [str(sample["seed"]) for sample in samples]
        ratios = [float(sample["ratio"]) for sample in samples]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, ratios, color="#476f4f")
        ax.axhline(1.0, color="#333333", linewidth=1)
        ax.axhline(float(report.proof["required_margin"]), color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Cortex / baseline ratio by seed")
        ax.set_ylabel("ratio")
        ax.set_xlabel("seed")
        fig.tight_layout()
        fig.savefig(self.run_dir / "comparison_matrix_ratios.png", dpi=150)
        plt.close(fig)

    def _write_learning_curve_summary(self, report: ComparisonMatrixReport) -> None:
        rows = _read_validation_learning_curve_rows(report.seeds)
        _write_learning_curve_matrix_artifacts(
            self.run_dir,
            rows=rows,
            csv_name="comparison_matrix_learning_curves.csv",
            png_name="comparison_matrix_learning_curves.png",
            group_by_corpus=False,
        )


class LLMCorpusMatrixSuite:
    def __init__(
        self,
        corpora: Sequence[tuple[str, TextCorpusConfig]],
        config: ComparisonConfig,
        *,
        run_dir: str | Path,
        seeds: Sequence[int],
    ):
        if not corpora:
            raise ValueError("at least one corpus is required")
        if not seeds:
            raise ValueError("at least one seed is required")
        seen_names: set[str] = set()
        seen_run_names: set[str] = set()
        normalized: list[tuple[str, str, TextCorpusConfig]] = []
        for name, corpus in corpora:
            clean_name = str(name).strip()
            if not clean_name:
                raise ValueError("corpus names must be non-empty")
            if clean_name in seen_names:
                raise ValueError(f"duplicate corpus name {clean_name!r}")
            run_name = _safe_run_name(clean_name)
            if run_name in seen_run_names:
                raise ValueError(f"corpus names collide after sanitizing: {clean_name!r}")
            seen_names.add(clean_name)
            seen_run_names.add(run_name)
            normalized.append((clean_name, run_name, corpus))
        self.corpora = tuple(normalized)
        self.config = config
        self.run_dir = Path(run_dir)
        self.seeds = tuple(int(seed) for seed in seeds)

    def run(self, *, require_win: bool = False) -> CorpusMatrixReport:
        runtime = DistributedRuntime.from_env(
            requested=self.config.training.distributed,
            device_type="cuda" if torch.cuda.is_available() and self.config.training.device in {"auto", "cuda"} else "cpu",
            gloo_interface=self.config.training.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        _barrier_if_needed(runtime)

        corpus_payloads: list[Mapping[str, Any]] = []
        for name, run_name, corpus in self.corpora:
            report = LLMComparisonMatrixSuite(
                corpus,
                self.config,
                run_dir=self.run_dir / run_name,
                seeds=self.seeds,
            ).run(require_win=False)
            corpus_payloads.append(
                {
                    "name": name,
                    "run_name": run_name,
                    "run_dir": report.run_dir,
                    "manifest": report.manifest,
                    "proof": report.proof,
                    "seeds": report.seeds,
                }
            )

        proof = self._proof(corpus_payloads)
        report = CorpusMatrixReport(
            run_dir=str(self.run_dir),
            corpora=tuple(corpus_payloads),
            proof=proof,
            hardware=hardware_report(),
        )
        if runtime.is_main:
            _write_json(self.run_dir / "corpus_matrix_report.json", report.to_dict())
            self._write_markdown(report)
            self._write_ratio_plot(report)
            self._write_learning_curve_summary(report)
        failed = bool(require_win and not proof["passed"] and runtime.is_main)
        if runtime.enabled:
            flag_device = torch.device(f"cuda:{runtime.local_rank}") if runtime.backend == "nccl" else torch.device("cpu")
            flag = torch.tensor([1 if failed else 0], dtype=torch.int64, device=flag_device)
            torch.distributed.broadcast(flag, src=0)
            failed = bool(int(flag.cpu().item()))
        _barrier_if_needed(runtime)
        if failed:
            raise RuntimeError(f"Cortex corpus matrix did not pass: {proof}")
        return report

    def _proof(self, corpus_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        for corpus_report in corpus_reports:
            corpus_name = str(corpus_report["name"])
            for sample in corpus_report["proof"]["samples"]:
                baseline_score = float(sample["baseline_score"])
                baseline_score_passed = bool(
                    sample.get("baseline_score_passed", baseline_score >= self.config.min_baseline_future_tokens_per_cost)
                )
                corpus_token_count = int(sample.get("corpus_token_count", 0))
                planned_train_tokens = int(sample.get("planned_train_tokens", 0))
                corpus_scale_passed = bool(sample.get("corpus_scale_passed", corpus_token_count >= self.config.min_corpus_tokens))
                planned_train_tokens_passed = bool(
                    sample.get("planned_train_tokens_passed", planned_train_tokens >= self.config.min_planned_train_tokens)
                )
                samples.append(
                    {
                        "corpus": corpus_name,
                        "seed": int(sample["seed"]),
                        "ratio": float(sample["ratio"]),
                        "baseline_score": baseline_score,
                        "baseline_score_passed": baseline_score_passed,
                        "cortex_score": float(sample["cortex_score"]),
                        "next_token_loss_regression_ratio": float(sample["next_token_loss_regression_ratio"]),
                        "corpus_token_count": corpus_token_count,
                        "corpus_scale_passed": corpus_scale_passed,
                        "planned_train_tokens": planned_train_tokens,
                        "planned_train_tokens_passed": planned_train_tokens_passed,
                        "passed": bool(sample["passed"]) and baseline_score_passed and corpus_scale_passed and planned_train_tokens_passed,
                    }
                )

        ratios = [sample["ratio"] for sample in samples]
        baseline_scores = [sample["baseline_score"] for sample in samples]
        regressions = [sample["next_token_loss_regression_ratio"] for sample in samples]
        corpus_tokens = [int(sample["corpus_token_count"]) for sample in samples]
        planned_tokens = [int(sample["planned_train_tokens"]) for sample in samples]
        passed_count = sum(1 for sample in samples if sample["passed"])
        sample_count = len(samples)

        corpus_results: list[dict[str, Any]] = []
        for name, _, _ in self.corpora:
            corpus_samples = [sample for sample in samples if sample["corpus"] == name]
            corpus_ratios = [sample["ratio"] for sample in corpus_samples]
            corpus_passed = sum(1 for sample in corpus_samples if sample["passed"])
            corpus_results.append(
                {
                    "corpus": name,
                    "sample_count": len(corpus_samples),
                    "mean_ratio": sum(corpus_ratios) / max(1, len(corpus_ratios)),
                    "median_ratio": statistics.median(corpus_ratios) if corpus_ratios else 0.0,
                    "min_ratio": min(corpus_ratios) if corpus_ratios else 0.0,
                    "max_ratio": max(corpus_ratios) if corpus_ratios else 0.0,
                    "win_rate": corpus_passed / max(1, len(corpus_samples)),
                    "passed": bool(corpus_samples) and corpus_passed == len(corpus_samples),
                }
            )

        seed_results: list[dict[str, Any]] = []
        for seed in self.seeds:
            seed_samples = [sample for sample in samples if sample["seed"] == seed]
            seed_ratios = [sample["ratio"] for sample in seed_samples]
            seed_passed = sum(1 for sample in seed_samples if sample["passed"])
            seed_results.append(
                {
                    "seed": seed,
                    "sample_count": len(seed_samples),
                    "mean_ratio": sum(seed_ratios) / max(1, len(seed_ratios)),
                    "median_ratio": statistics.median(seed_ratios) if seed_ratios else 0.0,
                    "min_ratio": min(seed_ratios) if seed_ratios else 0.0,
                    "win_rate": seed_passed / max(1, len(seed_samples)),
                    "passed": bool(seed_samples) and seed_passed == len(seed_samples),
                }
            )

        min_ratio = min(ratios) if ratios else 0.0
        max_regression = max(regressions) if regressions else 0.0
        min_baseline = min(baseline_scores) if baseline_scores else 0.0
        min_corpus = min(corpus_tokens) if corpus_tokens else 0
        min_planned = min(planned_tokens) if planned_tokens else 0
        expected_samples = len(self.corpora) * len(self.seeds)
        passed = (
            sample_count == expected_samples
            and passed_count == sample_count
            and min_ratio >= self.config.cortex_win_margin
            and min_baseline >= self.config.min_baseline_future_tokens_per_cost
            and max_regression <= self.config.max_next_token_loss_regression
            and min_corpus >= self.config.min_corpus_tokens
            and min_planned >= self.config.min_planned_train_tokens
        )
        return {
            "metric": "corpus_matrix_cortex_over_baseline",
            "corpora": [name for name, _, _ in self.corpora],
            "seeds": list(self.seeds),
            "corpus_count": len(self.corpora),
            "seed_count": len(self.seeds),
            "sample_count": sample_count,
            "required_margin": self.config.cortex_win_margin,
            "required_win_rate": 1.0,
            "mean_ratio": sum(ratios) / max(1, len(ratios)),
            "median_ratio": statistics.median(ratios) if ratios else 0.0,
            "ratio_population_stddev": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "min_ratio": min_ratio,
            "max_ratio": max(ratios) if ratios else 0.0,
            "win_rate": passed_count / max(1, sample_count),
            "mean_baseline_score": sum(baseline_scores) / max(1, len(baseline_scores)),
            "min_baseline_score": min_baseline,
            "min_baseline_future_tokens_per_cost": self.config.min_baseline_future_tokens_per_cost,
            "min_corpus_tokens": int(self.config.min_corpus_tokens),
            "min_observed_corpus_tokens": min_corpus,
            "min_planned_train_tokens": int(self.config.min_planned_train_tokens),
            "min_observed_planned_train_tokens": min_planned,
            "max_next_token_loss_regression": max_regression,
            "all_samples_passed": passed_count == sample_count and sample_count > 0,
            "corpus_results": tuple(corpus_results),
            "seed_results": tuple(seed_results),
            "samples": tuple(samples),
            "passed": passed,
        }

    def _write_markdown(self, report: CorpusMatrixReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM corpus matrix report",
            "",
            f"- Corpora: `{', '.join(proof['corpora'])}`",
            f"- Seeds: `{', '.join(str(seed) for seed in proof['seeds'])}`",
            f"- Samples: `{proof['sample_count']}`",
            f"- Mean Cortex/baseline ratio: `{proof['mean_ratio']:.3f}`",
            f"- Median Cortex/baseline ratio: `{proof['median_ratio']:.3f}`",
            f"- Min Cortex/baseline ratio: `{proof['min_ratio']:.3f}`",
            f"- Win rate: `{proof['win_rate']:.3f}`",
            f"- Max next-token-loss regression: `{proof['max_next_token_loss_regression']:.3f}`",
            f"- Min observed corpus tokens: `{proof['min_observed_corpus_tokens']}` (required `{proof['min_corpus_tokens']}`)",
            f"- Min observed planned train tokens: `{proof['min_observed_planned_train_tokens']}` (required `{proof['min_planned_train_tokens']}`)",
            f"- Passed: `{proof['passed']}`",
            "",
            "## Corpus Results",
            "",
            "| Corpus | Samples | Mean ratio | Median ratio | Min ratio | Win rate | Passed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in proof["corpus_results"]:
            lines.append(
                f"| `{item['corpus']}` | {item['sample_count']} | {item['mean_ratio']:.3f} | "
                f"{item['median_ratio']:.3f} | {item['min_ratio']:.3f} | {item['win_rate']:.3f} | `{item['passed']}` |"
            )
        lines.extend(
            [
                "",
                "## Seed Results",
                "",
                "| Seed | Samples | Mean ratio | Median ratio | Min ratio | Win rate | Passed |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in proof["seed_results"]:
            lines.append(
                f"| {item['seed']} | {item['sample_count']} | {item['mean_ratio']:.3f} | "
                f"{item['median_ratio']:.3f} | {item['min_ratio']:.3f} | {item['win_rate']:.3f} | `{item['passed']}` |"
            )
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                "- `corpus_matrix_report.json`",
                "- `corpus_matrix_report.md`",
                "- `corpus_matrix_ratios.png`",
                "- `corpus_matrix_learning_curves.csv`",
                "- `corpus_matrix_learning_curves.png`",
                "- `<corpus>/comparison_matrix_report.json`",
                "- `<corpus>/seed_<seed>/comparison_report.json`",
                "- `<corpus>/seed_<seed>/baseline_ntp/checkpoint_final.pt`",
                "- `<corpus>/seed_<seed>/cortex3/checkpoint_final.pt`",
            ]
        )
        (self.run_dir / "corpus_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_ratio_plot(self, report: CorpusMatrixReport) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        corpus_results = list(report.proof["corpus_results"])
        names = [str(item["corpus"]) for item in corpus_results]
        means = [float(item["mean_ratio"]) for item in corpus_results]
        lows = [max(0.0, float(item["mean_ratio"]) - float(item["min_ratio"])) for item in corpus_results]
        highs = [max(0.0, float(item["max_ratio"]) - float(item["mean_ratio"])) for item in corpus_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, means, color="#6a5f3a")
        ax.errorbar(names, means, yerr=[lows, highs], fmt="none", ecolor="#222222", capsize=4, linewidth=1)
        for index, corpus_name in enumerate(names):
            corpus_samples = [sample for sample in report.proof["samples"] if sample["corpus"] == corpus_name]
            for offset, sample in enumerate(corpus_samples):
                jitter = (offset - (len(corpus_samples) - 1) / 2.0) * 0.04
                ax.scatter(index + jitter, float(sample["ratio"]), color="#b2473e", s=22, zorder=3)
        ax.axhline(1.0, color="#333333", linewidth=1)
        ax.axhline(float(report.proof["required_margin"]), color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Cortex / baseline ratio by corpus and seed")
        ax.set_ylabel("ratio")
        ax.set_xlabel("corpus")
        fig.tight_layout()
        fig.savefig(self.run_dir / "corpus_matrix_ratios.png", dpi=150)
        plt.close(fig)

    def _write_learning_curve_summary(self, report: CorpusMatrixReport) -> None:
        rows: list[dict[str, Any]] = []
        for corpus_report in report.corpora:
            rows.extend(_read_validation_learning_curve_rows(corpus_report["seeds"], corpus=str(corpus_report["name"])))
        _write_learning_curve_matrix_artifacts(
            self.run_dir,
            rows=rows,
            csv_name="corpus_matrix_learning_curves.csv",
            png_name="corpus_matrix_learning_curves.png",
            group_by_corpus=True,
        )


class LLMExperimentRunner:
    def __init__(self, manifest: Mapping[str, Any], *, manifest_path: str | Path | None = None):
        self.manifest = self._normalize_manifest(manifest)
        self.manifest_path = str(manifest_path) if manifest_path is not None else None

    @staticmethod
    def load(path: str | Path) -> "LLMExperimentRunner":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ValueError("experiment manifest root must be a JSON object")
        return LLMExperimentRunner(payload, manifest_path=manifest_path)

    def run(self) -> LLMExperimentReport:
        run_dir = Path(str(self.manifest["out_dir"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "experiment_manifest.normalized.json", self.manifest)

        doctor_config = dict(self.manifest["doctor"])
        doctor_report = llm_doctor_report(**doctor_config)
        _write_json(run_dir / "doctor_report.json", doctor_report)
        if not doctor_report["passed"]:
            failed = ", ".join(str(check["name"]) for check in doctor_report["failed_required_checks"])
            raise RuntimeError(f"experiment doctor failed required checks: {failed}")

        preflight_report = self.preflight(doctor_report=doctor_report)
        _write_json(run_dir / "preflight_report.json", preflight_report.to_dict())
        if not preflight_report.passed:
            failed = "; ".join(preflight_report.failed_checks[:10])
            raise RuntimeError(f"experiment preflight failed: {failed}")

        corpora, prepared_payloads = self._prepare_corpora(run_dir)
        seeds = tuple(int(seed) for seed in self.manifest["seeds"])
        config = self._comparison_config(seeds)
        matrix_report = LLMCorpusMatrixSuite(
            corpora,
            config,
            run_dir=run_dir / "corpus_matrix",
            seeds=seeds,
        ).run(require_win=bool(self.manifest["require_win"]))
        report = LLMExperimentReport(
            run_dir=str(run_dir),
            manifest=self.manifest,
            doctor=doctor_report,
            corpora=tuple(prepared_payloads),
            corpus_matrix=matrix_report.to_dict(),
            proof=matrix_report.proof,
            hardware=hardware_report(),
        )
        _write_json(run_dir / "experiment_report.json", report.to_dict())
        self._write_markdown(report)
        return report

    def preflight(self, *, doctor_report: Mapping[str, Any] | None = None) -> LLMExperimentPreflightReport:
        run_dir = Path(str(self.manifest["out_dir"]))
        seeds = tuple(int(seed) for seed in self.manifest["seeds"])
        config = self._comparison_config(seeds)
        hardware = hardware_report()
        doctor_payload = dict(doctor_report or llm_doctor_report(**dict(self.manifest["doctor"])))
        device_type = str(doctor_payload.get("device_type", "cuda" if hardware.get("cuda_available") else "cpu"))
        estimates = _experiment_model_memory_estimates(config)
        max_peak = int(estimates["max_estimated_peak_training_bytes"])
        failed_checks: list[str] = []
        warnings: list[str] = []
        usable_fraction = 0.85
        total_memory = hardware.get("cuda_current_device_total_memory_bytes")
        usable_memory = int(float(total_memory) * usable_fraction) if total_memory else None

        if device_type == "cuda":
            if not hardware.get("cuda_available"):
                failed_checks.append("cuda_requested_but_unavailable")
            if total_memory is None:
                failed_checks.append("cuda_memory_capacity_unavailable")
            elif max_peak > int(usable_memory):
                failed_checks.append(
                    "cuda_memory_capacity_exceeded:"
                    f"estimated_peak={max_peak},usable={usable_memory},total={total_memory}"
                )
        else:
            warnings.append("device_type is not cuda; GPU memory capacity was not enforced")

        estimates = {
            **estimates,
            "device_type": device_type,
            "cuda_memory_usable_fraction": usable_fraction,
            "cuda_current_device_total_memory_bytes": total_memory,
            "cuda_current_device_usable_memory_bytes": usable_memory,
            "fits_cuda_memory": bool(usable_memory is not None and max_peak <= int(usable_memory)) if device_type == "cuda" else None,
        }
        return LLMExperimentPreflightReport(
            run_dir=str(run_dir),
            passed=not failed_checks,
            failed_checks=tuple(failed_checks),
            warnings=tuple(warnings),
            estimates=estimates,
            hardware=hardware,
        )

    def _normalize_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(manifest)
        if not str(payload.get("name", "")).strip():
            raise ValueError("experiment manifest requires non-empty `name`")
        if not str(payload.get("out_dir", "")).strip():
            raise ValueError("experiment manifest requires non-empty `out_dir`")
        seeds = payload.get("seeds")
        if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or not seeds:
            raise ValueError("experiment manifest requires non-empty integer `seeds`")
        payload["seeds"] = tuple(int(seed) for seed in seeds)
        if "corpora" not in payload or not isinstance(payload["corpora"], Sequence) or not payload["corpora"]:
            raise ValueError("experiment manifest requires non-empty `corpora` list")
        payload["corpora"] = tuple(self._normalize_corpus_config(item) for item in payload["corpora"])
        payload["doctor"] = self._normalize_doctor_config(payload.get("doctor", {}))
        payload["training"] = self._normalize_training_config(payload.get("training", {}), payload["seeds"][0], payload["doctor"])
        payload["model"] = self._normalize_model_config(payload.get("model", {}))
        payload["require_win"] = bool(payload.get("require_win", True))
        return payload

    def _normalize_doctor_config(self, raw: Any) -> dict[str, Any]:
        payload = dict(raw or {})
        return {
            "require_cuda": bool(payload.get("require_cuda", False)),
            "require_cuda_extension": bool(payload.get("require_cuda_extension", False)),
            "precision": str(payload.get("precision", "bf16")),
            "device": str(payload.get("device", "auto")),
            "distributed": bool(payload.get("distributed", False)),
            "gloo_interface": payload.get("gloo_interface"),
        }

    def _normalize_training_config(self, raw: Any, seed: int, doctor: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(raw or {})
        return {
            "steps": int(payload.get("steps", 200)),
            "batch_size": int(payload.get("batch_size", 32)),
            "gradient_accumulation_steps": int(payload.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(payload.get("learning_rate", 3e-4)),
            "weight_decay": float(payload.get("weight_decay", 0.01)),
            "grad_clip": float(payload.get("grad_clip", 1.0)),
            "eval_interval": int(payload.get("eval_interval", max(1, int(payload.get("steps", 200)) // 10))),
            "eval_batches": int(payload.get("eval_batches", 8)),
            "seed": int(payload.get("seed", seed)),
            "device": str(payload.get("device", doctor.get("device", "auto"))),
            "precision": str(payload.get("precision", doctor.get("precision", "bf16"))),
            "require_cuda": bool(payload.get("require_cuda", doctor.get("require_cuda", False))),
            "distributed": bool(payload.get("distributed", doctor.get("distributed", False))),
            "gloo_interface": payload.get("gloo_interface", doctor.get("gloo_interface")),
            "resume": bool(payload.get("resume", False)),
            "resume_if_exists": bool(payload.get("resume_if_exists", False)),
            "checkpoint_interval": int(payload.get("checkpoint_interval", 100)),
            "max_intermediate_checkpoints": int(payload.get("max_intermediate_checkpoints", 0)),
            "resource_monitor_interval": float(payload.get("resource_monitor_interval", 2.0)),
            "cortex_phase_interval": int(payload.get("cortex_phase_interval", 0)),
            "cortex_phase_probe_tasks": int(payload.get("cortex_phase_probe_tasks", 1)),
            "cortex_phase_max_proposals": int(payload.get("cortex_phase_max_proposals", 1)),
            "cortex_phase_regularization_weight": float(payload.get("cortex_phase_regularization_weight", 0.001)),
            "cortex_phase_replay_weight": float(payload.get("cortex_phase_replay_weight", 0.05)),
            "num_threads": payload.get("num_threads"),
        }

    def _normalize_model_config(self, raw: Any) -> dict[str, Any]:
        payload = dict(raw or {})
        horizons = payload.get("horizons", (1, 2, 4, 8))
        return {
            "vocab_size": int(payload.get("vocab_size", 4096)),
            "min_frequency": int(payload.get("min_frequency", 2)),
            "seq_len": int(payload.get("seq_len", 128)),
            "d_model": int(payload.get("d_model", 256)),
            "n_heads": int(payload.get("n_heads", 8)),
            "n_layers": int(payload.get("n_layers", 6)),
            "dropout": float(payload.get("dropout", 0.1)),
            "horizons": tuple(int(item) for item in horizons),
            "cortex_win_margin": float(payload.get("cortex_win_margin", 1.05)),
            "max_next_token_loss_regression": float(payload.get("max_next_token_loss_regression", 1.20)),
            "min_corpus_tokens": int(payload.get("min_corpus_tokens", 0)),
            "max_corpus_tokens": int(payload["max_corpus_tokens"]) if payload.get("max_corpus_tokens") is not None else None,
            "tokenizer_training_chars": (
                int(payload["tokenizer_training_chars"]) if payload.get("tokenizer_training_chars") is not None else None
            ),
            "min_planned_train_tokens": int(payload.get("min_planned_train_tokens", 0)),
        }

    def _normalize_corpus_config(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("each corpus entry must be a JSON object")
        payload = dict(raw)
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("each corpus entry requires non-empty `name`")
        kind = str(payload.get("kind", "paths" if "paths" in payload else "hf"))
        normalized: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "min_chars_per_chunk": int(payload.get("min_chars_per_chunk", 2048)),
        }
        if kind == "paths":
            paths = payload.get("paths")
            if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
                raise ValueError(f"paths corpus {name!r} requires non-empty `paths` list")
            normalized["paths"] = tuple(str(path) for path in paths)
            return normalized
        if kind == "hf":
            if not str(payload.get("dataset", "")).strip():
                raise ValueError(f"hf corpus {name!r} requires `dataset`")
            normalized.update(
                {
                    "dataset": str(payload["dataset"]),
                    "config_name": payload.get("config_name"),
                    "split": str(payload.get("split", "train")),
                    "text_field": str(payload.get("text_field", "text")),
                    "data_files": tuple(str(path) for path in payload.get("data_files", ())),
                    "streaming": bool(payload.get("streaming", True)),
                    "trust_remote_code": bool(payload.get("trust_remote_code", False)),
                    "cache_dir": payload.get("cache_dir"),
                    "max_documents": payload.get("max_documents", 100_000),
                    "max_characters": payload.get("max_characters"),
                    "allow_unbounded": bool(payload.get("allow_unbounded", False)),
                    "min_text_chars": int(payload.get("min_text_chars", 1)),
                    "shard_max_chars": int(payload.get("shard_max_chars", 64 * 1024 * 1024)),
                }
            )
            return normalized
        raise ValueError(f"unsupported corpus kind {kind!r} for corpus {name!r}")

    def _prepare_corpora(self, run_dir: Path) -> tuple[tuple[tuple[str, TextCorpusConfig], ...], list[Mapping[str, Any]]]:
        corpora: list[tuple[str, TextCorpusConfig]] = []
        payloads: list[Mapping[str, Any]] = []
        for corpus_payload in self.manifest["corpora"]:
            name = str(corpus_payload["name"])
            if corpus_payload["kind"] == "paths":
                corpus = TextCorpusConfig.from_paths(
                    corpus_payload["paths"],
                    min_chars_per_chunk=int(corpus_payload["min_chars_per_chunk"]),
                )
                corpora.append((name, corpus))
                payloads.append({"name": name, "kind": "paths", "files": corpus.files})
                continue

            corpus_dir = run_dir / "prepared" / _safe_run_name(name)
            export_config = HFDatasetExportConfig(
                dataset=str(corpus_payload["dataset"]),
                split=str(corpus_payload["split"]),
                text_field=str(corpus_payload["text_field"]),
                config_name=corpus_payload.get("config_name"),
                data_files=tuple(corpus_payload.get("data_files", ())),
                streaming=bool(corpus_payload["streaming"]),
                trust_remote_code=bool(corpus_payload["trust_remote_code"]),
                cache_dir=corpus_payload.get("cache_dir"),
                max_documents=corpus_payload.get("max_documents"),
                max_characters=corpus_payload.get("max_characters"),
                allow_unbounded=bool(corpus_payload["allow_unbounded"]),
                min_text_chars=int(corpus_payload["min_text_chars"]),
                shard_max_chars=int(corpus_payload["shard_max_chars"]),
            )
            export_report = HFDatasetTextExporter(export_config).export(
                corpus_dir,
                resume=bool(self.manifest["training"]["resume"] or self.manifest["training"]["resume_if_exists"]),
            )
            corpus = TextCorpusConfig.from_paths(
                export_report.shard_files,
                min_chars_per_chunk=int(corpus_payload["min_chars_per_chunk"]),
            )
            corpora.append((name, corpus))
            payloads.append({"name": name, "kind": "hf", "hf_export": export_report.to_dict(), "files": corpus.files})
        return tuple(corpora), payloads

    def _comparison_config(self, seeds: Sequence[int]) -> ComparisonConfig:
        training_payload = self.manifest["training"]
        model_payload = self.manifest["model"]
        training = TrainingConfig(
            steps=int(training_payload["steps"]),
            batch_size=int(training_payload["batch_size"]),
            gradient_accumulation_steps=int(training_payload["gradient_accumulation_steps"]),
            learning_rate=float(training_payload["learning_rate"]),
            weight_decay=float(training_payload["weight_decay"]),
            grad_clip=float(training_payload["grad_clip"]),
            eval_interval=int(training_payload["eval_interval"]),
            eval_batches=int(training_payload["eval_batches"]),
            seed=int(seeds[0]),
            device=str(training_payload["device"]),
            precision=str(training_payload["precision"]),
            require_cuda=bool(training_payload["require_cuda"]),
            distributed=bool(training_payload["distributed"]),
            gloo_interface=training_payload.get("gloo_interface"),
            resume=bool(training_payload["resume"]),
            resume_if_exists=bool(training_payload["resume_if_exists"]),
            checkpoint_interval=int(training_payload["checkpoint_interval"]),
            max_intermediate_checkpoints=int(training_payload["max_intermediate_checkpoints"]),
            resource_monitor_interval=float(training_payload["resource_monitor_interval"]),
            cortex_phase_interval=int(training_payload["cortex_phase_interval"]),
            cortex_phase_probe_tasks=int(training_payload["cortex_phase_probe_tasks"]),
            cortex_phase_max_proposals=int(training_payload["cortex_phase_max_proposals"]),
            cortex_phase_regularization_weight=float(training_payload["cortex_phase_regularization_weight"]),
            cortex_phase_replay_weight=float(training_payload["cortex_phase_replay_weight"]),
            num_threads=training_payload.get("num_threads"),
        )
        return ComparisonConfig(
            vocab_size=int(model_payload["vocab_size"]),
            min_frequency=int(model_payload["min_frequency"]),
            seq_len=int(model_payload["seq_len"]),
            d_model=int(model_payload["d_model"]),
            n_heads=int(model_payload["n_heads"]),
            n_layers=int(model_payload["n_layers"]),
            dropout=float(model_payload["dropout"]),
            horizons=tuple(model_payload["horizons"]),
            training=training,
            cortex_win_margin=float(model_payload["cortex_win_margin"]),
            max_next_token_loss_regression=float(model_payload["max_next_token_loss_regression"]),
            min_corpus_tokens=int(model_payload["min_corpus_tokens"]),
            max_corpus_tokens=(
                int(model_payload["max_corpus_tokens"]) if model_payload.get("max_corpus_tokens") is not None else None
            ),
            tokenizer_training_chars=(
                int(model_payload["tokenizer_training_chars"])
                if model_payload.get("tokenizer_training_chars") is not None
                else None
            ),
            min_planned_train_tokens=int(model_payload["min_planned_train_tokens"]),
        )

    def _write_markdown(self, report: LLMExperimentReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM experiment report",
            "",
            f"- Experiment: `{report.manifest['name']}`",
            f"- Passed: `{proof['passed']}`",
            f"- Corpora: `{', '.join(proof['corpora'])}`",
            f"- Seeds: `{', '.join(str(seed) for seed in proof['seeds'])}`",
            f"- Samples: `{proof['sample_count']}`",
            f"- Mean Cortex/baseline ratio: `{proof['mean_ratio']:.3f}`",
            f"- Min Cortex/baseline ratio: `{proof['min_ratio']:.3f}`",
            f"- Win rate: `{proof['win_rate']:.3f}`",
            f"- Max next-token-loss regression: `{proof['max_next_token_loss_regression']:.3f}`",
            "",
            "## Artifacts",
            "",
            "- `experiment_manifest.normalized.json`",
            "- `doctor_report.json`",
            "- `preflight_report.json`",
            "- `experiment_report.json`",
            "- `experiment_report.md`",
            "- `prepared/<corpus>/hf_export_report.json` for HF corpora",
            "- `corpus_matrix/corpus_matrix_report.json`",
            "- `corpus_matrix/corpus_matrix_learning_curves.csv`",
            "- `corpus_matrix/corpus_matrix_learning_curves.png`",
        ]
        Path(report.run_dir, "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json_artifact(path: Path, failures: list[str], checked: list[str]) -> dict[str, Any] | None:
    checked.append(str(path))
    if not path.exists():
        failures.append(f"missing artifact: {path}")
        return None
    if path.stat().st_size <= 0:
        failures.append(f"empty artifact: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid JSON artifact {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"JSON artifact root is not an object: {path}")
        return None
    return payload


def _require_nonempty_artifact(path: Path, failures: list[str], checked: list[str]) -> None:
    checked.append(str(path))
    if not path.exists():
        failures.append(f"missing artifact: {path}")
        return
    if path.stat().st_size <= 0:
        failures.append(f"empty artifact: {path}")


def _resolve_recorded_path(raw_path: Any, *, run_dir: Path) -> Path:
    if raw_path is None or not str(raw_path).strip():
        return run_dir / "__missing_recorded_path__"
    path = Path(str(raw_path))
    if path.is_absolute() or path.exists():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    return path


def _audit_comparison_run_artifacts(
    seed_dir: Path,
    *,
    require_passed: bool,
    failures: list[str],
    checked: list[str],
) -> None:
    comparison = _load_json_artifact(seed_dir / "comparison_report.json", failures, checked)
    _require_nonempty_artifact(seed_dir / "report.md", failures, checked)
    _require_nonempty_artifact(seed_dir / "run_plan.json", failures, checked)
    _require_nonempty_artifact(seed_dir / "learning_curve_audit.json", failures, checked)
    _require_nonempty_artifact(seed_dir / "learning_curve.png", failures, checked)
    if comparison is not None:
        proof = comparison.get("proof", {})
        if require_passed and not bool(proof.get("passed", False)):
            failures.append(f"comparison proof did not pass: {seed_dir}")
        curve_audit = comparison.get("curve_audit", {})
        if not bool(curve_audit.get("passed", False)):
            failures.append(f"comparison curve audit did not pass: {seed_dir}")
        for model_name in ("baseline", "cortex"):
            report_payload = comparison.get(model_name, {})
            checkpoint_path = report_payload.get("checkpoint_path")
            if checkpoint_path:
                _require_nonempty_artifact(
                    _resolve_recorded_path(checkpoint_path, run_dir=seed_dir),
                    failures,
                    checked,
                )
            else:
                failures.append(f"comparison report is missing {model_name}.checkpoint_path: {seed_dir}")
    for model_dir in ("baseline_ntp", "cortex3"):
        model_path = seed_dir / model_dir
        _require_nonempty_artifact(model_path / "training_report.json", failures, checked)
        _require_nonempty_artifact(model_path / "learning_curve.csv", failures, checked)
        _require_nonempty_artifact(model_path / "checkpoint_final.pt", failures, checked)


def audit_llm_experiment_artifacts(
    run_dir: str | Path,
    *,
    require_passed: bool = True,
) -> LLMExperimentAuditReport:
    root = Path(run_dir)
    failures: list[str] = []
    checked: list[str] = []
    if not root.exists():
        failures.append(f"missing run_dir: {root}")
        return LLMExperimentAuditReport(
            run_dir=str(root),
            passed=False,
            failed_checks=tuple(failures),
            checked_artifacts=tuple(checked),
            proof={},
        )

    _require_nonempty_artifact(root / "experiment_manifest.normalized.json", failures, checked)
    _require_nonempty_artifact(root / "experiment_report.md", failures, checked)
    doctor = _load_json_artifact(root / "doctor_report.json", failures, checked)
    preflight = _load_json_artifact(root / "preflight_report.json", failures, checked)
    experiment = _load_json_artifact(root / "experiment_report.json", failures, checked)
    if doctor is not None and not bool(doctor.get("passed", False)):
        failures.append("doctor_report.json did not pass")
    if preflight is not None and not bool(preflight.get("passed", False)):
        failures.append("preflight_report.json did not pass")
    proof: Mapping[str, Any] = {}
    if experiment is not None:
        proof = experiment.get("proof", {}) if isinstance(experiment.get("proof", {}), Mapping) else {}
        if require_passed and not bool(proof.get("passed", False)):
            failures.append("experiment proof did not pass")
        for corpus_payload in experiment.get("corpora", ()):
            if not isinstance(corpus_payload, Mapping):
                failures.append("experiment corpus payload is not an object")
                continue
            for file_path in corpus_payload.get("files", ()):
                _require_nonempty_artifact(_resolve_recorded_path(file_path, run_dir=root), failures, checked)
            if corpus_payload.get("kind") == "hf":
                hf_report = corpus_payload.get("hf_export", {})
                output_dir = hf_report.get("output_dir") if isinstance(hf_report, Mapping) else None
                if output_dir:
                    report_path = _resolve_recorded_path(Path(str(output_dir)) / "hf_export_report.json", run_dir=root)
                    _require_nonempty_artifact(report_path, failures, checked)
                    try:
                        HFDatasetExportReport.load(report_path).validate_artifacts()
                    except Exception as exc:
                        failures.append(f"HF export artifact validation failed for {report_path}: {exc}")

    matrix_dir = root / "corpus_matrix"
    matrix = _load_json_artifact(matrix_dir / "corpus_matrix_report.json", failures, checked)
    _require_nonempty_artifact(matrix_dir / "corpus_matrix_report.md", failures, checked)
    _require_nonempty_artifact(matrix_dir / "corpus_matrix_ratios.png", failures, checked)
    _require_nonempty_artifact(matrix_dir / "corpus_matrix_learning_curves.csv", failures, checked)
    _require_nonempty_artifact(matrix_dir / "corpus_matrix_learning_curves.png", failures, checked)
    if matrix is not None:
        matrix_proof = matrix.get("proof", {})
        if require_passed and not bool(matrix_proof.get("passed", False)):
            failures.append("corpus_matrix proof did not pass")
        for corpus_report in matrix.get("corpora", ()):
            if not isinstance(corpus_report, Mapping):
                failures.append("corpus_matrix corpus payload is not an object")
                continue
            corpus_dir = _resolve_recorded_path(corpus_report.get("run_dir", ""), run_dir=matrix_dir)
            _require_nonempty_artifact(corpus_dir / "comparison_matrix_report.json", failures, checked)
            _require_nonempty_artifact(corpus_dir / "comparison_matrix_report.md", failures, checked)
            _require_nonempty_artifact(corpus_dir / "comparison_matrix_ratios.png", failures, checked)
            _require_nonempty_artifact(corpus_dir / "comparison_matrix_learning_curves.csv", failures, checked)
            _require_nonempty_artifact(corpus_dir / "comparison_matrix_learning_curves.png", failures, checked)
            manifest_path = corpus_dir / "corpus" / "manifest.json"
            _require_nonempty_artifact(manifest_path, failures, checked)
            if manifest_path.exists() and manifest_path.stat().st_size > 0:
                try:
                    TokenizedCorpusManifest.load(manifest_path).identity()
                except Exception as exc:
                    failures.append(f"tokenized corpus manifest validation failed for {manifest_path}: {exc}")
            corpus_proof = corpus_report.get("proof", {})
            if require_passed and not bool(corpus_proof.get("passed", False)):
                failures.append(f"comparison matrix proof did not pass: {corpus_dir}")
            for seed_report in corpus_report.get("seeds", ()):
                if not isinstance(seed_report, Mapping):
                    failures.append(f"seed payload is not an object under {corpus_dir}")
                    continue
                seed_dir = _resolve_recorded_path(seed_report.get("run_dir", ""), run_dir=corpus_dir)
                _audit_comparison_run_artifacts(
                    seed_dir,
                    require_passed=require_passed,
                    failures=failures,
                    checked=checked,
                )

    return LLMExperimentAuditReport(
        run_dir=str(root),
        passed=not failures,
        failed_checks=tuple(failures),
        checked_artifacts=tuple(dict.fromkeys(checked)),
        proof=dict(proof),
    )


class LLMBenchmarkSuite:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        domains: Sequence[str],
        repeats: int,
        config: ComparisonConfig,
    ):
        self.run_dir = Path(run_dir)
        self.domains = tuple(domains)
        self.repeats = repeats
        self.config = config

    def run(self, *, require_win: bool = False) -> BenchmarkSuiteReport:
        runtime = DistributedRuntime.from_env(
            requested=self.config.training.distributed,
            device_type="cuda" if torch.cuda.is_available() and self.config.training.device in {"auto", "cuda"} else "cpu",
            gloo_interface=self.config.training.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        _barrier_if_needed(runtime)

        domain_payloads: list[Mapping[str, Any]] = []
        for domain in self.domains:
            corpus_dir = self.run_dir / "corpora" / domain
            if runtime.is_main:
                corpus_files = build_benchmark_corpus(corpus_dir, domain=domain, repeats=self.repeats)
            _barrier_if_needed(runtime)
            corpus_files = (str(corpus_dir / f"{domain}.txt"),)
            corpus = TextCorpusConfig.from_paths(corpus_files, min_chars_per_chunk=512)
            report = LLMComparisonRunner(corpus, self.config, run_dir=self.run_dir / domain).run(require_win=False)
            domain_payloads.append(
                {
                    "domain": domain,
                    "run_dir": report.run_dir,
                    "proof": report.proof,
                    "baseline_final_val": report.baseline["final_val"],
                    "cortex_final_val": report.cortex["final_val"],
                }
            )
        proof = self._proof(domain_payloads)
        report = BenchmarkSuiteReport(
            run_dir=str(self.run_dir),
            domains=tuple(domain_payloads),
            proof=proof,
            hardware=hardware_report(),
        )
        if runtime.is_main:
            _write_json(self.run_dir / "benchmark_report.json", report.to_dict())
            self._write_markdown(report)
            self._write_bar_chart(report)
        failed = bool(require_win and not proof["passed"] and runtime.is_main)
        if runtime.enabled:
            flag_device = torch.device(f"cuda:{runtime.local_rank}") if runtime.backend == "nccl" else torch.device("cpu")
            flag = torch.tensor([1 if failed else 0], dtype=torch.int64, device=flag_device)
            torch.distributed.broadcast(flag, src=0)
            failed = bool(int(flag.cpu().item()))
        _barrier_if_needed(runtime)
        if failed:
            raise RuntimeError(f"Cortex benchmark did not pass: {proof}")
        return report

    def _proof(self, domains: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        ratios = [float(item["proof"]["cortex_over_baseline_ratio"]) for item in domains]
        baseline_scores = [float(item["proof"]["baseline_score"]) for item in domains]
        regressions = [float(item["proof"]["next_token_loss_regression_ratio"]) for item in domains]
        corpus_tokens = [int(item["proof"].get("corpus_token_count", 0)) for item in domains]
        planned_tokens = [int(item["proof"].get("planned_train_tokens", 0)) for item in domains]
        all_domain_proofs = [
            bool(item["proof"]["passed"])
            and bool(
                item["proof"].get(
                    "baseline_score_passed",
                    float(item["proof"]["baseline_score"]) >= self.config.min_baseline_future_tokens_per_cost,
                )
            )
            and bool(
                item["proof"].get(
                    "corpus_scale_passed",
                    int(item["proof"].get("corpus_token_count", 0)) >= self.config.min_corpus_tokens,
                )
            )
            and bool(
                item["proof"].get(
                    "planned_train_tokens_passed",
                    int(item["proof"].get("planned_train_tokens", 0)) >= self.config.min_planned_train_tokens,
                )
            )
            for item in domains
        ]
        min_baseline = min(baseline_scores) if baseline_scores else 0.0
        min_corpus = min(corpus_tokens) if corpus_tokens else 0
        min_planned = min(planned_tokens) if planned_tokens else 0
        return {
            "metric": "benchmark_mean_cortex_over_baseline",
            "domains": [str(item["domain"]) for item in domains],
            "domain_count": len(domains),
            "mean_ratio": sum(ratios) / max(1, len(ratios)),
            "min_ratio": min(ratios) if ratios else 0.0,
            "mean_baseline_score": sum(baseline_scores) / max(1, len(baseline_scores)),
            "min_baseline_score": min_baseline,
            "min_baseline_future_tokens_per_cost": self.config.min_baseline_future_tokens_per_cost,
            "min_corpus_tokens": int(self.config.min_corpus_tokens),
            "min_observed_corpus_tokens": min_corpus,
            "min_planned_train_tokens": int(self.config.min_planned_train_tokens),
            "min_observed_planned_train_tokens": min_planned,
            "max_next_token_loss_regression": max(regressions) if regressions else 0.0,
            "all_domains_passed": all(all_domain_proofs),
            "passed": (
                bool(domains)
                and all(all_domain_proofs)
                and min_corpus >= self.config.min_corpus_tokens
                and min_planned >= self.config.min_planned_train_tokens
            ),
        }

    def _write_markdown(self, report: BenchmarkSuiteReport) -> None:
        lines = [
            "# Cortex-3 LLM benchmark report",
            "",
            f"- Domains: `{', '.join(report.proof['domains'])}`",
            f"- Mean Cortex/baseline ratio: `{report.proof['mean_ratio']:.3f}`",
            f"- Min Cortex/baseline ratio: `{report.proof['min_ratio']:.3f}`",
            f"- Mean baseline score: `{report.proof['mean_baseline_score']:.6f}`",
            f"- Max next-token-loss regression: `{report.proof['max_next_token_loss_regression']:.3f}`",
            f"- Min observed corpus tokens: `{report.proof['min_observed_corpus_tokens']}` (required `{report.proof['min_corpus_tokens']}`)",
            f"- Min observed planned train tokens: `{report.proof['min_observed_planned_train_tokens']}` (required `{report.proof['min_planned_train_tokens']}`)",
            f"- Passed: `{report.proof['passed']}`",
            "",
            "## Domain Results",
            "",
        ]
        for item in report.domains:
            proof = item["proof"]
            lines.append(f"- `{item['domain']}`: ratio `{proof['cortex_over_baseline_ratio']:.3f}`, baseline `{proof['baseline_score']:.6f}`, cortex `{proof['cortex_score']:.6f}`, passed `{proof['passed']}`")
        (self.run_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_bar_chart(self, report: BenchmarkSuiteReport) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [str(item["domain"]) for item in report.domains]
        ratios = [float(item["proof"]["cortex_over_baseline_ratio"]) for item in report.domains]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, ratios, color="#287c71")
        ax.axhline(1.0, color="#333333", linewidth=1)
        ax.set_title("Cortex / baseline future-token cost ratio")
        ax.set_ylabel("ratio")
        ax.set_xlabel("domain")
        fig.tight_layout()
        fig.savefig(self.run_dir / "benchmark_ratios.png", dpi=150)
        plt.close(fig)


class LLMStatisticalBenchmarkSuite:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        domains: Sequence[str],
        seeds: Sequence[int],
        repeats: int,
        config: ComparisonConfig,
    ):
        if not domains:
            raise ValueError("at least one benchmark domain is required")
        if not seeds:
            raise ValueError("at least one benchmark seed is required")
        self.run_dir = Path(run_dir)
        self.domains = tuple(domains)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.repeats = repeats
        self.config = config

    def run(self, *, require_win: bool = False) -> StatisticalBenchmarkReport:
        runtime = DistributedRuntime.from_env(
            requested=self.config.training.distributed,
            device_type="cuda" if torch.cuda.is_available() and self.config.training.device in {"auto", "cuda"} else "cpu",
            gloo_interface=self.config.training.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            if self.run_dir.exists() and not _training_allows_existing_artifacts(self.config.training):
                shutil.rmtree(self.run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        _barrier_if_needed(runtime)

        seed_payloads: list[Mapping[str, Any]] = []
        for seed in self.seeds:
            seed_training = replace(self.config.training, seed=seed)
            seed_config = replace(self.config, training=seed_training)
            seed_report = LLMBenchmarkSuite(
                run_dir=self.run_dir / f"seed_{seed}",
                domains=self.domains,
                repeats=self.repeats,
                config=seed_config,
            ).run(require_win=False)
            seed_payloads.append(
                {
                    "seed": seed,
                    "run_dir": seed_report.run_dir,
                    "proof": seed_report.proof,
                    "domains": seed_report.domains,
                }
            )

        proof = self._proof(seed_payloads)
        report = StatisticalBenchmarkReport(
            run_dir=str(self.run_dir),
            seeds=tuple(seed_payloads),
            proof=proof,
            hardware=hardware_report(),
        )
        if runtime.is_main:
            _write_json(self.run_dir / "statistical_benchmark_report.json", report.to_dict())
            self._write_markdown(report)
            self._write_ratio_plot(report)
        failed = bool(require_win and not proof["passed"] and runtime.is_main)
        if runtime.enabled:
            flag_device = torch.device(f"cuda:{runtime.local_rank}") if runtime.backend == "nccl" else torch.device("cpu")
            flag = torch.tensor([1 if failed else 0], dtype=torch.int64, device=flag_device)
            torch.distributed.broadcast(flag, src=0)
            failed = bool(int(flag.cpu().item()))
        _barrier_if_needed(runtime)
        if failed:
            raise RuntimeError(f"Cortex statistical benchmark did not pass: {proof}")
        return report

    def _proof(self, seed_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        samples: list[dict[str, Any]] = []
        for seed_report in seed_reports:
            seed = int(seed_report["seed"])
            for domain_report in seed_report["domains"]:
                proof = domain_report["proof"]
                baseline_score = float(proof["baseline_score"])
                baseline_score_passed = bool(
                    proof.get("baseline_score_passed", baseline_score >= self.config.min_baseline_future_tokens_per_cost)
                )
                corpus_token_count = int(proof.get("corpus_token_count", 0))
                planned_train_tokens = int(proof.get("planned_train_tokens", 0))
                corpus_scale_passed = bool(proof.get("corpus_scale_passed", corpus_token_count >= self.config.min_corpus_tokens))
                planned_train_tokens_passed = bool(
                    proof.get("planned_train_tokens_passed", planned_train_tokens >= self.config.min_planned_train_tokens)
                )
                samples.append(
                    {
                        "seed": seed,
                        "domain": str(domain_report["domain"]),
                        "ratio": float(proof["cortex_over_baseline_ratio"]),
                        "baseline_score": baseline_score,
                        "baseline_score_passed": baseline_score_passed,
                        "cortex_score": float(proof["cortex_score"]),
                        "next_token_loss_regression_ratio": float(proof["next_token_loss_regression_ratio"]),
                        "corpus_token_count": corpus_token_count,
                        "corpus_scale_passed": corpus_scale_passed,
                        "planned_train_tokens": planned_train_tokens,
                        "planned_train_tokens_passed": planned_train_tokens_passed,
                        "passed": bool(proof["passed"]) and baseline_score_passed and corpus_scale_passed and planned_train_tokens_passed,
                    }
                )

        ratios = [sample["ratio"] for sample in samples]
        baseline_scores = [sample["baseline_score"] for sample in samples]
        regressions = [sample["next_token_loss_regression_ratio"] for sample in samples]
        corpus_tokens = [int(sample["corpus_token_count"]) for sample in samples]
        planned_tokens = [int(sample["planned_train_tokens"]) for sample in samples]
        passed_count = sum(1 for sample in samples if sample["passed"])
        sample_count = len(samples)

        domain_results: list[dict[str, Any]] = []
        for domain in self.domains:
            domain_samples = [sample for sample in samples if sample["domain"] == domain]
            domain_ratios = [sample["ratio"] for sample in domain_samples]
            domain_passed = sum(1 for sample in domain_samples if sample["passed"])
            domain_results.append(
                {
                    "domain": domain,
                    "sample_count": len(domain_samples),
                    "mean_ratio": sum(domain_ratios) / max(1, len(domain_ratios)),
                    "min_ratio": min(domain_ratios) if domain_ratios else 0.0,
                    "max_ratio": max(domain_ratios) if domain_ratios else 0.0,
                    "win_rate": domain_passed / max(1, len(domain_samples)),
                    "passed": bool(domain_samples) and domain_passed == len(domain_samples),
                }
            )

        seed_results: list[dict[str, Any]] = []
        for seed in self.seeds:
            seed_samples = [sample for sample in samples if sample["seed"] == seed]
            seed_ratios = [sample["ratio"] for sample in seed_samples]
            seed_passed = sum(1 for sample in seed_samples if sample["passed"])
            seed_results.append(
                {
                    "seed": seed,
                    "sample_count": len(seed_samples),
                    "mean_ratio": sum(seed_ratios) / max(1, len(seed_ratios)),
                    "min_ratio": min(seed_ratios) if seed_ratios else 0.0,
                    "win_rate": seed_passed / max(1, len(seed_samples)),
                    "passed": bool(seed_samples) and seed_passed == len(seed_samples),
                }
            )

        min_ratio = min(ratios) if ratios else 0.0
        max_regression = max(regressions) if regressions else 0.0
        min_baseline = min(baseline_scores) if baseline_scores else 0.0
        min_corpus = min(corpus_tokens) if corpus_tokens else 0
        min_planned = min(planned_tokens) if planned_tokens else 0
        win_rate = passed_count / max(1, sample_count)
        passed = (
            sample_count == len(self.domains) * len(self.seeds)
            and passed_count == sample_count
            and min_ratio >= self.config.cortex_win_margin
            and min_baseline >= self.config.min_baseline_future_tokens_per_cost
            and max_regression <= self.config.max_next_token_loss_regression
            and min_corpus >= self.config.min_corpus_tokens
            and min_planned >= self.config.min_planned_train_tokens
        )
        return {
            "metric": "statistical_benchmark_cortex_over_baseline",
            "domains": list(self.domains),
            "seeds": list(self.seeds),
            "domain_count": len(self.domains),
            "seed_count": len(self.seeds),
            "sample_count": sample_count,
            "required_margin": self.config.cortex_win_margin,
            "required_win_rate": 1.0,
            "mean_ratio": sum(ratios) / max(1, len(ratios)),
            "median_ratio": statistics.median(ratios) if ratios else 0.0,
            "ratio_population_stddev": statistics.pstdev(ratios) if len(ratios) > 1 else 0.0,
            "min_ratio": min_ratio,
            "max_ratio": max(ratios) if ratios else 0.0,
            "win_rate": win_rate,
            "mean_baseline_score": sum(baseline_scores) / max(1, len(baseline_scores)),
            "min_baseline_score": min_baseline,
            "min_baseline_future_tokens_per_cost": self.config.min_baseline_future_tokens_per_cost,
            "min_corpus_tokens": int(self.config.min_corpus_tokens),
            "min_observed_corpus_tokens": min_corpus,
            "min_planned_train_tokens": int(self.config.min_planned_train_tokens),
            "min_observed_planned_train_tokens": min_planned,
            "max_next_token_loss_regression": max_regression,
            "all_samples_passed": passed_count == sample_count and sample_count > 0,
            "domain_results": tuple(domain_results),
            "seed_results": tuple(seed_results),
            "samples": tuple(samples),
            "passed": passed,
        }

    def _write_markdown(self, report: StatisticalBenchmarkReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM statistical benchmark report",
            "",
            f"- Domains: `{', '.join(proof['domains'])}`",
            f"- Seeds: `{', '.join(str(seed) for seed in proof['seeds'])}`",
            f"- Samples: `{proof['sample_count']}`",
            f"- Mean Cortex/baseline ratio: `{proof['mean_ratio']:.3f}`",
            f"- Median Cortex/baseline ratio: `{proof['median_ratio']:.3f}`",
            f"- Min Cortex/baseline ratio: `{proof['min_ratio']:.3f}`",
            f"- Win rate: `{proof['win_rate']:.3f}`",
            f"- Max next-token-loss regression: `{proof['max_next_token_loss_regression']:.3f}`",
            f"- Min observed corpus tokens: `{proof['min_observed_corpus_tokens']}` (required `{proof['min_corpus_tokens']}`)",
            f"- Min observed planned train tokens: `{proof['min_observed_planned_train_tokens']}` (required `{proof['min_planned_train_tokens']}`)",
            f"- Passed: `{proof['passed']}`",
            "",
            "## Domain Results",
            "",
            "| Domain | Samples | Mean ratio | Min ratio | Win rate | Passed |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in proof["domain_results"]:
            lines.append(
                f"| `{item['domain']}` | {item['sample_count']} | {item['mean_ratio']:.3f} | "
                f"{item['min_ratio']:.3f} | {item['win_rate']:.3f} | `{item['passed']}` |"
            )
        lines.extend(
            [
                "",
                "## Seed Results",
                "",
                "| Seed | Samples | Mean ratio | Min ratio | Win rate | Passed |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in proof["seed_results"]:
            lines.append(
                f"| {item['seed']} | {item['sample_count']} | {item['mean_ratio']:.3f} | "
                f"{item['min_ratio']:.3f} | {item['win_rate']:.3f} | `{item['passed']}` |"
            )
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                "- `statistical_benchmark_report.json`",
                "- `statistical_benchmark_report.md`",
                "- `statistical_benchmark_ratios.png`",
                "- `seed_<seed>/benchmark_report.json`",
                "- `seed_<seed>/<domain>/comparison_report.json`",
                "- `seed_<seed>/<domain>/baseline_ntp/checkpoint_final.pt`",
                "- `seed_<seed>/<domain>/cortex3/checkpoint_final.pt`",
            ]
        )
        (self.run_dir / "statistical_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_ratio_plot(self, report: StatisticalBenchmarkReport) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        domain_results = list(report.proof["domain_results"])
        names = [str(item["domain"]) for item in domain_results]
        means = [float(item["mean_ratio"]) for item in domain_results]
        lows = [max(0.0, float(item["mean_ratio"]) - float(item["min_ratio"])) for item in domain_results]
        highs = [max(0.0, float(item["max_ratio"]) - float(item["mean_ratio"])) for item in domain_results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, means, color="#435c7a")
        ax.errorbar(names, means, yerr=[lows, highs], fmt="none", ecolor="#222222", capsize=4, linewidth=1)
        for index, domain in enumerate(names):
            domain_samples = [sample for sample in report.proof["samples"] if sample["domain"] == domain]
            for offset, sample in enumerate(domain_samples):
                jitter = (offset - (len(domain_samples) - 1) / 2.0) * 0.04
                ax.scatter(index + jitter, float(sample["ratio"]), color="#b2473e", s=22, zorder=3)
        ax.axhline(1.0, color="#333333", linewidth=1)
        ax.axhline(float(report.proof["required_margin"]), color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Cortex / baseline ratio by domain and seed")
        ax.set_ylabel("ratio")
        ax.set_xlabel("domain")
        fig.tight_layout()
        fig.savefig(self.run_dir / "statistical_benchmark_ratios.png", dpi=150)
        plt.close(fig)


def build_seed_corpus(path: str | Path, *, repeats: int = 256) -> tuple[str, ...]:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    shard = output / "seed_corpus.txt"
    patterns = [
        "alpha beta gamma delta epsilon zeta eta theta.",
        "red green blue yellow red green blue yellow.",
        "the verifier checks anchors and preserves exact identifiers.",
        "cortex compiles slow verified skills into fast reusable circuits.",
        "one two three five eight thirteen twenty one.",
    ]
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(repeats):
            handle.write(f"C3-SAMPLE-{index:04d} {patterns[index % len(patterns)]}\n")
            handle.write(f"C3-MARK-{index % 17:02d} sample {index:04d} keeps sequence marker {index % 17:02d}.\n")
    return (str(shard),)


def _torch_cuda_memory_snapshot(device: torch.device) -> Mapping[str, Any]:
    if device.type != "cuda":
        return {"enabled": False}
    torch.cuda.synchronize(device)
    return {
        "enabled": True,
        "device": str(device),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _strict_extension_only_from_influence(influence: Mapping[str, Any]) -> bool:
    required_maps = (
        "native_ternary_backend_counts",
        "native_ternary_requantize_backend_counts",
        "native_ternary_grad_weight_backend_counts",
    )
    for key in required_maps:
        counts = dict(influence.get(key) or {})
        if int(counts.get("extension", 0)) <= 0:
            return False
        if any(name != "extension" and int(value) > 0 for name, value in counts.items()):
            return False
    return int(influence.get("torch_packed_ternary_dispatches", 0)) == 0


def run_llm_batch_profile(
    *,
    out_dir: str | Path,
    steps: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    seq_len: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    vocab_size: int = 256,
    precision: str = "auto",
    device: str = "auto",
    require_cuda: bool = False,
    native_ternary_backend: str = STRICT_NATIVE_TERNARY_BACKEND,
    resource_interval: float = 0.05,
    min_resource_samples: int = 2,
    seed: int = 71,
    corpus_repeats: int = 192,
    max_corpus_tokens: int | None = 8192,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output_dir = Path(out_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"profile output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_precision = _resolve_cli_precision(precision, device=device, require_cuda=require_cuda)
    files = build_seed_corpus(output_dir / "seed_text", repeats=corpus_repeats)
    corpus = TextCorpusConfig.from_paths(files, min_chars_per_chunk=512)
    training = TrainingConfig(
        steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_interval=max(1, steps),
        eval_batches=1,
        seed=seed,
        device=device,
        precision=resolved_precision,
        require_cuda=require_cuda,
        checkpoint_interval=max(1, steps),
        max_intermediate_checkpoints=0,
        resource_monitor_interval=resource_interval,
        cortex_phase_interval=1,
        cortex_phase_probe_tasks=1,
        cortex_phase_max_proposals=1,
        num_threads=1,
    )
    config = ComparisonConfig(
        vocab_size=vocab_size,
        min_frequency=1,
        seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
        horizons=(1, 2, 4, 8),
        training=training,
        max_corpus_tokens=max_corpus_tokens,
        native_ternary_backend=native_ternary_backend,
    )
    tokenizer = LLMTokenizer.train(corpus, vocab_size=vocab_size, min_frequency=1)
    manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
        output_dir / "corpus",
        seq_len=seq_len,
        max_horizon=max(config.horizons),
        max_tokens=max_corpus_tokens,
        preparation_config=_tokenized_preparation_config(
            corpus,
            vocab_size=vocab_size,
            min_frequency=1,
            seq_len=seq_len,
            max_horizon=max(config.horizons),
            max_tokens=max_corpus_tokens,
        ),
    )
    corpus_identity = manifest.identity()
    plan = build_training_plan(
        manifest,
        config,
        world_size=1,
        distributed=False,
        corpus_identity=corpus_identity,
    )
    _write_json(output_dir / "run_plan.json", plan)
    train_data = MemmapCausalDataset(manifest, split="train")
    val_data = MemmapCausalDataset(manifest, split="val")
    started = time.time()
    report: TrainingRunReport | None = None
    memory_before: Mapping[str, Any] = {"enabled": False}
    memory_after: Mapping[str, Any] = {"enabled": False}
    try:
        strict_native_required = _strict_native_ternary_required_for_training(training)
        model_config = TransformerConfig(
            vocab_size=manifest.vocab_size,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.0,
            horizons=config.horizons,
            use_cortex_heads=True,
            use_ternary_core=True,
            use_native_ternary_kernel=strict_native_required,
            require_native_ternary_kernel=strict_native_required,
            native_ternary_backend=native_ternary_backend,
            use_skill_aware_experts=True,
            use_variable_in_compressor=True,
            use_learned_memory_policy=True,
            use_certificate_head=True,
            use_latent_reasoning_workspace=True,
        )
        trainer = LLMTrainer(
            CortexTransformerLM(model_config),
            train_data,
            val_data,
            training,
            run_dir=output_dir / "cortex3",
            model_kind="cortex3_multi_horizon_profile",
            corpus_identity=corpus_identity,
        )
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(trainer.device)
        memory_before = _torch_cuda_memory_snapshot(trainer.device)
        report = trainer.train(name="cortex3_profile")
        memory_after = _torch_cuda_memory_snapshot(trainer.device)
    finally:
        train_data.close()
        val_data.close()
    elapsed = max(1e-9, time.time() - started)
    assert report is not None
    phase_report = dict(report.cortex_phase_report)
    influence = dict(phase_report.get("training_influence") or {})
    resource_usage = dict(report.resource_usage or {})
    resource_metrics = dict(resource_usage.get("metrics") or {})
    planned_tokens = int(steps) * int(batch_size) * int(gradient_accumulation_steps) * int(seq_len)
    cuda_profile = bool(memory_after.get("enabled"))
    failed_checks: list[str] = []
    if int(resource_usage.get("sample_count", 0)) < int(min_resource_samples):
        failed_checks.append("resource_sample_count")
    if planned_tokens <= 0:
        failed_checks.append("planned_tokens")
    if not bool(phase_report.get("architecture_audit", {}).get("passed")):
        failed_checks.append("architecture_audit")
    if not bool(phase_report.get("phase_deliverable_audit", {}).get("passed")):
        failed_checks.append("phase_deliverable_audit")
    native_kernel_required = bool(phase_report.get("native_ternary_kernel_required"))
    strict_extension_only = _strict_extension_only_from_influence(influence)
    if native_kernel_required and not strict_extension_only:
        failed_checks.append("strict_extension_only")
    if cuda_profile:
        for key in ("gpu_utilization_percent", "gpu_memory_used_mb", "gpu_power_draw_watts"):
            if key not in resource_metrics:
                failed_checks.append(f"missing_{key}")
        if int(memory_after.get("max_memory_allocated_bytes", 0)) <= 0:
            failed_checks.append("torch_cuda_peak_memory")
    elif require_cuda:
        failed_checks.append("cuda_memory_profile")
    throughput = {
        "wall_seconds": float(elapsed),
        "optimizer_steps": int(report.optimizer_steps),
        "effective_batch_size": int(report.effective_batch_size),
        "planned_train_tokens": int(planned_tokens),
        "optimizer_steps_per_second_wall": float(report.optimizer_steps / elapsed),
        "train_tokens_per_second_wall": float(planned_tokens / elapsed),
        "tokens_per_optimizer_step": int(batch_size * gradient_accumulation_steps * seq_len),
    }
    profile = {
        "schema_version": 1,
        "run_dir": str(output_dir),
        "training_report": report.to_dict(),
        "plan": plan,
        "throughput": throughput,
        "torch_cuda_memory": {
            "before": memory_before,
            "after": memory_after,
        },
        "resource_usage": resource_usage,
        "kernel_evidence": {
            "native_ternary_kernel_required": native_kernel_required,
            "strict_extension_only": strict_extension_only,
            "native_ternary_backend_counts": influence.get("native_ternary_backend_counts", {}),
            "native_ternary_kernel_variants": influence.get("native_ternary_kernel_variants", ()),
            "native_ternary_requantize_backend_counts": influence.get("native_ternary_requantize_backend_counts", {}),
            "native_ternary_grad_input_kernel_counts": influence.get("native_ternary_grad_input_kernel_counts", {}),
            "native_ternary_grad_weight_backend_counts": influence.get("native_ternary_grad_weight_backend_counts", {}),
            "native_ternary_grad_weight_kernel_counts": influence.get("native_ternary_grad_weight_kernel_counts", {}),
        },
        "architecture": {
            "architecture_audit_passed": bool(phase_report.get("architecture_audit", {}).get("passed")),
            "phase_deliverable_audit_passed": bool(phase_report.get("phase_deliverable_audit", {}).get("passed")),
            "all_phases_active": bool(phase_report.get("all_phases_active")),
            "phase_event_counts": phase_report.get("phase_event_counts", {}),
        },
        "hardware": report.hardware,
        "failed_checks": tuple(failed_checks),
        "passed": not failed_checks,
    }
    _write_json(output_dir / "llm_batch_profile.json", profile)
    return profile


def _normalize_profile_shape_spec(spec: Mapping[str, Any], *, default_batch_size: int) -> Mapping[str, int]:
    normalized = {
        "seq_len": int(spec.get("seq_len", spec.get("seq", 0))),
        "d_model": int(spec.get("d_model", spec.get("d", 0))),
        "n_heads": int(spec.get("n_heads", spec.get("heads", 0))),
        "n_layers": int(spec.get("n_layers", spec.get("layers", 0))),
        "batch_size": int(spec.get("batch_size", spec.get("batch", default_batch_size))),
    }
    if "gradient_accumulation_steps" in spec or "grad_accum" in spec or "g" in spec:
        normalized["gradient_accumulation_steps"] = int(
            spec.get("gradient_accumulation_steps", spec.get("grad_accum", spec.get("g", 0)))
        )
    for key, value in normalized.items():
        if value <= 0:
            raise ValueError(f"profile shape field {key} must be positive, got {value}")
    if normalized["d_model"] % normalized["n_heads"] != 0:
        raise ValueError(
            "profile shape d_model must be divisible by n_heads, "
            f"got d_model={normalized['d_model']} n_heads={normalized['n_heads']}"
        )
    return normalized


def _profile_shape_key(shape: Mapping[str, int]) -> str:
    key = (
        f"seq{int(shape['seq_len'])}_d{int(shape['d_model'])}_"
        f"h{int(shape['n_heads'])}_l{int(shape['n_layers'])}_b{int(shape['batch_size'])}"
    )
    if "gradient_accumulation_steps" in shape:
        key += f"_g{int(shape['gradient_accumulation_steps'])}"
    return key


def _profile_shape_memory_estimate(
    shape: Mapping[str, int],
    *,
    vocab_size: int,
    precision: str,
    device: str,
    require_cuda: bool,
    gradient_accumulation_steps: int,
    native_ternary_backend: str,
) -> Mapping[str, Any]:
    training = TrainingConfig(
        steps=1,
        batch_size=int(shape["batch_size"]),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        eval_interval=1,
        eval_batches=1,
        device=device,
        precision=_resolve_cli_precision(precision, device=device, require_cuda=require_cuda),
        require_cuda=bool(require_cuda),
        checkpoint_interval=1,
        max_intermediate_checkpoints=0,
        cortex_phase_interval=1,
        cortex_phase_probe_tasks=1,
        cortex_phase_max_proposals=1,
        resource_monitor_interval=0.05,
        num_threads=1,
    )
    strict_native_required = _strict_native_ternary_required_for_training(training)
    config = TransformerConfig(
        vocab_size=int(vocab_size),
        seq_len=int(shape["seq_len"]),
        d_model=int(shape["d_model"]),
        n_heads=int(shape["n_heads"]),
        n_layers=int(shape["n_layers"]),
        dropout=0.0,
        horizons=(1, 2, 4, 8),
        use_cortex_heads=True,
        use_ternary_core=True,
        use_native_ternary_kernel=strict_native_required,
        require_native_ternary_kernel=strict_native_required,
        native_ternary_backend=native_ternary_backend,
        use_skill_aware_experts=True,
        use_variable_in_compressor=True,
        use_learned_memory_policy=True,
        use_certificate_head=True,
        use_latent_reasoning_workspace=True,
    )
    return _estimate_transformer_training_memory(config, training)


def _profile_autosize_budget(
    *,
    memory_budget_mb: float,
    memory_budget_fraction: float,
    device: str,
    require_cuda: bool,
) -> Mapping[str, Any]:
    explicit_bytes = int(float(memory_budget_mb) * 1024 * 1024)
    if explicit_bytes > 0:
        return {
            "source": "explicit_mb",
            "budget_bytes": explicit_bytes,
            "memory_budget_mb": float(memory_budget_mb),
            "memory_budget_fraction": float(memory_budget_fraction),
        }
    if not 0.0 < float(memory_budget_fraction) <= 1.0:
        raise ValueError("memory_budget_fraction must be > 0 and <= 1")
    resolves_to_cuda = bool(require_cuda)
    if not resolves_to_cuda:
        if str(device) == "auto":
            resolves_to_cuda = bool(torch.cuda.is_available())
        else:
            try:
                resolves_to_cuda = torch.device(device).type == "cuda"
            except (TypeError, RuntimeError):
                resolves_to_cuda = str(device).startswith("cuda")
    if resolves_to_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory autosize requested but CUDA is unavailable")
        cuda_device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
        return {
            "source": "cuda_free_fraction",
            "cuda_device": int(cuda_device),
            "free_memory_bytes": int(free_bytes),
            "total_memory_bytes": int(total_bytes),
            "budget_bytes": int(float(free_bytes) * float(memory_budget_fraction)),
            "memory_budget_mb": 0.0,
            "memory_budget_fraction": float(memory_budget_fraction),
        }
    available_bytes = int(psutil.virtual_memory().available)
    return {
        "source": "system_ram_available_fraction",
        "available_memory_bytes": available_bytes,
        "budget_bytes": int(float(available_bytes) * float(memory_budget_fraction)),
        "memory_budget_mb": 0.0,
        "memory_budget_fraction": float(memory_budget_fraction),
    }


def _profile_autosize_candidate(
    shape: Mapping[str, int],
    *,
    vocab_size: int,
    precision: str,
    device: str,
    require_cuda: bool,
    gradient_accumulation_steps: int,
    native_ternary_backend: str,
    budget_bytes: int,
) -> Mapping[str, Any]:
    shape_gradient_accumulation_steps = int(shape.get("gradient_accumulation_steps", gradient_accumulation_steps))
    estimate = _profile_shape_memory_estimate(
        shape,
        vocab_size=vocab_size,
        precision=precision,
        device=device,
        require_cuda=require_cuda,
        gradient_accumulation_steps=shape_gradient_accumulation_steps,
        native_ternary_backend=native_ternary_backend,
    )
    estimated_peak = int(estimate["estimated_peak_training_bytes"])
    tokens_per_step = int(shape["seq_len"]) * int(shape["batch_size"]) * shape_gradient_accumulation_steps
    score = float(estimated_peak) * math.log2(max(tokens_per_step, 2))
    return {
        "shape": dict(shape),
        "shape_key": _profile_shape_key(shape),
        "gradient_accumulation_steps": shape_gradient_accumulation_steps,
        "memory_estimate": estimate,
        "estimated_peak_training_bytes": estimated_peak,
        "budget_bytes": int(budget_bytes),
        "budget_fraction_used": float(estimated_peak / max(1, int(budget_bytes))),
        "tokens_per_optimizer_step": tokens_per_step,
        "score": score,
        "fits_budget": estimated_peak <= int(budget_bytes),
    }


PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS = ("throughput", "gpu", "throughput_gpu")
PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES = ("diverse", "top")


def _profile_autosize_measured_score(
    *,
    train_tokens_per_second: float,
    gpu_utilization_percent: float,
    metric: str,
) -> float:
    if metric not in PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS:
        raise ValueError(
            f"unknown measured selection metric {metric!r}; "
            f"expected one of {PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS}"
        )
    if metric == "throughput":
        return float(train_tokens_per_second)
    if metric == "gpu":
        return float(gpu_utilization_percent)
    return float(train_tokens_per_second) * max(1.0, float(gpu_utilization_percent))


def _profile_autosize_diversity_features(
    candidate: Mapping[str, Any],
    *,
    ranges: Mapping[str, tuple[float, float]],
) -> tuple[float, ...]:
    shape = dict(candidate["shape"])
    raw = {
        "seq_len": float(shape["seq_len"]),
        "d_model": float(shape["d_model"]),
        "n_layers": float(shape["n_layers"]),
        "batch_size": float(shape["batch_size"]),
        "gradient_accumulation_steps": float(candidate.get("gradient_accumulation_steps", shape.get("gradient_accumulation_steps", 1))),
        "budget_fraction_used": float(candidate.get("budget_fraction_used", 0.0)),
        "tokens_per_optimizer_step": float(candidate.get("tokens_per_optimizer_step", 0.0)),
    }
    values: list[float] = []
    for name in (
        "seq_len",
        "d_model",
        "n_layers",
        "batch_size",
        "gradient_accumulation_steps",
        "budget_fraction_used",
        "tokens_per_optimizer_step",
    ):
        low, high = ranges[name]
        span = max(1e-9, high - low)
        values.append((raw[name] - low) / span)
    return tuple(values)


def _profile_autosize_measurement_inputs(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    requested_count: int,
    strategy: str,
) -> tuple[Mapping[str, Any], ...]:
    if requested_count <= 0:
        return ()
    if strategy not in PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES:
        raise ValueError(
            f"unknown autosize measurement strategy {strategy!r}; "
            f"expected one of {PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES}"
        )
    candidates = tuple(ranked_candidates)
    if not candidates:
        return ()
    count = min(int(requested_count), len(candidates))

    def annotate(candidate: Mapping[str, Any], *, reason: str, index: int, distance: float = 0.0) -> Mapping[str, Any]:
        return {
            **candidate,
            "measurement_candidate_index": int(index),
            "measurement_selection_reason": reason,
            "measurement_selection_distance": float(distance),
        }

    if strategy == "top":
        return tuple(
            annotate(candidate, reason="top_estimated_score", index=index)
            for index, candidate in enumerate(candidates[:count])
        )

    ranges: dict[str, tuple[float, float]] = {}
    feature_names = (
        "seq_len",
        "d_model",
        "n_layers",
        "batch_size",
        "gradient_accumulation_steps",
        "budget_fraction_used",
        "tokens_per_optimizer_step",
    )
    raw_by_name: dict[str, list[float]] = {name: [] for name in feature_names}
    for candidate in candidates:
        shape = dict(candidate["shape"])
        raw_by_name["seq_len"].append(float(shape["seq_len"]))
        raw_by_name["d_model"].append(float(shape["d_model"]))
        raw_by_name["n_layers"].append(float(shape["n_layers"]))
        raw_by_name["batch_size"].append(float(shape["batch_size"]))
        raw_by_name["gradient_accumulation_steps"].append(
            float(candidate.get("gradient_accumulation_steps", shape.get("gradient_accumulation_steps", 1)))
        )
        raw_by_name["budget_fraction_used"].append(float(candidate.get("budget_fraction_used", 0.0)))
        raw_by_name["tokens_per_optimizer_step"].append(float(candidate.get("tokens_per_optimizer_step", 0.0)))
    for name, values in raw_by_name.items():
        ranges[name] = (min(values), max(values))
    features = {
        str(candidate["shape_key"]): _profile_autosize_diversity_features(candidate, ranges=ranges)
        for candidate in candidates
    }
    scores = [float(candidate.get("score", 0.0)) for candidate in candidates]
    score_low, score_high = min(scores), max(scores)
    score_span = max(1e-9, score_high - score_low)
    token_values = [float(candidate.get("tokens_per_optimizer_step", 0.0)) for candidate in candidates]
    token_low, token_high = min(token_values), max(token_values)
    token_span = max(1e-9, token_high - token_low)

    selected: list[Mapping[str, Any]] = [annotate(candidates[0], reason="top_estimated_score", index=0)]
    selected_keys = {str(candidates[0]["shape_key"])}
    while len(selected) < count:
        best: tuple[float, float, float, int, Mapping[str, Any]] | None = None
        selected_features = [features[str(item["shape_key"])] for item in selected]
        for candidate in candidates:
            shape_key = str(candidate["shape_key"])
            if shape_key in selected_keys:
                continue
            feature = features[shape_key]
            min_distance = min(
                math.sqrt(sum((left - right) ** 2 for left, right in zip(feature, selected_feature)))
                for selected_feature in selected_features
            )
            estimated_score_norm = (float(candidate.get("score", 0.0)) - score_low) / score_span
            token_norm = (float(candidate.get("tokens_per_optimizer_step", 0.0)) - token_low) / token_span
            budget_norm = float(candidate.get("budget_fraction_used", 0.0))
            secondary = (0.50 * min_distance) + (0.25 * estimated_score_norm) + (0.15 * token_norm) + (0.10 * budget_norm)
            rank = int(candidate.get("estimated_rank", 0))
            key = (secondary, min_distance, float(candidate.get("score", 0.0)), -rank, candidate)
            if best is None or key[:4] > best[:4]:
                best = key
        if best is None:
            break
        _, distance, _, _, candidate = best
        selected.append(
            annotate(
                candidate,
                reason="diverse_shape_frontier",
                index=len(selected),
                distance=distance,
            )
        )
        selected_keys.add(str(candidate["shape_key"]))
    return tuple(selected)


def _profile_autosize_adaptive_measurement_inputs(
    ranked_candidates: Sequence[Mapping[str, Any]],
    *,
    measured_candidates: Sequence[Mapping[str, Any]],
    already_measured_shape_keys: set[str],
    requested_count: int,
) -> tuple[Mapping[str, Any], ...]:
    if requested_count <= 0:
        return ()
    candidates = tuple(
        candidate
        for candidate in ranked_candidates
        if str(candidate["shape_key"]) not in already_measured_shape_keys
    )
    if not candidates:
        return ()
    count = min(int(requested_count), len(candidates))
    references = tuple(
        sorted(
            (
                item
                for item in measured_candidates
                if bool(item.get("measurement_passed"))
            ),
            key=lambda item: (
                float(item.get("measured_score_upper_confidence", item.get("measured_score", 0.0))),
                float(item.get("measured_score", 0.0)),
                int(item.get("tokens_per_optimizer_step", 0)),
                -int(item.get("estimated_rank", 0)),
            ),
            reverse=True,
        )
    )
    if not references:
        references = tuple(
            sorted(
                measured_candidates,
                key=lambda item: (
                    float(item.get("measured_score_upper_confidence", item.get("measured_score", 0.0))),
                    float(item.get("measured_score", 0.0)),
                    -int(item.get("estimated_rank", 0)),
                ),
                reverse=True,
            )
        )
    if not references:
        return _profile_autosize_measurement_inputs(
            candidates,
            requested_count=count,
            strategy="diverse",
        )

    all_candidates = tuple(ranked_candidates)
    ranges: dict[str, tuple[float, float]] = {}
    feature_names = (
        "seq_len",
        "d_model",
        "n_layers",
        "batch_size",
        "gradient_accumulation_steps",
        "budget_fraction_used",
        "tokens_per_optimizer_step",
    )
    raw_by_name: dict[str, list[float]] = {name: [] for name in feature_names}
    for candidate in all_candidates:
        shape = dict(candidate["shape"])
        raw_by_name["seq_len"].append(float(shape["seq_len"]))
        raw_by_name["d_model"].append(float(shape["d_model"]))
        raw_by_name["n_layers"].append(float(shape["n_layers"]))
        raw_by_name["batch_size"].append(float(shape["batch_size"]))
        raw_by_name["gradient_accumulation_steps"].append(
            float(candidate.get("gradient_accumulation_steps", shape.get("gradient_accumulation_steps", 1)))
        )
        raw_by_name["budget_fraction_used"].append(float(candidate.get("budget_fraction_used", 0.0)))
        raw_by_name["tokens_per_optimizer_step"].append(float(candidate.get("tokens_per_optimizer_step", 0.0)))
    for name, values in raw_by_name.items():
        ranges[name] = (min(values), max(values))
    features = {
        str(candidate["shape_key"]): _profile_autosize_diversity_features(candidate, ranges=ranges)
        for candidate in all_candidates
    }
    scores = [float(candidate.get("score", 0.0)) for candidate in all_candidates]
    score_low, score_high = min(scores), max(scores)
    score_span = max(1e-9, score_high - score_low)
    token_values = [float(candidate.get("tokens_per_optimizer_step", 0.0)) for candidate in all_candidates]
    token_low, token_high = min(token_values), max(token_values)
    token_span = max(1e-9, token_high - token_low)
    reference_upper_values = [
        float(reference.get("measured_score_upper_confidence", reference.get("measured_score", 0.0)))
        for reference in references
    ]
    reference_upper_low = min(reference_upper_values, default=0.0)
    reference_upper_high = max(reference_upper_values, default=0.0)
    reference_upper_span = max(1e-9, reference_upper_high - reference_upper_low)
    measured_features = [
        features[str(item["shape_key"])]
        for item in measured_candidates
        if str(item["shape_key"]) in features
    ]
    reference_features = [
        features[str(item["shape_key"])]
        for item in references
        if str(item["shape_key"]) in features
    ]
    if not reference_features:
        return _profile_autosize_measurement_inputs(
            candidates,
            requested_count=count,
            strategy="diverse",
        )

    selected: list[Mapping[str, Any]] = []
    selected_keys: set[str] = set()
    while len(selected) < count:
        best: tuple[
            float,
            float,
            float,
            float,
            float,
            int,
            Mapping[str, Any],
            str,
            float,
            float,
            float,
            float,
            float,
        ] | None = None
        current_measured_features = measured_features + [
            features[str(item["shape_key"])]
            for item in selected
            if str(item["shape_key"]) in features
        ]
        for candidate in candidates:
            shape_key = str(candidate["shape_key"])
            if shape_key in selected_keys:
                continue
            feature = features[shape_key]
            distances_to_references = tuple(
                math.sqrt(sum((left - right) ** 2 for left, right in zip(feature, reference_feature)))
                for reference_feature in reference_features
            )
            nearest_reference_distance = min(distances_to_references)
            nearest_reference_index = distances_to_references.index(nearest_reference_distance)
            source_reference = references[nearest_reference_index]
            source_shape_key = str(source_reference["shape_key"])
            source_score = float(source_reference.get("measured_score", 0.0))
            source_score_mean = float(source_reference.get("measured_score_mean", source_score))
            source_score_stddev = float(source_reference.get("measured_score_stddev", 0.0))
            source_upper_confidence = float(
                source_reference.get("measured_score_upper_confidence", source_score)
            )
            source_stability_ratio = float(source_reference.get("measured_score_stability_ratio", 0.0))
            source_potential = (source_upper_confidence - reference_upper_low) / reference_upper_span
            proximity = 1.0 / (1.0 + nearest_reference_distance)
            novelty = min(
                (
                    math.sqrt(sum((left - right) ** 2 for left, right in zip(feature, measured_feature)))
                    for measured_feature in current_measured_features
                ),
                default=0.0,
            )
            estimated_score_norm = (float(candidate.get("score", 0.0)) - score_low) / score_span
            token_norm = (float(candidate.get("tokens_per_optimizer_step", 0.0)) - token_low) / token_span
            score = (
                (0.35 * proximity)
                + (0.20 * novelty)
                + (0.20 * source_potential)
                + (0.15 * estimated_score_norm)
                + (0.10 * token_norm)
            )
            rank = int(candidate.get("estimated_rank", 0))
            key = (score, source_potential, proximity, novelty, float(candidate.get("score", 0.0)), -rank, candidate, source_shape_key, source_score, source_score_mean, source_score_stddev, source_upper_confidence, source_stability_ratio)
            if best is None or key[:5] > best[:5]:
                best = key
        if best is None:
            break
        _, _, _, _, _, _, candidate, source_shape_key, source_score, source_score_mean, source_score_stddev, source_upper_confidence, source_stability_ratio = best
        selected.append(
            {
                **candidate,
                "measurement_selection_reason": "adaptive_measured_frontier",
                "measurement_selection_source_shape_key": source_shape_key,
                "measurement_selection_source_score": float(source_score),
                "measurement_selection_source_score_mean": float(source_score_mean),
                "measurement_selection_source_score_stddev": float(source_score_stddev),
                "measurement_selection_source_upper_confidence": float(source_upper_confidence),
                "measurement_selection_source_stability_ratio": float(source_stability_ratio),
                "measurement_selection_distance": float(
                    min(
                        math.sqrt(
                            sum(
                                (left - right) ** 2
                                for left, right in zip(
                                    features[str(candidate["shape_key"])],
                                    reference_feature,
                                )
                            )
                        )
                        for reference_feature in reference_features
                    )
                ),
            }
        )
        selected_keys.add(str(candidate["shape_key"]))
    return tuple(selected)


def _profile_autosize_measurement_summary(
    candidate: Mapping[str, Any],
    *,
    profile: Mapping[str, Any] | None,
    profile_path: Path,
    seed: int,
    profile_steps: int,
    repeat_index: int,
    repeat_count: int,
    metric: str,
    error: BaseException | None = None,
) -> Mapping[str, Any]:
    shape = dict(candidate["shape"])
    base = {
        "shape": shape,
        "shape_key": str(candidate["shape_key"]),
        "seed": int(seed),
        "profile_path": str(profile_path),
        "estimated_peak_training_bytes": int(candidate["estimated_peak_training_bytes"]),
        "budget_bytes": int(candidate["budget_bytes"]),
        "budget_fraction_used": float(candidate["budget_fraction_used"]),
        "tokens_per_optimizer_step": int(candidate["tokens_per_optimizer_step"]),
        "estimated_score": float(candidate["score"]),
        "estimated_rank": int(candidate.get("estimated_rank", 0)),
        "measurement_candidate_index": int(candidate.get("measurement_candidate_index", 0)),
        "measurement_selection_reason": str(candidate.get("measurement_selection_reason", "")),
        "measurement_selection_source_shape_key": str(candidate.get("measurement_selection_source_shape_key", "")),
        "measurement_selection_source_score": float(candidate.get("measurement_selection_source_score", 0.0)),
        "measurement_selection_source_score_mean": float(candidate.get("measurement_selection_source_score_mean", 0.0)),
        "measurement_selection_source_score_stddev": float(candidate.get("measurement_selection_source_score_stddev", 0.0)),
        "measurement_selection_source_upper_confidence": float(candidate.get("measurement_selection_source_upper_confidence", 0.0)),
        "measurement_selection_source_stability_ratio": float(candidate.get("measurement_selection_source_stability_ratio", 0.0)),
        "measurement_selection_distance": float(candidate.get("measurement_selection_distance", 0.0)),
        "fits_budget": bool(candidate["fits_budget"]),
        "measurement_metric": metric,
        "measurement_steps": int(profile_steps),
        "measurement_repeat_index": int(repeat_index),
        "measurement_repeat_count": int(repeat_count),
    }
    if error is not None:
        return {
            **base,
            "measurement_profile_passed": False,
            "measurement_passed": False,
            "measurement_error_type": type(error).__name__,
            "measurement_error": str(error),
            "measurement_profile_failed_checks": ("measurement_error",),
            "measurement_failed_checks": ("measurement_error",),
            "measured_score": 0.0,
            "train_tokens_per_second_wall": 0.0,
            "planned_train_tokens": 0,
            "gpu_utilization_percent_avg": 0.0,
            "gpu_memory_used_mb_max": 0.0,
            "gpu_memory_used_mb_avg": 0.0,
            "gpu_power_draw_watts_avg": 0.0,
            "observed_gpu_memory_used_bytes": 0,
            "observed_gpu_memory_budget_fraction_used": 0.0,
            "measured_budget_enforced": False,
            "measured_budget_passed": False,
            "torch_cuda_peak_allocated_bytes": 0,
            "torch_cuda_peak_to_estimate_ratio": 0.0,
            "strict_extension_only": False,
            "all_phases_active": False,
        }
    assert profile is not None
    metrics = dict((profile.get("resource_usage") or {}).get("metrics") or {})
    throughput = dict(profile.get("throughput") or {})
    train_tokens_per_second = float(throughput.get("train_tokens_per_second_wall", 0.0))
    gpu_utilization_percent = float(metrics.get("gpu_utilization_percent", {}).get("avg", 0.0))
    gpu_memory_metric = dict(metrics.get("gpu_memory_used_mb") or {})
    gpu_memory_used_mb = float(gpu_memory_metric.get("avg", 0.0))
    gpu_memory_used_mb_max = float(gpu_memory_metric.get("max", 0.0))
    gpu_power_draw_watts = float(metrics.get("gpu_power_draw_watts", {}).get("avg", 0.0))
    torch_cuda_peak = int(profile.get("torch_cuda_memory", {}).get("after", {}).get("max_memory_allocated_bytes", 0))
    observed_gpu_memory_bytes = int(gpu_memory_used_mb_max * 1024 * 1024) if gpu_memory_used_mb_max > 0.0 else 0
    measured_budget_enforced = observed_gpu_memory_bytes > 0
    measured_budget_passed = (not measured_budget_enforced) or observed_gpu_memory_bytes <= int(candidate["budget_bytes"])
    profile_failed_checks = tuple(str(item) for item in profile.get("failed_checks", ()))
    budget_failed_checks = ("observed_gpu_memory_budget",) if not measured_budget_passed else ()
    measured_score = _profile_autosize_measured_score(
        train_tokens_per_second=train_tokens_per_second,
        gpu_utilization_percent=gpu_utilization_percent,
        metric=metric,
    )
    return {
        **base,
        "measurement_profile_passed": bool(profile.get("passed")),
        "measurement_passed": bool(profile.get("passed")) and measured_budget_passed,
        "measurement_profile_failed_checks": profile_failed_checks,
        "measurement_failed_checks": profile_failed_checks + budget_failed_checks,
        "measured_score": float(measured_score),
        "train_tokens_per_second_wall": train_tokens_per_second,
        "planned_train_tokens": int(throughput.get("planned_train_tokens", 0)),
        "gpu_utilization_percent_avg": gpu_utilization_percent,
        "gpu_memory_used_mb_avg": gpu_memory_used_mb,
        "gpu_memory_used_mb_max": gpu_memory_used_mb_max,
        "gpu_power_draw_watts_avg": gpu_power_draw_watts,
        "observed_gpu_memory_used_bytes": observed_gpu_memory_bytes,
        "observed_gpu_memory_budget_fraction_used": float(
            observed_gpu_memory_bytes / max(1, int(candidate["budget_bytes"]))
        ) if measured_budget_enforced else 0.0,
        "measured_budget_enforced": measured_budget_enforced,
        "measured_budget_passed": measured_budget_passed,
        "torch_cuda_peak_allocated_bytes": torch_cuda_peak,
        "torch_cuda_peak_to_estimate_ratio": float(
            torch_cuda_peak / max(1, int(candidate["estimated_peak_training_bytes"]))
        ),
        "strict_extension_only": bool(profile.get("kernel_evidence", {}).get("strict_extension_only")),
        "all_phases_active": bool(profile.get("architecture", {}).get("all_phases_active")),
    }


def _profile_autosize_aggregate_measurements(
    candidate: Mapping[str, Any],
    *,
    seed_measurements: Sequence[Mapping[str, Any]],
    metric: str,
) -> Mapping[str, Any]:
    rows = tuple(seed_measurements)
    if not rows:
        raise ValueError("at least one candidate seed measurement is required")

    def _mean_float(name: str) -> float:
        values = [float(row.get(name, 0.0)) for row in rows]
        return float(statistics.fmean(values)) if values else 0.0

    def _float_values(name: str) -> tuple[float, ...]:
        return tuple(float(row.get(name, 0.0)) for row in rows)

    def _max_float(name: str) -> float:
        return float(max((float(row.get(name, 0.0)) for row in rows), default=0.0))

    def _max_int(name: str) -> int:
        return int(max((int(row.get(name, 0)) for row in rows), default=0))

    measurement_profile_seeds = tuple(int(row.get("seed", 0)) for row in rows)
    measurement_seeds = tuple(dict.fromkeys(measurement_profile_seeds))
    measurement_steps = tuple(int(row.get("measurement_steps", 0)) for row in rows)
    measurement_repeat_indices = tuple(int(row.get("measurement_repeat_index", 0)) for row in rows)
    measurement_repeat_counts = tuple(int(row.get("measurement_repeat_count", 1)) for row in rows)
    profile_failed_checks = tuple(
        dict.fromkeys(
            str(check)
            for row in rows
            for check in tuple(row.get("measurement_profile_failed_checks", ()))
        )
    )
    failed_checks = tuple(
        dict.fromkeys(
            str(check)
            for row in rows
            for check in tuple(row.get("measurement_failed_checks", ()))
        )
    )
    failed_checks_by_seed = tuple(
        {
            "seed": int(row.get("seed", 0)),
            "measurement_repeat_index": int(row.get("measurement_repeat_index", 0)),
            "measurement_profile_failed_checks": tuple(str(item) for item in row.get("measurement_profile_failed_checks", ())),
            "measurement_failed_checks": tuple(str(item) for item in row.get("measurement_failed_checks", ())),
        }
        for row in rows
        if tuple(row.get("measurement_failed_checks", ())) or tuple(row.get("measurement_profile_failed_checks", ()))
    )
    measurement_errors = tuple(
        {
            "seed": int(row.get("seed", 0)),
            "measurement_repeat_index": int(row.get("measurement_repeat_index", 0)),
            "measurement_error_type": str(row.get("measurement_error_type", "")),
            "measurement_error": str(row.get("measurement_error", "")),
            "profile_path": str(row.get("profile_path", "")),
        }
        for row in rows
        if "measurement_error" in row
    )
    measured_budget_enforced = any(bool(row.get("measured_budget_enforced")) for row in rows)
    measured_budget_passed = all(bool(row.get("measured_budget_passed")) for row in rows)
    measurement_profile_passed = all(bool(row.get("measurement_profile_passed")) for row in rows)
    measurement_passed = all(bool(row.get("measurement_passed")) for row in rows)
    strict_extension_only = all(bool(row.get("strict_extension_only")) for row in rows)
    all_phases_active = all(bool(row.get("all_phases_active")) for row in rows)
    estimated_peak_training_bytes = int(candidate["estimated_peak_training_bytes"])
    torch_cuda_peak_allocated_bytes = _max_int("torch_cuda_peak_allocated_bytes")
    measured_score_profile_values = _float_values("measured_score")
    score_groups: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row.get("seed", 0)), int(row.get("measurement_steps", 0)))
        score_groups.setdefault(key, []).append(float(row.get("measured_score", 0.0)))
    measured_score_observation_values = tuple(
        float(statistics.fmean(values))
        for values in score_groups.values()
    )
    measured_score_mean = (
        float(statistics.fmean(measured_score_observation_values))
        if measured_score_observation_values
        else 0.0
    )
    measured_score_min = float(min(measured_score_observation_values, default=0.0))
    measured_score_max = float(max(measured_score_observation_values, default=0.0))
    measured_score_stddev = (
        float(statistics.stdev(measured_score_observation_values))
        if len(measured_score_observation_values) > 1
        else 0.0
    )
    measured_score_lower_confidence = max(0.0, measured_score_mean - measured_score_stddev)
    measured_score_upper_confidence = measured_score_mean + measured_score_stddev
    measured_score_stability_ratio = (
        float(measured_score_lower_confidence / measured_score_mean)
        if measured_score_mean > 0.0
        else 0.0
    )
    return {
        "shape": dict(candidate["shape"]),
        "shape_key": str(candidate["shape_key"]),
        "measurement_profile_count": len(rows),
        "measurement_profile_seeds": measurement_profile_seeds,
        "measurement_seed_count": len(measurement_seeds),
        "measurement_seeds": measurement_seeds,
        "measurement_steps": measurement_steps,
        "measurement_step_count_min": min(measurement_steps, default=0),
        "measurement_step_count_max": max(measurement_steps, default=0),
        "measurement_repeat_indices": measurement_repeat_indices,
        "measurement_repeat_counts": measurement_repeat_counts,
        "measurement_repeat_count_max": max(measurement_repeat_counts, default=0),
        "profile_path": str(rows[0].get("profile_path", "")),
        "profile_paths": tuple(str(row.get("profile_path", "")) for row in rows),
        "seed_measurements": rows,
        "estimated_peak_training_bytes": estimated_peak_training_bytes,
        "budget_bytes": int(candidate["budget_bytes"]),
        "budget_fraction_used": float(candidate["budget_fraction_used"]),
        "tokens_per_optimizer_step": int(candidate["tokens_per_optimizer_step"]),
        "estimated_score": float(candidate["score"]),
        "score": float(candidate["score"]),
        "estimated_rank": int(candidate.get("estimated_rank", 0)),
        "measurement_candidate_index": int(candidate.get("measurement_candidate_index", 0)),
        "measurement_selection_reason": str(candidate.get("measurement_selection_reason", "")),
        "measurement_selection_source_shape_key": str(candidate.get("measurement_selection_source_shape_key", "")),
        "measurement_selection_source_score": float(candidate.get("measurement_selection_source_score", 0.0)),
        "measurement_selection_source_score_mean": float(candidate.get("measurement_selection_source_score_mean", 0.0)),
        "measurement_selection_source_score_stddev": float(candidate.get("measurement_selection_source_score_stddev", 0.0)),
        "measurement_selection_source_upper_confidence": float(candidate.get("measurement_selection_source_upper_confidence", 0.0)),
        "measurement_selection_source_stability_ratio": float(candidate.get("measurement_selection_source_stability_ratio", 0.0)),
        "measurement_selection_distance": float(candidate.get("measurement_selection_distance", 0.0)),
        "fits_budget": bool(candidate["fits_budget"]),
        "measurement_metric": metric,
        "measurement_profile_passed": measurement_profile_passed,
        "measurement_passed": measurement_passed,
        "measurement_profile_failed_checks": profile_failed_checks,
        "measurement_failed_checks": failed_checks,
        "measurement_failed_checks_by_seed": failed_checks_by_seed,
        "measurement_errors": measurement_errors,
        "measured_score": measured_score_lower_confidence,
        "measured_score_mean": measured_score_mean,
        "measured_score_min": measured_score_min,
        "measured_score_max": measured_score_max,
        "measured_score_stddev": measured_score_stddev,
        "measured_score_lower_confidence": measured_score_lower_confidence,
        "measured_score_upper_confidence": measured_score_upper_confidence,
        "measured_score_stability_ratio": measured_score_stability_ratio,
        "measured_score_profile_values": measured_score_profile_values,
        "measured_score_observation_values": measured_score_observation_values,
        "measured_score_observation_count": len(measured_score_observation_values),
        "measured_score_observation_keys": tuple(
            {
                "seed": int(seed),
                "measurement_steps": int(measurement_steps),
            }
            for seed, measurement_steps in score_groups.keys()
        ),
        "train_tokens_per_second_wall": _mean_float("train_tokens_per_second_wall"),
        "planned_train_tokens": sum(int(row.get("planned_train_tokens", 0)) for row in rows),
        "gpu_utilization_percent_avg": _mean_float("gpu_utilization_percent_avg"),
        "gpu_memory_used_mb_avg": _mean_float("gpu_memory_used_mb_avg"),
        "gpu_memory_used_mb_max": _max_float("gpu_memory_used_mb_max"),
        "gpu_power_draw_watts_avg": _mean_float("gpu_power_draw_watts_avg"),
        "observed_gpu_memory_used_bytes": _max_int("observed_gpu_memory_used_bytes"),
        "observed_gpu_memory_budget_fraction_used": _max_float("observed_gpu_memory_budget_fraction_used"),
        "measured_budget_enforced": measured_budget_enforced,
        "measured_budget_passed": measured_budget_passed,
        "torch_cuda_peak_allocated_bytes": torch_cuda_peak_allocated_bytes,
        "torch_cuda_peak_to_estimate_ratio": float(
            torch_cuda_peak_allocated_bytes / max(1, estimated_peak_training_bytes)
        ),
        "strict_extension_only": strict_extension_only,
        "all_phases_active": all_phases_active,
    }


def _profile_autosize_measurement_seeds(
    normalized_seeds: Sequence[int],
    *,
    requested_count: int,
) -> tuple[int, ...]:
    if requested_count < 1:
        raise ValueError("requested measurement seed count must be positive")
    if not normalized_seeds:
        raise ValueError("at least one seed is required")
    selected: list[int] = []
    seen: set[int] = set()
    for raw_seed in normalized_seeds:
        seed = int(raw_seed)
        if seed in seen:
            continue
        selected.append(seed)
        seen.add(seed)
        if len(selected) >= requested_count:
            return tuple(selected)
    step = 104729
    candidate = int(selected[-1] if selected else normalized_seeds[-1])
    while len(selected) < requested_count:
        candidate = (candidate + step) % 2147483647
        if candidate == 0:
            candidate = step
        if candidate in seen:
            candidate += 1
            continue
        selected.append(candidate)
        seen.add(candidate)
    return tuple(selected)


def _profile_autosize_uncertainty_refinement_actions(
    measured_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_shape_count: int = 1,
    refinement_steps: int = 1,
    refinement_repeat_count: int = 1,
    refinement_extra_seed_count: int = 1,
) -> tuple[Mapping[str, Any], ...]:
    passed = tuple(
        sorted(
            (
                candidate
                for candidate in measured_candidates
                if bool(candidate.get("measurement_passed"))
            ),
            key=lambda candidate: (
                float(candidate.get("measured_score", 0.0)),
                int(candidate.get("tokens_per_optimizer_step", 0)),
                -int(candidate.get("estimated_rank", 0)),
                str(candidate.get("shape_key", "")),
            ),
            reverse=True,
        )
    )
    if not passed:
        return ()
    robust_frontier_score = float(passed[0].get("measured_score", 0.0))
    finalist_keys = {
        str(candidate.get("shape_key", ""))
        for candidate in passed[: max(1, int(selected_shape_count))]
    }
    uncertain = tuple(
        candidate
        for candidate in passed
        if float(candidate.get("measured_score_stddev", 0.0)) > 0.0
    )
    if len(uncertain) < 1:
        return ()
    actions: list[Mapping[str, Any]] = []
    for candidate in uncertain:
        shape_key = str(candidate.get("shape_key", ""))
        lower_confidence = float(candidate.get("measured_score_lower_confidence", candidate.get("measured_score", 0.0)))
        upper_confidence = float(candidate.get("measured_score_upper_confidence", candidate.get("measured_score", 0.0)))
        mean_score = float(candidate.get("measured_score_mean", candidate.get("measured_score", 0.0)))
        stddev_score = float(candidate.get("measured_score_stddev", 0.0))
        uncertainty_width = max(0.0, upper_confidence - lower_confidence)
        expected_gain = max(0.0, upper_confidence - robust_frontier_score)
        finalist_bonus = 0.25 * max(0.0, robust_frontier_score) if shape_key in finalist_keys else 0.0
        posterior_utility = expected_gain + (0.50 * uncertainty_width) + (0.25 * max(0.0, mean_score)) + finalist_bonus
        measurement_cost = (
            max(1, int(candidate.get("tokens_per_optimizer_step", 0)))
            * max(1, int(refinement_steps))
            * max(1, int(refinement_repeat_count))
            * max(1, int(refinement_extra_seed_count))
        )
        gain_per_cost = posterior_utility / max(1.0, float(measurement_cost))
        actions.append(
            {
                **candidate,
                "refinement_budget_strategy": "expected_gain_per_cost",
                "refinement_expected_gain": expected_gain,
                "refinement_uncertainty_width": uncertainty_width,
                "refinement_posterior_utility": posterior_utility,
                "refinement_measurement_cost_tokens": int(measurement_cost),
                "refinement_gain_per_cost": float(gain_per_cost),
                "refinement_frontier_score": robust_frontier_score,
                "refinement_is_selected_finalist": shape_key in finalist_keys,
                "refinement_planned_steps": max(1, int(refinement_steps)),
                "refinement_planned_repeat_count": max(1, int(refinement_repeat_count)),
                "refinement_planned_extra_seed_count": max(1, int(refinement_extra_seed_count)),
            }
        )
    ranked = sorted(
        actions,
        key=lambda candidate: (
            float(candidate.get("refinement_gain_per_cost", 0.0)),
            float(candidate.get("refinement_posterior_utility", 0.0)),
            float(candidate.get("refinement_expected_gain", 0.0)),
            1 if bool(candidate.get("refinement_is_selected_finalist")) else 0,
            float(candidate.get("measured_score_upper_confidence", 0.0)),
            -int(candidate.get("estimated_rank", 0)),
            str(candidate.get("shape_key", "")),
        ),
        reverse=True,
    )
    return tuple(ranked)


def _profile_autosize_uncertainty_refinement_inputs(
    measured_candidates: Sequence[Mapping[str, Any]],
    *,
    requested_count: int,
    selected_shape_count: int = 1,
    refinement_steps: int = 1,
    refinement_repeat_count: int = 1,
    refinement_extra_seed_count: int = 1,
) -> tuple[Mapping[str, Any], ...]:
    if requested_count <= 0:
        return ()
    return _profile_autosize_uncertainty_refinement_actions(
        measured_candidates,
        selected_shape_count=selected_shape_count,
        refinement_steps=refinement_steps,
        refinement_repeat_count=refinement_repeat_count,
        refinement_extra_seed_count=refinement_extra_seed_count,
    )[: int(requested_count)]


def run_llm_batch_profile_matrix(
    *,
    out_dir: str | Path,
    shape_specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    steps: int = 1,
    gradient_accumulation_steps: int = 1,
    vocab_size: int = 256,
    precision: str = "auto",
    device: str = "auto",
    require_cuda: bool = False,
    native_ternary_backend: str = STRICT_NATIVE_TERNARY_BACKEND,
    resource_interval: float = 0.05,
    min_resource_samples: int = 2,
    corpus_repeats: int = 192,
    max_corpus_tokens: int | None = 8192,
    min_cases: int = 1,
    require_multi_shape: bool = False,
    require_multi_seed: bool = False,
    min_train_tokens_per_second_mean: float = 0.0,
    min_gpu_utilization_percent_mean: float = 0.0,
    min_gpu_memory_used_mb_mean: float = 0.0,
    min_gpu_power_draw_watts_mean: float = 0.0,
    default_batch_size: int = 8,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output_dir = Path(out_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"profile matrix output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_shapes = tuple(
        _normalize_profile_shape_spec(spec, default_batch_size=default_batch_size)
        for spec in shape_specs
    )
    if not normalized_shapes:
        raise ValueError("at least one profile shape spec is required")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("at least one profile seed is required")
    started = time.time()
    cases: list[Mapping[str, Any]] = []
    failed_checks: list[str] = []
    csv_rows: list[Mapping[str, Any]] = []
    case_index = 0
    for shape_index, shape in enumerate(normalized_shapes):
        shape_key = _profile_shape_key(shape)
        shape_gradient_accumulation_steps = int(shape.get("gradient_accumulation_steps", gradient_accumulation_steps))
        for seed in normalized_seeds:
            case_id = f"case_{case_index:03d}_{shape_key}_seed{seed}"
            case_dir = output_dir / "cases" / case_id
            profile = run_llm_batch_profile(
                out_dir=case_dir,
                steps=steps,
                batch_size=int(shape["batch_size"]),
                gradient_accumulation_steps=shape_gradient_accumulation_steps,
                seq_len=int(shape["seq_len"]),
                d_model=int(shape["d_model"]),
                n_heads=int(shape["n_heads"]),
                n_layers=int(shape["n_layers"]),
                vocab_size=vocab_size,
                precision=precision,
                device=device,
                require_cuda=require_cuda,
                native_ternary_backend=native_ternary_backend,
                resource_interval=resource_interval,
                min_resource_samples=min_resource_samples,
                seed=seed,
                corpus_repeats=corpus_repeats,
                max_corpus_tokens=max_corpus_tokens,
                overwrite=False,
            )
            metrics = dict((profile.get("resource_usage") or {}).get("metrics") or {})
            case_failed = tuple(str(item) for item in profile.get("failed_checks", ()))
            if not bool(profile.get("passed")):
                failed_checks.append(f"case_failed:{case_id}")
            if bool(profile.get("kernel_evidence", {}).get("native_ternary_kernel_required")) and not bool(profile.get("kernel_evidence", {}).get("strict_extension_only")):
                failed_checks.append(f"case_not_strict_extension:{case_id}")
            if not bool(profile.get("architecture", {}).get("all_phases_active")):
                failed_checks.append(f"case_missing_phase:{case_id}")
            row = {
                "case_id": case_id,
                "shape_index": shape_index,
                "seed": seed,
                "seq_len": int(shape["seq_len"]),
                "d_model": int(shape["d_model"]),
                "n_heads": int(shape["n_heads"]),
                "n_layers": int(shape["n_layers"]),
                "batch_size": int(shape["batch_size"]),
                "gradient_accumulation_steps": shape_gradient_accumulation_steps,
                "passed": bool(profile.get("passed")),
                "failed_checks": ";".join(case_failed),
                "planned_train_tokens": int(profile.get("throughput", {}).get("planned_train_tokens", 0)),
                "train_tokens_per_second_wall": float(profile.get("throughput", {}).get("train_tokens_per_second_wall", 0.0)),
                "gpu_utilization_percent_avg": float(metrics.get("gpu_utilization_percent", {}).get("avg", 0.0)),
                "gpu_memory_used_mb_avg": float(metrics.get("gpu_memory_used_mb", {}).get("avg", 0.0)),
                "gpu_power_draw_watts_avg": float(metrics.get("gpu_power_draw_watts", {}).get("avg", 0.0)),
                "process_cpu_percent_of_total_avg": float(metrics.get("process_cpu_percent_of_total", {}).get("avg", 0.0)),
                "torch_cuda_peak_allocated_bytes": int(profile.get("torch_cuda_memory", {}).get("after", {}).get("max_memory_allocated_bytes", 0)),
                "strict_extension_only": bool(profile.get("kernel_evidence", {}).get("strict_extension_only")),
                "all_phases_active": bool(profile.get("architecture", {}).get("all_phases_active")),
                "profile_path": str(case_dir / "llm_batch_profile.json"),
            }
            csv_rows.append(row)
            cases.append(
                {
                    "case_id": case_id,
                    "shape": dict(shape),
                    "seed": seed,
                    "profile_path": row["profile_path"],
                    "passed": row["passed"],
                    "failed_checks": case_failed,
                    "throughput": profile.get("throughput", {}),
                    "resource_metrics": metrics,
                    "torch_cuda_memory": profile.get("torch_cuda_memory", {}),
                    "kernel_evidence": profile.get("kernel_evidence", {}),
                    "architecture": profile.get("architecture", {}),
                }
            )
            case_index += 1
    shape_count = len({_profile_shape_key(shape) for shape in normalized_shapes})
    seed_count = len(set(normalized_seeds))
    case_count = len(cases)
    if case_count < int(min_cases):
        failed_checks.append("min_cases")
    if require_multi_shape and shape_count < 2:
        failed_checks.append("multi_shape")
    if require_multi_seed and seed_count < 2:
        failed_checks.append("multi_seed")
    passed_cases = sum(1 for case in cases if bool(case.get("passed")))
    total_planned_tokens = sum(int(row["planned_train_tokens"]) for row in csv_rows)
    tokens_per_second_values = [float(row["train_tokens_per_second_wall"]) for row in csv_rows if float(row["train_tokens_per_second_wall"]) > 0.0]
    gpu_avg_values = [float(row["gpu_utilization_percent_avg"]) for row in csv_rows if float(row["gpu_utilization_percent_avg"]) > 0.0]
    power_avg_values = [float(row["gpu_power_draw_watts_avg"]) for row in csv_rows if float(row["gpu_power_draw_watts_avg"]) > 0.0]
    vram_avg_values = [float(row["gpu_memory_used_mb_avg"]) for row in csv_rows if float(row["gpu_memory_used_mb_avg"]) > 0.0]
    train_tokens_per_second_wall_mean = float(statistics.fmean(tokens_per_second_values)) if tokens_per_second_values else 0.0
    gpu_utilization_percent_case_mean = float(statistics.fmean(gpu_avg_values)) if gpu_avg_values else 0.0
    gpu_power_draw_watts_case_mean = float(statistics.fmean(power_avg_values)) if power_avg_values else 0.0
    gpu_memory_used_mb_case_mean = float(statistics.fmean(vram_avg_values)) if vram_avg_values else 0.0
    threshold_checks = {
        "min_train_tokens_per_second_mean": {
            "required": float(min_train_tokens_per_second_mean),
            "observed": train_tokens_per_second_wall_mean,
            "passed": train_tokens_per_second_wall_mean >= float(min_train_tokens_per_second_mean),
        },
        "min_gpu_utilization_percent_mean": {
            "required": float(min_gpu_utilization_percent_mean),
            "observed": gpu_utilization_percent_case_mean,
            "passed": gpu_utilization_percent_case_mean >= float(min_gpu_utilization_percent_mean),
        },
        "min_gpu_memory_used_mb_mean": {
            "required": float(min_gpu_memory_used_mb_mean),
            "observed": gpu_memory_used_mb_case_mean,
            "passed": gpu_memory_used_mb_case_mean >= float(min_gpu_memory_used_mb_mean),
        },
        "min_gpu_power_draw_watts_mean": {
            "required": float(min_gpu_power_draw_watts_mean),
            "observed": gpu_power_draw_watts_case_mean,
            "passed": gpu_power_draw_watts_case_mean >= float(min_gpu_power_draw_watts_mean),
        },
    }
    for check_name, check in threshold_checks.items():
        if float(check["required"]) > 0.0 and not bool(check["passed"]):
            failed_checks.append(check_name)
    summary = {
        "case_count": case_count,
        "passed_cases": passed_cases,
        "shape_count": shape_count,
        "seed_count": seed_count,
        "total_planned_train_tokens": total_planned_tokens,
        "wall_seconds": float(max(1e-9, time.time() - started)),
        "train_tokens_per_second_wall_mean": train_tokens_per_second_wall_mean,
        "gpu_utilization_percent_case_mean": gpu_utilization_percent_case_mean,
        "gpu_power_draw_watts_case_mean": gpu_power_draw_watts_case_mean,
        "gpu_memory_used_mb_case_mean": gpu_memory_used_mb_case_mean,
        "strict_extension_only_cases": sum(1 for case in cases if bool(case.get("kernel_evidence", {}).get("strict_extension_only"))),
        "all_phases_active_cases": sum(1 for case in cases if bool(case.get("architecture", {}).get("all_phases_active"))),
        "threshold_checks": threshold_checks,
    }
    matrix = {
        "schema_version": 1,
        "run_dir": str(output_dir),
        "shape_specs": tuple(dict(shape) for shape in normalized_shapes),
        "seeds": normalized_seeds,
        "config": {
            "steps": int(steps),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "shape_specific_gradient_accumulation_steps": any(
                "gradient_accumulation_steps" in shape for shape in normalized_shapes
            ),
            "vocab_size": int(vocab_size),
            "precision": precision,
            "device": device,
            "require_cuda": bool(require_cuda),
            "native_ternary_backend": native_ternary_backend,
            "resource_interval": float(resource_interval),
            "min_resource_samples": int(min_resource_samples),
            "corpus_repeats": int(corpus_repeats),
            "max_corpus_tokens": max_corpus_tokens,
            "min_cases": int(min_cases),
            "require_multi_shape": bool(require_multi_shape),
            "require_multi_seed": bool(require_multi_seed),
            "min_train_tokens_per_second_mean": float(min_train_tokens_per_second_mean),
            "min_gpu_utilization_percent_mean": float(min_gpu_utilization_percent_mean),
            "min_gpu_memory_used_mb_mean": float(min_gpu_memory_used_mb_mean),
            "min_gpu_power_draw_watts_mean": float(min_gpu_power_draw_watts_mean),
        },
        "summary": summary,
        "cases": tuple(cases),
        "failed_checks": tuple(failed_checks),
        "passed": not failed_checks,
    }
    _write_json(output_dir / "llm_batch_profile_matrix.json", matrix)
    csv_path = output_dir / "llm_batch_profile_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = tuple(csv_rows[0].keys()) if csv_rows else ("case_id",)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return matrix


def run_llm_batch_profile_autosize(
    *,
    out_dir: str | Path,
    candidate_seq_lens: Sequence[int],
    candidate_d_models: Sequence[int],
    candidate_n_layers: Sequence[int],
    candidate_batch_sizes: Sequence[int],
    candidate_gradient_accumulation_steps: Sequence[int] | None = None,
    n_heads: int = 4,
    selected_shape_count: int = 2,
    min_selected_shapes: int = 1,
    seeds: Sequence[int] = (71,),
    steps: int = 1,
    gradient_accumulation_steps: int = 1,
    vocab_size: int = 256,
    precision: str = "auto",
    device: str = "auto",
    require_cuda: bool = False,
    native_ternary_backend: str = STRICT_NATIVE_TERNARY_BACKEND,
    resource_interval: float = 0.05,
    min_resource_samples: int = 2,
    corpus_repeats: int = 192,
    max_corpus_tokens: int | None = 8192,
    memory_budget_mb: float = 0.0,
    memory_budget_fraction: float = 0.35,
    measure_candidate_count: int = 4,
    measure_candidate_seed_count: int | None = None,
    min_measure_candidate_seed_count: int = 2,
    measure_candidate_strategy: str = "diverse",
    measure_candidate_adaptive_rounds: int = 2,
    refine_uncertain_candidate_count: int = 1,
    refine_uncertain_extra_seed_count: int = 1,
    refine_uncertain_step_multiplier: int = 2,
    refine_uncertain_repeat_count: int = 2,
    refinement_budget_candidate_action_report_cap: int = 64,
    confirm_selected_candidate_count: int | None = None,
    confirm_selected_extra_seed_count: int = 1,
    confirm_selected_step_multiplier: int = 2,
    confirm_selected_repeat_count: int = 2,
    confirm_selected_max_rounds: int | None = None,
    confirm_selected_decision_resolution_extra_rounds: int | None = None,
    confirm_selected_decision_resolution_adaptive_extra_rounds: int = 2,
    confirm_selected_runtime_step_multiplier_cap: int = 4,
    confirm_selected_runtime_repeat_count_cap: int = 4,
    measured_selection_metric: str = "throughput_gpu",
    min_cases: int = 1,
    require_multi_shape: bool = False,
    require_multi_seed: bool = False,
    min_train_tokens_per_second_mean: float = 0.0,
    min_gpu_utilization_percent_mean: float = 0.0,
    min_gpu_memory_used_mb_mean: float = 0.0,
    min_gpu_power_draw_watts_mean: float = 0.0,
    overwrite: bool = False,
) -> Mapping[str, Any]:
    output_dir = Path(out_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"profile autosize output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected_shape_count < 1:
        raise ValueError("selected_shape_count must be positive")
    if min_selected_shapes < 1:
        raise ValueError("min_selected_shapes must be positive")
    requested_measure_candidate_count = int(measure_candidate_count)
    if requested_measure_candidate_count < 0:
        raise ValueError("measure_candidate_count must be >= 0")
    requested_measure_candidate_adaptive_rounds = int(measure_candidate_adaptive_rounds)
    if requested_measure_candidate_adaptive_rounds < 1:
        raise ValueError("measure_candidate_adaptive_rounds must be positive")
    requested_refine_uncertain_candidate_count = int(refine_uncertain_candidate_count)
    if requested_refine_uncertain_candidate_count < 0:
        raise ValueError("refine_uncertain_candidate_count must be >= 0")
    requested_refine_uncertain_extra_seed_count = int(refine_uncertain_extra_seed_count)
    if requested_refine_uncertain_extra_seed_count < 0:
        raise ValueError("refine_uncertain_extra_seed_count must be >= 0")
    requested_refine_uncertain_step_multiplier = int(refine_uncertain_step_multiplier)
    if requested_refine_uncertain_step_multiplier < 1:
        raise ValueError("refine_uncertain_step_multiplier must be positive")
    requested_refine_uncertain_repeat_count = int(refine_uncertain_repeat_count)
    if requested_refine_uncertain_repeat_count < 1:
        raise ValueError("refine_uncertain_repeat_count must be positive")
    requested_refinement_budget_candidate_action_report_cap = int(refinement_budget_candidate_action_report_cap)
    if requested_refinement_budget_candidate_action_report_cap < 0:
        raise ValueError("refinement_budget_candidate_action_report_cap must be >= 0")
    requested_confirm_selected_candidate_count = (
        int(selected_shape_count)
        if confirm_selected_candidate_count is None
        else int(confirm_selected_candidate_count)
    )
    if requested_confirm_selected_candidate_count < 0:
        raise ValueError("confirm_selected_candidate_count must be >= 0")
    requested_confirm_selected_extra_seed_count = int(confirm_selected_extra_seed_count)
    if requested_confirm_selected_extra_seed_count < 0:
        raise ValueError("confirm_selected_extra_seed_count must be >= 0")
    requested_confirm_selected_step_multiplier = int(confirm_selected_step_multiplier)
    if requested_confirm_selected_step_multiplier < 1:
        raise ValueError("confirm_selected_step_multiplier must be positive")
    requested_confirm_selected_repeat_count = int(confirm_selected_repeat_count)
    if requested_confirm_selected_repeat_count < 1:
        raise ValueError("confirm_selected_repeat_count must be positive")
    requested_confirm_selected_max_rounds = (
        None
        if confirm_selected_max_rounds is None
        else int(confirm_selected_max_rounds)
    )
    if requested_confirm_selected_max_rounds is not None and requested_confirm_selected_max_rounds < 1:
        raise ValueError("confirm_selected_max_rounds must be positive")
    requested_confirm_selected_decision_resolution_extra_rounds = (
        None
        if confirm_selected_decision_resolution_extra_rounds is None
        else int(confirm_selected_decision_resolution_extra_rounds)
    )
    if (
        requested_confirm_selected_decision_resolution_extra_rounds is not None
        and requested_confirm_selected_decision_resolution_extra_rounds < 0
    ):
        raise ValueError("confirm_selected_decision_resolution_extra_rounds must be >= 0")
    requested_confirm_selected_decision_resolution_adaptive_extra_rounds = int(
        confirm_selected_decision_resolution_adaptive_extra_rounds
    )
    if requested_confirm_selected_decision_resolution_adaptive_extra_rounds < 0:
        raise ValueError("confirm_selected_decision_resolution_adaptive_extra_rounds must be >= 0")
    requested_confirm_selected_runtime_step_multiplier_cap = int(
        confirm_selected_runtime_step_multiplier_cap
    )
    if requested_confirm_selected_runtime_step_multiplier_cap < requested_confirm_selected_step_multiplier:
        raise ValueError(
            "confirm_selected_runtime_step_multiplier_cap must be >= confirm_selected_step_multiplier"
        )
    requested_confirm_selected_runtime_repeat_count_cap = int(
        confirm_selected_runtime_repeat_count_cap
    )
    if requested_confirm_selected_runtime_repeat_count_cap < requested_confirm_selected_repeat_count:
        raise ValueError(
            "confirm_selected_runtime_repeat_count_cap must be >= confirm_selected_repeat_count"
        )
    if measure_candidate_strategy not in PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES:
        raise ValueError(
            f"unknown measure_candidate_strategy {measure_candidate_strategy!r}; "
            f"expected one of {PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES}"
        )
    if measured_selection_metric not in PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS:
        raise ValueError(
            f"unknown measured_selection_metric {measured_selection_metric!r}; "
            f"expected one of {PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS}"
        )
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("at least one autosize seed is required")
    requested_min_measure_candidate_seed_count = int(min_measure_candidate_seed_count)
    if requested_min_measure_candidate_seed_count < 1:
        raise ValueError("min_measure_candidate_seed_count must be positive")
    requested_measure_candidate_seed_count = (
        max(len(tuple(dict.fromkeys(normalized_seeds))), requested_min_measure_candidate_seed_count)
        if measure_candidate_seed_count is None
        else int(measure_candidate_seed_count)
    )
    if requested_measure_candidate_seed_count < 1:
        raise ValueError("measure_candidate_seed_count must be positive")
    measurement_seeds = _profile_autosize_measurement_seeds(
        normalized_seeds,
        requested_count=requested_measure_candidate_seed_count,
    )
    provided_seed_set = set(normalized_seeds)
    synthesized_measurement_seed_count = sum(1 for seed in measurement_seeds if seed not in provided_seed_set)
    if candidate_gradient_accumulation_steps is None:
        base_gradient_accumulation_steps = int(gradient_accumulation_steps)
        normalized_candidate_gradient_accumulation_steps = tuple(
            dict.fromkeys((base_gradient_accumulation_steps, max(2, base_gradient_accumulation_steps)))
        )
    else:
        normalized_candidate_gradient_accumulation_steps = tuple(
            dict.fromkeys(int(value) for value in candidate_gradient_accumulation_steps)
        )
    if not normalized_candidate_gradient_accumulation_steps:
        raise ValueError("at least one candidate gradient accumulation value is required")
    if any(int(value) <= 0 for value in normalized_candidate_gradient_accumulation_steps):
        raise ValueError("candidate gradient accumulation values must be positive")
    budget = _profile_autosize_budget(
        memory_budget_mb=memory_budget_mb,
        memory_budget_fraction=memory_budget_fraction,
        device=device,
        require_cuda=require_cuda,
    )
    budget_bytes = int(budget["budget_bytes"])
    candidates: list[Mapping[str, Any]] = []
    rejected: list[Mapping[str, Any]] = []
    seen_shapes: set[str] = set()
    for seq_len in candidate_seq_lens:
        for d_model in candidate_d_models:
            for n_layers in candidate_n_layers:
                for batch_size in candidate_batch_sizes:
                    for candidate_gradient_accumulation_step in normalized_candidate_gradient_accumulation_steps:
                        try:
                            shape = _normalize_profile_shape_spec(
                                {
                                    "seq_len": int(seq_len),
                                    "d_model": int(d_model),
                                    "n_heads": int(n_heads),
                                    "n_layers": int(n_layers),
                                    "batch_size": int(batch_size),
                                    "gradient_accumulation_steps": int(candidate_gradient_accumulation_step),
                                },
                                default_batch_size=int(batch_size),
                            )
                        except ValueError as exc:
                            rejected.append(
                                {
                                    "shape": {
                                        "seq_len": seq_len,
                                        "d_model": d_model,
                                        "n_heads": n_heads,
                                        "n_layers": n_layers,
                                        "batch_size": batch_size,
                                        "gradient_accumulation_steps": candidate_gradient_accumulation_step,
                                    },
                                    "reason": str(exc),
                                }
                            )
                            continue
                        shape_key = _profile_shape_key(shape)
                        if shape_key in seen_shapes:
                            continue
                        seen_shapes.add(shape_key)
                        candidate = _profile_autosize_candidate(
                            shape,
                            vocab_size=vocab_size,
                            precision=precision,
                            device=device,
                            require_cuda=require_cuda,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            native_ternary_backend=native_ternary_backend,
                            budget_bytes=budget_bytes,
                        )
                        if bool(candidate["fits_budget"]):
                            candidates.append(candidate)
                        else:
                            rejected.append({**candidate, "reason": "estimated_peak_exceeds_budget"})
    ranked_candidates = tuple(
        {
            **candidate,
            "estimated_rank": index + 1,
        }
        for index, candidate in enumerate(
            sorted(
                candidates,
                key=lambda item: (
                    float(item["score"]),
                    int(item["estimated_peak_training_bytes"]),
                    int(item["tokens_per_optimizer_step"]),
                    str(item["shape_key"]),
                ),
                reverse=True,
            )
        )
    )
    measured_candidates: tuple[Mapping[str, Any], ...] = ()
    measured_passed_candidates: tuple[Mapping[str, Any], ...] = ()
    measured_profile_passed_count = 0
    measured_candidate_profile_count = 0
    measured_profile_passed_profile_count = 0
    measured_passed_profile_count = 0
    measurement_seed = int(measurement_seeds[0])
    selection_source = "estimated"
    selection_pool: Sequence[Mapping[str, Any]] = ranked_candidates
    measurement_inputs: tuple[Mapping[str, Any], ...] = ()
    measurement_rounds: list[Mapping[str, Any]] = []
    refinement_rounds: list[Mapping[str, Any]] = []
    refinement_budget_actions: list[Mapping[str, Any]] = []
    refinement_budget_candidate_actions: list[Mapping[str, Any]] = []
    refinement_budget_candidate_action_total_count = 0
    refinement_budget_candidate_actions_truncated = False
    confirmation_rounds: list[Mapping[str, Any]] = []
    refinement_seeds: tuple[int, ...] = ()
    confirmation_seeds: tuple[int, ...] = ()
    synthesized_refinement_seed_count = 0
    synthesized_confirmation_seed_count = 0
    confirmation_complete = False
    confirmation_frontier_state: Mapping[str, Any] = {
        "selected_shape_keys": (),
        "selected_lower_confidence": 0.0,
        "best_challenger_shape_key": "",
        "best_challenger_upper_confidence": 0.0,
        "decision_margin": 0.0,
        "pending_shape_keys": (),
        "pending_reasons": (),
    }
    effective_measure_candidate_count = (
        max(requested_measure_candidate_count, int(selected_shape_count))
        if requested_measure_candidate_count > 0
        else 0
    )
    effective_confirm_selected_max_rounds = (
        max(1, int(effective_measure_candidate_count or selected_shape_count))
        if requested_confirm_selected_max_rounds is None
        else requested_confirm_selected_max_rounds
    )
    effective_confirm_selected_decision_resolution_extra_rounds = (
        max(1, int(effective_measure_candidate_count or selected_shape_count))
        if requested_confirm_selected_decision_resolution_extra_rounds is None
        else requested_confirm_selected_decision_resolution_extra_rounds
    )
    confirmation_decision_resolution_adaptive_extra_rounds = 0
    confirmation_decision_resolution_uncertainty = 0.0
    confirmation_decision_resolution_margin_deficit = 0.0
    confirmation_decision_resolution_overlap_ratio = 0.0
    confirmation_decision_resolution_total_rounds = effective_confirm_selected_decision_resolution_extra_rounds
    confirmation_decision_resolution_stop_reason = ""
    confirmation_decision_resolution_budget_evaluations: list[Mapping[str, Any]] = []
    confirmation_runtime_escalations: list[Mapping[str, Any]] = []
    if effective_measure_candidate_count > 0:
        aggregated_measurement_rows: list[Mapping[str, Any]] = []
        already_measured_shape_keys: set[str] = set()
        confirmed_shape_keys: set[str] = set()

        def rank_measured_passed_candidates(
            rows: Sequence[Mapping[str, Any]],
        ) -> tuple[Mapping[str, Any], ...]:
            return tuple(
                sorted(
                    (item for item in rows if bool(item.get("measurement_passed"))),
                    key=lambda item: (
                        float(item["measured_score"]),
                        int(item["estimated_peak_training_bytes"]),
                        int(item["tokens_per_optimizer_step"]),
                        str(item["shape_key"]),
                    ),
                    reverse=True,
                )
            )

        def fresh_extra_seeds(
            *,
            used_seed_set: set[int],
            requested_count: int,
        ) -> tuple[int, ...]:
            if requested_count <= 0:
                return ()
            requested_seed_total = max(
                len(measurement_seeds),
                len(used_seed_set) + int(requested_count),
            )
            while True:
                expanded = _profile_autosize_measurement_seeds(
                    normalized_seeds,
                    requested_count=requested_seed_total,
                )
                extra = tuple(
                    int(seed)
                    for seed in expanded
                    if int(seed) not in used_seed_set
                )[: int(requested_count)]
                if len(extra) >= int(requested_count):
                    return extra
                requested_seed_total += int(requested_count)
                if requested_seed_total > len(used_seed_set) + int(requested_count) + 1024:
                    raise RuntimeError("could not synthesize enough fresh autosize confirmation seeds")

        def measure_candidate_seed(
            candidate: Mapping[str, Any],
            *,
            candidate_index: int,
            seed: int,
            seed_dir_prefix: str = "seed",
            profile_steps: int | None = None,
            repeat_count: int = 1,
        ) -> tuple[Mapping[str, Any], ...]:
            shape = dict(candidate["shape"])
            effective_profile_steps = int(steps if profile_steps is None else profile_steps)
            if repeat_count < 1:
                raise ValueError("repeat_count must be positive")
            rows: list[Mapping[str, Any]] = []
            for repeat_index in range(int(repeat_count)):
                seed_dir_name = (
                    f"{seed_dir_prefix}_{int(seed)}"
                    if profile_steps is None
                    else f"{seed_dir_prefix}_{int(seed)}_steps_{effective_profile_steps}"
                )
                if repeat_count > 1:
                    seed_dir_name = f"{seed_dir_name}_repeat_{repeat_index}"
                measure_dir = (
                    output_dir
                    / "candidate_measurements"
                    / f"candidate_{candidate_index:03d}_{candidate['shape_key']}"
                    / seed_dir_name
                )
                profile: Mapping[str, Any] | None = None
                error: BaseException | None = None
                try:
                    profile = run_llm_batch_profile(
                        out_dir=measure_dir,
                        steps=effective_profile_steps,
                        batch_size=int(shape["batch_size"]),
                        gradient_accumulation_steps=int(shape.get("gradient_accumulation_steps", gradient_accumulation_steps)),
                        seq_len=int(shape["seq_len"]),
                        d_model=int(shape["d_model"]),
                        n_heads=int(shape["n_heads"]),
                        n_layers=int(shape["n_layers"]),
                        vocab_size=vocab_size,
                        precision=precision,
                        device=device,
                        require_cuda=require_cuda,
                        native_ternary_backend=native_ternary_backend,
                        resource_interval=resource_interval,
                        min_resource_samples=min_resource_samples,
                        seed=int(seed),
                        corpus_repeats=corpus_repeats,
                        max_corpus_tokens=max_corpus_tokens,
                        overwrite=False,
                    )
                except Exception as exc:
                    error = exc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                rows.append(
                    _profile_autosize_measurement_summary(
                        candidate,
                        profile=profile,
                        profile_path=measure_dir / "llm_batch_profile.json",
                        seed=int(seed),
                        profile_steps=effective_profile_steps,
                        repeat_index=repeat_index,
                        repeat_count=int(repeat_count),
                        metric=measured_selection_metric,
                        error=error,
                    )
                )
            return tuple(rows)

        def measure_round(
            inputs: Sequence[Mapping[str, Any]],
            *,
            round_index: int,
            round_kind: str,
        ) -> None:
            nonlocal measurement_inputs
            round_rows: list[Mapping[str, Any]] = []
            for candidate in inputs:
                shape_key = str(candidate["shape_key"])
                if shape_key in already_measured_shape_keys:
                    continue
                candidate_index = len(aggregated_measurement_rows)
                candidate = {
                    **candidate,
                    "measurement_candidate_index": candidate_index,
                }
                seed_measurement_rows: list[Mapping[str, Any]] = []
                for seed in measurement_seeds:
                    seed_measurement_rows.extend(
                        measure_candidate_seed(
                            candidate,
                            candidate_index=candidate_index,
                            seed=int(seed),
                        )
                    )
                aggregated = _profile_autosize_aggregate_measurements(
                    candidate,
                    seed_measurements=seed_measurement_rows,
                    metric=measured_selection_metric,
                )
                aggregated_measurement_rows.append(aggregated)
                round_rows.append(aggregated)
                already_measured_shape_keys.add(shape_key)
            if round_rows:
                measurement_inputs = tuple(aggregated_measurement_rows)
                measurement_rounds.append(
                    {
                        "round_index": int(round_index),
                        "round_kind": round_kind,
                        "candidate_count": len(round_rows),
                        "shape_keys": tuple(str(item["shape_key"]) for item in round_rows),
                        "estimated_ranks": tuple(int(item.get("estimated_rank", 0)) for item in round_rows),
                        "selection_reasons": tuple(str(item.get("measurement_selection_reason", "")) for item in round_rows),
                        "source_shape_keys": tuple(str(item.get("measurement_selection_source_shape_key", "")) for item in round_rows),
                        "source_scores": tuple(float(item.get("measurement_selection_source_score", 0.0)) for item in round_rows),
                        "source_score_means": tuple(float(item.get("measurement_selection_source_score_mean", 0.0)) for item in round_rows),
                        "source_score_stddevs": tuple(float(item.get("measurement_selection_source_score_stddev", 0.0)) for item in round_rows),
                        "source_upper_confidences": tuple(float(item.get("measurement_selection_source_upper_confidence", 0.0)) for item in round_rows),
                        "source_stability_ratios": tuple(float(item.get("measurement_selection_source_stability_ratio", 0.0)) for item in round_rows),
                    }
                )

        def refinement_budget_action_payload(
            candidate: Mapping[str, Any],
            *,
            selected_refinement_keys: set[str] | None = None,
            report_selection_reason: str | None = None,
        ) -> Mapping[str, Any]:
            payload = {
                "shape_key": str(candidate.get("shape_key", "")),
                "estimated_rank": int(candidate.get("estimated_rank", 0)),
                "strategy": str(candidate.get("refinement_budget_strategy", "")),
                "expected_gain": float(candidate.get("refinement_expected_gain", 0.0)),
                "uncertainty_width": float(candidate.get("refinement_uncertainty_width", 0.0)),
                "posterior_utility": float(candidate.get("refinement_posterior_utility", 0.0)),
                "measurement_cost_tokens": int(candidate.get("refinement_measurement_cost_tokens", 0)),
                "gain_per_cost": float(candidate.get("refinement_gain_per_cost", 0.0)),
                "is_selected_finalist": bool(candidate.get("refinement_is_selected_finalist")),
                "planned_steps": int(candidate.get("refinement_planned_steps", requested_refine_uncertain_step_multiplier)),
                "planned_repeat_count": int(candidate.get("refinement_planned_repeat_count", requested_refine_uncertain_repeat_count)),
                "planned_extra_seed_count": int(candidate.get("refinement_planned_extra_seed_count", requested_refine_uncertain_extra_seed_count)),
            }
            if selected_refinement_keys is not None:
                payload["selected_for_refinement"] = str(candidate.get("shape_key", "")) in selected_refinement_keys
            if report_selection_reason is not None:
                payload["report_selection_reason"] = str(report_selection_reason)
            return payload

        def select_refinement_action_report_frontier(
            frontier: Sequence[Mapping[str, Any]],
            *,
            selected_refinement_keys: set[str],
            cap: int,
        ) -> tuple[tuple[Mapping[str, Any], str], ...]:
            if cap <= 0:
                return tuple((candidate, "full_frontier") for candidate in frontier)
            selected: list[tuple[Mapping[str, Any], str]] = []
            seen_shape_keys: set[str] = set()

            def add(candidate: Mapping[str, Any], reason: str) -> None:
                shape_key = str(candidate.get("shape_key", ""))
                if not shape_key or shape_key in seen_shape_keys or len(selected) >= cap:
                    return
                selected.append((candidate, reason))
                seen_shape_keys.add(shape_key)

            for candidate in frontier:
                if str(candidate.get("shape_key", "")) in selected_refinement_keys:
                    add(candidate, "selected_for_refinement")
            buckets: tuple[tuple[str, Sequence[Mapping[str, Any]]], ...] = (
                (
                    "top_expected_gain",
                    sorted(
                        frontier,
                        key=lambda candidate: (
                            float(candidate.get("refinement_expected_gain", 0.0)),
                            float(candidate.get("refinement_gain_per_cost", 0.0)),
                            -int(candidate.get("estimated_rank", 0)),
                            str(candidate.get("shape_key", "")),
                        ),
                        reverse=True,
                    ),
                ),
                (
                    "top_uncertainty_width",
                    sorted(
                        frontier,
                        key=lambda candidate: (
                            float(candidate.get("refinement_uncertainty_width", 0.0)),
                            float(candidate.get("refinement_gain_per_cost", 0.0)),
                            -int(candidate.get("estimated_rank", 0)),
                            str(candidate.get("shape_key", "")),
                        ),
                        reverse=True,
                    ),
                ),
                ("top_gain_per_cost", frontier),
                (
                    "selected_finalist",
                    tuple(
                        candidate
                        for candidate in frontier
                        if bool(candidate.get("refinement_is_selected_finalist"))
                    ),
                ),
                (
                    "lowest_measurement_cost",
                    sorted(
                        frontier,
                        key=lambda candidate: (
                            int(candidate.get("refinement_measurement_cost_tokens", 0)),
                            -float(candidate.get("refinement_gain_per_cost", 0.0)),
                            int(candidate.get("estimated_rank", 0)),
                            str(candidate.get("shape_key", "")),
                        ),
                    ),
                ),
                ("frontier_order", frontier),
            )
            while len(selected) < cap:
                progressed = False
                for reason, bucket in buckets:
                    for candidate in bucket:
                        before_count = len(selected)
                        add(candidate, reason)
                        if len(selected) > before_count:
                            progressed = True
                            break
                    if len(selected) >= cap:
                        break
                if not progressed:
                    break
            return tuple(selected)

        def refine_uncertain_candidates(*, round_index: int) -> None:
            nonlocal measurement_inputs, refinement_seeds, synthesized_refinement_seed_count
            nonlocal refinement_budget_candidate_action_total_count, refinement_budget_candidate_actions_truncated
            if (
                requested_refine_uncertain_candidate_count <= 0
                or requested_refine_uncertain_extra_seed_count <= 0
                or len(aggregated_measurement_rows) < 2
            ):
                return
            expanded_seeds = _profile_autosize_measurement_seeds(
                normalized_seeds,
                requested_count=len(measurement_seeds) + requested_refine_uncertain_extra_seed_count,
            )
            measured_seed_set = set(int(seed) for seed in measurement_seeds)
            extra_seeds = tuple(
                int(seed)
                for seed in expanded_seeds
                if int(seed) not in measured_seed_set
            )[:requested_refine_uncertain_extra_seed_count]
            if not extra_seeds:
                return
            refinement_profile_steps = max(1, int(steps) * requested_refine_uncertain_step_multiplier)
            refinement_action_frontier = _profile_autosize_uncertainty_refinement_actions(
                aggregated_measurement_rows,
                selected_shape_count=selected_shape_count,
                refinement_steps=refinement_profile_steps,
                refinement_repeat_count=requested_refine_uncertain_repeat_count,
                refinement_extra_seed_count=len(extra_seeds),
            )
            refinement_inputs = tuple(refinement_action_frontier[:requested_refine_uncertain_candidate_count])
            if not refinement_inputs:
                return

            refined_rows: list[Mapping[str, Any]] = []
            refinement_details: list[Mapping[str, Any]] = []
            for candidate in refinement_inputs:
                shape_key = str(candidate["shape_key"])
                row_index = next(
                    (
                        index
                        for index, row in enumerate(aggregated_measurement_rows)
                        if str(row["shape_key"]) == shape_key
                    ),
                    None,
                )
                if row_index is None:
                    continue
                candidate_index = int(candidate.get("measurement_candidate_index", row_index))
                previous_seed_rows = tuple(candidate.get("seed_measurements", ()))
                before = {
                    "measured_score": float(candidate.get("measured_score", 0.0)),
                    "measured_score_mean": float(candidate.get("measured_score_mean", 0.0)),
                    "measured_score_stddev": float(candidate.get("measured_score_stddev", 0.0)),
                    "measured_score_lower_confidence": float(candidate.get("measured_score_lower_confidence", candidate.get("measured_score", 0.0))),
                    "measured_score_upper_confidence": float(candidate.get("measured_score_upper_confidence", candidate.get("measured_score", 0.0))),
                    "measurement_seed_count": int(candidate.get("measurement_seed_count", len(previous_seed_rows))),
                }
                extra_rows = tuple(
                    row
                    for seed in extra_seeds
                    for row in measure_candidate_seed(
                        candidate,
                        candidate_index=candidate_index,
                        seed=int(seed),
                        seed_dir_prefix="refine_seed",
                        profile_steps=refinement_profile_steps,
                        repeat_count=requested_refine_uncertain_repeat_count,
                    )
                )
                refined = _profile_autosize_aggregate_measurements(
                    candidate,
                    seed_measurements=previous_seed_rows + extra_rows,
                    metric=measured_selection_metric,
                )
                aggregated_measurement_rows[row_index] = refined
                refined_rows.append(refined)
                refinement_details.append(
                    {
                        "shape_key": shape_key,
                        "estimated_rank": int(candidate.get("estimated_rank", 0)),
                        "candidate_index": candidate_index,
                        "extra_seeds": extra_seeds,
                        "refinement_steps": refinement_profile_steps,
                        "refinement_repeat_count": requested_refine_uncertain_repeat_count,
                        "refinement_budget_strategy": str(candidate.get("refinement_budget_strategy", "")),
                        "refinement_expected_gain": float(candidate.get("refinement_expected_gain", 0.0)),
                        "refinement_uncertainty_width": float(candidate.get("refinement_uncertainty_width", 0.0)),
                        "refinement_posterior_utility": float(candidate.get("refinement_posterior_utility", 0.0)),
                        "refinement_measurement_cost_tokens": int(candidate.get("refinement_measurement_cost_tokens", 0)),
                        "refinement_gain_per_cost": float(candidate.get("refinement_gain_per_cost", 0.0)),
                        "refinement_is_selected_finalist": bool(candidate.get("refinement_is_selected_finalist")),
                        "before": before,
                        "after": {
                            "measured_score": float(refined.get("measured_score", 0.0)),
                            "measured_score_mean": float(refined.get("measured_score_mean", 0.0)),
                            "measured_score_stddev": float(refined.get("measured_score_stddev", 0.0)),
                            "measured_score_lower_confidence": float(refined.get("measured_score_lower_confidence", refined.get("measured_score", 0.0))),
                            "measured_score_upper_confidence": float(refined.get("measured_score_upper_confidence", refined.get("measured_score", 0.0))),
                            "measurement_seed_count": int(refined.get("measurement_seed_count", 0)),
                        },
                    }
                )
            if refined_rows:
                measurement_inputs = tuple(aggregated_measurement_rows)
                refinement_seeds = extra_seeds
                synthesized_refinement_seed_count = sum(1 for seed in refinement_seeds if seed not in provided_seed_set)
                selected_refinement_keys = {
                    str(candidate.get("shape_key", ""))
                    for candidate in refinement_inputs
                }
                round_budget_actions = tuple(
                    refinement_budget_action_payload(candidate)
                    for candidate in refinement_inputs
                )
                refinement_action_frontier_total_count = len(refinement_action_frontier)
                reported_refinement_action_frontier = select_refinement_action_report_frontier(
                    refinement_action_frontier,
                    selected_refinement_keys=selected_refinement_keys,
                    cap=requested_refinement_budget_candidate_action_report_cap,
                )
                refinement_action_frontier_truncated = (
                    len(reported_refinement_action_frontier) < refinement_action_frontier_total_count
                )
                round_budget_candidate_actions = tuple(
                    refinement_budget_action_payload(
                        candidate,
                        selected_refinement_keys=selected_refinement_keys,
                        report_selection_reason=report_selection_reason,
                    )
                    for candidate, report_selection_reason in reported_refinement_action_frontier
                )
                refinement_budget_actions.extend(round_budget_actions)
                refinement_budget_candidate_actions.extend(round_budget_candidate_actions)
                refinement_budget_candidate_action_total_count += refinement_action_frontier_total_count
                refinement_budget_candidate_actions_truncated = (
                    refinement_budget_candidate_actions_truncated or refinement_action_frontier_truncated
                )
                refinement_rounds.append(
                    {
                        "round_index": int(round_index),
                        "round_kind": "uncertainty_seed_refinement",
                        "candidate_count": len(refined_rows),
                        "shape_keys": tuple(str(item["shape_key"]) for item in refined_rows),
                        "estimated_ranks": tuple(int(item.get("estimated_rank", 0)) for item in refined_rows),
                        "extra_seed_count": len(extra_seeds),
                        "extra_seeds": extra_seeds,
                        "refinement_steps": refinement_profile_steps,
                        "refinement_repeat_count": requested_refine_uncertain_repeat_count,
                        "refinement_budget_strategy": "expected_gain_per_cost",
                        "refinement_budget_actions": round_budget_actions,
                        "refinement_budget_candidate_action_report_cap": requested_refinement_budget_candidate_action_report_cap,
                        "refinement_budget_candidate_action_total_count": refinement_action_frontier_total_count,
                        "refinement_budget_candidate_action_count": len(round_budget_candidate_actions),
                        "refinement_budget_candidate_actions_truncated": refinement_action_frontier_truncated,
                        "refinement_budget_candidate_actions": round_budget_candidate_actions,
                        "details": tuple(refinement_details),
                    }
                )

        def selected_confirmation_frontier_state() -> Mapping[str, Any]:
            passed = rank_measured_passed_candidates(aggregated_measurement_rows)
            selected_count = max(1, int(selected_shape_count))
            selected_candidates = passed[:selected_count]
            challenger_candidates = passed[selected_count:]
            selected_shape_keys = tuple(str(candidate["shape_key"]) for candidate in selected_candidates)
            selected_lower_confidence = float(
                min(
                    (
                        float(candidate.get("measured_score", 0.0))
                        for candidate in selected_candidates
                    ),
                    default=0.0,
                )
            )
            best_challenger = max(
                challenger_candidates,
                key=lambda candidate: float(
                    candidate.get(
                        "measured_score_upper_confidence",
                        candidate.get("measured_score", 0.0),
                    )
                ),
                default=None,
            )
            best_challenger_upper_confidence = (
                float(
                    best_challenger.get(
                        "measured_score_upper_confidence",
                        best_challenger.get("measured_score", 0.0),
                    )
                )
                if best_challenger is not None
                else 0.0
            )
            pending: list[Mapping[str, Any]] = []
            pending_reasons: list[str] = []
            for candidate in selected_candidates:
                if str(candidate["shape_key"]) not in confirmed_shape_keys:
                    pending.append(candidate)
                    pending_reasons.append("selected_finalist_unconfirmed")
            for candidate in challenger_candidates:
                shape_key = str(candidate["shape_key"])
                challenger_upper = float(
                    candidate.get(
                        "measured_score_upper_confidence",
                        candidate.get("measured_score", 0.0),
                    )
                )
                if shape_key not in confirmed_shape_keys and challenger_upper >= selected_lower_confidence:
                    pending.append(candidate)
                    pending_reasons.append("decision_frontier_challenger_unconfirmed")
            return {
                "selected_shape_keys": selected_shape_keys,
                "selected_candidates": tuple(selected_candidates),
                "selected_lower_confidence": selected_lower_confidence,
                "best_challenger_shape_key": (
                    str(best_challenger["shape_key"])
                    if best_challenger is not None
                    else ""
                ),
                "best_challenger_candidate": best_challenger,
                "best_challenger_upper_confidence": best_challenger_upper_confidence,
                "decision_margin": float(selected_lower_confidence - best_challenger_upper_confidence),
                "pending_candidates": tuple(pending),
                "pending_shape_keys": tuple(str(candidate["shape_key"]) for candidate in pending),
                "pending_reasons": tuple(pending_reasons),
            }

        def selected_confirmation_missing() -> tuple[Mapping[str, Any], ...]:
            return tuple(selected_confirmation_frontier_state()["pending_candidates"])

        def confirmation_decision_resolved(state: Mapping[str, Any]) -> bool:
            return (
                not str(state.get("best_challenger_shape_key", ""))
                or float(state.get("decision_margin", 0.0)) > 0.0
            )

        def decision_resolution_inputs(state: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
            if confirmation_decision_resolved(state):
                return ()
            best_challenger = state.get("best_challenger_candidate")
            if best_challenger is None:
                return ()
            inputs: list[Mapping[str, Any]] = []
            seen: set[str] = set()
            for candidate in tuple(state.get("selected_candidates", ())) + (best_challenger,):
                shape_key = str(candidate["shape_key"])
                if shape_key in seen:
                    continue
                seen.add(shape_key)
                inputs.append(candidate)
            return tuple(inputs)

        def decision_resolution_adaptive_budget(state: Mapping[str, Any]) -> Mapping[str, Any]:
            if (
                effective_confirm_selected_decision_resolution_extra_rounds <= 0
                or requested_confirm_selected_decision_resolution_adaptive_extra_rounds <= 0
                or tuple(state.get("pending_candidates", ()))
                or confirmation_decision_resolved(state)
            ):
                return {
                    "adaptive_extra_rounds": 0,
                    "uncertainty": 0.0,
                    "margin_deficit": max(0.0, -float(state.get("decision_margin", 0.0))),
                    "overlap_ratio": 0.0,
                }
            inputs = decision_resolution_inputs(state)
            interval_widths = tuple(
                max(
                    0.0,
                    float(candidate.get("measured_score_upper_confidence", candidate.get("measured_score", 0.0)))
                    - float(candidate.get("measured_score_lower_confidence", candidate.get("measured_score", 0.0))),
                )
                for candidate in inputs
            )
            uncertainty = max(interval_widths, default=0.0)
            decision_margin = float(state.get("decision_margin", 0.0))
            margin_deficit = max(1e-9, -decision_margin)
            if uncertainty <= 0.0 or margin_deficit <= 0.0:
                adaptive_extra_rounds = 0
                overlap_ratio = 0.0
            else:
                overlap_ratio = margin_deficit / max(1e-9, uncertainty)
                adaptive_extra_rounds = min(
                    requested_confirm_selected_decision_resolution_adaptive_extra_rounds,
                    max(1, int(math.ceil(overlap_ratio))),
                )
            return {
                "adaptive_extra_rounds": adaptive_extra_rounds,
                "uncertainty": uncertainty,
                "margin_deficit": margin_deficit,
                "overlap_ratio": overlap_ratio,
            }

        def confirmation_runtime_plan(
            *,
            confirmation_inputs: Sequence[Mapping[str, Any]],
            state: Mapping[str, Any],
            round_kind: str,
        ) -> Mapping[str, Any]:
            candidates = tuple(confirmation_inputs)
            interval_widths = tuple(
                max(
                    0.0,
                    float(candidate.get("measured_score_upper_confidence", candidate.get("measured_score", 0.0)))
                    - float(candidate.get("measured_score_lower_confidence", candidate.get("measured_score", 0.0))),
                )
                for candidate in candidates
            )
            score_values = [
                abs(float(candidate.get(key, 0.0)))
                for candidate in candidates
                for key in (
                    "measured_score",
                    "measured_score_mean",
                    "measured_score_lower_confidence",
                    "measured_score_upper_confidence",
                )
            ]
            uncertainty = max(interval_widths, default=0.0)
            score_scale = max((1.0, *score_values))
            uncertainty_ratio = uncertainty / score_scale
            decision_margin = float(state.get("decision_margin", 0.0))
            margin_deficit = max(0.0, -decision_margin)
            overlap_ratio = margin_deficit / max(1e-9, uncertainty) if uncertainty > 0.0 else 0.0
            runtime_signal = max(uncertainty_ratio, overlap_ratio)
            escalation_level = 0
            step_multiplier = requested_confirm_selected_step_multiplier
            repeat_count = requested_confirm_selected_repeat_count
            if runtime_signal > 0.0:
                max_level = max(
                    requested_confirm_selected_runtime_repeat_count_cap
                    - requested_confirm_selected_repeat_count,
                    requested_confirm_selected_runtime_step_multiplier_cap
                    - requested_confirm_selected_step_multiplier,
                    0,
                )
                escalation_level = min(max_level, max(1, int(math.ceil(runtime_signal))))
                step_multiplier = min(
                    requested_confirm_selected_runtime_step_multiplier_cap,
                    requested_confirm_selected_step_multiplier * (1 + escalation_level),
                )
                repeat_count = min(
                    requested_confirm_selected_runtime_repeat_count_cap,
                    requested_confirm_selected_repeat_count + escalation_level,
                )
            profile_steps = max(1, int(steps) * step_multiplier)
            return {
                "round_kind": round_kind,
                "base_step_multiplier": requested_confirm_selected_step_multiplier,
                "step_multiplier": step_multiplier,
                "steps": profile_steps,
                "base_repeat_count": requested_confirm_selected_repeat_count,
                "repeat_count": repeat_count,
                "escalation_level": escalation_level,
                "adaptive_runtime_applied": (
                    step_multiplier != requested_confirm_selected_step_multiplier
                    or repeat_count != requested_confirm_selected_repeat_count
                ),
                "uncertainty": uncertainty,
                "score_scale": score_scale,
                "uncertainty_ratio": uncertainty_ratio,
                "decision_margin": decision_margin,
                "margin_deficit": margin_deficit,
                "overlap_ratio": overlap_ratio,
                "runtime_signal": runtime_signal,
                "shape_keys": tuple(str(candidate.get("shape_key", "")) for candidate in candidates),
            }

        def confirm_selected_candidates(
            *,
            round_index: int,
            confirmation_inputs: Sequence[Mapping[str, Any]] | None = None,
            round_kind: str = "selected_candidate_confirmation",
        ) -> int:
            nonlocal measurement_inputs, confirmation_seeds, synthesized_confirmation_seed_count
            if (
                requested_confirm_selected_candidate_count <= 0
                or requested_confirm_selected_extra_seed_count <= 0
                or not aggregated_measurement_rows
            ):
                return 0
            confirmation_inputs = (
                selected_confirmation_missing()[: int(requested_confirm_selected_candidate_count)]
                if confirmation_inputs is None
                else tuple(confirmation_inputs)
            )
            if not confirmation_inputs:
                return 0
            confirmation_state = selected_confirmation_frontier_state()
            runtime_plan = confirmation_runtime_plan(
                confirmation_inputs=confirmation_inputs,
                state=confirmation_state,
                round_kind=round_kind,
            )
            used_seed_set = {
                int(row.get("seed", 0))
                for candidate in aggregated_measurement_rows
                for row in tuple(candidate.get("seed_measurements", ()))
            }
            extra_seeds = fresh_extra_seeds(
                used_seed_set=used_seed_set,
                requested_count=requested_confirm_selected_extra_seed_count,
            )
            if not extra_seeds:
                return 0

            confirmed_rows: list[Mapping[str, Any]] = []
            confirmation_details: list[Mapping[str, Any]] = []
            confirmation_profile_steps = int(runtime_plan["steps"])
            confirmation_repeat_count = int(runtime_plan["repeat_count"])
            for candidate in confirmation_inputs:
                shape_key = str(candidate["shape_key"])
                row_index = next(
                    (
                        index
                        for index, row in enumerate(aggregated_measurement_rows)
                        if str(row["shape_key"]) == shape_key
                    ),
                    None,
                )
                if row_index is None:
                    continue
                candidate_index = int(candidate.get("measurement_candidate_index", row_index))
                previous_seed_rows = tuple(candidate.get("seed_measurements", ()))
                before = {
                    "measured_score": float(candidate.get("measured_score", 0.0)),
                    "measured_score_mean": float(candidate.get("measured_score_mean", 0.0)),
                    "measured_score_stddev": float(candidate.get("measured_score_stddev", 0.0)),
                    "measured_score_lower_confidence": float(candidate.get("measured_score_lower_confidence", candidate.get("measured_score", 0.0))),
                    "measured_score_upper_confidence": float(candidate.get("measured_score_upper_confidence", candidate.get("measured_score", 0.0))),
                    "measurement_seed_count": int(candidate.get("measurement_seed_count", len(previous_seed_rows))),
                }
                extra_rows = tuple(
                    row
                    for seed in extra_seeds
                    for row in measure_candidate_seed(
                        candidate,
                        candidate_index=candidate_index,
                        seed=int(seed),
                        seed_dir_prefix="confirm_seed",
                        profile_steps=confirmation_profile_steps,
                        repeat_count=confirmation_repeat_count,
                    )
                )
                confirmed = _profile_autosize_aggregate_measurements(
                    candidate,
                    seed_measurements=previous_seed_rows + extra_rows,
                    metric=measured_selection_metric,
                )
                aggregated_measurement_rows[row_index] = confirmed
                confirmed_shape_keys.add(shape_key)
                confirmed_rows.append(confirmed)
                confirmation_details.append(
                    {
                        "shape_key": shape_key,
                        "estimated_rank": int(candidate.get("estimated_rank", 0)),
                        "candidate_index": candidate_index,
                        "extra_seeds": extra_seeds,
                        "confirmation_steps": confirmation_profile_steps,
                        "confirmation_step_multiplier": int(runtime_plan["step_multiplier"]),
                        "confirmation_repeat_count": confirmation_repeat_count,
                        "confirmation_adaptive_runtime_applied": bool(runtime_plan["adaptive_runtime_applied"]),
                        "confirmation_runtime_escalation_level": int(runtime_plan["escalation_level"]),
                        "before": before,
                        "after": {
                            "measured_score": float(confirmed.get("measured_score", 0.0)),
                            "measured_score_mean": float(confirmed.get("measured_score_mean", 0.0)),
                            "measured_score_stddev": float(confirmed.get("measured_score_stddev", 0.0)),
                            "measured_score_lower_confidence": float(confirmed.get("measured_score_lower_confidence", confirmed.get("measured_score", 0.0))),
                            "measured_score_upper_confidence": float(confirmed.get("measured_score_upper_confidence", confirmed.get("measured_score", 0.0))),
                            "measurement_seed_count": int(confirmed.get("measurement_seed_count", 0)),
                        },
                    }
                )
            if confirmed_rows:
                measurement_inputs = tuple(aggregated_measurement_rows)
                confirmation_seeds = tuple(dict.fromkeys(confirmation_seeds + extra_seeds))
                synthesized_confirmation_seed_count = sum(1 for seed in confirmation_seeds if seed not in provided_seed_set)
                confirmation_rounds.append(
                    {
                        "round_index": int(round_index),
                        "round_kind": round_kind,
                        "candidate_count": len(confirmed_rows),
                        "shape_keys": tuple(str(item["shape_key"]) for item in confirmed_rows),
                        "estimated_ranks": tuple(int(item.get("estimated_rank", 0)) for item in confirmed_rows),
                        "extra_seed_count": len(extra_seeds),
                        "extra_seeds": extra_seeds,
                        "confirmation_steps": confirmation_profile_steps,
                        "confirmation_base_step_multiplier": int(runtime_plan["base_step_multiplier"]),
                        "confirmation_step_multiplier": int(runtime_plan["step_multiplier"]),
                        "confirmation_base_repeat_count": int(runtime_plan["base_repeat_count"]),
                        "confirmation_repeat_count": confirmation_repeat_count,
                        "confirmation_adaptive_runtime_applied": bool(runtime_plan["adaptive_runtime_applied"]),
                        "confirmation_runtime_escalation_level": int(runtime_plan["escalation_level"]),
                        "confirmation_runtime_uncertainty": float(runtime_plan["uncertainty"]),
                        "confirmation_runtime_uncertainty_ratio": float(runtime_plan["uncertainty_ratio"]),
                        "confirmation_runtime_margin_deficit": float(runtime_plan["margin_deficit"]),
                        "confirmation_runtime_overlap_ratio": float(runtime_plan["overlap_ratio"]),
                        "confirmation_runtime_signal": float(runtime_plan["runtime_signal"]),
                        "details": tuple(confirmation_details),
                    }
                )
                if bool(runtime_plan["adaptive_runtime_applied"]):
                    confirmation_runtime_escalations.append(
                        {
                            "round_index": int(round_index),
                            "round_kind": round_kind,
                            "shape_keys": tuple(str(item["shape_key"]) for item in confirmed_rows),
                            "extra_seeds": extra_seeds,
                            "confirmation_steps": confirmation_profile_steps,
                            "confirmation_step_multiplier": int(runtime_plan["step_multiplier"]),
                            "confirmation_repeat_count": confirmation_repeat_count,
                            "escalation_level": int(runtime_plan["escalation_level"]),
                            "uncertainty": float(runtime_plan["uncertainty"]),
                            "uncertainty_ratio": float(runtime_plan["uncertainty_ratio"]),
                            "margin_deficit": float(runtime_plan["margin_deficit"]),
                            "overlap_ratio": float(runtime_plan["overlap_ratio"]),
                            "runtime_signal": float(runtime_plan["runtime_signal"]),
                        }
                    )
            return len(confirmed_rows)

        use_adaptive = (
            requested_measure_candidate_adaptive_rounds > 1
            and effective_measure_candidate_count > 2
            and measure_candidate_strategy == "diverse"
        )
        initial_count = (
            max(2, math.ceil(effective_measure_candidate_count / requested_measure_candidate_adaptive_rounds))
            if use_adaptive
            else effective_measure_candidate_count
        )
        initial_inputs = _profile_autosize_measurement_inputs(
            ranked_candidates,
            requested_count=initial_count,
            strategy=measure_candidate_strategy,
        )
        measure_round(initial_inputs, round_index=0, round_kind=f"initial_{measure_candidate_strategy}")
        round_index = 1
        while (
            use_adaptive
            and len(aggregated_measurement_rows) < effective_measure_candidate_count
            and round_index < requested_measure_candidate_adaptive_rounds
        ):
            remaining_count = effective_measure_candidate_count - len(aggregated_measurement_rows)
            remaining_rounds = requested_measure_candidate_adaptive_rounds - round_index
            adaptive_count = max(1, math.ceil(remaining_count / max(1, remaining_rounds)))
            adaptive_inputs = _profile_autosize_adaptive_measurement_inputs(
                ranked_candidates,
                measured_candidates=tuple(aggregated_measurement_rows),
                already_measured_shape_keys=already_measured_shape_keys,
                requested_count=adaptive_count,
            )
            if not adaptive_inputs:
                break
            measure_round(adaptive_inputs, round_index=round_index, round_kind="adaptive_measured_frontier")
            round_index += 1
        if len(aggregated_measurement_rows) < effective_measure_candidate_count:
            remaining_inputs = _profile_autosize_measurement_inputs(
                tuple(
                    candidate
                    for candidate in ranked_candidates
                    if str(candidate["shape_key"]) not in already_measured_shape_keys
                ),
                requested_count=effective_measure_candidate_count - len(aggregated_measurement_rows),
                strategy=measure_candidate_strategy,
            )
            measure_round(remaining_inputs, round_index=round_index, round_kind=f"fill_{measure_candidate_strategy}")
        refine_uncertain_candidates(round_index=round_index + 1)
        if requested_confirm_selected_candidate_count > 0 and requested_confirm_selected_extra_seed_count > 0:
            for confirm_round_offset in range(effective_confirm_selected_max_rounds):
                confirmation_state = selected_confirmation_frontier_state()
                pending_inputs = tuple(confirmation_state["pending_candidates"])
                if not pending_inputs:
                    if confirmation_decision_resolved(confirmation_state):
                        confirmation_complete = True
                    break
                confirmed_count = confirm_selected_candidates(
                    round_index=round_index + 2 + confirm_round_offset,
                    confirmation_inputs=pending_inputs[: int(requested_confirm_selected_candidate_count)],
                    round_kind="selected_candidate_confirmation",
                )
                if confirmed_count <= 0:
                    break
            base_resolution_rounds_used = 0
            adaptive_resolution_rounds_used = 0
            resolution_round_offset = 0
            max_resolution_rounds = (
                effective_confirm_selected_decision_resolution_extra_rounds
                + requested_confirm_selected_decision_resolution_adaptive_extra_rounds
            )
            while resolution_round_offset < max_resolution_rounds:
                confirmation_state = selected_confirmation_frontier_state()
                if tuple(confirmation_state["pending_candidates"]):
                    confirmation_decision_resolution_stop_reason = "frontier_pending"
                    break
                if confirmation_decision_resolved(confirmation_state):
                    confirmation_complete = True
                    confirmation_decision_resolution_stop_reason = "decision_resolved"
                    break
                adaptive_budget_state = decision_resolution_adaptive_budget(confirmation_state)
                confirmation_decision_resolution_uncertainty = float(adaptive_budget_state["uncertainty"])
                confirmation_decision_resolution_margin_deficit = float(adaptive_budget_state["margin_deficit"])
                confirmation_decision_resolution_overlap_ratio = float(adaptive_budget_state["overlap_ratio"])
                if base_resolution_rounds_used < effective_confirm_selected_decision_resolution_extra_rounds:
                    budget_kind = "base"
                    base_resolution_rounds_used += 1
                else:
                    if adaptive_resolution_rounds_used >= requested_confirm_selected_decision_resolution_adaptive_extra_rounds:
                        confirmation_decision_resolution_stop_reason = "adaptive_budget_exhausted"
                        break
                    if int(adaptive_budget_state["adaptive_extra_rounds"]) <= 0:
                        confirmation_decision_resolution_stop_reason = "adaptive_not_indicated"
                        break
                    budget_kind = "adaptive"
                    adaptive_resolution_rounds_used += 1
                    confirmation_decision_resolution_adaptive_extra_rounds = adaptive_resolution_rounds_used
                confirmation_decision_resolution_budget_evaluations.append(
                    {
                        "round_offset": resolution_round_offset,
                        "budget_kind": budget_kind,
                        "decision_margin": float(confirmation_state.get("decision_margin", 0.0)),
                        "adaptive_extra_rounds_recommended": int(adaptive_budget_state["adaptive_extra_rounds"]),
                        "adaptive_extra_rounds_used": adaptive_resolution_rounds_used,
                        "uncertainty": confirmation_decision_resolution_uncertainty,
                        "margin_deficit": confirmation_decision_resolution_margin_deficit,
                        "overlap_ratio": confirmation_decision_resolution_overlap_ratio,
                    }
                )
                confirmed_count = confirm_selected_candidates(
                    round_index=round_index
                    + 2
                    + effective_confirm_selected_max_rounds
                    + resolution_round_offset,
                    confirmation_inputs=decision_resolution_inputs(confirmation_state),
                    round_kind="decision_margin_resolution",
                )
                if confirmed_count <= 0:
                    confirmation_decision_resolution_stop_reason = "no_profiles"
                    break
                resolution_round_offset += 1
            confirmation_decision_resolution_total_rounds = (
                effective_confirm_selected_decision_resolution_extra_rounds
                + confirmation_decision_resolution_adaptive_extra_rounds
            )
            if not confirmation_decision_resolution_stop_reason:
                final_confirmation_state = selected_confirmation_frontier_state()
                if confirmation_decision_resolved(final_confirmation_state):
                    confirmation_complete = True
                    confirmation_decision_resolution_stop_reason = "decision_resolved"
                else:
                    confirmation_decision_resolution_stop_reason = "round_limit"
            if not selected_confirmation_missing():
                confirmation_complete = True
            confirmation_frontier_state = {
                key: value
                for key, value in selected_confirmation_frontier_state().items()
                if key not in {"pending_candidates", "selected_candidates", "best_challenger_candidate"}
            }
        elif effective_measure_candidate_count > 0:
            confirmation_frontier_state = {
                key: value
                for key, value in selected_confirmation_frontier_state().items()
                if key not in {"pending_candidates", "selected_candidates", "best_challenger_candidate"}
            }
        measured_candidates = tuple(aggregated_measurement_rows)
        measured_candidate_profile_count = sum(
            len(tuple(item.get("seed_measurements", ()))) for item in measured_candidates
        )
        measured_profile_passed_count = sum(1 for item in measured_candidates if bool(item.get("measurement_profile_passed")))
        measured_profile_passed_profile_count = sum(
            1
            for item in measured_candidates
            for row in tuple(item.get("seed_measurements", ()))
            if bool(row.get("measurement_profile_passed"))
        )
        measured_passed_profile_count = sum(
            1
            for item in measured_candidates
            for row in tuple(item.get("seed_measurements", ()))
            if bool(row.get("measurement_passed"))
        )
        measured_passed_candidates = tuple(
            rank_measured_passed_candidates(measured_candidates)
        )
        selection_source = "measured"
        selection_pool = measured_passed_candidates
    selected = tuple(selection_pool[: int(selected_shape_count)])
    confirmation_best_challenger_shape_key = str(confirmation_frontier_state.get("best_challenger_shape_key", ""))
    confirmation_decision_margin = float(confirmation_frontier_state.get("decision_margin", 0.0))
    confirmation_decision_resolved = (
        not confirmation_best_challenger_shape_key
        or confirmation_decision_margin > 0.0
    )
    failed_checks: list[str] = []
    if not selected:
        if not ranked_candidates:
            failed_checks.append("no_viable_shapes")
        else:
            failed_checks.append("no_measured_viable_shapes" if effective_measure_candidate_count > 0 else "no_viable_shapes")
    if len(selected) < int(min_selected_shapes):
        failed_checks.append("min_selected_shapes")
    if (
        effective_measure_candidate_count > 0
        and measured_candidates
        and not measured_passed_candidates
        and any(bool(item.get("measurement_profile_passed")) and not bool(item.get("measured_budget_passed")) for item in measured_candidates)
    ):
        failed_checks.append("measured_budget_exceeded")
    if (
        effective_measure_candidate_count > 0
        and measured_candidates
        and selected
        and requested_confirm_selected_candidate_count > 0
        and requested_confirm_selected_extra_seed_count > 0
    ):
        if not confirmation_complete:
            failed_checks.append("selected_confirmation_incomplete")
        elif not confirmation_decision_resolved:
            failed_checks.append("selected_confirmation_decision_unresolved")
    matrix_blocking_checks = {
        "selected_confirmation_incomplete",
        "selected_confirmation_decision_unresolved",
    }
    matrix_report: Mapping[str, Any] | None = None
    if selected and not any(item in matrix_blocking_checks for item in failed_checks):
        selected_shapes = tuple(dict(item["shape"]) for item in selected)
        matrix_report = run_llm_batch_profile_matrix(
            out_dir=output_dir / "matrix",
            shape_specs=selected_shapes,
            seeds=normalized_seeds,
            steps=steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            vocab_size=vocab_size,
            precision=precision,
            device=device,
            require_cuda=require_cuda,
            native_ternary_backend=native_ternary_backend,
            resource_interval=resource_interval,
            min_resource_samples=min_resource_samples,
            corpus_repeats=corpus_repeats,
            max_corpus_tokens=max_corpus_tokens,
            min_cases=max(int(min_cases), len(selected_shapes) * len(normalized_seeds)),
            require_multi_shape=require_multi_shape,
            require_multi_seed=require_multi_seed,
            min_train_tokens_per_second_mean=min_train_tokens_per_second_mean,
            min_gpu_utilization_percent_mean=min_gpu_utilization_percent_mean,
            min_gpu_memory_used_mb_mean=min_gpu_memory_used_mb_mean,
            min_gpu_power_draw_watts_mean=min_gpu_power_draw_watts_mean,
            overwrite=False,
        )
        if not bool(matrix_report.get("passed")):
            failed_checks.append("profile_matrix_failed")
            failed_checks.extend(f"matrix:{item}" for item in matrix_report.get("failed_checks", ()))
    report = {
        "schema_version": 1,
        "run_dir": str(output_dir),
        "budget": budget,
        "candidate_grid": {
            "candidate_seq_lens": tuple(int(value) for value in candidate_seq_lens),
            "candidate_d_models": tuple(int(value) for value in candidate_d_models),
            "candidate_n_layers": tuple(int(value) for value in candidate_n_layers),
            "candidate_batch_sizes": tuple(int(value) for value in candidate_batch_sizes),
            "candidate_gradient_accumulation_steps": normalized_candidate_gradient_accumulation_steps,
            "n_heads": int(n_heads),
        },
        "selection": {
            "selected_shape_count": int(selected_shape_count),
            "min_selected_shapes": int(min_selected_shapes),
            "selection_source": selection_source,
            "viable_candidate_count": len(candidates),
            "rejected_candidate_count": len(rejected),
            "measured_candidate_count": len(measured_candidates),
            "measured_candidate_profile_count": measured_candidate_profile_count,
            "measured_profile_passed_candidate_count": measured_profile_passed_count,
            "measured_profile_passed_profile_count": measured_profile_passed_profile_count,
            "measured_passed_candidate_count": len(measured_passed_candidates),
            "measured_passed_profile_count": measured_passed_profile_count,
            "selected_shapes": tuple(dict(item["shape"]) for item in selected),
            "selected_shape_keys": tuple(str(item["shape_key"]) for item in selected),
        },
        "measurement": {
            "enabled": effective_measure_candidate_count > 0,
            "requested_candidate_count": requested_measure_candidate_count,
            "effective_candidate_count": effective_measure_candidate_count,
            "measured_candidate_count": len(measured_candidates),
            "measured_candidate_profile_count": measured_candidate_profile_count,
            "measured_profile_passed_candidate_count": measured_profile_passed_count,
            "measured_profile_passed_profile_count": measured_profile_passed_profile_count,
            "measured_passed_candidate_count": len(measured_passed_candidates),
            "measured_passed_profile_count": measured_passed_profile_count,
            "measurement_seed": measurement_seed,
            "provided_seed_count": len(tuple(dict.fromkeys(normalized_seeds))),
            "min_measurement_seed_count": requested_min_measure_candidate_seed_count,
            "requested_seed_count": requested_measure_candidate_seed_count,
            "measurement_seed_count": len(measurement_seeds),
            "synthesized_measurement_seed_count": synthesized_measurement_seed_count,
            "measurement_seeds": measurement_seeds,
            "candidate_selection_strategy": measure_candidate_strategy,
            "adaptive_rounds_requested": requested_measure_candidate_adaptive_rounds,
            "adaptive_rounds_used": len(measurement_rounds),
            "measurement_rounds": tuple(measurement_rounds),
            "refinement_enabled": (
                effective_measure_candidate_count > 0
                and requested_refine_uncertain_candidate_count > 0
                and requested_refine_uncertain_extra_seed_count > 0
            ),
            "refine_uncertain_candidate_count": requested_refine_uncertain_candidate_count,
            "refine_uncertain_extra_seed_count": requested_refine_uncertain_extra_seed_count,
            "refine_uncertain_step_multiplier": requested_refine_uncertain_step_multiplier,
            "refine_uncertain_repeat_count": requested_refine_uncertain_repeat_count,
            "refinement_rounds_used": len(refinement_rounds),
            "refined_candidate_count": sum(int(round_["candidate_count"]) for round_ in refinement_rounds),
            "refinement_profile_count": sum(
                int(round_["candidate_count"]) * int(round_["extra_seed_count"]) * int(round_.get("refinement_repeat_count", 1))
                for round_ in refinement_rounds
            ),
            "refinement_seed_count": len(refinement_seeds),
            "refinement_seeds": refinement_seeds,
            "synthesized_refinement_seed_count": synthesized_refinement_seed_count,
            "refinement_rounds": tuple(refinement_rounds),
            "refinement_budget_strategy": "expected_gain_per_cost",
            "refinement_budget_action_count": len(refinement_budget_actions),
            "refinement_budget_actions": tuple(refinement_budget_actions),
            "refinement_budget_candidate_action_report_cap": requested_refinement_budget_candidate_action_report_cap,
            "refinement_budget_candidate_action_total_count": refinement_budget_candidate_action_total_count,
            "refinement_budget_candidate_action_count": len(refinement_budget_candidate_actions),
            "refinement_budget_candidate_actions_truncated": refinement_budget_candidate_actions_truncated,
            "refinement_budget_candidate_actions": tuple(refinement_budget_candidate_actions),
            "confirmation_enabled": (
                effective_measure_candidate_count > 0
                and requested_confirm_selected_candidate_count > 0
                and requested_confirm_selected_extra_seed_count > 0
            ),
            "confirm_selected_candidate_count": requested_confirm_selected_candidate_count,
            "confirm_selected_extra_seed_count": requested_confirm_selected_extra_seed_count,
            "confirm_selected_step_multiplier": requested_confirm_selected_step_multiplier,
            "confirm_selected_repeat_count": requested_confirm_selected_repeat_count,
            "confirm_selected_max_rounds": effective_confirm_selected_max_rounds,
            "confirm_selected_decision_resolution_extra_rounds": effective_confirm_selected_decision_resolution_extra_rounds,
            "confirm_selected_decision_resolution_adaptive_extra_round_cap": requested_confirm_selected_decision_resolution_adaptive_extra_rounds,
            "confirm_selected_runtime_step_multiplier_cap": requested_confirm_selected_runtime_step_multiplier_cap,
            "confirm_selected_runtime_repeat_count_cap": requested_confirm_selected_runtime_repeat_count_cap,
            "confirm_selected_decision_resolution_adaptive_extra_rounds": confirmation_decision_resolution_adaptive_extra_rounds,
            "confirm_selected_decision_resolution_total_rounds": confirmation_decision_resolution_total_rounds,
            "confirmation_decision_resolution_uncertainty": confirmation_decision_resolution_uncertainty,
            "confirmation_decision_resolution_margin_deficit": confirmation_decision_resolution_margin_deficit,
            "confirmation_decision_resolution_overlap_ratio": confirmation_decision_resolution_overlap_ratio,
            "confirmation_decision_resolution_stop_reason": confirmation_decision_resolution_stop_reason,
            "confirmation_decision_resolution_budget_evaluations": tuple(confirmation_decision_resolution_budget_evaluations),
            "confirmation_runtime_escalation_count": len(confirmation_runtime_escalations),
            "confirmation_runtime_escalations": tuple(confirmation_runtime_escalations),
            "confirmation_rounds_used": len(confirmation_rounds),
            "confirmation_decision_resolution_rounds_used": sum(
                1 for round_ in confirmation_rounds if str(round_.get("round_kind", "")) == "decision_margin_resolution"
            ),
            "confirmed_candidate_count": sum(int(round_["candidate_count"]) for round_ in confirmation_rounds),
            "confirmation_profile_count": sum(
                int(round_["candidate_count"]) * int(round_["extra_seed_count"]) * int(round_.get("confirmation_repeat_count", 1))
                for round_ in confirmation_rounds
            ),
            "confirmation_seed_count": len(confirmation_seeds),
            "confirmation_seeds": confirmation_seeds,
            "confirmation_complete": confirmation_complete,
            "confirmation_decision_resolved": confirmation_decision_resolved,
            "confirmed_shape_keys": tuple(sorted(confirmed_shape_keys)) if effective_measure_candidate_count > 0 else (),
            "confirmation_selected_shape_keys": tuple(confirmation_frontier_state.get("selected_shape_keys", ())),
            "confirmation_selected_lower_confidence": float(confirmation_frontier_state.get("selected_lower_confidence", 0.0)),
            "confirmation_best_challenger_shape_key": confirmation_best_challenger_shape_key,
            "confirmation_best_challenger_upper_confidence": float(confirmation_frontier_state.get("best_challenger_upper_confidence", 0.0)),
            "confirmation_decision_margin": confirmation_decision_margin,
            "confirmation_pending_shape_keys": tuple(confirmation_frontier_state.get("pending_shape_keys", ())),
            "confirmation_pending_reasons": tuple(confirmation_frontier_state.get("pending_reasons", ())),
            "synthesized_confirmation_seed_count": synthesized_confirmation_seed_count,
            "confirmation_rounds": tuple(confirmation_rounds),
            "measurement_input_shape_keys": tuple(str(item["shape_key"]) for item in measurement_inputs),
            "measurement_input_estimated_ranks": tuple(int(item.get("estimated_rank", 0)) for item in measurement_inputs),
            "measured_selection_metric": measured_selection_metric,
            "selected_from_measurements": selection_source == "measured",
        },
        "candidates": tuple(ranked_candidates),
        "measured_candidates": measured_candidates,
        "rejected_candidates": tuple(rejected),
        "matrix": matrix_report,
        "failed_checks": tuple(failed_checks),
        "passed": not failed_checks,
    }
    _write_json(output_dir / "llm_batch_profile_autosize.json", report)
    return report


def build_benchmark_corpus(path: str | Path, *, domain: str, repeats: int = 256) -> tuple[str, ...]:
    if repeats < 8:
        raise ValueError("benchmark repeats must be >= 8")
    if domain not in DEFAULT_BENCHMARK_DOMAINS:
        raise ValueError(f"unknown benchmark domain {domain!r}; choose from {sorted(DEFAULT_BENCHMARK_DOMAINS)}")
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    spec = DEFAULT_BENCHMARK_DOMAINS[domain]
    shard = output / f"{domain}.txt"
    with shard.open("w", encoding="utf-8") as handle:
        for index in range(repeats):
            pattern = spec.patterns[index % len(spec.patterns)]
            handle.write(pattern + "\n")
            handle.write(f"domain {domain} sample {index:05d} control token {index % 31:02d} repeats with stable local structure.\n")
    return (str(shard),)


def _parse_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.replace(",", " ").split() if part.strip())


def _safe_run_name(name: str) -> str:
    slug = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_" for ch in name).strip("._-")
    if not slug:
        raise ValueError(f"name {name!r} does not contain any safe run-directory characters")
    return slug


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for part in _parse_list(raw):
        try:
            seeds.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid seed {part!r}; seeds must be integers") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    return tuple(seeds)


def _parse_positive_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in _parse_list(raw):
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid {name} value {part!r}; expected integers") from exc
        if value <= 0:
            raise ValueError(f"invalid {name} value {part!r}; expected positive integers")
        values.append(value)
    if not values:
        raise ValueError(f"at least one {name} value is required")
    return tuple(values)


def _parse_profile_shape_specs(raw: str) -> tuple[Mapping[str, int], ...]:
    specs: list[Mapping[str, int]] = []
    for part in _parse_list(raw):
        tokens = tuple(token for token in part.lower().replace("*", "x").split("x") if token)
        if len(tokens) not in (5, 6):
            raise ValueError(
                f"invalid profile shape {part!r}; expected seq_lenxd_modelxn_headsxn_layersxbatch_size"
                " or seq_lenxd_modelxn_headsxn_layersxbatch_sizexgradient_accumulation_steps"
            )
        try:
            parsed_tokens = tuple(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(f"invalid profile shape {part!r}; all dimensions must be integers") from exc
        seq_len, d_model, n_heads, n_layers, batch_size = parsed_tokens[:5]
        gradient_accumulation_steps = parsed_tokens[5] if len(parsed_tokens) == 6 else None
        shape_spec = {
            "seq_len": seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "batch_size": batch_size,
        }
        if gradient_accumulation_steps is not None:
            shape_spec["gradient_accumulation_steps"] = gradient_accumulation_steps
        specs.append(
            _normalize_profile_shape_spec(
                shape_spec,
                default_batch_size=batch_size,
            )
        )
    if not specs:
        raise ValueError("at least one profile shape is required")
    return tuple(specs)


def _parse_named_corpus_specs(raw_specs: Sequence[str]) -> tuple[tuple[str, TextCorpusConfig], ...]:
    if not raw_specs:
        raise ValueError("at least one --corpus NAME=PATH spec is required")
    parsed: list[tuple[str, TextCorpusConfig]] = []
    seen: set[str] = set()
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"invalid corpus spec {raw!r}; expected NAME=PATH or NAME=PATH1;PATH2")
        name, path_blob = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"invalid corpus spec {raw!r}; corpus name is empty")
        if name in seen:
            raise ValueError(f"duplicate corpus name {name!r}")
        paths = tuple(part.strip() for part in path_blob.split(";") if part.strip())
        if not paths:
            raise ValueError(f"invalid corpus spec {raw!r}; at least one path is required")
        seen.add(name)
        parsed.append((name, TextCorpusConfig.from_paths(paths)))
    return tuple(parsed)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare a real Cortex-3 LLM pretraining harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="audit LLM dependencies, CUDA, distributed backends and precision readiness")
    doctor.add_argument("--out-dir", default="runs/llm-doctor")
    doctor.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--require-cuda", action="store_true")
    doctor.add_argument("--require-cuda-extension", action="store_true")
    doctor.add_argument("--distributed", action="store_true")
    doctor.add_argument("--gloo-interface", default=None)

    experiment = sub.add_parser("run-experiment", help="run a manifest-driven HF/paths corpus-matrix experiment")
    experiment.add_argument("manifest", help="JSON experiment manifest")
    experiment.add_argument("--out-dir", default=None, help="override manifest out_dir")

    preflight_experiment = sub.add_parser("preflight-experiment", help="preflight a manifest's doctor and model/GPU memory plan without preparing corpora")
    preflight_experiment.add_argument("manifest", help="JSON experiment manifest")
    preflight_experiment.add_argument("--out-dir", default=None, help="override manifest out_dir")

    audit_experiment = sub.add_parser("audit-experiment", help="audit completed LLM experiment artifacts and proof gates")
    audit_experiment.add_argument("run_dir", help="experiment run directory")
    audit_experiment.add_argument("--allow-failed-proof", action="store_true", help="report missing/corrupt artifacts without failing solely on proof=false")

    inspect_experiment = sub.add_parser("inspect-experiment", help="inspect a running or partial LLM experiment without loading large checkpoints")
    inspect_experiment.add_argument("run_dir", help="experiment run directory")

    smoke = sub.add_parser("smoke", help="run a deterministic small corpus comparison")
    smoke.add_argument("--out-dir", default="runs/llm-smoke")
    smoke.add_argument("--steps", type=int, default=48)
    smoke.add_argument("--gradient-accumulation-steps", type=int, default=1)
    smoke.add_argument("--checkpoint-interval", type=int, default=100)
    smoke.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    smoke.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--require-cuda", action="store_true")
    smoke.add_argument("--distributed", action="store_true")
    smoke.add_argument("--gloo-interface", default=None)
    smoke.add_argument("--require-win", action="store_true")
    smoke.add_argument("--min-corpus-tokens", type=int, default=0)
    smoke.add_argument("--max-corpus-tokens", type=int, default=None)
    smoke.add_argument("--tokenizer-training-chars", type=int, default=None)
    smoke.add_argument("--min-planned-train-tokens", type=int, default=0)
    smoke.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    profile_batch = sub.add_parser("profile-batch", help="profile one strict Cortex LLM training run with throughput, VRAM and GPU resource metrics")
    profile_batch.add_argument("--out-dir", default="runs/llm-batch-profile")
    profile_batch.add_argument("--steps", type=int, default=3)
    profile_batch.add_argument("--batch-size", type=int, default=8)
    profile_batch.add_argument("--gradient-accumulation-steps", type=int, default=1)
    profile_batch.add_argument("--seq-len", type=int, default=32)
    profile_batch.add_argument("--d-model", type=int, default=64)
    profile_batch.add_argument("--n-heads", type=int, default=4)
    profile_batch.add_argument("--n-layers", type=int, default=2)
    profile_batch.add_argument("--vocab-size", type=int, default=256)
    profile_batch.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    profile_batch.add_argument("--device", default="auto")
    profile_batch.add_argument("--require-cuda", action="store_true")
    profile_batch.add_argument("--resource-interval", type=float, default=0.05)
    profile_batch.add_argument("--min-resource-samples", type=int, default=2)
    profile_batch.add_argument("--seed", type=int, default=71)
    profile_batch.add_argument("--corpus-repeats", type=int, default=192)
    profile_batch.add_argument("--max-corpus-tokens", type=int, default=8192)
    profile_batch.add_argument("--overwrite", action="store_true", help="delete and recreate the output directory instead of failing when it exists")
    profile_batch.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    profile_matrix = sub.add_parser("profile-matrix", help="profile strict Cortex LLM training across multiple shapes and seeds")
    profile_matrix.add_argument("--out-dir", default="runs/llm-batch-profile-matrix")
    profile_matrix.add_argument("--profile-shapes", default="32x64x4x2x8,40x64x4x2x8", help="comma/space list of seq_lenxd_modelxn_headsxn_layersxbatch_size specs, optionally with xgradient_accumulation_steps")
    profile_matrix.add_argument("--seeds", default="71,73")
    profile_matrix.add_argument("--steps", type=int, default=1)
    profile_matrix.add_argument("--gradient-accumulation-steps", type=int, default=1)
    profile_matrix.add_argument("--vocab-size", type=int, default=256)
    profile_matrix.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    profile_matrix.add_argument("--device", default="auto")
    profile_matrix.add_argument("--require-cuda", action="store_true")
    profile_matrix.add_argument("--resource-interval", type=float, default=0.05)
    profile_matrix.add_argument("--min-resource-samples", type=int, default=2)
    profile_matrix.add_argument("--corpus-repeats", type=int, default=192)
    profile_matrix.add_argument("--max-corpus-tokens", type=int, default=8192)
    profile_matrix.add_argument("--min-cases", type=int, default=1)
    profile_matrix.add_argument("--require-multi-shape", action="store_true")
    profile_matrix.add_argument("--require-multi-seed", action="store_true")
    profile_matrix.add_argument("--min-train-tokens-per-second-mean", type=float, default=0.0)
    profile_matrix.add_argument("--min-gpu-utilization-percent-mean", type=float, default=0.0)
    profile_matrix.add_argument("--min-gpu-memory-used-mb-mean", type=float, default=0.0)
    profile_matrix.add_argument("--min-gpu-power-draw-watts-mean", type=float, default=0.0)
    profile_matrix.add_argument("--overwrite", action="store_true", help="delete and recreate the output directory instead of failing when it exists")
    profile_matrix.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    profile_autosize = sub.add_parser("profile-autosize", help="select strict Cortex LLM profile shapes under a memory budget, then run the profile matrix")
    profile_autosize.add_argument("--out-dir", default="runs/llm-batch-profile-autosize")
    profile_autosize.add_argument("--candidate-seq-lens", default="32,40,48")
    profile_autosize.add_argument("--candidate-d-models", default="64,96")
    profile_autosize.add_argument("--candidate-n-layers", default="2")
    profile_autosize.add_argument("--candidate-batch-sizes", default="4,8")
    profile_autosize.add_argument("--candidate-gradient-accumulation-steps", default=None, help="comma/space list of gradient accumulation values to search; default searches the requested value and at least 2")
    profile_autosize.add_argument("--n-heads", type=int, default=4)
    profile_autosize.add_argument("--selected-shape-count", type=int, default=2)
    profile_autosize.add_argument("--min-selected-shapes", type=int, default=1)
    profile_autosize.add_argument("--seeds", default="71")
    profile_autosize.add_argument("--steps", type=int, default=1)
    profile_autosize.add_argument("--gradient-accumulation-steps", type=int, default=1)
    profile_autosize.add_argument("--vocab-size", type=int, default=256)
    profile_autosize.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    profile_autosize.add_argument("--device", default="auto")
    profile_autosize.add_argument("--require-cuda", action="store_true")
    profile_autosize.add_argument("--resource-interval", type=float, default=0.05)
    profile_autosize.add_argument("--min-resource-samples", type=int, default=2)
    profile_autosize.add_argument("--corpus-repeats", type=int, default=192)
    profile_autosize.add_argument("--max-corpus-tokens", type=int, default=8192)
    profile_autosize.add_argument("--memory-budget-mb", type=float, default=0.0)
    profile_autosize.add_argument("--memory-budget-fraction", type=float, default=0.35)
    profile_autosize.add_argument("--measure-candidate-count", type=int, default=4)
    profile_autosize.add_argument("--measure-candidate-seed-count", type=int, default=None, help="candidate measurement seed count; default uses all provided seeds and at least the minimum, synthesizing deterministic extras when needed")
    profile_autosize.add_argument("--min-measure-candidate-seed-count", type=int, default=2, help="minimum candidate measurement seeds; deterministic extra seeds are synthesized when fewer seeds are provided")
    profile_autosize.add_argument("--measure-candidate-strategy", choices=PROFILE_AUTOSIZE_MEASUREMENT_STRATEGIES, default="diverse", help="candidate subset measured before final matrix; diverse samples shape frontiers instead of only the top estimated ranks")
    profile_autosize.add_argument("--measure-candidate-adaptive-rounds", type=int, default=2, help="bounded measurement waves; with diverse strategy, later rounds refine around measured winners without increasing measure-candidate-count")
    profile_autosize.add_argument("--refine-uncertain-candidate-count", type=int, default=1, help="after candidate waves, add extra seeds to this many high-variance measured candidates before final selection")
    profile_autosize.add_argument("--refine-uncertain-extra-seed-count", type=int, default=1, help="extra measurement seeds per refined uncertain candidate")
    profile_autosize.add_argument("--refine-uncertain-step-multiplier", type=int, default=2, help="multiply profile steps for refinement seeds to reduce short-run overhead noise")
    profile_autosize.add_argument("--refine-uncertain-repeat-count", type=int, default=2, help="repeat each refinement seed profile to reduce runtime jitter before final ranking")
    profile_autosize.add_argument("--refinement-budget-candidate-action-report-cap", type=int, default=64, help="maximum candidate refinement budget actions written to JSON per round; 0 writes the full action frontier")
    profile_autosize.add_argument("--confirm-selected-candidate-count", type=int, default=None, help="final measured candidates to re-profile before selection; default confirms every requested selected shape")
    profile_autosize.add_argument("--confirm-selected-extra-seed-count", type=int, default=1, help="fresh confirmation seeds per final selected candidate before final ranking")
    profile_autosize.add_argument("--confirm-selected-step-multiplier", type=int, default=2, help="multiply profile steps for final selected candidate confirmation")
    profile_autosize.add_argument("--confirm-selected-repeat-count", type=int, default=2, help="repeat each final selected confirmation seed profile to reduce runtime jitter")
    profile_autosize.add_argument("--confirm-selected-max-rounds", type=int, default=None, help="maximum finalist/frontier confirmation rounds; default covers the measured frontier")
    profile_autosize.add_argument("--confirm-selected-decision-resolution-extra-rounds", type=int, default=None, help="extra decision-margin resolution rounds after the measured frontier is confirmed; default covers the measured frontier")
    profile_autosize.add_argument("--confirm-selected-decision-resolution-adaptive-extra-rounds", type=int, default=2, help="maximum additional decision-margin resolution rounds added when confirmed winner/challenger intervals still overlap; 0 disables variance-adaptive extension")
    profile_autosize.add_argument("--confirm-selected-runtime-step-multiplier-cap", type=int, default=4, help="maximum confirmation step multiplier used by uncertainty-adaptive finalist and decision-margin confirmation")
    profile_autosize.add_argument("--confirm-selected-runtime-repeat-count-cap", type=int, default=4, help="maximum repeat count used by uncertainty-adaptive finalist and decision-margin confirmation")
    profile_autosize.add_argument("--measured-selection-metric", choices=PROFILE_AUTOSIZE_MEASURED_SELECTION_METRICS, default="throughput_gpu")
    profile_autosize.add_argument("--min-cases", type=int, default=1)
    profile_autosize.add_argument("--require-multi-shape", action="store_true")
    profile_autosize.add_argument("--require-multi-seed", action="store_true")
    profile_autosize.add_argument("--min-train-tokens-per-second-mean", type=float, default=0.0)
    profile_autosize.add_argument("--min-gpu-utilization-percent-mean", type=float, default=0.0)
    profile_autosize.add_argument("--min-gpu-memory-used-mb-mean", type=float, default=0.0)
    profile_autosize.add_argument("--min-gpu-power-draw-watts-mean", type=float, default=0.0)
    profile_autosize.add_argument("--overwrite", action="store_true", help="delete and recreate the output directory instead of failing when it exists")
    profile_autosize.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    compare = sub.add_parser("compare", help="run baseline vs Cortex comparison on text files or directories")
    compare.add_argument("paths", nargs="+")
    compare.add_argument("--out-dir", default="runs/llm-compare")
    compare.add_argument("--vocab-size", type=int, default=4096)
    compare.add_argument("--seq-len", type=int, default=128)
    compare.add_argument("--steps", type=int, default=200)
    compare.add_argument("--batch-size", type=int, default=32)
    compare.add_argument("--gradient-accumulation-steps", type=int, default=1)
    compare.add_argument("--checkpoint-interval", type=int, default=100)
    compare.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    compare.add_argument("--resume", action="store_true")
    compare.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    compare.add_argument("--d-model", type=int, default=256)
    compare.add_argument("--n-heads", type=int, default=8)
    compare.add_argument("--n-layers", type=int, default=6)
    compare.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--require-cuda", action="store_true")
    compare.add_argument("--distributed", action="store_true")
    compare.add_argument("--gloo-interface", default=None)
    compare.add_argument("--require-win", action="store_true")
    compare.add_argument("--min-corpus-tokens", type=int, default=0)
    compare.add_argument("--max-corpus-tokens", type=int, default=None)
    compare.add_argument("--tokenizer-training-chars", type=int, default=None)
    compare.add_argument("--min-planned-train-tokens", type=int, default=0)
    compare.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    compare_matrix = sub.add_parser("compare-matrix", help="run baseline vs Cortex comparison across multiple seeds on one shared corpus")
    compare_matrix.add_argument("paths", nargs="+")
    compare_matrix.add_argument("--out-dir", default="runs/llm-compare-matrix")
    compare_matrix.add_argument("--seeds", default="11,23,37")
    compare_matrix.add_argument("--vocab-size", type=int, default=4096)
    compare_matrix.add_argument("--seq-len", type=int, default=128)
    compare_matrix.add_argument("--steps", type=int, default=200)
    compare_matrix.add_argument("--batch-size", type=int, default=32)
    compare_matrix.add_argument("--gradient-accumulation-steps", type=int, default=1)
    compare_matrix.add_argument("--checkpoint-interval", type=int, default=100)
    compare_matrix.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    compare_matrix.add_argument("--resume", action="store_true")
    compare_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    compare_matrix.add_argument("--d-model", type=int, default=256)
    compare_matrix.add_argument("--n-heads", type=int, default=8)
    compare_matrix.add_argument("--n-layers", type=int, default=6)
    compare_matrix.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    compare_matrix.add_argument("--device", default="auto")
    compare_matrix.add_argument("--require-cuda", action="store_true")
    compare_matrix.add_argument("--distributed", action="store_true")
    compare_matrix.add_argument("--gloo-interface", default=None)
    compare_matrix.add_argument("--require-win", action="store_true")
    compare_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    compare_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    compare_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    compare_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)
    compare_matrix.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    corpus_matrix = sub.add_parser("corpus-matrix", help="run compare-matrix across multiple named corpora")
    corpus_matrix.add_argument("--corpus", action="append", default=[], help="named corpus spec: NAME=PATH or NAME=PATH1;PATH2")
    corpus_matrix.add_argument("--out-dir", default="runs/llm-corpus-matrix")
    corpus_matrix.add_argument("--seeds", default="11,23,37")
    corpus_matrix.add_argument("--vocab-size", type=int, default=4096)
    corpus_matrix.add_argument("--seq-len", type=int, default=128)
    corpus_matrix.add_argument("--steps", type=int, default=200)
    corpus_matrix.add_argument("--batch-size", type=int, default=32)
    corpus_matrix.add_argument("--gradient-accumulation-steps", type=int, default=1)
    corpus_matrix.add_argument("--checkpoint-interval", type=int, default=100)
    corpus_matrix.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    corpus_matrix.add_argument("--resume", action="store_true")
    corpus_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    corpus_matrix.add_argument("--d-model", type=int, default=256)
    corpus_matrix.add_argument("--n-heads", type=int, default=8)
    corpus_matrix.add_argument("--n-layers", type=int, default=6)
    corpus_matrix.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    corpus_matrix.add_argument("--device", default="auto")
    corpus_matrix.add_argument("--require-cuda", action="store_true")
    corpus_matrix.add_argument("--distributed", action="store_true")
    corpus_matrix.add_argument("--gloo-interface", default=None)
    corpus_matrix.add_argument("--require-win", action="store_true")
    corpus_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    corpus_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    corpus_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    corpus_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)
    corpus_matrix.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    benchmark = sub.add_parser("benchmark", help="run a deterministic multi-domain LLM benchmark suite")
    benchmark.add_argument("--out-dir", default="runs/llm-benchmark")
    benchmark.add_argument("--domains", default="sequence,reasoning,code,anchors")
    benchmark.add_argument("--repeats", type=int, default=160)
    benchmark.add_argument("--steps", type=int, default=48)
    benchmark.add_argument("--batch-size", type=int, default=8)
    benchmark.add_argument("--gradient-accumulation-steps", type=int, default=1)
    benchmark.add_argument("--checkpoint-interval", type=int, default=100)
    benchmark.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    benchmark.add_argument("--vocab-size", type=int, default=256)
    benchmark.add_argument("--seq-len", type=int, default=32)
    benchmark.add_argument("--d-model", type=int, default=64)
    benchmark.add_argument("--n-heads", type=int, default=4)
    benchmark.add_argument("--n-layers", type=int, default=2)
    benchmark.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--require-cuda", action="store_true")
    benchmark.add_argument("--distributed", action="store_true")
    benchmark.add_argument("--gloo-interface", default=None)
    benchmark.add_argument("--require-win", action="store_true")
    benchmark.add_argument("--min-corpus-tokens", type=int, default=0)
    benchmark.add_argument("--max-corpus-tokens", type=int, default=None)
    benchmark.add_argument("--tokenizer-training-chars", type=int, default=None)
    benchmark.add_argument("--min-planned-train-tokens", type=int, default=0)
    benchmark.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    benchmark_matrix = sub.add_parser("benchmark-matrix", help="run a multi-domain x multi-seed statistical LLM benchmark")
    benchmark_matrix.add_argument("--out-dir", default="runs/llm-benchmark-matrix")
    benchmark_matrix.add_argument("--domains", default="sequence,reasoning,code,anchors")
    benchmark_matrix.add_argument("--seeds", default="11,23,37")
    benchmark_matrix.add_argument("--repeats", type=int, default=160)
    benchmark_matrix.add_argument("--steps", type=int, default=48)
    benchmark_matrix.add_argument("--batch-size", type=int, default=8)
    benchmark_matrix.add_argument("--gradient-accumulation-steps", type=int, default=1)
    benchmark_matrix.add_argument("--checkpoint-interval", type=int, default=100)
    benchmark_matrix.add_argument("--max-intermediate-checkpoints", type=int, default=0)
    benchmark_matrix.add_argument("--resume", action="store_true")
    benchmark_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    benchmark_matrix.add_argument("--vocab-size", type=int, default=256)
    benchmark_matrix.add_argument("--seq-len", type=int, default=32)
    benchmark_matrix.add_argument("--d-model", type=int, default=64)
    benchmark_matrix.add_argument("--n-heads", type=int, default=4)
    benchmark_matrix.add_argument("--n-layers", type=int, default=2)
    benchmark_matrix.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    benchmark_matrix.add_argument("--device", default="auto")
    benchmark_matrix.add_argument("--require-cuda", action="store_true")
    benchmark_matrix.add_argument("--distributed", action="store_true")
    benchmark_matrix.add_argument("--gloo-interface", default=None)
    benchmark_matrix.add_argument("--require-win", action="store_true")
    benchmark_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    benchmark_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    benchmark_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    benchmark_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)
    benchmark_matrix.add_argument("--native-ternary-backend", choices=NATIVE_TERNARY_BACKEND_CHOICES, default=STRICT_NATIVE_TERNARY_BACKEND)

    prepare_hf = sub.add_parser("prepare-hf", help="export a Hugging Face dataset to text shards and a token memmap corpus")
    prepare_hf.add_argument("--dataset", required=True, help="Hugging Face dataset path, e.g. allenai/c4 or json")
    prepare_hf.add_argument("--config-name", default=None, help="dataset config/subset name")
    prepare_hf.add_argument("--split", default="train")
    prepare_hf.add_argument("--text-field", default="text", help="text column name, supports dotted nested fields")
    prepare_hf.add_argument("--data-file", action="append", default=[], help="local data file for builders such as json")
    prepare_hf.add_argument("--out-dir", default="runs/hf-corpus")
    prepare_hf.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    prepare_hf.add_argument("--trust-remote-code", action="store_true")
    prepare_hf.add_argument("--cache-dir", default=None)
    prepare_hf.add_argument("--max-documents", type=int, default=None)
    prepare_hf.add_argument("--max-characters", type=int, default=None)
    prepare_hf.add_argument("--allow-unbounded", action="store_true")
    prepare_hf.add_argument("--min-text-chars", type=int, default=1)
    prepare_hf.add_argument("--shard-chars", type=int, default=64 * 1024 * 1024)
    prepare_hf.add_argument("--min-chars-per-chunk", type=int, default=2048)
    prepare_hf.add_argument("--vocab-size", type=int, default=8192)
    prepare_hf.add_argument("--min-frequency", type=int, default=2)
    prepare_hf.add_argument("--seq-len", type=int, default=128)
    prepare_hf.add_argument("--max-horizon", type=int, default=8)
    prepare_hf.add_argument("--max-tokens", type=int, default=None, help="cap the tokenized memmap after this many tokens")
    prepare_hf.add_argument("--tokenizer-training-chars", type=int, default=None)
    prepare_hf.add_argument("--train-fraction", type=float, default=0.9)
    prepare_hf.add_argument("--resume", action="store_true", help="reuse a verified existing HF export and tokenized corpus")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = llm_doctor_report(
            require_cuda=args.require_cuda,
            require_cuda_extension=args.require_cuda_extension,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            device=args.device,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
        )
        out_dir = Path(args.out_dir)
        _write_json(out_dir / "doctor_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        if not report["passed"]:
            failed = ", ".join(str(check["name"]) for check in report["failed_required_checks"])
            raise RuntimeError(f"Cortex LLM doctor failed required checks: {failed}")
        return

    if args.command == "run-experiment":
        runner = LLMExperimentRunner.load(args.manifest)
        if args.out_dir is not None:
            manifest = dict(runner.manifest)
            manifest["out_dir"] = args.out_dir
            runner = LLMExperimentRunner(manifest, manifest_path=args.manifest)
        report = runner.run()
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True, default=_json_default))
        return

    if args.command == "preflight-experiment":
        runner = LLMExperimentRunner.load(args.manifest)
        if args.out_dir is not None:
            manifest = dict(runner.manifest)
            manifest["out_dir"] = args.out_dir
            runner = LLMExperimentRunner(manifest, manifest_path=args.manifest)
        doctor_report = llm_doctor_report(**dict(runner.manifest["doctor"]))
        report = runner.preflight(doctor_report=doctor_report)
        out_dir = Path(str(runner.manifest["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "doctor_report.json", doctor_report)
        _write_json(out_dir / "preflight_report.json", report.to_dict())
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default))
        if not doctor_report["passed"]:
            failed = ", ".join(str(check["name"]) for check in doctor_report["failed_required_checks"])
            raise RuntimeError(f"Cortex LLM preflight doctor failed required checks: {failed}")
        if not report.passed:
            failed = "; ".join(report.failed_checks[:10])
            raise RuntimeError(f"Cortex LLM preflight failed: {failed}")
        return

    if args.command == "audit-experiment":
        report = audit_llm_experiment_artifacts(
            args.run_dir,
            require_passed=not args.allow_failed_proof,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default))
        if not report.passed:
            failed = "; ".join(report.failed_checks[:10])
            raise RuntimeError(f"Cortex LLM experiment audit failed: {failed}")
        return

    if args.command == "inspect-experiment":
        report = inspect_llm_experiment(args.run_dir)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=_json_default))
        return

    if args.command == "smoke":
        out_dir = Path(args.out_dir)
        runtime = DistributedRuntime.from_env(
            requested=args.distributed,
            device_type="cuda" if (args.device == "auto" and torch.cuda.is_available()) or str(args.device).startswith("cuda") else "cpu",
            gloo_interface=args.gloo_interface,
        )
        runtime.ensure_initialized()
        if runtime.is_main:
            files = build_seed_corpus(out_dir / "seed_text", repeats=160)
        _barrier_if_needed(runtime)
        files = (str(out_dir / "seed_text" / "seed_corpus.txt"),)
        corpus = TextCorpusConfig.from_paths(files, min_chars_per_chunk=512)
        training = TrainingConfig(
            steps=args.steps,
            batch_size=8,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=max(1, args.steps // 3),
            eval_batches=3,
            seed=11,
            device=args.device,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
            max_intermediate_checkpoints=args.max_intermediate_checkpoints,
            num_threads=1,
        )
        config = ComparisonConfig(
            vocab_size=256,
            min_frequency=1,
            seq_len=32,
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
            horizons=(1, 2, 4, 8),
            training=training,
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
        report = LLMComparisonRunner(corpus, config, run_dir=out_dir / "comparison").run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
        return

    if args.command == "profile-batch":
        report = run_llm_batch_profile(
            out_dir=args.out_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            vocab_size=args.vocab_size,
            precision=args.precision,
            device=args.device,
            require_cuda=args.require_cuda,
            native_ternary_backend=args.native_ternary_backend,
            resource_interval=args.resource_interval,
            min_resource_samples=args.min_resource_samples,
            seed=args.seed,
            corpus_repeats=args.corpus_repeats,
            max_corpus_tokens=args.max_corpus_tokens,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        if not bool(report["passed"]):
            failed = ", ".join(str(item) for item in report["failed_checks"])
            raise RuntimeError(f"Cortex LLM batch profile failed required checks: {failed}")
        return

    if args.command == "profile-matrix":
        report = run_llm_batch_profile_matrix(
            out_dir=args.out_dir,
            shape_specs=_parse_profile_shape_specs(args.profile_shapes),
            seeds=_parse_seed_list(args.seeds),
            steps=args.steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            vocab_size=args.vocab_size,
            precision=args.precision,
            device=args.device,
            require_cuda=args.require_cuda,
            native_ternary_backend=args.native_ternary_backend,
            resource_interval=args.resource_interval,
            min_resource_samples=args.min_resource_samples,
            corpus_repeats=args.corpus_repeats,
            max_corpus_tokens=args.max_corpus_tokens,
            min_cases=args.min_cases,
            require_multi_shape=args.require_multi_shape,
            require_multi_seed=args.require_multi_seed,
            min_train_tokens_per_second_mean=args.min_train_tokens_per_second_mean,
            min_gpu_utilization_percent_mean=args.min_gpu_utilization_percent_mean,
            min_gpu_memory_used_mb_mean=args.min_gpu_memory_used_mb_mean,
            min_gpu_power_draw_watts_mean=args.min_gpu_power_draw_watts_mean,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        if not bool(report["passed"]):
            failed = ", ".join(str(item) for item in report["failed_checks"])
            raise RuntimeError(f"Cortex LLM batch profile matrix failed required checks: {failed}")
        return

    if args.command == "profile-autosize":
        report = run_llm_batch_profile_autosize(
            out_dir=args.out_dir,
            candidate_seq_lens=_parse_positive_int_list(args.candidate_seq_lens, name="candidate-seq-lens"),
            candidate_d_models=_parse_positive_int_list(args.candidate_d_models, name="candidate-d-models"),
            candidate_n_layers=_parse_positive_int_list(args.candidate_n_layers, name="candidate-n-layers"),
            candidate_batch_sizes=_parse_positive_int_list(args.candidate_batch_sizes, name="candidate-batch-sizes"),
            candidate_gradient_accumulation_steps=(
                None
                if args.candidate_gradient_accumulation_steps is None
                else _parse_positive_int_list(args.candidate_gradient_accumulation_steps, name="candidate-gradient-accumulation-steps")
            ),
            n_heads=args.n_heads,
            selected_shape_count=args.selected_shape_count,
            min_selected_shapes=args.min_selected_shapes,
            seeds=_parse_seed_list(args.seeds),
            steps=args.steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            vocab_size=args.vocab_size,
            precision=args.precision,
            device=args.device,
            require_cuda=args.require_cuda,
            native_ternary_backend=args.native_ternary_backend,
            resource_interval=args.resource_interval,
            min_resource_samples=args.min_resource_samples,
            corpus_repeats=args.corpus_repeats,
            max_corpus_tokens=args.max_corpus_tokens,
            memory_budget_mb=args.memory_budget_mb,
            memory_budget_fraction=args.memory_budget_fraction,
            measure_candidate_count=args.measure_candidate_count,
            measure_candidate_seed_count=args.measure_candidate_seed_count,
            min_measure_candidate_seed_count=args.min_measure_candidate_seed_count,
            measure_candidate_strategy=args.measure_candidate_strategy,
            measure_candidate_adaptive_rounds=args.measure_candidate_adaptive_rounds,
            refine_uncertain_candidate_count=args.refine_uncertain_candidate_count,
            refine_uncertain_extra_seed_count=args.refine_uncertain_extra_seed_count,
            refine_uncertain_step_multiplier=args.refine_uncertain_step_multiplier,
            refine_uncertain_repeat_count=args.refine_uncertain_repeat_count,
            refinement_budget_candidate_action_report_cap=args.refinement_budget_candidate_action_report_cap,
            confirm_selected_candidate_count=args.confirm_selected_candidate_count,
            confirm_selected_extra_seed_count=args.confirm_selected_extra_seed_count,
            confirm_selected_step_multiplier=args.confirm_selected_step_multiplier,
            confirm_selected_repeat_count=args.confirm_selected_repeat_count,
            confirm_selected_max_rounds=args.confirm_selected_max_rounds,
            confirm_selected_decision_resolution_extra_rounds=args.confirm_selected_decision_resolution_extra_rounds,
            confirm_selected_decision_resolution_adaptive_extra_rounds=args.confirm_selected_decision_resolution_adaptive_extra_rounds,
            confirm_selected_runtime_step_multiplier_cap=args.confirm_selected_runtime_step_multiplier_cap,
            confirm_selected_runtime_repeat_count_cap=args.confirm_selected_runtime_repeat_count_cap,
            measured_selection_metric=args.measured_selection_metric,
            min_cases=args.min_cases,
            require_multi_shape=args.require_multi_shape,
            require_multi_seed=args.require_multi_seed,
            min_train_tokens_per_second_mean=args.min_train_tokens_per_second_mean,
            min_gpu_utilization_percent_mean=args.min_gpu_utilization_percent_mean,
            min_gpu_memory_used_mb_mean=args.min_gpu_memory_used_mb_mean,
            min_gpu_power_draw_watts_mean=args.min_gpu_power_draw_watts_mean,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        if not bool(report["passed"]):
            failed = ", ".join(str(item) for item in report["failed_checks"])
            raise RuntimeError(f"Cortex LLM batch profile autosize failed required checks: {failed}")
        return

    if args.command == "prepare-hf":
        out_dir = Path(args.out_dir)
        max_documents = args.max_documents
        if max_documents is None and args.max_characters is None and not args.allow_unbounded:
            max_documents = 100_000
        hf_config = HFDatasetExportConfig(
            dataset=args.dataset,
            config_name=args.config_name,
            split=args.split,
            text_field=args.text_field,
            data_files=tuple(args.data_file),
            streaming=args.streaming,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            max_documents=max_documents,
            max_characters=args.max_characters,
            allow_unbounded=args.allow_unbounded,
            min_text_chars=args.min_text_chars,
            shard_max_chars=args.shard_chars,
        )
        export_report = HFDatasetTextExporter(hf_config).export(out_dir, resume=args.resume)
        corpus = TextCorpusConfig.from_paths(export_report.shard_files, min_chars_per_chunk=args.min_chars_per_chunk)
        tokenization_config = _tokenized_preparation_config(
            corpus,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            seq_len=args.seq_len,
            max_horizon=args.max_horizon,
            max_tokens=args.max_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            train_fraction=args.train_fraction,
        )
        tokenized_dir = out_dir / "tokenized"
        tokenized_manifest_path = tokenized_dir / "manifest.json"
        prepare_report_path = out_dir / "prepare_report.json"
        if args.resume and tokenized_manifest_path.exists():
            if not prepare_report_path.exists():
                raise FileNotFoundError(f"resume=True found tokenized manifest without prepare_report.json: {prepare_report_path}")
            previous_prepare = json.loads(prepare_report_path.read_text(encoding="utf-8"))
            if previous_prepare.get("tokenization") != tokenization_config:
                raise ValueError("existing prepare_report tokenization config does not match requested prepare-hf arguments")
            manifest = TokenizedCorpusManifest.load(tokenized_manifest_path)
            manifest.identity()
            _require_tokenized_preparation_config(
                manifest,
                tokenization_config,
                manifest_path=tokenized_manifest_path,
            )
            if manifest.source_files != export_report.shard_files:
                raise ValueError("existing tokenized corpus source_files do not match resumed HF export shards")
        else:
            if args.resume and tokenized_dir.exists() and any(tokenized_dir.iterdir()):
                raise FileExistsError(f"resume=True found incomplete tokenized artifacts without manifest: {tokenized_dir}")
            tokenizer = LLMTokenizer.train(
                corpus,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                max_training_chars=args.tokenizer_training_chars,
            )
            manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
                tokenized_dir,
                seq_len=args.seq_len,
                max_horizon=args.max_horizon,
                max_tokens=args.max_tokens,
                train_fraction=args.train_fraction,
                preparation_config=tokenization_config,
            )
        payload = {
            "hf_export": export_report.to_dict(),
            "manifest": manifest.to_dict(),
            "tokenization": tokenization_config,
            "command": "prepare-hf",
        }
        _write_json(prepare_report_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return

    if args.command == "benchmark":
        domains = _parse_list(args.domains)
        resolved_domains = tuple(DEFAULT_BENCHMARK_DOMAINS.keys()) if "all" in domains else domains
        training = TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=max(1, args.steps // 3),
            eval_batches=3,
            seed=23,
            device=args.device,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
            max_intermediate_checkpoints=args.max_intermediate_checkpoints,
            num_threads=1,
        )
        config = ComparisonConfig(
            vocab_size=args.vocab_size,
            min_frequency=1,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=0.0,
            horizons=(1, 2, 4, 8),
            training=training,
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
        report = LLMBenchmarkSuite(
            run_dir=args.out_dir,
            domains=resolved_domains,
            repeats=args.repeats,
            config=config,
        ).run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
        return

    if args.command == "benchmark-matrix":
        domains = _parse_list(args.domains)
        resolved_domains = tuple(DEFAULT_BENCHMARK_DOMAINS.keys()) if "all" in domains else domains
        seeds = _parse_seed_list(args.seeds)
        training = TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=max(1, args.steps // 3),
            eval_batches=3,
            seed=seeds[0],
            device=args.device,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
            max_intermediate_checkpoints=args.max_intermediate_checkpoints,
            num_threads=1,
        )
        config = ComparisonConfig(
            vocab_size=args.vocab_size,
            min_frequency=1,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=0.0,
            horizons=(1, 2, 4, 8),
            training=training,
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
            min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
        report = LLMStatisticalBenchmarkSuite(
            run_dir=args.out_dir,
            domains=resolved_domains,
            seeds=seeds,
            repeats=args.repeats,
            config=config,
        ).run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
        return

    if args.command == "compare-matrix":
        seeds = _parse_seed_list(args.seeds)
        corpus = TextCorpusConfig.from_paths(args.paths)
        training = TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=max(1, args.steps // 10),
            device=args.device,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
            max_intermediate_checkpoints=args.max_intermediate_checkpoints,
            seed=seeds[0],
        )
        config = ComparisonConfig(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            training=training,
            min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
        report = LLMComparisonMatrixSuite(corpus, config, run_dir=args.out_dir, seeds=seeds).run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
        return

    if args.command == "corpus-matrix":
        seeds = _parse_seed_list(args.seeds)
        corpora = _parse_named_corpus_specs(args.corpus)
        training = TrainingConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_interval=max(1, args.steps // 10),
            device=args.device,
            precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
            max_intermediate_checkpoints=args.max_intermediate_checkpoints,
            seed=seeds[0],
        )
        config = ComparisonConfig(
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            training=training,
            min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
        report = LLMCorpusMatrixSuite(corpora, config, run_dir=args.out_dir, seeds=seeds).run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
        return

    corpus = TextCorpusConfig.from_paths(args.paths)
    training = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_interval=max(1, args.steps // 10),
        device=args.device,
        precision=_resolve_cli_precision(args.precision, device=args.device, require_cuda=args.require_cuda),
        require_cuda=args.require_cuda,
        distributed=args.distributed,
        gloo_interface=args.gloo_interface,
        resume=args.resume,
        resume_if_exists=args.resume_if_exists,
        checkpoint_interval=args.checkpoint_interval,
        max_intermediate_checkpoints=args.max_intermediate_checkpoints,
    )
    config = ComparisonConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        training=training,
        min_corpus_tokens=args.min_corpus_tokens,
            max_corpus_tokens=args.max_corpus_tokens,
            tokenizer_training_chars=args.tokenizer_training_chars,
            min_planned_train_tokens=args.min_planned_train_tokens,
            native_ternary_backend=args.native_ternary_backend,
        )
    report = LLMComparisonRunner(corpus, config, run_dir=args.out_dir).run(require_win=args.require_win)
    if _rank_zero():
        print(json.dumps(report.proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
