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

from cortex3_llm import CortexObjective, CortexTransformerLM, LossBreakdown, TransformerConfig


def _loss_dict(breakdown: LossBreakdown) -> dict[str, float]:
    return {
        "total": float(breakdown.total),
        "next_token": float(breakdown.next_token),
        "mtp": float(breakdown.mtp),
        "temporal_consistency": float(breakdown.temporal_consistency),
        "confidence": float(breakdown.confidence),
        "variable_input": float(breakdown.variable_input),
        "learned_memory": float(breakdown.learned_memory),
        "certificate": float(breakdown.certificate),
    }


def _policy_summary(policy: Any) -> dict[str, float | int]:
    probs = policy.probs.detach()
    modes = probs.argmax(dim=-1)
    return {
        "exact_prob_mean": float(policy.exact_prob.detach().mean().cpu()),
        "latent_prob_mean": float(policy.latent_prob.detach().mean().cpu()),
        "drop_prob_mean": float(policy.drop_prob.detach().mean().cpu()),
        "storage_ratio_mean": float(policy.storage_ratio.detach().mean().cpu()),
        "entropy_mean": float(policy.entropy.detach().mean().cpu()),
        "exact_decisions": int(modes.eq(0).sum().cpu()),
        "latent_decisions": int(modes.eq(1).sum().cpu()),
        "drop_decisions": int(modes.eq(2).sum().cpu()),
        "tokens": int(modes.numel()),
    }


def _load_shared_weights(source: CortexTransformerLM, target: CortexTransformerLM) -> int:
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        key: value.detach().clone()
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target.load_state_dict(shared, strict=False)
    return len(shared)


def _synthetic_batch(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = 2
    future = torch.stack((y, y, y, y), dim=-1)
    return x, y, future


def _evaluate(
    model: CortexTransformerLM,
    objective: CortexObjective,
    x: torch.Tensor,
    y: torch.Tensor,
    future: torch.Tensor,
) -> tuple[torch.Tensor, LossBreakdown, Any]:
    output = model(x)
    loss, breakdown = objective.compute(output, y, future, use_cortex_terms=True)
    return loss, breakdown, output.learned_memory_policy


def run_learned_memory_ablation(
    *,
    seed: int = 202,
    steps: int = 8,
    learning_rate: float = 0.03,
    device: str = "auto",
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 128,
    d_model: int = 32,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for learned-memory ablation, but torch.cuda.is_available() is false")
        torch_device = torch.device("cuda")
    elif device == "cpu":
        torch_device = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cpu, cuda")

    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    common_config = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.0,
        "horizons": (1, 2, 4, 8),
        "use_cortex_heads": True,
        "use_ternary_core": True,
        "use_native_ternary_kernel": torch_device.type == "cuda",
        "require_native_ternary_kernel": False,
        "use_skill_aware_experts": False,
        "use_variable_in_compressor": True,
        "use_certificate_head": False,
    }
    learned = CortexTransformerLM(TransformerConfig(**common_config, use_learned_memory_policy=True)).to(torch_device)
    disabled = CortexTransformerLM(TransformerConfig(**common_config, use_learned_memory_policy=False)).to(torch_device)
    shared_weight_tensors = _load_shared_weights(learned, disabled)

    for name, parameter in learned.named_parameters():
        parameter.requires_grad_(name.startswith("learned_memory."))
    trainable = [parameter for parameter in learned.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("learned-memory ablation found no trainable memory parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    objective = CortexObjective()
    x, y, future = _synthetic_batch(
        vocab_size=vocab_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=torch_device,
    )

    with torch.no_grad():
        _, disabled_breakdown, _ = _evaluate(disabled, objective, x, y, future)
    loss, before_breakdown, before_policy = _evaluate(learned, objective, x, y, future)
    before_policy_summary = _policy_summary(before_policy)
    step_reports: list[dict[str, Any]] = []
    max_gradient_l1 = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, breakdown, policy = _evaluate(learned, objective, x, y, future)
        loss.backward()
        gradient_l1 = sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in trainable
            if parameter.grad is not None
        )
        max_gradient_l1 = max(max_gradient_l1, gradient_l1)
        optimizer.step()
        step_reports.append({
            "step": int(step),
            "loss": _loss_dict(breakdown),
            "policy": _policy_summary(policy),
            "learned_memory_gradient_l1": gradient_l1,
        })

    with torch.no_grad():
        _, after_breakdown, after_policy = _evaluate(learned, objective, x, y, future)
    after_policy_summary = _policy_summary(after_policy)
    before_loss = _loss_dict(before_breakdown)
    after_loss = _loss_dict(after_breakdown)
    disabled_loss = _loss_dict(disabled_breakdown)
    return {
        "seed": int(seed),
        "device": str(torch_device),
        "shared_weight_tensors": int(shared_weight_tensors),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "disabled_memory_loss": disabled_loss,
        "learned_memory_before_loss": before_loss,
        "learned_memory_after_loss": after_loss,
        "loss_delta_before_minus_after": {
            key: float(before_loss[key] - after_loss[key])
            for key in before_loss
        },
        "loss_delta_disabled_minus_after": {
            key: float(disabled_loss[key] - after_loss[key])
            for key in disabled_loss
        },
        "learned_memory_before_policy": before_policy_summary,
        "learned_memory_after_policy": after_policy_summary,
        "policy_probability_shift_l1": float(
            abs(before_policy_summary["exact_prob_mean"] - after_policy_summary["exact_prob_mean"])
            + abs(before_policy_summary["latent_prob_mean"] - after_policy_summary["latent_prob_mean"])
            + abs(before_policy_summary["drop_prob_mean"] - after_policy_summary["drop_prob_mean"])
        ),
        "max_learned_memory_gradient_l1": float(max_gradient_l1),
        "step_reports": step_reports,
        "passed_short_ablation": bool(
            max_gradient_l1 > 0.0
            and after_loss["total"] < before_loss["total"]
            and after_loss["next_token"] < before_loss["next_token"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short learned-memory policy ablation.")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    started = time()
    report = run_learned_memory_ablation(
        seed=args.seed,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
    )
    report["elapsed_seconds"] = float(time() - started)
    if torch.cuda.is_available():
        report["cuda_device"] = torch.cuda.get_device_name()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
