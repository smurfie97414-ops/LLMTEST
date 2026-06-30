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
    CandidateAnswer,
    CostTrace,
    CorruptedCompressedAgent,
    DynamicSkillVerifier,
    ReferenceRuleAgent,
    Task,
    default_skill_specs,
)
from cortex3_attribution import CausalAttributionEngine
from cortex3_certificates import CertificateType, CertificateVerifier, LatentProofState, build_certificate, evaluate_certificate_efficiency
from cortex3_cycle import CortexCycle
from cortex3_future import FutureContractEngine, FutureContractLedger, MTPFSPConfig
from cortex3_improvement import RecursiveImprovementEngine
from cortex3_inference import InferenceConfig, UltraFastInferenceEngine
from cortex3_memory import CognitiveMemory, CognitiveMemoryConfig
from cortex3_objective import build_objective_report
from cortex3_phases import CORTEX3_PHASES
from cortex3_regrowth import MinimalRegrowthEngine
from cortex3_sleep import ExampleOrigin, SleepPhaseConsolidator, TrainingExample
from cortex3_ternary import BitLinear, BitLinearConfig, CompressionTraceLedger


SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


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
class LLMForwardOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    mtp_logits: Mapping[int, torch.Tensor]
    confidence: torch.Tensor | None


class CortexTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.compression_ledger = CompressionTraceLedger() if config.use_ternary_core else None
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, ledger=self.compression_ledger, layer_index=index)
            for index in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
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
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> LLMForwardOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch, time_steps = input_ids.shape
        if time_steps > self.config.seq_len:
            raise ValueError(f"input sequence length {time_steps} exceeds model seq_len {self.config.seq_len}")
        positions = torch.arange(0, time_steps, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.drop(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        mtp_logits = {
            int(horizon): head(hidden)
            for horizon, head in ((int(key), module) for key, module in self.mtp_heads.items())
        }
        confidence = torch.sigmoid(self.confidence_head(hidden)).squeeze(-1) if self.confidence_head else None
        return LLMForwardOutput(logits=logits, hidden=hidden, mtp_logits=mtp_logits, confidence=confidence)

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


@dataclass(frozen=True)
class LossWeights:
    next_token: float = 1.0
    mtp: float = 0.35
    temporal_consistency: float = 0.05
    confidence: float = 0.05


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    next_token: float
    mtp: float = 0.0
    temporal_consistency: float = 0.0
    confidence: float = 0.0

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
        next_loss = F.cross_entropy(output.logits.reshape(-1, vocab_size), next_targets.reshape(-1))
        total = self.weights.next_token * next_loss
        mtp_loss = output.logits.new_tensor(0.0)
        temporal_loss = output.logits.new_tensor(0.0)
        confidence_loss = output.logits.new_tensor(0.0)
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
            total = total + self.weights.temporal_consistency * temporal_loss
        return total, LossBreakdown(
            total=float(total.detach().cpu()),
            next_token=float(next_loss.detach().cpu()),
            mtp=float(mtp_loss.detach().cpu()),
            temporal_consistency=float(temporal_loss.detach().cpu()),
            confidence=float(confidence_loss.detach().cpu()),
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
    resource_monitor_interval: float = 2.0
    cortex_phase_interval: int = 0
    cortex_phase_probe_tasks: int = 1
    cortex_phase_max_proposals: int = 1
    cortex_phase_regularization_weight: float = 0.001
    cortex_phase_replay_weight: float = 0.05
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
        if self.resource_monitor_interval <= 0:
            raise ValueError("resource_monitor_interval must be positive")
        if self.cortex_phase_interval < 0:
            raise ValueError("cortex_phase_interval must be non-negative")
        if self.cortex_phase_probe_tasks < 1:
            raise ValueError("cortex_phase_probe_tasks must be positive")
        if self.cortex_phase_max_proposals < 0:
            raise ValueError("cortex_phase_max_proposals must be non-negative")
        if self.cortex_phase_regularization_weight < 0:
            raise ValueError("cortex_phase_regularization_weight must be non-negative")
        if self.cortex_phase_replay_weight < 0:
            raise ValueError("cortex_phase_replay_weight must be non-negative")
        if self.resume and self.resume_if_exists:
            raise ValueError("resume and resume_if_exists are mutually exclusive")


def _training_allows_existing_artifacts(training: TrainingConfig) -> bool:
    return bool(training.resume or training.resume_if_exists)


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
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
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
            return {
                "gpu_index": index,
                "gpu_utilization_percent": float(parts[1]),
                "gpu_memory_utilization_percent": float(parts[2]),
                "gpu_memory_used_mb": float(parts[3]),
                "gpu_memory_total_mb": float(parts[4]),
            }
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


def llm_doctor_report(
    *,
    require_cuda: bool = False,
    precision: str = "bf16",
    device: str = "auto",
    distributed: bool = False,
    gloo_interface: str | None = None,
) -> dict[str, Any]:
    hardware = hardware_report()
    dependencies = _package_versions()
    device_type = "cuda" if (device == "auto" and torch.cuda.is_available()) or str(device).startswith("cuda") else "cpu"
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str, *, required: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "required": required, "detail": detail})

    for package, payload in dependencies.items():
        add_check(f"dependency:{package}", bool(payload["installed"]), f"version={payload['version']}")
    add_check("torch:cuda_available", bool(hardware["cuda_available"]), f"cuda_device_count={hardware['cuda_device_count']}", required=require_cuda)
    if require_cuda:
        add_check("torch:require_cuda", device_type == "cuda" and bool(hardware["cuda_available"]), f"resolved_device_type={device_type}")

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
            "precision": precision,
            "device": device,
            "distributed": distributed,
            "gloo_interface": gloo_interface,
        },
        "hardware": hardware,
        "dependencies": dependencies,
        "checks": tuple(checks),
        "failed_required_checks": tuple(failed_required),
    }


def _last_items(items: Sequence[Any], limit: int = 3) -> list[Any]:
    if limit <= 0:
        return []
    return list(items[-limit:])


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
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.run_dir = Path(run_dir)
        self.verifier = DynamicSkillVerifier(default_skill_specs())
        self.reference_agent = ReferenceRuleAgent()
        self.trial_agent = CorruptedCompressedAgent()
        self.cycle = CortexCycle(self.verifier)
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
        self.attribution = CausalAttributionEngine(self.verifier)
        self.regrowth = MinimalRegrowthEngine(self.verifier)
        self.inference = UltraFastInferenceEngine(
            self.verifier,
            self.reference_agent,
            memory=self.memory,
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
        self.phase_counts: dict[str, int] = {phase.id: 0 for phase in CORTEX3_PHASES}
        self.errors: list[dict[str, Any]] = []
        self.batch_contract_samples: list[dict[str, Any]] = []
        self.phase_audits: list[dict[str, Any]] = []
        self.replay_batches: list[torch.Tensor] = []
        self.replay_cursor = 0
        self.regularization_steps = 0
        self.replay_updates = 0
        self.phase_replay_examples: dict[str, int] = {phase.id: 0 for phase in CORTEX3_PHASES}
        self.phase_replay_example_ids: list[str] = []

    def _touch(self, phase_id: str) -> None:
        self.phase_counts[phase_id] = self.phase_counts.get(phase_id, 0) + 1

    def _record_error(self, phase_id: str, exc: Exception) -> None:
        self.errors.append({"phase": phase_id, "type": type(exc).__name__, "message": str(exc)})

    def interval(self) -> int:
        return int(self.config.cortex_phase_interval or self.config.eval_interval)

    def should_sample_step(self, step: int) -> bool:
        return step == 0 or step % self.interval() == 0 or step == self.config.steps

    def auxiliary_loss(self, output: LLMForwardOutput) -> torch.Tensor:
        if output.confidence is None or self.config.cortex_phase_regularization_weight <= 0:
            return output.logits.new_tensor(0.0)
        self.regularization_steps += 1
        confidence_pressure = (1.0 - output.confidence.mean()).clamp_min(0.0)
        return confidence_pressure * float(self.config.cortex_phase_regularization_weight)

    def observe_batch_contract(
        self,
        *,
        step: int,
        output: LLMForwardOutput,
        future_targets: torch.Tensor,
        breakdown: LossBreakdown,
    ) -> None:
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
            self._touch("P3")
            decision_label = "ACCEPT" if decision.accepted else "REJECT"
            decision_task = Task(
                f"phase-p3-{step}-{len(self.batch_contract_samples)}",
                "instruction_following",
                f"Output the future contract gate result exactly: {decision_label}",
                decision_label,
            )
            self._add_verified_phase_replay(
                "P3",
                decision_task,
                metadata={
                    "contract_id": decision.contract.contract_id,
                    "accepted_horizon": decision.contract.accepted_horizon,
                    "observed_token_count": len(observed),
                    "reason": decision.reason,
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
        batch = self.replay_batches[self.replay_cursor % len(self.replay_batches)].to(device)
        self.replay_cursor += 1
        x = batch[:, :-1]
        y = batch[:, 1:]
        future = self._future_targets_from_next_tokens(y)
        with precision.autocast(device.type):
            output = model_forward(x)
            loss, _ = objective.compute(output, y, future, use_cortex_terms=self.model.config.use_cortex_heads)
        self.replay_updates += 1
        return loss * float(self.config.cortex_phase_replay_weight)

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
        self.phase_replay_examples[phase_id] = self.phase_replay_examples.get(phase_id, 0) + 1
        self.phase_replay_example_ids.append(example.example_id)
        return example

    def _add_sleep_replay(self, examples: Sequence[TrainingExample]) -> None:
        for example in examples[:8]:
            self.replay_batches.append(self._batch_from_example(example))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase_counts": dict(self.phase_counts),
            "errors": list(self.errors),
            "batch_contract_samples": list(self.batch_contract_samples),
            "phase_audits": list(self.phase_audits),
            "replay_batches": [batch.detach().cpu() for batch in self.replay_batches],
            "replay_cursor": int(self.replay_cursor),
            "regularization_steps": int(self.regularization_steps),
            "replay_updates": int(self.replay_updates),
            "phase_replay_examples": dict(self.phase_replay_examples),
            "phase_replay_example_ids": list(self.phase_replay_example_ids),
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
        self.replay_batches = [
            batch.detach().cpu().to(dtype=torch.long)
            if isinstance(batch, torch.Tensor)
            else torch.as_tensor(batch, dtype=torch.long)
            for batch in payload.get("replay_batches", ())
        ]
        self.replay_cursor = int(payload.get("replay_cursor", 0))
        self.regularization_steps = int(payload.get("regularization_steps", 0))
        self.replay_updates = int(payload.get("replay_updates", 0))
        self.phase_replay_examples.update({
            str(key): int(value)
            for key, value in dict(payload.get("phase_replay_examples", {})).items()
        })
        self.phase_replay_example_ids = [str(item) for item in payload.get("phase_replay_example_ids", ())]

    def checkpoint_state_summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase_event_counts": dict(self.phase_counts),
            "replay_batch_count": len(self.replay_batches),
            "replay_cursor": int(self.replay_cursor),
            "regularization_steps": int(self.regularization_steps),
            "replay_updates": int(self.replay_updates),
            "phase_replay_examples": sum(self.phase_replay_examples.values()),
            "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
            "error_count": len(self.errors),
        }

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
            self._touch("P4")
            audit["memory"] = {
                "selected_segment_ids": reconstruction.selected_segment_ids,
                "fidelity": asdict(reconstruction.fidelity),
                "compression": self.memory.compression_report(),
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
            efficiency = evaluate_certificate_efficiency(
                "slow visible reasoning " * 32,
                certificate,
                verification,
                reference_uncertainty=0.05,
            )
            self._touch("P5")
            audit["certificate"] = {
                "verified": verification.to_dict(),
                "efficiency": asdict(efficiency),
                "certificate": certificate.to_dict(),
            }
            if verification.passed:
                self._add_verified_phase_replay(
                    "P5",
                    task,
                    metadata={
                        "certificate_id": certificate.certificate_id,
                        "reduction_ratio": efficiency.reduction_ratio,
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
                regrowth_plan = self.regrowth.plan(
                    attribution_report,
                    self.trial_agent,
                    (attribution_report.failure.task,),
                    budget=10.0,
                )
                self._touch("P7")
                audit["regrowth"] = regrowth_plan.to_dict()
                self._add_verified_phase_replay(
                    "P7",
                    regrowth_plan.failure.task,
                    metadata={
                        "selected_action": regrowth_plan.selected_action,
                        "candidate_count": len(regrowth_plan.candidates),
                    },
                )
        except Exception as exc:
            self._record_error("P7", exc)

        inference_results: list[Any] = []
        try:
            task = first_failure.task if first_failure is not None else Task("phase-infer", "instruction_following", "Output OK exactly.", "OK")
            inferred = self.inference.infer(task)
            inference_results.append(inferred)
            self._touch("P8")
            audit["inference"] = inferred.to_dict()
            if inferred.passed:
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
                sleep_report = self.sleep.ingest_cycle(cycle_report, seed=self.config.seed + step)
                self._add_sleep_replay(sleep_report.accepted_examples)
                self._touch("P9")
                audit["sleep"] = sleep_report.to_dict()
        except Exception as exc:
            self._record_error("P9", exc)

        improvement_report = None
        try:
            if cycle_report is not None and self.config.cortex_phase_max_proposals > 0:
                improvement_report = self.improvement.run(
                    cycle_report,
                    baseline_agent=self.trial_agent,
                    reference_agent=self.reference_agent,
                    max_proposals=self.config.cortex_phase_max_proposals,
                    seed=self.config.seed + step,
                    n_per_skill=self.config.cortex_phase_probe_tasks,
                )
                self._touch("P10")
                audit["recursive_improvement"] = improvement_report.to_dict()
                if improvement_report.decisions:
                    accepted_decision = next((decision for decision in improvement_report.decisions if decision.accepted), None)
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
        except Exception as exc:
            self._record_error("objective", exc)

        self.phase_audits.append(audit)
        return audit

    def summary(self) -> Mapping[str, Any]:
        compression_trace = self.model.compression_trace()
        trace_counts = {}
        if compression_trace.get("enabled"):
            trace_counts = {
                "compression_decisions": len(compression_trace.get("compression_decisions", ())),
                "activation_quantizations": len(compression_trace.get("activation_quantizations", ())),
                "kv_events": len(compression_trace.get("kv_events", ())),
                "mtp_fsp_events": len(compression_trace.get("mtp_fsp_events", ())),
                "layer_forward_events": len(compression_trace.get("layer_forward_events", ())),
            }
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
        return {
            "schema_version": 1,
            "enabled": True,
            "all_phases_active": all(item["active_in_llm_training"] for item in phases),
            "phases": phases,
            "phase_event_counts": dict(self.phase_counts),
            "training_influence": {
                "ternary_core_forward_events": trace_counts.get("layer_forward_events", 0),
                "future_contract_decisions": len(self.future_ledger.decisions),
                "confidence_regularization_steps": self.regularization_steps,
                "sleep_replay_batches_available": len(self.replay_batches),
                "sleep_replay_updates": self.replay_updates,
                "phase_replay_examples": sum(self.phase_replay_examples.values()),
                "phase_replay_examples_by_phase": dict(self.phase_replay_examples),
            },
            "trace_counts": trace_counts,
            "future_ledger": self.future_ledger.to_dict(),
            "phase_audits": _last_items(self.phase_audits, 2),
            "batch_contract_samples": _last_items(self.batch_contract_samples, 5),
            "phase_replay_example_ids": _last_items(self.phase_replay_example_ids, 10),
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
        self.precision = PrecisionPolicy(config.precision, require_cuda=config.require_cuda)
        self.corpus_identity = dict(corpus_identity or train_data.manifest.identity())
        self.tokenizer = LLMTokenizer.load(self.train_data.manifest.tokenizer_file)
        if self.val_data.manifest.identity(verify=False) != self.train_data.manifest.identity(verify=False):
            raise ValueError("train and validation datasets must come from the same tokenized corpus manifest")
        if config.num_threads is not None:
            torch.set_num_threads(config.num_threads)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

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
                if start_step > self.config.steps:
                    raise ValueError(f"checkpoint step {start_step} is greater than target steps {self.config.steps}")
                if runtime.enabled:
                    resumed_seed = self.config.seed + runtime.rank * 100_003 + start_step * 997
                    torch.manual_seed(resumed_seed)
                    random.seed(resumed_seed)
                    np.random.seed(resumed_seed)
                    self.generator.manual_seed(resumed_seed)
                if phase_controller is not None and start_step >= self.config.steps:
                    self._prime_phase_controller(phase_controller, step=start_step)
            else:
                curve.append(self.evaluate(self.train_data, split="train", step=0))
                curve.append(self.evaluate(self.val_data, split="val", step=0))
                if phase_controller is not None:
                    self._prime_phase_controller(phase_controller, step=0)
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
            code_state=code_state_report(),
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
        for path in self.run_dir.glob("checkpoint_step_*.pt"):
            raw_step = path.stem.removeprefix("checkpoint_step_")
            if raw_step.isdigit():
                candidates.append((int(raw_step), path))
        if candidates:
            return sorted(candidates)[-1][1]
        if self.config.resume_if_exists:
            return None
        raise FileNotFoundError(f"resume=True but no checkpoint was found in {self.run_dir}")

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
        checkpoint_model_config = payload.get("model_config")
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
        code_state = code_state_report()
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
            output,
        )
        _write_json(
            output.with_name(output.name + ".json"),
            {
                "schema_version": 1,
                "checkpoint": str(output),
                "checkpoint_size_bytes": int(output.stat().st_size),
                "step": step,
                "model_kind": self.model_kind,
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
    return {
        "exists": model_dir.exists(),
        "latest_checkpoint_step": max((item["step"] for item in step_checkpoints), default=None),
        "checkpoint_count": len(step_checkpoints),
        "latest_checkpoint": step_checkpoints[-1] if step_checkpoints else None,
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
    cortex_config = TransformerConfig(**{**asdict(baseline_config), "use_cortex_heads": True, "use_ternary_core": True})
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
    cortex_config = TransformerConfig(**{**asdict(baseline_config), "use_cortex_heads": True, "use_ternary_core": True})
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
            cortex_config = TransformerConfig(**{**asdict(model_config), "use_cortex_heads": True, "use_ternary_core": True})
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
            and tuple(int(value) for value in raw_horizons) == (1, 2, 4, 8)
        )
        phase_report = cortex.cortex_phase_report if isinstance(cortex.cortex_phase_report, Mapping) else {}
        cortex_phase_integration_passed = (
            (not cortex_full_phase_required)
            or (bool(phase_report.get("all_phases_active")) and not phase_report.get("errors"))
        )
        checks = {
            "finite_metrics": finite_metrics,
            "baseline_score_passed": baseline_score_passed,
            "ratio_passed": ratio_passed,
            "next_token_regression_passed": next_token_regression_passed,
            "learning_curve_audit_passed": learning_curve_audit_passed,
            "corpus_scale_passed": corpus_scale_passed,
            "planned_train_tokens_passed": planned_train_tokens_passed,
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
            handle.write(patterns[index % len(patterns)] + "\n")
            handle.write(f"sample {index:04d} keeps sequence marker {index % 17:02d}.\n")
    return (str(shard),)


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
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    smoke.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--require-cuda", action="store_true")
    smoke.add_argument("--distributed", action="store_true")
    smoke.add_argument("--gloo-interface", default=None)
    smoke.add_argument("--require-win", action="store_true")
    smoke.add_argument("--min-corpus-tokens", type=int, default=0)
    smoke.add_argument("--max-corpus-tokens", type=int, default=None)
    smoke.add_argument("--tokenizer-training-chars", type=int, default=None)
    smoke.add_argument("--min-planned-train-tokens", type=int, default=0)

    compare = sub.add_parser("compare", help="run baseline vs Cortex comparison on text files or directories")
    compare.add_argument("paths", nargs="+")
    compare.add_argument("--out-dir", default="runs/llm-compare")
    compare.add_argument("--vocab-size", type=int, default=4096)
    compare.add_argument("--seq-len", type=int, default=128)
    compare.add_argument("--steps", type=int, default=200)
    compare.add_argument("--batch-size", type=int, default=32)
    compare.add_argument("--gradient-accumulation-steps", type=int, default=1)
    compare.add_argument("--checkpoint-interval", type=int, default=100)
    compare.add_argument("--resume", action="store_true")
    compare.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    compare.add_argument("--d-model", type=int, default=256)
    compare.add_argument("--n-heads", type=int, default=8)
    compare.add_argument("--n-layers", type=int, default=6)
    compare.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--require-cuda", action="store_true")
    compare.add_argument("--distributed", action="store_true")
    compare.add_argument("--gloo-interface", default=None)
    compare.add_argument("--require-win", action="store_true")
    compare.add_argument("--min-corpus-tokens", type=int, default=0)
    compare.add_argument("--max-corpus-tokens", type=int, default=None)
    compare.add_argument("--tokenizer-training-chars", type=int, default=None)
    compare.add_argument("--min-planned-train-tokens", type=int, default=0)

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
    compare_matrix.add_argument("--resume", action="store_true")
    compare_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    compare_matrix.add_argument("--d-model", type=int, default=256)
    compare_matrix.add_argument("--n-heads", type=int, default=8)
    compare_matrix.add_argument("--n-layers", type=int, default=6)
    compare_matrix.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    compare_matrix.add_argument("--device", default="auto")
    compare_matrix.add_argument("--require-cuda", action="store_true")
    compare_matrix.add_argument("--distributed", action="store_true")
    compare_matrix.add_argument("--gloo-interface", default=None)
    compare_matrix.add_argument("--require-win", action="store_true")
    compare_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    compare_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    compare_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    compare_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)

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
    corpus_matrix.add_argument("--resume", action="store_true")
    corpus_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    corpus_matrix.add_argument("--d-model", type=int, default=256)
    corpus_matrix.add_argument("--n-heads", type=int, default=8)
    corpus_matrix.add_argument("--n-layers", type=int, default=6)
    corpus_matrix.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    corpus_matrix.add_argument("--device", default="auto")
    corpus_matrix.add_argument("--require-cuda", action="store_true")
    corpus_matrix.add_argument("--distributed", action="store_true")
    corpus_matrix.add_argument("--gloo-interface", default=None)
    corpus_matrix.add_argument("--require-win", action="store_true")
    corpus_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    corpus_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    corpus_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    corpus_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)

    benchmark = sub.add_parser("benchmark", help="run a deterministic multi-domain LLM benchmark suite")
    benchmark.add_argument("--out-dir", default="runs/llm-benchmark")
    benchmark.add_argument("--domains", default="sequence,reasoning,code,anchors")
    benchmark.add_argument("--repeats", type=int, default=160)
    benchmark.add_argument("--steps", type=int, default=48)
    benchmark.add_argument("--batch-size", type=int, default=8)
    benchmark.add_argument("--gradient-accumulation-steps", type=int, default=1)
    benchmark.add_argument("--checkpoint-interval", type=int, default=100)
    benchmark.add_argument("--resume", action="store_true")
    benchmark.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    benchmark.add_argument("--vocab-size", type=int, default=256)
    benchmark.add_argument("--seq-len", type=int, default=32)
    benchmark.add_argument("--d-model", type=int, default=64)
    benchmark.add_argument("--n-heads", type=int, default=4)
    benchmark.add_argument("--n-layers", type=int, default=2)
    benchmark.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--require-cuda", action="store_true")
    benchmark.add_argument("--distributed", action="store_true")
    benchmark.add_argument("--gloo-interface", default=None)
    benchmark.add_argument("--require-win", action="store_true")
    benchmark.add_argument("--min-corpus-tokens", type=int, default=0)
    benchmark.add_argument("--max-corpus-tokens", type=int, default=None)
    benchmark.add_argument("--tokenizer-training-chars", type=int, default=None)
    benchmark.add_argument("--min-planned-train-tokens", type=int, default=0)

    benchmark_matrix = sub.add_parser("benchmark-matrix", help="run a multi-domain x multi-seed statistical LLM benchmark")
    benchmark_matrix.add_argument("--out-dir", default="runs/llm-benchmark-matrix")
    benchmark_matrix.add_argument("--domains", default="sequence,reasoning,code,anchors")
    benchmark_matrix.add_argument("--seeds", default="11,23,37")
    benchmark_matrix.add_argument("--repeats", type=int, default=160)
    benchmark_matrix.add_argument("--steps", type=int, default=48)
    benchmark_matrix.add_argument("--batch-size", type=int, default=8)
    benchmark_matrix.add_argument("--gradient-accumulation-steps", type=int, default=1)
    benchmark_matrix.add_argument("--checkpoint-interval", type=int, default=100)
    benchmark_matrix.add_argument("--resume", action="store_true")
    benchmark_matrix.add_argument("--resume-if-exists", action="store_true", help="resume from existing verified artifacts if present, otherwise start fresh")
    benchmark_matrix.add_argument("--vocab-size", type=int, default=256)
    benchmark_matrix.add_argument("--seq-len", type=int, default=32)
    benchmark_matrix.add_argument("--d-model", type=int, default=64)
    benchmark_matrix.add_argument("--n-heads", type=int, default=4)
    benchmark_matrix.add_argument("--n-layers", type=int, default=2)
    benchmark_matrix.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    benchmark_matrix.add_argument("--device", default="auto")
    benchmark_matrix.add_argument("--require-cuda", action="store_true")
    benchmark_matrix.add_argument("--distributed", action="store_true")
    benchmark_matrix.add_argument("--gloo-interface", default=None)
    benchmark_matrix.add_argument("--require-win", action="store_true")
    benchmark_matrix.add_argument("--min-corpus-tokens", type=int, default=0)
    benchmark_matrix.add_argument("--max-corpus-tokens", type=int, default=None)
    benchmark_matrix.add_argument("--tokenizer-training-chars", type=int, default=None)
    benchmark_matrix.add_argument("--min-planned-train-tokens", type=int, default=0)

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
            precision=args.precision,
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
            precision=args.precision,
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
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
        )
        report = LLMComparisonRunner(corpus, config, run_dir=out_dir / "comparison").run(require_win=args.require_win)
        if _rank_zero():
            print(json.dumps(report.proof, indent=2, sort_keys=True))
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
            precision=args.precision,
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
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
            precision=args.precision,
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
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
            precision=args.precision,
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
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
            precision=args.precision,
            require_cuda=args.require_cuda,
            distributed=args.distributed,
            gloo_interface=args.gloo_interface,
            resume=args.resume,
            resume_if_exists=args.resume_if_exists,
            checkpoint_interval=args.checkpoint_interval,
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
        precision=args.precision,
        require_cuda=args.require_cuda,
        distributed=args.distributed,
        gloo_interface=args.gloo_interface,
        resume=args.resume,
        resume_if_exists=args.resume_if_exists,
        checkpoint_interval=args.checkpoint_interval,
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
    )
    report = LLMComparisonRunner(corpus, config, run_dir=args.out_dir).run(require_win=args.require_win)
    if _rank_zero():
        print(json.dumps(report.proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
