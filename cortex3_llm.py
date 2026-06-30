from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


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
    truncated_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HFDatasetTextExporter:
    def __init__(self, config: HFDatasetExportConfig):
        self.config = config

    def export(self, output_dir: str | Path) -> HFDatasetExportReport:
        output = Path(output_dir)
        shard_dir = output / "text_shards"
        output.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
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
            truncated_reason=truncated_reason,
        )
        _write_json(output / "hf_export_report.json", report.to_dict())
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
        return load_dataset(self.config.dataset, **kwargs)

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
    def train(config: TextCorpusConfig, *, vocab_size: int = 4096, min_frequency: int = 2) -> "LLMTokenizer":
        if vocab_size < len(SPECIAL_TOKENS) + 16:
            raise ValueError("vocab_size is too small for a useful BPE tokenizer")
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
        tokenizer.train_from_iterator(reader.iter_chunks(), trainer=trainer)
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load(path: str | Path) -> "TokenizedCorpusManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["source_files"] = tuple(payload["source_files"])
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
    ) -> TokenizedCorpusManifest:
        if seq_len < 8:
            raise ValueError("seq_len must be >= 8")
        if max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if not 0.1 <= train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.1, 1.0)")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        token_path = out / "tokens.uint32"
        tokenizer_path = out / "tokenizer.json"
        manifest_path = out / "manifest.json"
        self.tokenizer.save(tokenizer_path)

        token_count = 0
        for tokens in self._iter_token_chunks():
            token_count += len(tokens)
        minimum = seq_len + max_horizon + 2
        if token_count < minimum:
            raise ValueError(f"corpus produced {token_count} tokens, need at least {minimum}")

        memmap = np.memmap(token_path, dtype=np.uint32, mode="w+", shape=(token_count,))
        offset = 0
        for tokens in self._iter_token_chunks():
            size = len(tokens)
            memmap[offset:offset + size] = np.asarray(tokens, dtype=np.uint32)
            offset += size
        memmap.flush()
        del memmap

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
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        futures: list[torch.Tensor] = []
        for offset in offsets.tolist():
            x, y, future = self.item(int(offset))
            xs.append(x)
            ys.append(y)
            futures.append(future)
        return (
            torch.stack(xs).to(device),
            torch.stack(ys).to(device),
            torch.stack(futures).to(device),
        )

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


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model)
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
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.GELU(),
            nn.Linear(config.d_model * 4, config.d_model),
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleDict({
            str(horizon): nn.Linear(config.d_model, config.vocab_size)
            for horizon in config.horizons
        }) if config.use_cortex_heads else nn.ModuleDict()
        self.confidence_head = nn.Linear(config.d_model, 1) if config.use_cortex_heads else None
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
    resume_from_checkpoint: str | None = None
    checkpoint_interval: int = 100
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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["curve"] = [point.to_dict() for point in self.curve]
        payload["final_train"] = self.final_train.to_dict()
        payload["final_val"] = self.final_val.to_dict()
        return payload


def hardware_report() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "distributed_available": bool(torch.distributed.is_available()),
        "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
        "gloo_available": bool(torch.distributed.is_available() and torch.distributed.is_gloo_available()),
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
        optimizer = torch.optim.AdamW(trainable.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scaler = self.precision.scaler(self.device.type)
        curve: list[TrainingPoint] = []
        resumed_from = self._resolve_resume_checkpoint()
        start_step = 0
        if resumed_from is not None:
            start_step, curve = self.load_checkpoint(resumed_from, optimizer=optimizer, scaler=scaler, restore_rng=not runtime.enabled)
            if start_step > self.config.steps:
                raise ValueError(f"checkpoint step {start_step} is greater than target steps {self.config.steps}")
            if runtime.enabled:
                resumed_seed = self.config.seed + runtime.rank * 100_003 + start_step * 997
                torch.manual_seed(resumed_seed)
                random.seed(resumed_seed)
                np.random.seed(resumed_seed)
                self.generator.manual_seed(resumed_seed)
        else:
            curve.append(self.evaluate(self.train_data, split="train", step=0))
            curve.append(self.evaluate(self.val_data, split="val", step=0))
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
                        loss, _ = self.objective.compute(
                            output,
                            y,
                            future,
                            use_cortex_terms=self.model.config.use_cortex_heads,
                        )
                        loss = loss / self.config.gradient_accumulation_steps
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
            if step % self.config.eval_interval == 0 or step == self.config.steps:
                curve.append(self.evaluate(self.train_data, split="train", step=step))
                curve.append(self.evaluate(self.val_data, split="val", step=step))
            if runtime.is_main and step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(optimizer, self.run_dir / f"checkpoint_step_{step}.pt", step=step, curve=curve, scaler=scaler)
        checkpoint_path = self.run_dir / "checkpoint_final.pt"
        if runtime.is_main:
            checkpoint_path = self.save_checkpoint(optimizer, checkpoint_path, step=self.config.steps, curve=curve, scaler=scaler)
            self._write_curve(curve)
        final_train = [point for point in curve if point.split == "train"][-1]
        final_val = [point for point in curve if point.split == "val"][-1]
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
            },
            hardware=hardware_report(),
        )
        if runtime.is_main:
            _write_json(self.run_dir / "training_report.json", report.to_dict())
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
        if not self.config.resume:
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
        raise FileNotFoundError(f"resume=True but no checkpoint was found in {self.run_dir}")

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        restore_rng: bool = True,
    ) -> tuple[int, list[TrainingPoint]]:
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if payload.get("model_kind") != self.model_kind:
            raise ValueError(f"checkpoint model_kind={payload.get('model_kind')!r} does not match {self.model_kind!r}")
        checkpoint_model_config = payload.get("model_config")
        if checkpoint_model_config != asdict(self.model.config):
            raise ValueError("checkpoint model_config does not match the current model")
        self.model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(payload["scaler_state_dict"])
        if restore_rng:
            rng_state = payload.get("rng_state", {})
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"].cpu())
            if self.device.type == "cuda" and rng_state.get("torch_cuda_all"):
                torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])
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
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
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
                "curve": [point.to_dict() for point in curve],
                "rng_state": self._rng_state(),
            },
            output,
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


@dataclass(frozen=True)
class ComparisonReport:
    run_dir: str
    manifest: Mapping[str, Any]
    baseline: Mapping[str, Any]
    cortex: Mapping[str, Any]
    proof: Mapping[str, Any]
    hardware: Mapping[str, Any]

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


class LLMComparisonRunner:
    def __init__(self, corpus: TextCorpusConfig, config: ComparisonConfig, *, run_dir: str | Path):
        self.corpus = corpus
        self.config = config
        self.run_dir = Path(run_dir)

    def prepare_corpus(self) -> TokenizedCorpusManifest:
        tokenizer = LLMTokenizer.train(self.corpus, vocab_size=self.config.vocab_size, min_frequency=self.config.min_frequency)
        return TokenizedCorpusBuilder(self.corpus, tokenizer).build(
            self.run_dir / "corpus",
            seq_len=self.config.seq_len,
            max_horizon=max(self.config.horizons),
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
        if runtime.is_main:
            if self.run_dir.exists() and not self.config.training.resume:
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
            ).train(name="baseline_ntp")
            cortex_config = TransformerConfig(**{**asdict(model_config), "use_cortex_heads": True})
            cortex = LLMTrainer(
                CortexTransformerLM(cortex_config),
                train_data,
                val_data,
                self.config.training,
                run_dir=self.run_dir / "cortex3",
                model_kind="cortex3_multi_horizon",
            ).train(name="cortex3")
        finally:
            train_data.close()
            val_data.close()
        proof = self._proof_payload(baseline, cortex)
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
        )
        if runtime.is_main:
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

    def _proof_payload(self, baseline: TrainingRunReport, cortex: TrainingRunReport) -> dict[str, Any]:
        baseline_score = baseline.final_val.future_tokens_per_cost
        cortex_score = cortex.final_val.future_tokens_per_cost
        ratio = cortex_score / max(1e-9, baseline_score)
        next_token_regression = cortex.final_val.next_token_loss / max(1e-9, baseline.final_val.next_token_loss)
        passed = ratio >= self.config.cortex_win_margin and next_token_regression <= self.config.max_next_token_loss_regression
        return {
            "metric": "verified_future_tokens_per_forward_cost",
            "baseline_score": baseline_score,
            "cortex_score": cortex_score,
            "cortex_over_baseline_ratio": ratio,
            "required_margin": self.config.cortex_win_margin,
            "next_token_loss_regression_ratio": next_token_regression,
            "max_next_token_loss_regression": self.config.max_next_token_loss_regression,
            "passed": passed,
        }

    def _write_markdown(self, report: ComparisonReport) -> None:
        proof = report.proof
        lines = [
            "# Cortex-3 LLM comparison report",
            "",
            f"- Proof metric: `{proof['metric']}`",
            f"- Baseline score: `{proof['baseline_score']:.6f}`",
            f"- Cortex score: `{proof['cortex_score']:.6f}`",
            f"- Cortex/baseline ratio: `{proof['cortex_over_baseline_ratio']:.3f}`",
            f"- Next-token loss regression ratio: `{proof['next_token_loss_regression_ratio']:.3f}`",
            f"- Passed: `{proof['passed']}`",
            "",
            "## Artifacts",
            "",
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
            if self.run_dir.exists() and not self.config.training.resume:
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
        all_domain_proofs = [bool(item["proof"]["passed"]) and float(item["proof"]["baseline_score"]) > 0.0 for item in domains]
        return {
            "metric": "benchmark_mean_cortex_over_baseline",
            "domains": [str(item["domain"]) for item in domains],
            "domain_count": len(domains),
            "mean_ratio": sum(ratios) / max(1, len(ratios)),
            "min_ratio": min(ratios) if ratios else 0.0,
            "mean_baseline_score": sum(baseline_scores) / max(1, len(baseline_scores)),
            "max_next_token_loss_regression": max(regressions) if regressions else 0.0,
            "all_domains_passed": all(all_domain_proofs),
            "passed": bool(domains) and all(all_domain_proofs),
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and compare a real Cortex-3 LLM pretraining harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="run a deterministic small corpus comparison")
    smoke.add_argument("--out-dir", default="runs/llm-smoke")
    smoke.add_argument("--steps", type=int, default=48)
    smoke.add_argument("--gradient-accumulation-steps", type=int, default=1)
    smoke.add_argument("--checkpoint-interval", type=int, default=100)
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--require-cuda", action="store_true")
    smoke.add_argument("--distributed", action="store_true")
    smoke.add_argument("--gloo-interface", default=None)
    smoke.add_argument("--require-win", action="store_true")

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
    compare.add_argument("--d-model", type=int, default=256)
    compare.add_argument("--n-heads", type=int, default=8)
    compare.add_argument("--n-layers", type=int, default=6)
    compare.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    compare.add_argument("--device", default="auto")
    compare.add_argument("--require-cuda", action="store_true")
    compare.add_argument("--distributed", action="store_true")
    compare.add_argument("--gloo-interface", default=None)
    compare.add_argument("--require-win", action="store_true")

    benchmark = sub.add_parser("benchmark", help="run a deterministic multi-domain LLM benchmark suite")
    benchmark.add_argument("--out-dir", default="runs/llm-benchmark")
    benchmark.add_argument("--domains", default="sequence,reasoning,code,anchors")
    benchmark.add_argument("--repeats", type=int, default=160)
    benchmark.add_argument("--steps", type=int, default=48)
    benchmark.add_argument("--batch-size", type=int, default=8)
    benchmark.add_argument("--gradient-accumulation-steps", type=int, default=1)
    benchmark.add_argument("--checkpoint-interval", type=int, default=100)
    benchmark.add_argument("--resume", action="store_true")
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
    prepare_hf.add_argument("--train-fraction", type=float, default=0.9)

    args = parser.parse_args(argv)
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
            horizons=(1, 2, 4),
            training=training,
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
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
        export_report = HFDatasetTextExporter(hf_config).export(out_dir)
        corpus = TextCorpusConfig.from_paths(export_report.shard_files, min_chars_per_chunk=args.min_chars_per_chunk)
        tokenizer = LLMTokenizer.train(corpus, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
        manifest = TokenizedCorpusBuilder(corpus, tokenizer).build(
            out_dir / "tokenized",
            seq_len=args.seq_len,
            max_horizon=args.max_horizon,
            train_fraction=args.train_fraction,
        )
        payload = {
            "hf_export": export_report.to_dict(),
            "manifest": manifest.to_dict(),
            "command": "prepare-hf",
        }
        _write_json(out_dir / "prepare_report.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
        return

    if args.command == "benchmark":
        domains = tuple(part.strip() for part in args.domains.replace(",", " ").split() if part.strip())
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
            horizons=(1, 2, 4),
            training=training,
            cortex_win_margin=1.02,
            max_next_token_loss_regression=1.50,
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
        checkpoint_interval=args.checkpoint_interval,
    )
    config = ComparisonConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        training=training,
    )
    report = LLMComparisonRunner(corpus, config, run_dir=args.out_dir).run(require_win=args.require_win)
    if _rank_zero():
        print(json.dumps(report.proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
