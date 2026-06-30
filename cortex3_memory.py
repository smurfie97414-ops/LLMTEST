from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import torch

from cortex3 import Anchor, CostTrace, ExactAnchorLedger, extract_anchors


_TOKEN_RE = re.compile(r"[\wÀ-ÿ-]+", re.UNICODE)
_ANCHOR_INTENT: Mapping[str, tuple[str, ...]] = {
    "code": ("identifier",),
    "identifiant": ("identifier",),
    "id": ("identifier",),
    "identifier": ("identifier",),
    "montant": ("number", "amount"),
    "amount": ("number", "amount"),
    "nombre": ("number",),
    "number": ("number",),
    "date": ("number", "date"),
    "chemin": ("path",),
    "path": ("path",),
    "ville": ("city",),
    "city": ("city",),
    "personne": ("person",),
    "person": ("person",),
    "nom": ("person",),
    "item": ("item",),
    "objet": ("item",),
    "contrainte": ("constraint",),
    "variable": ("variable",),
}


class MemoryMode(str, Enum):
    EXACT = "exact"
    LATENT = "latent"


@dataclass(frozen=True)
class CognitiveMemoryConfig:
    recent_exact_limit: int = 2
    embedding_dim: int = 64
    top_k_exact: int = 2
    top_k_latent: int = 3
    max_summary_terms: int = 16
    anchor_boost: float = 0.40

    def __post_init__(self) -> None:
        if self.recent_exact_limit < 1:
            raise ValueError("recent_exact_limit must be at least 1")
        if self.embedding_dim < 8:
            raise ValueError("embedding_dim must be at least 8")
        if self.top_k_exact < 0 or self.top_k_latent < 0:
            raise ValueError("top_k values cannot be negative")
        if self.max_summary_terms < 1:
            raise ValueError("max_summary_terms must be at least 1")


@dataclass(frozen=True)
class MemorySegment:
    segment_id: str
    mode: MemoryMode
    exact_text: str
    latent_summary: str
    anchors: tuple[Anchor, ...]
    token_counts: Mapping[str, int]
    embedding: torch.Tensor
    original_token_count: int
    stored_token_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        return self.stored_token_count / max(self.original_token_count, 1)

    def rendered(self) -> str:
        if self.mode == MemoryMode.EXACT:
            return self.exact_text
        anchor_text = " ".join(f"{anchor.kind}={anchor.value}" for anchor in self.anchors)
        return f"{self.latent_summary}\n{anchor_text}".strip()


@dataclass(frozen=True)
class AnchorFidelityResult:
    required: int
    preserved: int
    missing: tuple[Anchor, ...]
    score: float

    @property
    def passed(self) -> bool:
        return self.required == self.preserved


@dataclass(frozen=True)
class MemoryReconstruction:
    query: str
    exact_context: tuple[str, ...]
    latent_context: tuple[str, ...]
    anchors: tuple[Anchor, ...]
    selected_segment_ids: tuple[str, ...]
    fidelity: AnchorFidelityResult
    cost: CostTrace

    @property
    def rendered_context(self) -> str:
        sections = list(self.exact_context) + list(self.latent_context)
        if self.anchors:
            sections.append("ANCHORS " + " ".join(f"{anchor.kind}={anchor.value}" for anchor in self.anchors))
        return "\n".join(section for section in sections if section)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(text))


def _stable_hash_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def embed_tokens(tokens: Iterable[str], dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    for token in tokens:
        hashed = _stable_hash_int(token)
        index = hashed % dim
        sign = 1.0 if (hashed >> 8) & 1 else -1.0
        vector[index] += sign
    norm = vector.norm()
    return vector / norm if float(norm) > 0.0 else vector


def embed_text(text: str, dim: int) -> torch.Tensor:
    return embed_tokens(tokenize(text), dim)


def _summary_from_counts(counts: Mapping[str, int], anchors: Sequence[Anchor], max_terms: int) -> str:
    anchor_values = {anchor.value.lower() for anchor in anchors}
    filtered = [
        (token, count)
        for token, count in counts.items()
        if len(token) > 2 and token not in anchor_values
    ]
    ranked = sorted(filtered, key=lambda item: (-item[1], item[0]))[:max_terms]
    return " ".join(token for token, _ in ranked)


class RecentExactKV:
    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("recent exact KV limit must be at least 1")
        self.limit = limit
        self.segments: list[MemorySegment] = []

    def push(self, segment: MemorySegment) -> MemorySegment | None:
        if segment.mode != MemoryMode.EXACT:
            raise ValueError("RecentExactKV only stores exact segments")
        self.segments.append(segment)
        if len(self.segments) > self.limit:
            return self.segments.pop(0)
        return None


class LatentKVStore:
    def __init__(self, config: CognitiveMemoryConfig):
        self.config = config
        self.segments: list[MemorySegment] = []

    def compress_from_exact(self, segment: MemorySegment) -> MemorySegment:
        summary_budget = max(0, min(self.config.max_summary_terms, segment.original_token_count - len(segment.anchors) - 1))
        summary = _summary_from_counts(segment.token_counts, segment.anchors, summary_budget)
        summary_tokens = tokenize(summary)
        latent = MemorySegment(
            segment_id=segment.segment_id,
            mode=MemoryMode.LATENT,
            exact_text="",
            latent_summary=summary,
            anchors=segment.anchors,
            token_counts=segment.token_counts,
            embedding=segment.embedding.detach().clone(),
            original_token_count=segment.original_token_count,
            stored_token_count=len(summary_tokens) + len(segment.anchors),
            metadata={**dict(segment.metadata), "compressed_from": MemoryMode.EXACT.value},
        )
        self.segments.append(latent)
        return latent

    def retrieve(self, query_embedding: torch.Tensor, query_tokens: set[str], anchor_kinds: set[str], top_k: int) -> list[MemorySegment]:
        scored = []
        for segment in self.segments:
            score = float(torch.dot(query_embedding, segment.embedding))
            if query_tokens.intersection(segment.token_counts):
                score += 0.10
            if any(anchor.kind in anchor_kinds or anchor.value.lower() in query_tokens for anchor in segment.anchors):
                score += self.config.anchor_boost
            scored.append((score, segment))
        return [segment for score, segment in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k] if score > 0.0]


class AnchorFidelityVerifier:
    def verify(self, text: str, required: Sequence[Anchor]) -> AnchorFidelityResult:
        if not required:
            return AnchorFidelityResult(0, 0, tuple(), 1.0)
        missing = tuple(anchor for anchor in required if anchor.value not in text)
        preserved = len(required) - len(missing)
        return AnchorFidelityResult(len(required), preserved, missing, preserved / len(required))


class CognitiveMemory:
    def __init__(self, config: CognitiveMemoryConfig | None = None):
        self.config = config or CognitiveMemoryConfig()
        self.anchor_ledger = ExactAnchorLedger()
        self.recent = RecentExactKV(self.config.recent_exact_limit)
        self.latent = LatentKVStore(self.config)
        self.fidelity = AnchorFidelityVerifier()

    def _make_exact_segment(self, segment_id: str, text: str, metadata: Mapping[str, Any] | None = None, extra_anchors: Iterable[Anchor] = ()) -> MemorySegment:
        extracted = self.anchor_ledger.ingest(text, segment_id)
        anchors = tuple(dict.fromkeys(tuple(extracted) + tuple(extra_anchors)))
        tokens = tokenize(text)
        counts = Counter(tokens)
        anchor_tokens = [token for anchor in anchors for token in tokenize(anchor.value)]
        embedding = embed_tokens(tuple(tokens) + tuple(anchor_tokens), self.config.embedding_dim)
        return MemorySegment(
            segment_id=segment_id,
            mode=MemoryMode.EXACT,
            exact_text=text,
            latent_summary="",
            anchors=anchors,
            token_counts=dict(counts),
            embedding=embedding,
            original_token_count=len(tokens),
            stored_token_count=len(tokens),
            metadata=dict(metadata or {}),
        )

    def ingest(self, segment_id: str, text: str, metadata: Mapping[str, Any] | None = None, extra_anchors: Iterable[Anchor] = ()) -> MemorySegment:
        if not segment_id:
            raise ValueError("segment_id cannot be empty")
        exact = self._make_exact_segment(segment_id, text, metadata, extra_anchors)
        evicted = self.recent.push(exact)
        if evicted is not None:
            self.latent.compress_from_exact(evicted)
        return exact

    def _query_anchor_kinds(self, query_tokens: set[str]) -> set[str]:
        kinds: set[str] = set()
        for token in query_tokens:
            kinds.update(_ANCHOR_INTENT.get(token, ()))
        return kinds

    def _score_exact(self, segment: MemorySegment, query_embedding: torch.Tensor, query_tokens: set[str], anchor_kinds: set[str]) -> float:
        score = float(torch.dot(query_embedding, segment.embedding))
        if query_tokens.intersection(segment.token_counts):
            score += 0.10
        if any(anchor.kind in anchor_kinds or anchor.value.lower() in query_tokens for anchor in segment.anchors):
            score += self.config.anchor_boost
        return score

    def _required_anchors(self, selected: Sequence[MemorySegment], query_tokens: set[str], anchor_kinds: set[str], explicit: Sequence[Anchor] | None) -> tuple[Anchor, ...]:
        if explicit is not None:
            return tuple(explicit)
        required: list[Anchor] = []
        for segment in selected:
            for anchor in segment.anchors:
                if anchor.kind in anchor_kinds or anchor.value.lower() in query_tokens:
                    required.append(anchor)
        seen: set[tuple[str, str]] = set()
        out: list[Anchor] = []
        for anchor in required:
            key = (anchor.kind, anchor.value)
            if key not in seen:
                seen.add(key)
                out.append(anchor)
        return tuple(out)

    def reconstruct(self, query: str, required_anchors: Sequence[Anchor] | None = None) -> MemoryReconstruction:
        query_tokens = set(tokenize(query))
        query_embedding = embed_text(query, self.config.embedding_dim)
        anchor_kinds = self._query_anchor_kinds(query_tokens)

        exact_scored = [
            (self._score_exact(segment, query_embedding, query_tokens, anchor_kinds), segment)
            for segment in self.recent.segments
        ]
        exact_segments = [
            segment
            for score, segment in sorted(exact_scored, key=lambda item: item[0], reverse=True)[:self.config.top_k_exact]
            if score > 0.0
        ]
        latent_segments = self.latent.retrieve(query_embedding, query_tokens, anchor_kinds, self.config.top_k_latent)
        selected = exact_segments + [segment for segment in latent_segments if segment.segment_id not in {exact.segment_id for exact in exact_segments}]
        required = self._required_anchors(selected, query_tokens, anchor_kinds, required_anchors)

        anchors: list[Anchor] = []
        for segment in selected:
            anchors.extend(segment.anchors)
        seen: set[tuple[str, str]] = set()
        unique_anchors: list[Anchor] = []
        for anchor in anchors:
            key = (anchor.kind, anchor.value)
            if key not in seen:
                seen.add(key)
                unique_anchors.append(anchor)

        exact_context = tuple(segment.exact_text for segment in exact_segments)
        latent_context = tuple(segment.rendered() for segment in latent_segments)
        rendered = "\n".join(list(exact_context) + list(latent_context) + [anchor.value for anchor in unique_anchors])
        fidelity = self.fidelity.verify(rendered, required)
        cost = CostTrace(
            kv_bytes=sum(len(text.encode("utf-8")) for text in exact_context)
            + sum(len(text.encode("utf-8")) for text in latent_context),
            verifier_steps=1,
        )
        return MemoryReconstruction(
            query=query,
            exact_context=exact_context,
            latent_context=latent_context,
            anchors=tuple(unique_anchors),
            selected_segment_ids=tuple(segment.segment_id for segment in selected),
            fidelity=fidelity,
            cost=cost,
        )

    def compression_report(self) -> dict[str, Any]:
        latent = self.latent.segments
        original = sum(segment.original_token_count for segment in latent)
        stored = sum(segment.stored_token_count for segment in latent)
        return {
            "recent_exact_segments": len(self.recent.segments),
            "latent_segments": len(latent),
            "latent_original_tokens": original,
            "latent_stored_tokens": stored,
            "latent_compression_ratio": stored / max(original, 1),
            "anchors": [asdict(anchor) for anchor in self.anchor_ledger.anchors],
        }
