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
    DROP = "drop"


def _coerce_memory_mode(value: Any, *, default: MemoryMode | None = None) -> MemoryMode | None:
    try:
        return MemoryMode(str(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MemoryRetentionDecision:
    segment_id: str
    requested_mode: MemoryMode
    applied_mode: MemoryMode | None
    exact_prob: float = 0.0
    latent_prob: float = 0.0
    drop_prob: float = 0.0
    storage_ratio: float = 1.0
    confidence: float = 1.0
    source: str = "manual"
    reason: str = ""
    anchor_count: int = 0
    anchor_safety_override: bool = False
    stored: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "requested_mode": self.requested_mode.value,
            "applied_mode": self.applied_mode.value if self.applied_mode is not None else None,
            "exact_prob": float(self.exact_prob),
            "latent_prob": float(self.latent_prob),
            "drop_prob": float(self.drop_prob),
            "storage_ratio": float(self.storage_ratio),
            "confidence": float(self.confidence),
            "source": self.source,
            "reason": self.reason,
            "anchor_count": int(self.anchor_count),
            "anchor_safety_override": bool(self.anchor_safety_override),
            "stored": bool(self.stored),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "MemoryRetentionDecision":
        requested = _coerce_memory_mode(payload.get("requested_mode"), default=MemoryMode.EXACT) or MemoryMode.EXACT
        applied = _coerce_memory_mode(payload.get("applied_mode"), default=None)
        return MemoryRetentionDecision(
            segment_id=str(payload.get("segment_id", "")),
            requested_mode=requested,
            applied_mode=applied,
            exact_prob=float(payload.get("exact_prob", 0.0)),
            latent_prob=float(payload.get("latent_prob", 0.0)),
            drop_prob=float(payload.get("drop_prob", 0.0)),
            storage_ratio=float(payload.get("storage_ratio", 1.0)),
            confidence=float(payload.get("confidence", 1.0)),
            source=str(payload.get("source", "manual")),
            reason=str(payload.get("reason", "")),
            anchor_count=int(payload.get("anchor_count", 0)),
            anchor_safety_override=bool(payload.get("anchor_safety_override", False)),
            stored=bool(payload.get("stored", applied is not None)),
        )


@dataclass(frozen=True)
class MemoryUtilityCredit:
    segment_id: str
    source: str
    phase: str
    query: str
    selected: bool
    fidelity_passed: bool
    fidelity_score: float
    required_anchor_count: int
    utility: float
    requested_mode: MemoryMode | None = None
    applied_mode: MemoryMode | None = None
    retention_source: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source": self.source,
            "phase": self.phase,
            "query": self.query,
            "selected": bool(self.selected),
            "fidelity_passed": bool(self.fidelity_passed),
            "fidelity_score": float(self.fidelity_score),
            "required_anchor_count": int(self.required_anchor_count),
            "utility": float(self.utility),
            "requested_mode": self.requested_mode.value if self.requested_mode is not None else None,
            "applied_mode": self.applied_mode.value if self.applied_mode is not None else None,
            "retention_source": self.retention_source,
            "reason": self.reason,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "MemoryUtilityCredit":
        return MemoryUtilityCredit(
            segment_id=str(payload.get("segment_id", "")),
            source=str(payload.get("source", "")),
            phase=str(payload.get("phase", "")),
            query=str(payload.get("query", "")),
            selected=bool(payload.get("selected", True)),
            fidelity_passed=bool(payload.get("fidelity_passed", False)),
            fidelity_score=float(payload.get("fidelity_score", 0.0)),
            required_anchor_count=int(payload.get("required_anchor_count", 0)),
            utility=float(payload.get("utility", 0.0)),
            requested_mode=_coerce_memory_mode(payload.get("requested_mode"), default=None),
            applied_mode=_coerce_memory_mode(payload.get("applied_mode"), default=None),
            retention_source=str(payload.get("retention_source", "")),
            reason=str(payload.get("reason", "")),
        )


@dataclass(frozen=True)
class CognitiveMemoryConfig:
    recent_exact_limit: int = 2
    embedding_dim: int = 64
    top_k_exact: int = 2
    top_k_latent: int = 3
    max_summary_terms: int = 16
    anchor_boost: float = 0.40
    utility_score_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.recent_exact_limit < 1:
            raise ValueError("recent_exact_limit must be at least 1")
        if self.embedding_dim < 8:
            raise ValueError("embedding_dim must be at least 8")
        if self.top_k_exact < 0 or self.top_k_latent < 0:
            raise ValueError("top_k values cannot be negative")
        if self.max_summary_terms < 1:
            raise ValueError("max_summary_terms must be at least 1")
        if self.utility_score_weight < 0.0:
            raise ValueError("utility_score_weight cannot be negative")


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


@dataclass(frozen=True)
class CompiledCircuitMemoryBinding:
    binding_id: str
    circuit_id: str
    skill: str
    source_kind: str
    segment_id: str
    source_failure_ids: tuple[str, ...]
    frontier_task_ids: tuple[str, ...]
    heldout_task_ids: tuple[str, ...]
    prompt_obligations: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    anchor_values: tuple[str, ...]
    fidelity: AnchorFidelityResult
    selected_segment_ids: tuple[str, ...]
    cost: CostTrace

    @property
    def passed(self) -> bool:
        return self.fidelity.passed and self.segment_id in self.selected_segment_ids

    @property
    def required_anchors(self) -> tuple[Anchor, ...]:
        source = self.segment_id
        anchors = [
            Anchor("circuit", self.circuit_id, source, 1.0),
            Anchor("skill", self.skill, source, 1.0),
        ]
        if self.source_kind:
            anchors.append(Anchor("source_kind", self.source_kind, source, 1.0))
        return tuple(anchors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "circuit_id": self.circuit_id,
            "skill": self.skill,
            "source_kind": self.source_kind,
            "segment_id": self.segment_id,
            "source_failure_ids": list(self.source_failure_ids),
            "frontier_task_ids": list(self.frontier_task_ids),
            "heldout_task_ids": list(self.heldout_task_ids),
            "prompt_obligations": list(self.prompt_obligations),
            "metadata_keys": list(self.metadata_keys),
            "anchor_values": list(self.anchor_values),
            "fidelity": {
                "required": self.fidelity.required,
                "preserved": self.fidelity.preserved,
                "missing": [asdict(anchor) for anchor in self.fidelity.missing],
                "score": self.fidelity.score,
            },
            "selected_segment_ids": list(self.selected_segment_ids),
            "cost": asdict(self.cost),
            "passed": self.passed,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "CompiledCircuitMemoryBinding":
        fidelity_payload = dict(payload.get("fidelity") or {})
        cost_payload = dict(payload.get("cost") or {})
        return CompiledCircuitMemoryBinding(
            binding_id=str(payload.get("binding_id", "")),
            circuit_id=str(payload.get("circuit_id", "")),
            skill=str(payload.get("skill", "")),
            source_kind=str(payload.get("source_kind", "")),
            segment_id=str(payload.get("segment_id", "")),
            source_failure_ids=tuple(str(item) for item in payload.get("source_failure_ids", ())),
            frontier_task_ids=tuple(str(item) for item in payload.get("frontier_task_ids", ())),
            heldout_task_ids=tuple(str(item) for item in payload.get("heldout_task_ids", ())),
            prompt_obligations=tuple(str(item) for item in payload.get("prompt_obligations", ())),
            metadata_keys=tuple(str(item) for item in payload.get("metadata_keys", ())),
            anchor_values=tuple(str(item) for item in payload.get("anchor_values", ())),
            fidelity=AnchorFidelityResult(
                required=int(fidelity_payload.get("required", 0)),
                preserved=int(fidelity_payload.get("preserved", 0)),
                missing=tuple(
                    Anchor(
                        kind=str(anchor.get("kind", "")),
                        value=str(anchor.get("value", "")),
                        source_id=str(anchor.get("source_id", "")),
                        importance=float(anchor.get("importance", 1.0)),
                    )
                    for anchor in fidelity_payload.get("missing", ())
                ),
                score=float(fidelity_payload.get("score", 0.0)),
            ),
            selected_segment_ids=tuple(str(item) for item in payload.get("selected_segment_ids", ())),
            cost=CostTrace(
                weight_bits_read=float(cost_payload.get("weight_bits_read", 0.0)),
                activation_bits=float(cost_payload.get("activation_bits", 0.0)),
                kv_bytes=float(cost_payload.get("kv_bytes", 0.0)),
                generated_tokens=int(cost_payload.get("generated_tokens", 0)),
                latent_steps=int(cost_payload.get("latent_steps", 0)),
                experts_activated=int(cost_payload.get("experts_activated", 0)),
                verifier_steps=int(cost_payload.get("verifier_steps", 0)),
                wall_time_ms=float(cost_payload.get("wall_time_ms", 0.0)),
            ),
        )


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

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        query_tokens: set[str],
        anchor_kinds: set[str],
        top_k: int,
        score_bias_by_segment: Mapping[str, float] | None = None,
    ) -> list[MemorySegment]:
        score_bias_by_segment = score_bias_by_segment or {}
        scored = []
        for segment in self.segments:
            score = float(torch.dot(query_embedding, segment.embedding))
            if query_tokens.intersection(segment.token_counts):
                score += 0.10
            if any(anchor.kind in anchor_kinds or anchor.value.lower() in query_tokens for anchor in segment.anchors):
                score += self.config.anchor_boost
            score += float(score_bias_by_segment.get(segment.segment_id, 0.0))
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
        self.compiled_circuit_bindings: dict[str, CompiledCircuitMemoryBinding] = {}
        self.retention_decisions: list[MemoryRetentionDecision] = []
        self.utility_credits: list[MemoryUtilityCredit] = []

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

    def _normalize_retention_decision(
        self,
        decision: MemoryRetentionDecision | Mapping[str, Any] | None,
        exact: MemorySegment,
    ) -> MemoryRetentionDecision | None:
        if decision is None:
            return None
        if isinstance(decision, MemoryRetentionDecision):
            payload = decision
        else:
            payload = MemoryRetentionDecision.from_dict(decision)
        requested = payload.requested_mode
        applied: MemoryMode | None = requested if requested != MemoryMode.DROP else None
        stored = requested != MemoryMode.DROP
        override = False
        reason = payload.reason
        if requested == MemoryMode.DROP and exact.anchors:
            applied = MemoryMode.EXACT
            stored = True
            override = True
            suffix = "anchor_safety_promoted_drop_to_exact"
            reason = f"{reason};{suffix}" if reason else suffix
        return MemoryRetentionDecision(
            segment_id=exact.segment_id,
            requested_mode=requested,
            applied_mode=applied,
            exact_prob=payload.exact_prob,
            latent_prob=payload.latent_prob,
            drop_prob=payload.drop_prob,
            storage_ratio=payload.storage_ratio,
            confidence=payload.confidence,
            source=payload.source,
            reason=reason,
            anchor_count=len(exact.anchors),
            anchor_safety_override=payload.anchor_safety_override or override,
            stored=stored,
        )

    def _record_retention_decision(self, decision: MemoryRetentionDecision | None) -> None:
        if decision is not None:
            self.retention_decisions.append(decision)

    def _compiled_circuit_retention_decision(
        self,
        *,
        segment_id: str,
        anchor_count: int,
    ) -> MemoryRetentionDecision:
        return MemoryRetentionDecision(
            segment_id=segment_id,
            requested_mode=MemoryMode.LATENT,
            applied_mode=MemoryMode.LATENT,
            exact_prob=0.28,
            latent_prob=0.66,
            drop_prob=0.06,
            storage_ratio=0.35,
            confidence=0.66,
            source="learned_memory_compiled_circuit_policy",
            reason="compiled_circuit_anchor_preserving_latent_retention",
            anchor_count=max(0, int(anchor_count)),
            stored=True,
        )

    def ingest(
        self,
        segment_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
        extra_anchors: Iterable[Anchor] = (),
        retention_decision: MemoryRetentionDecision | Mapping[str, Any] | None = None,
    ) -> MemorySegment | None:
        if not segment_id:
            raise ValueError("segment_id cannot be empty")
        exact = self._make_exact_segment(segment_id, text, metadata, extra_anchors)
        decision = self._normalize_retention_decision(retention_decision, exact)
        if decision is not None and decision.applied_mode is None:
            self._record_retention_decision(decision)
            return None
        if decision is not None and decision.applied_mode == MemoryMode.LATENT:
            latent = self.latent.compress_from_exact(exact)
            self._record_retention_decision(decision)
            return latent
        evicted = self.recent.push(exact)
        if evicted is not None:
            self.latent.compress_from_exact(evicted)
        self._record_retention_decision(decision)
        return exact

    def _query_anchor_kinds(self, query_tokens: set[str]) -> set[str]:
        kinds: set[str] = set()
        for token in query_tokens:
            kinds.update(_ANCHOR_INTENT.get(token, ()))
        return kinds

    def _learned_utility_score_biases(self) -> dict[str, float]:
        if not self.utility_credits or self.config.utility_score_weight <= 0.0:
            return {}
        by_segment: dict[str, list[float]] = {}
        for credit in self.utility_credits:
            if not credit.retention_source.startswith("learned_memory"):
                continue
            by_segment.setdefault(credit.segment_id, []).append(float(credit.utility))
        biases: dict[str, float] = {}
        for segment_id, utilities in by_segment.items():
            if not utilities:
                continue
            # Recent credits matter more, but old credits still contribute.
            tail = utilities[-8:]
            mean_utility = sum(tail) / float(len(tail))
            clipped = max(-1.0, min(1.0, mean_utility))
            biases[segment_id] = clipped * self.config.utility_score_weight
        return biases

    def _score_exact(
        self,
        segment: MemorySegment,
        query_embedding: torch.Tensor,
        query_tokens: set[str],
        anchor_kinds: set[str],
        score_bias_by_segment: Mapping[str, float] | None = None,
    ) -> float:
        score = float(torch.dot(query_embedding, segment.embedding))
        if query_tokens.intersection(segment.token_counts):
            score += 0.10
        if any(anchor.kind in anchor_kinds or anchor.value.lower() in query_tokens for anchor in segment.anchors):
            score += self.config.anchor_boost
        if score_bias_by_segment is not None:
            score += float(score_bias_by_segment.get(segment.segment_id, 0.0))
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
        score_bias_by_segment = self._learned_utility_score_biases()

        exact_scored = [
            (self._score_exact(segment, query_embedding, query_tokens, anchor_kinds, score_bias_by_segment), segment)
            for segment in self.recent.segments
        ]
        exact_segments = [
            segment
            for score, segment in sorted(exact_scored, key=lambda item: item[0], reverse=True)[:self.config.top_k_exact]
            if score > 0.0
        ]
        latent_segments = self.latent.retrieve(query_embedding, query_tokens, anchor_kinds, self.config.top_k_latent, score_bias_by_segment)
        if required_anchors is not None:
            required_keys = {(anchor.kind, anchor.value) for anchor in required_anchors}

            def has_required_anchor(segment: MemorySegment) -> bool:
                return any((anchor.kind, anchor.value) in required_keys for anchor in segment.anchors)

            exact_ids = {segment.segment_id for segment in exact_segments}
            for segment in self.recent.segments:
                if segment.segment_id not in exact_ids and has_required_anchor(segment):
                    exact_segments.append(segment)
                    exact_ids.add(segment.segment_id)
            latent_ids = {segment.segment_id for segment in latent_segments}
            for segment in self.latent.segments:
                if segment.segment_id not in latent_ids and has_required_anchor(segment):
                    latent_segments.append(segment)
                    latent_ids.add(segment.segment_id)
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

    def bind_compiled_circuit(
        self,
        *,
        circuit_id: str,
        skill: str,
        source_kind: str,
        source_failure_ids: Sequence[str] = (),
        frontier_task_ids: Sequence[str] = (),
        heldout_task_ids: Sequence[str] = (),
        prompt_obligations: Sequence[str] = (),
        metadata_keys: Sequence[str] = (),
        anchors: Iterable[Anchor] = (),
    ) -> CompiledCircuitMemoryBinding:
        if not circuit_id:
            raise ValueError("compiled circuit memory binding requires a circuit_id")
        if not skill:
            raise ValueError("compiled circuit memory binding requires a skill")
        existing = self.compiled_circuit_bindings.get(circuit_id)
        if existing is not None:
            try:
                _, reconstruction = self.reconstruct_compiled_circuit_binding(circuit_id)
                if reconstruction.fidelity.passed and existing.segment_id in reconstruction.selected_segment_ids:
                    return existing
            except KeyError:
                pass

        segment_id = f"compiled-circuit-{circuit_id[:16]}"
        required = (
            Anchor("circuit", circuit_id, segment_id, 1.0),
            Anchor("skill", skill, segment_id, 1.0),
            Anchor("source_kind", source_kind or "frontier_discovery", segment_id, 1.0),
        )
        source_failure_ids = tuple(str(item) for item in source_failure_ids)
        frontier_task_ids = tuple(str(item) for item in frontier_task_ids)
        heldout_task_ids = tuple(str(item) for item in heldout_task_ids)
        prompt_obligations = tuple(str(item) for item in prompt_obligations)
        metadata_keys = tuple(str(item) for item in metadata_keys)
        extra_anchor_tuple = tuple(anchors)
        text = "\n".join(
            (
                f"Compiled circuit id: {circuit_id}",
                f"Skill: {skill}",
                f"Source kind: {source_kind or 'frontier_discovery'}",
                "Source failures: " + " ".join(source_failure_ids),
                "Frontier tasks: " + " ".join(frontier_task_ids),
                "Held-out tasks: " + " ".join(heldout_task_ids),
                "Prompt obligations: " + " ".join(prompt_obligations),
                "Metadata keys: " + " ".join(metadata_keys),
                "Anchors: " + " ".join(anchor.value for anchor in extra_anchor_tuple),
            )
        )
        retention_decision = self._compiled_circuit_retention_decision(
            segment_id=segment_id,
            anchor_count=len(tuple(required) + extra_anchor_tuple),
        )
        segment = self.ingest(
            segment_id,
            text,
            metadata={
                "source": "compiled_circuit_memory_binding",
                "circuit_id": circuit_id,
                "skill": skill,
                "source_kind": source_kind or "frontier_discovery",
                "frontier_task_ids": frontier_task_ids,
                "heldout_task_ids": heldout_task_ids,
            },
            extra_anchors=tuple(required) + extra_anchor_tuple,
            retention_decision=retention_decision,
        )
        if segment is None:
            raise ValueError(f"compiled circuit memory binding unexpectedly dropped {circuit_id}")
        reconstruction = self.reconstruct(
            f"compiled circuit {circuit_id} skill {skill} {source_kind}",
            required_anchors=required,
        )
        binding = CompiledCircuitMemoryBinding(
            binding_id=f"memory-binding-{circuit_id}",
            circuit_id=circuit_id,
            skill=skill,
            source_kind=source_kind or "frontier_discovery",
            segment_id=segment.segment_id,
            source_failure_ids=source_failure_ids,
            frontier_task_ids=frontier_task_ids,
            heldout_task_ids=heldout_task_ids,
            prompt_obligations=prompt_obligations,
            metadata_keys=metadata_keys,
            anchor_values=tuple(anchor.value for anchor in tuple(required) + extra_anchor_tuple),
            fidelity=reconstruction.fidelity,
            selected_segment_ids=reconstruction.selected_segment_ids,
            cost=reconstruction.cost,
        )
        if not binding.passed:
            missing = ", ".join(anchor.value for anchor in reconstruction.fidelity.missing)
            raise ValueError(f"compiled circuit memory binding failed fidelity for {circuit_id}: missing {missing}")
        self.compiled_circuit_bindings[circuit_id] = binding
        return binding

    def reconstruct_compiled_circuit_binding(
        self,
        circuit_id: str,
        *,
        query: str = "",
    ) -> tuple[CompiledCircuitMemoryBinding, MemoryReconstruction]:
        binding = self.compiled_circuit_bindings.get(circuit_id)
        if binding is None:
            raise KeyError(f"compiled circuit memory binding not found for {circuit_id}")
        reconstruction = self.reconstruct(
            query or f"compiled circuit {binding.circuit_id} skill {binding.skill} {binding.source_kind}",
            required_anchors=binding.required_anchors,
        )
        return binding, reconstruction

    def record_utility(
        self,
        reconstruction: MemoryReconstruction,
        *,
        phase: str,
        source: str,
        utility: float | None = None,
        reason: str = "",
    ) -> tuple[MemoryUtilityCredit, ...]:
        decision_by_segment: dict[str, MemoryRetentionDecision] = {}
        for decision in self.retention_decisions:
            decision_by_segment[decision.segment_id] = decision
        selected_ids = tuple(str(segment_id) for segment_id in reconstruction.selected_segment_ids)
        if not selected_ids:
            return ()
        selected_id_set = set(selected_ids)
        base_utility = (
            float(utility)
            if utility is not None
            else (float(reconstruction.fidelity.score) if reconstruction.fidelity.passed else -1.0)
        )
        credits: list[MemoryUtilityCredit] = []
        for segment_id in selected_ids:
            decision = decision_by_segment.get(segment_id)
            credit = MemoryUtilityCredit(
                segment_id=segment_id,
                source=str(source),
                phase=str(phase),
                query=reconstruction.query,
                selected=True,
                fidelity_passed=bool(reconstruction.fidelity.passed),
                fidelity_score=float(reconstruction.fidelity.score),
                required_anchor_count=int(reconstruction.fidelity.required),
                utility=base_utility,
                requested_mode=decision.requested_mode if decision is not None else None,
                applied_mode=decision.applied_mode if decision is not None else None,
                retention_source=decision.source if decision is not None else "",
                reason=str(reason),
            )
            credits.append(credit)
        if reconstruction.fidelity.passed:
            current_segment_ids = {
                segment.segment_id
                for segment in tuple(self.recent.segments) + tuple(self.latent.segments)
            }
            for segment_id, decision in decision_by_segment.items():
                if not decision.source.startswith("learned_memory"):
                    continue
                if decision.applied_mode is None:
                    continue
                if segment_id in selected_id_set or segment_id not in current_segment_ids:
                    continue
                if decision.applied_mode == MemoryMode.EXACT:
                    unselected_utility = -0.12
                else:
                    unselected_utility = -0.06
                credits.append(
                    MemoryUtilityCredit(
                        segment_id=segment_id,
                        source=str(source),
                        phase=str(phase),
                        query=reconstruction.query,
                        selected=False,
                        fidelity_passed=True,
                        fidelity_score=float(reconstruction.fidelity.score),
                        required_anchor_count=int(reconstruction.fidelity.required),
                        utility=unselected_utility,
                        requested_mode=decision.requested_mode,
                        applied_mode=decision.applied_mode,
                        retention_source=decision.source,
                        reason=f"{reason};unselected_retained" if reason else "unselected_retained",
                    )
                )
        self.utility_credits.extend(credits)
        return tuple(credits)

    def utility_report(self) -> dict[str, Any]:
        learned_credits = [
            credit
            for credit in self.utility_credits
            if credit.retention_source.startswith("learned_memory")
        ]
        compiled_credits = [
            credit
            for credit in learned_credits
            if credit.retention_source.startswith("learned_memory_compiled_circuit")
        ]

        def mode_count(mode: MemoryMode | None, *, learned: bool = False) -> int:
            credits = learned_credits if learned else self.utility_credits
            return sum(1 for credit in credits if credit.applied_mode == mode)

        def compiled_mode_count(mode: MemoryMode | None) -> int:
            return sum(1 for credit in compiled_credits if credit.applied_mode == mode)

        def mean_utility(credits: Sequence[MemoryUtilityCredit]) -> float:
            if not credits:
                return 0.0
            return sum(float(credit.utility) for credit in credits) / float(len(credits))

        return {
            "memory_utility_credit_count": len(self.utility_credits),
            "memory_utility_positive_count": sum(1 for credit in self.utility_credits if credit.utility > 0.0),
            "memory_utility_negative_count": sum(1 for credit in self.utility_credits if credit.utility <= 0.0),
            "memory_utility_selected_count": sum(1 for credit in self.utility_credits if credit.selected),
            "memory_utility_unselected_count": sum(1 for credit in self.utility_credits if not credit.selected),
            "memory_utility_mean": mean_utility(self.utility_credits),
            "memory_utility_exact_count": mode_count(MemoryMode.EXACT),
            "memory_utility_latent_count": mode_count(MemoryMode.LATENT),
            "memory_utility_drop_count": mode_count(None),
            "learned_memory_utility_credit_count": len(learned_credits),
            "learned_memory_utility_positive_count": sum(1 for credit in learned_credits if credit.utility > 0.0),
            "learned_memory_utility_negative_count": sum(1 for credit in learned_credits if credit.utility <= 0.0),
            "learned_memory_utility_selected_count": sum(1 for credit in learned_credits if credit.selected),
            "learned_memory_utility_unselected_count": sum(1 for credit in learned_credits if not credit.selected),
            "learned_memory_utility_mean": mean_utility(learned_credits),
            "learned_memory_utility_exact_count": mode_count(MemoryMode.EXACT, learned=True),
            "learned_memory_utility_latent_count": mode_count(MemoryMode.LATENT, learned=True),
            "learned_memory_utility_drop_count": mode_count(None, learned=True),
            "learned_memory_utility_credits": [credit.to_dict() for credit in learned_credits[-16:]],
            "memory_utility_credits": [credit.to_dict() for credit in self.utility_credits[-16:]],
            "compiled_circuit_memory_utility_credit_count": len(compiled_credits),
            "compiled_circuit_memory_utility_positive_count": sum(1 for credit in compiled_credits if credit.utility > 0.0),
            "compiled_circuit_memory_utility_negative_count": sum(1 for credit in compiled_credits if credit.utility <= 0.0),
            "compiled_circuit_memory_utility_selected_count": sum(1 for credit in compiled_credits if credit.selected),
            "compiled_circuit_memory_utility_unselected_count": sum(1 for credit in compiled_credits if not credit.selected),
            "compiled_circuit_memory_utility_mean": mean_utility(compiled_credits),
            "compiled_circuit_memory_utility_exact_count": compiled_mode_count(MemoryMode.EXACT),
            "compiled_circuit_memory_utility_latent_count": compiled_mode_count(MemoryMode.LATENT),
            "compiled_circuit_memory_utility_drop_count": compiled_mode_count(None),
            "compiled_circuit_memory_utility_credits": [credit.to_dict() for credit in compiled_credits[-16:]],
        }

    def compression_report(self) -> dict[str, Any]:
        latent = self.latent.segments
        original = sum(segment.original_token_count for segment in latent)
        stored = sum(segment.stored_token_count for segment in latent)
        learned_decisions = [
            decision
            for decision in self.retention_decisions
            if decision.source.startswith("learned_memory")
        ]
        compiled_decisions = [
            decision
            for decision in learned_decisions
            if decision.source.startswith("learned_memory_compiled_circuit")
        ]

        def decision_count(mode: MemoryMode | None, *, applied: bool, compiled: bool = False) -> int:
            decisions = compiled_decisions if compiled else learned_decisions
            if applied:
                return sum(1 for decision in decisions if decision.applied_mode == mode)
            return sum(1 for decision in decisions if decision.requested_mode == mode)

        return {
            "recent_exact_segments": len(self.recent.segments),
            "latent_segments": len(latent),
            "latent_original_tokens": original,
            "latent_stored_tokens": stored,
            "latent_compression_ratio": stored / max(original, 1),
            "retention_decision_count": len(self.retention_decisions),
            "learned_retention_decision_count": len(learned_decisions),
            "learned_retention_requested_exact": decision_count(MemoryMode.EXACT, applied=False),
            "learned_retention_requested_latent": decision_count(MemoryMode.LATENT, applied=False),
            "learned_retention_requested_drop": decision_count(MemoryMode.DROP, applied=False),
            "learned_retention_applied_exact": decision_count(MemoryMode.EXACT, applied=True),
            "learned_retention_applied_latent": decision_count(MemoryMode.LATENT, applied=True),
            "learned_retention_applied_drop": sum(1 for decision in learned_decisions if decision.applied_mode is None),
            "learned_retention_anchor_overrides": sum(1 for decision in learned_decisions if decision.anchor_safety_override),
            "learned_retention_decisions": [decision.to_dict() for decision in learned_decisions[-16:]],
            "compiled_circuit_learned_retention_count": len(compiled_decisions),
            "compiled_circuit_learned_retention_requested_exact": decision_count(MemoryMode.EXACT, applied=False, compiled=True),
            "compiled_circuit_learned_retention_requested_latent": decision_count(MemoryMode.LATENT, applied=False, compiled=True),
            "compiled_circuit_learned_retention_requested_drop": decision_count(MemoryMode.DROP, applied=False, compiled=True),
            "compiled_circuit_learned_retention_applied_exact": decision_count(MemoryMode.EXACT, applied=True, compiled=True),
            "compiled_circuit_learned_retention_applied_latent": decision_count(MemoryMode.LATENT, applied=True, compiled=True),
            "compiled_circuit_learned_retention_applied_drop": sum(1 for decision in compiled_decisions if decision.applied_mode is None),
            "compiled_circuit_learned_retention_decisions": [decision.to_dict() for decision in compiled_decisions[-16:]],
            "anchors": [asdict(anchor) for anchor in self.anchor_ledger.anchors],
            "compiled_circuit_memory_bindings": [
                binding.to_dict()
                for binding in self.compiled_circuit_bindings.values()
            ],
            "compiled_circuit_memory_binding_count": len(self.compiled_circuit_bindings),
            **self.utility_report(),
        }
