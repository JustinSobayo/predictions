"""Deterministic same-article dedupe for prediction candidates.

This module keeps the human-annotation queue clean without adding extra model
calls. It operates only on the final ``potential_predictions`` rows and leaves
``processed_sentences`` untouched so the raw extracted sentence universe stays
auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .clean import sanitize_output_sentence_text


_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CandidateDedupeConfig:
    containment_ratio_threshold: float = 0.9
    similarity_threshold: float = 0.93
    dedupe_version: str = "same_article_text_v1"


@dataclass
class CandidateDedupeReport:
    rows: list[dict[str, Any]]
    exact_duplicates_removed: int = 0
    containment_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    total_before: int = 0
    total_after: int = 0
    config: CandidateDedupeConfig = CandidateDedupeConfig()

    def as_json(self) -> dict[str, Any]:
        return {
            "dedupe_version": self.config.dedupe_version,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "containment_duplicates_removed": self.containment_duplicates_removed,
            "near_duplicates_removed": self.near_duplicates_removed,
            "total_before": self.total_before,
            "total_after": self.total_after,
            "similarity_threshold": self.config.similarity_threshold,
            "containment_ratio_threshold": self.config.containment_ratio_threshold,
        }


def _normalize_candidate_text(text: Any) -> str:
    cleaned = sanitize_output_sentence_text(str(text or ""))
    cleaned = cleaned.casefold()
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _candidate_priority(row: dict[str, Any], original_index: int) -> tuple[int, int, int, int, int]:
    span_ids = row.get("span_sentence_ids") or []
    span_width = len(span_ids) if isinstance(span_ids, list) else 0
    raw_len = len((row.get("Base Sentence (raw)") or "").strip())
    has_reason = 1 if str(row.get("candidate_reason") or "").strip() else 0
    context_needed = 1 if bool(row.get("context_needed")) else 0
    return (context_needed, span_width, has_reason, raw_len, -original_index)


def _containment_ratio(a: str, b: str) -> float:
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if not shorter or not longer or shorter not in longer:
        return 0.0
    return len(shorter) / max(1, len(longer))


def _token_sequence_contained(a: str, b: str) -> bool:
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    short_tokens = shorter.split()
    long_tokens = longer.split()
    if len(short_tokens) < 5 or len(short_tokens) > len(long_tokens):
        return False
    window = len(short_tokens)
    for start in range(len(long_tokens) - window + 1):
        if long_tokens[start : start + window] == short_tokens:
            return True
    return False


def _duplicate_kind(
    a_norm: str,
    b_norm: str,
    *,
    config: CandidateDedupeConfig,
) -> str | None:
    if not a_norm or not b_norm:
        return None
    if a_norm == b_norm:
        return "exact"
    if (
        _containment_ratio(a_norm, b_norm) >= config.containment_ratio_threshold
        or _token_sequence_contained(a_norm, b_norm)
    ):
        return "containment"
    if SequenceMatcher(None, a_norm, b_norm).ratio() >= config.similarity_threshold:
        return "near"
    return None


def dedupe_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    config: CandidateDedupeConfig | None = None,
) -> CandidateDedupeReport:
    """Drop deterministic same-article duplicate candidates.

    Only rows with the same ``article_id`` are compared, which keeps the logic
    conservative and avoids collapsing legitimate repeated predictions across
    different publishers.
    """

    cfg = config or CandidateDedupeConfig()
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        article_id = row.get("article_id")
        if not isinstance(article_id, int):
            article_id = -1
        grouped.setdefault(article_id, []).append((idx, row))

    survivors: list[tuple[int, dict[str, Any]]] = []
    exact_removed = 0
    containment_removed = 0
    near_removed = 0

    for article_rows in grouped.values():
        article_survivors: list[tuple[int, dict[str, Any], str]] = []
        for original_index, row in article_rows:
            current_norm = _normalize_candidate_text(row.get("Base Sentence"))
            match_index: int | None = None
            match_kind: str | None = None

            for s_idx, (_, survivor_row, survivor_norm) in enumerate(article_survivors):
                duplicate_kind = _duplicate_kind(
                    current_norm,
                    survivor_norm,
                    config=cfg,
                )
                if duplicate_kind is not None:
                    match_index = s_idx
                    match_kind = duplicate_kind
                    break

            if match_index is None:
                article_survivors.append((original_index, row, current_norm))
                continue

            survivor_original_index, survivor_row, _ = article_survivors[match_index]
            current_priority = _candidate_priority(row, original_index)
            survivor_priority = _candidate_priority(survivor_row, survivor_original_index)

            if current_priority > survivor_priority:
                article_survivors[match_index] = (original_index, row, current_norm)

            if match_kind == "exact":
                exact_removed += 1
            elif match_kind == "containment":
                containment_removed += 1
            else:
                near_removed += 1

        survivors.extend((original_index, row) for original_index, row, _ in article_survivors)

    survivors.sort(key=lambda item: item[0])
    deduped_rows = [row for _, row in survivors]

    return CandidateDedupeReport(
        rows=deduped_rows,
        exact_duplicates_removed=exact_removed,
        containment_duplicates_removed=containment_removed,
        near_duplicates_removed=near_removed,
        total_before=len(rows),
        total_after=len(deduped_rows),
        config=cfg,
    )
