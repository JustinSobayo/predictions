"""Sentence segmentation (``pysbd``) and overlapping LLM-window planning.

The segmenter is run once per article, against the combined title +
description + content text. Each sentence is tagged with which of those three
source fields it came from, so the transform step can populate
``Source_Field`` and ``field_order`` for the annotator schema.

For LLM windowing we emit overlapping windows with a configurable target
sentence count and overlap, and we mark each window's "owned core" range. A
window only owns predictions whose ``primary_sentence_id`` falls inside its
core; this prevents adjacent windows from emitting duplicate candidates for
the same primary sentence (per the plan's "owned core sentence ranges"
guidance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pysbd


@dataclass
class Sentence:
    """One segmented sentence with provenance."""

    sentence_id: int
    text: str
    source_field: str  # "title" | "description" | "content"
    field_order: int  # 0/1/2, matches the existing annotator schema


@dataclass
class LLMWindow:
    """One LLM-input window over a contiguous slice of an article's sentences."""

    window_index: int
    sentence_id_start: int
    sentence_id_end: int  # inclusive
    owned_sentence_id_start: int
    owned_sentence_id_end: int  # inclusive
    sentences: list[Sentence] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sentences)


_SOURCE_FIELD_ORDER = {"title": 0, "description": 1, "content": 2}


# pysbd Segmenter is not thread-safe; build lazily per call.
def _new_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="en", clean=False)


def segment_field(text: str | None) -> list[str]:
    """Run pysbd on a single field's text and return non-empty sentences."""
    if not text:
        return []
    seg = _new_segmenter()
    out = []
    for sent in seg.segment(text):
        s = (sent or "").strip()
        if s:
            out.append(s)
    return out


def segment_article(
    *,
    title: str | None,
    description: str | None,
    content: str | None,
) -> list[Sentence]:
    """Return sentences in title -> description -> content order, with ids."""
    sentences: list[Sentence] = []
    sid = 0

    if title:
        for s in segment_field(title):
            sentences.append(
                Sentence(
                    sentence_id=sid,
                    text=s,
                    source_field="title",
                    field_order=_SOURCE_FIELD_ORDER["title"],
                )
            )
            sid += 1

    if description:
        for s in segment_field(description):
            sentences.append(
                Sentence(
                    sentence_id=sid,
                    text=s,
                    source_field="description",
                    field_order=_SOURCE_FIELD_ORDER["description"],
                )
            )
            sid += 1

    if content:
        for s in segment_field(content):
            sentences.append(
                Sentence(
                    sentence_id=sid,
                    text=s,
                    source_field="content",
                    field_order=_SOURCE_FIELD_ORDER["content"],
                )
            )
            sid += 1

    return sentences


def plan_windows(
    sentences: list[Sentence],
    *,
    target_sentences: int = 40,
    hard_max_sentences: int = 60,
    overlap: int = 5,
) -> list[LLMWindow]:
    """Cut a sentence list into overlapping windows with contiguous owned cores.

    Each window contains up to ``target_sentences`` sentences (capped at
    ``hard_max_sentences``). Adjacent windows overlap by ``overlap`` sentences
    on each side as context, but their *owned* cores partition the article
    exactly: every sentence is owned by exactly one window. Predictions whose
    ``primary_sentence_id`` lies outside the owned range are dropped during
    span-level dedupe.
    """
    if not sentences:
        return []
    n = len(sentences)
    target = max(1, min(target_sentences, hard_max_sentences))
    overlap = max(0, min(overlap, target - 1)) if target > 1 else 0

    if n <= target:
        return [
            LLMWindow(
                window_index=0,
                sentence_id_start=sentences[0].sentence_id,
                sentence_id_end=sentences[-1].sentence_id,
                owned_sentence_id_start=sentences[0].sentence_id,
                owned_sentence_id_end=sentences[-1].sentence_id,
                sentences=list(sentences),
            )
        ]

    # Step = how many *new* sentences the owned core advances per window.
    step = max(1, target - overlap)

    # Build contiguous owned-core index ranges first; absorb a tiny tail into
    # the previous window so the last window isn't a one-sentence stub.
    cores: list[tuple[int, int]] = []
    core_start = 0
    while core_start < n:
        core_end = min(core_start + step - 1, n - 1)
        cores.append((core_start, core_end))
        core_start = core_end + 1
    if len(cores) > 1 and cores[-1][1] - cores[-1][0] + 1 < max(1, step // 4):
        prev_start, _ = cores[-2]
        last_start, last_end = cores.pop()
        cores[-1] = (prev_start, last_end)

    windows: list[LLMWindow] = []
    for wnum, (cstart, cend) in enumerate(cores):
        win_start = max(0, cstart - overlap)
        win_end = min(n - 1, cend + overlap)
        # Cap at hard_max to be safe.
        if win_end - win_start + 1 > hard_max_sentences:
            win_end = win_start + hard_max_sentences - 1
        chunk = sentences[win_start : win_end + 1]
        windows.append(
            LLMWindow(
                window_index=wnum,
                sentence_id_start=chunk[0].sentence_id,
                sentence_id_end=chunk[-1].sentence_id,
                owned_sentence_id_start=sentences[cstart].sentence_id,
                owned_sentence_id_end=sentences[cend].sentence_id,
                sentences=list(chunk),
            )
        )
    return windows
