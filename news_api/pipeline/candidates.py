"""Run domain assignment + sequential span extraction over an article.

This module takes a cleaned article and a configured ``LLMProvider``, runs
windowed candidate extraction, applies span-level dedupe (per the plan's
"owned core sentence ranges" rule), and returns a flat list of unique
candidate spans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from .llm_client import (
    CandidateSpan,
    DomainResponse,
    LLMProvider,
    TokenEstimate,
    WindowResponse,
)
from .segment import LLMWindow, Sentence
from .utils import fingerprint


@dataclass
class WindowCallResult:
    window: LLMWindow
    response: WindowResponse
    owned_spans: list[CandidateSpan] = field(default_factory=list)
    dropped_unowned: int = 0


@dataclass
class ArticleLLMResult:
    """Aggregated output of running the LLM over one article's windows."""

    domain: DomainResponse | None
    windows: list[WindowCallResult] = field(default_factory=list)
    unique_spans: list[CandidateSpan] = field(default_factory=list)
    dedupe_drops: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _filter_owned(window: LLMWindow, response: WindowResponse) -> tuple[list[CandidateSpan], int]:
    """Drop spans whose primary sentence isn't inside the window's owned core."""
    keep: list[CandidateSpan] = []
    dropped = 0
    for span in response.spans:
        if (
            window.owned_sentence_id_start
            <= span.primary_sentence_id
            <= window.owned_sentence_id_end
        ):
            keep.append(span)
        else:
            dropped += 1
    return keep, dropped


def _span_dedupe_key(span: CandidateSpan, canonical_article_id: str) -> str:
    """Stable dedupe key per the plan: ``(article_id, primary_id, span_ids)``."""
    return fingerprint(
        canonical_article_id,
        span.primary_sentence_id,
        ",".join(str(i) for i in sorted(span.span_sentence_ids)),
    )


def run_llm_for_article(
    *,
    provider: LLMProvider,
    article_title: str | None,
    article_description: str | None,
    article_url: str | None,
    article_content: str | None,
    query_domain_hint: str | None,
    sentences: list[Sentence],
    windows: list[LLMWindow],
    canonical_article_id: str,
    skip_domain: bool = False,
    window_status_logger: Callable[[str], None] | None = None,
) -> ArticleLLMResult:
    """Run domain assignment + every window's candidate-extraction call.

    Span-level dedupe is applied across all windows of one article using the
    ``(article_id, primary_sentence_id, span_sentence_ids)`` key. The result
    is the deduped list plus per-window diagnostics.
    """
    domain_resp: DomainResponse | None = None
    if not skip_domain:
        domain_resp = provider.assign_domain(
            article_title=article_title,
            article_description=article_description,
            article_url=article_url,
            article_content=article_content,
            query_domain_hint=query_domain_hint,
        )

    seen: set[str] = set()
    unique: list[CandidateSpan] = []
    win_results: list[WindowCallResult] = []
    drops = 0
    in_tokens = 0
    out_tokens = 0

    total_windows = len(windows)
    for window in windows:
        if window_status_logger is not None:
            window_status_logger(
                f"window {window.window_index + 1}/{total_windows} start "
                f"sentences={len(window.sentences)} "
                f"owned={window.owned_sentence_id_start}-{window.owned_sentence_id_end}"
            )
        est = provider.estimate_window_tokens(window)
        in_tokens += est.input_tokens
        out_tokens += est.output_tokens

        response = provider.extract_candidates(
            window=window,
            article_title=article_title,
            article_description=article_description,
        )
        owned, dropped_unowned = _filter_owned(window, response)
        win_results.append(
            WindowCallResult(
                window=window,
                response=response,
                owned_spans=owned,
                dropped_unowned=dropped_unowned,
            )
        )
        if window_status_logger is not None:
            window_status_logger(
                f"window {window.window_index + 1}/{total_windows} done "
                f"spans={len(response.spans)} owned={len(owned)} "
                f"dropped_unowned={dropped_unowned}"
            )
        for span in owned:
            key = _span_dedupe_key(span, canonical_article_id)
            if key in seen:
                drops += 1
                continue
            seen.add(key)
            unique.append(span)

    return ArticleLLMResult(
        domain=domain_resp,
        windows=win_results,
        unique_spans=unique,
        dedupe_drops=drops,
        total_input_tokens=in_tokens,
        total_output_tokens=out_tokens,
    )


def estimate_article_tokens(
    *,
    provider: LLMProvider,
    windows: Iterable[LLMWindow],
) -> TokenEstimate:
    """Sum estimated tokens across every window for dry-run planning."""
    total_in = 0
    total_out = 0
    for w in windows:
        est = provider.estimate_window_tokens(w)
        total_in += est.input_tokens
        total_out += est.output_tokens
    return TokenEstimate(input_tokens=total_in, output_tokens=total_out)
