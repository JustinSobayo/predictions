"""Build the JSON + predictions-CSV row dicts that match the annotator schema.

The shape is dictated by the existing ``annotators/*_full-v*.json`` and
``annotators/*_predictions-v*.csv`` files. The CSV is always derived from the
JSON's ``potential_predictions`` list (per the plan, it's a flattened view of
the full source-of-truth artifact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .candidates import ArticleLLMResult
from .clean import sanitize_output_sentence_text
from .dedupe_candidates import dedupe_prediction_rows
from .extract import CachedArticle
from .segment import Sentence


# The exact column order used by the existing annotators/*.csv files.
PREDICTIONS_CSV_COLUMNS: tuple[str, ...] = (
    "Base Sentence",
    "Base Sentence (raw)",
    "Sentence Label",
    "Human Annotation",
    "Human Reasoning",
    "Source",
    "Date",
    "Source Meta Data",
    "title",
    "description",
    "URL",
    "Image URL",
    "content",
    "Query Domain",
    "query_string",
    "query_slug",
    "source_cache_file",
    "article_id",
    "prediction_visible",
    "Source_Field",
    "field_order",
)


@dataclass
class ProcessedArticle:
    """One article plus its sentences, candidate spans, and final domain."""

    article_id: int
    cached: CachedArticle
    sentences: list[Sentence]
    llm_result: ArticleLLMResult
    final_query_domain: str
    domain_source: str  # "llm" | "hint" | "fallback"
    domain_confidence: float
    domain_reason: str
    llm_cache_key: str | None = None
    llm_cache_hit: bool | None = None


@dataclass
class TransformOutput:
    raw_articles: list[dict[str, Any]] = field(default_factory=list)
    processed_sentences: list[dict[str, Any]] = field(default_factory=list)
    potential_predictions: list[dict[str, Any]] = field(default_factory=list)
    predictions_rows: list[dict[str, Any]] = field(default_factory=list)
    dedupe_meta: dict[str, Any] = field(default_factory=dict)


def _article_metadata_block(p: ProcessedArticle) -> dict[str, Any]:
    """Repeated per-row article metadata, mirroring annotators/*.json."""
    out = {
        "Source": p.cached.author,
        "Date": p.cached.published_at,
        "Source Meta Data": p.cached.source_meta,
        "title": sanitize_output_sentence_text(p.cached.title),
        "description": sanitize_output_sentence_text(p.cached.description),
        "URL": p.cached.url,
        "Image URL": p.cached.image_url,
        "content": sanitize_output_sentence_text(p.cached.content),
        "Query Domain": p.final_query_domain,
    }
    if p.llm_cache_key is not None:
        out["llm_cache_key"] = p.llm_cache_key
    if p.llm_cache_hit is not None:
        out["llm_cache_hit"] = p.llm_cache_hit
    return out


def _build_processed_sentence_row(
    p: ProcessedArticle,
    sent: Sentence,
    *,
    label: int,
    base_sentence: str,
    base_sentence_raw: str,
    primary_sentence_id: int | None = None,
    span_sentence_ids: list[int] | None = None,
    span_text: str | None = None,
    candidate_reason: str | None = None,
    reason_category: str | None = None,
    context_needed: bool | None = None,
    uncertainty_note: str | None = None,
) -> dict[str, Any]:
    base_sentence = sanitize_output_sentence_text(base_sentence)
    base_sentence_raw = sanitize_output_sentence_text(base_sentence_raw)
    base = {
        "Base Sentence": base_sentence,
        "Base Sentence (raw)": base_sentence_raw,
        "Sentence Label": label,
        "Human Annotation": "",
        "Human Reasoning": "",
    }
    base.update(_article_metadata_block(p))
    base.update(
        {
            "article_id": p.article_id,
            "prediction_visible": label,
            "Source_Field": sent.source_field,
            "field_order": sent.field_order,
        }
    )
    if primary_sentence_id is not None:
        base["primary_sentence_id"] = primary_sentence_id
    if span_sentence_ids is not None:
        base["span_sentence_ids"] = list(span_sentence_ids)
    if span_text is not None:
        base["span_text"] = sanitize_output_sentence_text(span_text)
    if candidate_reason is not None:
        base["candidate_reason"] = candidate_reason
    if reason_category is not None:
        base["reason_category"] = reason_category
    if context_needed is not None:
        base["context_needed"] = context_needed
    if uncertainty_note is not None:
        base["uncertainty_note"] = uncertainty_note
    base["query_domain_source"] = p.domain_source
    base["query_domain_confidence"] = p.domain_confidence
    base["query_domain_reason"] = p.domain_reason
    return base


def _build_raw_article_row(p: ProcessedArticle) -> dict[str, Any]:
    return {
        "Source Meta Data": p.cached.source_meta,
        "Source": p.cached.author,
        "title": sanitize_output_sentence_text(p.cached.title),
        "description": sanitize_output_sentence_text(p.cached.description),
        "URL": p.cached.url,
        "Image URL": p.cached.image_url,
        "Date": p.cached.published_at,
        "content": sanitize_output_sentence_text(p.cached.content),
        "Query Domain": p.final_query_domain,
        "article_id": p.article_id,
        "query_domain_source": p.domain_source,
        "query_domain_confidence": p.domain_confidence,
        "query_domain_reason": p.domain_reason,
    }


def _predictions_csv_row(json_row: dict[str, Any]) -> dict[str, Any]:
    """Project a JSON ``potential_predictions`` row to CSV columns."""
    out: dict[str, Any] = {}
    for col in PREDICTIONS_CSV_COLUMNS:
        v = json_row.get(col, "")
        if isinstance(v, dict):
            # Source Meta Data is stored as a python-repr string in CSV (matching
            # the existing annotator files).
            out[col] = str(v)
        elif isinstance(v, list):
            out[col] = ",".join(str(x) for x in v)
        elif v is None:
            out[col] = ""
        else:
            out[col] = v
    return out


def build_predictions_csv_rows(json_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project JSON ``potential_predictions`` rows to the canonical CSV schema."""
    return [_predictions_csv_row(row) for row in json_rows]


def add_run_provenance(
    transform_output: TransformOutput,
    *,
    query_string: str,
    query_slug: str,
    source_cache_file: str,
) -> None:
    """Attach run/query provenance to every row that may be exported.

    The cached CSV itself remains a separate artifact. These lightweight
    columns make each extracted annotation row traceable back to the query/cache
    that produced it.
    """
    provenance = {
        "query_string": query_string,
        "query_slug": query_slug,
        "source_cache_file": source_cache_file,
    }
    for section in (
        transform_output.raw_articles,
        transform_output.processed_sentences,
        transform_output.potential_predictions,
    ):
        for row in section:
            row.update(provenance)
    transform_output.predictions_rows = build_predictions_csv_rows(
        transform_output.potential_predictions
    )


def transform_articles(processed: list[ProcessedArticle]) -> TransformOutput:
    """Build the three JSON arrays + the flattened predictions CSV rows."""
    out = TransformOutput()

    for p in processed:
        out.raw_articles.append(_build_raw_article_row(p))

        # Build a sentence-id -> Sentence lookup so we can map LLM owned spans.
        sent_lookup = {s.sentence_id: s for s in p.sentences}

        # Index of which sentence ids are flagged as candidate primary sentences.
        primary_ids = {span.primary_sentence_id for span in p.llm_result.unique_spans}

        # 1) Per-sentence rows (every sentence in the article, label=0/1).
        # Group sentences by source field with a stable field_order = 0/1/2.
        for sent in p.sentences:
            label = 1 if sent.sentence_id in primary_ids else 0
            base_raw_text = _raw_text_for_field(p, sent.source_field)
            row = _build_processed_sentence_row(
                p,
                sent,
                label=label,
                base_sentence=sent.text,
                base_sentence_raw=base_raw_text,
            )
            out.processed_sentences.append(row)

        # 2) Per-prediction rows (one per unique candidate span). These also
        #    populate the predictions CSV.
        for span in p.llm_result.unique_spans:
            primary_sent = sent_lookup.get(span.primary_sentence_id)
            if primary_sent is None:
                continue
            row = _build_processed_sentence_row(
                p,
                primary_sent,
                label=1,
                base_sentence=primary_sent.text,
                base_sentence_raw=span.span_text or _raw_text_for_field(p, primary_sent.source_field),
                primary_sentence_id=span.primary_sentence_id,
                span_sentence_ids=span.span_sentence_ids,
                span_text=span.span_text,
                candidate_reason=span.candidate_reason,
                reason_category=span.reason_category,
                context_needed=span.context_needed,
                uncertainty_note=span.uncertainty_note,
            )
            out.potential_predictions.append(row)

    dedupe_report = dedupe_prediction_rows(out.potential_predictions)
    out.potential_predictions = dedupe_report.rows
    out.predictions_rows = build_predictions_csv_rows(out.potential_predictions)
    out.dedupe_meta = dedupe_report.as_json()

    return out


def _raw_text_for_field(p: ProcessedArticle, source_field: str) -> str:
    """The full source-field text, used as ``Base Sentence (raw)`` for non-prediction rows."""
    if source_field == "title":
        return p.cached.title or ""
    if source_field == "description":
        return p.cached.description or ""
    return p.cached.content or ""


def build_query_meta(
    *,
    query: str,
    query_domain: str,
    language: str,
    from_date: str | None,
    to_date: str | None,
    sort_by: str,
    page_size: int,
    timestamp_utc: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "query": query,
        "query_domain": query_domain,
        "language": language,
        "from_date": from_date,
        "to_date": to_date,
        "sort_by": sort_by,
        "page_size": page_size,
        "timestamp_utc": timestamp_utc,
    }
    if extra:
        out.update(extra)
    return out


def build_llm_meta(
    *,
    provider_name: str,
    model_display_name: str,
    model_api_id: str,
    prompt_version: str,
    domain_prompt_version: str,
    temperature: float,
    request_mode: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "provider": provider_name,
        "model_display_name": model_display_name,
        "model_api_id": model_api_id,
        "prompt_version": prompt_version,
        "domain_prompt_version": domain_prompt_version,
        "temperature": temperature,
        "request_mode": request_mode,
    }
    if extra:
        out.update(extra)
    return out


def build_full_json(
    *,
    query_meta: dict[str, Any],
    llm_meta: dict[str, Any],
    transform_output: TransformOutput,
) -> dict[str, Any]:
    payload = {
        "query_meta": query_meta,
        "llm_meta": llm_meta,
        "counts": {
            "num_raw_articles": len(transform_output.raw_articles),
            "num_sentences": len(transform_output.processed_sentences),
            "num_potential_predictions": len(transform_output.potential_predictions),
        },
        "raw_articles": transform_output.raw_articles,
        "processed_sentences": transform_output.processed_sentences,
        "potential_predictions": transform_output.potential_predictions,
    }
    if transform_output.dedupe_meta:
        payload["dedupe_meta"] = transform_output.dedupe_meta
    return payload
