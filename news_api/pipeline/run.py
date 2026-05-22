"""End-to-end orchestrator with ``--dry-run`` and prototype caps.

Process one cached CSV (default: the first one in
``input_csvs/newsapi_cache/``) and produce one matching JSON + predictions
CSV pair under ``annotators(1)/``. Use ``--dry-run`` to plan + estimate
tokens without making any LLM calls; use ``--fake-llm`` to replace Gemini
with a deterministic heuristic provider for testing.

Usage examples:
    python -m pipeline.run --list
    python -m pipeline.run --dry-run --max-articles 3
    python -m pipeline.run --fake-llm --max-articles 3
    python -m pipeline.run --csv input_csvs/newsapi_cache/news_articles_2026-04-07_13-15-13.csv-v1.csv
    python -m pipeline.run --csv input_csvs/newsapi_cache/news_articles_large.csv-v1.csv --progress-every 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .candidates import (
    ArticleLLMResult,
    estimate_article_tokens,
    run_llm_for_article,
)
from .clean import clean_article_text
from .config import (
    EXCLUDED_BUCKETS,
    PipelineConfig,
    is_allowed_query_domain,
    is_excluded_query_domain,
    load_config,
)
from .extract import (
    LoadStats,
    LoadedArticle,
    iter_cached_articles,
    list_cached_csvs,
)
from .llm_client import (
    DomainResponse,
    FakeLLMProvider,
    LLMProvider,
    make_provider,
)
from .llm_result_cache import LLMResultCache
from .load import next_available_version, output_paths, write_outputs
from .newsapi_ai_client import (
    EventRegistryArticleFetcher,
    fetch_plan_lines,
    fetch_specs_to_cache,
    get_newsapi_ai_key,
    load_query_specs,
    render_query_string,
)
from .package_quality_data import QualityPackageError, package_quality_data
from .segment import LLMWindow, Sentence, plan_windows, segment_article
from .state import ArticleStatus, StateStore, WindowStatus
from .transform import (
    ProcessedArticle,
    TransformOutput,
    add_run_provenance,
    build_predictions_csv_rows,
    build_full_json,
    build_llm_meta,
    build_query_meta,
    transform_articles,
)
from .utils import CanonicalArticleKey, fingerprint, now_utc_iso, parse_source_meta
from .validate import ValidationReport, validate_full_json


LOGGER = logging.getLogger("pipeline.run")
StatusLogger = Callable[[str], None]


@dataclass
class LLMCacheRunStats:
    enabled: bool = False
    hits: int = 0
    misses: int = 0
    bypassed: int = 0
    writes: int = 0
    refreshes: int = 0
    skipped_hits: int = 0
    skipped_state: int = 0
    skipped_prior_outputs: int = 0


@dataclass
class RunResult:
    csv_path: Path
    base_stem: str
    output_version: int
    json_path: Path | None
    csv_out_path: Path | None
    num_articles: int
    num_sentences: int
    num_predictions: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    excluded_articles: int
    duplicate_articles: int
    invalid_articles: int
    llm_cache_enabled: bool = False
    llm_cache_hits: int = 0
    llm_cache_misses: int = 0
    llm_cache_bypassed: int = 0
    llm_cache_writes: int = 0
    llm_cache_refreshes: int = 0
    llm_cache_skipped_hits: int = 0
    skipped_processed_articles: int = 0
    skipped_prior_output_articles: int = 0
    validation: ValidationReport | None = None
    notes: list[str] = field(default_factory=list)


def _stderr_status(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds % 60)
    return f"{minutes}m{rem:02d}s"


def _parse_progress_every_env() -> int:
    raw = os.environ.get("PIPELINE_PROGRESS_EVERY")
    if raw is None or raw.strip() == "":
        return 1
    try:
        return max(0, int(raw))
    except ValueError:
        LOGGER.warning("Invalid PIPELINE_PROGRESS_EVERY=%r; using 1", raw)
        return 1


def _new_cache_run_label() -> str:
    return time.strftime("run-%Y%m%dT%H%M%SZ", time.gmtime())


def _progress_prefix(label: str | None) -> str:
    return f"[progress:{label}]" if label else "[progress]"


def _count_processable_articles(
    csv_path: Path,
    cfg: PipelineConfig,
    *,
    max_articles: int | None,
) -> int:
    total = 0
    for loaded in iter_cached_articles(csv_path, cfg, stats=LoadStats()):
        if loaded.skip_reason:
            continue
        if max_articles is not None and total >= max_articles:
            break
        total += 1
    return total


def _infer_query_slug_from_stem(stem: str) -> str:
    value = stem
    if value.startswith("news_articles_"):
        value = value[len("news_articles_"):]
    match = re.search(r"_(?:\d{4}-\d{2}-\d{2}|any)_to_(?:\d{4}-\d{2}-\d{2}|any)$", value)
    if match:
        value = value[: match.start()]
    return value or stem


def _matching_query_spec_meta(cfg: PipelineConfig, stem: str) -> dict[str, Any] | None:
    queries_dir = cfg.repo_root / "pipeline" / "queries"
    try:
        specs = load_query_specs(queries_dir)
    except Exception as exc:
        LOGGER.debug("Could not load query specs for provenance: %s", exc)
        return None
    inferred_slug = _infer_query_slug_from_stem(stem)
    for spec in specs:
        if stem == spec.filename_stem or inferred_slug == spec.query_slug:
            return {
                "query": render_query_string(spec),
                "query_domain": spec.query_domain,
                "language": spec.lang,
                "from_date": spec.date_start,
                "to_date": spec.date_end,
                "sort_by": spec.sort_by,
                "page_size": spec.max_items,
                "query_slug": spec.query_slug,
                "extra": {
                    "newsapi_query_slug": spec.query_slug,
                    "newsapi_data_type": spec.data_type,
                    "newsapi_maxItems": spec.max_items,
                    "newsapi_bodyLen": spec.body_len,
                    "newsapi_isDuplicateFilter": spec.is_duplicate_filter,
                },
            }
    return None


def _extract_query_meta_from_stem(cfg: PipelineConfig, stem: str) -> dict[str, Any]:
    """Pull a sensible default query/domain hint out of a cached-CSV filename.

    The cached CSVs are named ``news_articles_<timestamp>.csv-v1`` and don't
    embed the query or domain. We synthesise a passthrough query_meta block
    so the output JSON is still well-formed; v1 query metadata is
    best-effort and a future hybrid mode can supersede it via a ``--query``
    flag.
    """
    matched = _matching_query_spec_meta(cfg, stem)
    if matched:
        matched["timestamp_utc"] = now_utc_iso()
        return matched

    query_slug = _infer_query_slug_from_stem(stem)
    return {
        "query": stem,
        "query_domain": "misc-general",
        "language": "en",
        "from_date": None,
        "to_date": None,
        "sort_by": "relevancy",
        "page_size": 0,
        "timestamp_utc": now_utc_iso(),
        "query_slug": query_slug,
    }


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _resolve_query_domain(
    domain_resp: DomainResponse | None,
    *,
    query_hint: str | None,
) -> tuple[str, str, float, str]:
    """Decide the article-level Query Domain. Returns ``(domain, source, conf, reason)``."""
    if domain_resp:
        candidate = (domain_resp.query_domain or "").strip().lower()
        if candidate.startswith("misc-"):
            slug = _SLUG_RE.sub("-", candidate[len("misc-"):]).strip("-") or "general"
            candidate = f"misc-{slug}"
        if is_excluded_query_domain(candidate):
            # rejected → fall through to misc-general but record the reason
            return (
                "misc-general",
                "fallback",
                domain_resp.confidence,
                f"llm flagged excluded ({candidate!r}); fell back to misc-general",
            )
        if is_allowed_query_domain(candidate):
            return candidate, "llm", domain_resp.confidence, domain_resp.domain_reason or ""

    # fallback: try the provided hint
    if query_hint and is_allowed_query_domain(query_hint):
        return query_hint, "hint", 0.0, "no LLM domain; used query hint"

    return "misc-general", "fallback", 0.0, "no LLM domain or valid hint"


def _article_fingerprint(loaded: LoadedArticle) -> str:
    a = loaded.article
    return fingerprint(
        a.canonical_url or "",
        a.title or "",
        (a.source_meta or {}).get("name") or "",
        a.published_at or "",
    )


def _row_article_key(row: dict[str, Any]) -> CanonicalArticleKey:
    source_meta = row.get("Source Meta Data")
    if not isinstance(source_meta, dict):
        source_meta = parse_source_meta(source_meta)
    return CanonicalArticleKey.from_row(
        url=row.get("URL") or row.get("url"),
        title=row.get("title"),
        source_name=source_meta.get("name") if source_meta else None,
        published_at=row.get("Date") or row.get("publishedAt"),
    )


def _load_prior_output_article_keys(paths: list[Path]) -> set[CanonicalArticleKey]:
    keys: set[CanonicalArticleKey] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        rows = payload.get("raw_articles")
        if not isinstance(rows, list):
            rows = payload.get("processed_sentences") or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = _row_article_key(row)
            if not key.is_empty():
                keys.add(key)
    return keys


def _llm_cache_inputs(
    *,
    cfg: PipelineConfig,
    provider: LLMProvider,
    loaded: LoadedArticle,
    cleaned_text: str,
    sentences: list[Sentence],
    windows: list[LLMWindow],
    query_hint: str | None,
    target: int,
    hard_max: int,
    overlap: int,
    request_mode: str,
) -> tuple[str, dict[str, Any]]:
    article_fp = _article_fingerprint(loaded)
    cleaned_fp = fingerprint(cleaned_text)
    sentence_fp = fingerprint("|".join(sentence.text for sentence in sentences))
    candidate_prompt_version = str(
        cfg.llm.candidate_param("prompt_version", "candidate_span_extraction_v1")
    )
    domain_prompt_version = str(
        cfg.llm.domain_param("prompt_version", "query_domain_assignment_v1")
    )
    temperature = float(cfg.llm.candidate_param("temperature", 0))
    query_hint_value = query_hint or ""
    key = LLMResultCache.build_key(
        article_fingerprint=article_fp,
        cleaned_text_fingerprint=cleaned_fp,
        sentence_text_fingerprint=sentence_fp,
        query_domain_hint=query_hint_value,
        provider_name=provider.name,
        model_api_id=provider.model_api_id,
        candidate_prompt_version=candidate_prompt_version,
        domain_prompt_version=domain_prompt_version,
        temperature=temperature,
        target_sentences_per_call=target,
        hard_max_sentences_per_call=hard_max,
        overlap=overlap,
    )
    article = loaded.article
    metadata = {
        "article": {
            "canonical_url": article.canonical_url,
            "input_url": article.url,
            "input_title": article.title,
            "source_name": (article.source_meta or {}).get("name"),
            "published_at": article.published_at,
            "article_fingerprint": article_fp,
            "input_csv_path": str(article.input_csv_path),
            "input_row_index": article.input_row_index,
        },
        "extraction_input": {
            "cleaned_text_fingerprint": cleaned_fp,
            "sentence_text_fingerprint": sentence_fp,
            "sentence_count": len(sentences),
            "window_count": len(windows),
            "query_domain_hint": query_hint_value,
            "windowing_config": {
                "target_sentences_per_call": target,
                "hard_max_sentences_per_call": hard_max,
                "overlap": overlap,
            },
        },
        "model": {
            "provider_name": provider.name,
            "model_api_id": provider.model_api_id,
            "model_display_name": provider.model_display_name,
            "candidate_prompt_version": candidate_prompt_version,
            "domain_prompt_version": domain_prompt_version,
            "temperature": temperature,
            "request_mode": request_mode,
        },
    }
    return key, metadata


def _process_one_csv(
    *,
    cfg: PipelineConfig,
    csv_path: Path,
    provider: LLMProvider,
    state_store: StateStore | None,
    llm_result_cache: LLMResultCache | None,
    refresh_llm_cache: bool,
    skip_processed_articles: bool,
    skip_article_keys: set[CanonicalArticleKey] | None,
    max_articles: int | None,
    dry_run: bool,
    query_hint: str | None,
    request_mode: str,
    progress_every: int = 0,
    progress_windows: bool = False,
    progress_label: str | None = None,
    status_logger: StatusLogger | None = None,
) -> tuple[list[ProcessedArticle], LoadStats, int, int, LLMCacheRunStats]:
    """Run extraction → segmentation → (optional) LLM for one cached CSV."""
    stats = LoadStats()
    cache_stats = LLMCacheRunStats(enabled=llm_result_cache is not None)
    processed: list[ProcessedArticle] = []
    article_id = 0
    est_in = 0
    est_out = 0
    start_time = time.monotonic()
    total_articles: int | None = None
    prefix = _progress_prefix(progress_label)
    if status_logger is not None and progress_every > 0:
        if skip_processed_articles:
            total_articles = None
        else:
            total_articles = _count_processable_articles(
                csv_path,
                cfg,
                max_articles=max_articles,
            )
        status_logger(
            f"{prefix} queued={total_articles if total_articles is not None else 'unknown'} "
            f"dry_run={dry_run} progress_every={progress_every}"
        )

    for loaded in iter_cached_articles(csv_path, cfg, stats=stats):
        if loaded.skip_reason:
            LOGGER.debug("skip %s: %s", loaded.article.url, loaded.skip_reason)
            if state_store is not None:
                state_store.upsert_article(
                    ArticleStatus(
                        input_csv_path=str(csv_path),
                        input_row_index=loaded.article.input_row_index,
                        canonical_url=loaded.article.canonical_url,
                        article_fingerprint=_article_fingerprint(loaded),
                        article_id=-1,
                        status="skipped",
                        failure_reason=loaded.skip_reason,
                    )
                )
            continue

        article_fp = _article_fingerprint(loaded)
        if skip_article_keys and loaded.article.canonical_key in skip_article_keys:
            cache_stats.skipped_prior_outputs += 1
            LOGGER.debug(
                "skip previously output article %s",
                loaded.article.url,
            )
            continue

        if (
            skip_processed_articles
            and state_store is not None
            and not dry_run
            and not refresh_llm_cache
        ):
            previous = state_store.get_article(
                input_row_index=loaded.article.input_row_index,
                article_fingerprint=article_fp,
            )
            if previous is not None and previous.status == "llm_done":
                cache_stats.skipped_state += 1
                LOGGER.debug(
                    "skip previously processed %s: state llm_done",
                    loaded.article.url,
                )
                continue

        if max_articles is not None and article_id >= max_articles:
            break

        cleaned = clean_article_text(
            title=loaded.article.title,
            description=loaded.article.description,
            content=loaded.article.content,
        )
        sentences = segment_article(
            title=loaded.article.title,
            description=loaded.article.description,
            content=cleaned.text or loaded.article.content,
        )
        if not sentences:
            stats.rows_skipped_invalid += 1
            if state_store is not None:
                state_store.upsert_article(
                    ArticleStatus(
                        input_csv_path=str(csv_path),
                        input_row_index=loaded.article.input_row_index,
                        canonical_url=loaded.article.canonical_url,
                        article_fingerprint=article_fp,
                        article_id=-1,
                        status="skipped",
                        failure_reason="no_sentences_after_segmentation",
                    )
                )
            continue

        # Window plan from llm.yaml (with safe defaults).
        target = int(cfg.llm.candidate_param("target_sentences_per_call", 40))
        hard_max = int(cfg.llm.candidate_param("hard_max_sentences_per_call", 60))
        overlap = int(cfg.llm.candidate_param("overlap_sentences", 5))
        windows = plan_windows(
            sentences,
            target_sentences=target,
            hard_max_sentences=hard_max,
            overlap=overlap,
        )

        if dry_run:
            est = estimate_article_tokens(provider=provider, windows=windows)
            est_in += est.input_tokens
            est_out += est.output_tokens
            domain, source, conf, reason = _resolve_query_domain(
                None, query_hint=query_hint
            )
            llm_result = ArticleLLMResult(
                domain=None,
                windows=[],
                unique_spans=[],
                dedupe_drops=0,
                total_input_tokens=est.input_tokens,
                total_output_tokens=est.output_tokens,
            )
            processed.append(
                ProcessedArticle(
                    article_id=article_id,
                    cached=loaded.article,
                    sentences=sentences,
                    llm_result=llm_result,
                    final_query_domain=domain,
                    domain_source=source,
                    domain_confidence=conf,
                    domain_reason=reason,
                )
            )
        else:
            canonical_id = loaded.article.canonical_url or f"row:{loaded.article.input_row_index}"
            article_num = article_id + 1

            def log_window_progress(message: str) -> None:
                if status_logger is None:
                    return
                total_text = f"/{total_articles}" if total_articles is not None else ""
                status_logger(
                    f"{prefix} article={article_num}{total_text} {message} "
                    f"elapsed={_format_elapsed(time.monotonic() - start_time)}"
                )

            cache_key: str | None = None
            cache_hit = False
            cache_metadata: dict[str, Any] | None = None
            if llm_result_cache is not None:
                cache_key, cache_metadata = _llm_cache_inputs(
                    cfg=cfg,
                    provider=provider,
                    loaded=loaded,
                    cleaned_text=cleaned.text,
                    sentences=sentences,
                    windows=windows,
                    query_hint=query_hint,
                    target=target,
                    hard_max=hard_max,
                    overlap=overlap,
                    request_mode=request_mode,
                )
                if not refresh_llm_cache:
                    llm_result = llm_result_cache.get(cache_key)
                    if llm_result is not None:
                        cache_hit = True
                        cache_stats.hits += 1
                        if skip_processed_articles:
                            cache_stats.skipped_hits += 1
                            LOGGER.debug(
                                "skip previously processed %s: llm cache hit",
                                loaded.article.url,
                            )
                            continue
                    else:
                        cache_stats.misses += 1
                else:
                    llm_result = None
                    cache_stats.refreshes += 1
            else:
                llm_result = None
                cache_stats.bypassed += 1

            if llm_result is None:
                llm_result = run_llm_for_article(
                    provider=provider,
                    article_title=loaded.article.title,
                    article_description=loaded.article.description,
                    article_url=loaded.article.url,
                    article_content=cleaned.text,
                    query_domain_hint=query_hint,
                    sentences=sentences,
                    windows=windows,
                    canonical_article_id=canonical_id,
                    window_status_logger=log_window_progress if progress_windows else None,
                )
                if llm_result_cache is not None and cache_key and cache_metadata is not None:
                    llm_result_cache.put(cache_key, llm_result, cache_metadata)
                    cache_stats.writes += 1
            est_in += llm_result.total_input_tokens
            est_out += llm_result.total_output_tokens

            domain, source, conf, reason = _resolve_query_domain(
                llm_result.domain, query_hint=query_hint
            )
            processed.append(
                ProcessedArticle(
                    article_id=article_id,
                    cached=loaded.article,
                    sentences=sentences,
                    llm_result=llm_result,
                    final_query_domain=domain,
                    domain_source=source,
                    domain_confidence=conf,
                    domain_reason=reason,
                    llm_cache_key=cache_key,
                    llm_cache_hit=cache_hit if cache_key is not None else None,
                )
            )

            if state_store is not None:
                state_store.upsert_article(
                    ArticleStatus(
                        input_csv_path=str(csv_path),
                        input_row_index=loaded.article.input_row_index,
                        canonical_url=loaded.article.canonical_url,
                        article_fingerprint=article_fp,
                        article_id=article_id,
                        status="llm_done",
                        fetch_method="cached_csv",
                    )
                )
                for win_result in llm_result.windows:
                    win = win_result.window
                    state_store.upsert_window(
                        WindowStatus(
                            article_id=article_id,
                            window_index=win.window_index,
                            sentence_id_start=win.sentence_id_start,
                            sentence_id_end=win.sentence_id_end,
                            owned_sentence_id_start=win.owned_sentence_id_start,
                            owned_sentence_id_end=win.owned_sentence_id_end,
                            window_fingerprint=fingerprint(
                                canonical_id,
                                win.window_index,
                                "|".join(s.text for s in win.sentences),
                            ),
                            prompt_version=str(
                                cfg.llm.candidate_param(
                                    "prompt_version",
                                    "candidate_span_extraction_v1",
                                )
                            ),
                            model_api_id=provider.model_api_id,
                            status="llm_done",
                        )
                    )

        completed = article_id + 1
        if (
            status_logger is not None
            and progress_every > 0
            and (
                completed % progress_every == 0
                or (total_articles is not None and completed == total_articles)
            )
        ):
            llm_result_for_progress = processed[-1].llm_result
            total_text = f"/{total_articles}" if total_articles is not None else ""
            status_logger(
                f"{prefix} article={completed}{total_text} "
                f"row={loaded.article.input_row_index} "
                f"sentences={len(sentences)} windows={len(windows)} "
                f"candidates={len(llm_result_for_progress.unique_spans)} "
                f"elapsed={_format_elapsed(time.monotonic() - start_time)}"
            )

        article_id += 1

    return processed, stats, est_in, est_out, cache_stats


def _build_payload(
    *,
    cfg: PipelineConfig,
    provider: LLMProvider,
    processed: list[ProcessedArticle],
    csv_path: Path,
    csv_stem: str,
    request_mode: str,
    cache_stats: LLMCacheRunStats | None = None,
    query_meta_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_meta = _extract_query_meta_from_stem(cfg, csv_stem)
    if query_meta_override:
        base_meta.update(
            {k: v for k, v in query_meta_override.items() if v is not None}
        )
    query_meta = build_query_meta(
        query=base_meta["query"],
        query_domain=base_meta["query_domain"],
        language=base_meta["language"],
        from_date=base_meta["from_date"],
        to_date=base_meta["to_date"],
        sort_by=base_meta["sort_by"],
        page_size=base_meta["page_size"],
        timestamp_utc=base_meta["timestamp_utc"],
        extra={
            "input_csv_stem": csv_stem,
            "query_slug": base_meta.get("query_slug") or _infer_query_slug_from_stem(csv_stem),
            **(base_meta.get("extra") or {}),
        },
    )
    llm_meta = build_llm_meta(
        provider_name=provider.name,
        model_display_name=provider.model_display_name,
        model_api_id=provider.model_api_id,
        prompt_version=str(
            cfg.llm.candidate_param("prompt_version", "candidate_span_extraction_v1")
        ),
        domain_prompt_version=str(
            cfg.llm.domain_param("prompt_version", "query_domain_assignment_v1")
        ),
        temperature=float(cfg.llm.candidate_param("temperature", 0)),
        request_mode=request_mode,
        extra={
            "llm_cache_enabled": bool(cache_stats and cache_stats.enabled),
            "llm_cache_hits": cache_stats.hits if cache_stats else 0,
            "llm_cache_misses": cache_stats.misses if cache_stats else 0,
            "llm_cache_bypassed": cache_stats.bypassed if cache_stats else 0,
            "llm_cache_writes": cache_stats.writes if cache_stats else 0,
            "llm_cache_refreshes": cache_stats.refreshes if cache_stats else 0,
            "llm_cache_skipped_hits": cache_stats.skipped_hits if cache_stats else 0,
            "skipped_processed_articles": (
                (cache_stats.skipped_state + cache_stats.skipped_hits)
                if cache_stats
                else 0
            ),
            "skipped_prior_output_articles": (
                cache_stats.skipped_prior_outputs if cache_stats else 0
            ),
        },
    )
    transform_output: TransformOutput = transform_articles(processed)
    query_slug = str(
        query_meta.get("newsapi_query_slug")
        or query_meta.get("query_slug")
        or base_meta.get("query_slug")
        or _infer_query_slug_from_stem(csv_stem)
    )
    add_run_provenance(
        transform_output,
        query_string=str(query_meta.get("query") or ""),
        query_slug=query_slug,
        source_cache_file=csv_path.name,
    )
    full = build_full_json(
        query_meta=query_meta,
        llm_meta=llm_meta,
        transform_output=transform_output,
    )
    return full, transform_output.predictions_rows


def run_one_csv(
    *,
    cfg: PipelineConfig,
    csv_path: Path,
    provider: LLMProvider,
    dry_run: bool,
    max_articles: int | None,
    overwrite: bool,
    output_version: int | None = None,
    query_hint: str | None = None,
    query_meta_override: dict[str, Any] | None = None,
    request_mode: str = "free_tier",
    enable_state: bool = True,
    llm_cache_enabled: bool | None = None,
    llm_cache_dir: Path | None = None,
    refresh_llm_cache: bool = False,
    skip_processed_articles: bool = False,
    skip_article_keys: set[CanonicalArticleKey] | None = None,
    progress_every: int = 0,
    progress_windows: bool = False,
    progress_label: str | None = None,
    status_logger: StatusLogger | None = None,
) -> RunResult:
    """Run the pipeline against one cached CSV. Returns a RunResult summary."""
    csv_stem = _csv_stem(csv_path)

    state_store: StateStore | None = None
    if enable_state and not dry_run:
        state_dir = cfg.state_dir / csv_stem
        state_store = StateStore(state_dir)

    effective_llm_cache = False
    if not dry_run:
        effective_llm_cache = (
            provider.name != "fake" if llm_cache_enabled is None else llm_cache_enabled
        )
    cache: LLMResultCache | None = None
    if effective_llm_cache:
        resolved_cache_dir = llm_cache_dir or (cfg.state_dir / "llm_result_cache")
        if not resolved_cache_dir.is_absolute():
            resolved_cache_dir = (cfg.repo_root / resolved_cache_dir).resolve()
        cache = LLMResultCache(resolved_cache_dir)

    processed, stats, est_in, est_out, cache_stats = _process_one_csv(
        cfg=cfg,
        csv_path=csv_path,
        provider=provider,
        state_store=state_store,
        llm_result_cache=cache,
        refresh_llm_cache=refresh_llm_cache,
        skip_processed_articles=skip_processed_articles,
        skip_article_keys=skip_article_keys,
        max_articles=max_articles,
        dry_run=dry_run,
        query_hint=query_hint,
        request_mode=request_mode,
        progress_every=progress_every,
        progress_windows=progress_windows,
        progress_label=progress_label,
        status_logger=status_logger,
    )

    if dry_run:
        return RunResult(
            csv_path=csv_path,
            base_stem=csv_stem,
            output_version=output_version or 1,
            json_path=None,
            csv_out_path=None,
            num_articles=len(processed),
            num_sentences=sum(len(p.sentences) for p in processed),
            num_predictions=0,
            estimated_input_tokens=est_in,
            estimated_output_tokens=est_out,
            excluded_articles=stats.rows_excluded_host + stats.rows_excluded_path,
            duplicate_articles=stats.rows_deduped,
            invalid_articles=stats.rows_skipped_invalid,
            llm_cache_enabled=cache_stats.enabled,
            llm_cache_hits=cache_stats.hits,
            llm_cache_misses=cache_stats.misses,
            llm_cache_bypassed=cache_stats.bypassed,
            llm_cache_writes=cache_stats.writes,
            llm_cache_refreshes=cache_stats.refreshes,
            llm_cache_skipped_hits=cache_stats.skipped_hits,
            skipped_processed_articles=cache_stats.skipped_state
            + cache_stats.skipped_hits,
            skipped_prior_output_articles=cache_stats.skipped_prior_outputs,
            notes=[
                f"cached_csv_rows_seen={stats.rows_seen}",
                f"cached_csv_rows_kept={stats.rows_kept}",
            ],
        )

    full_payload, prediction_rows = _build_payload(
        cfg=cfg,
        provider=provider,
        processed=processed,
        csv_path=csv_path,
        csv_stem=csv_stem,
        request_mode=request_mode,
        cache_stats=cache_stats,
        query_meta_override=query_meta_override,
    )

    report = validate_full_json(full_payload)
    prediction_rows = build_predictions_csv_rows(
        full_payload.get("potential_predictions", [])
    )

    version = output_version or next_available_version(
        output_dir=cfg.output_dir, base_stem=csv_stem
    )
    outcome = write_outputs(
        output_dir=cfg.output_dir,
        base_stem=csv_stem,
        version=version,
        full_payload=full_payload,
        predictions_rows=prediction_rows,
        overwrite=overwrite,
    )

    return RunResult(
        csv_path=csv_path,
        base_stem=csv_stem,
        output_version=version,
        json_path=outcome.json_path,
        csv_out_path=outcome.csv_path,
        num_articles=len(processed),
        num_sentences=len(full_payload.get("processed_sentences", [])),
        num_predictions=len(full_payload.get("potential_predictions", [])),
        estimated_input_tokens=est_in,
        estimated_output_tokens=est_out,
        excluded_articles=stats.rows_excluded_host + stats.rows_excluded_path,
        duplicate_articles=stats.rows_deduped,
        invalid_articles=stats.rows_skipped_invalid,
        llm_cache_enabled=cache_stats.enabled,
        llm_cache_hits=cache_stats.hits,
        llm_cache_misses=cache_stats.misses,
        llm_cache_bypassed=cache_stats.bypassed,
        llm_cache_writes=cache_stats.writes,
        llm_cache_refreshes=cache_stats.refreshes,
        llm_cache_skipped_hits=cache_stats.skipped_hits,
        skipped_processed_articles=cache_stats.skipped_state
        + cache_stats.skipped_hits,
        skipped_prior_output_articles=cache_stats.skipped_prior_outputs,
        validation=report,
    )


def _csv_stem(path: Path) -> str:
    """``news_articles_2026-...csv-v1.csv`` -> ``news_articles_2026-...``.

    Strips the longest known cache suffix first so we don't double-stamp
    the version segment in output filenames.
    """
    name = path.name
    for suffix in (".csv-v1.csv", ".csv-v2.csv", ".csv-v3.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Cached NewsAPI CSV to process. Default: the first file in the cache dir.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the cached CSVs the pipeline would consider and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan windows and estimate tokens without making any LLM calls.",
    )
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="Force the deterministic FakeLLMProvider (no API calls).",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Cap number of articles processed (prototype mode).",
    )
    parser.add_argument(
        "--query-hint",
        default=None,
        help=(
            "Optional canonical query domain hint (e.g. 'sport', 'misc-tesla'). "
            "Falls back to misc-general."
        ),
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Output version. Default: lowest free v<N> in annotators(1)/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Skip persistent pipeline_state writes.",
    )
    parser.add_argument(
        "--llm-cache",
        dest="llm_cache",
        action="store_true",
        default=None,
        help=(
            "Enable article-level LLM result reuse. Default: enabled for real "
            "LLM runs, disabled for dry-run/fake LLM runs."
        ),
    )
    parser.add_argument(
        "--no-llm-cache",
        dest="llm_cache",
        action="store_false",
        help="Disable article-level LLM result reuse and always call the LLM.",
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=Path("pipeline_state/llm_result_cache"),
        help="Directory for article-level LLM result JSON cache files.",
    )
    parser.add_argument(
        "--refresh-llm-cache",
        action="store_true",
        help="Call the LLM and overwrite matching cache entries.",
    )
    parser.add_argument(
        "--skip-processed-articles",
        action="store_true",
        help=(
            "Skip articles already marked llm_done in pipeline_state or already "
            "present in the LLM cache. With --max-articles, the cap applies to "
            "newly processed articles."
        ),
    )
    parser.add_argument(
        "--skip-output-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Path to a prior *_full-v*.json whose raw_articles should be skipped. "
            "May be passed multiple times. With --max-articles, the cap applies "
            "after these articles are skipped."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every cached CSV under input_csvs/newsapi_cache/.",
    )
    parser.add_argument(
        "--fetch-newsapi",
        action="store_true",
        help="Fetch fresh NewsAPI.ai/Event Registry results into the cache before processing.",
    )
    parser.add_argument(
        "--skip-existing-cache",
        action="store_true",
        help=(
            "With --fetch-newsapi, skip query specs whose target cache CSV already "
            "exists instead of aborting the fetch."
        ),
    )
    parser.add_argument(
        "--process-existing-cache",
        action="store_true",
        help=(
            "With --skip-existing-cache, also process existing cache CSVs that were "
            "skipped during fetch. Default is to process only newly fetched CSVs."
        ),
    )
    parser.add_argument(
        "--new-cache-per-run",
        action="store_true",
        help=(
            "With --fetch-newsapi, write a fresh timestamped cache CSV for this run "
            "instead of reusing the stable query/date cache filename."
        ),
    )
    parser.add_argument(
        "--cache-run-label",
        default=None,
        help=(
            "Optional filename suffix for --new-cache-per-run. Default: "
            "run-YYYYMMDDTHHMMSSZ."
        ),
    )
    parser.add_argument(
        "--package-quality-data",
        action="store_true",
        help=(
            "After writing annotators(1) outputs, package them into the "
            "quality_data/<domain>_predictions/<domain>_N human-annotation layout."
        ),
    )
    parser.add_argument(
        "--quality-out-root",
        type=Path,
        default=Path("quality_data"),
        help="Output root for --package-quality-data. Default: quality_data/.",
    )
    parser.add_argument(
        "--quality-domain",
        default=None,
        help="Optional domain override for --package-quality-data, e.g. policy.",
    )
    parser.add_argument(
        "--quality-max-rows-per-file",
        type=int,
        default=300,
        help="Maximum annotation rows per quality batch file.",
    )
    parser.add_argument(
        "--quality-start-index",
        type=int,
        default=None,
        help="Optional starting batch index for --package-quality-data.",
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=Path("pipeline/queries"),
        help="Directory containing Event Registry query YAML specs.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help=(
            "Query spec filename or slug to fetch. May be passed multiple times. "
            "Default with --fetch-newsapi: all YAML files in --queries-dir."
        ),
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="With --fetch-newsapi, write cache CSVs and exit before Gemini processing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=_parse_progress_every_env(),
        help=(
            "Write Gemini processing progress to stderr every K completed articles. "
            "Use 0 to disable. Default: PIPELINE_PROGRESS_EVERY or 1."
        ),
    )
    parser.add_argument(
        "--progress-windows",
        action="store_true",
        help=(
            "Also write per-window Gemini start/done lines to stderr. "
            "Useful for very long articles; default off."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase log verbosity."
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = load_config()
    progress_every = max(0, args.progress_every)
    skip_article_keys: set[CanonicalArticleKey] = set()
    if args.skip_output_json:
        skip_paths: list[Path] = []
        for path in args.skip_output_json:
            resolved = path if path.is_absolute() else (cfg.repo_root / path).resolve()
            if not resolved.exists():
                print(f"Prior output JSON not found: {resolved}", file=sys.stderr)
                return 2
            skip_paths.append(resolved)
        try:
            skip_article_keys = _load_prior_output_article_keys(skip_paths)
        except Exception as exc:
            print(
                f"Failed to load --skip-output-json: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
        print(
            f"[run] skip_output_json: {len(skip_paths)} file(s), "
            f"{len(skip_article_keys)} article key(s)"
        )

    fetched_csvs: list[Path] = []
    fetched_query_hints: dict[Path, str] = {}
    fetched_query_meta: dict[Path, dict[str, Any]] = {}
    if args.fetch_newsapi:
        cache_run_label = None
        if args.new_cache_per_run:
            cache_run_label = args.cache_run_label or _new_cache_run_label()
            print(f"[fetch] cache_run_label={cache_run_label}")
        queries_dir = args.queries_dir
        if not queries_dir.is_absolute():
            queries_dir = (cfg.repo_root / queries_dir).resolve()
        specs = load_query_specs(queries_dir, selected=args.query)
        if not specs:
            print(f"No query specs found under {queries_dir}.", file=sys.stderr)
            return 2

        print("[fetch] Event Registry query plan:")
        for line in fetch_plan_lines(specs):
            print(f"[fetch] {line}")

        if args.dry_run:
            print("[fetch] dry-run: no Event Registry requests will be made.")
            return 0
        else:
            api_key = get_newsapi_ai_key()
            if not api_key:
                print(
                    "Missing NEWSAPI_AI_API_KEY (or NEWSAPI_API_KEY) in environment/.env.",
                    file=sys.stderr,
                )
                return 2
            fetcher = EventRegistryArticleFetcher(
                api_key=api_key,
                verbose_output=args.verbose >= 1,
                repeat_failed_request_count=2,
            )
            try:
                fetch_results = fetch_specs_to_cache(
                    specs=specs,
                    fetcher=fetcher,
                    cfg=cfg,
                    dry_run=False,
                    overwrite=args.overwrite,
                    skip_existing=args.skip_existing_cache,
                    cache_label=cache_run_label,
                    dedupe_existing_cache=not args.new_cache_per_run,
                    progress_every=25,
                    status_logger=print,
                )
            except FileExistsError as exc:
                print(f"[fetch] SKIPPED (cache exists): {exc}", file=sys.stderr)
                return 3
            except Exception as exc:
                print(f"[fetch] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 4
            for res in fetch_results:
                if res.csv_path is not None:
                    resolved = res.csv_path.resolve()
                    if res.skipped_existing and not args.process_existing_cache:
                        print(f"[fetch] skipped existing cache: {res.csv_path}")
                        continue
                    fetched_csvs.append(res.csv_path)
                    fetched_query_hints[resolved] = res.spec.query_domain
                    fetched_query_meta[resolved] = {
                        "query": render_query_string(res.spec),
                        "query_domain": res.spec.query_domain,
                        "language": res.spec.lang,
                        "from_date": res.spec.date_start,
                        "to_date": res.spec.date_end,
                        "sort_by": res.spec.sort_by,
                        "page_size": res.spec.max_items,
                        "query_slug": res.spec.query_slug,
                        "extra": {
                            "newsapi_provider": "newsapi.ai/event_registry",
                            "newsapi_query_slug": res.spec.query_slug,
                            "newsapi_data_type": res.spec.data_type,
                            "newsapi_maxItems": res.spec.max_items,
                            "newsapi_bodyLen": res.spec.body_len,
                            "newsapi_isDuplicateFilter": res.spec.is_duplicate_filter,
                        },
                    }
                if res.skipped_existing:
                    print(
                        f"[fetch] existing={res.csv_path} "
                        f"process={args.process_existing_cache}"
                    )
                    continue
                print(
                    f"[fetch] wrote={res.csv_path} "
                    f"fetched={res.rows_fetched} kept={res.rows_written} "
                    f"dedupe={res.rows_deduped} excluded={res.rows_excluded}"
                )
            if args.fetch_only:
                return 0

    if args.list:
        csvs = list_cached_csvs(cfg.input_cache_dir)
        if not csvs:
            print(f"No cached CSVs found under {cfg.input_cache_dir}.")
            return 0
        for p in csvs:
            print(p)
        return 0

    csv_targets: list[Path]
    if args.fetch_newsapi and not args.batch and not args.csv:
        if not fetched_csvs:
            print("[run] no newly fetched CSVs to process.")
            return 0
        csv_targets = fetched_csvs
    elif args.batch:
        csv_targets = list_cached_csvs(cfg.input_cache_dir)
        if not csv_targets:
            print(f"No cached CSVs found under {cfg.input_cache_dir}.", file=sys.stderr)
            return 2
    elif args.csv:
        target = args.csv
        if not target.is_absolute():
            target = (cfg.repo_root / target).resolve()
        if not target.exists():
            print(f"CSV not found: {target}", file=sys.stderr)
            return 2
        csv_targets = [target]
    else:
        csvs = list_cached_csvs(cfg.input_cache_dir)
        if not csvs:
            print(f"No cached CSVs found under {cfg.input_cache_dir}.", file=sys.stderr)
            return 2
        csv_targets = [csvs[0]]

    provider: LLMProvider
    if args.dry_run or args.fake_llm:
        provider = FakeLLMProvider()
    else:
        provider = make_provider(cfg.llm, fake=False)

    print(f"[run] provider:  {provider.name} ({provider.model_api_id})")
    print(f"[run] dry_run:   {args.dry_run}  fake_llm: {args.fake_llm}")
    effective_llm_cache = (
        False
        if args.dry_run
        else (provider.name != "fake" if args.llm_cache is None else args.llm_cache)
    )
    print(f"[run] llm_cache: {effective_llm_cache}")
    if effective_llm_cache:
        print(f"[run] llm_cache_dir: {args.llm_cache_dir}")
        if args.refresh_llm_cache:
            print("[run] refresh_llm_cache: true")
    if args.skip_processed_articles:
        print("[run] skip_processed_articles: true")
    if args.max_articles is not None:
        print(f"[run] max_articles: {args.max_articles}")
    if args.batch:
        print(f"[run] batch mode: {len(csv_targets)} cached CSV(s)")

    failures = 0
    last_result: RunResult | None = None
    for csv_path in csv_targets:
        print()
        print(f"[run] input csv: {csv_path}")
        try:
            query_hint = args.query_hint or fetched_query_hints.get(csv_path.resolve())
            query_meta_override = fetched_query_meta.get(csv_path.resolve())
            result = run_one_csv(
                cfg=cfg,
                csv_path=csv_path,
                provider=provider,
                dry_run=args.dry_run,
                max_articles=args.max_articles,
                overwrite=args.overwrite,
                output_version=args.version,
                query_hint=query_hint,
                query_meta_override=query_meta_override,
                enable_state=not args.no_state,
                llm_cache_enabled=args.llm_cache,
                llm_cache_dir=args.llm_cache_dir,
                refresh_llm_cache=args.refresh_llm_cache,
                skip_processed_articles=args.skip_processed_articles,
                skip_article_keys=skip_article_keys or None,
                progress_every=progress_every,
                progress_windows=args.progress_windows,
                progress_label=_csv_stem(csv_path) if args.batch else None,
                status_logger=(
                    _stderr_status
                    if progress_every > 0 or args.progress_windows
                    else None
                ),
            )
        except FileExistsError as exc:
            print(f"[run] SKIPPED (output exists): {exc}", file=sys.stderr)
            failures += 1
            continue
        last_result = result
        _print_result_summary(result, args, cfg)
        if args.package_quality_data and not args.dry_run:
            _package_quality_result(result, args, cfg)

    if last_result is None:
        return 2 if failures else 1
    return 3 if failures else 0


def _resolve_repo_path(cfg: PipelineConfig, path: Path) -> Path:
    return path if path.is_absolute() else (cfg.repo_root / path).resolve()


def _package_quality_result(result: RunResult, args, cfg: PipelineConfig) -> None:
    if not result.json_path or not result.csv_out_path:
        return
    out_root = _resolve_repo_path(cfg, args.quality_out_root)
    queries_dir = _resolve_repo_path(cfg, args.queries_dir)
    try:
        summary = package_quality_data(
            predictions_csv=result.csv_out_path,
            full_json=result.json_path,
            cached_csv=result.csv_path,
            out_root=out_root,
            domain=args.quality_domain,
            max_rows_per_file=args.quality_max_rows_per_file,
            start_index=args.quality_start_index,
            queries_dir=queries_dir,
        )
    except QualityPackageError as exc:
        print(f"[quality] ERROR: {exc}", file=sys.stderr)
        return

    print(f"[quality] package_dir={summary.package_dir}")
    print(f"[quality] total_rows={summary.total_rows}")
    print(f"[quality] total_articles={summary.total_articles}")
    print(f"[quality] batches={summary.batch_count}")
    for batch in summary.batches:
        print(
            f"[quality] {batch.batch_id}: rows={batch.row_count} "
            f"articles={batch.article_count} folder={batch.folder}"
        )
    print(f"[quality] manifest={summary.manifest_path}")


def _print_result_summary(result: RunResult, args, cfg: PipelineConfig) -> None:

    print(f"[run] articles processed:  {result.num_articles}")
    print(f"[run] sentences total:     {result.num_sentences}")
    print(f"[run] predictions:         {result.num_predictions}")
    print(
        f"[run] tokens (est.) input/output: "
        f"{result.estimated_input_tokens} / {result.estimated_output_tokens}"
    )
    print(
        f"[run] excluded={result.excluded_articles} "
        f"duplicate={result.duplicate_articles} "
        f"invalid={result.invalid_articles}"
    )
    print(
        f"[run] llm_cache enabled={result.llm_cache_enabled} "
        f"hits={result.llm_cache_hits} misses={result.llm_cache_misses} "
        f"bypassed={result.llm_cache_bypassed} writes={result.llm_cache_writes}"
    )
    if result.skipped_processed_articles:
        print(
            f"[run] skipped previously processed: {result.skipped_processed_articles} "
            f"(state={result.skipped_processed_articles - result.llm_cache_skipped_hits}, "
            f"cache={result.llm_cache_skipped_hits})"
        )
    if result.skipped_prior_output_articles:
        print(f"[run] skipped prior-output articles={result.skipped_prior_output_articles}")
    if result.llm_cache_refreshes:
        print(f"[run] llm_cache refreshes={result.llm_cache_refreshes}")

    if args.dry_run:
        json_path, csv_out = output_paths(
            output_dir=cfg.output_dir,
            base_stem=result.base_stem,
            version=result.output_version,
        )
        print(f"[dry-run] would write: {json_path}")
        print(f"[dry-run] would write: {csv_out}")
        return

    if result.validation and result.validation.warnings:
        for w in result.validation.warnings[:5]:
            print(f"[validate] WARN {w}")
        if len(result.validation.warnings) > 5:
            print(
                f"[validate] ... +{len(result.validation.warnings) - 5} more warnings"
            )

    if result.validation and result.validation.errors:
        for e in result.validation.errors:
            print(f"[validate] ERROR {e}")

    if result.json_path:
        print(f"[run] wrote: {result.json_path}")
    if result.csv_out_path:
        print(f"[run] wrote: {result.csv_out_path}")


if __name__ == "__main__":
    sys.exit(main())
