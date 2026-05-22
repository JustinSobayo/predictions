"""Cached-CSV loader (the input side of the hybrid extract step).

Reads ``input_csvs/newsapi_cache/news_articles_*.csv-v1.csv`` and yields
normalised article dicts ready for cleaning, segmentation, and the LLM pass.
Live NewsAPI.ai / Event Registry querying writes into this same cache via
``pipeline.newsapi_ai_client`` so the downstream path stays identical.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .config import PipelineConfig
from .excluded_domains import ExclusionVerdict, evaluate_url
from .utils import (
    CanonicalArticleKey,
    canonicalize_url,
    parse_source_meta,
    url_host,
)


# Standard NewsAPI.org / cached-CSV columns.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "source",
    "author",
    "title",
    "description",
    "url",
    "urlToImage",
    "publishedAt",
    "content",
)


@dataclass
class CachedArticle:
    """One row from a cached NewsAPI CSV, in pipeline-friendly form."""

    input_csv_path: Path
    input_row_index: int
    raw: dict[str, str | None]
    source_meta: dict[str, str | None]
    title: str | None
    description: str | None
    url: str | None
    canonical_url: str | None
    image_url: str | None
    published_at: str | None
    content: str | None
    author: str | None

    canonical_key: CanonicalArticleKey = field(init=False)

    def __post_init__(self) -> None:
        self.canonical_key = CanonicalArticleKey.from_row(
            url=self.url,
            title=self.title,
            source_name=self.source_meta.get("name") if self.source_meta else None,
            published_at=self.published_at,
        )


@dataclass
class LoadedArticle:
    """A cached article plus the verdict from the excluded-domain detector."""

    article: CachedArticle
    exclusion: ExclusionVerdict
    skip_reason: str | None = None


@dataclass
class LoadStats:
    rows_seen: int = 0
    rows_kept: int = 0
    rows_excluded_host: int = 0
    rows_excluded_path: int = 0
    rows_skipped_invalid: int = 0
    rows_deduped: int = 0


def list_cached_csvs(cache_dir: Path) -> list[Path]:
    """Return all cached NewsAPI CSVs in alphabetical order."""
    if not cache_dir.exists():
        return []
    return sorted(p for p in cache_dir.glob("news_articles_*.csv*") if p.is_file())


def _read_rows(path: Path) -> Iterator[tuple[int, dict[str, str | None]]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            yield idx, {k: (v if v != "" else None) for k, v in row.items()}


def _coerce_article(path: Path, idx: int, row: dict[str, str | None]) -> CachedArticle:
    url = (row.get("url") or "").strip() or None
    return CachedArticle(
        input_csv_path=path,
        input_row_index=idx,
        raw=row,
        source_meta=parse_source_meta(row.get("source")),
        title=(row.get("title") or "").strip() or None,
        description=(row.get("description") or "").strip() or None,
        url=url,
        canonical_url=canonicalize_url(url),
        image_url=(row.get("urlToImage") or "").strip() or None,
        published_at=(row.get("publishedAt") or "").strip() or None,
        content=row.get("content") or None,
        author=(row.get("author") or "").strip() or None,
    )


def iter_cached_articles(
    csv_path: Path,
    cfg: PipelineConfig,
    *,
    stats: LoadStats | None = None,
) -> Iterator[LoadedArticle]:
    """Yield every article from one cached CSV, with exclusion verdicts attached.

    Rows that are duplicates inside the same CSV (by canonical URL or fallback
    key) are dropped. Rejected rows are still yielded so callers can audit /
    log them; check ``LoadedArticle.exclusion.rejected`` and ``skip_reason``.
    """
    seen: set[CanonicalArticleKey] = set()
    for idx, row in _read_rows(csv_path):
        article = _coerce_article(csv_path, idx, row)
        if stats is not None:
            stats.rows_seen += 1

        # basic validity: must have either a url or *some* identifying text
        if article.canonical_key.is_empty() and not article.title:
            if stats is not None:
                stats.rows_skipped_invalid += 1
            yield LoadedArticle(article, ExclusionVerdict(False, None, None, None), "invalid_no_id")
            continue

        if article.canonical_key in seen:
            if stats is not None:
                stats.rows_deduped += 1
            yield LoadedArticle(article, ExclusionVerdict(False, None, None, None), "duplicate_in_csv")
            continue
        seen.add(article.canonical_key)

        verdict = evaluate_url(article.url, cfg.excluded)
        if verdict.rejected:
            if stats is not None:
                if verdict.tier == "host":
                    stats.rows_excluded_host += 1
                elif verdict.tier == "path":
                    stats.rows_excluded_path += 1
            yield LoadedArticle(article, verdict, verdict.reason)
            continue

        if stats is not None:
            stats.rows_kept += 1
        yield LoadedArticle(article, verdict, None)


def cross_csv_dedupe(
    sources: Iterable[Iterable[LoadedArticle]],
) -> Iterator[LoadedArticle]:
    """Stream rows from many CSVs, suppressing cross-CSV duplicates."""
    seen: set[CanonicalArticleKey] = set()
    for source in sources:
        for loaded in source:
            if loaded.skip_reason:
                yield loaded
                continue
            if loaded.article.canonical_key in seen:
                yield LoadedArticle(loaded.article, loaded.exclusion, "duplicate_cross_csv")
                continue
            seen.add(loaded.article.canonical_key)
            yield loaded
