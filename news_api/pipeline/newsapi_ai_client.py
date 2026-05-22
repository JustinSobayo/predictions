"""NewsAPI.ai / Event Registry fetch layer.

The rest of the ETL already knows how to process cached NewsAPI-shaped CSVs.
This module is responsible for turning Event Registry query specs into those
same CSVs so the downstream Gemini/sentence pipeline does not need a second
input path.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import yaml

from .config import PipelineConfig, is_allowed_query_domain, is_excluded_query_domain
from .excluded_domains import evaluate_url
from .extract import EXPECTED_COLUMNS
from .utils import CanonicalArticleKey, parse_source_meta


NEWSAPI_AI_KEY_ENVS: tuple[str, ...] = ("NEWSAPI_AI_API_KEY", "NEWSAPI_API_KEY")


@dataclass(frozen=True)
class NewsAPIQuerySpec:
    query_slug: str
    query_domain: str
    lang: str = "eng"
    date_start: str | None = None
    date_end: str | None = None
    sort_by: str = "rel"
    sort_by_asc: bool = False
    max_items: int = 100
    data_type: str = "news"
    body_len: int = -1
    is_duplicate_filter: str = "skipDuplicates"
    keyword_groups: dict[str, list[str]] = field(default_factory=dict)
    keywords: str | list[str] | None = None
    complex_query: dict[str, Any] | str | None = None
    keyword_search_mode: str = "phrase"
    keywords_loc: str = "body"
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def filename_stem(self) -> str:
        return f"news_articles_{self.query_slug}_{self.date_start or 'any'}_to_{self.date_end or 'any'}"


@dataclass
class NewsAPIFetchResult:
    spec: NewsAPIQuerySpec
    csv_path: Path | None
    rows_fetched: int
    rows_written: int
    rows_deduped: int
    rows_excluded: int
    dry_run: bool = False
    skipped_existing: bool = False


class ArticleFetcher(Protocol):
    def fetch_articles(self, spec: NewsAPIQuerySpec) -> Iterable[dict[str, Any]]: ...


def get_newsapi_ai_key() -> str | None:
    for env_name in NEWSAPI_AI_KEY_ENVS:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def load_query_specs(
    queries_dir: Path,
    *,
    selected: list[str] | None = None,
) -> list[NewsAPIQuerySpec]:
    """Load and validate Event Registry query specs from YAML files."""
    if not queries_dir.exists():
        return []

    selected_set = {item for item in selected or []}
    paths: list[Path] = []
    if selected_set:
        for item in selected_set:
            p = Path(item)
            if not p.is_absolute():
                if p.exists():
                    p = p.resolve()
                else:
                    p = queries_dir / item
            if p.suffix not in {".yaml", ".yml"}:
                p = p.with_suffix(".yaml")
            paths.append(p)
    else:
        paths = sorted(
            p for p in queries_dir.glob("*.y*ml") if p.is_file()
        )

    specs: list[NewsAPIQuerySpec] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Query spec not found: {path}")
        specs.append(load_query_spec(path))
    return specs


def load_query_spec(path: Path) -> NewsAPIQuerySpec:
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Query spec must be a YAML mapping: {path}")

    query_slug = str(raw.get("query_slug") or path.stem).strip()
    query_domain = str(raw.get("query_domain") or "").strip().lower()
    if not query_slug:
        raise ValueError(f"{path}: query_slug is required")
    if is_excluded_query_domain(query_domain):
        raise ValueError(f"{path}: excluded query_domain is not allowed: {query_domain!r}")
    if not is_allowed_query_domain(query_domain):
        raise ValueError(f"{path}: unsupported query_domain: {query_domain!r}")

    max_items = int(raw.get("maxItems", raw.get("max_items", 100)))
    if max_items < 0:
        raise ValueError(f"{path}: maxItems must be >= 0")

    body_len = int(raw.get("bodyLen", raw.get("body_len", -1)))
    keyword_groups = raw.get("keyword_groups", {}) or {}
    if not isinstance(keyword_groups, dict):
        raise ValueError(f"{path}: keyword_groups must be a mapping")

    return NewsAPIQuerySpec(
        query_slug=query_slug,
        query_domain=query_domain,
        lang=str(raw.get("lang", "eng")),
        date_start=raw.get("dateStart") or raw.get("date_start"),
        date_end=raw.get("dateEnd") or raw.get("date_end"),
        sort_by=str(raw.get("sortBy", raw.get("sort_by", "rel"))),
        sort_by_asc=bool(raw.get("sortByAsc", raw.get("sort_by_asc", False))),
        max_items=max_items,
        data_type=str(raw.get("dataType", raw.get("data_type", "news"))),
        body_len=body_len,
        is_duplicate_filter=str(
            raw.get("isDuplicateFilter", raw.get("is_duplicate_filter", "skipDuplicates"))
        ),
        keyword_groups={
            str(k): [str(v) for v in (values or [])]
            for k, values in keyword_groups.items()
        },
        keywords=raw.get("keywords"),
        complex_query=raw.get("complex_query") or raw.get("complexQuery"),
        keyword_search_mode=str(
            raw.get("keywordSearchMode", raw.get("keyword_search_mode", "phrase"))
        ),
        keywords_loc=str(raw.get("keywordsLoc", raw.get("keywords_loc", "body"))),
        raw=raw,
        path=path,
    )


def render_query_string(spec: NewsAPIQuerySpec) -> str:
    """Return a compact, human-readable representation of a query spec.

    This is used as row-level provenance in extracted annotation CSVs. It is
    intentionally readable rather than an exact SDK object dump.
    """
    if spec.keywords:
        if isinstance(spec.keywords, list):
            return " OR ".join(str(item) for item in spec.keywords)
        return str(spec.keywords)
    if spec.keyword_groups:
        parts: list[str] = []
        for group, values in spec.keyword_groups.items():
            clean_values = [str(value) for value in values if str(value).strip()]
            if clean_values:
                parts.append(f"{group}=({' OR '.join(clean_values)})")
        if parts:
            return " AND ".join(parts)
    if spec.complex_query:
        if isinstance(spec.complex_query, str):
            return spec.complex_query
        return json.dumps(spec.complex_query, sort_keys=True)
    return spec.query_slug


class EventRegistryArticleFetcher:
    """Thin wrapper around the official Event Registry SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        allow_use_of_archive: bool = False,
        verbose_output: bool = False,
        repeat_failed_request_count: int = 2,
        min_delay_between_requests: float = 0.5,
    ) -> None:
        from eventregistry import EventRegistry

        self._er = EventRegistry(
            apiKey=api_key,
            allowUseOfArchive=allow_use_of_archive,
            verboseOutput=verbose_output,
            repeatFailedRequestCount=repeat_failed_request_count,
            minDelayBetweenRequests=min_delay_between_requests,
        )

    def fetch_articles(self, spec: NewsAPIQuerySpec) -> Iterable[dict[str, Any]]:
        from eventregistry import ArticleInfoFlags, ReturnInfo

        query = build_event_registry_query(spec)
        return_info = ReturnInfo(articleInfo=ArticleInfoFlags(bodyLen=spec.body_len))
        iterator = query.execQuery(
            self._er,
            sortBy=spec.sort_by,
            sortByAsc=spec.sort_by_asc,
            returnInfo=return_info,
            maxItems=spec.max_items,
        )
        return iterator


def build_event_registry_query(spec: NewsAPIQuerySpec):
    """Build a ``QueryArticlesIter`` from a query spec."""
    from eventregistry import QueryArticlesIter

    if spec.complex_query:
        return QueryArticlesIter.initWithComplexQuery(spec.complex_query)
    if spec.keyword_groups and not spec.keywords:
        return QueryArticlesIter.initWithComplexQuery(
            _build_complex_query_from_keyword_groups(spec)
        )

    return QueryArticlesIter(
        keywords=_build_keywords(spec),
        lang=spec.lang,
        dateStart=spec.date_start,
        dateEnd=spec.date_end,
        keywordSearchMode=spec.keyword_search_mode,
        keywordsLoc=spec.keywords_loc,
        isDuplicateFilter=spec.is_duplicate_filter,
        dataType=spec.data_type,
    )


def _build_complex_query_from_keyword_groups(spec: NewsAPIQuerySpec) -> dict[str, Any]:
    """Build a nested boolean keyword query for Event Registry complex mode.

    The Event Registry SDK only supports a single operator level for
    ``keywords=QueryItems.*`` in ``QueryArticlesIter``. Nested objects like
    ``AND([..., OR([...])])`` are not JSON-serializable in that code path.
    This helper maps grouped keyword YAML into a proper complex query object.
    """
    from eventregistry import BaseQuery, CombinedQuery, ComplexArticleQuery, QueryItems

    groups = spec.keyword_groups
    required_terms = _unique_terms(
        [*groups.get("all", []), *groups.get("required", []), *groups.get("topics", [])]
    )
    disjunction_groups: list[list[str]] = []
    for key in ("any", "event_nouns", "prediction_triggers", "extra_any"):
        values = _unique_terms(groups.get(key, []))
        if values:
            disjunction_groups.append(values)

    if not required_terms and not disjunction_groups:
        raise ValueError(
            f"{spec.path or spec.query_slug}: provide keywords, complex_query, or keyword_groups"
        )

    base_clause_kwargs: dict[str, Any] = {}
    if spec.lang:
        base_clause_kwargs["lang"] = spec.lang
    if spec.date_start:
        base_clause_kwargs["dateStart"] = spec.date_start
    if spec.date_end:
        base_clause_kwargs["dateEnd"] = spec.date_end
    if spec.keywords_loc and spec.keywords_loc != "body":
        base_clause_kwargs["keywordLoc"] = spec.keywords_loc

    clauses: list[Any] = []
    if base_clause_kwargs:
        # Ensure date/lang restrictions apply regardless of keyword clauses.
        clauses.append(BaseQuery(**base_clause_kwargs))

    for term in required_terms:
        kwargs = dict(base_clause_kwargs)
        kwargs["keyword"] = term
        clauses.append(BaseQuery(**kwargs))

    for values in disjunction_groups:
        kwargs = dict(base_clause_kwargs)
        kwargs["keyword"] = values[0] if len(values) == 1 else QueryItems.OR(values)
        clauses.append(BaseQuery(**kwargs))

    query_obj = clauses[0] if len(clauses) == 1 else CombinedQuery.AND(clauses)
    return ComplexArticleQuery(
        query=query_obj,
        dataType=spec.data_type,
        isDuplicateFilter=spec.is_duplicate_filter,
    ).getQuery()


def _unique_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        term = str(raw).strip()
        if not term or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def _build_keywords(spec: NewsAPIQuerySpec):
    from eventregistry import QueryItems

    if spec.keywords:
        return spec.keywords

    groups = spec.keyword_groups
    and_items: list[Any] = []

    for key in ("all", "required", "topics"):
        and_items.extend(groups.get(key, []))

    for key in ("any", "event_nouns", "prediction_triggers", "extra_any"):
        values = groups.get(key, [])
        if len(values) == 1:
            and_items.append(values[0])
        elif values:
            and_items.append(QueryItems.OR(values))

    if not and_items:
        raise ValueError(
            f"{spec.path or spec.query_slug}: provide keywords, complex_query, or keyword_groups"
        )
    if len(and_items) == 1:
        return and_items[0]
    return QueryItems.AND(and_items)


def fetch_specs_to_cache(
    *,
    specs: list[NewsAPIQuerySpec],
    fetcher: ArticleFetcher,
    cfg: PipelineConfig,
    dry_run: bool = False,
    overwrite: bool = False,
    skip_existing: bool = False,
    cache_label: str | None = None,
    dedupe_existing_cache: bool = True,
    progress_every: int = 25,
    status_logger: Callable[[str], None] | None = None,
) -> list[NewsAPIFetchResult]:
    """Fetch each spec and persist the result set as a cached CSV."""
    log = status_logger or (lambda *_: None)
    seen_new_cache_rows: set[CanonicalArticleKey] = set()
    results: list[NewsAPIFetchResult] = []

    for spec in specs:
        stem = spec.filename_stem
        if cache_label:
            stem = f"{stem}_{cache_label}"
        csv_path = cfg.input_cache_dir / f"{stem}.csv-v1.csv"
        if dry_run:
            results.append(
                NewsAPIFetchResult(
                    spec=spec,
                    csv_path=csv_path,
                    rows_fetched=0,
                    rows_written=0,
                    rows_deduped=0,
                    rows_excluded=0,
                    dry_run=True,
                )
            )
            continue

        if csv_path.exists() and not overwrite:
            if skip_existing:
                log(f"[fetch] skipped existing cache: {csv_path}")
                results.append(
                    NewsAPIFetchResult(
                        spec=spec,
                        csv_path=csv_path,
                        rows_fetched=0,
                        rows_written=0,
                        rows_deduped=0,
                        rows_excluded=0,
                        skipped_existing=True,
                    )
                )
                continue
            raise FileExistsError(
                f"Refusing to overwrite existing fetched cache CSV: {csv_path}. "
                "Pass --overwrite or change query_slug/date window."
            )

        seen_existing = (
            _existing_cache_keys(
                cfg.input_cache_dir,
                exclude={csv_path.resolve()} if overwrite else None,
            )
            if dedupe_existing_cache
            else set()
        )
        seen_existing.update(seen_new_cache_rows)

        rows: list[dict[str, str]] = []
        rows_fetched = 0
        rows_deduped = 0
        rows_excluded = 0
        seen_this_spec: set[CanonicalArticleKey] = set()
        log(
            f"[fetch] starting {spec.query_slug}: maxItems={spec.max_items}, "
            f"date={spec.date_start}..{spec.date_end}, sortBy={spec.sort_by}"
        )

        for article in fetcher.fetch_articles(spec):
            rows_fetched += 1
            row = event_registry_article_to_csv_row(article)
            verdict = evaluate_url(row.get("url"), cfg.excluded)
            if verdict.rejected:
                rows_excluded += 1
                continue

            key = _row_key(row)
            if key in seen_existing or key in seen_this_spec:
                rows_deduped += 1
                continue
            seen_this_spec.add(key)
            rows.append(row)
            if progress_every > 0 and rows_fetched % progress_every == 0:
                log(
                    f"[fetch] progress {spec.query_slug}: "
                    f"fetched={rows_fetched} kept={len(rows)} "
                    f"dedupe={rows_deduped} excluded={rows_excluded}"
                )

        write_newsapi_cache_csv(csv_path, rows, overwrite=overwrite)
        seen_new_cache_rows.update(seen_this_spec)
        results.append(
            NewsAPIFetchResult(
                spec=spec,
                csv_path=csv_path,
                rows_fetched=rows_fetched,
                rows_written=len(rows),
                rows_deduped=rows_deduped,
                rows_excluded=rows_excluded,
            )
        )

    return results


def event_registry_article_to_csv_row(article: dict[str, Any]) -> dict[str, str]:
    source = article.get("source") if isinstance(article.get("source"), dict) else {}
    source_name = (
        source.get("title")
        or source.get("name")
        or source.get("uri")
        or article.get("sourceUri")
        or ""
    )
    source_id = source.get("uri") or article.get("sourceUri")

    body = article.get("body") or article.get("content") or ""
    description = article.get("description") or article.get("summary") or ""
    if not description and body:
        description = _derive_description_from_body(str(body))
    authors = _authors_to_string(article.get("authors"))

    return {
        "source": str({"id": source_id, "name": source_name}),
        "author": authors,
        "title": str(article.get("title") or ""),
        "description": str(description or ""),
        "url": str(article.get("url") or ""),
        "urlToImage": str(article.get("image") or article.get("urlToImage") or ""),
        "publishedAt": str(
            article.get("dateTime")
            or article.get("date")
            or article.get("publishedAt")
            or ""
        ),
        "content": str(body or ""),
    }


def _derive_description_from_body(body: str, *, max_len: int = 280) -> str:
    """Create a short description when provider description/summary is missing.

    Event Registry sometimes returns only ``title`` + ``body`` with empty
    ``description`` and ``authors`` for some sources. We keep the missing
    author blank, but derive a compact description from body text so the cached
    CSV stays useful for quick triage and later filtering.
    """
    if not body:
        return ""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if len(first) <= max_len:
        return first
    return first[: max_len - 1].rstrip() + "…"


def write_newsapi_cache_csv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(EXPECTED_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in EXPECTED_COLUMNS})


def _existing_cache_keys(
    cache_dir: Path,
    *,
    exclude: set[Path] | None = None,
) -> set[CanonicalArticleKey]:
    seen: set[CanonicalArticleKey] = set()
    if not cache_dir.exists():
        return seen
    for csv_path in sorted(cache_dir.glob("news_articles_*.csv*")):
        if exclude and csv_path.resolve() in exclude:
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                seen.add(_row_key({k: v or "" for k, v in row.items()}))
    return seen


def _row_key(row: dict[str, str]) -> CanonicalArticleKey:
    source_meta = parse_source_meta(row.get("source"))
    return CanonicalArticleKey.from_row(
        url=row.get("url"),
        title=row.get("title"),
        source_name=source_meta.get("name") if source_meta else None,
        published_at=row.get("publishedAt"),
    )


def _authors_to_string(authors: Any) -> str:
    if not authors:
        return ""
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        names: list[str] = []
        for item in authors:
            if isinstance(item, dict):
                name = item.get("name") or item.get("uri")
                if name:
                    names.append(str(name))
            elif item:
                names.append(str(item))
        return ", ".join(names)
    return str(authors)


def fetch_plan_lines(specs: list[NewsAPIQuerySpec]) -> list[str]:
    lines: list[str] = []
    for spec in specs:
        lines.append(
            " | ".join(
                [
                    spec.query_slug,
                    f"domain={spec.query_domain}",
                    f"dateStart={spec.date_start}",
                    f"dateEnd={spec.date_end}",
                    f"maxItems={spec.max_items}",
                    f"sortBy={spec.sort_by}",
                    f"dataType={spec.data_type}",
                ]
            )
        )
    return lines


def timestamped_query_slug(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}"
