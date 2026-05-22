"""Schema/data validation + dedupe for the final JSON before write."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import is_allowed_query_domain, is_excluded_query_domain
from .utils import canonicalize_url


REQUIRED_PROCESSED_KEYS: tuple[str, ...] = (
    "Base Sentence",
    "Base Sentence (raw)",
    "Sentence Label",
    "Source Meta Data",
    "URL",
    "Query Domain",
    "article_id",
    "Source_Field",
    "field_order",
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_valid_url(url: Any) -> bool:
    if not isinstance(url, str) or not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def _validate_row(row: dict[str, Any], idx: int, location: str) -> list[str]:
    errs: list[str] = []
    for key in REQUIRED_PROCESSED_KEYS:
        if key not in row:
            errs.append(f"{location}[{idx}] missing key: {key!r}")
    base = row.get("Base Sentence")
    if not isinstance(base, str) or not base.strip():
        errs.append(f"{location}[{idx}] empty/non-string Base Sentence")
    url = row.get("URL")
    if url is not None and not _is_valid_url(url):
        errs.append(f"{location}[{idx}] invalid URL: {url!r}")
    qd = row.get("Query Domain")
    if not isinstance(qd, str):
        errs.append(f"{location}[{idx}] non-string Query Domain")
    elif is_excluded_query_domain(qd):
        errs.append(f"{location}[{idx}] excluded query_domain: {qd!r}")
    elif not is_allowed_query_domain(qd):
        errs.append(f"{location}[{idx}] unsupported query_domain: {qd!r}")
    aid = row.get("article_id")
    if not isinstance(aid, int):
        errs.append(f"{location}[{idx}] article_id must be int")
    fo = row.get("field_order")
    if not isinstance(fo, int):
        errs.append(f"{location}[{idx}] field_order must be int")
    return errs


def _row_dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        canonicalize_url(row.get("URL")) or "",
        (row.get("Base Sentence") or "").strip(),
        (row.get("Source_Field") or "").strip(),
    )


def validate_full_json(payload: dict[str, Any]) -> ValidationReport:
    """Validate every section. Strict-mode: drop bad rows and report errors."""
    report = ValidationReport()

    # Top-level keys
    for key in ("query_meta", "llm_meta", "counts", "raw_articles", "processed_sentences", "potential_predictions"):
        if key not in payload:
            report.errors.append(f"missing top-level key: {key!r}")

    # processed_sentences
    seen: set[tuple[str, str, str]] = set()
    cleaned_sentences: list[dict[str, Any]] = []
    for idx, row in enumerate(payload.get("processed_sentences", [])):
        errs = _validate_row(row, idx, "processed_sentences")
        if errs:
            report.dropped_rows += 1
            report.warnings.extend(errs)
            continue
        key = _row_dedupe_key(row)
        if key in seen:
            report.dropped_rows += 1
            report.warnings.append(f"processed_sentences[{idx}] duplicate row dropped")
            continue
        seen.add(key)
        cleaned_sentences.append(row)
    payload["processed_sentences"] = cleaned_sentences

    # potential_predictions (a strict subset of processed_sentences with label=1)
    seen_pred: set[tuple[str, str, str]] = set()
    cleaned_pred: list[dict[str, Any]] = []
    for idx, row in enumerate(payload.get("potential_predictions", [])):
        errs = _validate_row(row, idx, "potential_predictions")
        if errs:
            report.dropped_rows += 1
            report.warnings.extend(errs)
            continue
        if row.get("Sentence Label") != 1 or row.get("prediction_visible") != 1:
            report.warnings.append(
                f"potential_predictions[{idx}] has Sentence Label/prediction_visible != 1"
            )
            continue
        key = _row_dedupe_key(row)
        if key in seen_pred:
            report.dropped_rows += 1
            report.warnings.append(
                f"potential_predictions[{idx}] duplicate row dropped"
            )
            continue
        seen_pred.add(key)
        cleaned_pred.append(row)
    payload["potential_predictions"] = cleaned_pred

    # Refresh counts.
    counts = payload.get("counts") or {}
    counts["num_raw_articles"] = len(payload.get("raw_articles", []))
    counts["num_sentences"] = len(payload.get("processed_sentences", []))
    counts["num_potential_predictions"] = len(payload.get("potential_predictions", []))
    payload["counts"] = counts

    return report
