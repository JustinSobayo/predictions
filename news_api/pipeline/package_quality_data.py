"""Package annotation-ready prediction CSVs with their source artifacts.

This utility creates a reviewer-friendly folder layout without changing the
original pipeline outputs:

    quality_data/
      policy_predictions/
        policy_1/
          <cached NewsAPI CSV>
          <full pipeline JSON>
          <human annotation CSV, <= --max-rows-per-file rows>

Each packaged annotation CSV gets provenance columns such as ``query_string``,
``query_slug``, ``source_cache_file``, and ``quality_batch_id`` so the files can
be recombined later with a plain CSV concatenation.

Example:
    python -m pipeline.package_quality_data \
      --predictions-csv 'annotators(1)/news_articles_example_predictions-v7.csv' \
      --full-json 'annotators(1)/news_articles_example_full-v7.json' \
      --cached-csv 'input_csvs/newsapi_cache/news_articles_example.csv-v1.csv' \
      --domain policy \
      --max-rows-per-file 300
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .newsapi_ai_client import NewsAPIQuerySpec, load_query_specs, render_query_string


PROVENANCE_COLUMNS: tuple[str, ...] = (
    "query_string",
    "query_slug",
    "source_cache_file",
    "quality_batch_id",
)


@dataclass
class QualityBatch:
    batch_id: str
    folder: str
    annotation_csv: str
    full_json: str
    cached_csv: str
    row_count: int
    article_count: int
    article_ids: list[str] = field(default_factory=list)


@dataclass
class QualityPackageSummary:
    domain: str
    package_dir: str
    max_rows_per_file: int
    query_string: str
    query_slug: str
    source_predictions_csv: str
    source_full_json: str
    source_cached_csv: str
    total_rows: int
    total_articles: int
    batch_count: int
    batches: list[QualityBatch]
    manifest_path: str


class QualityPackageError(ValueError):
    """Raised when a quality data package cannot be built safely."""


def _csv_stem(path: Path) -> str:
    name = path.name
    for suffix in (".csv-v1.csv", ".csv-v2.csv", ".csv-v3.csv", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _predictions_stem(path: Path) -> str:
    return path.name[:-4] if path.name.endswith(".csv") else path.stem


def _infer_query_slug_from_cache(cached_csv: Path) -> str:
    stem = _csv_stem(cached_csv)
    if stem.startswith("news_articles_"):
        stem = stem[len("news_articles_"):]
    marker = "_202"
    if marker in stem:
        stem = stem.split(marker, 1)[0]
    return stem or cached_csv.stem


def _find_query_spec(cached_csv: Path, queries_dir: Path) -> NewsAPIQuerySpec | None:
    if not queries_dir.exists():
        return None
    try:
        specs = load_query_specs(queries_dir)
    except Exception:
        return None
    cached_stem = _csv_stem(cached_csv)
    inferred_slug = _infer_query_slug_from_cache(cached_csv)
    for spec in specs:
        if cached_stem == spec.filename_stem or inferred_slug == spec.query_slug:
            return spec
    return None


def _load_json_meta(full_json: Path) -> dict[str, Any]:
    if not full_json.exists():
        raise QualityPackageError(f"Full JSON not found: {full_json}")
    with full_json.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    query_meta = payload.get("query_meta") or {}
    if not isinstance(query_meta, dict):
        return {}
    return query_meta


def _resolve_provenance(
    *,
    full_json: Path,
    cached_csv: Path,
    queries_dir: Path,
    query_string: str | None,
    query_slug: str | None,
    domain: str | None,
) -> tuple[str, str, str]:
    meta = _load_json_meta(full_json)
    spec = _find_query_spec(cached_csv, queries_dir)
    fallback_query = str(meta.get("query") or _infer_query_slug_from_cache(cached_csv))
    looks_like_cache_name = fallback_query.startswith("news_articles_")

    resolved_query_string = query_string or fallback_query
    if spec is not None and (query_string is None and looks_like_cache_name):
        resolved_query_string = render_query_string(spec)

    resolved_query_slug = (
        query_slug
        or str(meta.get("newsapi_query_slug") or meta.get("query_slug") or "").strip()
        or (spec.query_slug if spec is not None else _infer_query_slug_from_cache(cached_csv))
    )
    resolved_domain = (
        domain
        or (spec.query_domain if spec is not None else None)
        or str(meta.get("query_domain") or "misc-general")
    )
    return resolved_query_string, resolved_query_slug, resolved_domain


def _output_header(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise QualityPackageError("Predictions CSV has no header")
    required = {"article_id", "URL"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise QualityPackageError(f"Predictions CSV missing required columns: {missing}")
    header = list(fieldnames)
    for col in PROVENANCE_COLUMNS:
        if col not in header:
            header.append(col)
    return header


def _next_batch_index(package_dir: Path, domain: str) -> int:
    highest = 0
    prefix = f"{domain}_"
    for child in package_dir.iterdir() if package_dir.exists() else []:
        if not child.is_dir() or not child.name.startswith(prefix):
            continue
        suffix = child.name[len(prefix):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def _package_dir_name(domain: str) -> str:
    """Return the reviewer-facing package directory name for a domain."""
    if domain == "sport":
        return "sports_predictions"
    if domain == "misc" or domain.startswith("misc-"):
        return "misc_predictions"
    return f"{domain}_predictions"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        raise QualityPackageError(f"Refusing to overwrite existing file: {dst}")
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_manifest(package_dir: Path, summary: QualityPackageSummary) -> Path:
    manifest_path = package_dir / "quality_manifest.json"
    existing: dict[str, Any] = {"packages": []}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            existing = loaded
            existing.setdefault("packages", [])
    existing["packages"].append(asdict(summary))
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(existing, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return manifest_path


def package_quality_data(
    *,
    predictions_csv: Path,
    full_json: Path,
    cached_csv: Path,
    out_root: Path = REPO_ROOT / "quality_data",
    domain: str | None = None,
    max_rows_per_file: int = 300,
    start_index: int | None = None,
    query_string: str | None = None,
    query_slug: str | None = None,
    queries_dir: Path = REPO_ROOT / "pipeline" / "queries",
) -> QualityPackageSummary:
    """Create quality-data folders and article-preserving annotation CSVs."""
    if max_rows_per_file <= 0:
        raise QualityPackageError("--max-rows-per-file must be greater than 0")
    for path, label in (
        (predictions_csv, "Predictions CSV"),
        (full_json, "Full JSON"),
        (cached_csv, "Cached CSV"),
    ):
        if not path.exists():
            raise QualityPackageError(f"{label} not found: {path}")

    resolved_query, resolved_slug, resolved_domain = _resolve_provenance(
        full_json=full_json,
        cached_csv=cached_csv,
        queries_dir=queries_dir,
        query_string=query_string,
        query_slug=query_slug,
        domain=domain,
    )
    package_dir = out_root / _package_dir_name(resolved_domain)
    package_dir.mkdir(parents=True, exist_ok=True)
    next_index = start_index or _next_batch_index(package_dir, resolved_domain)
    header: list[str]
    batches: list[QualityBatch] = []
    seen_articles: set[str] = set()
    total_rows = 0

    current_rows: list[dict[str, str]] = []
    current_article_ids: list[str] = []
    current_article_id: str | None = None
    current_article_rows: list[dict[str, str]] = []
    closed_articles: set[str] = set()

    def flush_batch() -> None:
        nonlocal current_rows, current_article_ids, next_index
        if not current_rows:
            return
        batch_id = f"{resolved_domain}_{next_index}"
        batch_dir = package_dir / batch_id
        if batch_dir.exists():
            raise QualityPackageError(f"Batch folder already exists: {batch_dir}")
        batch_dir.mkdir(parents=True)

        cached_dst = batch_dir / cached_csv.name
        json_dst = batch_dir / full_json.name
        annotation_dst = batch_dir / f"{_predictions_stem(predictions_csv)}_{batch_id}_human_annotation.csv"
        _link_or_copy(cached_csv, cached_dst)
        _link_or_copy(full_json, json_dst)

        rows_to_write: list[dict[str, str]] = []
        for row in current_rows:
            enriched = dict(row)
            enriched["query_string"] = resolved_query
            enriched["query_slug"] = resolved_slug
            enriched["source_cache_file"] = cached_csv.name
            enriched["quality_batch_id"] = batch_id
            rows_to_write.append(enriched)

        with annotation_dst.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows_to_write)

        batches.append(
            QualityBatch(
                batch_id=batch_id,
                folder=str(batch_dir),
                annotation_csv=str(annotation_dst),
                full_json=str(json_dst),
                cached_csv=str(cached_dst),
                row_count=len(rows_to_write),
                article_count=len(current_article_ids),
                article_ids=list(current_article_ids),
            )
        )
        next_index += 1
        current_rows = []
        current_article_ids = []

    def add_article(article_id: str, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        if current_rows and len(current_rows) + len(rows) > max_rows_per_file:
            flush_batch()
        current_rows.extend(rows)
        current_article_ids.append(article_id)
        if len(rows) > max_rows_per_file:
            print(
                "[quality] warning: "
                f"article_id={article_id} has {len(rows)} rows, exceeding "
                f"--max-rows-per-file={max_rows_per_file}; keeping article intact",
                file=sys.stderr,
            )

    with predictions_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        header = _output_header(reader.fieldnames)
        for row in reader:
            total_rows += 1
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                raise QualityPackageError(
                    f"Row {total_rows} has empty article_id; cannot package safely"
                )
            if current_article_id is None:
                current_article_id = article_id
                seen_articles.add(article_id)
            elif article_id != current_article_id:
                closed_articles.add(current_article_id)
                add_article(current_article_id, current_article_rows)
                current_article_rows = []
                current_article_id = article_id
                if article_id in closed_articles:
                    raise QualityPackageError(
                        "Predictions CSV is not grouped by article_id; "
                        f"article_id={article_id!r} reappeared after another article."
                    )
                seen_articles.add(article_id)
            current_article_rows.append(row)

    if current_article_id is not None:
        add_article(current_article_id, current_article_rows)
    flush_batch()

    summary = QualityPackageSummary(
        domain=resolved_domain,
        package_dir=str(package_dir),
        max_rows_per_file=max_rows_per_file,
        query_string=resolved_query,
        query_slug=resolved_slug,
        source_predictions_csv=str(predictions_csv),
        source_full_json=str(full_json),
        source_cached_csv=str(cached_csv),
        total_rows=total_rows,
        total_articles=len(seen_articles),
        batch_count=len(batches),
        batches=batches,
        manifest_path=str(package_dir / "quality_manifest.json"),
    )
    manifest_path = _write_manifest(package_dir, summary)
    summary.manifest_path = str(manifest_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--cached-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "quality_data")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=300)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--query-string", default=None)
    parser.add_argument("--query-slug", default=None)
    parser.add_argument("--queries-dir", type=Path, default=REPO_ROOT / "pipeline" / "queries")
    args = parser.parse_args(argv)

    try:
        summary = package_quality_data(
            predictions_csv=args.predictions_csv,
            full_json=args.full_json,
            cached_csv=args.cached_csv,
            out_root=args.out_root,
            domain=args.domain,
            max_rows_per_file=args.max_rows_per_file,
            start_index=args.start_index,
            query_string=args.query_string,
            query_slug=args.query_slug,
            queries_dir=args.queries_dir,
        )
    except QualityPackageError as exc:
        print(f"[quality] ERROR: {exc}", file=sys.stderr)
        return 2

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
