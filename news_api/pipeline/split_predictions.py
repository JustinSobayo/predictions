"""Split large prediction CSVs into article-preserving batches.

The pipeline's predictions CSV can become too large for spreadsheet-style
annotation tools. This utility writes smaller CSVs while guaranteeing that all
rows for a given ``article_id`` stay in the same output file.

Example:
    python -m pipeline.split_predictions \
      'annotators(1)/news_articles_example_predictions-v1.csv' \
      --out-dir 'annotators(1)/news_articles_example_predictions-v1_batches' \
      --articles-per-file 300 \
      --max-rows-per-file 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BatchSummary:
    path: str
    article_count: int
    row_count: int
    min_article_id: str | None
    max_article_id: str | None
    article_ids: list[str] = field(default_factory=list)


@dataclass
class SplitSummary:
    input_csv: str
    output_dir: str
    input_rows: int
    output_rows: int
    input_articles: int
    output_articles: int
    batch_count: int
    manifest_path: str | None
    batches: list[BatchSummary]


class PredictionSplitError(ValueError):
    """Raised when the input CSV cannot be split safely."""


def _input_stem(path: Path) -> str:
    return path.name[:-4] if path.name.endswith(".csv") else path.stem


def _article_sort_key(article_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(article_id))
    except ValueError:
        return (1, article_id)


def _minmax_article_ids(article_ids: list[str]) -> tuple[str | None, str | None]:
    if not article_ids:
        return None, None
    ordered = sorted(article_ids, key=_article_sort_key)
    return ordered[0], ordered[-1]


def _batch_path(out_dir: Path, input_stem: str, batch_index: int) -> Path:
    return out_dir / f"{input_stem}_batch_{batch_index:04d}.csv"


def _validate_header(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise PredictionSplitError("Input CSV has no header")
    required = {"article_id", "URL"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise PredictionSplitError(f"Input CSV missing required columns: {missing}")
    if "Base Sentence" not in fieldnames and "Base Sentence (raw)" not in fieldnames:
        raise PredictionSplitError(
            "Input CSV must include Base Sentence or Base Sentence (raw)"
        )
    return list(fieldnames)


def split_predictions_csv(
    *,
    input_csv: Path,
    out_dir: Path | None = None,
    articles_per_file: int = 300,
    max_rows_per_file: int | None = None,
    write_manifest: bool = True,
) -> SplitSummary:
    """Split a predictions CSV into smaller article-preserving CSV batches.

    The generated predictions CSV is ordered by article. To stay memory-safe on
    million-row files, this splitter streams that order and raises if an
    ``article_id`` reappears after another article has started.
    """
    if articles_per_file <= 0:
        raise PredictionSplitError("--articles-per-file must be greater than 0")
    if max_rows_per_file is not None and max_rows_per_file <= 0:
        raise PredictionSplitError("--max-rows-per-file must be greater than 0")
    if not input_csv.exists():
        raise PredictionSplitError(f"Input CSV not found: {input_csv}")

    input_stem = _input_stem(input_csv)
    target_dir = out_dir or (input_csv.parent / f"{input_stem}_batches")
    target_dir.mkdir(parents=True, exist_ok=True)

    batches: list[BatchSummary] = []
    seen_articles: set[str] = set()
    closed_articles: set[str] = set()
    current_article_id: str | None = None
    current_article_rows: list[dict[str, str]] = []
    batch_rows: list[dict[str, str]] = []
    batch_article_ids: list[str] = []
    input_rows = 0
    batch_index = 0
    header: list[str]

    def flush_batch() -> None:
        nonlocal batch_index, batch_rows, batch_article_ids
        if not batch_rows:
            return
        batch_index += 1
        path = _batch_path(target_dir, input_stem, batch_index)
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=header)
            writer.writeheader()
            writer.writerows(batch_rows)
        min_id, max_id = _minmax_article_ids(batch_article_ids)
        batches.append(
            BatchSummary(
                path=str(path),
                article_count=len(batch_article_ids),
                row_count=len(batch_rows),
                min_article_id=min_id,
                max_article_id=max_id,
                article_ids=list(batch_article_ids),
            )
        )
        batch_rows = []
        batch_article_ids = []

    def add_article_to_batch(article_id: str, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        over_article_limit = len(batch_article_ids) >= articles_per_file
        over_row_limit = (
            max_rows_per_file is not None
            and batch_rows
            and len(batch_rows) + len(rows) > max_rows_per_file
        )
        if over_article_limit or over_row_limit:
            flush_batch()
        batch_rows.extend(rows)
        batch_article_ids.append(article_id)
        if max_rows_per_file is not None and len(rows) > max_rows_per_file:
            print(
                "[split] warning: "
                f"article_id={article_id} has {len(rows)} rows, "
                f"exceeding --max-rows-per-file={max_rows_per_file}; "
                "emitting article intact",
                file=sys.stderr,
            )

    with input_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        header = _validate_header(reader.fieldnames)
        for row in reader:
            input_rows += 1
            article_id = (row.get("article_id") or "").strip()
            if not article_id:
                raise PredictionSplitError(
                    f"Row {input_rows} has empty article_id; cannot split safely"
                )

            if current_article_id is None:
                current_article_id = article_id
                seen_articles.add(article_id)
            elif article_id != current_article_id:
                closed_articles.add(current_article_id)
                add_article_to_batch(current_article_id, current_article_rows)
                current_article_id = article_id
                current_article_rows = []
                if article_id in closed_articles:
                    raise PredictionSplitError(
                        "Input is not grouped by article_id; "
                        f"article_id={article_id!r} reappeared after another article. "
                        "Refusing to split because that could separate article context."
                    )
                seen_articles.add(article_id)

            current_article_rows.append(row)

    if current_article_id is not None:
        add_article_to_batch(current_article_id, current_article_rows)
    flush_batch()

    output_rows = sum(batch.row_count for batch in batches)
    output_articles = sum(batch.article_count for batch in batches)
    if output_rows != input_rows:
        raise PredictionSplitError(
            f"Sanity check failed: input rows={input_rows}, output rows={output_rows}"
        )
    if output_articles != len(seen_articles):
        raise PredictionSplitError(
            "Sanity check failed: "
            f"input articles={len(seen_articles)}, output articles={output_articles}"
        )

    manifest_path: Path | None = None
    summary = SplitSummary(
        input_csv=str(input_csv),
        output_dir=str(target_dir),
        input_rows=input_rows,
        output_rows=output_rows,
        input_articles=len(seen_articles),
        output_articles=output_articles,
        batch_count=len(batches),
        manifest_path=None,
        batches=batches,
    )
    if write_manifest:
        manifest_path = target_dir / f"{input_stem}_manifest.json"
        summary.manifest_path = str(manifest_path)
        manifest_path.write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="Predictions CSV to split.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for split CSVs. Default: <input_stem>_batches beside input.",
    )
    parser.add_argument(
        "--articles-per-file",
        type=int,
        default=300,
        help="Maximum distinct article_id groups per output file.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help=(
            "Optional row budget per output file. Whole articles are never split; "
            "a single oversized article is emitted alone."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write the manifest JSON sanity report.",
    )
    args = parser.parse_args(argv)

    try:
        summary = split_predictions_csv(
            input_csv=args.input_csv,
            out_dir=args.out_dir,
            articles_per_file=args.articles_per_file,
            max_rows_per_file=args.max_rows_per_file,
            write_manifest=not args.no_manifest,
        )
    except PredictionSplitError as exc:
        print(f"[split] ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[split] input_rows={summary.input_rows}")
    print(f"[split] output_rows={summary.output_rows}")
    print(f"[split] articles={summary.input_articles}")
    print(f"[split] batches={summary.batch_count}")
    print(f"[split] out_dir={summary.output_dir}")
    if summary.manifest_path:
        print(f"[split] manifest={summary.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
