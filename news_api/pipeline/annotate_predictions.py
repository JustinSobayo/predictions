"""Annotate prediction CSV rows from the command line.

Example:
    python -m pipeline.annotate_predictions \
      quality_data/policy_predictions/policy_2/example_human_annotation.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO


VALID_ANNOTATIONS = {"0", "1"}


@dataclass
class AnnotationConfig:
    path: Path
    text_column: str = "Base Sentence"
    annotation_column: str = "Human Annotation"
    reasoning_column: str = "Human Reasoning"
    start_row: int | None = None
    include_annotated: bool = False
    limit: int | None = None
    dry_run: bool = False
    backup: bool = True
    encoding: str = "utf-8-sig"


@dataclass
class AnnotationCounts:
    total_rows: int
    annotated_rows: int
    unannotated_rows: int
    invalid_rows: list[tuple[int, str]] = field(default_factory=list)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_rows)


@dataclass
class UndoEntry:
    row_index: int
    old_annotation: str
    new_annotation: str


class AnnotationCsvError(ValueError):
    """Raised when an annotation CSV cannot be processed safely."""


def _annotation_value(row: dict[str, str], column: str) -> str:
    return (row.get(column) or "").strip()


def _is_unannotated(row: dict[str, str], column: str) -> bool:
    return _annotation_value(row, column) == ""


def load_csv(path: Path, encoding: str = "utf-8-sig") -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV while preserving its header order."""
    if not path.exists():
        raise AnnotationCsvError(f"CSV file not found: {path}")
    if not path.is_file():
        raise AnnotationCsvError(f"Path is not a file: {path}")

    with path.open("r", encoding=encoding, newline="") as fp:
        reader = csv.DictReader(fp, restval="")
        if not reader.fieldnames:
            raise AnnotationCsvError("CSV has no header row")
        fieldnames = list(reader.fieldnames)
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=1):
            if None in row:
                raise AnnotationCsvError(
                    f"Row {row_number} has more fields than the header; "
                    "refusing to rewrite a malformed CSV"
                )
            rows.append({name: row.get(name, "") or "" for name in fieldnames})
    return fieldnames, rows


def validate_config(config: AnnotationConfig, fieldnames: list[str]) -> None:
    """Validate CLI options and required columns."""
    if config.start_row is not None and config.start_row <= 0:
        raise AnnotationCsvError("--start-row must be a 1-based row number")
    if config.limit is not None and config.limit <= 0:
        raise AnnotationCsvError("--limit must be greater than 0")

    missing = [
        column
        for column in (config.text_column, config.annotation_column)
        if column not in fieldnames
    ]
    if missing:
        raise AnnotationCsvError(f"CSV missing required columns: {missing}")


def count_annotations(rows: list[dict[str, str]], annotation_column: str) -> AnnotationCounts:
    annotated = 0
    unannotated = 0
    invalid: list[tuple[int, str]] = []
    for index, row in enumerate(rows, start=1):
        value = _annotation_value(row, annotation_column)
        if value == "":
            unannotated += 1
        elif value in VALID_ANNOTATIONS:
            annotated += 1
        else:
            invalid.append((index, value))
    return AnnotationCounts(
        total_rows=len(rows),
        annotated_rows=annotated,
        unannotated_rows=unannotated,
        invalid_rows=invalid,
    )


def find_next_row(
    rows: list[dict[str, str]],
    annotation_column: str,
    start_index: int,
    skipped: set[int] | None = None,
    include_annotated: bool = False,
) -> int | None:
    """Return the next row index to show, or None when no row is eligible."""
    skipped = skipped or set()
    for index in range(max(start_index, 0), len(rows)):
        if index in skipped:
            continue
        if include_annotated or _is_unannotated(rows[index], annotation_column):
            return index
    return None


def save_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    encoding: str = "utf-8-sig",
) -> None:
    """Write rows to a temp file and atomically replace the target CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fp:
            temp_path = Path(fp.name)
            writer = csv.DictWriter(
                fp,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def create_backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak-{stamp}")
    counter = 2
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.bak-{stamp}-{counter}")
        counter += 1
    shutil.copy2(path, backup_path)
    return backup_path


def _shorten(value: str, width: int) -> str:
    value = " ".join(value.split())
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3].rstrip() + "..."


def format_row(
    row: dict[str, str],
    *,
    row_index: int,
    total_rows: int,
    remaining: int,
    text_column: str,
    width: int | None = None,
) -> str:
    """Build the terminal display for one annotation row."""
    width = width or shutil.get_terminal_size((100, 24)).columns
    wrap_width = max(40, min(width, 120))
    lines = [
        "",
        f"[row {row_index + 1}/{total_rows}] remaining={remaining}",
    ]

    article_bits = []
    for column in ("article_id", "field_order"):
        value = row.get(column)
        if value:
            article_bits.append(f"{column}={value}")
    if article_bits:
        lines[-1] += " " + " ".join(article_bits)

    if row.get("Sentence Label") or row.get("prediction_visible"):
        label = row.get("Sentence Label") or "?"
        visible = row.get("prediction_visible") or "?"
        lines.append(f"Model label: {label} | visible: {visible}")

    for column in ("title", "URL"):
        value = row.get(column)
        if value:
            lines.append(f"{column}: {_shorten(value, wrap_width)}")

    text = row.get(text_column, "") or ""
    lines.extend(
        [
            "",
            f"{text_column}:",
            textwrap.fill(text, width=wrap_width) if text else "(empty)",
            "",
            "Keys: 1=yes  0=no  s=skip  r=reason  u=undo  q=quit  ?=help",
        ]
    )
    return "\n".join(lines)


def read_single_key(stdin: TextIO | None = None) -> str:
    """Read one key from stdin, with a line-based fallback for scripts/tests."""
    stdin = stdin or sys.stdin
    if not hasattr(stdin, "isatty") or not stdin.isatty():
        return stdin.readline().strip()[:1]

    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()

    import termios
    import tty

    fd = stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _print_help(output: TextIO) -> None:
    output.write(
        "\n"
        "Commands:\n"
        "  1  mark candidate as a prediction\n"
        "  0  mark candidate as not a prediction\n"
        "  s  skip this row for the current session\n"
        "  r  edit Human Reasoning for this row\n"
        "  u  undo the last annotation made in this session\n"
        "  q  save and quit\n"
        "  ?  show this help\n"
    )


def run_session(
    config: AnnotationConfig,
    *,
    key_reader: Callable[[], str] | None = None,
    reason_reader: Callable[[str], str] | None = None,
    output: TextIO | None = None,
    error: TextIO | None = None,
) -> int:
    """Run an annotation session and return a process-style exit code."""
    output = output or sys.stdout
    error = error or sys.stderr
    key_reader = key_reader or read_single_key

    fieldnames, rows = load_csv(config.path, config.encoding)
    validate_config(config, fieldnames)

    skipped: set[int] = set()
    undo_stack: list[UndoEntry] = []
    backup_path: Path | None = None
    annotated_this_session = 0
    reasoning_edits = 0

    def save_changes() -> bool:
        nonlocal backup_path
        if config.dry_run:
            return True
        try:
            if config.backup and backup_path is None:
                backup_path = create_backup(config.path)
                output.write(f"[annotate] backup={backup_path}\n")
            save_csv_atomic(config.path, fieldnames, rows, config.encoding)
            return True
        except Exception as exc:
            print(f"[annotate] ERROR: save failed: {exc}", file=error)
            return False

    counts = count_annotations(rows, config.annotation_column)
    output.write(f"[annotate] loaded={config.path}\n")
    output.write(
        "[annotate] "
        f"rows={counts.total_rows} annotated={counts.annotated_rows} "
        f"remaining={counts.unannotated_rows} invalid={counts.invalid_count}\n"
    )
    if counts.invalid_rows:
        preview = ", ".join(
            f"row {row_number}={value!r}" for row_number, value in counts.invalid_rows[:5]
        )
        suffix = "" if len(counts.invalid_rows) <= 5 else " ..."
        output.write(f"[annotate] warning: invalid annotation values: {preview}{suffix}\n")

    start_index = (config.start_row - 1) if config.start_row else 0
    current_index = find_next_row(
        rows,
        config.annotation_column,
        start_index,
        skipped,
        config.include_annotated,
    )

    try:
        while True:
            if current_index is None:
                if not undo_stack:
                    break
                output.write("\n[annotate] no eligible rows remain. Keys: u=undo  q=quit  ?=help\n")
                output.write("Choice: ")
                output.flush()
                key = (key_reader() or "").lower()
                output.write("\n")
                if key == "u":
                    entry = undo_stack.pop()
                    rows[entry.row_index][config.annotation_column] = entry.old_annotation
                    if save_changes():
                        annotated_this_session = max(0, annotated_this_session - 1)
                        current_index = entry.row_index
                        output.write(f"[annotate] undid row {entry.row_index + 1}\n")
                    else:
                        rows[entry.row_index][config.annotation_column] = entry.new_annotation
                        undo_stack.append(entry)
                    continue
                if key == "q":
                    output.write("[annotate] quit requested\n")
                    break
                if key == "?":
                    _print_help(output)
                    continue
                output.write("[annotate] unknown key; press ? for help\n")
                continue

            if config.limit is not None and annotated_this_session >= config.limit:
                output.write(f"[annotate] reached --limit={config.limit}\n")
                break

            counts = count_annotations(rows, config.annotation_column)
            output.write(
                format_row(
                    rows[current_index],
                    row_index=current_index,
                    total_rows=len(rows),
                    remaining=counts.unannotated_rows,
                    text_column=config.text_column,
                )
            )
            output.write("\nChoice: ")
            output.flush()
            key = (key_reader() or "").lower()
            output.write("\n")

            if key in VALID_ANNOTATIONS:
                row = rows[current_index]
                old_value = row.get(config.annotation_column, "") or ""
                if old_value.strip() and not config.include_annotated:
                    output.write(
                        "[annotate] row already has an annotation; "
                        "use --include-annotated to overwrite\n"
                    )
                    current_index = find_next_row(
                        rows,
                        config.annotation_column,
                        current_index + 1,
                        skipped,
                        config.include_annotated,
                    )
                    continue
                row[config.annotation_column] = key
                undo_stack.append(
                    UndoEntry(
                        row_index=current_index,
                        old_annotation=old_value,
                        new_annotation=key,
                    )
                )
                if save_changes():
                    annotated_this_session += 1
                    current_index = find_next_row(
                        rows,
                        config.annotation_column,
                        current_index + 1,
                        skipped,
                        config.include_annotated,
                    )
                else:
                    row[config.annotation_column] = old_value
                    undo_stack.pop()
                continue

            if key == "s":
                skipped.add(current_index)
                output.write(f"[annotate] skipped row {current_index + 1}\n")
                current_index = find_next_row(
                    rows,
                    config.annotation_column,
                    current_index + 1,
                    skipped,
                    config.include_annotated,
                )
                continue

            if key == "r":
                if config.reasoning_column not in fieldnames:
                    output.write(
                        f"[annotate] reasoning column not found: {config.reasoning_column!r}\n"
                    )
                    continue
                prompt = "Reasoning: "
                if reason_reader is not None:
                    reason = reason_reader(prompt)
                else:
                    output.write(prompt)
                    output.flush()
                    reason = sys.stdin.readline().rstrip("\n")
                old_reason = rows[current_index].get(config.reasoning_column, "") or ""
                rows[current_index][config.reasoning_column] = reason
                if save_changes():
                    reasoning_edits += 1
                    output.write(f"[annotate] updated reasoning for row {current_index + 1}\n")
                else:
                    rows[current_index][config.reasoning_column] = old_reason
                continue

            if key == "u":
                if not undo_stack:
                    output.write("[annotate] nothing to undo\n")
                    continue
                entry = undo_stack.pop()
                rows[entry.row_index][config.annotation_column] = entry.old_annotation
                if save_changes():
                    annotated_this_session = max(0, annotated_this_session - 1)
                    current_index = entry.row_index
                    output.write(f"[annotate] undid row {entry.row_index + 1}\n")
                else:
                    rows[entry.row_index][config.annotation_column] = entry.new_annotation
                    undo_stack.append(entry)
                continue

            if key == "q":
                output.write("[annotate] quit requested\n")
                break

            if key == "?":
                _print_help(output)
                continue

            output.write("[annotate] unknown key; press ? for help\n")
    except KeyboardInterrupt:
        output.write("\n[annotate] interrupted; completed saves are already on disk\n")
        return 130

    counts = count_annotations(rows, config.annotation_column)
    output.write(
        "[annotate] summary "
        f"annotated_this_session={annotated_this_session} "
        f"reasoning_edits={reasoning_edits} "
        f"remaining={counts.unannotated_rows} "
        f"dry_run={config.dry_run}\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Prediction CSV to annotate.")
    parser.add_argument(
        "--text-column",
        default="Base Sentence",
        help="Column shown as the main candidate text.",
    )
    parser.add_argument(
        "--annotation-column",
        default="Human Annotation",
        help="Column updated with 0 or 1.",
    )
    parser.add_argument(
        "--reasoning-column",
        default="Human Reasoning",
        help="Column updated when pressing r.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help="1-based data row to start at. Default: first empty annotation.",
    )
    parser.add_argument(
        "--include-annotated",
        action="store_true",
        help="Visit rows even when an annotation already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N new annotations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate changes without writing to disk.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Do not create a timestamped backup before the first write.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding. Default: utf-8-sig.",
    )
    parser.set_defaults(backup=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AnnotationConfig(
        path=args.file,
        text_column=args.text_column,
        annotation_column=args.annotation_column,
        reasoning_column=args.reasoning_column,
        start_row=args.start_row,
        include_annotated=args.include_annotated,
        limit=args.limit,
        dry_run=args.dry_run,
        backup=args.backup,
        encoding=args.encoding,
    )

    try:
        return run_session(config)
    except AnnotationCsvError as exc:
        print(f"[annotate] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
