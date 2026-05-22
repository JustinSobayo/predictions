"""Output writers: JSON + predictions CSV with overwrite protection."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transform import PREDICTIONS_CSV_COLUMNS
from .utils import write_json


@dataclass
class WriteOutcome:
    json_path: Path
    csv_path: Path
    overwrote: bool


def output_paths(
    *,
    output_dir: Path,
    base_stem: str,
    version: int,
) -> tuple[Path, Path]:
    json_path = output_dir / f"{base_stem}_full-v{version}.json"
    csv_path = output_dir / f"{base_stem}_predictions-v{version}.csv"
    return json_path, csv_path


def next_available_version(
    *,
    output_dir: Path,
    base_stem: str,
    start: int = 1,
) -> int:
    """Return the lowest ``v<N>`` for which both target files are free."""
    v = start
    while True:
        j, c = output_paths(output_dir=output_dir, base_stem=base_stem, version=v)
        if not j.exists() and not c.exists():
            return v
        v += 1


def write_outputs(
    *,
    output_dir: Path,
    base_stem: str,
    version: int,
    full_payload: dict[str, Any],
    predictions_rows: list[dict[str, Any]],
    overwrite: bool = False,
) -> WriteOutcome:
    """Write the JSON + CSV pair. Refuse to overwrite unless ``overwrite=True``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_paths(
        output_dir=output_dir, base_stem=base_stem, version=version
    )
    overwrote = False
    for path in (json_path, csv_path):
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {path}. "
                    "Bump --version or pass --overwrite."
                )
            overwrote = True

    write_json(json_path, full_payload)

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(PREDICTIONS_CSV_COLUMNS))
        writer.writeheader()
        for row in predictions_rows:
            writer.writerow({col: row.get(col, "") for col in PREDICTIONS_CSV_COLUMNS})

    return WriteOutcome(json_path=json_path, csv_path=csv_path, overwrote=overwrote)
