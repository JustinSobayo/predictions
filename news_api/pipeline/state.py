"""Per-article and per-window resume state, persisted as JSONL."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from .utils import now_utc_iso


ARTICLE_STATUS_VALUES = {
    "pending",
    "fetched",
    "segmented",
    "llm_done",
    "skipped",
    "failed",
    "needs_rerun",
}
WINDOW_STATUS_VALUES = {
    "pending",
    "llm_done",
    "failed",
    "rate_limited",
    "needs_rerun",
}


@dataclass
class ArticleStatus:
    input_csv_path: str
    input_row_index: int
    canonical_url: str | None
    article_fingerprint: str
    article_id: int
    status: str
    fetch_method: str | None = None
    failure_reason: str | None = None
    raw_html_cache_path: str | None = None
    clean_text_cache_path: str | None = None
    updated_at_utc: str = ""


@dataclass
class WindowStatus:
    article_id: int
    window_index: int
    sentence_id_start: int
    sentence_id_end: int
    owned_sentence_id_start: int
    owned_sentence_id_end: int
    window_fingerprint: str
    prompt_version: str
    model_api_id: str
    status: str
    response_cache_path: str | None = None
    failure_reason: str | None = None
    updated_at_utc: str = ""


class StateStore:
    """Append-only JSONL state for one cached-CSV processing unit."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.run_dir / "articles_status.jsonl"
        self.windows_path = self.run_dir / "llm_windows_status.jsonl"
        self._article_index: dict[tuple[int, str], ArticleStatus] = {}
        self._window_index: dict[tuple[int, int], WindowStatus] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if self.articles_path.exists():
            for line in self.articles_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = ArticleStatus(**data)
                key = (rec.input_row_index, rec.article_fingerprint)
                self._article_index[key] = rec
        if self.windows_path.exists():
            for line in self.windows_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = WindowStatus(**data)
                self._window_index[(rec.article_id, rec.window_index)] = rec

    def get_article(
        self, *, input_row_index: int, article_fingerprint: str
    ) -> ArticleStatus | None:
        return self._article_index.get((input_row_index, article_fingerprint))

    def upsert_article(self, status: ArticleStatus) -> None:
        if status.status not in ARTICLE_STATUS_VALUES:
            raise ValueError(f"Invalid article status: {status.status!r}")
        status.updated_at_utc = now_utc_iso()
        key = (status.input_row_index, status.article_fingerprint)
        self._article_index[key] = status
        with self.articles_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(status), ensure_ascii=False) + "\n")

    def get_window(self, *, article_id: int, window_index: int) -> WindowStatus | None:
        return self._window_index.get((article_id, window_index))

    def upsert_window(self, status: WindowStatus) -> None:
        if status.status not in WINDOW_STATUS_VALUES:
            raise ValueError(f"Invalid window status: {status.status!r}")
        status.updated_at_utc = now_utc_iso()
        self._window_index[(status.article_id, status.window_index)] = status
        with self.windows_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(status), ensure_ascii=False) + "\n")

    def article_statuses(self) -> Iterable[ArticleStatus]:
        return list(self._article_index.values())

    def window_statuses(self) -> Iterable[WindowStatus]:
        return list(self._window_index.values())
