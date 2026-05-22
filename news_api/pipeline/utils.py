"""Small shared helpers: URL canonicalization, fingerprints, JSON safety."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


_TRACKING_PARAM_PREFIXES: tuple[str, ...] = (
    "utm_",
    "fbclid",
    "gclid",
    "mc_",
    "_ga",
    "ref",
    "ref_",
    "ref_src",
    "ocid",
    "cmpid",
    "ncid",
    "spm",
)


def now_utc_iso() -> str:
    """ISO-8601 timestamp with seconds precision (UTC, no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonicalize_url(url: str | None) -> str | None:
    """Normalise a URL for dedupe.

    Steps: lowercase scheme/host, strip ``www.``, drop common tracking query
    params, normalise trailing slashes and the empty query/fragment. Returns
    ``None`` for empty / un-parseable input.
    """
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if not parsed.scheme or not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # strip default ports
    if host.endswith(":80") and scheme == "http":
        host = host[:-3]
    if host.endswith(":443") and scheme == "https":
        host = host[:-4]

    path = parsed.path or "/"
    # collapse trailing slash on non-root paths
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    # drop tracking query params, sort the rest for stability
    kept_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(k)
    ]
    kept_params.sort()
    query = urlencode(kept_params, doseq=True)

    return urlunparse((scheme, host, path, "", query, ""))


def _is_tracking_param(key: str) -> bool:
    key = key.lower()
    return any(key == prefix or key.startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES)


def url_host(url: str | None) -> str | None:
    """Return the lowercased host (no ``www.``) of a URL or ``None``."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def url_path(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    return parsed.path or "/"


def fingerprint(*parts: Any) -> str:
    """Stable short fingerprint over arbitrary string-able parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(b"\x00")
        if part is None:
            h.update(b"<none>")
        else:
            h.update(str(part).encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``NaN``/``inf`` floats to ``None`` for strict JSON.

    Pandas-style ``NaN`` floats and numpy types sneak in through the cached
    CSVs. ``json.dump(..., allow_nan=False)`` will raise on those, so we run
    everything through this filter first.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    # fallback to repr-friendly string
    return str(value)


def write_json(path, payload: Any) -> None:
    """Write strict JSON (UTF-8, ``allow_nan=False``, indent=4)."""
    safe = to_jsonable(payload)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(safe, fp, ensure_ascii=False, indent=4, allow_nan=False)
        fp.write("\n")


def parse_source_meta(raw: Any) -> dict[str, Any]:
    """Parse the ``source`` field from cached CSVs.

    NewsAPI's CSV dumps store the source as a Python-repr ``dict`` string,
    e.g. ``"{'id': None, 'name': 'Fast Company'}"``. Try ``ast.literal_eval``
    first, then JSON, then fall back to a best-effort name-only structure.
    """
    if isinstance(raw, dict):
        return {"id": raw.get("id"), "name": raw.get("name")}
    if not isinstance(raw, str):
        return {"id": None, "name": None}
    s = raw.strip()
    if not s:
        return {"id": None, "name": None}

    import ast

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, dict):
            return {"id": parsed.get("id"), "name": parsed.get("name")}
    except (ValueError, SyntaxError):
        pass
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return {"id": parsed.get("id"), "name": parsed.get("name")}
    except json.JSONDecodeError:
        pass
    return {"id": None, "name": s or None}


@dataclass(frozen=True)
class CanonicalArticleKey:
    """Cheap dedupe key. Prefers canonical URL; falls back to ``(title, source_name, publishedAt)``."""

    canonical_url: str | None
    fallback: tuple[str, str, str] | None

    @classmethod
    def from_row(
        cls,
        url: str | None,
        title: str | None,
        source_name: str | None,
        published_at: str | None,
    ) -> "CanonicalArticleKey":
        canonical = canonicalize_url(url)
        if canonical:
            return cls(canonical_url=canonical, fallback=None)
        norm = lambda v: (v or "").strip().lower()
        fallback = (norm(title), norm(source_name), norm(published_at))
        if any(fallback):
            return cls(canonical_url=None, fallback=fallback)
        return cls(canonical_url=None, fallback=None)

    def is_empty(self) -> bool:
        return self.canonical_url is None and not self.fallback


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace, keep newlines."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
