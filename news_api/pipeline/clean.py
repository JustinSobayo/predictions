"""Text cleaning: NewsAPI truncation marker + lightweight ad/boilerplate filter.

This is the v1 cleaner. Live web scraping is out of scope until we hit cases
where the cached NewsAPI text is not "complete enough"; for the prototype we
work directly off the NewsAPI ``content`` field (and supplement with title /
description) after stripping the truncation marker and obvious boilerplate.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .utils import normalize_whitespace


# NewsAPI truncates long article bodies with a marker like
# ``"... [+3230 chars]"``. Cut everything from that marker onward, with no
# trailing period (per the plan: "just cut it off there").
TRUNCATION_RE = re.compile(r"\s*…\s*\[\+\d+\s+chars\]\s*$|\s*\.\.\.\s*\[\+\d+\s+chars\]\s*$")
# Same marker but anywhere in the string (rare; mid-content truncation).
TRUNCATION_INLINE_RE = re.compile(r"\s*…\s*\[\+\d+\s+chars\]\s*|\s*\.\.\.\s*\[\+\d+\s+chars\]\s*")

# NewsAPI sometimes leaves a parenthetical ``(+2968)`` in the middle of a
# sentence where the number is the **omitted character count** (same meaning
# as ``[+2968 chars]``). Real editorial text rarely uses ``(+dddd)`` with
# **four or more digits** after the plus; smaller values like ``(+10)`` may be
# odds, units, or other semantics — so we only strip when ``\d{4,}``.
TRUNCATION_PAREN_CHAR_COUNT_RE = re.compile(r"\(\s*\+\s*\d{4,}\s*\)")
HTML_TAG_RE = re.compile(r"</?[^>]+>")


# Obvious ad / boilerplate stub patterns that occasionally bleed in via the
# NewsAPI ``content`` field. These are case-insensitive line-level matches.
# Keep the list conservative; the goal in the plan is to maximise recall and
# let the human annotator drop anything ambiguous.
AD_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"^advertisement\s*$",
    r"^advertisement\b.*",
    r"^sponsored content\s*$",
    r"^sponsored\s*$",
    r"^sponsored by\b.*",
    r"^subscribe to .*",
    r"^subscribe now.*",
    r"^sign up (for|to) .*",
    r"^read more:.*",
    r"^read also:.*",
    r"^share this article.*",
    r"^follow us on .*",
    r"^©.*all rights reserved.*",
    r"^all rights reserved\.?$",
    r"^this content is for subscribers only.*",
    r"^accept cookies.*",
    r"^by clicking .*you (agree|consent).*",
    r"^by continuing,? you (agree|consent).*",
    r"^you can unsubscribe.*",
    r"^this article was originally published.*",
    r"^cookie (policy|preferences|settings).*",
)
_BOILERPLATE_RE = re.compile(
    r"|".join(f"(?:{pat})" for pat in AD_BOILERPLATE_PATTERNS),
    flags=re.IGNORECASE,
)


@dataclass
class CleanedText:
    text: str
    had_truncation_marker: bool
    removed_boilerplate_lines: int
    used_fields: list[str]
    raw_combined: str

    def is_complete_enough(self) -> bool:
        """Heuristic: at least one non-trivial sentence-ish chunk after cleaning.

        The plan defines "complete enough" loosely as: no truncation marker,
        at least one valid article-body sentence, not mostly boilerplate. We
        approximate that here without doing the full segmentation pass: if
        the cleaned body has <40 chars and is mostly the title, it's not
        complete enough.
        """
        body = self.text.strip()
        if not body:
            return False
        if len(body) < 40:
            return False
        return True


def strip_truncation_parentheticals(text: str) -> tuple[str, int]:
    """Remove ``(+NNNN)`` blobs where ``NNNN`` has at least 4 digits.

    These are almost always leaked NewsAPI truncation metadata, e.g.
    ``89-53 (+2968) on his picks``. Shorter runs like ``(+123)`` or ``(+99)``
    are left intact so legitimate numeric parentheticals are not stripped.
    """
    if not text:
        return "", 0
    matches = TRUNCATION_PAREN_CHAR_COUNT_RE.findall(text)
    out = TRUNCATION_PAREN_CHAR_COUNT_RE.sub(" ", text)
    return out, len(matches)


def sanitize_output_sentence_text(text: str | None) -> str:
    """Final pass for any user-facing sentence / span field.

    Applies truncation-marker removal, parenthetical ``(+dddd+)`` char-count
    removal, and whitespace normalisation. Does **not** re-run ad/boilerplate
    stripping (that runs once per article in ``clean_article_text``).
    """
    if not text:
        return ""
    t, _ = strip_truncation_marker(text)
    t, _ = strip_truncation_parentheticals(t)
    t = strip_html_markup(t)
    return normalize_whitespace(t)


def strip_html_markup(text: str) -> str:
    """Remove lightweight HTML markup and decode entities.

    NewsAPI content occasionally leaks fragments like ``</li><li>`` into the
    visible text. Strip tags conservatively and decode HTML entities so the
    user-facing sentence fields stay readable.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    return text


def strip_truncation_marker(text: str) -> tuple[str, bool]:
    """Strip ``"... [+N chars]"`` (and the unicode ``…`` variant) from text.

    Returns the cleaned text and a flag indicating whether the marker was
    present.
    """
    if not text:
        return "", False
    had = bool(TRUNCATION_RE.search(text)) or bool(TRUNCATION_INLINE_RE.search(text))
    text = TRUNCATION_INLINE_RE.sub(" ", text)
    text = TRUNCATION_RE.sub("", text)
    return text.rstrip(), had


def strip_boilerplate(text: str) -> tuple[str, int]:
    """Drop lines that match obvious ad / boilerplate stubs.

    Returns the rebuilt text and the count of removed lines.
    """
    if not text:
        return "", 0
    lines = text.splitlines()
    kept: list[str] = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if _BOILERPLATE_RE.match(stripped):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def build_combined_text(
    *,
    title: str | None,
    description: str | None,
    content: str | None,
) -> tuple[str, list[str]]:
    """Stitch title / description / content into one string for segmentation.

    Returns ``(combined, used_fields)`` where ``used_fields`` lists which of
    title/description/content actually contributed (used by the
    transform step to attribute sentences back to their source field).
    """
    parts: list[str] = []
    used: list[str] = []
    if title:
        parts.append(title.strip())
        used.append("title")
    if description and description.strip() != (title or "").strip():
        parts.append(description.strip())
        used.append("description")
    if content and content.strip():
        parts.append(content.strip())
        used.append("content")
    return "\n\n".join(parts), used


def clean_article_text(
    *,
    title: str | None,
    description: str | None,
    content: str | None,
) -> CleanedText:
    """Run the full cleaning pipeline against one article."""
    combined, used = build_combined_text(
        title=title,
        description=description,
        content=content,
    )
    raw_combined = combined

    cleaned, had_marker = strip_truncation_marker(combined)
    cleaned, _ = strip_truncation_parentheticals(cleaned)
    cleaned = strip_html_markup(cleaned)
    cleaned, removed = strip_boilerplate(cleaned)
    cleaned = normalize_whitespace(cleaned)

    return CleanedText(
        text=cleaned,
        had_truncation_marker=had_marker,
        removed_boilerplate_lines=removed,
        used_fields=used,
        raw_combined=raw_combined,
    )
