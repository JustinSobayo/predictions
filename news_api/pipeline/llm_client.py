"""LLM clients: a real Gemini provider and a deterministic ``Fake`` provider.

The fake provider is used by ``--dry-run`` and by the e2e tests, so the rest
of the pipeline can be exercised without an API key or quota. It applies a
small set of heuristic rules to flag candidate sentences, mirroring the kind
of output we expect from Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import LLMSettings
from .segment import LLMWindow, Sentence


LOGGER = logging.getLogger("pipeline.llm_client")


@dataclass
class DomainResponse:
    """Article-level domain assignment from the LLM."""

    query_domain: str
    top_level_domain: str
    misc_subtopic: str | None
    confidence: float
    domain_reason: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSpan:
    """One LLM-flagged candidate prediction span (window-scoped)."""

    primary_sentence_id: int
    span_sentence_ids: list[int]
    span_text: str
    candidate_reason: str
    reason_category: str
    context_needed: bool
    uncertainty_note: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowResponse:
    """A list of candidate spans returned by one LLM window call."""

    spans: list[CandidateSpan]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenEstimate:
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    """Minimal interface used by ``candidates.py``."""

    name: str
    model_api_id: str
    model_display_name: str

    def assign_domain(
        self,
        *,
        article_title: str | None,
        article_description: str | None,
        article_url: str | None,
        article_content: str | None,
        query_domain_hint: str | None,
    ) -> DomainResponse: ...

    def extract_candidates(
        self,
        *,
        window: LLMWindow,
        article_title: str | None,
        article_description: str | None,
    ) -> WindowResponse: ...

    def estimate_window_tokens(self, window: LLMWindow) -> TokenEstimate: ...


# -----------------------------------------------------------------------------
# Helpers used by both providers.
# -----------------------------------------------------------------------------


_PREDICTION_TRIGGER_RE = re.compile(
    r"\b("
    r"will|won't|wo n't|shall|should|would|could|might|may|must|"
    r"expect(?:s|ed|ing)?|forecast(?:s|ed|ing)?|predict(?:s|ed|ing|ion|ions)?|"
    r"project(?:s|ed|ing|ion|ions)?|anticipat(?:e|es|ed|ing)|"
    r"plan(?:s|ned|ning)?|aim(?:s|ed|ing)?|set to|on track to|"
    r"likely|unlikely|expected|forecasted|projected|"
    r"by\s+\d{4}|by\s+(?:next|the\s+end\s+of)|"
    r"in\s+the\s+coming\s+(?:weeks?|months?|years?|decade)|"
    r"next\s+(?:year|month|week|quarter|decade)"
    r")\b",
    flags=re.IGNORECASE,
)
_FUTURE_DATE_RE = re.compile(r"\b20[3-9]\d\b|\b2[1-9]\d{2}\b")
_TRANSIENT_GEMINI_STATUS_CODES = {429, 500, 502, 503, 504}
_STRICT_REASON_CATEGORIES = {
    "forecast",
    "projection",
    "expected_outcome",
    "probabilistic_claim",
    "conditional_forecast",
    "warning",
    "estimate",
    "polling_or_odds",
    "policy_or_legal_outcome",
}
_STRICT_PREDICTION_SIGNAL_RE = re.compile(
    r"\b("
    r"expect(?:s|ed|ing)?|forecast(?:s|ed|ing)?|predict(?:s|ed|ing|ion|ions)?|"
    r"project(?:s|ed|ing|ion|ions)?|anticipat(?:e|es|ed|ing)|"
    r"estimat(?:e|es|ed|ing)|likely|unlikely|odds|poll(?:s|ing)?|"
    r"warn(?:s|ed|ing)?|risk|chance|probab(?:le|ility)"
    r")\b",
    flags=re.IGNORECASE,
)
_WEAK_MODAL_RE = re.compile(r"\b(could|might|may)\b", flags=re.IGNORECASE)
_PAST_MODAL_RE = re.compile(r"\b(could|might|may)\s+have\s+been\b", flags=re.IGNORECASE)
_FUTURE_YEAR_PATTERN = r"(?:202[6-9]|20[3-9]\d)"
_FUTURE_ORIENTATION_RE = re.compile(
    r"\b("
    r"will|won't|shall|could|might|may|"
    r"expected to|forecast(?:ed)? to|projected to|predicted to|likely to|"
    r"unlikely to|on track to|"
    rf"by\s+(?:{_FUTURE_YEAR_PATTERN}|next|the end of)|"
    rf"in\s+(?:{_FUTURE_YEAR_PATTERN}|the coming|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)|"
    r"next\s+(?:year|month|week|quarter|decade|election|january|february|"
    r"march|april|may|june|july|august|september|october|november|december)|"
    r"upcoming|future|coming"
    r")\b",
    flags=re.IGNORECASE,
)
_CONCRETE_OUTCOME_RE = re.compile(
    r"\b("
    r"win|lose|pass|fail|approve|reject|block|overturn|rule|vote|turnout|"
    r"adopt|ratify|enact|become law|sign(?:ed|s)? into law|victory|defeat|"
    r"majority|margin of victory|vote margin|percent|percentage|rate|count|increase|decrease|"
    r"rise|fall|drop|grow|shrink|lead|trail|nomination|lawsuit|court|ballot|"
    r"loss|shortage|gain|snatch(?:ing)?(?: [a-z-]+){0,4} seat|"
    r"hold(?:ing)?(?: [a-z-]+){0,4} seat|retain(?:ing)?(?: [a-z-]+){0,4} seat"
    r")\b",
    flags=re.IGNORECASE,
)
_SCHEDULE_OR_PROCESS_RE = re.compile(
    r"\b("
    r"scheduled|will take place|set to (?:begin|start|meet|hold|vote|launch|"
    r"appear|speak)|plans? to|announced (?:that )?(?:it|he|she|they) will|"
    r"deadline|calendar|hearing|meeting|debate|rally|primary election day"
    r")\b",
    flags=re.IGNORECASE,
)
_PROCESS_ONLY_RE = re.compile(
    r"\b("
    r"expected to be (?:heard|considered|debated|scheduled)|"
    r"will be (?:heard|considered|debated|scheduled)|"
    r"expected to receive a final vote|"
    r"heard for consideration|take effect|if enacted|if passed|"
    r"may be required|could be reported|could lead to disciplinary action|"
    r"authorizes?|provides that|would allow|would prohibit|would require|"
    r"now awaits .*signature|generally expected to approve"
    r")\b",
    flags=re.IGNORECASE,
)
_PRESENT_ESTIMATE_RE = re.compile(
    r"\ban estimated\b.{0,80}\b(?:is|are|was|were)\b",
    flags=re.IGNORECASE,
)
_SUBJECTIVE_FUTURE_RE = re.compile(
    r"\bwill (?:be|remain|feel|seem|look|know)\b",
    flags=re.IGNORECASE,
)


def _heuristic_is_candidate(text: str) -> tuple[bool, str, str]:
    """Cheap rule-based prediction-ish detector.

    Returns ``(is_candidate, reason_category, reason_text)``. Used by the
    ``Fake`` provider and as a tie-breaker when the real LLM is offline.
    """
    if not text:
        return False, "none", ""
    t = text.strip()
    if len(t) < 4:
        return False, "none", ""
    trigger = _PREDICTION_TRIGGER_RE.search(t)
    future_date = _FUTURE_DATE_RE.search(t)
    if trigger:
        return True, "linguistic_indicator", f"matched trigger: {trigger.group(0)!r}"
    if future_date:
        return True, "future_orientation", f"matched future date: {future_date.group(0)!r}"
    return False, "none", ""


def _approx_token_count(text: str) -> int:
    """Rough token estimator used for dry-run / batch planning."""
    if not text:
        return 0
    # ~4 characters per token is a reasonable heuristic for English news.
    return max(1, len(text) // 4)


def _domain_hint_or_misc(query_hint: str | None) -> tuple[str, str, str | None]:
    """Map a noisy query hint to ``(query_domain, top_level, misc_subtopic)``."""
    if not query_hint:
        return "misc-general", "misc", "general"
    hint = query_hint.strip().lower()
    if hint.startswith("sports") or hint == "sport":
        return "sport", "sport", None
    if hint.startswith("health") or hint.startswith("medical"):
        return "health", "health", None
    if hint.startswith("policy") or hint.startswith("politic") or hint.startswith("law"):
        return "policy", "policy", None
    if hint.startswith("misc-"):
        suffix = hint[len("misc-"):] or "general"
        return f"misc-{suffix}", "misc", suffix
    # everything else becomes a misc bucket whose subtopic is the (slugged) hint
    slug = re.sub(r"[^a-z0-9]+", "-", hint).strip("-") or "general"
    return f"misc-{slug}", "misc", slug


@dataclass
class FakeLLMProvider:
    """Heuristic stand-in for Gemini. Used by --dry-run and tests."""

    model_api_id: str = "fake-llm-v1"
    model_display_name: str = "Fake LLM (deterministic)"
    name: str = "fake"

    def assign_domain(
        self,
        *,
        article_title: str | None,
        article_description: str | None,
        article_url: str | None,
        article_content: str | None,
        query_domain_hint: str | None,
    ) -> DomainResponse:
        domain, top_level, misc = _domain_hint_or_misc(query_domain_hint)
        return DomainResponse(
            query_domain=domain,
            top_level_domain=top_level,
            misc_subtopic=misc,
            confidence=0.5,
            domain_reason="fake_provider:hint_passthrough",
            raw={"source": "fake"},
        )

    def extract_candidates(
        self,
        *,
        window: LLMWindow,
        article_title: str | None,
        article_description: str | None,
    ) -> WindowResponse:
        spans: list[CandidateSpan] = []
        for sent in window.sentences:
            is_cand, category, reason_text = _heuristic_is_candidate(sent.text)
            if not is_cand:
                continue
            spans.append(
                CandidateSpan(
                    primary_sentence_id=sent.sentence_id,
                    span_sentence_ids=[sent.sentence_id],
                    span_text=sent.text,
                    candidate_reason=reason_text or "heuristic candidate",
                    reason_category=category,
                    context_needed=False,
                    uncertainty_note="heuristic; verify manually",
                    raw={"source": "fake"},
                )
            )
        return WindowResponse(spans=spans, raw={"source": "fake"})

    def estimate_window_tokens(self, window: LLMWindow) -> TokenEstimate:
        body = "\n".join(s.text for s in window.sentences)
        return TokenEstimate(
            input_tokens=_approx_token_count(body) + 400,  # +prompt
            output_tokens=200,
        )


# Real Gemini provider (uses google-genai if available + API key set).


_DOMAIN_PROMPT = """You are a precise news topic classifier.

You will see one news article (title, description, optional URL, optional body).
Pick the SINGLE best canonical domain bucket from this closed set:
  - "health"   (clinical medicine, public health, healthcare policy, drugs, disease, mental health)
  - "policy"   (government, legislation, courts, elections, regulation, geopolitics, public administration)
  - "sport"    (any sports league/event/team/athlete coverage)
  - "misc-<subtopic>"  (catch-all; pick a short kebab-case subtopic, e.g. "misc-tesla", "misc-ai", "misc-aviation")

Strict rules:
- If the article's primary topic is finance, markets, weather, or climate change, set "rejected_excluded": true and use "misc-general" as a placeholder.
- Otherwise prefer the most specific accurate bucket.
- "query_domain_hint" is a hint from the user's original query; you may use it but the article's actual content takes priority.

Return JSON only, no prose:
{
  "query_domain": "<one of: health|policy|sport|misc-<subtopic>>",
  "top_level_domain": "<one of: health|policy|sport|misc>",
  "misc_subtopic": "<lowercase kebab-case subtopic if top_level_domain is misc, else null>",
  "confidence": <float between 0 and 1>,
  "domain_reason": "<one short sentence>",
  "rejected_excluded": <true|false>
}
"""


_CANDIDATE_PROMPT = """You are an annotation assistant building a HIGH-PRECISION dataset of PREDICTION sentences.

Only return strong prediction candidates. A "prediction" must satisfy ALL of:
  1. Future orientation: refers to a state/event after the time of utterance.
  2. Forecasting signal: forecast verbs or probabilistic language such as expect, predict, project, forecast, anticipate, estimate, likely/unlikely, may/could/might when used to express uncertainty, or quantified polling/odds/projection language.
  3. Falsifiable outcome: a concrete claim that can later be checked as true/false.
  4. Uncertainty/risk: the outcome is not already fixed, scheduled, legally mandated, or a calendar fact.
  5. Measurable outcome: the future event/state has an observable result, amount, winner, vote, count, rate, policy outcome, legal outcome, adoption outcome, health outcome, or similar.

Use the "revised information test": include the sentence only if new information arriving today could reasonably change the claim. If no new information could change it, it is not a prediction.

Do NOT return:
  - pure background facts, historical summaries, biographical context, or explanations of how a process works;
  - hedged reporting without a concrete outcome ("could be important", "may face challenges", "might affect voters") unless it names a specific future result that can be checked;
  - generic "could/may/might" hypotheticals that describe possibility but not a forecast;
  - process narration ("the bill moved to committee", "lawmakers are debating", "the campaign is preparing") without a predicted outcome;
  - attribution-only quotes or campaign rhetoric ("X says Y will win", "the candidate vowed to fight") unless the quoted claim itself is a concrete forecast with an outcome;
  - price targets, rankings, odds, polling, or projections that are historical/descriptive rather than explicitly forward-looking and checkable;
  - fixed schedules, calendars, event dates, deadlines, planned meetings, hearings, debates, votes, elections, launches, or trials unless the sentence predicts an uncertain outcome of that event;
  - statements of intent, promises, goals, hopes, campaign messaging, or plans unless they predict an outcome beyond the actor's control;
  - generic future tense with "will" when it simply reports an already scheduled action;
  - historical claims, summaries, background facts, or procedural descriptions;
  - vague possibilities without a concrete falsifiable outcome;
  - quotes/opinions that are not making a checkable forecast.

Micro-examples:
  INCLUDE: "Analysts project the party will lose three House seats in November." -> concrete, future, checkable seat outcome.
  INCLUDE: "Officials warn turnout could fall below 50% if the rule remains in place." -> conditional, measurable future outcome.
  REJECT: "The committee will meet Tuesday to discuss the bill." -> scheduled process, no uncertain outcome.
  REJECT: "The candidate said the plan would help families." -> rhetoric/claim about intent, not a checkable forecast.

A prediction span MAY contain multiple sentences, but keep spans tight. Prefer a single primary sentence unless an adjacent sentence is required to understand the predicted outcome. For every prediction span you return, the primary sentence must be the most prediction-bearing one.

You will be given a window of sentences plus an "owned range" of sentence ids. Only return spans whose PRIMARY sentence id falls inside the owned range. (Other sentences are context.)

Goal: HIGH PRECISION. When in doubt, leave it out. It is better to return fewer rows than to send many non-predictions to human annotators.
If a sentence is only weakly predictive or merely prediction-adjacent, return an empty spans array for that window.

Return JSON only, no prose:
{
  "spans": [
    {
      "primary_sentence_id": <int>,
      "span_sentence_ids": [<int>, ...],
      "span_text": "<verbatim concatenation of the span sentences>",
      "candidate_reason": "<one sentence naming the concrete future outcome and why it is uncertain/checkable>",
      "reason_category": "<one of: forecast | projection | expected_outcome | probabilistic_claim | conditional_forecast | warning | estimate | polling_or_odds | policy_or_legal_outcome>",
      "context_needed": <true|false>,
      "uncertainty_note": "<brief note about edge-case-ness, or empty string>"
    }
  ]
}
If no spans, return {"spans": []}.
"""


def _safe_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    v = d.get(key)
    return default if v is None else v


def _coerce_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_id_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [i for i in (_coerce_int(v) for v in value) if i >= 0]


def _gemini_error_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    match = re.search(r"\b(429|5\d{2})\b", text)
    return int(match.group(1)) if match else None


def _is_transient_gemini_error(exc: BaseException) -> bool:
    status = _gemini_error_status(exc)
    return status in _TRANSIENT_GEMINI_STATUS_CODES


def _is_strict_candidate_span(span: CandidateSpan, *, primary_text: str | None = None) -> bool:
    span_text = span.span_text.strip()
    gate_text = (primary_text or span_text).strip()
    if not span_text or not gate_text:
        return False
    reason = span.reason_category.strip().lower()
    if _PAST_MODAL_RE.search(gate_text):
        return False
    if _PRESENT_ESTIMATE_RE.search(gate_text):
        return False
    if _SUBJECTIVE_FUTURE_RE.search(gate_text):
        return False
    has_future_orientation = bool(_FUTURE_ORIENTATION_RE.search(gate_text))
    has_strict_signal = bool(_STRICT_PREDICTION_SIGNAL_RE.search(gate_text))
    has_concrete_outcome = bool(_CONCRETE_OUTCOME_RE.search(gate_text))
    has_weak_modal_forecast = bool(
        _WEAK_MODAL_RE.search(gate_text) and has_concrete_outcome
    )
    if not has_future_orientation:
        return False
    if reason and reason not in _STRICT_REASON_CATEGORIES and not has_strict_signal:
        return False
    if _PROCESS_ONLY_RE.search(gate_text):
        return False
    if _SCHEDULE_OR_PROCESS_RE.search(gate_text) and not (
        has_strict_signal or has_weak_modal_forecast
    ):
        return False
    return has_concrete_outcome and (has_strict_signal or has_weak_modal_forecast)


@dataclass
class GeminiProvider:
    """Wrapper around ``google-genai`` for Gemini 2.5 Flash-Lite."""

    settings: LLMSettings
    name: str = "gemini"
    _client: Any | None = None

    @property
    def model_api_id(self) -> str:
        return self.settings.model_api_id

    @property
    def model_display_name(self) -> str:
        return self.settings.model_display_name

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai 
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        api_key = self.settings.api_key
        if not api_key:
            raise RuntimeError(
                f"Missing API key (env var {self.settings.api_key_env}). "
                "Set it in .env or use --fake-llm for dry runs."
            )
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _generate_content_once(self, full_prompt: str):
        client = self._get_client()
        try:
            from google.genai import types

            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=float(self.settings.candidate_param("temperature", 0)),
            )
            resp = client.models.generate_content(
                model=self.model_api_id,
                contents=full_prompt,
                config=cfg,
            )
        except TypeError:
            resp = client.models.generate_content(
                model=self.model_api_id,
                contents=full_prompt,
            )
        return resp

    def _generate_json(self, prompt: str, payload: str) -> dict[str, Any]:
        full_prompt = prompt + "\n\nINPUT:\n" + payload
        max_retries = int(self.settings.candidate_param("max_retries", 5))
        base_delay = float(self.settings.candidate_param("retry_base_delay_seconds", 10))
        max_delay = float(self.settings.candidate_param("retry_max_delay_seconds", 120))

        attempt = 0
        while True:
            try:
                resp = self._generate_content_once(full_prompt)
                break
            except Exception as exc:
                if not _is_transient_gemini_error(exc) or attempt >= max_retries:
                    raise
                attempt += 1
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                status = _gemini_error_status(exc)
                LOGGER.warning(
                    "Gemini transient error status=%s attempt=%s/%s; retrying in %.1fs",
                    status or "unknown",
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)

        text = getattr(resp, "text", None) or ""
        if not text:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = ""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                return {}
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}

    def assign_domain(
        self,
        *,
        article_title: str | None,
        article_description: str | None,
        article_url: str | None,
        article_content: str | None,
        query_domain_hint: str | None,
    ) -> DomainResponse:
        payload = json.dumps(
            {
                "title": article_title or "",
                "description": article_description or "",
                "url": article_url or "",
                "content_excerpt": (article_content or "")[:2000],
                "query_domain_hint": query_domain_hint or "",
            },
            ensure_ascii=False,
        )
        raw = self._generate_json(_DOMAIN_PROMPT, payload)
        domain = str(_safe_get(raw, "query_domain", "")).strip() or "misc-general"
        top_level = str(_safe_get(raw, "top_level_domain", "")).strip() or (
            "misc" if domain.startswith("misc-") else domain
        )
        return DomainResponse(
            query_domain=domain,
            top_level_domain=top_level,
            misc_subtopic=raw.get("misc_subtopic"),
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            domain_reason=str(raw.get("domain_reason", "")),
            raw=raw,
        )

    def extract_candidates(
        self,
        *,
        window: LLMWindow,
        article_title: str | None,
        article_description: str | None,
    ) -> WindowResponse:
        sent_payload = [
            {"id": s.sentence_id, "source_field": s.source_field, "text": s.text}
            for s in window.sentences
        ]
        payload = json.dumps(
            {
                "article_title": article_title or "",
                "article_description": article_description or "",
                "owned_sentence_id_start": window.owned_sentence_id_start,
                "owned_sentence_id_end": window.owned_sentence_id_end,
                "sentences": sent_payload,
            },
            ensure_ascii=False,
        )
        raw = self._generate_json(_CANDIDATE_PROMPT, payload)
        spans_raw = raw.get("spans") if isinstance(raw, dict) else None
        if not isinstance(spans_raw, list):
            spans_raw = []
        spans: list[CandidateSpan] = []
        valid_ids = {s.sentence_id for s in window.sentences}
        for item in spans_raw:
            if not isinstance(item, dict):
                continue
            primary = _coerce_int(item.get("primary_sentence_id"))
            if primary < 0 or primary not in valid_ids:
                continue
            ids = _coerce_id_list(item.get("span_sentence_ids")) or [primary]
            ids = [i for i in ids if i in valid_ids]
            if primary not in ids:
                ids.append(primary)
            ids = sorted(set(ids))
            text = str(item.get("span_text", "")).strip()
            if not text:
                # rebuild span text from window if missing
                lookup = {s.sentence_id: s.text for s in window.sentences}
                text = " ".join(lookup[i] for i in ids if i in lookup).strip()
            spans.append(
                CandidateSpan(
                    primary_sentence_id=primary,
                    span_sentence_ids=ids,
                    span_text=text,
                    candidate_reason=str(item.get("candidate_reason", "")),
                    reason_category=str(item.get("reason_category", "")),
                    context_needed=bool(item.get("context_needed", False)),
                    uncertainty_note=str(item.get("uncertainty_note", "")),
                    raw=item,
                )
            )
        sentence_lookup = {s.sentence_id: s.text for s in window.sentences}
        strict_spans = [
            span
            for span in spans
            if _is_strict_candidate_span(
                span,
                primary_text=sentence_lookup.get(span.primary_sentence_id),
            )
        ]
        return WindowResponse(spans=strict_spans, raw=raw if isinstance(raw, dict) else {})

    def estimate_window_tokens(self, window: LLMWindow) -> TokenEstimate:
        body = "\n".join(s.text for s in window.sentences)
        return TokenEstimate(
            input_tokens=_approx_token_count(_CANDIDATE_PROMPT)
            + _approx_token_count(body),
            output_tokens=400,
        )


def make_provider(settings: LLMSettings, *, fake: bool = False) -> LLMProvider:
    """Build the right provider for the current run."""
    if fake:
        return FakeLLMProvider()
    if settings.provider != "gemini":
        raise RuntimeError(
            f"Unsupported LLM provider: {settings.provider!r}. v1 only ships Gemini."
        )
    if not settings.api_key:
        return FakeLLMProvider()
    return GeminiProvider(settings=settings)
