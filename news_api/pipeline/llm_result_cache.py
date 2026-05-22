"""Persistent article-level cache for reusable LLM extraction results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from .candidates import ArticleLLMResult, WindowCallResult
from .llm_client import CandidateSpan, DomainResponse, WindowResponse
from .segment import LLMWindow, Sentence
from .utils import now_utc_iso, to_jsonable


LOGGER = logging.getLogger("pipeline.llm_result_cache")
CACHE_SCHEMA_VERSION = 1


def sentence_to_dict(sentence: Sentence) -> dict[str, Any]:
    return {
        "sentence_id": sentence.sentence_id,
        "text": sentence.text,
        "source_field": sentence.source_field,
        "field_order": sentence.field_order,
    }


def sentence_from_dict(data: dict[str, Any]) -> Sentence:
    return Sentence(
        sentence_id=int(data["sentence_id"]),
        text=str(data["text"]),
        source_field=str(data["source_field"]),
        field_order=int(data["field_order"]),
    )


def llm_window_to_dict(window: LLMWindow) -> dict[str, Any]:
    return {
        "window_index": window.window_index,
        "sentence_id_start": window.sentence_id_start,
        "sentence_id_end": window.sentence_id_end,
        "owned_sentence_id_start": window.owned_sentence_id_start,
        "owned_sentence_id_end": window.owned_sentence_id_end,
        "sentences": [sentence_to_dict(sentence) for sentence in window.sentences],
    }


def llm_window_from_dict(data: dict[str, Any]) -> LLMWindow:
    sentences_raw = data.get("sentences")
    if not isinstance(sentences_raw, list):
        raise ValueError("cached LLM window is missing sentences")
    return LLMWindow(
        window_index=int(data["window_index"]),
        sentence_id_start=int(data["sentence_id_start"]),
        sentence_id_end=int(data["sentence_id_end"]),
        owned_sentence_id_start=int(data["owned_sentence_id_start"]),
        owned_sentence_id_end=int(data["owned_sentence_id_end"]),
        sentences=[sentence_from_dict(item) for item in sentences_raw],
    )


def domain_response_to_dict(response: DomainResponse) -> dict[str, Any]:
    return {
        "query_domain": response.query_domain,
        "top_level_domain": response.top_level_domain,
        "misc_subtopic": response.misc_subtopic,
        "confidence": response.confidence,
        "domain_reason": response.domain_reason,
        "raw": response.raw,
    }


def domain_response_from_dict(data: dict[str, Any]) -> DomainResponse:
    raw = data.get("raw")
    return DomainResponse(
        query_domain=str(data["query_domain"]),
        top_level_domain=str(data["top_level_domain"]),
        misc_subtopic=(
            None if data.get("misc_subtopic") is None else str(data.get("misc_subtopic"))
        ),
        confidence=float(data["confidence"]),
        domain_reason=str(data["domain_reason"]),
        raw=raw if isinstance(raw, dict) else {},
    )


def candidate_span_to_dict(span: CandidateSpan) -> dict[str, Any]:
    return {
        "primary_sentence_id": span.primary_sentence_id,
        "span_sentence_ids": list(span.span_sentence_ids),
        "span_text": span.span_text,
        "candidate_reason": span.candidate_reason,
        "reason_category": span.reason_category,
        "context_needed": span.context_needed,
        "uncertainty_note": span.uncertainty_note,
        "raw": span.raw,
    }


def candidate_span_from_dict(data: dict[str, Any]) -> CandidateSpan:
    raw = data.get("raw")
    ids_raw = data.get("span_sentence_ids")
    if not isinstance(ids_raw, list):
        raise ValueError("cached candidate span is missing span_sentence_ids")
    return CandidateSpan(
        primary_sentence_id=int(data["primary_sentence_id"]),
        span_sentence_ids=[int(value) for value in ids_raw],
        span_text=str(data["span_text"]),
        candidate_reason=str(data["candidate_reason"]),
        reason_category=str(data["reason_category"]),
        context_needed=bool(data["context_needed"]),
        uncertainty_note=str(data["uncertainty_note"]),
        raw=raw if isinstance(raw, dict) else {},
    )


def window_response_to_dict(response: WindowResponse) -> dict[str, Any]:
    return {
        "spans": [candidate_span_to_dict(span) for span in response.spans],
        "raw": response.raw,
    }


def window_response_from_dict(data: dict[str, Any]) -> WindowResponse:
    spans_raw = data.get("spans")
    raw = data.get("raw")
    if not isinstance(spans_raw, list):
        raise ValueError("cached window response is missing spans")
    return WindowResponse(
        spans=[candidate_span_from_dict(item) for item in spans_raw],
        raw=raw if isinstance(raw, dict) else {},
    )


def window_call_result_to_dict(result: WindowCallResult) -> dict[str, Any]:
    return {
        "window": llm_window_to_dict(result.window),
        "response": window_response_to_dict(result.response),
        "owned_spans": [candidate_span_to_dict(span) for span in result.owned_spans],
        "dropped_unowned": result.dropped_unowned,
    }


def window_call_result_from_dict(data: dict[str, Any]) -> WindowCallResult:
    owned_raw = data.get("owned_spans")
    if not isinstance(owned_raw, list):
        raise ValueError("cached window call result is missing owned_spans")
    return WindowCallResult(
        window=llm_window_from_dict(data["window"]),
        response=window_response_from_dict(data["response"]),
        owned_spans=[candidate_span_from_dict(item) for item in owned_raw],
        dropped_unowned=int(data["dropped_unowned"]),
    )


def article_llm_result_to_dict(result: ArticleLLMResult) -> dict[str, Any]:
    return {
        "domain": (
            domain_response_to_dict(result.domain) if result.domain is not None else None
        ),
        "windows": [window_call_result_to_dict(window) for window in result.windows],
        "unique_spans": [candidate_span_to_dict(span) for span in result.unique_spans],
        "dedupe_drops": result.dedupe_drops,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
    }


def article_llm_result_from_dict(data: dict[str, Any]) -> ArticleLLMResult:
    windows_raw = data.get("windows")
    spans_raw = data.get("unique_spans")
    if not isinstance(windows_raw, list):
        raise ValueError("cached article result is missing windows")
    if not isinstance(spans_raw, list):
        raise ValueError("cached article result is missing unique_spans")
    domain_raw = data.get("domain")
    return ArticleLLMResult(
        domain=(
            domain_response_from_dict(domain_raw)
            if isinstance(domain_raw, dict)
            else None
        ),
        windows=[window_call_result_from_dict(item) for item in windows_raw],
        unique_spans=[candidate_span_from_dict(item) for item in spans_raw],
        dedupe_drops=int(data.get("dedupe_drops", 0)),
        total_input_tokens=int(data.get("total_input_tokens", 0)),
        total_output_tokens=int(data.get("total_output_tokens", 0)),
    )


class LLMResultCache:
    """JSON-file cache keyed by article/model/prompt/extraction inputs."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    @staticmethod
    def build_key(
        *,
        article_fingerprint: str,
        cleaned_text_fingerprint: str,
        sentence_text_fingerprint: str,
        query_domain_hint: str,
        provider_name: str,
        model_api_id: str,
        candidate_prompt_version: str,
        domain_prompt_version: str,
        temperature: float,
        target_sentences_per_call: int,
        hard_max_sentences_per_call: int,
        overlap: int,
    ) -> str:
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "article_fingerprint": article_fingerprint,
            "cleaned_text_fingerprint": cleaned_text_fingerprint,
            "sentence_text_fingerprint": sentence_text_fingerprint,
            "query_domain_hint": query_domain_hint,
            "provider_name": provider_name,
            "model_api_id": model_api_id,
            "candidate_prompt_version": candidate_prompt_version,
            "domain_prompt_version": domain_prompt_version,
            "temperature": temperature,
            "windowing_config": {
                "target_sentences_per_call": target_sentences_per_call,
                "hard_max_sentences_per_call": hard_max_sentences_per_call,
                "overlap": overlap,
            },
        }
        encoded = json.dumps(
            to_jsonable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> ArticleLLMResult | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
                return None
            if payload.get("key") != key:
                return None
            result = payload.get("result")
            if not isinstance(result, dict):
                return None
            return article_llm_result_from_dict(result)
        except Exception as exc:
            LOGGER.debug("Treating malformed LLM cache file as miss: %s (%s)", path, exc)
            return None

    def put(
        self,
        key: str,
        result: ArticleLLMResult,
        metadata: dict[str, Any],
    ) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for_key(key)
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "created_at_utc": now_utc_iso(),
            "key": key,
            "metadata": metadata,
            "result": article_llm_result_to_dict(result),
        }
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fp:
            json.dump(
                to_jsonable(payload),
                fp,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            fp.write("\n")
        os.replace(tmp_path, path)
        return path

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
