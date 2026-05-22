"""Config + env loading for the ETL pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_DIR / "config"

DEFAULT_INPUT_CACHE_DIR = REPO_ROOT / "input_csvs" / "newsapi_cache"
DEFAULT_STATE_DIR = REPO_ROOT / "pipeline_state"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "annotators(1)"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "annotators"

ALLOWED_TOP_LEVEL_DOMAINS = {"health", "policy", "sport"}
EXCLUDED_BUCKETS = {"financial", "finance", "weather", "climate"}
MISC_SUBTOPIC_RE = re.compile(r"^misc-[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp)
    return loaded or {}


@dataclass
class LLMSettings:
    """Resolved LLM configuration (provider + extraction knobs)."""

    provider: str
    model_api_id: str
    model_display_name: str
    api_key_env: str
    candidate: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    def candidate_param(self, key: str, default: Any = None) -> Any:
        return self.candidate.get(key, default)

    def domain_param(self, key: str, default: Any = None) -> Any:
        return self.domain.get(key, default)


@dataclass
class ExcludedDomainConfig:
    """Tier 1 host blocklist + Tier 2 path blocklist."""

    hosts: dict[str, set[str]]
    paths: dict[str, list[str]]

    def all_hosts(self) -> set[str]:
        out: set[str] = set()
        for bucket in self.hosts.values():
            out.update(bucket)
        return out

    def host_bucket(self, host: str | None) -> str | None:
        if not host:
            return None
        host_l = host.lower()
        for bucket, members in self.hosts.items():
            for member in members:
                if host_l == member or host_l.endswith("." + member):
                    return bucket
        return None

    def path_bucket(self, path: str | None) -> str | None:
        if not path:
            return None
        path_l = path.lower()
        if not path_l.startswith("/"):
            path_l = "/" + path_l
        for bucket, prefixes in self.paths.items():
            for prefix in prefixes:
                prefix_l = prefix.lower()
                if not prefix_l.startswith("/"):
                    prefix_l = "/" + prefix_l
                if path_l == prefix_l.rstrip("/") or path_l.startswith(prefix_l):
                    return bucket
        return None


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration container."""

    repo_root: Path
    input_cache_dir: Path
    state_dir: Path
    output_dir: Path
    reference_dir: Path
    llm: LLMSettings
    excluded: ExcludedDomainConfig
    raw_llm_yaml: dict[str, Any] = field(default_factory=dict)


def _load_excluded_domains() -> ExcludedDomainConfig:
    hosts_yaml = _load_yaml(CONFIG_DIR / "excluded_hosts.yaml")
    paths_yaml = _load_yaml(CONFIG_DIR / "excluded_paths.yaml")

    hosts: dict[str, set[str]] = {}
    for bucket, entries in hosts_yaml.items():
        hosts[bucket.lower()] = {
            entry.strip().lower().lstrip(".")
            for entry in (entries or [])
            if isinstance(entry, str) and entry.strip()
        }

    paths: dict[str, list[str]] = {}
    for bucket, entries in paths_yaml.items():
        normalized: list[str] = []
        for entry in entries or []:
            if not isinstance(entry, str):
                continue
            entry = entry.strip().lower()
            if not entry:
                continue
            if not entry.startswith("/"):
                entry = "/" + entry
            normalized.append(entry)
        paths[bucket.lower()] = normalized

    # Sanity: ensure the excluded buckets named in the plan are at least
    # defined (even if empty) so callers can rely on the keys.
    for bucket in ("financial", "weather", "climate"):
        hosts.setdefault(bucket, set())
        paths.setdefault(bucket, [])

    return ExcludedDomainConfig(hosts=hosts, paths=paths)


def _load_llm_settings() -> LLMSettings:
    raw = _load_yaml(CONFIG_DIR / "llm.yaml")
    provider = raw.get("default_provider", "gemini")
    providers = raw.get("providers", {}) or {}
    provider_cfg = providers.get(provider, {}) or {}

    return LLMSettings(
        provider=provider,
        model_api_id=str(provider_cfg.get("model", "")),
        model_display_name=str(
            provider_cfg.get("model_display_name", "Gemini 3.1 Flash-Lite")
        ),
        api_key_env=str(provider_cfg.get("api_key_env", "GEMINI_API_KEY")),
        candidate=raw.get("candidate_extraction", {}) or {},
        domain=raw.get("domain_assignment", {}) or {},
    )


def load_config(
    *,
    input_cache_dir: Path | None = None,
    state_dir: Path | None = None,
    output_dir: Path | None = None,
    reference_dir: Path | None = None,
    load_dotenv_file: bool = True,
) -> PipelineConfig:
    """Load configuration from yaml + .env (optional)."""
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env", override=False)
        except ImportError:
            pass

    return PipelineConfig(
        repo_root=REPO_ROOT,
        input_cache_dir=Path(input_cache_dir or DEFAULT_INPUT_CACHE_DIR),
        state_dir=Path(state_dir or DEFAULT_STATE_DIR),
        output_dir=Path(output_dir or DEFAULT_OUTPUT_DIR),
        reference_dir=Path(reference_dir or DEFAULT_REFERENCE_DIR),
        llm=_load_llm_settings(),
        excluded=_load_excluded_domains(),
        raw_llm_yaml=_load_yaml(CONFIG_DIR / "llm.yaml"),
    )


def is_allowed_query_domain(value: str | None) -> bool:
    """Return True if ``value`` is one of the allowed canonical domains."""
    if not value:
        return False
    if value in ALLOWED_TOP_LEVEL_DOMAINS:
        return True
    if MISC_SUBTOPIC_RE.match(value):
        # also reject misc forms of excluded buckets
        suffix = value[len("misc-"):]
        for excluded in EXCLUDED_BUCKETS:
            if suffix == excluded or suffix.startswith(excluded + "-"):
                return False
        return True
    return False


def is_excluded_query_domain(value: str | None) -> bool:
    """Return True if ``value`` matches an excluded bucket (or misc-* of one)."""
    if not value:
        return False
    if value in EXCLUDED_BUCKETS:
        return True
    if MISC_SUBTOPIC_RE.match(value):
        suffix = value[len("misc-"):]
        for excluded in EXCLUDED_BUCKETS:
            if suffix == excluded or suffix.startswith(excluded + "-"):
                return True
    return False
