"""Tiered excluded-domain detector (host blocklist + URL-path blocklist)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ExcludedDomainConfig
from .utils import url_host, url_path


@dataclass(frozen=True)
class ExclusionVerdict:
    """Result of running the excluded-domain detector against a URL."""

    rejected: bool
    bucket: str | None
    tier: str | None  # "host" | "path" | None
    matched: str | None  # the host or path prefix that matched

    @property
    def reason(self) -> str | None:
        if not self.rejected:
            return None
        if self.tier == "host":
            return f"host_blocklist:{self.bucket}:{self.matched}"
        if self.tier == "path":
            return f"path_blocklist:{self.bucket}:{self.matched}"
        return f"excluded:{self.bucket}"


_ADMIT = ExclusionVerdict(False, None, None, None)


def evaluate_url(url: str | None, cfg: ExcludedDomainConfig) -> ExclusionVerdict:
    """Decide whether ``url`` should be rejected as belonging to an excluded bucket.

    Default-admit / reject-list policy:

    1. Tier 1 host blocklist — if the canonical host (or any parent suffix) is
       in ``cfg.hosts``, reject.
    2. Tier 2 path blocklist — if the URL path starts with any prefix in
       ``cfg.paths``, reject.
    3. Otherwise admit.
    """
    if not url:
        return _ADMIT

    host = url_host(url)
    if host:
        host_l = host.lower()
        for bucket, members in cfg.hosts.items():
            for member in members:
                if not member:
                    continue
                if host_l == member or host_l.endswith("." + member):
                    return ExclusionVerdict(True, bucket, "host", member)

    path = url_path(url)
    if path:
        path_l = path.lower()
        for bucket, prefixes in cfg.paths.items():
            for prefix in prefixes:
                prefix_l = prefix.lower()
                if not prefix_l:
                    continue
                if not prefix_l.startswith("/"):
                    prefix_l = "/" + prefix_l
                # Tier 2 wants a true path-segment prefix match: "/markets/" must
                # match "/markets" or "/markets/foo" but NOT "/marketsplace".
                trimmed = prefix_l.rstrip("/")
                if path_l == trimmed or path_l.startswith(trimmed + "/"):
                    return ExclusionVerdict(True, bucket, "path", prefix)

    return _ADMIT
