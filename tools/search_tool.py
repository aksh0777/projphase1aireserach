"""
Web Search Tool for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 3 and Chapter 13

First Principle:
Grounding: an LLM is parametric memory frozen at training time. An external
search tool supplies non-parametric retrieval tokens, grounding generation in
verifiable reality.

Integrity rule (Chapter 13): fabricated "simulated" sources are ONLY produced
when the caller explicitly allows them (MOCK mode). In LIVE mode a failed
search returns an empty list, so the researcher reports a failed sub-question
instead of silently grounding the report on invented URLs.
Every hit carries ``provenance``: "live" or "simulated".
"""

import logging
import time
from typing import Dict, List

# Handle various distributions (prefer modern ddgs package)
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        DDGS = None  # Handled gracefully at runtime

logger = logging.getLogger("SearchTool")

# Domains that only ever appear in simulated/mock sources (used to audit legacy data).
SIMULATED_SOURCE_DOMAINS = (
    "research-index.org",
    "standards-tracker.net",
    "energy-storage-review.org",
    "bloomberg-energy.com/bess-cost-parity",
    "iea.org/reports/battery-storage-outlook-2026",
)


def _simulated_search(query: str, max_results: int) -> List[Dict[str, str]]:
    """Deterministic placeholder sources for MOCK mode only. Clearly tagged provenance='simulated'."""
    import hashlib

    cleaned = query.strip()
    h = int(hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:6], 16)
    return [
        {
            "title": f"[SIMULATED] Placeholder source A: {cleaned}",
            "url": f"https://research-index.org/papers/{h % 10000}",
            "snippet": f"Simulated placeholder snippet echoing the query: {cleaned}.",
            "provenance": "simulated",
        },
        {
            "title": f"[SIMULATED] Placeholder source B: {cleaned}",
            "url": f"https://standards-tracker.net/reports/2026/{h % 5000}",
            "snippet": f"Second simulated placeholder snippet echoing the query: {cleaned}.",
            "provenance": "simulated",
        },
    ][:max_results]


def search_web(
    query: str,
    max_results: int = 5,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: float = 8.0,
    allow_simulated: bool = False,
    simulate_only: bool = False,
) -> List[Dict[str, str]]:
    """
    Execute a web search with retry + exponential backoff.

    Returns a list of {"title", "url", "snippet", "provenance"} dicts.
    On total failure returns [] unless ``allow_simulated`` (MOCK mode) is True.
    ``simulate_only`` skips the network entirely (used by forced-MOCK runs and tests).
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()

    if simulate_only:
        return _simulated_search(clean_query, max_results)

    for attempt in range(1, max_retries + 1):
        try:
            if DDGS is not None:
                with DDGS(timeout=timeout) as ddgs:
                    raw_results = list(ddgs.text(clean_query, max_results=max_results))
                    normalized: List[Dict[str, str]] = []
                    for item in raw_results:
                        title = (item.get("title") or "Untitled Source").strip()
                        url = item.get("href") or item.get("link") or item.get("url") or ""
                        snippet = item.get("body") or item.get("snippet") or item.get("abstract") or ""
                        if url and snippet:
                            normalized.append({
                                "title": title,
                                "url": url.strip(),
                                "snippet": snippet.strip(),
                                "provenance": "live",
                            })
                    if normalized:
                        return normalized[:max_results]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search attempt %d/%d for '%s' failed: %s", attempt, max_retries, clean_query, str(exc))

        if attempt < max_retries:
            time.sleep(backoff_factor ** attempt)

    if allow_simulated:
        logger.info("MOCK mode: using simulated placeholder sources for '%s'", clean_query)
        return _simulated_search(clean_query, max_results)

    logger.warning("Live search returned no usable results for '%s' (no simulated fallback in LIVE mode).", clean_query)
    return []
