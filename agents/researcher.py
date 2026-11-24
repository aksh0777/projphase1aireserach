"""
Researcher Agent with Parallel Fan-Out Execution
Architecture Reference: docs/EXPLANATIONS.md - Section 5 and Chapter 13

First Principle:
Divide-and-Conquer (Map-Reduce Retrieval): parallel worker nodes execute isolated
retrieval and extraction passes. Every execution emits a provenance-tagged
RESEARCH RECORD (run_id, sub_question, sources, findings, status) so the
validation gate can verify that findings belong to the right sub-question
and the right run.

Integrity: in LIVE mode a failed search/LLM call yields a record with
status != "ok" (never invented evidence). Mock text is generic, labelled, and
built only from this sub-question - it contains no domain-specific content.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from agents.llm import LLMUnavailableError, call_llm, is_live_mode, usage_record
from state import stable_key
from tools.search_tool import search_web

logger = logging.getLogger("ResearcherAgent")

_URL_TAG = re.compile(r"\[(https?://[^\]\s]+)\]")


def _mock_research_generator(sub_question: str, hits: List[Dict[str, str]]) -> str:
    """Generic, clearly-labelled placeholder findings built only from this sub-question."""
    urls = [h["url"] for h in hits]
    if not urls:
        return ""
    u0, u1 = urls[0], urls[min(1, len(urls) - 1)]
    return (
        f"- [MOCK] Placeholder finding 1 for the sub-question \"{sub_question}\" - no language model was called [{u0}].\n"
        f"- [MOCK] Placeholder finding 2 for the sub-question \"{sub_question}\" - generated without real evidence [{u1}]."
    )


def _parse_findings(content: str, hit_urls: set) -> Dict[str, Any]:
    """Extract bullets; keep only bullets with a citation URL; flag citations that were not retrieved."""
    findings: List[str] = []
    unverified: List[str] = []
    dropped_uncited = 0
    for line in content.strip().splitlines():
        s = line.strip()
        if not (s.startswith(("-", "*", "•")) or re.match(r"^\d+\.", s)):
            continue
        cleaned = re.sub(r"^[-*•\d\.]+\s*", "", s).strip()
        if not cleaned:
            continue
        urls = _URL_TAG.findall(cleaned)
        if not urls:
            dropped_uncited += 1  # never attribute an uncited claim to an arbitrary source
            continue
        bad = [u for u in urls if u not in hit_urls]
        if bad:
            unverified.extend(bad)
            continue
        findings.append(cleaned)
    return {"findings": findings, "unverified_citations": unverified, "dropped_uncited": dropped_uncited}


def research_subquestion(
    sub_question: str,
    run_id: str = "",
    question: str = "",
    retry: bool = False,
) -> Dict[str, Any]:
    """
    Execute search and evidence extraction for ONE sub-question of ONE run.
    Designed for parallel execution across LangGraph worker nodes.
    """
    clean_sub_q = sub_question.strip()
    key = stable_key(clean_sub_q)
    live = is_live_mode()

    search_query = clean_sub_q
    if retry and question:
        search_query = f"{clean_sub_q} ({question})"[:300]

    hits = search_web(search_query, max_results=5, allow_simulated=not live, simulate_only=not live)
    provenance = (hits[0].get("provenance", "live") if hits else "none")
    hit_urls = {h["url"] for h in hits}

    record: Dict[str, Any] = {
        "run_id": run_id,
        "question": question,
        "sub_question": clean_sub_q,
        "status": "ok",
        "mode": "live" if live else "mock",
        "search_provenance": provenance,
        "findings": [],
        "sources": hits,
        "error": "",
        "retry": retry,
        "unverified_citations": [],
        "dropped_uncited": 0,
    }
    meta: Dict[str, Any] = {"search_count": 1, "llm_calls": []}
    notes: Dict[str, List[str]] = {}

    if not hits:
        record["status"] = "no_evidence"
        record["error"] = "Search returned no usable sources."
    else:
        sources_text = ""
        for idx, hit in enumerate(hits, 1):
            sources_text += f"[{idx}] Title: {hit['title']}\n    URL: {hit['url']}\n    Snippet: {hit['snippet']}\n\n"

        system_instruction = (
            "You are an objective academic research analyst. Extract concise, factual bullet-point findings "
            "addressing the research question, strictly based on the provided search snippets.\n"
            "RULES:\n"
            "1. Extract 3 to 5 clear, high-density factual bullet points.\n"
            "2. EVERY single bullet point MUST end with its exact source URL formatted as [source URL].\n"
            "3. Use ONLY URLs that appear in the provided search evidence.\n"
            "4. Emphasize empirical metrics, dates, and concrete facts relevant to the question.\n"
            "5. Do not speculate beyond the evidence provided. If the evidence is irrelevant, say so in one bullet "
            "starting with 'NO RELEVANT EVIDENCE' and still cite a URL."
        )
        ctx = f"Overall research topic: {question}\n" if question else ""
        prompt = (
            f"{ctx}Research Question: {clean_sub_q}\n\n"
            f"Search Evidence:\n{sources_text}\n\n"
            "Synthesize the evidence into grounded bullet points with source URLs:"
        )

        try:
            content, tokens = call_llm(
                prompt=prompt,
                system_instruction=system_instruction,
                mock_generator=lambda p: _mock_research_generator(clean_sub_q, hits),
                node="researcher",
            )
            meta["llm_calls"].append(usage_record("researcher", tokens))
            meta[f"tokens_researcher_{key}"] = tokens["total_tokens"]
            parsed = _parse_findings(content, hit_urls)
            record["findings"] = parsed["findings"]
            record["unverified_citations"] = parsed["unverified_citations"]
            record["dropped_uncited"] = parsed["dropped_uncited"]
            if not parsed["findings"]:
                record["status"] = "no_valid_findings"
                record["error"] = "No bullet carried a citation to a retrieved source."
        except LLMUnavailableError as exc:
            record["status"] = "failed"
            record["error"] = str(exc)[:300]

    if record["status"] == "ok":
        notes[clean_sub_q] = list(record["findings"])

    banner = [
        "", "=" * 60,
        f"========== [2] RESEARCHER OUTPUT ({clean_sub_q[:45]}...) ==========",
        "=" * 60,
        f"Sub-Question: {clean_sub_q}",
        f"Run: {run_id or 'n/a'} | Status: {record['status']} | Search provenance: {provenance}",
        f"Sources Queried: {len(hits)} results",
        "Key Findings Extracted:",
    ]
    for note in record["findings"]:
        banner.append(f"  • {note}")
    if record["error"]:
        banner.append(f"  ! {record['error']}")
    banner += ["=" * 60, ""]
    print("\n".join(banner))

    return {
        "search_results": {clean_sub_q: hits},
        "notes": notes,
        "research_records": [record],
        "metadata": meta,
    }
