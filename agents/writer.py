"""
Writer Agent for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 7 and Chapter 13

First Principle:
Grounding & Verifiable Attribution: every factual claim is anchored to numbered
in-text citations ([1], [2]) mapping to a verifiable bibliography.

Isolation / integrity:
* The Writer consumes ONLY the validated evidence of the current run
  (``validated_notes`` / ``validated_search_results``) and the run's own question.
* There is no hard-coded domain content anywhere in this module. The MOCK report is
  built from the run's own plan and notes, and starts with an unmistakable banner.
* Sub-questions that failed validation are declared in an "Evidence Gaps" section
  instead of being papered over.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Set

from agents.llm import call_llm, usage_record
from state import ResearchState

logger = logging.getLogger("WriterAgent")

MOCK_BANNER = (
    "> ⚠️ **MOCK EXECUTION** — this report was produced by placeholder generators "
    "(no language model and no real sources). It must not be used as research evidence."
)


def _extract_sources(notes_dict: Dict[str, List[str]], search_results: Dict[str, List[Dict[str, str]]]) -> List[str]:
    """Deduplicated URLs: cited-in-notes first, then retrieved hits - for THIS run's validated sub-questions only."""
    urls: List[str] = []
    seen: Set[str] = set()
    for _, n_list in notes_dict.items():
        for note in n_list:
            for u in re.findall(r"\[(https?://[^\]]+)\]", note):
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
    for _, hits in search_results.items():
        for hit in hits:
            u = hit.get("url", "")
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
    return urls


def _mock_report_generator(topic: str, plan: List[str], notes: Dict[str, List[str]], sources: List[str]) -> str:
    """Generic report assembled from this run's own plan + notes. Contains no domain-specific boilerplate."""
    idx_of = {u: i + 1 for i, u in enumerate(sources)}
    sections = []
    for i, sq in enumerate(plan, 1):
        bullets = []
        for n in notes.get(sq, []):
            cites = [idx_of[u] for u in re.findall(r"\[(https?://[^\]]+)\]", n) if u in idx_of]
            text = re.sub(r"\s*\[https?://[^\]]+\]", "", n).strip()
            bullets.append(f"- {text} " + "".join(f"[{c}]" for c in cites))
        sections.append(f"## {i}. {sq}\n\n" + ("\n".join(bullets) if bullets else "_No validated evidence._"))
    refs = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(sources)) or "_No sources._"
    return (
        f"# Deep Research Report: {topic}\n\n{MOCK_BANNER}\n\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
        "## Introduction & Context\nPlaceholder introduction (MOCK).\n\n"
        + "\n\n".join(sections)
        + "\n\n## Synthesis & Strategic Conclusion\nPlaceholder conclusion (MOCK).\n\n"
        f"## References\n{refs}\n"
    )


def _evidence_gap_section(failed: List[Dict[str, str]]) -> str:
    if not failed:
        return ""
    lines = ["", "## Evidence Gaps", "The following sub-questions could not be supported by validated evidence in this run:"]
    for f in failed:
        lines.append(f"- {f.get('sub_question', '?')} — {f.get('reason', 'validation failed')}")
    return "\n".join(lines) + "\n"


def write_report(state: ResearchState) -> Dict[str, Any]:
    """
    Generate the academic markdown report from the current run's validated evidence,
    save it to outputs/report_<timestamp>_<run suffix>.md and register draft version 0.
    """
    run_id = state.get("run_id", "")
    topic = state.get("question", "")
    brief = state.get("brief", "")
    validated = state.get("validated_notes")
    if validated is None or not state.get("validation"):
        validated = state.get("notes", {})
        results = state.get("search_results", {})
    else:
        results = state.get("validated_search_results", {})
    plan = [q for q in state.get("plan", []) if q in validated] or list(validated.keys())
    failed = state.get("failed_subquestions", [])

    sources = _extract_sources(validated, results)
    sources_catalog = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(sources))

    system_instruction = (
        "You are an authoritative academic scientific author. Author a formal, comprehensive "
        "markdown research report synthesizing ONLY the provided evidence, answering the stated research question.\n\n"
        "MANDATORY REPORT STRUCTURE:\n"
        "# [Full Comprehensive Title]\n\n"
        "## Introduction & Context\n"
        "## [Dedicated Section For Each Sub-Question in the Research Plan]\n"
        "## Synthesis & Strategic Conclusion\n"
        "## References\n\n"
        "STRICT RULES:\n"
        "1. Every factual statement, metric and percentage MUST carry a numbered citation like [1], [2].\n"
        "2. The numbers MUST correspond EXACTLY to the numbered references in the bibliography catalog.\n"
        "3. Every sub-question from the plan MUST have its own detailed heading and analysis.\n"
        "4. Use only the provided brief and sources. Do not add outside facts or content about any other topic.\n"
        "5. Complete every section; do not stop mid-sentence."
    )
    plan_str = "\n".join(f"- {q}" for q in plan)
    prompt = (
        f"Research Question (the report must answer exactly this): {topic}\n\n"
        f"Research Plan (Sub-Questions to cover):\n{plan_str}\n\n"
        f"Synthesized Research Brief:\n{brief}\n\n"
        f"Available Verified Source Catalog:\n{sources_catalog}\n\n"
        "Draft the full academic markdown research report with inline numbered citations and complete references:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_report_generator(topic, plan, validated, sources),
        node="writer",
    )
    content = content.strip()
    if tokens.get("mode") != "live" and MOCK_BANNER not in content:
        content = content.replace("\n", "\n\n" + MOCK_BANNER + "\n", 1)

    gap = _evidence_gap_section(failed)
    if "## References" in content:
        content = content.replace("## References", gap + "\n## References", 1) if gap else content
    else:
        content += gap + f"\n\n## References\n{sources_catalog}\n"

    os.makedirs("outputs", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_id.split("_")[-1] if run_id else "norun"
    report_path = os.path.join("outputs", f"report_{stamp}_{suffix}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    print("\n".join(["", "=" * 60, "========== [4] FINAL REPORT ==========", "=" * 60,
                     f"Saved to: {report_path}", "", content.strip(), "=" * 60, ""]))

    return {
        "report": content.strip(),
        "draft": content.strip(),
        "draft_history": [{"version": 0, "author": "writer", "text": content.strip(), "chars": len(content.strip()), "run_id": run_id}],
        "metadata": {
            "tokens_writer": tokens["total_tokens"],
            "report_path": report_path,
            "llm_calls": [usage_record("writer", tokens)],
        },
    }
