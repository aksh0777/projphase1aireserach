"""
Synthesizer Agent for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 6 and Chapter 13

First Principle:
Information Synthesis & Entropy Reduction: parallel workers produce overlapping and
potentially contradictory evidence. The Synthesizer reconciles discrepancies and
structures the findings into a cohesive brief.

Isolation: the Synthesizer consumes ONLY research records stamped with the current
run_id (see ``state.current_run_records``). Records from any other run are dropped
and reported in metadata['isolation_violations'].
"""

import logging
import re
from typing import Any, Dict, List

from agents.llm import LLMUnavailableError, call_llm, usage_record
from state import ResearchState, current_run_records

logger = logging.getLogger("SynthesizerAgent")


def _mock_synthesis_generator(topic: str, notes_dict: Dict[str, List[str]]) -> str:
    """Generic labelled brief assembled strictly from this run's notes (no canned domain content)."""
    all_notes: List[str] = []
    sources: List[str] = []
    for _, n_list in notes_dict.items():
        for note in n_list:
            all_notes.append(note)
            for u in re.findall(r"\[(https?://[^\]]+)\]", note):
                if u not in sources:
                    sources.append(u)
    return (
        f"# Research Brief: {topic}\n\n"
        "> MOCK EXECUTION: this brief was assembled without a language model.\n\n"
        "## 1. Executive Synthesis\nPlaceholder synthesis of the findings below.\n\n"
        "## 2. Thematic Analysis\n" + "\n".join(f"- {n}" for n in all_notes[:8]) + "\n\n"
        "## 3. Cross-Source Contradictions & Divergences\nNot assessed in MOCK mode.\n\n"
        "## 4. Consolidated Source Index\n" + "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(sources))
    )


def synthesize_findings(state: ResearchState) -> Dict[str, Any]:
    """Synthesize notes from this run's researchers into a coherent research brief."""
    topic = state.get("question", "")
    valid, foreign = current_run_records(state)
    notes_dict: Dict[str, List[str]] = {r["sub_question"]: r["findings"] for r in valid if r.get("findings")}

    meta: Dict[str, Any] = {"llm_calls": []}
    if foreign:
        meta["isolation_violations"] = [
            {"node": "synthesizer", "record_run_id": r.get("run_id"), "sub_question": r.get("sub_question")} for r in foreign
        ]

    if not notes_dict:
        print("\n[SYNTHESIZER] No valid research evidence for this run; skipping synthesis.\n")
        return {"brief": "", "metadata": meta}

    formatted_notes = ""
    for sq, findings in notes_dict.items():
        formatted_notes += f"### Sub-Question: {sq}\n"
        for item in findings:
            formatted_notes += f"- {item}\n"
        formatted_notes += "\n"

    system_instruction = (
        "You are a rigorous academic research synthesizer. Process findings collected across multiple parallel "
        "research nodes, eliminate redundant bullet points, detect and explicitly flag contradictions between "
        "sources, and compile an organized, cohesive research brief. Use ONLY the findings provided; do not add "
        "outside facts and do not drift to topics other than the stated research topic.\n\n"
        "Structure your brief with these exact sections:\n"
        "1. Executive Synthesis (macro-level summary of findings)\n"
        "2. Thematic Analysis (structured by major research sub-questions)\n"
        "3. Cross-Source Contradictions & Divergences (explicitly contrast conflicting data, ranges, or interpretations)\n"
        "4. Consolidated Source Index (all referenced URLs)"
    )
    prompt = (
        f"Research Topic: {topic}\n\n"
        f"Aggregated Findings From Researcher Agents:\n{formatted_notes}\n\n"
        "Compile the deduplicated research brief and flag any contradictions:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_synthesis_generator(topic, notes_dict),
        node="synthesizer",
    )
    meta["llm_calls"].append(usage_record("synthesizer", tokens))
    meta["tokens_synthesizer"] = tokens["total_tokens"]

    print("\n".join(["", "=" * 60, "========== [3] SYNTHESIZER BRIEF ==========", "=" * 60, content.strip(), "=" * 60, ""]))
    return {"brief": content.strip(), "metadata": meta}
