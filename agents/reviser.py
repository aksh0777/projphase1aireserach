"""
Reviser Agent for Multi-Agent Deep Research System (Phase 2)
Architecture Reference: docs/EXPLANATIONS.md - Section 11: "The Critic & Revision Loop"

First Principle:
Iterative Refinement & Error Gradient Descent:
In a single-pass feed-forward architecture, generation errors are immutable.
The Reviser takes targeted feedback from the Critic and updates the document draft,
closing the loop between evaluation and generation to systematically raise output quality.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List
from agents.llm import call_llm
from state import ResearchState

logger = logging.getLogger("ReviserAgent")


def _mock_reviser_generator(question: str, draft: str, flaws: List[str]) -> str:
    """Mock draft reviser for resilient demonstration mode."""
    flaw_notes = "\n".join([f"- Addressed flaw: {f}" for f in flaws])
    revision_header = (
        f"\n\n<!-- Peer Review Revisions Applied -->\n"
        f"<!--\n{flaw_notes}\n-->\n"
    )

    # Enhance draft by adding targeted refinement
    refined_draft = draft + revision_header
    if "## Strategic Synthesis & Conclusion" in refined_draft:
        refined_draft = refined_draft.replace(
            "## Strategic Synthesis & Conclusion",
            "### Peer-Review Quality Refinement\n"
            "This revised iteration addresses reviewer critiques by verifying cross-source numerical parity, "
            "standardizing citation bindings, and strengthening technical boundary definitions.\n\n"
            "## Strategic Synthesis & Conclusion"
        )
    return refined_draft


def revise_draft(state: ResearchState) -> Dict[str, Any]:
    """
    Revise the current draft report based on targeted critique flaws and feedback.
    CRITICAL RULE: The reviser does NOT receive raw search notes, operating strictly
    on the current draft, research question, and critic feedback.
    Outputs the complete updated markdown report draft.
    """
    question = state.get("question", "").strip()
    draft = state.get("draft") or state.get("report", "").strip()
    critique = state.get("critique", {})
    flaws = critique.get("flaws", [])
    scores = critique.get("scores", {})
    prev_rev_count = state.get("revision_count", 0)
    current_iter = prev_rev_count + 1

    flaws_text = "\n".join([f"[{i+1}] {f}" for i, f in enumerate(flaws)]) if flaws else "Address general analytical depth."
    scores_text = ", ".join([f"{k}: {v}/5" for k, v in scores.items()])

    system_instruction = (
        "You are an expert scientific editor revising an academic research report based on peer-review critique.\n\n"
        "MANDATORY REVISION RULES:\n"
        "1. You MUST address every specific flaw identified by the reviewer.\n"
        "2. Output the FULL REVISED REPORT in complete markdown. Do NOT output a diff, patch, or partial snippet.\n"
        "3. Preserve the core formal report structure:\n"
        "   - Title (# ...)\n"
        "   - Introduction & Context (## ...)\n"
        "   - Dedicated sections for each sub-domain\n"
        "   - Strategic Synthesis & Conclusion (## ...)\n"
        "   - References (## References)\n"
        "4. Retain all existing valid citations ([1], [2], etc.) and add precise citations for any newly introduced factual claims.\n"
        "5. Ensure numbered in-text citations map accurately to the references list.\n"
        "6. Elevate academic clarity, precision of technical metrics, and economic granularity."
    )

    prompt = (
        f"Research Question: {question}\n\n"
        f"Peer Review Assessment (Scores: {scores_text}):\n"
        f"Specific Flaws to Fix:\n{flaws_text}\n\n"
        f"Current Report Draft:\n{draft}\n\n"
        "Produce the full, revised, production-grade academic research report:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_reviser_generator(question, draft, flaws),
    )

    revised_draft = content.strip()
    # Ensure references section remains present
    if "## References" not in revised_draft and "# References" not in revised_draft:
        # Carry over references from original draft if dropped
        match = re.search(r"## References.*", draft, re.DOTALL)
        if match:
            revised_draft += "\n\n" + match.group(0)

    # Print DEMO-READY terminal banner
    banner = [
        "",
        "=" * 60,
        f"========== ITERATION {current_iter}: REVISER ==========",
        "=" * 60,
        f"Research Topic: {question}",
        f"Flaws Addressed: {len(flaws)}",
        f"Prior Scores: {scores_text}",
        f"Revised Draft Length: {len(revised_draft)} characters",
        "=" * 60,
        "",
    ]
    print("\n".join(banner))

    return {
        "draft": revised_draft,
        "revision_count": current_iter,
        "metadata": {
            f"tokens_reviser_iter_{current_iter}": tokens["total_tokens"],
        },
    }
