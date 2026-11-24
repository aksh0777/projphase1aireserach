"""
Reviser Agent for Multi-Agent Deep Research System (Phase 2)
Architecture Reference: docs/EXPLANATIONS.md - Section 11 and Chapter 13

First Principle:
Iterative Refinement: the Reviser takes targeted feedback from the Critic and updates the
document draft, closing the loop between evaluation and generation.

The Reviser receives the ORIGINAL QUESTION, the current draft and the critique only (no search
notes). Every revised draft is appended to ``draft_history`` so no version is ever lost.
"""

import logging
import re
from typing import Any, Dict, List

from agents.llm import LLMUnavailableError, call_llm, usage_record
from state import ResearchState

logger = logging.getLogger("ReviserAgent")


def _mock_reviser_generator(question: str, draft: str, flaws: List[str]) -> str:
    """Mock editor: appends a revision-notes line (topic-neutral)."""
    notes = "; ".join(flaws) if flaws else "general polish"
    return draft.rstrip() + f"\n\n_Revision notes (MOCK): addressed -> {notes}_\n"


def revise_draft(state: ResearchState) -> Dict[str, Any]:
    """Revise the current draft based on the critique. Outputs the complete revised markdown."""
    question = state.get("question", "").strip()
    draft = (state.get("draft") or state.get("report", "")).strip()
    critique = state.get("critique", {})
    flaws = critique.get("flaws", [])
    scores = critique.get("scores", {})
    run_id = state.get("run_id", "")
    current_iter = state.get("revision_count", 0) + 1

    flaws_text = "\n".join(f"[{i + 1}] {f}" for i, f in enumerate(flaws)) if flaws else "Address general analytical depth."
    scores_text = ", ".join(f"{k}: {v}/5" for k, v in scores.items())

    system_instruction = (
        "You are an editor revising a research report based on peer-review feedback.\n\n"
        "MANDATORY REVISION RULES:\n"
        "1. Address every specific flaw identified by the reviewer.\n"
        "2. Output the FULL REVISED REPORT in complete markdown (not a diff or snippet). Never stop mid-sentence.\n"
        "3. Preserve the report structure: Title, Introduction & Context, one section per sub-question, "
        "Synthesis & Conclusion, References.\n"
        "4. Retain valid citations ([1], [2], ...) and keep them mapped to the References list. "
        "Do not invent new sources or facts that are not already supported in the draft.\n"
        "5. Keep the report strictly on the original research question."
    )
    prompt = (
        f"Original Research Question: {question}\n\n"
        f"Peer Review Assessment (Scores: {scores_text}):\nSpecific Flaws to Fix:\n{flaws_text}\n\n"
        f"Current Report Draft:\n{draft}\n\n"
        "Produce the full, revised academic research report:"
    )

    try:
        content, tokens = call_llm(
            prompt=prompt, system_instruction=system_instruction,
            mock_generator=lambda p: _mock_reviser_generator(question, draft, flaws), node="reviser",
        )
    except LLMUnavailableError as exc:
        print(f"\n[REVISER] iteration {current_iter} FAILED: {exc}\n")
        return {"loop_error": f"reviser: {exc}", "metadata": {"llm_calls": []}}

    revised = content.strip()
    if "## References" not in revised and "# References" not in revised:
        match = re.search(r"## References.*", draft, re.DOTALL)
        if match:
            revised += "\n\n" + match.group(0)

    banner = ["", "=" * 60, f"========== ITERATION {current_iter}: REVISER ==========", "=" * 60,
              f"Research Topic: {question}", f"Flaws Addressed: {len(flaws)}", f"Prior Scores: {scores_text}",
              f"Revised Draft Length: {len(revised)} characters", "=" * 60, ""]
    print("\n".join(banner))

    return {
        "draft": revised,
        "revision_count": current_iter,
        "draft_history": [{"version": current_iter, "author": "reviser", "text": revised, "chars": len(revised), "run_id": run_id}],
        "metadata": {f"tokens_reviser_iter_{current_iter}": tokens["total_tokens"], "llm_calls": [usage_record("reviser", tokens)]},
    }
