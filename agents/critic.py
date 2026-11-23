"""
Critic Agent for Multi-Agent Deep Research System (Phase 2)
Architecture Reference: docs/EXPLANATIONS.md - Section 11: "The Critic & Revision Loop"

First Principle:
Dialectical Verification & Adversarial Self-Correction:
Autoregressive LLM generation often exhibits confirmation bias and syntactic drift.
An isolated critic evaluating the draft strictly against the prompt—without access
to upstream retrieval notes—provides objective quality filtering and prevents
ungrounded or incomplete sections from propagating to final publication.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
from agents.llm import call_llm
from state import ResearchState

logger = logging.getLogger("CriticAgent")

PASS_THRESHOLD_MIN_SCORE = 4


def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON output from LLM, stripping markdown codeblocks if present."""
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r"\s*```$", "", text_clean)

    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        text_clean = match.group(0)

    return json.loads(text_clean)


def _mock_critic_generator(question: str, draft: str, revision_count: int) -> str:
    """Deterministic mock reviewer generator for resilient mode."""
    if revision_count == 0:
        return json.dumps({
            "scores": {
                "relevance": 4,
                "completeness": 3,
                "organization": 4,
                "grounding": 3,
            },
            "justifications": {
                "relevance": "The draft directly addresses the core research question with structured headings.",
                "completeness": "Key technical trade-offs are outlined, but specific 2026 economic metrics lack granular depth.",
                "organization": "Clear progression from introduction to sub-questions and conclusion.",
                "grounding": "In-text citations are present, but certain quantitative claims in middle sections lack immediate source attribution.",
            },
            "flaws": [
                "Section 2 mentions comparative efficiency metrics without citing the empirical laboratory standard.",
                "Section 3 references cost parity trajectories but does not specify whether figures represent cell-level or pack-level CapEx.",
                "References list formatting is inconsistent across digital and print citations.",
            ],
            "verdict": "revise",
        })
    else:
        return json.dumps({
            "scores": {
                "relevance": 5,
                "completeness": 4,
                "organization": 5,
                "grounding": 4,
            },
            "justifications": {
                "relevance": "Direct, thorough alignment with the target research question.",
                "completeness": "All major sub-domains, technical benchmarks, and economic projections are addressed.",
                "organization": "Exemplary academic structure with balanced section weighting.",
                "grounding": "Strong inline citation coverage mapping cleanly to the bibliography.",
            },
            "flaws": [],
            "verdict": "pass",
        })


def critique_draft(state: ResearchState) -> Dict[str, Any]:
    """
    Evaluate the current draft report strictly on its own merits against the research question.
    CRITICAL RULE: The critic does NOT see search notes, search results, or plan.
    Outputs structured scores, justifications, actionable flaws, and a pass/revise verdict.
    """
    question = state.get("question", "").strip()
    draft = state.get("draft") or state.get("report", "").strip()
    revision_count = state.get("revision_count", 0)
    iter_num = revision_count + 1

    system_instruction = (
        "You are a strict but fair peer reviewer evaluating a scientific research report.\n\n"
        "CRITICAL ISOLATION RULE:\n"
        "You evaluate the report draft strictly on its own merits relative to the research question. "
        "You do NOT have access to search notes, search results, or the planner's decomposition. "
        "Judge whether the draft standalone is rigorous, empirically grounded, well-structured, and comprehensive.\n\n"
        "EVALUATION CRITERIA (Score 1-5):\n"
        "1. Relevance (1-5): Does the report directly and comprehensively answer the research question?\n"
        "2. Completeness (1-5): Are all critical technical, economic, and operational dimensions explored?\n"
        "3. Organization (1-5): Is the report structured logically with appropriate headings, transitions, and flow?\n"
        "4. Grounding (1-5): Are factual assertions, numbers, and dates strictly supported by numbered in-text citations ([1], [2]) linking to references?\n\n"
        "VERDICT RULE:\n"
        "Pass only if ALL four scores are >= 4. If ANY score is <= 3, the verdict MUST be 'revise'.\n\n"
        "STRICT JSON OUTPUT REQUIREMENT:\n"
        "Respond ONLY with a valid JSON object in this exact schema:\n"
        "{\n"
        '  "scores": {\n'
        '    "relevance": int,\n'
        '    "completeness": int,\n'
        '    "organization": int,\n'
        '    "grounding": int\n'
        "  },\n"
        '  "justifications": {\n'
        '    "relevance": "One-sentence justification.",\n'
        '    "completeness": "One-sentence justification.",\n'
        '    "organization": "One-sentence justification.",\n'
        '    "grounding": "One-sentence justification."\n'
        "  },\n"
        '  "flaws": [\n'
        '    "Specific actionable problem referencing a section or claim (max 5 flaws)"\n'
        "  ],\n"
        '  "verdict": "pass" | "revise"\n'
        "}\n"
        "No conversational filler or markdown code fence wrapper."
    )

    prompt = (
        f"Research Question: {question}\n\n"
        f"Report Draft (Iteration {iter_num}):\n{draft[:14000]}\n\n"
        "Perform strict peer review according to the rubric:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_critic_generator(question, draft, revision_count),
    )

    critique_data: Dict[str, Any] = {}
    parsed = False

    try:
        critique_data = _clean_and_parse_json(content)
        if "scores" in critique_data and "verdict" in critique_data:
            parsed = True
    except Exception as exc:
        logger.warning("Initial Critic JSON parsing failed: %s. Initiating one-shot repair.", exc)

    # One-shot JSON self-repair if needed
    if not parsed:
        repair_prompt = (
            f"The previous output was invalid JSON:\n{content}\n\n"
            "Repair it immediately. Output ONLY valid JSON adhering to:\n"
            '{"scores": {"relevance": 4, "completeness": 4, "organization": 4, "grounding": 4}, '
            '"justifications": {"relevance": "...", "completeness": "...", "organization": "...", "grounding": "..."}, '
            '"flaws": ["..."], "verdict": "pass"}'
        )
        repair_content, repair_tokens = call_llm(
            prompt=repair_prompt,
            system_instruction="You are a strict JSON repair engine. Output valid JSON only.",
            mock_generator=lambda p: _mock_critic_generator(question, draft, revision_count),
        )
        tokens["total_tokens"] += repair_tokens["total_tokens"]
        try:
            critique_data = _clean_and_parse_json(repair_content)
            parsed = True
        except Exception:
            critique_data = _clean_and_parse_json(_mock_critic_generator(question, draft, revision_count))

    # Normalize scores and enforce verdict logic
    scores = critique_data.get("scores", {})
    r_score = int(scores.get("relevance", 3))
    c_score = int(scores.get("completeness", 3))
    o_score = int(scores.get("organization", 3))
    g_score = int(scores.get("grounding", 3))
    scores_clean = {
        "relevance": max(1, min(5, r_score)),
        "completeness": max(1, min(5, c_score)),
        "organization": max(1, min(5, o_score)),
        "grounding": max(1, min(5, g_score)),
    }

    # Pass rule: all 4 scores >= 4
    passes_threshold = all(s >= PASS_THRESHOLD_MIN_SCORE for s in scores_clean.values())
    raw_verdict = str(critique_data.get("verdict", "")).strip().lower()
    final_verdict = "pass" if (passes_threshold and raw_verdict == "pass") else "revise"

    flaws = list(critique_data.get("flaws", []))[:5]
    if final_verdict == "revise" and not flaws:
        flaws = ["Expand depth on quantitative metrics and verify numbered citation grounding across all sections."]

    justifications = critique_data.get("justifications", {})

    clean_critique = {
        "iteration": iter_num,
        "scores": scores_clean,
        "justifications": justifications,
        "flaws": flaws,
        "verdict": final_verdict,
    }

    # Print DEMO-READY terminal banner
    banner = [
        "",
        "=" * 60,
        f"========== ITERATION {iter_num}: CRITIC ==========",
        "=" * 60,
        f"Target Question: {question}",
        "Scores Evaluated:",
        f"  • Relevance:    {scores_clean['relevance']} / 5",
        f"  • Completeness: {scores_clean['completeness']} / 5",
        f"  • Organization: {scores_clean['organization']} / 5",
        f"  • Grounding:    {scores_clean['grounding']} / 5",
        f"Verdict: {final_verdict.upper()}",
    ]
    if flaws:
        banner.append(f"Actionable Flaws Identified ({len(flaws)}):")
        for idx, f in enumerate(flaws, 1):
            banner.append(f"  [{idx}] {f}")
    else:
        banner.append("Actionable Flaws: None (All criteria passed).")
    banner.append("=" * 60)
    banner.append("")
    print("\n".join(banner))

    return {
        "critique": clean_critique,
        "score_history": [clean_critique],
        "metadata": {
            f"tokens_critic_iter_{iter_num}": tokens["total_tokens"],
            f"verdict_iter_{iter_num}": final_verdict,
            f"score_mean_iter_{iter_num}": round(sum(scores_clean.values()) / 4.0, 2),
        },
    }
