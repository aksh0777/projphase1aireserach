"""
Critic Agent for Multi-Agent Deep Research System (Phase 2)
Architecture Reference: docs/EXPLANATIONS.md - Section 11 and Chapter 13

First Principle:
Dialectical Verification: an isolated critic evaluating the draft strictly against the
question - without access to upstream retrieval notes - provides objective filtering.

Integrity fixes (Chapter 13):
* The FULL draft is shown to the Critic. It used to be cut at 14,000 characters, which made
  every long draft look "truncated mid-sentence" and sent the revision loop chasing a flaw
  that the truncation itself had created.
* Missing/invalid scores are NOT defaulted to 3. An unparsable critique after one repair
  attempt marks the loop as errored (LIVE) instead of inventing a verdict.
* Internal Critic scores are a control signal for the loop. They are never mixed with
  the external Judge scores used in the experiment.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.llm import LLMUnavailableError, call_llm, is_live_mode, merge_usage, usage_record
from state import ResearchState

logger = logging.getLogger("CriticAgent")

PASS_THRESHOLD_MIN_SCORE = 4
DIMENSIONS = ("relevance", "completeness", "organization", "grounding")


def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r"\s*```$", "", text_clean)
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        text_clean = match.group(0)
    return json.loads(text_clean)


_MOCK_SCORE_LADDER = [
    (4, 3, 4, 3),
    (4, 4, 4, 3),
    (5, 4, 5, 4),
]


def _mock_critic_generator(question: str, draft: str, revision_count: int) -> str:
    """Deterministic, topic-neutral mock reviewer whose scores vary with the revision number."""
    r, c, o, g = _MOCK_SCORE_LADDER[min(revision_count, len(_MOCK_SCORE_LADDER) - 1)]
    passed = min(r, c, o, g) >= PASS_THRESHOLD_MIN_SCORE
    flaws = [] if passed else [
        "[MOCK] Section 2 states quantitative claims without a supporting citation.",
        "[MOCK] The conclusion does not explicitly answer the research question.",
    ][: 2 if revision_count == 0 else 1]
    return json.dumps({
        "scores": {"relevance": r, "completeness": c, "organization": o, "grounding": g},
        "justifications": {d: f"[MOCK] Placeholder justification for {d}." for d in DIMENSIONS},
        "flaws": flaws,
        "verdict": "pass" if passed else "revise",
    })


def _validate_scores(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Return integer 1-5 scores for all four dimensions, or None if any is missing/invalid."""
    scores = data.get("scores")
    if not isinstance(scores, dict):
        return None
    clean: Dict[str, int] = {}
    for d in DIMENSIONS:
        v = scores.get(d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not (1 <= int(v) <= 5):
            return None
        clean[d] = int(v)
    return clean


def critique_draft(state: ResearchState) -> Dict[str, Any]:
    """
    Evaluate the current draft strictly on its own merits against the research question.
    The critic does NOT see search notes, search results, or the plan.
    """
    question = state.get("question", "").strip()
    draft = (state.get("draft") or state.get("report", "")).strip()
    revision_count = state.get("revision_count", 0)
    iter_num = revision_count + 1

    system_instruction = (
        "You are a strict but fair peer reviewer evaluating a research report.\n\n"
        "CRITICAL ISOLATION RULE:\n"
        "You receive ONLY the research question and the report draft. You do NOT have access to search notes, "
        "search results, or the planner's decomposition, so you must judge the draft on its own merits. "
        "The complete draft is provided; do not claim it is truncated unless it genuinely ends mid-sentence.\n\n"
        "EVALUATION CRITERIA (Score 1-5):\n"
        "1. Relevance: Does the report directly and comprehensively answer the research question?\n"
        "2. Completeness: Are all critical dimensions of the question explored?\n"
        "3. Organization: Is the report structured logically with headings, transitions, and flow?\n"
        "4. Grounding: Are factual assertions, numbers, and dates supported by numbered in-text citations ([1], [2]) linking to references?\n\n"
        "VERDICT RULE:\nPass only if ALL four scores are >= 4. If ANY score is <= 3, the verdict MUST be 'revise'.\n\n"
        "FLAWS: at most 5, each specific and actionable and referencing a section or claim.\n\n"
        "STRICT JSON OUTPUT (no markdown fence, no filler):\n"
        "{\n"
        '  "scores": {"relevance": int, "completeness": int, "organization": int, "grounding": int},\n'
        '  "justifications": {"relevance": "one sentence", "completeness": "one sentence", "organization": "one sentence", "grounding": "one sentence"},\n'
        '  "flaws": ["..."],\n'
        '  "verdict": "pass" | "revise"\n'
        "}"
    )
    prompt = f"Research Question: {question}\n\nReport Draft (version {revision_count}):\n{draft}\n\nPerform strict peer review according to the rubric:"

    llm_calls: List[Dict[str, Any]] = []
    try:
        content, tokens = call_llm(
            prompt=prompt, system_instruction=system_instruction,
            mock_generator=lambda p: _mock_critic_generator(question, draft, revision_count),
            node="critic",
        )
        llm_calls.append(usage_record("critic", tokens))

        data: Dict[str, Any] = {}
        scores_clean: Optional[Dict[str, int]] = None
        try:
            data = _clean_and_parse_json(content)
            scores_clean = _validate_scores(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Critic JSON parsing failed: %s. One-shot repair.", exc)

        if scores_clean is None:
            repair_prompt = (
                f"The previous output was invalid or incomplete:\n{content}\n\n"
                "Repair it. Output ONLY valid JSON with integer scores 1-5 for relevance, completeness, organization, "
                'grounding, plus "justifications", "flaws" and "verdict".'
            )
            rep, rep_tokens = call_llm(
                prompt=repair_prompt, system_instruction="You are a strict JSON repair engine. Output valid JSON only.",
                mock_generator=lambda p: _mock_critic_generator(question, draft, revision_count), node="critic_repair",
            )
            llm_calls.append(usage_record("critic_repair", rep_tokens))
            tokens = merge_usage(tokens, rep_tokens)
            try:
                data = _clean_and_parse_json(rep)
                scores_clean = _validate_scores(data)
            except Exception:  # noqa: BLE001
                scores_clean = None

        if scores_clean is None:
            raise LLMUnavailableError("Critic returned no valid integer scores after one repair attempt.")
    except LLMUnavailableError as exc:
        err = {"iteration": iter_num, "draft_version": revision_count, "verdict": "error", "error": str(exc)[:300], "scores": {}, "flaws": [], "justifications": {}}
        print(f"\n[CRITIC] iteration {iter_num} FAILED: {exc}\n")
        return {"critique": err, "loop_error": f"critic: {exc}", "metadata": {"llm_calls": llm_calls}}

    passes_threshold = all(s >= PASS_THRESHOLD_MIN_SCORE for s in scores_clean.values())
    raw_verdict = str(data.get("verdict", "")).strip().lower()
    final_verdict = "pass" if (passes_threshold and raw_verdict == "pass") else "revise"

    flaws = [str(f) for f in list(data.get("flaws", []))[:5]]
    if final_verdict == "revise" and not flaws:
        flaws = ["Expand depth on quantitative metrics and verify numbered citation grounding across all sections."]

    clean_critique = {
        "iteration": iter_num,
        "draft_version": revision_count,
        "scores": scores_clean,
        "justifications": data.get("justifications", {}) if isinstance(data.get("justifications"), dict) else {},
        "flaws": flaws,
        "verdict": final_verdict,
        "mean": round(sum(scores_clean.values()) / 4.0, 2),
        "draft_chars": len(draft),
    }

    banner = ["", "=" * 60, f"========== ITERATION {iter_num}: CRITIC ==========", "=" * 60,
              f"Target Question: {question}", f"Draft version: {revision_count} ({len(draft)} chars, shown in full)", "Scores Evaluated:"]
    for d in DIMENSIONS:
        banner.append(f"  • {d.title():<13} {scores_clean[d]} / 5")
    banner.append(f"Verdict: {final_verdict.upper()}")
    if flaws:
        banner.append(f"Actionable Flaws Identified ({len(flaws)}):")
        banner += [f"  [{i}] {f}" for i, f in enumerate(flaws, 1)]
    else:
        banner.append("Actionable Flaws: None (All criteria passed).")
    banner += ["=" * 60, ""]
    print("\n".join(banner))

    return {
        "critique": clean_critique,
        "score_history": [clean_critique],
        "metadata": {
            f"tokens_critic_iter_{iter_num}": tokens["total_tokens"],
            f"verdict_iter_{iter_num}": final_verdict,
            f"score_mean_iter_{iter_num}": clean_critique["mean"],
            "llm_calls": llm_calls,
        },
    }
