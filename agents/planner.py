"""
Planner Agent for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 4 and Chapter 13

First Principle:
Problem Decomposition & Cognitive Load Reduction: breaking a complex, open-ended
question into 3-5 orthogonal sub-questions reduces entropy and allows parallel,
focused information retrieval.

Integrity: in LIVE mode an unparsable plan raises LLMUnavailableError (the run is
FAILED) instead of silently substituting a canned plan. The MOCK plan is generic.
"""

import json
import logging
import re
from typing import Any, Dict, List

from agents.llm import LLMUnavailableError, call_llm, is_live_mode, merge_usage, usage_record
from state import ResearchState

logger = logging.getLogger("PlannerAgent")


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


def _mock_plan_generator(topic: str) -> str:
    """Generic, topic-parametrised MOCK plan (no domain keyword routing, so it can never be off-topic)."""
    return json.dumps({
        "sub_questions": [
            f"What are the definition, scope and background of: {topic}?",
            f"What are the key comparisons, metrics or evidence relevant to: {topic}?",
            f"What recent developments or empirical findings exist regarding: {topic}?",
            f"What are the main limitations, risks and open questions for: {topic}?",
        ]
    })


def _valid_plan(data: Dict[str, Any]) -> bool:
    sq = data.get("sub_questions")
    return isinstance(sq, list) and len([q for q in sq if str(q).strip()]) >= 2


def plan_research(state: ResearchState) -> Dict[str, Any]:
    """Decompose the user topic into 3-5 targeted research sub-questions (strict JSON + one repair)."""
    topic = state.get("question", "").strip()

    system_instruction = (
        "You are an expert academic research director. Break down a complex research topic into 3 to 5 "
        "distinct, highly answerable, orthogonal sub-questions that all stay on the stated topic.\n"
        "STRICT REQUIREMENT: respond ONLY with a JSON object in this exact schema:\n"
        "{\n"
        '  "sub_questions": [\n'
        '    "First specific sub-question...",\n'
        '    "Second specific sub-question...",\n'
        '    "Third specific sub-question..."\n'
        "  ]\n"
        "}\n"
        "Do not include any conversational filler, markdown formatting, or preamble."
    )
    prompt = f"Deconstruct this research topic into 3-5 sub-questions:\nTopic: {topic}"

    llm_calls: List[Dict[str, Any]] = []
    content, tokens = call_llm(
        prompt=prompt, system_instruction=system_instruction,
        mock_generator=lambda p: _mock_plan_generator(topic), node="planner",
    )
    llm_calls.append(usage_record("planner", tokens))

    plan_data: Dict[str, Any] = {}
    parsed = False
    try:
        plan_data = _clean_and_parse_json(content)
        parsed = _valid_plan(plan_data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Initial JSON parsing failed: %s. Initiating one-shot JSON-repair.", exc)

    if not parsed:
        repair_prompt = (
            f"The previous output was invalid JSON:\n{content}\n\n"
            "Repair it immediately and output ONLY valid JSON conforming to:\n"
            '{"sub_questions": ["q1", "q2", "q3"]}'
        )
        repair_content, repair_tokens = call_llm(
            prompt=repair_prompt,
            system_instruction="You are a strict JSON repair engine. Output valid JSON only.",
            mock_generator=lambda p: _mock_plan_generator(topic), node="planner_repair",
        )
        llm_calls.append(usage_record("planner_repair", repair_tokens))
        tokens = merge_usage(tokens, repair_tokens)
        try:
            plan_data = _clean_and_parse_json(repair_content)
            parsed = _valid_plan(plan_data)
        except Exception:  # noqa: BLE001
            parsed = False

    if not parsed:
        if is_live_mode():
            raise LLMUnavailableError("Planner returned invalid JSON after one repair attempt (no mock plan is substituted in LIVE mode).")
        plan_data = _clean_and_parse_json(_mock_plan_generator(topic))

    sub_questions: List[str] = []
    for q in plan_data.get("sub_questions", []):
        q = str(q).strip()
        if q and q not in sub_questions:
            sub_questions.append(q)
    if len(sub_questions) < 3:
        sub_questions.extend([
            f"What are key empirical findings regarding {topic}?",
            f"What are emerging outlooks and limitations for {topic}?",
        ])
    sub_questions = sub_questions[:5]

    banner = ["", "=" * 60, "========== [1] PLANNER OUTPUT ==========", "=" * 60,
              f"Research Topic: {topic}", f"Generated Sub-Questions ({len(sub_questions)} total):"]
    for idx, sq in enumerate(sub_questions, 1):
        banner.append(f"  [{idx}] {sq}")
    banner += ["=" * 60, ""]
    print("\n".join(banner))

    return {
        "plan": sub_questions,
        "metadata": {"tokens_planner": tokens["total_tokens"], "llm_calls": llm_calls},
    }
