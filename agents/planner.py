"""
Planner Agent for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 4: "The Planner Agent"

First Principle:
Problem Decomposition & Cognitive Load Reduction:
An LLM is a next-token predictor with an effective receptive field limit.
Breaking a complex, open-ended question into 3-5 orthogonal sub-questions reduces
entropy and allows parallel, focused information retrieval.
"""

import json
import logging
import re
from typing import Any, Dict, List
from agents.llm import call_llm
from state import ResearchState

logger = logging.getLogger("PlannerAgent")


def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON output from LLM, stripping markdown codeblocks if present."""
    text_clean = text.strip()
    if text_clean.startswith("```"):
        # Remove ```json ... ``` wrapper
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r"\s*```$", "", text_clean)

    # Search for first { ... } block
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        text_clean = match.group(0)

    return json.loads(text_clean)


def _mock_plan_generator(topic: str) -> str:
    """Generate deterministic research plan if LLM is in placeholder/mock mode."""
    lower = topic.lower()
    if "battery" in lower or "lithium" in lower or "sodium" in lower:
        return json.dumps({
            "sub_questions": [
                "What are the 2026 cost and raw material supply chain differences between lithium-ion and sodium-ion batteries?",
                "How do energy density, cycle life, and round-trip efficiency compare for utility-scale grid storage?",
                "What are the thermal runaway risks, safety standards, and fire suppression requirements for both chemistries?",
                "What commercial grid-scale sodium-ion battery storage deployments and pilot projects exist as of 2026?"
            ]
        })
    elif "ai coding" in lower or "developer" in lower or "hiring" in lower:
        return json.dumps({
            "sub_questions": [
                "How have entry-level software engineering job postings and hiring volumes changed from 2024 to 2026?",
                "What new technical skill requirements (code review, AI orchestration, system design) are expected of junior engineers?",
                "How do engineering organizations measure junior developer productivity when using AI coding assistants?",
                "What are the long-term workforce risks of reduced junior hiring on senior engineering talent pipelines?"
            ]
        })
    else:
        return json.dumps({
            "sub_questions": [
                f"What is the historical background, core definition, and current state of {topic}?",
                f"What are the primary technical trade-offs, benchmarks, and economic factors in {topic}?",
                f"What recent breakthroughs, standards, or empirical developments occurred in {topic} by 2026?",
                f"What are the major challenges, risks, and projected future trajectories for {topic}?"
            ]
        })


def plan_research(state: ResearchState) -> Dict[str, Any]:
    """
    Decompose the user topic into 3-5 targeted research sub-questions.
    Enforces strict JSON schema with a one-shot repair loop.
    """
    topic = state.get("question", "").strip()

    system_instruction = (
        "You are an expert academic research director. Your task is to break down a complex "
        "research topic into 3 to 5 distinct, highly answerable, orthogonal sub-questions.\n"
        "STRICT REQUIREMENT: You must respond ONLY with a JSON object in this exact schema:\n"
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

    # First attempt
    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_plan_generator(topic),
    )

    plan_data: Dict[str, Any] = {}
    parsed = False

    try:
        plan_data = _clean_and_parse_json(content)
        if "sub_questions" in plan_data and isinstance(plan_data["sub_questions"], list) and len(plan_data["sub_questions"]) >= 2:
            parsed = True
    except Exception as exc:
        logger.warning("Initial JSON parsing failed: %s. Initiating one-shot JSON-repair.", exc)

    # One-shot repair retry if needed
    if not parsed:
        repair_prompt = (
            f"The previous output was invalid JSON:\n{content}\n\n"
            "Repair it immediately and output ONLY valid JSON conforming to:\n"
            '{"sub_questions": ["q1", "q2", "q3"]}'
        )
        repair_content, repair_tokens = call_llm(
            prompt=repair_prompt,
            system_instruction="You are a strict JSON repair engine. Output valid JSON only.",
            mock_generator=lambda p: _mock_plan_generator(topic),
        )
        tokens["prompt_tokens"] += repair_tokens["prompt_tokens"]
        tokens["completion_tokens"] += repair_tokens["completion_tokens"]
        tokens["total_tokens"] += repair_tokens["total_tokens"]

        try:
            plan_data = _clean_and_parse_json(repair_content)
            if "sub_questions" in plan_data and isinstance(plan_data["sub_questions"], list):
                parsed = True
        except Exception:
            # Fallback to mock plan
            plan_data = _clean_and_parse_json(_mock_plan_generator(topic))
            parsed = True

    sub_questions: List[str] = [str(q).strip() for q in plan_data.get("sub_questions", []) if str(q).strip()]
    # Ensure 3-5 sub-questions
    if len(sub_questions) < 3:
        sub_questions.extend([
            f"What are key empirical findings regarding {topic}?",
            f"What are emerging outlooks and limitations for {topic}?",
        ])
    sub_questions = sub_questions[:5]

    # Print DEMO-READY terminal banner
    banner = [
        "",
        "=" * 60,
        "========== [1] PLANNER OUTPUT ==========",
        "=" * 60,
        f"Research Topic: {topic}",
        f"Generated Sub-Questions ({len(sub_questions)} total):",
    ]
    for idx, sq in enumerate(sub_questions, 1):
        banner.append(f"  [{idx}] {sq}")
    banner.append("=" * 60)
    banner.append("")
    print("\n".join(banner))

    return {
        "plan": sub_questions,
        "metadata": {
            "tokens_planner": tokens["total_tokens"],
        },
    }
