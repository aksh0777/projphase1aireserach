"""
Synthesizer Agent for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 6: "The Synthesizer Agent"

First Principle:
Information Synthesis & Entropy Reduction:
Parallel workers produce overlapping and potentially contradictory evidence.
The Synthesizer reconciles discrepancies, performs cross-source fact-checking,
eliminates redundancy, and structures the findings into a cohesive brief.
"""

import logging
import re
from typing import Any, Dict, List
from agents.llm import call_llm
from state import ResearchState

logger = logging.getLogger("SynthesizerAgent")


def _mock_synthesis_generator(topic: str, notes_dict: Dict[str, List[str]]) -> str:
    """Generate structured research brief highlighting synthesis, deduplication, and contradictions."""
    all_notes = []
    sources = set()
    for sq, n_list in notes_dict.items():
        for note in n_list:
            all_notes.append(note)
            found_urls = re.findall(r"\[(https?://[^\]]+)\]", note)
            for u in found_urls:
                sources.add(u)

    source_list = sorted(list(sources)) or ["https://research-index.org/2026-report"]

    return (
        f"# Research Brief: {topic}\n\n"
        "## 1. Executive Synthesis\n"
        "This synthesis consolidates empirical findings gathered across parallel research streams. "
        "The evidence highlights critical inflection points in 2026, driven by technological scale-up, "
        "shifting cost structures, and evolving system-level requirements.\n\n"
        "## 2. Key Thematic Findings\n"
        + "\n".join([f"- {note}" for note in all_notes[:8]]) + "\n\n"
        "## 3. Cross-Source Contradictions & Divergent Perspectives\n"
        "- **Cost Variance vs. Maturity**: Early commercial reports claim manufacturing costs near parity with mature alternatives, "
        "whereas conservative supply chain analyses project a 12-18 month lag before gigawatt-scale economies of scale are fully realized.\n"
        "- **Cycle Life Benchmarks**: Accelerated lab cycle testing indicates up to 6,000 cycles under controlled thermal conditions, "
        "whereas field telemetry from early operational pilots demonstrates modest capacity fade under extreme ambient thermal stress.\n\n"
        "## 4. Consolidated Source Index\n"
        + "\n".join([f"[{idx+1}] {url}" for idx, url in enumerate(source_list)])
    )


def synthesize_findings(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesize notes from all parallel researchers into a coherent research brief.
    Deduplicates overlapping points and explicitly flags contradictions.
    """
    topic = state.get("question", "")
    notes_dict = state.get("notes", {})

    # Format all findings
    formatted_notes = ""
    all_urls = set()
    for sq, findings in notes_dict.items():
        formatted_notes += f"### Sub-Question: {sq}\n"
        for item in findings:
            formatted_notes += f"- {item}\n"
            urls = re.findall(r"\[(https?://[^\]]+)\]", item)
            for u in urls:
                all_urls.add(u)
        formatted_notes += "\n"

    system_instruction = (
        "You are a rigorous academic research synthesizer. Your task is to process findings collected "
        "across multiple parallel research nodes, eliminate redundant bullet points, detect and explicitly flag "
        "contradictions or discrepancies between sources, and compile an organized, cohesive research brief.\n\n"
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
    )

    # Print DEMO-READY terminal banner
    banner = [
        "",
        "=" * 60,
        "========== [3] SYNTHESIZER BRIEF ==========",
        "=" * 60,
        content.strip(),
        "=" * 60,
        "",
    ]
    print("\n".join(banner))

    return {
        "brief": content.strip(),
        "metadata": {
            "tokens_synthesizer": tokens["total_tokens"],
        },
    }
