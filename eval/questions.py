"""
Evaluation Benchmark Questions
Architecture Reference: docs/EXPLANATIONS.md - Section 10: "Evaluation Harness & LLM-as-a-Judge"

First Principle:
Benchmark Diversity & Ecological Validity:
Evaluating complex multi-agent reasoning requires diverse prompts covering
rapidly changing 2026 technical domains, economic trade-offs, and multi-disciplinary systems.
"""

from typing import List

EVAL_QUESTIONS: List[str] = [
    # 1. Energy & Electrochemistry
    "Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026",

    # 2. Tech Workforce & AI Economics
    "Impact of AI coding assistants on junior developer hiring in 2026",

    # 3. Nuclear Engineering & Energy Policy
    "Commercial viability and licensing status of Small Modular Reactors (SMRs) for data center power",

    # 4. Cybersecurity & Cryptography
    "Post-quantum cryptography migration challenges for core banking and financial transaction protocols",

    # 5. Biotechnology & Medicine
    "Current clinical pipeline and manufacturing constraints for mRNA therapeutics beyond vaccines",

    # 6. Computer Architecture
    "Neuromorphic computing hardware architectures vs ultra-low-power GPUs for edge inference",

    # 7. Climate Engineering
    "Levelized cost and thermodynamic limits of direct air capture (DAC) technologies at gigaton scale",

    # 8. Robotics & Industrial Automation
    "Deployment economics and safety certification of autonomous humanoid robots in automotive manufacturing",

    # 9. Distributed Systems & Telecommunications
    "Decentralized Physical Infrastructure Networks (DePIN) scalability limitations for wireless connectivity",

    # 10. Advanced Materials Science
    "Degradation mechanisms and dendrite suppression strategies in all-solid-state lithium batteries",
]


# ---------------------------------------------------------------------------
# Stable benchmark identifiers (Phase 2.2). The question TEXT above is frozen;
# IDs are assigned by position so Phase 1 and Phase 2 records can be joined on
# Q01..Q10 instead of fragile exact-string matching.
# ---------------------------------------------------------------------------
import re as _re
from difflib import SequenceMatcher as _SM
from typing import Dict as _Dict, Optional as _Optional

BENCHMARK_QUESTIONS: List[_Dict[str, str]] = [
    {"id": f"Q{i:02d}", "question": q} for i, q in enumerate(EVAL_QUESTIONS, 1)
]
BENCHMARK_IDS: List[str] = [b["id"] for b in BENCHMARK_QUESTIONS]


def normalize_question(text: str) -> str:
    """Lower-case, strip punctuation/extra whitespace so cosmetic differences never break joins."""
    return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


def benchmark_id_for(text: str, fuzzy_threshold: float = 0.93) -> _Optional[str]:
    """Map free text to a benchmark ID (normalised exact match, then high-threshold fuzzy match)."""
    n = normalize_question(text)
    if not n:
        return None
    best_id, best_ratio = None, 0.0
    for b in BENCHMARK_QUESTIONS:
        bn = normalize_question(b["question"])
        if n == bn:
            return b["id"]
        r = _SM(None, n, bn).ratio()
        if r > best_ratio:
            best_id, best_ratio = b["id"], r
    return best_id if best_ratio >= fuzzy_threshold else None


def question_for_id(benchmark_id: str) -> _Optional[str]:
    for b in BENCHMARK_QUESTIONS:
        if b["id"] == benchmark_id:
            return b["question"]
    return None
