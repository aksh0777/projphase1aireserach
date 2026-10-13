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
