"""Agents package for multi-agent deep research."""

__all__ = [
    "critique_draft",
    "plan_research",
    "research_subquestion",
    "revise_draft",
    "synthesize_findings",
    "validate_research",
    "write_report",
]


def __getattr__(name: str):
    """Lazy export of agent functions to eliminate circular import dependencies."""
    if name == "critique_draft":
        from agents.critic import critique_draft
        return critique_draft
    if name == "plan_research":
        from agents.planner import plan_research
        return plan_research
    if name == "research_subquestion":
        from agents.researcher import research_subquestion
        return research_subquestion
    if name == "revise_draft":
        from agents.reviser import revise_draft
        return revise_draft
    if name == "synthesize_findings":
        from agents.synthesizer import synthesize_findings
        return synthesize_findings
    if name == "validate_research":
        from agents.validator import validate_research
        return validate_research
    if name == "write_report":
        from agents.writer import write_report
        return write_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
