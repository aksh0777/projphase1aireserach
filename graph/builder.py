"""
Graph Builder for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 8 & Section 12

First Principle:
Deterministic Graph Topology & State Convergence:
LangGraph models agent pipelines as directed cyclic/acyclic state machines (FSMs),
guaranteeing checkpointing, fault-isolation, and deterministic convergence.

Phase 1: Deterministic feed-forward DAG (build_baseline_graph).
Phase 2: Closed-loop State Machine with Critic + Revision loop (build_improved_graph).
"""

from datetime import datetime
from functools import wraps
import logging
import os
import time
from typing import Any, Callable, Dict, List, Union
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.critic import critique_draft
from agents.planner import plan_research
from agents.researcher import research_subquestion
from agents.reviser import revise_draft
from agents.synthesizer import synthesize_findings
from agents.writer import write_report
from state import ResearchState

logger = logging.getLogger("GraphBuilder")

# =============================================================================
# Loop Constants
# =============================================================================
# Maximum number of revision passes allowed before forced termination
MAX_REVISIONS: int = 3

# Pass Threshold Logic:
# The Critic checks whether ALL 4 rubric criteria (relevance, completeness,
# organization, grounding) are >= 4. If any criterion is <= 3, verdict is "revise".
PASS_THRESHOLD_MIN_SCORE: int = 4


def timing_wrapper(node_name: str, func: Callable) -> Callable:
    """
    Wrap an agent node function to measure and record execution latency
    into state['metadata'].
    """
    @wraps(func)
    def wrapped(state_or_payload: Any) -> Dict[str, Any]:
        start_time = time.perf_counter()
        result = func(state_or_payload)
        elapsed = round(time.perf_counter() - start_time, 3)

        if isinstance(result, dict):
            metadata = dict(result.get("metadata", {}))
            if node_name == "researcher":
                sub_q = state_or_payload.get("sub_question", "") if isinstance(state_or_payload, dict) else ""
                key = f"latency_researcher_{abs(hash(sub_q)) % 1000}" if sub_q else "latency_researcher"
                metadata[key] = elapsed
            elif node_name == "critic":
                rev_num = state_or_payload.get("revision_count", 0) + 1 if isinstance(state_or_payload, dict) else 1
                metadata[f"latency_critic_iter_{rev_num}"] = elapsed
            elif node_name == "reviser":
                rev_num = state_or_payload.get("revision_count", 0) + 1 if isinstance(state_or_payload, dict) else 1
                metadata[f"latency_reviser_iter_{rev_num}"] = elapsed
            else:
                metadata[f"latency_{node_name}"] = elapsed
            result["metadata"] = metadata
        return result

    return wrapped


def _researcher_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker node executed per sub-question in parallel fan-out."""
    sub_q = payload.get("sub_question", "")
    return research_subquestion(sub_q)


def _fanout_researchers(state: ResearchState) -> List[Send]:
    """
    Conditional edge dispatcher mapping each sub-question in state['plan']
    to an independent parallel researcher node using LangGraph Send.
    """
    plan = state.get("plan", [])
    if not plan:
        topic = state.get("question", "General Analysis")
        plan = [f"Core overview and background of {topic}"]

    return [Send("researcher", {"sub_question": q}) for q in plan]


def _route_after_critic(state: ResearchState) -> str:
    """
    Routing condition after Critic evaluation:
    - If verdict == 'pass' OR revision_count >= MAX_REVISIONS: route to 'finalize' -> END
    - Else: route to 'reviser' to improve the draft
    """
    critique = state.get("critique", {})
    verdict = str(critique.get("verdict", "revise")).strip().lower()
    rev_count = state.get("revision_count", 0)

    # Check convergence condition
    if verdict == "pass" or rev_count >= MAX_REVISIONS:
        return "finalize"
    return "reviser"


def _finalize_node(state: ResearchState) -> Dict[str, Any]:
    """
    Final node in the improved graph: promotes working draft to final report
    and updates disk artifact with finalized content.
    """
    final_draft = state.get("draft") or state.get("report", "")
    metadata = dict(state.get("metadata", {}))

    report_path = metadata.get("report_path")
    if not report_path or not os.path.exists(report_path):
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("outputs", f"report_phase2_{timestamp}.md")
        metadata["report_path"] = report_path

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_draft.strip() + "\n")

    return {
        "report": final_draft,
        "metadata": {
            "iterations_used": state.get("revision_count", 0),
            "final_verdict": state.get("critique", {}).get("verdict", "pass"),
            "report_path": report_path,
        },
    }


def build_baseline_graph():
    """
    Phase 1 Baseline Graph: Strictly Feed-Forward DAG.
    START -> planner -> [parallel researchers] -> synthesizer -> writer -> END.
    """
    builder = StateGraph(ResearchState)

    # 1. Register nodes with timing instrumentation
    builder.add_node("planner", timing_wrapper("planner", plan_research))
    builder.add_node("researcher", timing_wrapper("researcher", _researcher_node))
    builder.add_node("synthesizer", timing_wrapper("synthesizer", synthesize_findings))
    builder.add_node("writer", timing_wrapper("writer", write_report))

    # 2. Add structural edges
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _fanout_researchers, ["researcher"])
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "writer")
    builder.add_edge("writer", END)

    # 3. Compile with in-memory checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def build_improved_graph():
    """
    Phase 2 Improved Graph: Closed-Loop State Machine with Critic + Revision Loop.
    Pipeline:
      START -> planner -> [researchers] -> synthesizer -> writer -> critic ->
      conditional_edge:
        - if verdict == 'pass' or revision_count >= MAX_REVISIONS -> finalize -> END
        - else -> reviser -> critic (loops back)
    """
    builder = StateGraph(ResearchState)

    # 1. Register all nodes with timing wrappers
    builder.add_node("planner", timing_wrapper("planner", plan_research))
    builder.add_node("researcher", timing_wrapper("researcher", _researcher_node))
    builder.add_node("synthesizer", timing_wrapper("synthesizer", synthesize_findings))
    builder.add_node("writer", timing_wrapper("writer", write_report))
    builder.add_node("critic", timing_wrapper("critic", critique_draft))
    builder.add_node("reviser", timing_wrapper("reviser", revise_draft))
    builder.add_node("finalize", _finalize_node)

    # 2. Add structural edges
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _fanout_researchers, ["researcher"])
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "writer")

    # Writer initiates draft and hands over to Critic
    builder.add_edge("writer", "critic")

    # Critic routes conditionally
    builder.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "reviser": "reviser",
            "finalize": "finalize",
        },
    )

    # Reviser loops back to Critic
    builder.add_edge("reviser", "critic")

    # Finalize promotes draft to report and ends workflow
    builder.add_edge("finalize", END)

    # 3. Compile with checkpointer
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Backward-compatible alias
build_research_graph = build_baseline_graph
