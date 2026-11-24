"""
Graph Builder for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 8, Section 12 and Chapter 13

First Principle:
Deterministic Graph Topology & State Convergence: LangGraph models agent pipelines as
directed (cyclic/acyclic) state machines.

Phase 1 (baseline):  planner -> researchers -> synthesizer -> validator -> writer
Phase 2 (improved):  ... -> writer -> critic <-> reviser -> finalize

The ``validator`` integrity gate is shared by BOTH phases so the controlled experiment stays
like-for-like. Run isolation: every node call is wrapped so that the run's API key / forced-mock
flag come from ``config['configurable']`` (never from process-global state).
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.critic import critique_draft
from agents.llm import run_llm_context
from agents.planner import plan_research
from agents.researcher import research_subquestion
from agents.reviser import revise_draft
from agents.synthesizer import synthesize_findings
from agents.validator import validate_research
from agents.writer import write_report
from state import ResearchState, stable_key

logger = logging.getLogger("GraphBuilder")

# =============================================================================
# Loop Constants
# =============================================================================
MAX_REVISIONS: int = 3
PASS_THRESHOLD_MIN_SCORE: int = 4
# Final draft policy: the latest draft if it PASSED, otherwise the best-scoring draft
# (internal Critic mean; ties resolved toward the later draft). All versions are retained.
SELECTION_POLICY: str = "latest_if_pass_else_best_critic_mean"


def timing_wrapper(node_name: str, func: Callable) -> Callable:
    """Measure node latency into state['metadata'] and scope run credentials to this node call."""

    def wrapped(state_or_payload: Any, config: RunnableConfig = None) -> Dict[str, Any]:
        cfg = ((config or {}).get("configurable") or {}) if isinstance(config, dict) else {}
        start_time = time.perf_counter()
        with run_llm_context(api_key=cfg.get("__api_key"), force_mock=bool(cfg.get("force_mock", False))):
            result = func(state_or_payload)
        elapsed = round(time.perf_counter() - start_time, 3)

        if isinstance(result, dict):
            metadata = dict(result.get("metadata", {}))
            if node_name == "researcher":
                sub_q = state_or_payload.get("sub_question", "") if isinstance(state_or_payload, dict) else ""
                metadata[f"latency_researcher_{stable_key(sub_q)}" if sub_q else "latency_researcher"] = elapsed
            elif node_name in ("critic", "reviser"):
                rev_num = (state_or_payload.get("revision_count", 0) + 1) if isinstance(state_or_payload, dict) else 1
                metadata[f"latency_{node_name}_iter_{rev_num}"] = elapsed
            else:
                metadata[f"latency_{node_name}"] = elapsed
            result["metadata"] = metadata
        return result

    wrapped.__name__ = f"{node_name}_node"
    return wrapped


def _researcher_node(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Worker node executed per sub-question in parallel fan-out (carries run_id + original question)."""
    return research_subquestion(
        payload.get("sub_question", ""), run_id=payload.get("run_id", ""), question=payload.get("question", "")
    )


def _fanout_researchers(state: ResearchState) -> List[Send]:
    """Map each planned sub-question to an independent parallel researcher via Send."""
    plan = state.get("plan", [])
    if not plan:
        plan = [f"Core overview and background of {state.get('question', 'the topic')}"]
    rid, q = state.get("run_id", ""), state.get("question", "")
    return [Send("researcher", {"sub_question": sq, "run_id": rid, "question": q}) for sq in plan]


def _route_after_validation(state: ResearchState) -> str:
    return END if state.get("execution_status") == "FAILED" else "writer"


def _route_after_critic(state: ResearchState) -> str:
    """pass OR revision cap OR critic error -> finalize; else revise."""
    critique = state.get("critique", {})
    verdict = str(critique.get("verdict", "revise")).strip().lower()
    if verdict in ("pass", "error") or state.get("revision_count", 0) >= MAX_REVISIONS:
        return "finalize"
    return "reviser"


def _route_after_reviser(state: ResearchState) -> str:
    return "finalize" if state.get("loop_error") else "critic"


def _finalize_node(state: ResearchState) -> Dict[str, Any]:
    """Select the final draft (all versions are retained), persist it and record loop metadata."""
    history = state.get("draft_history") or []
    by_version = {h["version"]: h["text"] for h in history}
    critiques = [c for c in (state.get("score_history") or []) if c.get("scores")]
    revisions = state.get("revision_count", 0)
    latest_v = max(by_version) if by_version else revisions

    last_crit = critiques[-1] if critiques else {}
    latest_passed = bool(last_crit) and last_crit.get("draft_version") == latest_v and last_crit.get("verdict") == "pass"
    best = None
    for c in critiques:  # chronological; '>=' resolves ties toward the later draft
        if best is None or c.get("mean", 0) >= best.get("mean", 0):
            best = c
    best_v = best["draft_version"] if best else latest_v
    final_v = latest_v if (latest_passed or not best) else best_v
    final_text = by_version.get(final_v) or state.get("draft") or state.get("report", "")

    metadata = dict(state.get("metadata", {}))
    report_path = metadata.get("report_path")
    if not report_path:
        os.makedirs("outputs", exist_ok=True)
        report_path = os.path.join("outputs", f"report_phase2_{state.get('run_id', 'norun')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_text.strip() + "\n")

    final_verdict = last_crit.get("verdict") or ("error" if state.get("loop_error") else "n/a")
    out: Dict[str, Any] = {
        "report": final_text,
        "metadata": {
            "iterations_used": revisions,
            "critic_iterations": len(critiques),
            "revisions": revisions,
            "final_verdict": final_verdict,
            "cap_reached": bool(not latest_passed and revisions >= MAX_REVISIONS and not state.get("loop_error")),
            "latest_draft_version": latest_v,
            "best_draft_version": best_v,
            "final_draft_version": final_v,
            "selection_policy": SELECTION_POLICY,
            "report_path": report_path,
        },
    }
    if state.get("loop_error"):
        out["execution_status"] = "PARTIAL" if state.get("execution_status") != "FAILED" else "FAILED"
        out["metadata"]["loop_error"] = state["loop_error"]
    return out


def _add_common_nodes(builder: StateGraph) -> None:
    builder.add_node("planner", timing_wrapper("planner", plan_research))
    builder.add_node("researcher", timing_wrapper("researcher", _researcher_node))
    builder.add_node("synthesizer", timing_wrapper("synthesizer", synthesize_findings))
    builder.add_node("validator", timing_wrapper("validator", validate_research))
    builder.add_node("writer", timing_wrapper("writer", write_report))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", _fanout_researchers, ["researcher"])
    builder.add_edge("researcher", "synthesizer")
    builder.add_edge("synthesizer", "validator")


def build_baseline_graph():
    """Phase 1: feed-forward DAG  START -> planner -> [researchers] -> synthesizer -> validator -> writer -> END."""
    builder = StateGraph(ResearchState)
    _add_common_nodes(builder)
    builder.add_conditional_edges("validator", _route_after_validation, {"writer": "writer", END: END})
    builder.add_edge("writer", END)
    return builder.compile(checkpointer=MemorySaver())


def build_improved_graph():
    """Phase 2: baseline + Critic <-> Reviser loop (cap MAX_REVISIONS) -> finalize."""
    builder = StateGraph(ResearchState)
    _add_common_nodes(builder)
    builder.add_node("critic", timing_wrapper("critic", critique_draft))
    builder.add_node("reviser", timing_wrapper("reviser", revise_draft))
    builder.add_node("finalize", _finalize_node)

    builder.add_conditional_edges("validator", _route_after_validation, {"writer": "writer", END: END})
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges("critic", _route_after_critic, {"reviser": "reviser", "finalize": "finalize"})
    builder.add_conditional_edges("reviser", _route_after_reviser, {"critic": "critic", "finalize": "finalize"})
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=MemorySaver())


# Backward-compatible alias
build_research_graph = build_baseline_graph
