"""
Graph Builder for Multi-Agent Deep Research System
Architecture Reference: docs/EXPLANATIONS.md - Section 8: "Graph Orchestration & Execution Flow"

First Principle:
Deterministic Graph Topology & State Convergence:
Complex workflows require explicit orchestration boundaries.
LangGraph models agent pipelines as directed cyclic/acyclic state machines (FSMs),
guaranteeing checkpointing, fault-isolation, and deterministic convergence.
"""

from functools import wraps
import logging
import time
from typing import Any, Callable, Dict, List, Union
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.planner import plan_research
from agents.researcher import research_subquestion
from agents.synthesizer import synthesize_findings
from agents.writer import write_report
from state import ResearchState

logger = logging.getLogger("GraphBuilder")


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
        # Fallback if plan was empty
        topic = state.get("question", "General Analysis")
        plan = [f"Core overview and background of {topic}"]

    return [Send("researcher", {"sub_question": q}) for q in plan]


def build_research_graph():
    """
    Construct, wire, and compile the multi-agent research StateGraph.
    Pipeline: START -> planner -> researcher (parallel fan-out) -> synthesizer -> writer -> END.
    """
    builder = StateGraph(ResearchState)

    # 1. Register nodes with timing instrumentation
    builder.add_node("planner", timing_wrapper("planner", plan_research))
    builder.add_node("researcher", timing_wrapper("researcher", _researcher_node))
    builder.add_node("synthesizer", timing_wrapper("synthesizer", synthesize_findings))
    builder.add_node("writer", timing_wrapper("writer", write_report))

    # 2. Add structural edges
    builder.add_edge(START, "planner")

    # Parallel Map Fan-Out
    builder.add_conditional_edges("planner", _fanout_researchers, ["researcher"])

    # Fan-In: All parallel researchers merge their outputs and proceed to synthesizer
    builder.add_edge("researcher", "synthesizer")

    # Sequential Synthesis to Publication
    builder.add_edge("synthesizer", "writer")
    builder.add_edge("writer", END)

    # 3. Compile with in-memory checkpointer for state persistence
    checkpointer = MemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)

    return compiled_graph
