"""
Unified Run Executor & Run Record Builder
Architecture Reference: docs/EXPLANATIONS.md - Chapter 13

Single code path used by main.py (CLI), app.py (Streamlit) and eval/runner.py (benchmark) so that
every execution:
  * gets a unique run_id and a FRESH ResearchState,
  * is executed with run-scoped credentials/mode (no process-global leakage),
  * is summarised into one schema-v2 RUN RECORD with ACTUAL token counts (estimates are flagged),
  * has an explicit status: LIVE / MOCK / PARTIAL / FAILED.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.llm import get_model_name, is_live_mode, run_llm_context
from graph.builder import build_baseline_graph, build_improved_graph
from state import ResearchState, new_run_state

RUNS_DIR = os.path.join("outputs", "runs")
SCHEMA_VERSION = 2


def _sum(calls: List[Dict[str, Any]], key: str) -> int:
    return int(sum(int(c.get(key, 0)) for c in calls))


def derive_mode_and_status(
    live_intent: bool, calls: List[Dict[str, Any]], state_status: str, has_report: bool, error: str
) -> Tuple[str, str, str]:
    """Return (execution_mode, execution_status, display_status)."""
    modes = {c.get("mode") for c in calls}
    if not calls:
        mode = "LIVE" if live_intent else "MOCK"
    elif modes == {"live"}:
        mode = "LIVE"
    elif modes <= {"mock"}:
        mode = "MOCK"
    else:
        mode = "MIXED"

    if error or state_status == "FAILED" or not has_report:
        status = "FAILED"
    elif state_status == "PARTIAL" or mode == "MIXED":
        status = "PARTIAL"
    else:
        status = "COMPLETE"

    if status == "FAILED":
        display = "FAILED"
    elif mode == "MOCK":
        display = "MOCK"
    elif status == "PARTIAL":
        display = "PARTIAL"
    else:
        display = "LIVE"
    return mode, status, display


def build_run_record(
    final_state: Dict[str, Any], *, phase: int, latency: float, live_intent: bool, error: str = "", started_at: str = ""
) -> Dict[str, Any]:
    meta = dict(final_state.get("metadata", {}) or {})
    calls: List[Dict[str, Any]] = list(meta.get("llm_calls", []) or [])
    report_path = meta.get("report_path", "")
    has_report = bool((final_state.get("report") or "").strip())

    mode, status, display = derive_mode_and_status(
        live_intent, calls, final_state.get("execution_status", ""), has_report, error
    )
    if not error and final_state.get("execution_error"):
        error = final_state["execution_error"]

    est = [bool(c.get("estimated")) for c in calls]
    if not calls:
        basis = "n/a"
    elif not any(est):
        basis = "actual"
    elif all(est):
        basis = "estimated"
    else:
        basis = "mixed"

    node_lat = {k: v for k, v in meta.items() if k.startswith("latency_") and isinstance(v, (int, float))}
    node_tok = {k: v for k, v in meta.items() if k.startswith("tokens_") and isinstance(v, int)}
    provs = {h.get("provenance", "live") for hits in (final_state.get("search_results") or {}).values() for h in hits}
    search_prov = "none" if not provs else (next(iter(provs)) if len(provs) == 1 else "mixed")

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": final_state.get("run_id", ""),
        "benchmark_id": final_state.get("benchmark_id", "") or None,
        "question": final_state.get("question", ""),
        "phase": phase,
        "architecture": "phase2_critic_loop" if phase == 2 else "phase1_baseline",
        "execution_mode": mode,
        "execution_status": status,
        "display_status": display,
        "error": error,
        "model": get_model_name() if mode != "MOCK" else "mock-generator",
        "report_path": report_path,
        "latency": round(latency, 2),
        "input_tokens": _sum(calls, "prompt_tokens"),
        "output_tokens": _sum(calls, "completion_tokens"),
        "total_tokens": _sum(calls, "total_tokens"),
        "tokens_estimated": any(est),
        "token_basis": basis,
        "llm_call_count": len(calls),
        "searches": meta.get("search_count", 0),
        "search_provenance": search_prov,
        "node_latencies": node_lat,
        "node_tokens": node_tok,
        "plan": final_state.get("plan", []),
        "validation": final_state.get("validation", {}),
        "failed_subquestions": final_state.get("failed_subquestions", []),
        "isolation_violations": meta.get("isolation_violations", []),
        "started_at": started_at,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if phase == 2:
        record.update({
            "critic_iterations": meta.get("critic_iterations", len(final_state.get("score_history", []))),
            "revisions": meta.get("revisions", final_state.get("revision_count", 0)),
            "iterations_used": meta.get("iterations_used", final_state.get("revision_count", 0)),
            "score_history": final_state.get("score_history", []),
            "final_verdict": meta.get("final_verdict", "n/a"),
            "cap_reached": bool(meta.get("cap_reached", False)),
            "critic_latency_total": round(sum(v for k, v in node_lat.items() if k.startswith("latency_critic")), 2),
            "reviser_latency_total": round(sum(v for k, v in node_lat.items() if k.startswith("latency_reviser")), 2),
            "latest_draft_version": meta.get("latest_draft_version"),
            "best_draft_version": meta.get("best_draft_version"),
            "final_draft_version": meta.get("final_draft_version"),
            "selection_policy": meta.get("selection_policy"),
            "loop_error": meta.get("loop_error", ""),
        })
    return record


def save_run_artifact(record: Dict[str, Any], final_state: Dict[str, Any]) -> str:
    """Persist the run record plus the full per-run snapshot (all draft versions, notes, brief). No secrets."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{record['run_id']}.json")
    snapshot = {
        "record": record,
        "snapshot": {
            "plan": final_state.get("plan", []),
            "validated_notes": final_state.get("validated_notes", {}),
            "research_records": final_state.get("research_records", []),
            "brief": final_state.get("brief", ""),
            "draft_history": final_state.get("draft_history", []),
            "score_history": final_state.get("score_history", []),
            "final_report": final_state.get("report", ""),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return path


def execute_run(
    question: str,
    with_critic: bool = False,
    *,
    benchmark_id: Optional[str] = None,
    run_id: Optional[str] = None,
    force_mock: bool = False,
    api_key: Optional[str] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    save: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute ONE isolated research run. Returns ``(run_record, final_state)``.
    Never raises for pipeline errors: they are captured as a FAILED record.
    """
    if not (question or "").strip():
        raise ValueError("question must be non-empty")

    phase = 2 if with_critic else 1
    state: ResearchState = new_run_state(question, run_id=run_id, benchmark_id=benchmark_id)
    rid = state["run_id"]

    graph = build_improved_graph() if with_critic else build_baseline_graph()
    config = {"configurable": {"thread_id": rid, "force_mock": force_mock, "__api_key": api_key}}

    with run_llm_context(api_key=api_key, force_mock=force_mock):
        live_intent = is_live_mode()

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    error = ""
    try:
        for event in graph.stream(state, config=config, stream_mode="updates"):
            if on_event:
                on_event(event)
    except Exception as exc:  # noqa: BLE001 - recorded as FAILED, never swallowed silently
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    latency = time.perf_counter() - t0

    try:
        final_state = dict(graph.get_state(config).values)
    except Exception:  # noqa: BLE001
        final_state = dict(state)
    final_state.setdefault("run_id", rid)

    record = build_run_record(final_state, phase=phase, latency=latency, live_intent=live_intent, error=error, started_at=started_at)
    if save:
        record["artifact_path"] = save_run_artifact(record, final_state)
    return record, final_state
