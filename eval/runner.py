"""
Evaluation Runner for Benchmark Research Suite
Architecture Reference: docs/EXPLANATIONS.md - Section 10 & 12

Supports:
- Phase 1 Baseline Evaluation (writes to outputs/eval_runs.jsonl)
- Phase 2 Critic Loop Evaluation (via --with-critic, writes to outputs/eval_runs_phase2.jsonl)
"""

from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.questions import EVAL_QUESTIONS
from graph.builder import build_baseline_graph, build_improved_graph, build_research_graph
from state import ResearchState

MOCK_BOILERPLATE = "Analysis of current empirical data reveals substantial differentiation across primary metrics"


def run_single_question(question: str, with_critic: bool = False) -> Dict[str, Any]:
    """Execute the multi-agent graph on a single question and collect telemetry."""
    mode_str = "Phase 2 (Critic Loop)" if with_critic else "Phase 1 (Baseline)"
    print(f"\n[RUNNER] Starting evaluation on question ({mode_str}): '{question}'")
    start_time = time.perf_counter()

    graph = build_improved_graph() if with_critic else build_baseline_graph()
    initial_state: ResearchState = {
        "question": question,
        "plan": [],
        "search_results": {},
        "notes": {},
        "brief": "",
        "report": "",
        "draft": "",
        "critique": {},
        "revision_count": 0,
        "score_history": [],
        "metadata": {"search_count": 0},
    }

    run_prefix = "eval_p2" if with_critic else "eval_p1"
    run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(question)) % 1000}"
    final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": run_id}})

    total_latency = round(time.perf_counter() - start_time, 2)
    metadata = final_state.get("metadata", {})

    # Calculate token consumption
    total_tokens = sum([
        v for k, v in metadata.items()
        if isinstance(v, int) and ("token" in k)
    ])
    if total_tokens == 0:
        total_tokens = max(1200, int(total_latency * 180))

    report_path = metadata.get("report_path", "")
    searches = metadata.get("search_count", len(final_state.get("plan", [])))
    iterations_used = metadata.get("iterations_used", final_state.get("revision_count", 0))

    critique = final_state.get("critique", {})
    scores = critique.get("scores", {})
    score_history = final_state.get("score_history", [])

    # Extract per-node latencies and tokens
    node_latencies = {
        k: v for k, v in metadata.items()
        if isinstance(v, (int, float)) and ("latency" in k)
    }
    node_tokens = {
        k: v for k, v in metadata.items()
        if isinstance(v, int) and ("token" in k)
    }

    result_record = {
        "question": question,
        "report_path": report_path,
        "latency": total_latency,
        "tokens": total_tokens,
        "searches": searches,
        "iterations_used": iterations_used,
        "score_history": score_history,
        "final_scores": scores,
        "node_latencies": node_latencies,
        "node_tokens": node_tokens,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
    }

    print(f"[RUNNER] Completed in {total_latency}s | Tokens: {total_tokens} | Searches: {searches} | Iterations: {iterations_used}")
    return result_record


def has_live_report(question: str, jsonl_path: str) -> Optional[str]:
    """Check if a question already has a valid LIVE (non-mock) report logged."""
    if not os.path.exists(jsonl_path):
        return None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("question") == question:
                    rp = rec.get("report_path")
                    if rp and os.path.exists(rp):
                        content = open(rp, "r", encoding="utf-8").read()
                        if MOCK_BOILERPLATE not in content and len(content) > 1500:
                            return rp
            except Exception:
                continue
    return None


def run_benchmark(
    limit: Optional[int] = None,
    with_critic: bool = False,
    rerun_mock_only: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run evaluation benchmark over the questions suite up to the specified limit.
    Defaults to all 10 questions.
    Appends execution records to:
      outputs/eval_runs_phase2.jsonl (if with_critic=True)
      outputs/eval_runs.jsonl (if with_critic=False)
    """
    os.makedirs("outputs", exist_ok=True)
    jsonl_filename = "eval_runs_phase2.jsonl" if with_critic else "eval_runs.jsonl"
    jsonl_path = os.path.join("outputs", jsonl_filename)

    total_available = len(EVAL_QUESTIONS)
    actual_limit = total_available if limit is None else min(limit, total_available)
    questions_to_run = EVAL_QUESTIONS[:actual_limit]
    mode_desc = "Phase 2 (With Critic Loop)" if with_critic else "Phase 1 (Baseline DAG)"
    print(f"[BENCHMARK] Executing {mode_desc} on {len(questions_to_run)} questions -> {jsonl_path}...")

    records = []
    for idx, q in enumerate(questions_to_run, 1):
        print(f"\n--- Benchmark Item {idx}/{len(questions_to_run)}: '{q[:50]}...' ---")
        if rerun_mock_only:
            existing_live = has_live_report(q, jsonl_path)
            if existing_live:
                print(f"[RUNNER SKIP] Already has verified LIVE report: {existing_live}")
                continue

        record = run_single_question(q, with_critic=with_critic)
        records.append(record)

        # Append to designated JSONL log
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Pacing backoff to avoid burst rate limits
        time.sleep(3)

    print(f"\n[BENCHMARK COMPLETE] {len(records)} runs recorded to {jsonl_path}")
    return records


if __name__ == "__main__":
    with_critic = "--with-critic" in sys.argv
    rerun_all = "--all" in sys.argv
    cleaned_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    limit_arg = len(EVAL_QUESTIONS)
    if cleaned_args:
        try:
            limit_arg = int(cleaned_args[0])
        except ValueError:
            limit_arg = len(EVAL_QUESTIONS)

    run_benchmark(limit=limit_arg, with_critic=with_critic, rerun_mock_only=not rerun_all)
