"""
Evaluation Runner for Benchmark Research Suite
Architecture Reference: docs/EXPLANATIONS.md - Section 10, 12 and Chapter 13

Supports:
- Phase 1 Baseline Evaluation
- Phase 2 Critic Loop Evaluation (via --with-critic)
- Explicit MOCK mode (--mock) for zero-cost testing and regression suites
- Stable Benchmark IDs (Q01..Q10) from eval.questions

Data Integrity:
- Appends to outputs/runs/eval_runs_v2.jsonl (schema v2, append-only).
- Preserves outputs/eval_runs.jsonl and outputs/phase1_baseline/ as frozen control data.
- Records actual input/output tokens from provider usage metadata, flags estimates.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.questions import BENCHMARK_QUESTIONS, question_for_id
from eval.data import DataPaths, load_runs, run_is_valid
from graph.run_record import execute_run


def has_valid_run(benchmark_id: str, phase: int, paths: Optional[DataPaths] = None) -> Optional[Dict[str, Any]]:
    """Check if a benchmark question already has a valid LIVE completed run."""
    paths = paths or DataPaths()
    all_runs = load_runs(paths)
    matches = [
        r for r in all_runs
        if r.get("benchmark_id") == benchmark_id and r.get("phase") == phase and run_is_valid(r)
    ]
    return matches[-1] if matches else None


def run_benchmark_item(
    item: Dict[str, str],
    with_critic: bool = False,
    force_mock: bool = False,
    api_key: Optional[str] = None,
    paths: Optional[DataPaths] = None,
) -> Dict[str, Any]:
    """Execute a single benchmark question with stable Q-ID tracking."""
    paths = paths or DataPaths()
    bid = item["id"]
    question = item["question"]
    phase = 2 if with_critic else 1
    mode_str = "Phase 2 (Critic Loop)" if with_critic else "Phase 1 (Baseline)"

    print(f"\n[BENCHMARK] Executing {bid} ({mode_str}): '{question}'")
    record, final_state = execute_run(
        question=question,
        with_critic=with_critic,
        benchmark_id=bid,
        force_mock=force_mock,
        api_key=api_key,
        save=True,
    )

    # Append to schema-v2 log
    v2_log_path = paths.v2_runs
    os.makedirs(os.path.dirname(v2_log_path), exist_ok=True)
    with open(v2_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[BENCHMARK] {bid} completed in {record['latency']}s | "
        f"Status: {record['display_status']} | Mode: {record['execution_mode']} | "
        f"Tokens: {record['total_tokens']} ({record['token_basis']})"
    )
    return record


def run_benchmark_suite(
    limit: Optional[int] = None,
    with_critic: bool = False,
    force_mock: bool = False,
    question_id: Optional[str] = None,
    rerun_all: bool = False,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run evaluation benchmark over the questions suite.
    Skips items that already have verified clean LIVE runs unless rerun_all=True.
    """
    paths = DataPaths()
    phase = 2 if with_critic else 1

    targets = list(BENCHMARK_QUESTIONS)
    if question_id:
        targets = [b for b in targets if b["id"].upper() == question_id.upper()]
        if not targets:
            print(f"[BENCHMARK] Unknown question ID: {question_id}")
            return []
    elif limit is not None:
        targets = targets[:limit]

    print(f"\n{'='*70}")
    print(f"=== BENCHMARK SUITE: {len(targets)} question(s) | Phase {phase} | Mock={force_mock} ===")
    print(f"{'='*70}\n")

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(targets, 1):
        bid = item["id"]
        if not rerun_all and not force_mock:
            existing = has_valid_run(bid, phase, paths)
            if existing:
                print(f"[BENCHMARK SKIP] {bid} already has verified LIVE run: {existing.get('report_path')}")
                results.append(existing)
                continue

        rec = run_benchmark_item(
            item,
            with_critic=with_critic,
            force_mock=force_mock,
            api_key=api_key,
            paths=paths,
        )
        results.append(rec)
        # Gentle inter-run pause
        time.sleep(2)

    print(f"\n[BENCHMARK SUITE COMPLETE] {len(results)} item(s) processed. Logged to: {paths.v2_runs}\n")
    return results


def main():
    with_critic = "--with-critic" in sys.argv
    force_mock = "--mock" in sys.argv
    rerun_all = "--all" in sys.argv
    cleaned_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    qid = None
    limit = None
    if cleaned_args:
        first = cleaned_args[0]
        if first.upper().startswith("Q") and len(first) == 3:
            qid = first.upper()
        else:
            try:
                limit = int(first)
            except ValueError:
                limit = None

    run_benchmark_suite(
        limit=limit,
        with_critic=with_critic,
        force_mock=force_mock,
        question_id=qid,
        rerun_all=rerun_all,
    )


if __name__ == "__main__":
    main()
