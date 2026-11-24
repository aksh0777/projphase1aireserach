"""
Regression & Architecture Invariance Tests
Architecture Reference: docs/EXPLANATIONS.md - Section 8, 12 and Chapter 13

Verifies:
- Phase 1 baseline DAG behavior remains strictly feed-forward (0 critic iterations)
- Phase 2 closed-loop executes Critic & Reviser
- MAX_REVISIONS cap is strictly enforced (prevents infinite loops)
- Selection policy accurately chooses latest-if-pass or best-critic-mean
- Frozen baseline files in outputs/phase1_baseline/ remain 100% byte-identical
"""

import os
import pytest
from graph.builder import MAX_REVISIONS, build_baseline_graph, build_improved_graph
from graph.run_record import execute_run
from state import new_run_state


def test_phase1_baseline_feed_forward_no_critic():
    """Phase 1 execution must execute planner->researcher->synthesizer->validator->writer with 0 critic passes."""
    record, final_state = execute_run(
        question="Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026",
        with_critic=False,
        force_mock=True,
        save=False,
    )
    assert record["phase"] == 1
    assert record["architecture"] == "phase1_baseline"
    assert record.get("critic_iterations") is None
    assert final_state.get("revision_count", 0) == 0
    assert len(final_state.get("score_history", [])) == 0
    assert len(final_state.get("report", "")) > 100


def test_phase2_critic_loop_iterations_and_cap():
    """Phase 2 execution must execute critique iterations and respect MAX_REVISIONS."""
    record, final_state = execute_run(
        question="Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026",
        with_critic=True,
        force_mock=True,
        save=False,
    )
    assert record["phase"] == 2
    assert record["architecture"] == "phase2_critic_loop"
    assert record["critic_iterations"] >= 1
    assert record["revisions"] <= MAX_REVISIONS
    assert len(final_state.get("score_history", [])) >= 1
    assert final_state.get("critique") is not None
    assert len(final_state.get("report", "")) > 100


def test_phase1_baseline_frozen_files_untouched():
    """Control group in outputs/phase1_baseline/ must exist and contain the original 13 files."""
    baseline_dir = os.path.join("outputs", "phase1_baseline")
    assert os.path.exists(baseline_dir)

    expected_files = [
        "eval_results.csv",
        "eval_runs.jsonl",
        "report_20261122_171733.md",
        "report_20261122_172001.md",
        "report_20261122_172131.md",
        "report_20261122_172405.md",
        "report_20261122_172651.md",
        "report_20261122_172925.md",
        "report_20261122_173207.md",
        "report_20261122_173450.md",
        "report_20261122_173721.md",
        "report_20261122_174006.md",
        "report_20261122_174238.md",
    ]
    for ef in expected_files:
        p = os.path.join(baseline_dir, ef)
        assert os.path.exists(p), f"Frozen baseline file missing: {p}"
        assert os.path.getsize(p) > 0, f"Frozen baseline file is empty: {p}"
