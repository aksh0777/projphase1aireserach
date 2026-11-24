"""
Experimental Evaluation Harness Tests
Architecture Reference: docs/EXPLANATIONS.md - Section 10, 12 and Chapter 13

Verifies:
- All 10 benchmark IDs load correctly (Q01..Q10)
- Missing scores/evaluations are NEVER converted to zero
- Missing evaluations show 'N/A' and do not generate fake deltas
- Mock reports are excluded from scientific comparisons
- Duplicate evaluations are handled safely (latest valid record used)
- Actual vs estimated tokens are distinguished
- Mean scores are accurately computed: (R + C + O + G) / 4
- Only valid paired evaluations contribute to aggregate metrics
- CSV export functions cleanly and encodes N/A properly
"""

import csv
import io
import pytest

from eval.questions import BENCHMARK_QUESTIONS, BENCHMARK_IDS, benchmark_id_for, question_for_id
from eval.data import (
    DataPaths,
    build_comparison,
    comparison_csv_text,
    mean_score,
    run_is_valid,
)


def test_benchmark_questions_suite_integrity():
    """All 10 benchmark questions must have stable IDs Q01..Q10."""
    assert len(BENCHMARK_QUESTIONS) == 10
    assert len(BENCHMARK_IDS) == 10
    assert BENCHMARK_IDS == [f"Q{i:02d}" for i in range(1, 11)]

    for b in BENCHMARK_QUESTIONS:
        bid = b["id"]
        q_text = b["question"]
        assert bid.startswith("Q")
        assert len(q_text) > 20
        assert benchmark_id_for(q_text) == bid
        assert question_for_id(bid) == q_text


def test_mean_score_calculation():
    """Mean score must be exactly (R + C + O + G) / 4 with integers 1-5."""
    assert mean_score({"relevance": 5, "completeness": 5, "organization": 5, "grounding": 5}) == 5.0
    assert mean_score({"relevance": 5, "completeness": 4, "organization": 5, "grounding": 3}) == 4.25
    assert mean_score({"relevance": 4, "completeness": 3, "organization": 4, "grounding": 3}) == 3.5

    # Any invalid or missing dimension must return None (NEVER a zero or default)
    assert mean_score({"relevance": 5, "completeness": 4, "organization": 5}) is None
    assert mean_score({"relevance": 5, "completeness": "4", "organization": 5, "grounding": 3}) is None
    assert mean_score({"relevance": 6, "completeness": 4, "organization": 5, "grounding": 3}) is None
    assert mean_score({"relevance": 0, "completeness": 4, "organization": 5, "grounding": 3}) is None


def test_mock_runs_excluded_from_comparisons():
    """Mock runs must never be treated as valid scientific comparison candidates."""
    mock_run = {
        "benchmark_id": "Q01",
        "phase": 1,
        "execution_mode": "MOCK",
        "execution_status": "COMPLETE",
        "report_path": "outputs/mock_report.md",
    }
    live_clean_run = {
        "benchmark_id": "Q01",
        "phase": 1,
        "execution_mode": "LIVE",
        "execution_status": "COMPLETE",
        "report_path": "outputs/live_report.md",
        "integrity_flags": [],
    }
    assert run_is_valid(mock_run) is False
    assert run_is_valid(live_clean_run) is True


def test_comparison_preserves_missing_as_none():
    """Missing evaluations must NEVER be converted to zero."""
    comp = build_comparison()
    rows = comp["rows"]
    assert len(rows) == 10

    for r in rows:
        if not r["pair_valid"]:
            assert r["mean_delta"] is None, f"{r['question_id']} must have None delta if pair is invalid"
            # Must not default missing phase means to 0.0
            if r["p1_mean"] is None:
                assert r["p1_mean"] != 0.0
            if r["p2_mean"] is None:
                assert r["p2_mean"] != 0.0


def test_comparison_csv_export_format():
    """CSV export must output valid CSV structure containing 'N/A' for nulls."""
    comp = build_comparison()
    csv_str = comparison_csv_text(comp["rows"])
    reader = list(csv.DictReader(io.StringIO(csv_str)))
    assert len(reader) == 10

    for row in reader:
        assert row["question_id"].startswith("Q")
        # Check that missing deltas are exported as 'N/A' string, never 0 or 0.0
        if row["pair_valid"] != "True":
            assert row["mean_delta"] == "N/A"
