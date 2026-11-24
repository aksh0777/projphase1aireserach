"""
Research Execution Isolation Tests
Architecture Reference: docs/EXPLANATIONS.md - Chapter 13

Verifies that consecutive research executions:
- Have unique run IDs
- Create fresh state containers without shared mutable defaults
- Do NOT leak cross-topic material (Cricket vs Batteries vs RAG vs Cancer)
- Associate researcher outputs strictly with the assigned sub-question
- Synthesize strictly within the current run's boundary
- Work reliably across both Phase 1 and Phase 2 pipelines
"""

import pytest
from graph.run_record import execute_run
from state import new_run_state, current_run_records

TOPIC_CRICKET = "How do the captaincy records of MS Dhoni compare to Virat Kohli across Tests, ODIs and T20Is?"
TOPIC_BATTERY = "Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026"
TOPIC_RAG = "Retrieval-Augmented Generation (RAG) vs fine-tuning for domain-specific enterprise knowledge retrieval"
TOPIC_CANCER = "Clinical efficacy and false positive rates of AI in early cancer detection from medical imaging"


def test_new_run_state_isolation():
    """State factory must produce decoupled, uniquely identified state objects."""
    s1 = new_run_state(TOPIC_CRICKET)
    s2 = new_run_state(TOPIC_BATTERY)

    assert s1["run_id"] != s2["run_id"]
    assert s1["question"] == TOPIC_CRICKET
    assert s2["question"] == TOPIC_BATTERY

    # Modifying s1 containers must not mutate s2
    s1["plan"].append("Cricket sub-question")
    s1["notes"]["Cricket sub-question"] = ["Note 1"]
    s1["metadata"]["search_count"] = 5

    assert len(s2["plan"]) == 0
    assert len(s2["notes"]) == 0
    assert s2["metadata"]["search_count"] == 0


def test_current_run_records_drops_foreign_runs():
    """Synthesizer & Validator must isolate records by run_id."""
    state = {
        "run_id": "run_current_123",
        "research_records": [
            {"run_id": "run_current_123", "sub_question": "Current Q1", "status": "ok", "findings": ["Finding 1"]},
            {"run_id": "run_OLD_LEAKED_456", "sub_question": "Old Battery Q", "status": "ok", "findings": ["Battery finding"]},
        ]
    }
    valid, foreign = current_run_records(state)
    assert len(valid) == 1
    assert valid[0]["sub_question"] == "Current Q1"
    assert len(foreign) == 1
    assert foreign[0]["run_id"] == "run_OLD_LEAKED_456"


@pytest.mark.parametrize("with_critic", [False, True])
def test_consecutive_runs_no_cross_contamination(with_critic):
    """
    Execute 4 consecutive runs on completely disparate topics in the same process session:
    1. Cricket (Dhoni vs Kohli)
    2. Batteries (Lithium vs Sodium)
    3. Software/AI (RAG vs Fine-Tuning)
    4. Medicine (Cancer Detection)
    Verify zero cross-topic leakage into subsequent reports.
    """
    topics = [
        (TOPIC_CRICKET, ["battery", "lithium", "sodium", "cathode", "anode", "cell-level"]),
        (TOPIC_BATTERY, ["dhoni", "kohli", "cricket", "odi", "t20i", "captaincy"]),
        (TOPIC_RAG, ["battery", "lithium", "sodium", "dhoni", "kohli", "cancer", "oncology"]),
        (TOPIC_CANCER, ["battery", "lithium", "sodium", "dhoni", "kohli", "rag", "fine-tuning"]),
    ]

    executed_records = []
    for topic, forbidden_words in topics:
        record, final_state = execute_run(
            question=topic,
            with_critic=with_critic,
            force_mock=True,
            save=False,
        )
        executed_records.append((record, final_state, forbidden_words))

    # Verification passes
    assert len({r[0]["run_id"] for r in executed_records}) == 4, "Every run must have a unique run_id"

    for record, final_state, forbidden in executed_records:
        report_text = final_state.get("report", "").lower()
        brief_text = final_state.get("brief", "").lower()

        # Check report does not contain forbidden cross-topic terms (whole words)
        import re
        for word in forbidden:
            pattern = rf"\b{re.escape(word.lower())}\b"
            assert not re.search(pattern, report_text), f"Report for '{record['question'][:30]}' contaminated with '{word}'"
            assert not re.search(pattern, brief_text), f"Brief for '{record['question'][:30]}' contaminated with '{word}'"

        # Check sub-questions correspond to the target question
        for sq in final_state.get("plan", []):
            sq_lower = sq.lower()
            for word in forbidden:
                pattern = rf"\b{re.escape(word.lower())}\b"
                assert not re.search(pattern, sq_lower), f"Subquestion '{sq}' contains forbidden term '{word}'"



def test_research_records_provenance_tagging():
    """All research records emitted during execution must carry the state run_id."""
    record, final_state = execute_run(
        question="How do the captaincy records of MS Dhoni compare to Virat Kohli?",
        with_critic=False,
        force_mock=True,
        save=False,
    )
    rid = record["run_id"]
    records = final_state.get("research_records", [])
    assert len(records) > 0
    for r in records:
        assert r.get("run_id") == rid
        assert r.get("status") in ("ok", "no_evidence", "failed", "no_valid_findings")
