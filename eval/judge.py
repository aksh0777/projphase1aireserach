"""
LLM-as-a-Judge Independent External Evaluator
Architecture Reference: docs/EXPLANATIONS.md - Section 10 and Chapter 13

First Principle:
Independent Post-Hoc Evaluation:
The internal Critic Agent operates as a closed-loop controller during draft revision.
The external Judge evaluates the COMPLETED final reports across the entire benchmark
to produce standardized, reproducible experimental measurements.

Integrity Guarantees:
- Context Window Preservation: Gemini 3.5/Flash has a large context window (>1M tokens).
  Reports are evaluated IN FULL without arbitrary 12k character truncation.
- Rejection of Mock / Ineligible Reports: Refuses to score mock templates or simulated
  data. Records status as "REJECTED:<reason>" without fabricating scores.
- Immutable Audit Trail: Every evaluation attempt (success or rejection) is appended
  with timestamp, evaluation_id, judge_model, rubric_version to outputs/runs/judge_evaluations.jsonl.
- Does NOT overwrite legacy files (outputs/eval_results.csv remains frozen).
"""

import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.llm import LLMUnavailableError, call_llm, get_model_name, is_live_mode, usage_record
from eval.data import (
    DIMS,
    LEGACY_MOCK_SENTENCE,
    DataPaths,
    has_simulated_sources,
    is_mock_text,
    load_runs,
    mean_score,
    read_text,
    run_is_valid,
)
from eval.questions import benchmark_id_for, question_for_id

logger = logging.getLogger("Judge")

JUDGE_RUBRIC_VERSION = "rubric-v1-4dim-1to5"
MAX_SAFE_REPORT_CHARS = 120_000  # Well within Gemini Flash context limit (~30k tokens)


def _clean_and_parse_judge_json(text: str) -> Dict[str, Any]:
    text_clean = text.strip()
    if text_clean.startswith("```"):
        text_clean = re.sub(r"^```(?:json)?\s*", "", text_clean, flags=re.IGNORECASE)
        text_clean = re.sub(r"\s*```$", "", text_clean)
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        text_clean = match.group(0)
    return json.loads(text_clean)


def _validate_judge_scores(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    clean: Dict[str, int] = {}
    for d in DIMS:
        v = data.get(d)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not (1 <= int(v) <= 5):
            return None
        clean[d] = int(v)
    return clean


def score_report_with_llm(
    question: str,
    report_text: str,
    judge_model: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[str, str], str, Dict[str, Any]]:
    """
    Execute external LLM-as-a-Judge scoring on full report text.
    Returns (scores, justifications, overall_justification, usage).
    """
    model_name = judge_model or get_model_name()

    system_instruction = (
        "You are an impartial academic journal referee. Evaluate the provided scientific research report "
        "relative to the original research question across four standard criteria.\n\n"
        "Score each criterion on an integer scale from 1 (poor) to 5 (flawless):\n"
        "1. Relevance (1-5): Does the report directly address the target inquiry without drift?\n"
        "2. Completeness (1-5): Does it address all technical, economic, and operational facets of the question?\n"
        "3. Organization (1-5): Is the report structured logically with clear headings, transitions, and conclusion?\n"
        "4. Grounding (1-5): Are factual claims strictly supported by numbered in-text citations ([1], [2]) "
        "linked to a verified references section?\n\n"
        "STRICT OUTPUT REQUIREMENT:\n"
        "Respond ONLY with a valid JSON object in this format:\n"
        "{\n"
        '  "relevance": int,\n'
        '  "completeness": int,\n'
        '  "organization": int,\n'
        '  "grounding": int,\n'
        '  "justifications": {\n'
        '    "relevance": "one-line justification",\n'
        '    "completeness": "one-line justification",\n'
        '    "organization": "one-line justification",\n'
        '    "grounding": "one-line justification"\n'
        '  },\n'
        '  "justification": "One-line overall justification summarizing the assessment."\n'
        "}"
    )

    # Use full report text (safely bound to context ceiling)
    evaluated_text = report_text[:MAX_SAFE_REPORT_CHARS]
    prompt = (
        f"Research Question: {question}\n\n"
        f"Full Research Report Content ({len(evaluated_text)} chars):\n"
        f"{evaluated_text}\n\n"
        "Evaluate the report strictly using the specified rubric:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.0,
        allow_mock=False,
        node="judge",
    )

    data = _clean_and_parse_judge_json(content)
    scores = _validate_judge_scores(data)
    if not scores:
        raise ValueError(f"Judge output contained invalid or missing scores: {content[:200]}")

    justifications = data.get("justifications", {})
    if not isinstance(justifications, dict):
        justifications = {}
    overall_just = str(data.get("justification", "")).strip()

    return scores, justifications, overall_just, tokens


def evaluate_run_record(
    run: Dict[str, Any],
    paths: Optional[DataPaths] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Score the report of a specific run record.
    Rejects mock reports, missing files, or non-LIVE runs.
    Appends the evaluation entry to outputs/runs/judge_evaluations.jsonl.
    """
    paths = paths or DataPaths()
    run_id = run.get("run_id", "")
    phase = int(run.get("phase", 1))
    bid = run.get("benchmark_id") or benchmark_id_for(run.get("question", ""))
    question = run.get("question") or (question_for_id(bid) if bid else "")
    rp = run.get("report_path", "")

    eval_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    record_template = {
        "evaluation_id": eval_id,
        "run_id": run_id,
        "benchmark_id": bid,
        "phase": phase,
        "question": question,
        "report_path": rp,
        "judge_model": get_model_name(),
        "rubric_version": JUDGE_RUBRIC_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Guard 1: LIVE mode check
    if not is_live_mode():
        rec = {**record_template, "status": "REJECTED:judge_not_live", "error": "LLM is not in LIVE mode"}
        _append_judge_record(rec, paths)
        print(f"[JUDGE] REJECTED: {bid} (Judge cannot execute without active live LLM)")
        return rec

    # Guard 2: Report existence
    text = read_text(rp) if rp else None
    if text is None:
        rec = {**record_template, "status": "REJECTED:report_missing", "error": f"Report file not found: {rp}"}
        _append_judge_record(rec, paths)
        print(f"[JUDGE] REJECTED: {bid} (Report file missing: {rp})")
        return rec

    # Guard 3: Mock boilerplate or simulated content check
    if is_mock_text(text) or run.get("execution_mode") == "MOCK":
        rec = {**record_template, "status": "REJECTED:mock_report", "error": "Report contains resilient mock template text"}
        _append_judge_record(rec, paths)
        print(f"[JUDGE] REJECTED: {bid} (Report contains mock template text, refusing to score)")
        return rec

    if has_simulated_sources(text) and not force:
        rec = {**record_template, "status": "REJECTED:simulated_sources", "error": "Report cites simulated search domains"}
        _append_judge_record(rec, paths)
        print(f"[JUDGE] REJECTED: {bid} (Report cites simulated search domains)")
        return rec

    # Execute LLM scoring
    print(f"\n[JUDGE] Scoring {bid} (Phase {phase}, {len(text)} chars) -> Model: {get_model_name()}...")
    try:
        scores, justifications, overall_just, usage = score_report_with_llm(question, text)
        mean_v = round(sum(scores.values()) / 4.0, 4)
        rec = {
            **record_template,
            "status": "OK",
            **scores,
            "mean_score": mean_v,
            "justification": overall_just,
            "justifications": justifications,
            "report_chars": len(text),
            "usage": usage_record("judge", usage),
        }
        _append_judge_record(rec, paths)

        print("\n" + "=" * 60)
        print(f"========== INDEPENDENT JUDGE EVALUATION ({bid} P{phase}) ==========")
        print("=" * 60)
        print(f"Report:         {rp}")
        for d in DIMS:
            print(f"  • {d.title():<13} {scores[d]} / 5")
        print(f"  • Mean Score:   {mean_v:.2f} / 5.0")
        print(f"Justification:  {overall_just}")
        print("=" * 60 + "\n")
        return rec

    except Exception as exc:
        rec = {**record_template, "status": f"REJECTED:error_{type(exc).__name__}", "error": str(exc)[:400]}
        _append_judge_record(rec, paths)
        print(f"[JUDGE ERROR] Failed to score {bid}: {exc}")
        return rec


def _append_judge_record(rec: Dict[str, Any], paths: DataPaths) -> None:
    os.makedirs(os.path.dirname(paths.v2_judge), exist_ok=True)
    with open(paths.v2_judge, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def evaluate_all_runs(
    phase: Optional[int] = None,
    benchmark_id: Optional[str] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate all eligible completed runs."""
    paths = DataPaths()
    runs = load_runs(paths)

    targets = [
        r for r in runs
        if run_is_valid(r, allow_flagged=force)
    ]
    if phase:
        targets = [r for r in targets if r.get("phase") == phase]
    if benchmark_id:
        targets = [r for r in targets if r.get("benchmark_id") == benchmark_id.upper()]

    print(f"[JUDGE] Found {len(targets)} eligible run(s) for independent scoring.")
    results = []
    for r in targets:
        res = evaluate_run_record(r, paths=paths, force=force)
        results.append(res)
    return results


def main():
    phase_arg = 2 if "--phase2" in sys.argv else (1 if "--phase1" in sys.argv else None)
    force = "--force" in sys.argv
    cleaned = [a for a in sys.argv[1:] if not a.startswith("--")]

    bid = cleaned[0].upper() if cleaned and cleaned[0].upper().startswith("Q") else None
    evaluate_all_runs(phase=phase_arg, benchmark_id=bid, force=force)


if __name__ == "__main__":
    main()
