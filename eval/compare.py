"""
Evaluation Comparison Engine: Phase 1 Baseline vs Phase 2 Critic Loop
Architecture Reference: docs/EXPLANATIONS.md - Section 12

Loads Phase 1 and Phase 2 evaluation artifacts, joins on research question,
calculates rubric score deltas, latency overhead, and token efficiency,
and exports the empirical comparison table to outputs/comparison_table.csv.
"""

import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_csv_scores(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load latest evaluation scores keyed by normalized question string."""
    scores_by_q: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(csv_path):
        return scores_by_q

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question", "").strip()
            if q:
                scores_by_q[q] = {
                    "relevance": float(row.get("Relevance", 0)),
                    "completeness": float(row.get("Completeness", 0)),
                    "organization": float(row.get("Organization", 0)),
                    "grounding": float(row.get("Grounding", 0)),
                    "mean_score": float(row.get("Mean_Score", 0)),
                    "justification": row.get("Justification", ""),
                    "report_path": row.get("Report_Path", ""),
                }
    return scores_by_q


def load_jsonl_telemetry(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    """Load latest run telemetry (latency, tokens, iterations) keyed by question."""
    telemetry_by_q: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        return telemetry_by_q

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                q = rec.get("question", "").strip()
                if q:
                    telemetry_by_q[q] = {
                        "latency": float(rec.get("latency", 0)),
                        "tokens": int(rec.get("tokens", 0)),
                        "iterations_used": int(rec.get("iterations_used", 0)),
                        "report_path": rec.get("report_path", ""),
                    }
            except Exception:
                continue
    return telemetry_by_q


def build_comparison_table(
    p1_csv: str = "outputs/eval_results.csv",
    p1_jsonl: str = "outputs/eval_runs.jsonl",
    p2_csv: str = "outputs/eval_results_phase2.csv",
    p2_jsonl: str = "outputs/eval_runs_phase2.jsonl",
    output_csv: str = "outputs/comparison_table.csv",
) -> List[Dict[str, Any]]:
    """Produce structured Phase 1 vs Phase 2 comparative matrix."""
    p1_scores = load_csv_scores(p1_csv)
    p1_telemetry = load_jsonl_telemetry(p1_jsonl)

    p2_scores = load_csv_scores(p2_csv)
    p2_telemetry = load_jsonl_telemetry(p2_jsonl)

    # Collect union of all questions
    all_questions = list(dict.fromkeys(
        list(p1_scores.keys()) + list(p2_scores.keys()) +
        list(p1_telemetry.keys()) + list(p2_telemetry.keys())
    ))

    rows: List[Dict[str, Any]] = []

    for q in all_questions:
        s1 = p1_scores.get(q, {})
        t1 = p1_telemetry.get(q, {})
        s2 = p2_scores.get(q, {})
        t2 = p2_telemetry.get(q, {})

        m1 = s1.get("mean_score", 0.0)
        m2 = s2.get("mean_score", 0.0)
        delta = round(m2 - m1, 2) if (m1 > 0 and m2 > 0) else round(m2 or (-m1), 2)

        p1_scores_str = f"R:{int(s1.get('relevance', 0))} C:{int(s1.get('completeness', 0))} O:{int(s1.get('organization', 0))} G:{int(s1.get('grounding', 0))}" if s1 else "N/A"
        p2_scores_str = f"R:{int(s2.get('relevance', 0))} C:{int(s2.get('completeness', 0))} O:{int(s2.get('organization', 0))} G:{int(s2.get('grounding', 0))}" if s2 else "N/A"

        row = {
            "question": q,
            "phase1_scores": p1_scores_str,
            "phase2_scores": p2_scores_str,
            "phase1_mean": m1,
            "phase2_mean": m2,
            "mean_delta": delta,
            "latency_p1": t1.get("latency", 0.0),
            "latency_p2": t2.get("latency", 0.0),
            "tokens_p1": t1.get("tokens", 0),
            "tokens_p2": t2.get("tokens", 0),
            "iterations_used": t2.get("iterations_used", 0),
        }
        rows.append(row)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "question",
            "phase1_scores",
            "phase2_scores",
            "phase1_mean",
            "phase2_mean",
            "mean_delta",
            "latency_p1",
            "latency_p2",
            "tokens_p1",
            "tokens_p2",
            "iterations_used",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print formatted console report
    print("\n" + "=" * 110)
    print("========== PHASE 1 (BASELINE) VS PHASE 2 (CRITIC LOOP) COMPARISON MATRIX ==========")
    print("=" * 110)
    print(f"{'#':<3} | {'Question':<38} | {'P1 Scores':<16} | {'P2 Scores':<16} | {'Delta':<6} | {'Lat P1':<7} | {'Lat P2':<7} | {'Iters':<5}")
    print("-" * 110)

    for idx, r in enumerate(rows, 1):
        q_disp = (r['question'][:35] + "...") if len(r['question']) > 35 else r['question']
        d_str = f"+{r['mean_delta']}" if r['mean_delta'] > 0 else f"{r['mean_delta']}"
        print(
            f"{idx:<3} | {q_disp:<38} | {r['phase1_scores']:<16} | {r['phase2_scores']:<16} | "
            f"{d_str:<6} | {r['latency_p1']:<6.1f}s | {r['latency_p2']:<6.1f}s | {r['iterations_used']:<5}"
        )

    print("=" * 110)
    print(f"Comparison saved to: {output_csv}\n")
    return rows


if __name__ == "__main__":
    build_comparison_table()
