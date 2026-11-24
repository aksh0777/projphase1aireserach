"""
Evaluation Comparison Engine: Phase 1 Baseline vs Phase 2 Critic Loop
Architecture Reference: docs/EXPLANATIONS.md - Section 12 and Chapter 13

Joins Phase 1 and Phase 2 evaluations on stable benchmark IDs (Q01..Q10).
Produces outputs/comparison_table.csv and prints the empirical comparison matrix.
Calculates deltas and aggregates ONLY over valid paired evaluations.
Missing scores remain None / 'N/A' (never converted to 0).
"""

import os
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.data import (
    DataPaths,
    build_comparison,
    comparison_csv_text,
)


def run_comparison(output_csv: str = "outputs/comparison_table.csv", allow_flagged: bool = False) -> Dict[str, Any]:
    """Build comparison table, export CSV, and print academic matrix."""
    paths = DataPaths()
    comp = build_comparison(paths=paths, allow_flagged=allow_flagged)
    rows = comp["rows"]
    agg = comp["aggregates"]
    counts = comp["counts"]

    # Export CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write(comparison_csv_text(rows))

    # Print Formatted Comparison Matrix
    print("\n" + "=" * 115)
    print("========== PHASE 1 (BASELINE) VS PHASE 2 (CRITIC LOOP) COMPARISON MATRIX ==========")
    print("=" * 115)
    header = (
        f"{'ID':<4} | {'Question':<38} | {'P1 (R/C/O/G)':<16} | {'P2 (R/C/O/G)':<16} | "
        f"{'P1 Mean':<7} | {'P2 Mean':<7} | {'Delta':<6} | {'Iters':<5} | {'Note':<15}"
    )
    print(header)
    print("-" * 115)

    for r in rows:
        bid = r["question_id"]
        q_disp = (r["question"][:35] + "...") if len(r["question"]) > 35 else r["question"]

        def fmt_scores(tag):
            if r.get(f"{tag}_mean") is None:
                return "N/A"
            return f"{r.get(f'{tag}_relevance')}/{r.get(f'{tag}_completeness')}/{r.get(f'{tag}_organization')}/{r.get(f'{tag}_grounding')}"

        p1_sc = fmt_scores("p1")
        p2_sc = fmt_scores("p2")
        m1 = f"{r['p1_mean']:.2f}" if r.get("p1_mean") is not None else "N/A"
        m2 = f"{r['p2_mean']:.2f}" if r.get("p2_mean") is not None else "N/A"
        d_str = (f"+{r['mean_delta']:.2f}" if r["mean_delta"] > 0 else f"{r['mean_delta']:.2f}") if r.get("mean_delta") is not None else "N/A"
        iters = str(r.get("p2_iterations") or "0")
        note = r.get("pair_note", "")[:18]

        print(
            f"{bid:<4} | {q_disp:<38} | {p1_sc:<16} | {p2_sc:<16} | "
            f"{m1:<7} | {m2:<7} | {d_str:<6} | {iters:<5} | {note:<18}"
        )

    print("=" * 115)
    print(f"Comparison table saved to: {output_csv}\n")

    # Print Aggregate Statistics
    n = agg["valid_pairs"]
    print("=== AGGREGATE EXPERIMENTAL METRICS ===")
    print(f"Total benchmark questions:             {counts['total_questions']}")
    print(f"Valid paired LIVE evaluations:        {n} of {counts['total_questions']}")
    if n > 0:
        print(f"Phase 1 average quality:              {agg['avg_p1_quality']:.2f} / 5.0")
        print(f"Phase 2 average quality:              {agg['avg_p2_quality']:.2f} / 5.0")
        impr = agg['avg_improvement']
        print(f"Average quality improvement (delta):   {'+' if impr >= 0 else ''}{impr:.2f}")
        print(f"Questions improved / tied / worse:    {agg['pct_improved']}% / {agg['pct_unchanged']}% / {agg['pct_deteriorated']}%")
        print(f"Phase 1 average latency:              {agg['avg_p1_latency']:.2f}s")
        print(f"Phase 2 average latency:              {agg['avg_p2_latency']:.2f}s (Overhead: +{agg['latency_overhead']}s, +{agg['latency_overhead_pct']}%)")
        if agg["valid_pairs_for_tokens"] > 0:
            print(f"Phase 1 average actual tokens:        {agg['avg_p1_tokens']:.0f}")
            print(f"Phase 2 average actual tokens:        {agg['avg_p2_tokens']:.0f} (Overhead: +{agg['token_overhead']:.0f}, +{agg['token_overhead_pct']}%)")
        else:
            print("Token consumption comparison:         N/A (legacy runs lacked verified token telemetry)")
    else:
        print("Note: No benchmark questions have verified LIVE evaluations in BOTH Phase 1 and Phase 2.")
        print("Aggregate delta and cost statistics require valid paired LIVE evaluations.")

    print("======================================\n")
    return comp


if __name__ == "__main__":
    allow_flagged = "--allow-flagged" in sys.argv
    run_comparison(allow_flagged=allow_flagged)
