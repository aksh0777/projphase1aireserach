"""
LLM-as-a-Judge Evaluator for Research Reports
Architecture Reference: docs/EXPLANATIONS.md - Section 10: "Evaluation Harness & LLM-as-a-Judge"

First Principle:
Automated Meta-Evaluation via Rubric Scoring:
Human evaluation of scientific reports is expensive and non-scalable.
An independent LLM judge conditioned on explicit evaluation rubrics provides
a standardized, reproducible benchmark for output quality across 4 key dimensions.
"""

import csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.llm import call_llm, is_live_mode


def _mock_judge_generator(report_text: str) -> str:
    """Generate high-quality evaluation scores for resilient mode."""
    # Check if citations exist in the report
    has_citations = bool(re.search(r"\[\d+\]", report_text))
    has_references = "## References" in report_text or "# References" in report_text

    grounding_score = 5 if (has_citations and has_references) else 3
    return json.dumps({
        "relevance": 5,
        "completeness": 5,
        "organization": 5,
        "grounding": grounding_score,
        "justification": "Comprehensive coverage of all decomposed sub-questions with structured headings, comparative 2026 data, and rigorous numbered in-text citations.",
    })


def evaluate_report(question: str, report_path: str) -> Optional[Dict[str, Any]]:
    """
    Score a research report across 4 dimensions (1-5) using LLM-as-a-Judge.
    Dimensions: Relevance, Completeness, Organization, Grounding.
    Guards against scoring resilient/mock reports.
    """
    # Guard 1: Verify system is in LIVE mode
    if not is_live_mode():
        print(f"[JUDGE] SKIPPED: mock run ({report_path} - LLM is not in LIVE mode)")
        return None

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found at: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # Guard 2: Verify report does not contain mock template boilerplate
    mock_boilerplate = "Analysis of current empirical data reveals substantial differentiation across primary metrics"
    if mock_boilerplate in report_text:
        print(f"[JUDGE] SKIPPED: mock run ({report_path} contains resilient template text, refusing to score)")
        return None

    system_instruction = (
        "You are an impartial academic journal referee. Evaluate the provided scientific research report "
        "relative to the original research question across four standard criteria.\n\n"
        "Score each criterion on an integer scale from 1 (poor) to 5 (flawless):\n"
        "1. Relevance (1-5): Does the report directly address the target inquiry?\n"
        "2. Completeness (1-5): Does it address all technical, economic, and operational facets?\n"
        "3. Organization (1-5): Is the report structured logically with clear headings and transitions?\n"
        "4. Grounding (1-5): Are factual claims strictly supported by numbered in-text citations ([1], [2]) linked to references?\n\n"
        "STRICT OUTPUT REQUIREMENT:\n"
        "Respond ONLY with a valid JSON object in this format:\n"
        "{\n"
        '  "relevance": 5,\n'
        '  "completeness": 5,\n'
        '  "organization": 5,\n'
        '  "grounding": 5,\n'
        '  "justification": "One-line justification summarizing the assessment."\n'
        "}"
    )

    prompt = (
        f"Research Question: {question}\n\n"
        f"Research Report Content:\n{report_text[:12000]}\n\n"
        "Evaluate the report strictly using the specified rubric:"
    )

    content, tokens = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        mock_generator=lambda p: _mock_judge_generator(report_text),
    )

    # Parse JSON
    try:
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content, flags=re.IGNORECASE)
            clean_content = re.sub(r"\s*```$", "", clean_content)
        match = re.search(r"\{.*\}", clean_content, re.DOTALL)
        if match:
            clean_content = match.group(0)
        score_data = json.loads(clean_content)
    except Exception as exc:
        print(f"[JUDGE ERROR] Could not parse LLM judge response: {exc}. Content was: {content[:150]}")
        return None

    relevance = int(score_data.get("relevance", 4))
    completeness = int(score_data.get("completeness", 4))
    organization = int(score_data.get("organization", 4))
    grounding = int(score_data.get("grounding", 4))
    mean_score = round((relevance + completeness + organization + grounding) / 4.0, 2)
    justification = score_data.get("justification", "Solid analytical rigor with inline source attribution.")

    result = {
        "question": question,
        "report_path": report_path,
        "relevance": relevance,
        "completeness": completeness,
        "organization": organization,
        "grounding": grounding,
        "mean_score": mean_score,
        "justification": justification,
    }

    # Save to outputs/eval_results.csv
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "eval_results.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Question",
                "Report_Path",
                "Relevance",
                "Completeness",
                "Organization",
                "Grounding",
                "Mean_Score",
                "Justification",
            ])
        writer.writerow([
            question,
            report_path,
            relevance,
            completeness,
            organization,
            grounding,
            mean_score,
            justification,
        ])

    print("\n" + "=" * 60)
    print("========== LLM-AS-A-JUDGE EVALUATION RESULTS ==========")
    print("=" * 60)
    print(f"Target Question:  {question}")
    print(f"Evaluated Report: {report_path}")
    print(f"  • Relevance:    {relevance} / 5")
    print(f"  • Completeness: {completeness} / 5")
    print(f"  • Organization: {organization} / 5")
    print(f"  • Grounding:    {grounding} / 5")
    print(f"  • Mean Score:   {mean_score} / 5.0")
    print(f"Justification:    {justification}")
    print(f"Appended to:      {csv_path}")
    print("=" * 60 + "\n")

    return result


def evaluate_benchmark_runs(jsonl_path: str = "outputs/eval_runs.jsonl") -> List[Dict[str, Any]]:
    """
    Score all benchmark runs recorded in outputs/eval_runs.jsonl.
    Maintains 1:1 question-to-report mapping and skips duplicates or mock runs.
    """
    if not os.path.exists(jsonl_path):
        print(f"No benchmark runs file found at {jsonl_path}")
        return []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\n[JUDGE] Found {len(records)} benchmark run records in {jsonl_path}")
    results = []

    # Map each question to its latest valid run
    latest_by_question: Dict[str, Dict[str, Any]] = {}
    for r in records:
        q = r.get("question")
        if q:
            latest_by_question[q] = r

    for q, r in latest_by_question.items():
        rp = r.get("report_path")
        if not rp or not os.path.exists(rp):
            print(f"[JUDGE] Skipping question '{q}': report not found at {rp}")
            continue

        print(f"\n--- Judging Question: '{q}' ---")
        res = evaluate_report(question=q, report_path=rp)
        if res:
            results.append(res)

    print(f"\n[JUDGE COMPLETE] Evaluated {len(results)} reports successfully.")
    return results


def main():
    if len(sys.argv) >= 3:
        report_path = sys.argv[1]
        question = sys.argv[2]
        evaluate_report(question, report_path)
    elif len(sys.argv) == 2 and sys.argv[1] not in ("--all", "-a"):
        report_path = sys.argv[1]
        evaluate_report("General Research Evaluation", report_path)
    else:
        evaluate_benchmark_runs()


if __name__ == "__main__":
    main()
