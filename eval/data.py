"""
Evaluation Data Layer (single source of truth for CSV export, compare.py and the Streamlit dashboard)
Architecture Reference: docs/EXPLANATIONS.md - Chapter 13

Responsibilities
* Read BOTH legacy files (read-only, never modified) and schema-v2 files:
    legacy runs : outputs/eval_runs.jsonl            (Phase 1)  outputs/eval_runs_phase2.jsonl   (Phase 2)
    legacy judge: outputs/eval_results.csv           (Phase 1)  outputs/eval_results_phase2.csv  (Phase 2)
    v2 runs     : outputs/runs/eval_runs_v2.jsonl    (append-only, both phases)
    v2 judge    : outputs/runs/judge_evaluations.jsonl (append-only, every Judge attempt incl. failures)
* Normalise everything onto stable benchmark IDs (Q01..Q10).
* Validate every record. Mock / failed / partial / inconsistent records are KEPT and shown, but are
  excluded from scientific comparisons. Missing values stay None (rendered "N/A") - never 0.
* Compute paired comparisons and aggregates only over VALID pairs.
"""

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from eval.questions import BENCHMARK_QUESTIONS, benchmark_id_for, normalize_question
from tools.search_tool import SIMULATED_SOURCE_DOMAINS

DIMS = ("relevance", "completeness", "organization", "grounding")
LEGACY_MOCK_SENTENCE = "Analysis of current empirical data reveals substantial differentiation across primary metrics"
MOCK_MARKER = "MOCK EXECUTION"
LEGACY_JUDGE_MODEL = "unrecorded (legacy)"
RUBRIC_VERSION = "rubric-v1-4dim-1to5"


@dataclass
class DataPaths:
    base: str = "outputs"

    @property
    def legacy_runs(self) -> Dict[int, str]:
        return {1: os.path.join(self.base, "eval_runs.jsonl"), 2: os.path.join(self.base, "eval_runs_phase2.jsonl")}

    @property
    def legacy_judge(self) -> Dict[int, str]:
        return {1: os.path.join(self.base, "eval_results.csv"), 2: os.path.join(self.base, "eval_results_phase2.csv")}

    @property
    def v2_runs(self) -> str:
        return os.path.join(self.base, "runs", "eval_runs_v2.jsonl")

    @property
    def v2_judge(self) -> str:
        return os.path.join(self.base, "runs", "judge_evaluations.jsonl")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.base, "runs")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def norm_path(p: Optional[str]) -> str:
    return (p or "").replace("\\", "/").lstrip("./").lower()


def _ts(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).astimezone().timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def is_mock_text(text: str) -> bool:
    return LEGACY_MOCK_SENTENCE in text or MOCK_MARKER in text


def has_simulated_sources(text: str) -> bool:
    return any(d in text for d in SIMULATED_SOURCE_DOMAINS)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec["_line"] = i
                out.append(rec)
            except json.JSONDecodeError:
                out.append({"_line": i, "_corrupt": True})
    return out


# --------------------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------------------
def _normalize_legacy_run(rec: Dict[str, Any], phase: int) -> Dict[str, Any]:
    question = rec.get("question", "")
    rp = rec.get("report_path", "")
    text = read_text(rp) if rp else None
    flags: List[str] = []
    mode = "LIVE"
    if text is None:
        flags.append("report_missing")
    else:
        if is_mock_text(text):
            mode = "MOCK"
        if has_simulated_sources(text) and mode != "MOCK":
            flags.append("simulated_sources_in_live_report")
    node_tokens = rec.get("node_tokens") or {}
    if node_tokens and abs(sum(node_tokens.values()) - int(rec.get("tokens", 0))) > max(5, 0.01 * int(rec.get("tokens", 1))):
        flags.append("token_total_inconsistent_with_node_tokens")
    if phase == 2 and len(rec.get("metadata", {}) or {}) <= 3 and node_tokens:
        flags.append("metadata_incomplete_not_instrument_output")
    status = "COMPLETE" if not flags or mode == "MOCK" else "PARTIAL"
    display = "MOCK" if mode == "MOCK" else ("PARTIAL" if flags else "LIVE")
    sh = rec.get("score_history") or []
    return {
        "source": "legacy", "schema_version": 1,
        "run_id": f"legacy-p{phase}-line{rec.get('_line')}",
        "benchmark_id": benchmark_id_for(question), "question": question, "phase": phase,
        "execution_mode": mode, "execution_status": status, "display_status": display,
        "model": "unrecorded (legacy)", "report_path": rp,
        "latency": rec.get("latency"), "input_tokens": None, "output_tokens": None,
        "total_tokens": rec.get("tokens"), "token_basis": "legacy-unverified",
        "searches": rec.get("searches"), "node_latencies": rec.get("node_latencies", {}), "node_tokens": node_tokens,
        "critic_iterations": len(sh) if phase == 2 else None,
        "revisions": rec.get("iterations_used") if phase == 2 else None,
        "iterations_used": rec.get("iterations_used"),
        "score_history": sh, "final_verdict": (rec.get("metadata", {}) or {}).get("final_verdict"),
        "cap_reached": None, "timestamp": rec.get("timestamp"), "error": "", "integrity_flags": flags,
        "validation": {}, "failed_subquestions": [],
    }


def _normalize_v2_run(rec: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(rec)
    r["source"] = "v2"
    r["benchmark_id"] = r.get("benchmark_id") or benchmark_id_for(r.get("question", ""))
    flags = list(r.get("integrity_flags", []))
    rp = r.get("report_path", "")
    text = read_text(rp) if rp else None
    if text is None and r.get("execution_status") != "FAILED":
        flags.append("report_missing")
    elif text is not None:
        if is_mock_text(text) and r.get("execution_mode") != "MOCK":
            flags.append("mock_text_in_non_mock_run")
        if has_simulated_sources(text) and r.get("execution_mode") == "LIVE":
            flags.append("simulated_sources_in_live_report")
    if r.get("isolation_violations"):
        flags.append("isolation_violation")
    r["integrity_flags"] = flags
    return r


def run_is_valid(r: Dict[str, Any], allow_flagged: bool = False) -> bool:
    """A run may enter a scientific comparison only if it is a clean, complete LIVE run."""
    if r.get("execution_mode") != "LIVE" or r.get("execution_status") != "COMPLETE":
        return False
    if not r.get("benchmark_id"):
        return False
    return allow_flagged or not r.get("integrity_flags")


def load_runs(paths: Optional[DataPaths] = None) -> List[Dict[str, Any]]:
    paths = paths or DataPaths()
    runs: List[Dict[str, Any]] = []
    for phase, p in paths.legacy_runs.items():
        for rec in _read_jsonl(p):
            if rec.get("_corrupt"):
                continue
            runs.append(_normalize_legacy_run(rec, phase))
    for rec in _read_jsonl(paths.v2_runs):
        if rec.get("_corrupt"):
            continue
        runs.append(_normalize_v2_run(rec))
    for r in runs:
        r["run_key"] = r.get("run_id")
        r["valid"] = run_is_valid(r)
    return runs


def select_latest_runs(runs: List[Dict[str, Any]], allow_flagged: bool = False) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Per (benchmark_id, phase): the latest VALID run; if none is valid, the latest run (for display)."""
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in sorted(runs, key=lambda x: _ts(x.get("timestamp"))):
        bid = r.get("benchmark_id")
        if not bid:
            continue
        key = (bid, r["phase"])
        cur = best.get(key)
        valid = run_is_valid(r, allow_flagged)
        if cur is None or valid or not run_is_valid(cur, allow_flagged):
            if cur is None or valid or not run_is_valid(cur, allow_flagged):
                best[key] = r
    return best


# --------------------------------------------------------------------------------------
# judge evaluations
# --------------------------------------------------------------------------------------
def _scores_ok(d: Dict[str, Any]) -> bool:
    return all(isinstance(d.get(k), int) and not isinstance(d.get(k), bool) and 1 <= d[k] <= 5 for k in DIMS)


def mean_score(d: Dict[str, Any]) -> Optional[float]:
    return round(sum(d[k] for k in DIMS) / 4.0, 4) if _scores_ok(d) else None


def load_judge_evaluations(paths: Optional[DataPaths] = None, runs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    paths = paths or DataPaths()
    runs = runs if runs is not None else load_runs(paths)
    by_path: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in runs:
        by_path[(r["phase"], norm_path(r.get("report_path")))] = r

    evals: List[Dict[str, Any]] = []
    for phase, p in paths.legacy_judge.items():
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                q, rp = row.get("Question", ""), row.get("Report_Path", "")
                run = by_path.get((phase, norm_path(rp)))
                text = read_text(rp) if rp else None
                reason = ""
                if run is None:
                    reason = "no_matching_run_record"
                elif benchmark_id_for(q) != run.get("benchmark_id"):
                    reason = "question_report_mismatch"
                elif text is None:
                    reason = "report_missing"
                elif is_mock_text(text):
                    reason = "mock_report"
                try:
                    sc = {"relevance": int(row["Relevance"]), "completeness": int(row["Completeness"]),
                          "organization": int(row["Organization"]), "grounding": int(row["Grounding"])}
                except (KeyError, ValueError, TypeError):
                    sc, reason = {}, reason or "unparsable_scores"
                ev = {
                    "evaluation_id": f"legacy-p{phase}-row{i}", "source": "legacy",
                    "benchmark_id": run.get("benchmark_id") if run else benchmark_id_for(q), "phase": phase,
                    "run_id": run.get("run_id") if run else None, "report_path": rp, "question": q,
                    "judge_model": LEGACY_JUDGE_MODEL, "rubric_version": "legacy", "timestamp": None,
                    "status": "OK" if not reason else f"REJECTED:{reason}",
                    "justification": row.get("Justification", ""), "justifications": {}, **sc,
                }
                ev["mean_score"] = mean_score(ev)
                evals.append(ev)
    for rec in _read_jsonl(paths.v2_judge):
        if rec.get("_corrupt"):
            continue
        rec["source"] = "v2"
        rec["mean_score"] = mean_score(rec) if rec.get("status") == "OK" else None
        evals.append(rec)
    for e in evals:
        e["valid"] = e.get("status") == "OK" and _scores_ok(e)
    return evals


def latest_valid_eval(evals: List[Dict[str, Any]], run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Newest OK evaluation of THIS run (matched by run_id, else report path). History is never deleted."""
    cands = [
        e for e in evals
        if e.get("valid") and e.get("phase") == run["phase"]
        and (e.get("run_id") == run.get("run_id") or (norm_path(e.get("report_path")) == norm_path(run.get("report_path")) and run.get("report_path")))
    ]
    return max(cands, key=lambda e: _ts(e.get("timestamp")), default=None) if cands else None


def eval_status_for(evals: List[Dict[str, Any]], run: Dict[str, Any]) -> str:
    """Judge status shown in the dashboard: EVALUATED / PENDING / NOT ELIGIBLE / FAILED:<reason>."""
    if latest_valid_eval(evals, run):
        return "EVALUATED"
    mine = [e for e in evals if e.get("phase") == run["phase"] and (e.get("run_id") == run.get("run_id") or norm_path(e.get("report_path")) == norm_path(run.get("report_path")))]
    if run.get("execution_mode") == "MOCK":
        return "NOT ELIGIBLE (mock)"
    if run.get("execution_status") == "FAILED":
        return "NOT ELIGIBLE (failed run)"
    if not run_is_valid(run):
        return "NOT ELIGIBLE (partial/unverified)"
    if mine:
        return "FAILED: " + (mine[-1].get("status") or "unknown")
    return "PENDING"


# --------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------
def _pair_note(r1, r2, e1, e2, allow_flagged: bool) -> str:
    def why(tag, r, e):
        if r is None:
            return f"{tag} run missing"
        if not run_is_valid(r, allow_flagged):
            if r.get("execution_mode") == "MOCK":
                return f"{tag} run is MOCK"
            if r.get("execution_status") == "FAILED":
                return f"{tag} run FAILED"
            return f"{tag} run not clean ({', '.join(r.get('integrity_flags', [])) or r.get('execution_status')})"
        if e is None:
            return f"{tag} judge evaluation missing"
        return ""
    notes = [n for n in (why("P1", r1, e1), why("P2", r2, e2)) if n]
    if not notes and e1 and e2 and (e1.get("judge_model") != e2.get("judge_model") or e1.get("rubric_version") != e2.get("rubric_version")):
        notes.append("judge model/rubric differ between phases")
    return "; ".join(notes)


def _avg(vals: List[float]) -> Optional[float]:
    return round(sum(vals) / len(vals), 4) if vals else None


def build_comparison(paths: Optional[DataPaths] = None, allow_flagged: bool = False) -> Dict[str, Any]:
    paths = paths or DataPaths()
    runs = load_runs(paths)
    evals = load_judge_evaluations(paths, runs)
    latest = select_latest_runs(runs, allow_flagged)

    rows: List[Dict[str, Any]] = []
    for b in BENCHMARK_QUESTIONS:
        bid = b["id"]
        r1, r2 = latest.get((bid, 1)), latest.get((bid, 2))
        e1 = latest_valid_eval(evals, r1) if r1 and run_is_valid(r1, allow_flagged) else None
        e2 = latest_valid_eval(evals, r2) if r2 and run_is_valid(r2, allow_flagged) else None
        note = _pair_note(r1, r2, e1, e2, allow_flagged)
        paired = not note
        row: Dict[str, Any] = {"question_id": bid, "question": b["question"], "pair_valid": paired, "pair_note": note or "valid pair"}
        for tag, r, e in (("p1", r1, e1), ("p2", r2, e2)):
            for d in DIMS:
                row[f"{tag}_{d}"] = e.get(d) if e else None
            row[f"{tag}_mean"] = e.get("mean_score") if e else None
            row[f"{tag}_latency"] = r.get("latency") if r else None
            row[f"{tag}_tokens"] = r.get("total_tokens") if r else None
            row[f"{tag}_token_basis"] = r.get("token_basis") if r else None
            row[f"{tag}_run_id"] = r.get("run_id") if r else None
            row[f"{tag}_status"] = r.get("display_status") if r else None
        row["p2_iterations"] = (r2.get("critic_iterations") if r2 else None)
        row["mean_delta"] = round(row["p2_mean"] - row["p1_mean"], 4) if paired and row["p1_mean"] is not None and row["p2_mean"] is not None else None
        rows.append(row)

    pairs = [r for r in rows if r["pair_valid"]]
    n = len(pairs)
    tok_pairs = [r for r in pairs if r["p1_token_basis"] == "actual" and r["p2_token_basis"] == "actual"]
    deltas = [r["mean_delta"] for r in pairs]
    agg: Dict[str, Any] = {
        "valid_pairs": n, "valid_pairs_for_tokens": len(tok_pairs),
        "avg_p1_quality": _avg([r["p1_mean"] for r in pairs]), "avg_p2_quality": _avg([r["p2_mean"] for r in pairs]),
        "avg_improvement": _avg(deltas),
        "avg_p1_latency": _avg([r["p1_latency"] for r in pairs]), "avg_p2_latency": _avg([r["p2_latency"] for r in pairs]),
        "avg_p1_tokens": _avg([r["p1_tokens"] for r in tok_pairs]), "avg_p2_tokens": _avg([r["p2_tokens"] for r in tok_pairs]),
        "pct_improved": round(100 * sum(1 for d in deltas if d > 1e-9) / n, 1) if n else None,
        "pct_deteriorated": round(100 * sum(1 for d in deltas if d < -1e-9) / n, 1) if n else None,
        "pct_unchanged": round(100 * sum(1 for d in deltas if abs(d) <= 1e-9) / n, 1) if n else None,
        "per_dimension": {d: {"p1": _avg([r[f"p1_{d}"] for r in pairs]), "p2": _avg([r[f"p2_{d}"] for r in pairs])} for d in DIMS},
    }
    agg["latency_overhead"] = round(agg["avg_p2_latency"] - agg["avg_p1_latency"], 2) if n else None
    agg["latency_overhead_pct"] = round(100 * (agg["avg_p2_latency"] / agg["avg_p1_latency"] - 1), 1) if n and agg["avg_p1_latency"] else None
    agg["token_overhead"] = round(agg["avg_p2_tokens"] - agg["avg_p1_tokens"], 1) if tok_pairs else None
    agg["token_overhead_pct"] = round(100 * (agg["avg_p2_tokens"] / agg["avg_p1_tokens"] - 1), 1) if tok_pairs and agg["avg_p1_tokens"] else None

    counts: Dict[str, Any] = {"total_questions": len(BENCHMARK_QUESTIONS)}
    for ph in (1, 2):
        pr = [r for r in runs if r["phase"] == ph]
        valid_runs = [r for r in pr if run_is_valid(r, allow_flagged)]
        evaluated = [r for r in valid_runs if latest_valid_eval(evals, r)]
        counts[f"p{ph}_records"] = len(pr)
        counts[f"p{ph}_questions_covered"] = len({r["benchmark_id"] for r in pr if r.get("benchmark_id")})
        counts[f"p{ph}_evaluated"] = len(evaluated)
        counts[f"p{ph}_pending"] = len(valid_runs) - len(evaluated)
        counts[f"p{ph}_failed"] = sum(1 for r in pr if r.get("execution_status") == "FAILED")
        counts[f"p{ph}_mock"] = sum(1 for r in pr if r.get("execution_mode") == "MOCK")
        counts[f"p{ph}_partial"] = sum(1 for r in pr if r.get("display_status") == "PARTIAL")
        counts[f"p{ph}_avg_quality"] = _avg([latest_valid_eval(evals, r)["mean_score"] for r in evaluated])
    counts["judge_records"] = len(evals)
    counts["judge_rejected"] = sum(1 for e in evals if str(e.get("status", "")).startswith("REJECTED"))
    return {"rows": rows, "aggregates": agg, "counts": counts, "runs": runs, "evals": evals}


CSV_COLUMNS = [
    "question_id", "question",
    "p1_relevance", "p2_relevance", "p1_completeness", "p2_completeness", "p1_organization", "p2_organization",
    "p1_grounding", "p2_grounding", "p1_mean", "p2_mean", "mean_delta",
    "p1_latency", "p2_latency", "p1_tokens", "p2_tokens", "p1_token_basis", "p2_token_basis", "p2_iterations", "pair_valid", "pair_note",
]


def comparison_csv_text(rows: List[Dict[str, Any]]) -> str:
    """CSV with 'N/A' for every missing value (never 0)."""
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in rows:
        w.writerow(["N/A" if r.get(c) is None else r.get(c) for c in CSV_COLUMNS])
    return buf.getvalue()


def load_run_artifact(run_id: str, paths: Optional[DataPaths] = None) -> Optional[Dict[str, Any]]:
    paths = paths or DataPaths()
    p = os.path.join(paths.runs_dir, f"{run_id}.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
