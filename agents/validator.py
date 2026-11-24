"""
Research Relevance Validator (shared by Phase 1 and Phase 2)
Architecture Reference: docs/EXPLANATIONS.md - Chapter 13

First Principle:
A generator should never be allowed to write from evidence that has not been checked.
This gate sits between the Synthesizer and the Writer and verifies, for the CURRENT run only:

 1. Isolation      - every research record carries this run's run_id (foreign records are dropped
                     and logged as isolation violations).
 2. Assignment     - each record answers a sub-question that is actually in this run's plan.
 3. Provenance     - cited URLs were retrieved for that very sub-question (the researcher already
                     drops uncited / unretrieved citations; the counts are recorded here).
 4. Semantic fit   - ONE batched, temperature-0 LLM check judges whether each sub-question's
                     findings (and the synthesis brief) are relevant to the original question.
                     It is deliberately tolerant of interdisciplinary evidence: a finding is
                     relevant if it helps answer the sub-question OR the overall question.
                     (No keyword matching is used.)

Failure handling: flagged sub-questions are re-researched ONCE with a context-enriched query.
Still-failing sub-questions are excluded and listed as evidence gaps (run -> PARTIAL). If no
sub-question survives, the run is FAILED and the graph ends before the Writer. Mock content
is never inserted.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.llm import LLMUnavailableError, call_llm, is_live_mode, usage_record
from agents.researcher import research_subquestion
from agents.synthesizer import synthesize_findings
from state import ResearchState, current_run_records

logger = logging.getLogger("ValidatorAgent")

MIN_RELEVANCE = 3  # 1-5 scale; <3 means the sub-question's evidence is judged off-topic


_URL_BRACKET = re.compile(r"\s*\[https?://[^\]]+\]")


def _strip_urls(text: str) -> str:
    return _URL_BRACKET.sub("", text)


def _parse_json(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    return json.loads(m.group(0) if m else t)


def _semantic_check(
    question: str, items: List[Dict[str, Any]], brief: str
) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    One batched LLM relevance judgement.
    Returns (per_item_result, brief_result, usage, status) with status in ok|unavailable|skipped_mock.
    """
    if not is_live_mode():
        return {}, None, None, "skipped_mock"
    if not items:
        return {}, None, None, "ok"

    blocks = []
    for it in items:
        lines = "\n".join("  (%d) %s" % (j, _strip_urls(f)) for j, f in enumerate(it["findings"]))
        blocks.append(f"Sub-question [{it['idx']}]: {it['sub_question']}\nFindings:\n{lines}")
    system_instruction = (
        "You are a research-integrity auditor. Decide whether collected findings are relevant to the ORIGINAL research "
        "question and to the sub-question they were collected for. Be tolerant of interdisciplinary or adjacent evidence "
        "that genuinely helps answer the question; be strict about findings about a DIFFERENT subject entirely.\n"
        "Respond ONLY with JSON:\n"
        '{"subquestions":[{"index":int,"relevance":1-5,"off_topic_findings":[int],"reason":"one sentence"}],'
        '"brief_on_topic":true|false,"brief_reason":"one sentence"}'
    )
    prompt = (
        f"ORIGINAL RESEARCH QUESTION: {question}\n\n" + "\n\n".join(blocks)
        + f"\n\nSYNTHESIS BRIEF (excerpt):\n{brief[:6000]}\n\nAudit now:"
    )
    try:
        content, usage = call_llm(prompt=prompt, system_instruction=system_instruction, temperature=0.0,
                                  allow_mock=False, node="validator")
        data = _parse_json(content)
        per = {int(s["index"]): s for s in data.get("subquestions", []) if "index" in s}
        brief_res = {"on_topic": bool(data.get("brief_on_topic", True)), "reason": str(data.get("brief_reason", ""))}
        return per, brief_res, usage, "ok"
    except (LLMUnavailableError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Semantic relevance check unavailable: %s", exc)
        return {}, None, None, "unavailable"


def validate_research(state: ResearchState) -> Dict[str, Any]:
    run_id = state.get("run_id", "")
    question = state.get("question", "")
    plan: List[str] = list(state.get("plan", []))
    brief = state.get("brief", "")

    valid, foreign = current_run_records(state)
    all_current = [r for r in (state.get("research_records") or []) if not run_id or r.get("run_id") == run_id]
    llm_calls: List[Dict[str, Any]] = []
    new_records: List[Dict[str, Any]] = []
    flagged: Dict[str, str] = {}
    retried: List[str] = []

    # --- 1/2. isolation + assignment + per-subquestion status ---------------------------
    accepted: Dict[str, Dict[str, Any]] = {}
    for sq in plan:
        recs = [r for r in all_current if r.get("sub_question") == sq]
        ok = [r for r in recs if r.get("status") == "ok" and r.get("findings")]
        if ok:
            accepted[sq] = ok[-1]
        elif recs:
            flagged[sq] = f"researcher status '{recs[-1].get('status')}': {recs[-1].get('error', '')}".strip()
        else:
            flagged[sq] = "no researcher output was produced for this sub-question"

    # --- 4. semantic relevance (batched) ---------------------------------------------------
    def run_semantic(candidates: Dict[str, Dict[str, Any]]):
        items = [{"idx": i, "sub_question": sq, "findings": r["findings"]} for i, (sq, r) in enumerate(candidates.items())]
        per, brief_res, usage, status = _semantic_check(question, items, brief)
        if usage:
            llm_calls.append(usage_record("validator", usage))
        return items, per, brief_res, status

    items, per, brief_res, sem_status = run_semantic(accepted)
    brief_flag = ""
    if sem_status == "ok":
        for it in items:
            res = per.get(it["idx"])
            if not res:
                continue
            sq = it["sub_question"]
            off = {int(i) for i in res.get("off_topic_findings", []) if str(i).lstrip("-").isdigit()}
            if off and len(off) < len(it["findings"]):
                accepted[sq] = {**accepted[sq], "findings": [f for j, f in enumerate(it["findings"]) if j not in off]}
            if int(res.get("relevance", 5)) < MIN_RELEVANCE or (off and len(off) >= len(it["findings"])):
                flagged[sq] = f"semantic relevance {res.get('relevance')}/5: {res.get('reason', '')}"
                accepted.pop(sq, None)
        if brief_res and not brief_res["on_topic"]:
            brief_flag = f"brief judged off-topic: {brief_res['reason']}"

    # --- retry flagged sub-questions once ----------------------------------------------------
    for sq in [q for q in plan if q in flagged]:
        retried.append(sq)
        out = research_subquestion(sq, run_id=run_id, question=question, retry=True)
        new_records.extend(out["research_records"])
        llm_calls.extend(out["metadata"].get("llm_calls", []))
        rec = out["research_records"][0]
        if rec["status"] == "ok" and rec["findings"]:
            r_items, r_per, _, r_status = run_semantic({sq: rec})
            res = r_per.get(0)
            if r_status == "ok" and res and int(res.get("relevance", 5)) < MIN_RELEVANCE:
                flagged[sq] = f"after retry, semantic relevance {res.get('relevance')}/5: {res.get('reason', '')}"
            else:
                accepted[sq] = rec
                flagged.pop(sq, None)
        else:
            flagged[sq] = f"after retry: {rec.get('error') or rec['status']}"

    failed_list = [{"sub_question": sq, "reason": flagged[sq]} for sq in plan if sq in flagged]

    # --- outcome -----------------------------------------------------------------------------------
    validated_notes = {sq: list(r["findings"]) for sq, r in accepted.items()}
    validated_results = {sq: list(r.get("sources", [])) for sq, r in accepted.items()}
    accepted_urls = {h["url"] for hits in validated_results.values() for h in hits}

    result_meta: Dict[str, Any] = {"validation_retries": len(retried), "llm_calls": llm_calls, "search_count": len(retried)}
    if foreign:
        result_meta["isolation_violations"] = [
            {"node": "validator", "record_run_id": r.get("run_id"), "sub_question": r.get("sub_question")} for r in foreign
        ]
    if llm_calls:
        result_meta["tokens_validator"] = sum(c["total_tokens"] for c in llm_calls if c["node"] == "validator")

    if not validated_notes:
        validation = {
            "run_id": run_id, "status": "failed", "semantic_check": sem_status, "retried": retried,
            "accepted": [], "flagged": failed_list, "foreign_records_dropped": len(foreign), "brief_check": brief_flag or "n/a",
        }
        result_meta["validation_status"] = "failed"
        print(f"\n[VALIDATOR] FAILED: no sub-question has relevant, validated evidence (run {run_id}).\n")
        return {
            "validated_notes": {}, "validated_search_results": {}, "validation": validation,
            "failed_subquestions": failed_list, "research_records": new_records,
            "execution_status": "FAILED",
            "execution_error": "No sub-question produced relevant, validated evidence.",
            "metadata": result_meta,
        }

    # brief provenance: unknown URLs or removed material -> re-synthesize from validated evidence only
    brief_urls = set(re.findall(r"https?://[^\s\]\)>]+", brief))
    unknown_urls = sorted(u for u in brief_urls if u not in accepted_urls)
    need_resynth = bool(failed_list or retried or brief_flag or unknown_urls or not brief)
    out_brief = brief
    if need_resynth:
        tmp_records = [
            {"run_id": run_id, "sub_question": sq, "status": "ok", "findings": f, "sources": validated_results[sq]}
            for sq, f in validated_notes.items()
        ]
        syn = synthesize_findings({"run_id": run_id, "question": question, "research_records": tmp_records})
        out_brief = syn["brief"]
        llm_calls.extend(syn["metadata"].get("llm_calls", []))
        result_meta["tokens_synthesizer_resynth"] = syn["metadata"].get("tokens_synthesizer", 0)

    status = "passed" if not failed_list and sem_status in ("ok", "skipped_mock") else "partial"
    validation = {
        "run_id": run_id, "status": status, "semantic_check": sem_status, "retried": retried,
        "accepted": list(validated_notes.keys()), "flagged": failed_list,
        "foreign_records_dropped": len(foreign), "resynthesized": need_resynth,
        "brief_check": brief_flag or ("unknown source urls re-synthesized" if unknown_urls else "ok"),
        "method": "run_id+provenance (deterministic) + batched semantic LLM audit" if sem_status == "ok"
                  else ("deterministic only (MOCK mode)" if sem_status == "skipped_mock" else "deterministic only (semantic audit unavailable)"),
    }
    result_meta["validation_status"] = status
    print(f"\n[VALIDATOR] status={status} accepted={len(validated_notes)}/{len(plan)} retried={len(retried)} semantic={sem_status}\n")

    out: Dict[str, Any] = {
        "validated_notes": validated_notes, "validated_search_results": validated_results, "validation": validation,
        "failed_subquestions": failed_list, "research_records": new_records, "brief": out_brief, "metadata": result_meta,
    }
    if status == "partial":
        out["execution_status"] = "PARTIAL"
    return out
