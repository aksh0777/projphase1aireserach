"""
Multi-Agent Deep Research System — Streamlit Academic Interface
Architecture Reference: docs/EXPLANATIONS.md

Sections:
1. 🔬 Research Workspace: Interactive live multi-agent execution (Phase 1 & Phase 2)
2. 📊 Evaluation Dashboard: Transparent experimental overview, status tracking, drill-down
3. ⚖️ Phase 1 vs Phase 2 Comparison: Empirical paired comparison, deltas, overhead, charts
"""

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is at the head of sys.path across all platforms & Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from agents.llm import get_model_name, is_live_mode, is_placeholder_key
from eval.data import (
    DataPaths,
    build_comparison,
    comparison_csv_text,
    eval_status_for,
    latest_valid_eval,
    load_judge_evaluations,
    load_run_artifact,
    load_runs,
    mean_score,
    read_text,
    run_is_valid,
)
from eval.questions import BENCHMARK_QUESTIONS
from graph.run_record import execute_run

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Deep Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic polish
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-live { background-color: #065F46; color: #D1FAE5; }
    .badge-mock { background-color: #92400E; color: #FEF3C7; }
    .badge-partial { background-color: #9A3412; color: #FFEDD5; }
    .badge-failed { background-color: #991B1B; color: #FEE2E2; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_run_record" not in st.session_state:
    st.session_state.current_run_record = None
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "topic_input_val" not in st.session_state:
    st.session_state.topic_input_val = "Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026"
if "use_critic_toggle" not in st.session_state:
    st.session_state.use_critic_toggle = True

# -----------------------------------------------------------------------------
# API Key Resolution
# -----------------------------------------------------------------------------
api_key = None
raw_key = None
placeholder_detected = False

try:
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        raw_key = str(st.secrets["GEMINI_API_KEY"]).strip()
except Exception:
    pass

if not raw_key:
    raw_key = os.getenv("GEMINI_API_KEY", "").strip()

if raw_key:
    if is_placeholder_key(raw_key):
        placeholder_detected = True
        api_key = None
    else:
        api_key = raw_key
        os.environ["GEMINI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Research Platform")
    st.caption("AI Multi-Agent Deep Research with Dialectical Peer Review")

    nav_section = st.radio(
        "Navigation",
        [
            "🔬 Research Workspace",
            "📊 Evaluation Dashboard",
            "⚖️ Phase 1 vs Phase 2 Comparison",
        ],
        index=0,
    )

    st.divider()
    st.subheader("⚙️ Credentials & Mode")

    if placeholder_detected:
        st.warning("⚠️ Placeholder key detected in environment.")

    if not api_key:
        user_key_input = st.text_input(
            "Gemini API Key:",
            type="password",
            placeholder="AIzaSy...",
            help="Add your Gemini API key for live research runs.",
        )
        if user_key_input.strip():
            if is_placeholder_key(user_key_input.strip()):
                st.error("⚠️ Entered key is a placeholder string.")
            else:
                api_key = user_key_input.strip()
                os.environ["GEMINI_API_KEY"] = api_key
                st.success("✅ API Key active for session")
                st.rerun()
    else:
        st.success("✅ Gemini API Key connected")

    is_live = bool(api_key and not is_placeholder_key(api_key))
    active_model = get_model_name() if is_live else "mock-generator"

    st.caption(f"**Execution Mode:** {'🟢 LIVE' if is_live else '🟠 MOCK'}")
    st.caption(f"**Active Model:** `{active_model}`")

    st.divider()
    st.caption("Academic Reference: `docs/EXPLANATIONS.md`")


# =============================================================================
# PAGE 1: RESEARCH WORKSPACE
# =============================================================================
if nav_section == "🔬 Research Workspace":
    st.header("🔬 Research Workspace")
    st.markdown(
        "Execute end-to-end multi-agent deep research with automated decomposition, "
        "grounded parallel retrieval, contradiction synthesis, and dialectical peer-review revision."
    )

    # Configuration Row
    col_mode, col_info = st.columns([2, 3])
    with col_mode:
        arch_choice = st.radio(
            "Architecture Pipeline:",
            [
                "Phase 1: Baseline DAG (Planner → Researchers → Synthesizer → Writer)",
                "Phase 2: Closed-Loop (Baseline + Critic ⇄ Reviser Loop)",
            ],
            index=1 if st.session_state.use_critic_toggle else 0,
            disabled=st.session_state.is_running,
        )
        with_critic = "Phase 2" in arch_choice
        st.session_state.use_critic_toggle = with_critic

    with col_info:
        if is_live:
            st.info(
                f"🟢 **LIVE EXECUTION MODE**\n\n"
                f"Agents will query live web search endpoints and invoke `{active_model}`. "
                f"Runs are isolated with unique IDs and verifiable provenance."
            )
        else:
            st.warning(
                "🟠 **MOCK EXECUTION MODE**\n\n"
                "No active API key detected. Running in deterministic mock generator mode "
                "for UI and workflow verification. No real search or LLM tokens will be consumed."
            )

    st.divider()

    # Topic Input
    c_input, c_btn = st.columns([5, 1])
    with c_input:
        topic_text = st.text_input(
            "Research Question / Topic:",
            value=st.session_state.topic_input_val,
            disabled=st.session_state.is_running,
            placeholder="Enter research topic...",
        )
    with c_btn:
        st.write("")
        st.write("")
        start_btn = st.button(
            "🚀 Execute Run",
            type="primary",
            disabled=st.session_state.is_running or not topic_text.strip(),
            use_container_width=True,
        )

    # Execution Flow
    if start_btn:
        st.session_state.is_running = True
        st.session_state.topic_input_val = topic_text.strip()
        st.session_state.current_report = None
        st.session_state.current_run_record = None

        t0 = time.perf_counter()

        st.subheader("⚡ Live Workflow Execution")
        planner_status = st.status("🧠 [1] Planner: Decomposing research question...", expanded=True)
        researcher_status = st.status("🔍 [2] Researchers: Awaiting plan...", expanded=False)
        synthesizer_status = st.status("📑 [3] Synthesizer: Awaiting evidence...", expanded=False)
        validator_status = st.status("🛡️ [4] Validator: Awaiting synthesis...", expanded=False)
        writer_status = st.status("✍️ [5] Writer: Awaiting validated brief...", expanded=False)
        critic_status = st.status("⚖️ [6] Critic & Revision Loop: Awaiting initial draft...", expanded=False) if with_critic else None

        accumulated_notes = {}
        accumulated_meta = {}

        def handle_event(event: Dict[str, Any]):
            # Planner
            if "planner" in event:
                plan = event["planner"].get("plan", [])
                accumulated_meta.update(event["planner"].get("metadata", {}))
                with planner_status:
                    st.write("**Decomposed Research Sub-Questions:**")
                    for i, q in enumerate(plan, 1):
                        st.markdown(f"**{i}.** {q}")
                planner_status.update(
                    label=f"✅ [1] Planner: Decomposed into {len(plan)} sub-questions",
                    state="complete",
                    expanded=False,
                )
                researcher_status.update(
                    label=f"🔍 [2] Researchers: Executing {len(plan)} parallel streams...",
                    state="running",
                    expanded=True,
                )

            # Researcher
            if "researcher" in event:
                r_notes = event["researcher"].get("notes", {})
                r_results = event["researcher"].get("search_results", {})
                accumulated_meta.update(event["researcher"].get("metadata", {}))
                for sq, findings in r_notes.items():
                    accumulated_notes[sq] = findings
                    with researcher_status:
                        with st.expander(f"📌 {sq}", expanded=True):
                            st.caption(f"Sources retrieved: {len(r_results.get(sq, []))}")
                            for note in findings:
                                st.markdown(f"- {note}")
                researcher_status.update(
                    label=f"🔍 [2] Researchers: {len(accumulated_notes)} streams completed",
                    state="running",
                )

            # Synthesizer
            if "synthesizer" in event:
                researcher_status.update(
                    label=f"✅ [2] Researchers: All parallel retrieval streams complete",
                    state="complete",
                    expanded=False,
                )
                brief = event["synthesizer"].get("brief", "")
                accumulated_meta.update(event["synthesizer"].get("metadata", {}))
                with synthesizer_status:
                    st.markdown(brief)
                synthesizer_status.update(
                    label="✅ [3] Synthesizer: Consolidated research brief compiled",
                    state="complete",
                    expanded=False,
                )
                validator_status.update(
                    label="🛡️ [4] Validator: Auditing evidence relevance and source provenance...",
                    state="running",
                    expanded=True,
                )

            # Validator
            if "validator" in event:
                val = event["validator"].get("validation", {})
                val_status = val.get("status", "passed")
                accumulated_meta.update(event["validator"].get("metadata", {}))
                with validator_status:
                    st.markdown(f"**Validation Status:** `{val_status.upper()}`")
                    st.markdown(f"**Audit Method:** {val.get('method', 'deterministic')}")
                    st.markdown(f"**Accepted Sub-Questions:** {len(val.get('accepted', []))}")
                    if val.get("flagged"):
                        st.warning(f"Flagged issues: {val.get('flagged')}")
                    if val.get("foreign_records_dropped", 0) > 0:
                        st.error(f"Dropped {val['foreign_records_dropped']} foreign isolation records!")
                validator_status.update(
                    label=f"✅ [4] Validator: Research relevance & isolation verified ({val_status})",
                    state="complete",
                    expanded=False,
                )
                writer_status.update(
                    label="✍️ [5] Writer: Drafting comprehensive cited academic report...",
                    state="running",
                    expanded=True,
                )

            # Writer
            if "writer" in event:
                draft_md = event["writer"].get("report", "")
                accumulated_meta.update(event["writer"].get("metadata", {}))
                writer_status.update(
                    label="✅ [5] Writer: Initial report drafted with inline numbered citations",
                    state="complete",
                    expanded=False,
                )
                if with_critic and critic_status:
                    critic_status.update(
                        label="⚖️ [6] Critic: Initiating adversarial peer review...",
                        state="running",
                        expanded=True,
                    )

            # Critic
            if "critic" in event and with_critic and critic_status:
                c_data = event["critic"].get("critique", {})
                iter_num = c_data.get("iteration", 1)
                scores = c_data.get("scores", {})
                verdict = str(c_data.get("verdict", "revise")).upper()
                flaws = c_data.get("flaws", [])
                accumulated_meta.update(event["critic"].get("metadata", {}))

                with critic_status:
                    with st.expander(f"🔎 Peer Review Iteration {iter_num} [{verdict}]", expanded=True):
                        st.markdown(f"**Verdict:** `{verdict}` | **Draft Version:** {c_data.get('draft_version', 0)}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Relevance", f"{scores.get('relevance', 'N/A')}/5")
                        c2.metric("Completeness", f"{scores.get('completeness', 'N/A')}/5")
                        c3.metric("Organization", f"{scores.get('organization', 'N/A')}/5")
                        c4.metric("Grounding", f"{scores.get('grounding', 'N/A')}/5")

                        justs = c_data.get("justifications", {})
                        if justs:
                            st.caption("Justifications: " + " · ".join(f"**{k[:4]}**: {v}" for k, v in justs.items()))

                        if flaws:
                            st.write("**Actionable Flaws to Resolve:**")
                            for f in flaws:
                                st.markdown(f"- ⚠️ {f}")
                        else:
                            st.success("✅ Flaw checklist cleared (All criteria passed threshold).")

                critic_status.update(
                    label=f"⚖️ [6] Critic: Iteration {iter_num} evaluated ({verdict})",
                    state="running",
                    expanded=True,
                )

            # Reviser
            if "reviser" in event and with_critic and critic_status:
                rev_num = event["reviser"].get("revision_count", 1)
                accumulated_meta.update(event["reviser"].get("metadata", {}))
                with critic_status:
                    st.info(f"✍️ **Reviser Iteration {rev_num}:** Generated updated draft. Re-submitting to Critic...")

            # Finalize
            if "finalize" in event and with_critic and critic_status:
                accumulated_meta.update(event["finalize"].get("metadata", {}))
                critic_status.update(
                    label=f"✅ [6] Critic Loop: Finalized after {accumulated_meta.get('iterations_used', 1)} iteration(s)",
                    state="complete",
                    expanded=False,
                )

        try:
            record, final_state = execute_run(
                question=st.session_state.topic_input_val,
                with_critic=with_critic,
                api_key=api_key,
                on_event=handle_event,
                save=True,
            )
            st.session_state.current_run_record = record
            st.session_state.current_report = final_state.get("report", "")
            st.session_state.current_state = final_state
        except Exception as exc:
            st.error(f"Execution failed: {exc}")
        finally:
            st.session_state.is_running = False
            st.rerun()

    # -------------------------------------------------------------------------
    # Display Results of Completed Run
    # -------------------------------------------------------------------------
    if st.session_state.current_run_record:
        rec = st.session_state.current_run_record
        st.divider()

        # Telemetry Summary Header
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        status_color = {
            "LIVE": "badge-live",
            "MOCK": "badge-mock",
            "PARTIAL": "badge-partial",
            "FAILED": "badge-failed",
        }.get(rec["display_status"], "badge-mock")

        with t_col1:
            st.markdown(f"**Status:** <span class='status-badge {status_color}'>{rec['display_status']}</span>", unsafe_allow_html=True)
            st.caption(f"Run ID: `{rec['run_id']}`")
        with t_col2:
            st.metric("Total Latency", f"{rec['latency']:.2f} s")
            st.caption(f"Model: `{rec['model']}`")
        with t_col3:
            tok_txt = f"{rec['total_tokens']:,}" if rec['total_tokens'] else "N/A"
            st.metric("Total Tokens", tok_txt)
            st.caption(f"Basis: `{rec['token_basis']}` (in: {rec['input_tokens']:,}, out: {rec['output_tokens']:,})")
        with t_col4:
            if rec["phase"] == 2:
                st.metric("Critic Iterations", f"{rec.get('critic_iterations', 0)}")
                st.caption(f"Verdict: `{rec.get('final_verdict', 'N/A')}` | Cap: `{rec.get('cap_reached', False)}`")
            else:
                st.metric("Searches Executed", f"{rec.get('searches', 0)}")
                st.caption(f"Search provenance: `{rec.get('search_provenance', 'live')}`")

        # Transparent failure / error diagnosis
        if rec.get("error"):
            st.error(f"⚠️ **Execution Diagnostic:** {rec['error']}")

        # Phase 2 Draft History & Version Comparison
        if rec["phase"] == 2 and rec.get("final_draft_version") is not None:
            with st.expander("🔎 Draft Version History & Selection Policy", expanded=False):
                st.markdown(f"**Selection Policy:** `{rec.get('selection_policy')}`")
                st.markdown(
                    f"- Latest Draft Version: `v{rec.get('latest_draft_version')}`  \n"
                    f"- Best Critic Mean Version: `v{rec.get('best_draft_version')}`  \n"
                    f"- Final Published Version: `v{rec.get('final_draft_version')}`"
                )
                sh = rec.get("score_history", [])
                if sh:
                    score_rows = []
                    for c in sh:
                        sc = c.get("scores", {})
                        score_rows.append({
                            "Iteration": c.get("iteration"),
                            "Draft Version": c.get("draft_version"),
                            "Verdict": c.get("verdict"),
                            "Relevance": sc.get("relevance"),
                            "Completeness": sc.get("completeness"),
                            "Organization": sc.get("organization"),
                            "Grounding": sc.get("grounding"),
                            "Mean": c.get("mean"),
                        })
                    st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

        # Report & Download Button
        st.subheader("📄 Published Research Report")
        col_down, col_path = st.columns([2, 4])
        with col_down:
            slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", rec["question"].lower())[:30]
            st.download_button(
                "💾 Download Report (.md)",
                data=st.session_state.current_report or "",
                file_name=f"report_p{rec['phase']}_{slug}_{rec['run_id']}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_path:
            if rec.get("report_path"):
                st.caption(f"Local disk artifact: `{rec['report_path']}`")

        st.markdown(st.session_state.current_report or "_No report generated._")


# =============================================================================
# PAGE 2: EVALUATION DASHBOARD
# =============================================================================
elif nav_section == "📊 Evaluation Dashboard":
    st.header("📊 Experimental Evaluation Dashboard")
    st.markdown(
        "Standardized academic evaluation tracking across the 10 benchmark questions. "
        "Records execution status, independent external LLM-as-a-Judge quality scores, "
        "per-node latency, and actual token consumption."
    )

    paths = DataPaths()
    all_runs = load_runs(paths)
    all_evals = load_judge_evaluations(paths, all_runs)
    comp = build_comparison(paths=paths)
    counts = comp["counts"]

    # Top-Level Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Benchmark Suite", f"{counts['total_questions']} Questions")
        st.caption("IDs: `Q01` through `Q10`")
    with m2:
        st.metric("Phase 1 Executions", f"{counts['p1_records']} runs")
        st.caption(f"{counts['p1_evaluated']} evaluated · {counts['p1_mock']} mock")
    with m3:
        st.metric("Phase 2 Executions", f"{counts['p2_records']} runs")
        st.caption(f"{counts['p2_evaluated']} evaluated · {counts['p2_mock']} mock")
    with m4:
        p1_q = counts.get('p1_avg_quality')
        st.metric("P1 Baseline Avg", f"{p1_q:.2f}/5.0" if p1_q is not None else "N/A")
        st.caption("Independent Judge Mean")
    with m5:
        p2_q = counts.get('p2_avg_quality')
        st.metric("P2 Critic Avg", f"{p2_q:.2f}/5.0" if p2_q is not None else "N/A")
        st.caption("Independent Judge Mean")

    st.divider()

    # Detailed Execution Table
    st.subheader("📋 Benchmark Execution & Audit Records")
    phase_filter = st.radio("Filter by Phase:", ["All", "Phase 1 Only", "Phase 2 Only"], horizontal=True)

    table_data = []
    for r in all_runs:
        if phase_filter == "Phase 1 Only" and r["phase"] != 1:
            continue
        if phase_filter == "Phase 2 Only" and r["phase"] != 2:
            continue

        bid = r.get("benchmark_id") or "N/A"
        j_eval = latest_valid_eval(all_evals, r)
        j_status = eval_status_for(all_evals, r)
        mean_s = f"{j_eval['mean_score']:.2f}" if j_eval else "N/A"
        lat = f"{r.get('latency'):.1f}s" if r.get('latency') is not None else "N/A"
        tok = f"{r.get('total_tokens'):,}" if r.get('total_tokens') is not None else "N/A"
        iters = str(r.get("critic_iterations") or r.get("iterations_used") or 0) if r["phase"] == 2 else "-"

        table_data.append({
            "Q-ID": bid,
            "Question": r.get("question", "")[:45] + "...",
            "Phase": f"P{r['phase']}",
            "Execution": r.get("display_status", "N/A"),
            "Judge Status": j_status,
            "Mean Score": mean_s,
            "Latency": lat,
            "Tokens": tok,
            "Iterations": iters,
            "Run ID": r.get("run_id", "N/A"),
            "Report Path": r.get("report_path", "N/A"),
        })

    if table_data:
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, hide_index=True, use_container_width=True)
    else:
        st.info("No benchmark run records found.")

    st.divider()

    # Question Drill-Down Inspection
    st.subheader("🔍 Individual Benchmark Question Inspection")
    q_options = [f"{b['id']}: {b['question']}" for b in BENCHMARK_QUESTIONS]
    selected_q_str = st.selectbox("Select Benchmark Question to Inspect:", q_options)
    selected_bid = selected_q_str.split(":")[0]

    matched_runs = [r for r in all_runs if r.get("benchmark_id") == selected_bid]
    if not matched_runs:
        st.warning(f"No execution records found for {selected_bid}.")
    else:
        tabs = st.tabs([f"Run `{r.get('run_id')}` (P{r['phase']} · {r.get('display_status')})" for r in matched_runs])
        for idx, r in enumerate(matched_runs):
            with tabs[idx]:
                c_a, c_b, c_c = st.columns(3)
                j_eval = latest_valid_eval(all_evals, r)
                c_a.markdown(f"**Execution Status:** `{r.get('display_status')}` (`{r.get('execution_mode')}`)")
                c_a.markdown(f"**Model:** `{r.get('model')}`")
                c_b.markdown(f"**Latency:** `{r.get('latency')} s`")
                c_b.markdown(f"**Tokens:** `{r.get('total_tokens')} ({r.get('token_basis')})`")
                c_c.markdown(f"**Judge Status:** `{eval_status_for(all_evals, r)}`")
                c_c.markdown(f"**Judge Mean:** `{j_eval['mean_score'] if j_eval else 'N/A'}`")

                if j_eval:
                    st.markdown("**Independent External Judge Rubric Scores:**")
                    j_col1, j_col2, j_col3, j_col4 = st.columns(4)
                    j_col1.metric("Relevance", f"{j_eval.get('relevance')}/5")
                    j_col2.metric("Completeness", f"{j_eval.get('completeness')}/5")
                    j_col3.metric("Organization", f"{j_eval.get('organization')}/5")
                    j_col4.metric("Grounding", f"{j_eval.get('grounding')}/5")
                    st.caption(f"Justification: {j_eval.get('justification')}")

                if r.get("integrity_flags"):
                    st.warning(f"Integrity Flags: {', '.join(r['integrity_flags'])}")

                rp = r.get("report_path")
                if rp:
                    text = read_text(rp)
                    if text:
                        with st.expander(f"📄 Inspect Report: `{rp}`", expanded=False):
                            st.markdown(text)
                    else:
                        st.error(f"Report artifact file missing from disk: {rp}")


# =============================================================================
# PAGE 3: PHASE 1 VS PHASE 2 COMPARISON
# =============================================================================
elif nav_section == "⚖️ Phase 1 vs Phase 2 Comparison":
    st.header("⚖️ Phase 1 (Baseline) vs Phase 2 (Critic Loop) Comparison")
    st.markdown(
        "Rigorous empirical evaluation comparing the single-pass feed-forward baseline (Phase 1) "
        "against the closed-loop dialectical peer review system (Phase 2). "
        "Scores reflect independent external LLM-as-a-Judge evaluations. "
        "Deltas and aggregates are calculated **strictly over valid paired LIVE evaluations**."
    )

    paths = DataPaths()
    comp = build_comparison(paths=paths)
    rows = comp["rows"]
    agg = comp["aggregates"]
    counts = comp["counts"]
    valid_n = agg["valid_pairs"]

    # Main Comparison Table
    st.subheader("📋 Empirical Comparison Matrix")

    display_rows = []
    for r in rows:
        def fmt_dim(tag, d):
            v = r.get(f"{tag}_{d}")
            return str(v) if v is not None else "N/A"

        m1 = f"{r['p1_mean']:.2f}" if r.get("p1_mean") is not None else "N/A"
        m2 = f"{r['p2_mean']:.2f}" if r.get("p2_mean") is not None else "N/A"
        d_val = r.get("mean_delta")
        delta_str = (f"+{d_val:.2f}" if d_val > 0 else f"{d_val:.2f}") if d_val is not None else "N/A"

        lat1 = f"{r['p1_latency']:.1f}s" if r.get("p1_latency") is not None else "N/A"
        lat2 = f"{r['p2_latency']:.1f}s" if r.get("p2_latency") is not None else "N/A"
        tok1 = f"{r['p1_tokens']:,}" if r.get("p1_tokens") is not None else "N/A"
        tok2 = f"{r['p2_tokens']:,}" if r.get("p2_tokens") is not None else "N/A"

        display_rows.append({
            "Q-ID": r["question_id"],
            "Question": r["question"][:38] + "...",
            "P1 Rel": fmt_dim("p1", "relevance"),
            "P2 Rel": fmt_dim("p2", "relevance"),
            "P1 Comp": fmt_dim("p1", "completeness"),
            "P2 Comp": fmt_dim("p2", "completeness"),
            "P1 Org": fmt_dim("p1", "organization"),
            "P2 Org": fmt_dim("p2", "organization"),
            "P1 Ground": fmt_dim("p1", "grounding"),
            "P2 Ground": fmt_dim("p2", "grounding"),
            "P1 Mean": m1,
            "P2 Mean": m2,
            "Mean Delta": delta_str,
            "P1 Lat": lat1,
            "P2 Lat": lat2,
            "P1 Tok": tok1,
            "P2 Tok": tok2,
            "P2 Iters": str(r.get("p2_iterations") or "0"),
            "Valid Pair": "✅ Yes" if r["pair_valid"] else f"❌ {r['pair_note'][:22]}",
        })

    df_comp = pd.DataFrame(display_rows)
    st.dataframe(df_comp, hide_index=True, use_container_width=True)

    # CSV Download Button
    csv_text = comparison_csv_text(rows)
    st.download_button(
        "💾 Download Comparison Table (CSV)",
        data=csv_text,
        file_name="comparison_table.csv",
        mime="text/csv",
    )

    st.divider()

    # Aggregate Metrics Section
    st.subheader(f"📈 Aggregate Experimental Metrics ({valid_n} of {counts['total_questions']} Valid Pairs)")
    if valid_n == 0:
        st.warning(
            "⚠️ **No benchmark questions currently have verified LIVE evaluations in BOTH Phase 1 and Phase 2.**\n\n"
            "Under rigorous scientific methodology, mock executions and incomplete runs are excluded from comparative statistics. "
            "Execute the live benchmark on both phases to populate aggregate quality and overhead metrics."
        )
    else:
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            st.metric("Avg Quality (P1)", f"{agg['avg_p1_quality']:.2f}/5.0")
            st.caption("Baseline Feed-Forward")
        with a_col2:
            st.metric("Avg Quality (P2)", f"{agg['avg_p2_quality']:.2f}/5.0")
            st.caption("Critic Loop")
        with a_col3:
            impr = agg['avg_improvement']
            st.metric("Quality Delta (P2 - P1)", f"{'+' if impr >= 0 else ''}{impr:.2f}")
            st.caption(f"Improved: {agg['pct_improved']}% · Tied: {agg['pct_unchanged']}% · Det: {agg['pct_deteriorated']}%")
        with a_col4:
            st.metric("Latency Overhead", f"+{agg['latency_overhead']}s")
            st.caption(f"Overhead: +{agg['latency_overhead_pct']}% (P1: {agg['avg_p1_latency']}s, P2: {agg['avg_p2_latency']}s)")

    st.divider()

    # Visualizations
    st.subheader("📊 Experimental Visualizations")
    if valid_n > 0:
        # Chart 1: Mean Scores by Question
        paired_data = [r for r in rows if r["pair_valid"]]
        chart_df = pd.DataFrame({
            "Question ID": [r["question_id"] for r in paired_data],
            "Phase 1 Mean": [r["p1_mean"] for r in paired_data],
            "Phase 2 Mean": [r["p2_mean"] for r in paired_data],
        }).set_index("Question ID")

        st.markdown("**1. Quality Score Comparison (Phase 1 vs Phase 2)**")
        st.bar_chart(chart_df)

        # Chart 2: 4-Dimension Breakdown
        dim_data = []
        for d in ["relevance", "completeness", "organization", "grounding"]:
            dim_data.append({
                "Dimension": d.capitalize(),
                "Phase 1": agg["per_dimension"][d]["p1"],
                "Phase 2": agg["per_dimension"][d]["p2"],
            })
        st.markdown("**2. Four-Dimensional Rubric Comparison (Averages Across Pairs)**")
        st.bar_chart(pd.DataFrame(dim_data).set_index("Dimension"))

        # Chart 3: Latency Comparison
        lat_df = pd.DataFrame({
            "Question ID": [r["question_id"] for r in paired_data],
            "Phase 1 (s)": [r["p1_latency"] for r in paired_data],
            "Phase 2 (s)": [r["p2_latency"] for r in paired_data],
        }).set_index("Question ID")
        st.markdown("**3. Computational Latency Overhead (Seconds)**")
        st.bar_chart(lat_df)
    else:
        st.info("Visual charts will render automatically once paired LIVE evaluations are recorded.")
