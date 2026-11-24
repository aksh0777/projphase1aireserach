"""
Main CLI Entry Point & Demo Trace Runner
Architecture Reference: docs/EXPLANATIONS.md - Section 9 and Chapter 13

Usage:
    python main.py "your question"             # Phase 1 baseline
    python main.py "your question" --critic    # Phase 2 critic + revision loop
    python main.py "your question" --mock      # force MOCK mode (no API calls)
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv

from graph.run_record import execute_run

load_dotenv()


class TeeStream:
    """Stream splitter that writes simultaneously to terminal stdout and a trace log file."""

    def __init__(self, trace_file_path: str):
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(trace_file_path), exist_ok=True)
        self.log_file = open(trace_file_path, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def run_research(topic: str, with_critic: bool = False, force_mock: bool = False) -> Dict[str, Any]:
    """Execute one isolated research run with real-time banners and trace persistence."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = "phase2_critic" if with_critic else "phase1_baseline"
    trace_path = os.path.join("outputs", f"demo_trace_{mode_label}_{timestamp}.txt")

    tee = TeeStream(trace_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        sys_title = "PHASE 2: CRITIC & REVISION LOOP DEMO" if with_critic else "PHASE 1: BASELINE DEMO"
        print("\n" + "#" * 68)
        print(f"### MULTI-AGENT DEEP RESEARCH SYSTEM ({sys_title})")
        print(f"### Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"### Target Topic: {topic}")
        print("#" * 68 + "\n")

        record, final_state = execute_run(topic, with_critic=with_critic, force_mock=force_mock)

        scores = (final_state.get("critique") or {}).get("scores", {})
        scores_str = (
            ", ".join(f"{k[:4].title()}: {v}" for k, v in scores.items()) if scores else "N/A (no Critic scores)"
        )
        tokens_txt = f"{record['total_tokens']} ({record['token_basis']})"
        lines = [
            "", "#" * 68, "========== FINAL EXECUTION SUMMARY ==========", "#" * 68,
            f"Run ID:         {record['run_id']}",
            f"Target Topic:   {topic}",
            f"Architecture:   {'Phase 2 (Critic + Revision Loop)' if with_critic else 'Phase 1 (Baseline DAG)'}",
            f"Status:         {record['display_status']}  (mode={record['execution_mode']}, status={record['execution_status']})",
            f"Model:          {record['model']}",
            f"Report Path:    {record['report_path'] or 'N/A'}",
            f"Total Latency:  {record['latency']}s",
            f"Tokens:         in={record['input_tokens']} out={record['output_tokens']} total={tokens_txt}",
            f"Searches Used:  {record['searches']}",
        ]
        if with_critic:
            lines += [
                f"Critic Iters:   {record.get('critic_iterations')}  Revisions: {record.get('revisions')}  "
                f"Cap reached: {record.get('cap_reached')}  Final verdict: {record.get('final_verdict')}",
                f"Draft versions: latest={record.get('latest_draft_version')} best={record.get('best_draft_version')} final={record.get('final_draft_version')}",
                f"Critic scores:  [{scores_str}]  (internal control signal, NOT the external Judge)",
            ]
        if record["error"]:
            lines.append(f"Error:          {record['error']}")
        lines += [
            f"Run artifact:   {record.get('artifact_path', 'n/a')}",
            f"Trace File:     {trace_path}", "#" * 68, "",
            f"SUMMARY: Status: {record['display_status']} | Phase{record['phase']} | Report: {record['report_path']} | "
            f"Latency: {record['latency']}s | Tokens: {tokens_txt} | Searches: {record['searches']}",
            "",
        ]
        print("\n".join(lines))
        return final_state
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"[Trace log saved to: {trace_path}]")


def main():
    with_critic = "--critic" in sys.argv
    force_mock = "--mock" in sys.argv
    cleaned = [a for a in sys.argv[1:] if a not in ("--critic", "--mock")]
    topic = " ".join(cleaned).strip() if cleaned else "Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026"
    run_research(topic, with_critic=with_critic, force_mock=force_mock)


if __name__ == "__main__":
    main()
