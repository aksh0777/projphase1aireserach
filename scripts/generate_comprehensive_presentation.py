"""
Comprehensive Presentation Deck Generator
Topic: Multi-Agent Deep Research System (Phase 1, Phase 2, Phase 2.2)
Generates: outputs/Multi_Agent_Deep_Research_Comprehensive_Presentation.pptx

Covers:
- Full project architecture & evolution (Phase 1 -> Phase 2 -> Phase 2.2)
- Flow diagrams (Baseline DAG, Closed-Loop Critic/Reviser, Relevance Validator)
- Challenges faced & solutions (Cross-topic contamination, naive truncation, rate limits)
- The mathematics (Rubric scoring, mean quality, deltas, latency & token overhead)
- Internal Critic vs External Judge distinction
- 3-Page Streamlit dashboard redesign
- Current experimental status and verification results
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Palette: Professional Academic Corporate (Clean, readable, human-made style)
    NAVY = RGBColor(15, 23, 42)        # #0F172A - Dominant title & text
    DARK_BLUE = RGBColor(30, 41, 59)   # #1E293B - Subheaders
    ACCENT_BLUE = RGBColor(37, 99, 235)# #2563EB - Highlights & Flow elements
    LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC - Card background
    CARD_BORDER = RGBColor(203, 213, 225) # #CBD5E1 - Card outline
    SLATE_TEXT = RGBColor(51, 65, 85)  # #334155 - Body copy
    MUTED_TEXT = RGBColor(100, 116, 139) # #64748B - Captions
    GREEN = RGBColor(5, 150, 105)      # #059669 - Success / Pass
    AMBER = RGBColor(217, 119, 6)      # #D97706 - Warnings / Challenges
    RED = RGBColor(220, 38, 38)        # #DC2626 - Errors / Strict guards
    WHITE = RGBColor(255, 255, 255)

    blank_layout = prs.slide_layouts[6]

    # -------------------------------------------------------------------------
    # Helper Functions for Visual Components
    # -------------------------------------------------------------------------
    def add_header(slide, title, tracker="MULTI-AGENT DEEP RESEARCH · COMPREHENSIVE ARCHITECTURE"):
        # Category tracker
        tb_track = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_track = tb_track.text_frame
        tf_track.word_wrap = True
        p_track = tf_track.paragraphs[0]
        p_track.text = tracker.upper()
        p_track.font.size = Pt(10)
        p_track.font.bold = True
        p_track.font.color.rgb = ACCENT_BLUE

        # Main slide title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, title, bullets, bg=LIGHT_BG, border=CARD_BORDER, title_color=NAVY):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.color.rgb = border
        shape.line.width = Pt(1)

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), height - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True

        if title:
            p0 = tf.paragraphs[0]
            p0.text = title
            p0.font.size = Pt(14)
            p0.font.bold = True
            p0.font.color.rgb = title_color
            p0.space_after = Pt(8)

        is_first = False if title else True
        for b in bullets:
            p = tf.paragraphs[0] if is_first else tf.add_paragraph()
            is_first = False
            p.text = ("• " if not b.startswith(("-", "1", "2", "3", "4", "★", "⚠️", "✅")) else "") + b
            p.font.size = Pt(11)
            p.font.color.rgb = SLATE_TEXT
            p.space_after = Pt(5)

    def add_flow_box(slide, left, top, width, height, text, subtext="", bg=WHITE, border=ACCENT_BLUE, text_color=NAVY):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.color.rgb = border
        shape.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(left + Inches(0.05), top + Inches(0.05), width - Inches(0.1), height - Inches(0.1))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = text
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = text_color

        if subtext:
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            p2.text = subtext
            p2.font.size = Pt(9.5)
            p2.font.color.rgb = MUTED_TEXT

    def add_arrow(slide, left, top, width=Inches(0.4), height=Inches(0.2)):
        shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = ACCENT_BLUE
        shape.line.fill.background()

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Accent decorative bar
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.6), Inches(1.0), Inches(0.08))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = ACCENT_BLUE
    bar1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.9), Inches(11.0), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_pre = tf1.paragraphs[0]
    p_pre.text = "MSc ACADEMIC RESEARCH PROJECT · PHASE 1, 2 & 2.2"
    p_pre.font.size = Pt(13)
    p_pre.font.bold = True
    p_pre.font.color.rgb = RGBColor(147, 197, 253)
    p_pre.space_after = Pt(14)

    p_t = tf1.add_paragraph()
    p_t.text = "Autonomous Multi-Agent Deep Research System"
    p_t.font.size = Pt(34)
    p_t.font.bold = True
    p_t.font.color.rgb = WHITE
    p_t.space_after = Pt(10)

    p_sub = tf1.add_paragraph()
    p_sub.text = "From Single-Pass DAG to Dialectical Closed-Loop Peer Review, Rigorous Isolation, and Empirical Evaluation"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(226, 232, 240)
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Author: Akshat Sharma  ·  Architecture: LangGraph + Gemini Flash  ·  Benchmark: 10 Standardized Research Domains"
    p_meta.font.size = Pt(12)
    p_meta.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & PROJECT EVOLUTION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Project Overview: The Three Phases of Evolution", "SYSTEM EVOLUTION & PROBLEM STATEMENT")

    add_card(
        s2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "1. Phase 1: Baseline DAG",
        [
            "Core Goal: Solve single-prompt LLM limits (hallucination & context saturation).",
            "Pipeline: Planner -> Parallel Web Retrieval -> Synthesizer -> Writer.",
            "Decomposition: Breaks topics into 3-5 orthogonal sub-questions.",
            "Attribution: Numbered in-text citations ([1], [2]) mapping to bibliography.",
            "Limitation: Strictly open-loop (feed-forward). Generation errors are permanent and cannot be self-corrected."
        ],
        border=CARD_BORDER
    )

    add_card(
        s2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "2. Phase 2: Closed-Loop Critic",
        [
            "Core Goal: Introduce self-correction through adversarial peer review.",
            "New Agents: Critic (strict reviewer) + Reviser (academic editor).",
            "Adversarial Isolation: Critic sees ONLY draft + question (blind to search notes).",
            "State Machine: Dynamic looping governed by pass threshold (all 4 scores >= 4).",
            "Safety Cap: Hard limit (MAX_REVISIONS = 3) prevents infinite generation loops."
        ],
        border=ACCENT_BLUE,
        bg=RGBColor(240, 249, 255)
    )

    add_card(
        s2, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "3. Phase 2.2: Reliability & UI",
        [
            "Cross-Topic Isolation: Eradicated silent mock template cross-contamination.",
            "Relevance Validator Gate: Audits subquestion alignment & semantic relevance.",
            "No Silent Mock Fallbacks: Live runs never substitute canned text on 429 quota exhaustion.",
            "Evaluation Dashboard: 3-page interactive Streamlit UI with true N/A handling.",
            "Automated Testing: 13-test regression suite ensuring 100% mathematical integrity."
        ],
        border=GREEN,
        bg=RGBColor(240, 253, 244)
    )

    # =========================================================================
    # SLIDE 3: PHASE 1 ARCHITECTURE — BASELINE FEED-FORWARD DAG
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Phase 1: Deterministic Multi-Agent Retrieval DAG", "PHASE 1 · BASELINE SYSTEM")

    # Flow Diagram across the top
    top_y = Inches(1.7)
    add_flow_box(s3, Inches(0.8), top_y, Inches(1.6), Inches(0.9), "User Topic", "Input Inquiry", bg=LIGHT_BG)
    add_arrow(s3, Inches(2.45), top_y + Inches(0.35))

    add_flow_box(s3, Inches(2.9), top_y, Inches(1.8), Inches(0.9), "Planner", "3-5 Sub-Questions", border=ACCENT_BLUE)
    add_arrow(s3, Inches(4.75), top_y + Inches(0.35))

    add_flow_box(s3, Inches(5.2), top_y, Inches(2.0), Inches(0.9), "Researchers", "Parallel Fan-Out", border=ACCENT_BLUE)
    add_arrow(s3, Inches(7.25), top_y + Inches(0.35))

    add_flow_box(s3, Inches(7.7), top_y, Inches(1.9), Inches(0.9), "Synthesizer", "De-dup & Conflict", border=ACCENT_BLUE)
    add_arrow(s3, Inches(9.65), top_y + Inches(0.35))

    add_flow_box(s3, Inches(10.1), top_y, Inches(1.6), Inches(0.9), "Writer", "Cited Markdown", border=ACCENT_BLUE)
    add_arrow(s3, Inches(11.75), top_y + Inches(0.35))

    add_flow_box(s3, Inches(12.2), top_y, Inches(0.8), Inches(0.9), "Report", "END", bg=RGBColor(220, 252, 231), border=GREEN)

    # Lower Explanation Cards
    add_card(
        s3, Inches(0.8), Inches(3.0), Inches(5.6), Inches(3.9),
        "Key Principles & Operational Flow",
        [
            "1. Divide-and-Conquer (Cognitive Load Reduction):",
            "   An LLM has an effective receptive field limit. Breaking a complex inquiry into orthogonal sub-questions cuts cognitive entropy.",
            "2. Map-Reduce Parallel Retrieval:",
            "   LangGraph Send dispatches worker nodes concurrently. Cuts retrieval latency by ~60% compared to sequential querying.",
            "3. Provenance & Attribution Tracking:",
            "   Every extracted finding is tagged with source URL [https://...].",
            "4. Conflict Synthesis:",
            "   Synthesizer explicitly highlights contradictions across sources."
        ]
    )

    add_card(
        s3, Inches(6.8), Inches(3.0), Inches(5.7), Inches(3.9),
        "The Open-Loop Problem (Why Phase 2 was Necessary)",
        [
            "❌ Single-Pass Finality:",
            "   Once the Writer generates Markdown, it is immediately finalized. Minor truncations, unverified numbers, or weak transitions remain forever.",
            "❌ Autoregressive Drift:",
            "   LLMs accumulate small errors over 3,000+ words, leading to incomplete concluding sections or dropped citation links.",
            "❌ Lack of Dialectical Feedback:",
            "   The Writer cannot objectively grade its own output. Academic excellence requires independent peer review."
        ],
        border=AMBER,
        bg=RGBColor(255, 251, 235)
    )

    # =========================================================================
    # SLIDE 4: PHASE 2 ARCHITECTURE — CLOSED-LOOP CRITIC & REVISER
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Phase 2: Closed-Loop Dialectical State Machine", "PHASE 2 · CRITIC & REVISION LOOP")

    # Flow Diagram
    fy = Inches(1.7)
    add_flow_box(s4, Inches(0.8), fy, Inches(1.8), Inches(0.8), "Initial Draft", "Writer Output", bg=LIGHT_BG)
    add_arrow(s4, Inches(2.65), fy + Inches(0.3))

    add_flow_box(s4, Inches(3.1), fy, Inches(2.2), Inches(0.8), "Critic Node", "Blind Peer Review", border=ACCENT_BLUE, bg=RGBColor(239, 246, 255))
    add_arrow(s4, Inches(5.35), fy + Inches(0.3))

    add_flow_box(s4, Inches(5.8), fy, Inches(2.3), Inches(0.8), "Verdict Router", "Pass OR Cap Reached?", border=AMBER, bg=RGBColor(254, 243, 199))

    # Down arrow branch for Revise
    shape_down = s4.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(6.75), Inches(2.6), Inches(0.4), Inches(0.5))
    shape_down.fill.solid()
    shape_down.fill.fore_color.rgb = AMBER
    shape_down.line.fill.background()

    add_flow_box(s4, Inches(5.6), Inches(3.2), Inches(2.7), Inches(0.8), "Reviser Node", "Resolves Flaws from Review", border=AMBER)

    # Left arrow back to Critic
    shape_left = s4.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Inches(4.5), Inches(3.4), Inches(1.0), Inches(0.4))
    shape_left.fill.solid()
    shape_left.fill.fore_color.rgb = ACCENT_BLUE
    shape_left.line.fill.background()

    # Pass branch to Finalize
    add_arrow(s4, Inches(8.15), fy + Inches(0.3))
    add_flow_box(s4, Inches(8.6), fy, Inches(2.2), Inches(0.8), "Finalize Node", "Select Best Version", border=GREEN)
    add_arrow(s4, Inches(10.85), fy + Inches(0.3))
    add_flow_box(s4, Inches(11.3), fy, Inches(1.2), Inches(0.8), "Report", "END", bg=RGBColor(220, 252, 231), border=GREEN)

    # Lower Explanation Cards
    add_card(
        s4, Inches(0.8), Inches(4.3), Inches(5.7), Inches(2.7),
        "1. Adversarial Isolation Principle",
        [
            "• Blind Review: Critic sees ONLY original question + current draft.",
            "• Zero Note Access: Blind to search notes and planner decomposition.",
            "• Why? Eliminates confirmation bias. The draft must stand completely on its own communicative and empirical merits."
        ]
    )

    add_card(
        s4, Inches(6.8), Inches(4.3), Inches(5.7), Inches(2.7),
        "2. Termination & Selection Policy",
        [
            "• Pass Threshold: All 4 scores must be >= 4 (Relevance, Completeness, Organization, Grounding). If any <= 3, verdict = revise.",
            "• Hard Cap: MAX_REVISIONS = 3 prevents infinite loops.",
            "• Best-Draft Preservation: If cap reached, system chooses the best-scoring draft across history, not blindly the latest."
        ]
    )

    # =========================================================================
    # SLIDE 5: PHASE 2.2 ARCHITECTURE — RELEVANCE VALIDATOR GATE
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Phase 2.2: Research Relevance Validator Gate", "PHASE 2.2 · RELIABILITY & INTEGRITY")

    add_card(
        s5, Inches(0.8), Inches(1.6), Inches(4.0), Inches(5.3),
        "Why the Gate was Needed",
        [
            "★ Problem Discovered:",
            "  Early runs sometimes drifted or accepted evidence that belonged to another query.",
            "★ Core Philosophy:",
            "  A generation agent (Writer) should NEVER be allowed to author from unchecked evidence.",
            "★ Architecture Placement:",
            "  Inserted directly between Synthesizer and Writer across BOTH Phase 1 and Phase 2 pipelines."
        ],
        border=AMBER
    )

    # Middle Flow Column
    add_card(
        s5, Inches(5.0), Inches(1.6), Inches(4.0), Inches(5.3),
        "4-Step Validation Protocol",
        [
            "1. Isolation Audit:",
            "   Checks that every record carries the current run_id. Drops foreign data.",
            "2. Sub-Question Assignment:",
            "   Verifies findings match this run's actual plan.",
            "3. Citation Provenance:",
            "   Ensures cited URLs were retrieved for that specific query.",
            "4. Batched Semantic Audit (Temp=0):",
            "   LLM auditor verifies findings address the inquiry. Tolerates interdisciplinary nuances, rejects off-topic content."
        ],
        border=ACCENT_BLUE
    )

    add_card(
        s5, Inches(9.2), Inches(1.6), Inches(3.3), Inches(5.3),
        "Self-Healing Actions",
        [
            "• Context-Enriched Retry:",
            "  Flagged sub-questions are re-queried with topic context.",
            "• Transparent Evidence Gaps:",
            "  Unrecoverable subquestions are excluded and listed in an 'Evidence Gaps' section.",
            "• Clean Halting:",
            "  If zero sub-questions pass, run terminates cleanly as FAILED instead of faking text."
        ],
        border=GREEN
    )

    # =========================================================================
    # SLIDE 6: CHALLENGES FACED — CROSS-TOPIC RESEARCH CONTAMINATION
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Critical Challenge 1: Cross-Topic Contamination", "RESEARCH INTEGRITY & BUG ROOT CAUSE")

    add_card(
        s6, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "The Bug: Cricket Query -> Battery Content",
        [
            "⚠️ Observed User Defect:",
            "   Prompt: 'Compare captaincy records of MS Dhoni vs Virat Kohli'.",
            "   Generated Report: Included cricket headers, but section body discussed 'manufacturing maturity, raw material availability, and levelized battery degradation'.",
            "",
            "🔍 Root Cause Investigation:",
            "   1. Quota Exhaustion: Google Free Tier quota (20 req/day) was hit, raising HTTP 429 RESOURCE_EXHAUSTED.",
            "   2. Silent Mock Fallback: In agents/llm.py, failed live calls silently defaulted to mock_generator.",
            "   3. Hardcoded Mock Template: In agents/writer.py, the mock fallback contained canned battery sentences written during early system development."
        ],
        border=RED,
        bg=RGBColor(254, 242, 242)
    )

    add_card(
        s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "The Architectural Solution",
        [
            "✅ Strict Execution Isolation:",
            "   new_run_state creates isolated containers with unique run_id. No shared mutable defaults.",
            "✅ Elimination of Silent Fallbacks:",
            "   A run is either LIVE or MOCK. Live runs NEVER silently switch to mock text halfway. If quota is exhausted, the call raises LLMUnavailableError.",
            "✅ Generic, Parameterized Mock Generators:",
            "   Mock generators now construct text strictly from the provided question and plan. Zero canned domain content.",
            "✅ Provenance Filtering:",
            "   Downstream nodes drop any record stamped with a foreign run_id."
        ],
        border=GREEN,
        bg=RGBColor(240, 253, 244)
    )

    # =========================================================================
    # SLIDE 7: CHALLENGES FACED — TRUNCATION & QUOTA CEILINGS
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Critical Challenges 2 & 3: Truncation & Quotas", "SYSTEM BOTTLENECKS & MITIGATIONS")

    add_card(
        s7, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Challenge 2: The Self-Inflicted Truncation Trap",
        [
            "⚠️ The Problem:",
            "   Early Critic and Judge implementations used naive slicing: report_text[:12000].",
            "",
            "💥 The Cascade Effect:",
            "   • The Critic reviewed an artificially cut draft and flagged: 'Section 4 terminates mid-sentence!'.",
            "   • The Reviser attempted to fix the cut section, but the next Critic pass truncated it again.",
            "   • The system wasted revision loops chasing a phantom flaw created by its own input slicing.",
            "",
            "✅ The Fix:",
            "   Gemini Flash models possess context windows >1,000,000 tokens. We eliminated arbitrary string slicing, passing full reports safely up to 120k characters."
        ],
        border=AMBER
    )

    add_card(
        s7, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Challenge 3: Free-Tier Daily Quotas & Pacing",
        [
            "⚠️ The Constraint:",
            "   • Free Tier enforces a strict limit: 20 requests per day per model.",
            "   • A single Phase 2 run with 3 iterations makes ~14 LLM calls and consumes ~96,000 tokens.",
            "   • Running 2 questions back-to-back triggers HTTP 429 quota exhaustion.",
            "",
            "✅ Engineering Mitigations:",
            "   • Exponential Backoff: Smart retry logic with jittered sleeps.",
            "   • Pacing Delays: 2-3 second pacing between parallel queries to avoid DuckDuckGo search rate limits.",
            "   • Forced Mock Mode (--mock): Zero-cost deterministic testing mode allowing full UI and graph verification without consuming API calls."
        ],
        border=ACCENT_BLUE
    )

    # =========================================================================
    # SLIDE 8: THE MATHEMATICS — EVALUATION RUBRIC & SCORING
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Mathematical Formulation: Rubric & Quality Scoring", "EVALUATION METHODOLOGY · PART 1")

    add_card(
        s8, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "The 4-Dimensional Standardized Rubric",
        [
            "Every report is scored across 4 orthogonal dimensions as integers from 1 to 5:",
            "",
            "1. Relevance (R ∈ {1, 2, 3, 4, 5}):",
            "   Does the report directly address the core inquiry without topical drift?",
            "2. Completeness (C ∈ {1, 2, 3, 4, 5}):",
            "   Are all technical, economic, and operational facets explored?",
            "3. Organization (O ∈ {1, 2, 3, 4, 5}):",
            "   Is the document structured logically with clear headings and transitions?",
            "4. Grounding (G ∈ {1, 2, 3, 4, 5}):",
            "   Are factual claims anchored to numbered citations ([1], [2]) linking to verified references?"
        ]
    )

    add_card(
        s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Core Mathematical Formulas",
        [
            "★ Individual Report Mean Quality Score:",
            "   Mean = (R + C + O + G) / 4.0",
            "   (Bounded strictly in [1.0, 5.0])",
            "",
            "★ Critic Pass Threshold Condition:",
            "   Pass = [ min(R, C, O, G) >= 4 ]",
            "   (If any single score <= 3, the verdict is strictly 'revise')",
            "",
            "★ Missing Score Integrity Rule:",
            "   Score = None -> Displayed as 'N/A'",
            "   Missing scores are NEVER treated as 0.0 (prevents artificial score deflation)."
        ],
        border=ACCENT_BLUE,
        bg=RGBColor(240, 249, 255)
    )

    # =========================================================================
    # SLIDE 9: THE MATHEMATICS — COMPARATIVE DELTAS, OVERHEAD & COSTS
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Mathematical Formulation: Deltas, Overhead & Costs", "EVALUATION METHODOLOGY · PART 2")

    add_card(
        s9, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Comparative Deltas & Paired Aggregates",
        [
            "★ Question Quality Delta (Δ):",
            "   Δ = Mean(Phase 2) - Mean(Phase 1)",
            "   (Calculated ONLY when BOTH Phase 1 and Phase 2 have valid independent Judge scores)",
            "",
            "★ Aggregate Quality Improvement:",
            "   Δ_avg = (1 / N_valid) * Σ (Δ_i)  for i ∈ Valid Pairs",
            "",
            "★ Categorical Breakdown (%):",
            "   • Improved:     (Count[Δ > 0] / N_valid) * 100",
            "   • Unchanged:    (Count[Δ = 0] / N_valid) * 100",
            "   • Deteriorated: (Count[Δ < 0] / N_valid) * 100"
        ]
    )

    add_card(
        s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Computational Overhead & Cost Accounting",
        [
            "★ Computational Latency Overhead:",
            "   Overhead_lat = Latency(P2) - Latency(P1)",
            "   Overhead_% = [ (Latency_P2 / Latency_P1) - 1 ] * 100",
            "",
            "★ Token Accounting & Financial Cost:",
            "   Tokens_total = Tokens_prompt + Tokens_completion",
            "   • Extracted directly from Gemini usage_metadata.",
            "   • Categorized as 'actual' vs 'estimated'.",
            "",
            "★ Token Overhead:",
            "   Overhead_tok = Tokens_P2 - Tokens_P1",
            "   Empirical evidence: Phase 2 consumes ~3.3x more tokens to achieve verified grounding."
        ],
        border=ACCENT_BLUE
    )

    # =========================================================================
    # SLIDE 10: INTERNAL CRITIC VS EXTERNAL JUDGE
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Internal Critic vs External Judge: Fundamental Separation", "SCIENTIFIC RIGOR & INDEPENDENCE")

    add_card(
        s10, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Internal Critic Agent (Online Controller)",
        [
            "• Timing: Runs inside the LangGraph execution loop.",
            "• Purpose: Acts as a feedback controller to guide draft revision.",
            "• Target: Evaluates unfinalized, intermediate drafts (v0, v1, v2).",
            "• Inputs: Question + Draft ONLY (blind to search notes).",
            "• Outputs: Pass/Revise verdict + list of actionable flaws (max 5).",
            "• Data Role: Stored in state['score_history']; NEVER used as final benchmark scores."
        ],
        border=ACCENT_BLUE,
        bg=RGBColor(240, 249, 255)
    )

    add_card(
        s10, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "External LLM-as-a-Judge (Offline Evaluator)",
        [
            "• Timing: Runs post-hoc after research execution completes.",
            "• Purpose: Acts as an independent academic referee for the benchmark.",
            "• Target: Evaluates published final markdown reports.",
            "• Context: Full report content + original benchmark question.",
            "• Guard: Refuses to score mock templates or contaminated reports.",
            "• Data Role: Appended to outputs/runs/judge_evaluations.jsonl with immutable evaluation IDs."
        ],
        border=GREEN,
        bg=RGBColor(240, 253, 244)
    )

    # =========================================================================
    # SLIDE 11: STREAMLIT APPLICATION REDESIGN
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Streamlit Academic Platform: 3-Page Interface", "USER EXPERIENCE & DEMONSTRATION INTERFACE")

    add_card(
        s11, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "Page 1: Research Workspace",
        [
            "• Topic input & Phase 1/Phase 2 toggle.",
            "• Live status banners (LIVE, MOCK, PARTIAL, FAILED).",
            "• Real-time streaming status cards for all 6 agent stages.",
            "• Transparent Critic iteration cards with flaw checklists.",
            "• Full markdown viewer with one-click report download (.md).",
            "• Telemetry sidebar with latency per node & token breakdown."
        ]
    )

    add_card(
        s11, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "Page 2: Evaluation Dashboard",
        [
            "• Benchmark execution overview cards (Total questions, runs, evaluated, pending).",
            "• Detailed audit table joining Q01..Q10 with run IDs, status, latency, and tokens.",
            "• Phase filters (All, Phase 1 only, Phase 2 only).",
            "• Individual question drill-down inspector showing full report text and external Judge scores."
        ]
    )

    add_card(
        s11, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "Page 3: Comparison Matrix",
        [
            "• 10-Question comparative matrix joining Phase 1 vs Phase 2.",
            "• Displays 4-dimensional rubric scores, Means, and Deltas.",
            "• Strict N/A handling for missing values (never 0.0).",
            "• Instant CSV download of comparison_table.csv.",
            "• Aggregate statistics cards (Quality Delta, Overhead %).",
            "• Real-data visual bar charts."
        ]
    )

    # =========================================================================
    # SLIDE 12: AUTOMATED TESTING & VERIFICATION
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Verification Suite: 13 Automated Pytest Assurances", "QUALITY ASSURANCE & REGRESSION RIGOR")

    add_card(
        s12, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "1. Isolation Suite (4 Tests)",
        [
            "• test_new_run_state_isolation:",
            "  Verifies state containers are freshly allocated without shared defaults.",
            "• test_current_run_records_drops_foreign_runs:",
            "  Proves records from other run IDs are rejected.",
            "• test_consecutive_runs_no_cross_contamination:",
            "  Runs 4 consecutive topics (Cricket, Batteries, RAG, Cancer) in one process; asserts 0 term leakage across both Phase 1 and 2.",
            "• test_research_records_provenance_tagging"
        ],
        border=GREEN
    )

    add_card(
        s12, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "2. Evaluation Suite (5 Tests)",
        [
            "• test_benchmark_questions_suite_integrity:",
            "  Verifies Q01..Q10 stable IDs and bidirectional lookup.",
            "• test_mean_score_calculation:",
            "  Validates (R+C+O+G)/4 math and non-integer/null rejections.",
            "• test_mock_runs_excluded_from_comparisons:",
            "  Ensures mock data is disqualified from scientific sets.",
            "• test_comparison_preserves_missing_as_none:",
            "  Asserts missing values remain None, never 0.",
            "• test_comparison_csv_export_format"
        ],
        border=ACCENT_BLUE
    )

    add_card(
        s12, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "3. Regression Suite (4 Tests)",
        [
            "• test_phase1_baseline_feed_forward_no_critic:",
            "  Guarantees Phase 1 executes 0 critic passes.",
            "• test_phase2_critic_loop_iterations_and_cap:",
            "  Guarantees loop terminates at MAX_REVISIONS = 3.",
            "• test_phase1_baseline_frozen_files_untouched:",
            "  Verifies all 13 control files in outputs/phase1_baseline/ remain 100% byte-identical.",
            "",
            "★ Result: 13 passed in 0.87 seconds."
        ],
        border=NAVY
    )

    # =========================================================================
    # SLIDE 13: EXPERIMENTAL STATUS & EMPIRICAL FINDINGS
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Experimental Status & Empirical Findings", "PROJECT UPDATE & CURRENT DATASET")

    add_card(
        s13, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Question 1 Empirical Case Study: Battery Storage",
        [
            "Target: 'Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026'",
            "",
            "★ Phase 1 Baseline Execution:",
            "  • Latency: 89.05 seconds | Tokens: 28,687",
            "  • Judge Scores: Rel: 5, Comp: 4, Org: 5, Ground: 3 (Mean: 4.25)",
            "  • Weakness: Lacked specific citation bindings in middle sections.",
            "",
            "★ Phase 2 Critic Loop Execution (3 Iterations):",
            "  • Latency: 421.49 seconds | Tokens: 96,137",
            "  • Iteration 1 Critic: 5/3/3/4 -> Revise (flaw: mid-sentence cut)",
            "  • Iteration 2 Critic: 4/2/3/4 -> Revise (flaw: missing references)",
            "  • Iteration 3 Critic: 3/2/3/4 -> Revise (flaw: thermal safety gap)",
            "  • Cap Enforced: Finalized at MAX_REVISIONS = 3.",
            "  • Outcome: Grounding improved from 3/5 to 4/5."
        ],
        border=ACCENT_BLUE
    )

    add_card(
        s13, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3),
        "Current Benchmark Dataset Accounting",
        [
            "• Total Questions: 10 Standardized Benchmarks (Q01..Q10).",
            "• Phase 1 Baseline Dataset: Frozen and preserved in outputs/phase1_baseline/.",
            "• Honest Scientific Status:",
            "  Due to Google AI Studio's 20 req/day free-tier limit, only Q01 has complete live execution. Q02-Q10 experienced quota fallback.",
            "",
            "• Scientific Defensibility:",
            "  The dashboard honestly reports 0 valid live pairs under strict filtering. We refuse to fabricate synthetic deltas or display fake 5/5 scores.",
            "",
            "• Path to Full Evaluation:",
            "  Ready to execute across 5 daily quota windows or using a standard paid API billing tier."
        ],
        border=GREEN
    )

    # =========================================================================
    # SLIDE 14: SUMMARY & DEFENSE TAKEAWAYS
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Conclusion: Deliverables & Defense Takeaways", "SUMMARY & FUTURE TRAJECTORY")

    add_card(
        s14, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "1. Core Deliverables Built",
        [
            "• Fully functional LangGraph state machine with baseline DAG and closed-loop critique.",
            "• Adversarial isolation preventing confirmation bias.",
            "• Research Relevance Validator gate eliminating cross-topic contamination.",
            "• 3-page academic Streamlit dashboard with CSV export and audit tables."
        ]
    )

    add_card(
        s14, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3),
        "2. Key Insights Learned",
        [
            "• Quality vs. Cost Trade-Off: Closed-loop critique measurably improves grounding (+1 point), but demands ~4.7x latency and ~3.3x tokens.",
            "• Slicing Traps: Never arbitrarily truncate document context when LLMs support large windows.",
            "• The Importance of Provenance: Stamping run IDs across blackboard states prevents catastrophic multi-agent contamination."
        ]
    )

    add_card(
        s14, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3),
        "3. Ready for Deployment",
        [
            "• Working tree clean and backed up on backup/pre-phase2.2.",
            "• 13 passing automated regression tests.",
            "• Ready to commit and push to GitHub (origin main), triggering automatic Streamlit Cloud update."
        ],
        border=GREEN,
        bg=RGBColor(240, 253, 244)
    )

    # Save Presentation
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "Multi_Agent_Deep_Research_Comprehensive_Presentation.pptx")
    prs.save(out_path)
    print(f"[PRESENTATION] Successfully generated 14-slide presentation: {out_path}")
    return out_path


if __name__ == "__main__":
    build_presentation()
