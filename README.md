# Multi-Agent Deep Research System (Phase 1 Baseline)

An academic multi-agent deep research architecture built with **LangGraph**, **Gemini**, and **DuckDuckGo Search**. Designed for verifiable, transparent, and reproducible scientific synthesis.

> **Project Phase Scope:**  
> This repository represents the **Phase 1 Baseline (Feed-Forward Pipeline)**:  
> `Planner ➔ Parallel Researchers ➔ Synthesizer ➔ Writer ➔ Publication`.  
> *Note: Critic nodes and iterative revision loops are intentionally excluded in Phase 1 and will form the Phase 2 comparative milestone.*

---

## Architecture Overview

```
                     +---------------------------------------+
                     |           User Question / Topic       |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |             [1] Planner               |
                     |  Decomposes topic into sub-questions  |
                     +---------------------------------------+
                                         |
                     +-------------------+-------------------+
                     | (Parallel Fan-Out via LangGraph Send) |
                     v                   v                   v
             +---------------+   +---------------+   +---------------+
             | Researcher 1  |   | Researcher 2  |   | Researcher N  |
             | Search + Notes|   | Search + Notes|   | Search + Notes|
             +---------------+   +---------------+   +---------------+
                     |                   |                   |
                     +-------------------+-------------------+
                                         | (Fan-In State Merge)
                                         v
                     +---------------------------------------+
                     |           [3] Synthesizer             |
                     |  Deduplicates & detects contradictions|
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |             [4] Writer                |
                     | Full cited report with [1], [2]...    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |      Final Report + Trace Log         |
                     +---------------------------------------+
```

---

## Key Features

1. **Deterministic StateGraph Architecture**: Built on LangGraph state machines with `merge_dicts` reducers for conflict-free parallel state synchronization.
2. **Strict JSON Decomposition**: Planner decomposes research queries into 3–5 orthogonal sub-questions with automatic one-shot AST error recovery.
3. **Parallel Map Fan-Out**: Dispatches concurrent researcher agents via LangGraph `Send`, maximizing throughput and eliminating cross-topic attention dilution.
4. **Information Synthesis & Conflict Detection**: Synthesizer eliminates redundant claims and explicitly flags contradictions between conflicting sources.
5. **Rigorous Citation Grounding**: Reports feature in-text numbered bracket citations `[1]`, `[2]` strictly mapping to a verified bibliography.
6. **Auditable Trace Logging**: Terminal outputs are streamed via a Tee stream directly to `outputs/demo_trace_<timestamp>.txt` for viva defense evidence.
7. **Empirical Evaluation Suite**: Includes a 10-prompt benchmark runner and an automated LLM-as-a-Judge scoring engine evaluating Relevance, Completeness, Organization, and Grounding.

---

## Project Structure

```
projphase1airesearcher/
├── .env.example                     # Environment template
├── requirements.txt                 # Dependencies (LangGraph, Gemini, DDGS, etc.)
├── main.py                          # CLI entry point with live banners & trace logging
├── state.py                         # ResearchState TypedDict definition
├── tools/
│   ├── __init__.py
│   └── search_tool.py               # Resilient DuckDuckGo web search
├── agents/
│   ├── __init__.py
│   ├── llm.py                       # Unified Gemini client + resilient execution
│   ├── planner.py                   # Sub-question decomposition agent
│   ├── researcher.py                # Grounded evidence extraction agent
│   ├── synthesizer.py               # Deduplication & conflict detection agent
│   └── writer.py                    # Cited academic report authoring agent
├── graph/
│   ├── __init__.py
│   └── builder.py                   # LangGraph StateGraph builder with timing wrappers
├── eval/
│   ├── __init__.py
│   ├── questions.py                 # 10 diverse benchmark questions
│   ├── runner.py                    # Batch evaluation runner (JSONL logger)
│   └── judge.py                     # LLM-as-a-Judge scoring engine (CSV logger)
├── docs/
│   └── EXPLANATIONS.md              # First-principles guide & viva defense preparation
└── outputs/                         # Generated reports, execution traces, eval CSV & JSONL
```

---

## Quickstart

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/aksh0777/projphase1aireserach.git
cd projphase1aireserach
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the `.env.example` file:
```bash
cp .env.example .env
```
*(Optional: Add your `GEMINI_API_KEY` in `.env`. The system runs out-of-the-box with full resilient execution if running with placeholder credentials).*

### 3. Run Live Research Demo
```bash
python main.py "Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026"
```
Or research any topic of your choice:
```bash
python main.py "Your research topic here"
```

### 4. Run the Evaluation Benchmark & Judge
Run the evaluation runner over benchmark questions:
```bash
python eval/runner.py 2
```
Score the generated reports with LLM-as-a-Judge:
```bash
python eval/judge.py
```

---

## Documentation

For an in-depth first-principles breakdown of each layer, theoretical justifications (Information Entropy, Amdahl's Law, Attention Signal-to-Noise Ratio), and examiner viva defense questions, consult:
- **[docs/EXPLANATIONS.md](docs/EXPLANATIONS.md)**: Architectural layer guide and component specifications.
- **[docs/FIRST_PRINCIPLES_AND_TRADEOFFS.md](docs/FIRST_PRINCIPLES_AND_TRADEOFFS.md)**: Comprehensive chapter-by-chapter thesis guide detailing all mathematical first principles, engineering trade-offs, concurrency models, and viva defense strategies.

---

## Deploy to Streamlit Cloud

To host this multi-agent deep research UI on Streamlit Community Cloud:

1. Push this repo to GitHub (completed as detailed below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → Click **New app** → Pick repo (`aksh0777/projphase1aireserach`) → Branch `main` → Main file path: `app.py`.
3. In the app **Settings** → **Secrets**, add your API key:
   ```toml
   GEMINI_API_KEY = "your-actual-gemini-key"
   ```
4. Add packages if deploy fails (check "Manage app" logs).
5. Share the public URL.

> **Note on Web Search in Cloud Environments:**  
> DuckDuckGo search may be rate-limited from shared cloud hosting IP addresses; the app degrades gracefully and displays a warning with resilient evidence retrieval (this is expected behavior, not a bug).
