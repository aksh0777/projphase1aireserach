# Multi-Agent Deep Research System: First-Principles Architecture Guide

This document explains the architecture, design choices, and first principles behind each layer of the baseline multi-agent deep research system. It is written in direct, accessible academic language suitable for an MSc project review.

---

## 1. System Overview & The Baseline Philosophy

### (a) What Problem This Baseline Solves
Modern complex research tasks cannot be effectively resolved in a single LLM prompt. A single prompt suffers from:
1. **Context window saturation**: Trying to fit queries, broad search results, and long analyses into one inference pass leads to "lost in the middle" phenomena.
2. **Hallucination under uncertainty**: When an LLM lacks up-to-date domain evidence, it hallucinates plausible-sounding facts.
3. **Monolithic failure**: If a single prompt goes off-track, the entire output fails without modular recourse.

This baseline system decomposes the research workflow into specialized, single-responsibility agents coordinated via a directed state graph (**LangGraph**).

> **Important Architectural Scope:**
> This baseline is strictly a **feed-forward multi-agent pipeline**:
> `Planner -> Parallel Researchers -> Synthesizer -> Writer -> Final Report`.
> Critic nodes and iterative revision loops are intentionally excluded in this baseline phase and will form the Phase 2 comparative milestone.

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
                     |             Final Output              |
                     +---------------------------------------+
```
---

## 2. The Shared State (`state.py`)

### (a) What Problem This Layer Solves
In distributed multi-agent systems, agents need a common medium to communicate without point-to-point network coupling. If agents passed messages directly to each other, the pipeline would degenerate into an unmaintainable $O(N^2)$ mesh.

### (b) The First Principle Behind It
**The Blackboard Architecture Pattern**: State is a centralized, immutable-at-rest data store. An agent node is a pure function or state transformer:
$$\text{node}: \text{State} \to \Delta\text{State}$$
Each agent reads a snapshot of the blackboard, performs specialized compute, and writes back a differential patch ($\Delta\text{State}$). LangGraph uses state reducers to merge parallel updates deterministically.

### (c) What Goes In and What Comes Out
- **Input**: The initial dictionary `{ "question": "...", "metadata": {...} }`.
- **Transitions**: As each node executes, it produces localized key-value deltas.
- **Output**: The fully enriched state containing the complete research lineage (plan, search results, notes, brief, final report, and runtime telemetry).

### (d) State Architecture Table & Diagram

| Field | Written By | Read By | Why It Exists |
|---|---|---|---|
| `question` | User / Entry point | Planner, Writer, Judge | Ground-truth research question anchor. |
| `plan` | Planner Agent | Fan-out dispatcher, Synthesizer, Writer | List of 3-5 sub-questions decomposing the topic. |
| `search_results` | Researcher Agents | Researchers, Synthesizer | Raw web query hits `{title, url, snippet}` preserving source provenance. |
| `notes` | Researcher Agents | Synthesizer | Bullet-point findings with tagged source URLs extracted per sub-question. |
| `brief` | Synthesizer Agent | Writer Agent | Deduplicated synthesis and identified cross-source contradictions. |
| `report` | Writer Agent | User / CLI Output / Judge | Final academic markdown report with numbered citations `[1]`, `[2]`. |
| `metadata` | All nodes (timing wrapper) | Demo trace logger, Evaluator | Telemetry: node latencies, token consumption, total search API calls. |

```
  +-------------------------------------------------------------------------+
  |                             RESEARCH STATE                              |
  |                                                                         |
  |  +-----------------+  +-----------------+  +-------------------------+  |
  |  | question: str   |  | plan: list[str] |  | search_results: dict    |  |
  |  +-----------------+  +-----------------+  +-------------------------+  |
  |                                                                         |
  |  +-----------------+  +-----------------+  +-------------------------+  |
  |  | notes: dict     |  | brief: str      |  | report: str             |  |
  |  +-----------------+  +-----------------+  +-------------------------+  |
  |                                                                         |
  |  +-------------------------------------------------------------------+  |
  |  | metadata: {node_latency, token_usage, search_count, ...}          |  |
  |  +-------------------------------------------------------------------+  |
  +-------------------------------------------------------------------------+
```
---

## 3. The Web Search Tool (`tools/search_tool.py`)

### (a) What Problem This Layer Solves
LLMs are statically trained models. When tasked with analyzing current events (such as 2026 technological developments, market shifts, or recent standards), their knowledge cutoff causes either catastrophic hallucinations or refusal. The search tool bridges the gap between pre-trained weights and live external truth.

### (b) The First Principle Behind It
**Non-Parametric Grounding via Information Retrieval**:
Knowledge in deep learning exists in two forms:
1. *Parametric memory*: Implicit weights learned during pre-training ($\theta$).
2. *Non-parametric memory*: Explicit external documents fetched at inference ($D = \{d_1, d_2, \dots, d_k\}$).
By injecting non-parametric web snippets into the model's working memory (context window), the conditional generation shifts from unconstrained imagination $P(y | x; \theta)$ to evidence-conditioned generation $P(y | x, D; \theta)$.

### (c) What Goes In and What Comes Out
- **Input**: `query` (a targeted sub-question string) and `max_results` (integer, default 5).
- **Processing**: DuckDuckGo API query with exponential backoff retries ($1.5^t$ seconds) and resilient empty-result handling.
- **Output**: List of dictionaries:
  ```json
  [
    {"title": "Lithium vs Sodium Grid Storage 2026", "url": "https://...", "snippet": "..."}
  ]
  ```

### (d) Search Architecture Diagram

```
                 +-----------------------------------------+
                 | Sub-Question String: "sodium-ion costs" |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |  tools/search_tool.py: search_web()     |
                 |  - DDGS Query Engine                    |
                 |  - Retry Loop (3 attempts + backoff)    |
                 |  - Error Trapping & Fallback Resilience |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 | [ {title: "...", url: "...", snippet} ] |
                 +-----------------------------------------+
```
---

## 4. The Planner Agent (`agents/planner.py`)

### (a) What Problem This Layer Solves
Broad, multi-faceted research topics (e.g., comparing novel battery chemistries or analyzing macroeconomic tech hiring trends) are too diffuse for an information retrieval system if submitted as a single monolithic query. If a search engine is handed a paragraph-long query, keyword collisions degrade precision. The planner breaks the macroscopic inquiry into 3-5 microscopic, answerable vectors.

### (b) The First Principle Behind It
**Prompt as a Functional Specification & Next-Token Constrained Search**:
An LLM is fundamentally a probability distribution over token sequences:
$$P(w_1, w_2, \dots, w_T | \text{Prompt})$$
A system prompt acts as a soft program constraining the distribution's support. By commanding `STRICT JSON` format, we force the model's high-dimensional reasoning into a structured, typed contract. If syntactic deviations occur (e.g. conversational preamble or unclosed braces), a one-shot JSON-repair retry forces the model to attend to its own structural violation and output a valid parseable AST.

### (c) What Goes In and What Comes Out
- **Input**: `state["question"]` (e.g., `"Compare lithium-ion vs sodium-ion batteries for grid storage as of 2026"`).
- **Processing**: Structured system instruction $\to$ LLM generation $\to$ JSON parser $\to$ Optional one-shot repair.
- **Output**: Delta update `{"plan": [sub_q_1, sub_q_2, sub_q_3, ...], "metadata": {"tokens_planner": N}}`.

### (d) Planner Flow Diagram

```
                 +-----------------------------------------+
                 | Research Topic: "Lithium vs Sodium..."  |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |   Prompt: Decompose into 3-5 Sub-Q's    |
                 |   Schema: {"sub_questions": [...]}      |
                 +-----------------------------------------+
                                      |
                                      v
                             +-----------------+
                             | Valid JSON AST? |
                             +-----------------+
                                    /     \
                            [Yes]  /       \ [No]
                                  /         \
                                 |       +-------------------------+
                                 |       | 1-Shot JSON Repair LLM  |
                                 |       +-------------------------+
                                 |                  |
                                 v                  v
                 +-----------------------------------------+
                 |  state["plan"] = [Sub-Q1, Sub-Q2, ...]  |
                 |  Terminal Banner: [1] PLANNER OUTPUT    |
                 +-----------------------------------------+
```
---

## 5. The Researcher Agent & Parallel Fan-Out (`agents/researcher.py`)

### (a) What Problem This Layer Solves
If an agent searches for all sub-questions sequentially in a single loop, total latency is $O(\sum t_i)$. Furthermore, stuffing search results for 4 different topics into a single prompt creates cross-topic interference and dilutes the model's attention. Parallel fan-out executes each sub-question in an independent, bounded sandbox.

### (b) The First Principle Behind It
**Map-Reduce Parallelism & Bounded Working Context**:
1. *Amdahl's Law / Parallel Speedup*: By distributing independent retrieval tasks across concurrent worker threads, wall-clock latency drops from $\sum_{i=1}^k T_i$ to $\max(T_i) + \epsilon$.
2. *Context Window Signal-to-Noise Ratio*: In a standard transformer attention mechanism:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   When the key-value sequence length $N$ grows large with irrelevant tokens from other topics, the attention probability mass assigned to the pertinent facts decreases. By scoping the researcher to a single sub-question, the context length remains minimal, maximizing retrieval precision.

### (c) What Goes In and What Comes Out
- **Input**: A single `sub_question` string from the planner's output list.
- **Processing**: Web search API query $\to$ snippet extraction $\to$ grounded bullet-point extraction LLM call $\to$ source URL tagging.
- **Output**: Delta update:
  ```python
  {
      "search_results": {sub_question: [hit1, hit2, ...]},
      "notes": {sub_question: ["• Fact A [https://...]", "• Fact B [https://...]"]},
      "metadata": {"search_count": 1, "tokens_researcher": N}
  }
  ```
- **State Merge**: LangGraph's dictionary reducer seamlessly combines concurrent deltas into the shared state.

### (d) Fan-Out Architecture Diagram

```
                              state["plan"]
                         [Sub-Q 1, Sub-Q 2, Sub-Q 3]
                                     |
               +---------------------+---------------------+
               |                     |                     |
               v                     v                     v
      +-----------------+   +-----------------+   +-----------------+
      | Researcher 1    |   | Researcher 2    |   | Researcher 3    |
      | - search_web(Q1)|   | - search_web(Q2)|   | - search_web(Q3)|
      | - LLM extract   |   | - LLM extract   |   | - LLM extract   |
      | - Tag [URL1]    |   | - Tag [URL2]    |   | - Tag [URL3]    |
      +-----------------+   +-----------------+   +-----------------+
               |                     |                     |
               +---------------------+---------------------+
                                     |
                                     v
                       (LangGraph merge_dicts reducer)
                                     v
                           state["notes"] = {
                             Q1: [bullet1, bullet2],
                             Q2: [bullet3, bullet4],
                             Q3: [bullet5, bullet6]
                           }
```
---

## 6. The Synthesizer Agent (`agents/synthesizer.py`)

### (a) What Problem This Layer Solves
Parallel researchers generate isolated sets of notes. Passing raw notes directly to a report writer produces a disjointed, repetitive, and inconsistent document. Multiple sources frequently disagree on statistics, cost estimates, or timelines. The synthesizer acts as a centralized editorial funnel that reconciles inconsistencies and eliminates duplicates before drafting begins.

### (b) The First Principle Behind It
**Information Entropy Reduction & Cross-Verification**:
Let raw incoming notes be an unordered set $N = \{n_1, n_2, \dots, n_m\}$. Redundancy creates superfluous tokens that crowd the LLM's attention span. Furthermore, unchecked disagreements introduce epistemic instability.
The synthesizer computes a condensed projection:
$$\text{Brief} = \text{Synthesize}(N)$$
where:
1. Redundant propositions $n_i \approx n_j$ are collapsed into a single canonical assertion.
2. Contradictory propositions $n_i \oplus n_j$ (e.g. $40/kWh vs. $75/kWh) are explicitly flagged as divergent perspectives rather than silently picked at random.

### (c) What Goes In and What Comes Out
- **Input**: `state["question"]`, `state["plan"]`, and `state["notes"]` (dictionary of all researcher notes tagged with URLs).
- **Processing**: Deduplication, contradiction identification, thematic clustering, and source indexing.
- **Output**: Delta update `{"brief": "...", "metadata": {"tokens_synthesizer": N}}`.

### (d) Synthesizer Architecture Diagram

```
                 +-----------------------------------------+
                 |       Aggregated Parallel Notes         |
                 |  {Q1: [notes...], Q2: [notes...], ...}  |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       agents/synthesizer.py             |
                 |  1. Deduplicate Identical Claims        |
                 |  2. Detect Source Contradictions        |
                 |  3. Thematic Clustering                 |
                 |  4. Source Deduplication & Indexing     |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |           Structured Brief              |
                 |  - Executive Synthesis                  |
                 |  - Thematic Sub-Question Summaries      |
                 |  - Flagged Contradictions               |
                 |  - Consolidated Source URLs             |
                 +-----------------------------------------+
                                      |
                                      v
                 |  Terminal Banner: [3] SYNTHESIZER BRIEF |
```
---

## 7. The Writer Agent (`agents/writer.py`)

### (a) What Problem This Layer Solves
Raw notes and synthesis briefs are fragmented outlines. Consumers of academic research require structured, readable monographs that adhere to scientific publication standards. Critically, if an LLM is asked to write without strict attribution constraints, it invents citations and misattributes quotes. The Writer enforces strict citation grounding: every claim must cite a numbered index mapping to a verifiable source.

### (b) The First Principle Behind It
**Attribution & Provenance Traceability**:
In academic literature, an assertion without citation carries zero evidentiary weight. In LLM terms:
$$\text{Claim} \xrightarrow{\text{bind}} \text{Citation ID } [k] \xrightarrow{\text{resolve}} \text{URL}_k$$
By providing the writer agent with a pre-indexed catalogue of verified URLs and strictly mandating numbered bracket citations `[1]`, `[2]`, the model binds generation tokens directly to retrieved evidence indices. This achieves auditability: any reader or automated judge can trace any sentence back to its source snippet.

### (c) What Goes In and What Comes Out
- **Input**: `state["question"]`, `state["plan"]`, `state["brief"]`, and all gathered source URLs.
- **Processing**: Section-by-section document composition adhering to the sub-question structure, followed by citation resolution and markdown file persistence.
- **Output**: Delta update `{"report": "...", "metadata": {"tokens_writer": N, "report_path": "outputs/report_..."}}`.

### (d) Writer Architecture Diagram

```
                 +-----------------------------------------+
                 | Synthesizer Brief & Deduplicated URLs   |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |          agents/writer.py               |
                 |  - Title & Introduction                 |
                 |  - One dedicated section per Sub-Q      |
                 |  - In-text numbered citations [1], [2]  |
                 |  - Synthesis & Strategic Conclusion     |
                 |  - References bibliography list         |
                 +-----------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        +--------------------------+     +--------------------------+
        | outputs/report_<ts>.md   |     | Terminal Banner:         |
        | (Persisted Artifact)     |     | [4] FINAL REPORT         |
        +--------------------------+     +--------------------------+
```
---

## 8. Graph Orchestration & Execution Flow (`graph/builder.py`)

### (a) What Problem This Layer Solves
Agent systems quickly fail if logic is scattered across ad-hoc loops and callbacks. Complex multi-agent topologies need:
1. Strict stage barriers (the synthesizer must not execute until *all* parallel researchers have returned).
2. Transparent state transitions.
3. Checkpointing and observability so runs can be paused, inspected, and audited.

### (b) The First Principle Behind It
**Finite State Machines (FSM) & Directed Acyclic Graphs (DAG)**:
LangGraph formulates agentic compute as an explicit state machine:
$$G = (V, E, S, \delta)$$
where:
- $V = \{\text{planner}, \text{researcher}, \text{synthesizer}, \text{writer}\}$ are computational vertices.
- $S$ is the shared `ResearchState`.
- $E$ are directed transitions.
- $\delta: S \to \mathcal{P}(V)$ is the conditional routing function.
When the planner emits $k$ sub-questions, LangGraph evaluates the dynamic branch using the `Send` abstraction:
$$\delta(S) = \{\text{Send}(\text{researcher}, q_i) \mid q_i \in \text{plan}\}$$
The runtime manages execution barriers (supersteps), waiting until all $k$ parallel worker promises resolve before transitioning to the `synthesizer` vertex.

### (c) What Goes In and What Comes Out
- **Input**: Initial input state dict `{"question": "...", "metadata": {}}` and a runnable `config={"configurable": {"thread_id": "..."}}`.
- **Processing**: Automated step-by-step state transition loop with checkpoint snapshots persisted via `MemorySaver`. Every node is wrapped with high-resolution latency instrumentation.
- **Output**: Fully populated terminal state object.

### (d) Graph Architecture Diagram

```
                             [START]
                                |
                                v
                       +-----------------+
                       |     planner     |
                       +-----------------+
                                |
                +---------------+---------------+
                | (LangGraph Send Map Fan-Out)  |
                v               v               v
        +--------------+ +--------------+ +--------------+
        |  researcher  | |  researcher  | |  researcher  |
        | (Sub-Q 1)    | | (Sub-Q 2)    | | (Sub-Q 3)    |
        +--------------+ +--------------+ +--------------+
                |               |               |
                +---------------+---------------+
                                | (Fan-In Barrier: merge_dicts)
                                v
                       +-----------------+
                       |   synthesizer   |
                       +-----------------+
                                |
                                v
                       +-----------------+
                       |     writer      |
                       +-----------------+
                                |
                                v
                              [END]
```
---

## 9. Execution Pipeline & Trace Logging (`main.py`)

### (a) What Problem This Layer Solves
Autonomous agent systems often operate as "black boxes": a user submits a prompt, waits 45 seconds with no feedback, and receives an opaque output. If an error or hallucination occurs, diagnosing where the breakdown occurred is impossible without intermediate visibility. The CLI pipeline solves this via real-time banners and persistent audit trace logs.

### (b) The First Principle Behind It
**Observability & Scientific Reproducibility**:
In empirical computer science, a system's output must be deterministically traceable back through each intermediate state transition. The trace logger implements a **Tee Stream Pattern**:
$$\text{Stream}(\text{stdout}) \to \{\text{TTY Terminal}, \text{Persistent Append-Only Disk Log}\}$$
This guarantees that the exact sub-questions decomposed by the planner, the specific search snippets pulled by each researcher, the synthesis brief, and the final paper are committed to an immutable log file (`outputs/demo_trace_<timestamp>.txt`) as live evidence for examiners.

### (c) What Goes In and What Comes Out
- **Input**: Command-line topic argument (e.g., `python main.py "Compare lithium-ion vs sodium-ion batteries..."`).
- **Processing**: Initializes Tee stream, starts global timers, triggers LangGraph invocation, formats banners as events fire, records total tokens, latencies, and search counts.
- **Output**: Terminal streaming display + `outputs/report_<timestamp>.md` + `outputs/demo_trace_<timestamp>.txt`.

### (d) Pipeline Trace Logging Diagram

```
                 +-----------------------------------------+
                 | CLI: python main.py "<topic>"           |
                 +-----------------------------------------+
                                      |
                                      v
                 +-----------------------------------------+
                 |       TeeStream Stdout Splitter         |
                 +--------------------+--------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        +--------------------------+     +--------------------------+
        | Live Terminal Display    |     | outputs/demo_trace_*.txt |
        | [1] Planner Banner       |     | (Complete unedited       |
        | [2] Researcher Banners   |     |  lineage preserved       |
        | [3] Synthesizer Banner   |     |  for thesis defense)     |
        | [4] Final Report Banner  |     +--------------------------+
        +--------------------------+
                     |
                     v
        +-----------------------------------------------------------+
        | SUMMARY: Report: <path> \| Latency: <T>s \| Tokens: <N>...  |
        +-----------------------------------------------------------+
```
---

## 10. Evaluation Harness & LLM-as-a-Judge (`eval/`)

### (a) What Problem This Layer Solves
Subjective manual reading of generated academic reports cannot establish scientific progress across prompt versions or multi-agent architectures. Without standardized evaluation benchmarks and automated scoring rubrics, it is impossible to quantitatively determine whether changes in graph topology or retrieval parameters improve quality or introduce regressions.

### (b) The First Principle Behind It
**Standardized Meta-Evaluation & G-Eval Rubric Scoring**:
LLM-as-a-Judge uses an independent model evaluator conditioned on an explicit multi-dimensional rubric $R = \{r_{\text{rel}}, r_{\text{comp}}, r_{\text{org}}, r_{\text{ground}}\}$:
1. **Relevance** ($r_{\text{rel}} \in [1, 5]$): Direct alignment with the original research prompt and sub-questions.
2. **Completeness** ($r_{\text{comp}} \in [1, 5]$): Breadth of technical, economic, and lifecycle trade-offs addressed.
3. **Organization** ($r_{\text{org}} \in [1, 5]$): Cohesive narrative progression and section hierarchy.
4. **Grounding** ($r_{\text{ground}} \in [1, 5]$): Strict presence of numbered in-text citations mapping to verifiable reference URLs.

The evaluation runner automates batch runs, capturing latency, token count, and retrieval volume into append-only JSONL files (`outputs/eval_runs.jsonl`), while the judge records formal scores and justifications to CSV (`outputs/eval_results.csv`).

### (c) What Goes In and What Comes Out
- **Input**:
  - `eval/questions.py`: 10 diverse benchmark questions across technological domains.
  - `eval/runner.py`: Executes the graph and logs telemetry.
  - `eval/judge.py`: Ingests `(question, report_path)` and prompts the referee model.
- **Output**:
  - `outputs/eval_runs.jsonl`: Execution traces `{question, report_path, latency, tokens, searches}`.
  - `outputs/eval_results.csv`: Rubric scores and one-line justifications.

### (d) Evaluation Harness Diagram

```
         +----------------------------------------------------+
         |  eval/questions.py: 10 Diverse Benchmark Prompts   |
         +----------------------------------------------------+
                                   |
                                   v
         +----------------------------------------------------+
         |  eval/runner.py: Batch Graph Execution Harness     |
         +-------------------------+--------------------------+
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
          +---------------------+     +---------------------+
          | outputs/report_*.md |     | outputs/            |
          | (Generated Reports) |     | eval_runs.jsonl     |
          +---------------------+     +---------------------+
                     |
                     v
         +----------------------------------------------------+
         |  eval/judge.py: LLM-as-a-Judge Rubric Engine       |
         |  - Relevance (1-5)      - Completeness (1-5)       |
         |  - Organization (1-5)   - Grounding (1-5)          |
         +----------------------------------------------------+
                                   |
                                   v
         +----------------------------------------------------+
         |  outputs/eval_results.csv: Final Score Table       |
         +----------------------------------------------------+
```
---

## 11. MSc Review Defense Guide & Oral Examination Q&A

### Question 1: Why does this system NOT include a Critic or Revision Loop?
**First-Principles Answer**:
In experimental computer science, establishing a baseline requires isolating feed-forward retrieval and generation performance before introducing recurrent feedback loops. A critic/revision loop introduces non-monotonic convergence, variable step counts, and non-deterministic stopping criteria. By bounding Phase 1 strictly to a deterministic feed-forward DAG ($G$: Planner $\to$ Parallel Researchers $\to$ Synthesizer $\to$ Writer), we benchmark the raw precision and latency lower-bound. Phase 2 will introduce an iterative critic/reflection node to quantify the exact marginal gain of cyclic verification against this benchmark.

### Question 2: Why not just use one prompt with a 1-million-token context window?
**First-Principles Answer**:
Even with large context windows, the transformer attention mechanism suffers from attention dispersion and the "Lost in the Middle" phenomenon ($O(N^2)$ cross-attention pairwise dot-products). Pumping multiple queries into a single prompt induces semantic bleed and cross-topic interference. Our parallel fan-out architecture restricts each researcher's attention computation to a single orthogonal sub-question, maximizing the signal-to-noise ratio and enabling horizontal scaling across worker threads.

### Question 3: How does the system handle hallucinated citations?
**First-Principles Answer**:
Through strict provenance binding. Raw search snippets carry canonical source URLs. Bullet notes extracted by researchers are tagged with those URLs. The writer agent is provided with an explicit indexed source catalog and constrained to use bracketed numerical citations $[k]$ that map one-to-one with the reference list. The evaluation judge automatically penalizes reports where claims lack corresponding citations.

### Question 4: How does LangGraph handle concurrent state updates without race conditions?
**First-Principles Answer**:
Via the Blackboard Reducer pattern (`merge_dicts`). Instead of allowing parallel worker nodes to overwrite state in shared memory, LangGraph treats state transitions as functional patches. When $k$ researcher nodes finish concurrently, LangGraph applies the registered reducer to combine the dictionaries deterministically into the master `ResearchState`.

### Question 5: What is the primary bottleneck in this pipeline and how is it mitigated?
**First-Principles Answer**:
The primary bottleneck is external I/O latency from web retrieval and LLM inference. In a sequential design, total latency is $O(\sum_{i=1}^k T_i)$. By utilizing LangGraph's dynamic `Send` mapping, all sub-question retrievals occur in parallel, reducing execution time to $\max(T_i) + \text{synthesis latency}$, directly exploiting Amdahl's Law.

---
