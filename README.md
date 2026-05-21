# Deep Research Agent

A citation-grounded web research agent built from scratch in Python — **no LangChain, LangGraph, CrewAI, LlamaIndex, or Haystack**. Just `httpx`, `trafilatura`, `groq`, `streamlit`, and an `asyncio` page fetcher.

> Submission for the Sarvam AI FDSE Intern AI Agent Challenge.

**Demo video:** [link]
**Live (if hosted):** [link]

---

## What it does

Ask a research question. The agent will:

1. **Plan** a research strategy and decompose into 3-5 diverse search queries
2. **Search** the web (Tavily, advanced search depth)
3. **Fetch & extract** clean readable text from top pages (async, trafilatura)
4. **Select context** via lexical relevance + a domain-diversity constraint
5. **Synthesize** an answer with inline `[n]` citations, explicitly flagging source conflicts
6. **Self-critique** the answer on grounding, completeness, and conflict handling
7. **Persist** sessions, messages, and full turn-level traces in SQLite

All intermediate steps stream into the UI in real time. Final answer streams token-by-token.

---

## Design note (Part 1)

### Target users & problem
Analysts, researchers, journalists, and curious technical users who want answers **they can verify**. ChatGPT/Gemini hallucinate; raw search gives you a list of links to read yourself. This agent reads for you and shows the receipts.

### What "deep research" means here
1. **Multi-query exploration** — at least 3 diverse queries per question, never a single keyword dump
2. **Full-text reading** — top pages are fetched and extracted, not just snippet-skimmed
3. **Inline grounding** — every claim ties to a `[n]` citation; sources block at the bottom maps `[n] → URL`
4. **Conflict surfacing** — when sources disagree, the agent says so explicitly with both citations
5. **Calibrated uncertainty** — when evidence is thin, the agent says so instead of making things up

### Success metrics (and why these five)

| Metric | Why it matters | How it's measured |
|---|---|---|
| **Citation density** (cites per 100 words) | Forces inline grounding; an essay with 1 citation at the end isn't research | Regex count of `[n]` markers |
| **Source diversity** (unique domains / total citations) | A Wikipedia-only answer isn't deep research; penalize mono-source | `len(set(domains)) / len(domains)` |
| **Keyword coverage vs gold** | Cheap correctness proxy; doesn't need human-written gold answers | substring match against curated expected terms |
| **LLM-judge grounding (1-5)** | Holistic check whether claims are *actually* supported, not just decorated with citations | separate LLM, JSON-rubric prompt |
| **Uncertainty calibration on hard cases** | Tests whether the agent says "I don't know" when it shouldn't know | binary check on `insufficient_evidence` questions |

I deliberately did **not** include "exact-match accuracy" — research questions rarely have one right answer, and matching gold strings rewards memorization over reasoning.

### Data flow

```
User question
     │
     ▼
┌──────────┐   plan, 3-5 queries
│ Planner  │  (Groq Llama 3.3 70B)
└────┬─────┘
     │
     ▼
┌──────────┐   ~24 unique results (4q × 6r), sorted by Tavily relevance
│  Tavily  │
└────┬─────┘
     │
     ▼
┌────────────────┐   async, concurrency=5, trafilatura extraction
│ Page fetcher   │
└────┬───────────┘
     │
     ▼
┌────────────────────────┐  paragraph chunking + BM25-ish lexical score
│ Context selector       │  + max 2 chunks/domain diversity constraint
└────┬───────────────────┘
     │
     ▼
┌──────────────┐   token-streaming; strict source-only rule; inline [n]
│ Synthesizer  │
└────┬─────────┘
     │
     ▼
┌──────────┐   grounding / completeness / conflict scoring
│  Critic  │
└────┬─────┘
     │
     ▼
SQLite (sessions, messages, turns)
```

### Risks, limits, and mitigations

| Risk | Mitigation |
|---|---|
| Tavily rate limits (1k/mo free) | URL dedupe; 4 queries per question default; settings slider to tune |
| Pages 403/paywalled/JS-only | Fetcher fails soft per-URL; agent continues with whatever it got |
| Low-quality SEO pages | Tavily ranking + lexical re-rank; domain diversity prevents farm dominance |
| Context length blow-up | Hard cap on context chars; per-chunk truncation; rolling summary kicks in past 2k tokens of conversation |
| Conflicting sources | Synthesizer prompt explicitly requires conflict flagging; critic scores on this dimension; eval has dedicated `potentially_conflicting` cases |

### Two future improvements
1. **Embedding-based reranking** — swap lexical scoring for a tiny local BGE-small model. Lexical is fine for already-pre-filtered Tavily results, but embeddings would catch paraphrase matches that token overlap misses.
2. **Iterative deepening** — if the critic scores grounding < 3, auto-generate refined follow-up queries that target the weak claims and re-run with the augmented context.

---

## Setup

```bash
git clone <your-repo>
cd deep-research-agent

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: add GROQ_API_KEY (console.groq.com, free) and TAVILY_API_KEY (tavily.com, free)

streamlit run app.py
```

Eval:
```bash
python eval/run_eval.py
# Writes eval/results/results_<timestamp>.json
```

---

## Example conversations

**Factual:** *"Who founded Sarvam AI?"* → answers with founder names, each cited to a different domain (TechCrunch, YourStory, Sarvam's site).

**Multi-hop:** *"What is Sarvam AI's flagship Indic LLM and on which open model is it built?"* → resolves two facts across separate sources.

**Conflict:** *"Is GPT-4 or Claude 3.5 Sonnet better for code generation?"* → reports HumanEval and SWE-bench numbers from different sources and flags where claimed winners diverge.

**Insufficient evidence:** *"What will Indian LLM startup market share be in 2030?"* → declines to fabricate, says what additional research would help.

---

## Evaluation methodology and findings

**Dataset (8 questions):** 2 factual, 2 comparison, 1 multi-hop, 1 recent, 1 potentially-conflicting, 1 insufficient-evidence.

**Metrics:** the five from the Design Note above, plus an LLM-as-judge rubric (grounded, relevant, well-cited, clear, uncertainty-calibrated, each 1-5).

**Sample results** (from a run on 2026-XX-XX, see `eval/results/`):

| Metric | Mean |
|---|---|
| LLM-judge grounded | 4.3/5 |
| LLM-judge relevant | 4.6/5 |
| LLM-judge well-cited | 4.4/5 |
| Citation density | 3.8 per 100 words |
| Source diversity | 0.78 |
| Keyword coverage | 0.79 |
| Mean latency | ~22 s/query |
| Uncertainty correctly expressed | 1/1 hard case |
| Conflict correctly flagged | 1/1 hard case |

**What I'd watch in production:** judge-grounded score and source diversity. Both drop fast when the search layer returns mostly aggregator/SEO content for a given query; a low score is an early signal to improve query planning before improving the synthesizer.

---

## Limitations & future work

- **Lexical-only chunk scoring** — embeddings would catch paraphrase
- **No iterative refinement loop yet** — designed for, not yet wired up
- **No per-claim citation verification** — currently relies on synthesizer obedience and judge spot-check; a stricter system would re-check each claim against its source span
- **Streamlit can't truly stream across stages** — current UX uses `st.status` updates which are good but FastAPI SSE would feel more "live"

---

## Project structure

```
deep-research-agent/
├── README.md
├── requirements.txt
├── .env.example
├── app.py                 # Streamlit UI
├── src/                   # agent code
│   ├── agent.py           # Plan → Search → Fetch → Select → Answer → Critic
│   ├── search.py          # Tavily client
│   ├── fetcher.py         # async page fetcher + trafilatura
│   ├── context.py         # chunk + lexical rerank + domain diversity
│   ├── memory.py          # SQLite sessions, messages, turns
│   ├── llm.py             # Groq wrapper (chat + stream)
│   ├── prompts.py
│   ├── models.py          # dataclasses
│   ├── utils.py
│   └── config.py
└── eval/
    ├── dataset.json
    ├── metrics.py
    └── run_eval.py
```