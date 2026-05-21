"""The agent loop. No framework — just Python, a callback for events, and four LLM calls.

Flow: Plan → Search → Fetch → Select Context → Synthesize → Critic → Persist.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import (
    DEFAULT_MAX_CHUNKS, DEFAULT_MAX_PAGES, DEFAULT_MAX_QUERIES,
    DEFAULT_RESULTS_PER_QUERY, MAX_CONTEXT_CHARS, SUMMARIZE_THRESHOLD_TOKENS,
)
from .context import chunk_pages, select_context
from .fetcher import fetch_many
from .llm import LLM
from .memory import (add_message, get_messages, get_summary, save_turn,
                     update_summary)
from .prompts import (CRITIC_PROMPT, PLANNER_PROMPT, SUMMARIZE_PROMPT,
                      SYNTHESIZER_PROMPT)
from .search import TavilySearch
from .utils import approx_tokens, extract_json, truncate


@dataclass
class Event:
    kind: str
    message: str
    data: dict = field(default_factory=dict)


class DeepResearchAgent:
    def __init__(
        self,
        llm: LLM,
        search_client: TavilySearch,
        max_search_queries: int = DEFAULT_MAX_QUERIES,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
        max_pages_to_fetch: int = DEFAULT_MAX_PAGES,
        max_context_chunks: int = DEFAULT_MAX_CHUNKS,
    ):
        self.llm = llm
        self.search = search_client
        self.k_queries = max_search_queries
        self.k_results = results_per_query
        self.k_fetch = max_pages_to_fetch
        self.k_chunks = max_context_chunks

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _emit(cb, kind: str, msg: str, data: dict = None):
        if cb:
            cb(Event(kind, msg, data or {}))

    # -- stages ----------------------------------------------------------

    def plan(self, query: str, summary: str = "") -> dict:
        prompt = PLANNER_PROMPT.format(
            query=query, context=summary or "(no prior context)"
        )
        raw = self.llm.chat([{"role": "user", "content": prompt}],
                            temperature=0.4, max_tokens=600)
        parsed = extract_json(raw)
        if not parsed or "queries" not in parsed or not parsed["queries"]:
            # fallback so we never crash on a bad LLM JSON
            parsed = {"plan": "Direct search.", "queries": [query]}
        parsed["queries"] = parsed["queries"][: self.k_queries]
        parsed["plan"] = parsed.get("plan", "")
        return parsed

    def search_all(self, queries: List[str]):
        merged, seen = [], set()
        for q in queries:
            for r in self.search.search(q, max_results=self.k_results):
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    merged.append(r)
        merged.sort(key=lambda r: (r.score or 0), reverse=True)
        return merged

    def fetch(self, results, top_n: int):
        urls = [r.url for r in results[:top_n]]
        pages = fetch_many(urls, concurrency=5)
        # backfill titles from search if extractor missed them
        title_lookup = {r.url: r.title for r in results}
        for p in pages:
            if not p.title:
                p.title = title_lookup.get(p.url, p.domain)
        return pages

    def build_context(self, query: str, pages):
        all_chunks = chunk_pages(pages)
        return select_context(query, all_chunks,
                              max_chunks=self.k_chunks,
                              max_per_domain=2)

    def synthesize(self, query: str, chunks, summary: str = "",
                   on_token: Optional[Callable] = None) -> str:
        # Pack sources, but enforce a total context cap so we don't overrun.
        parts, used = [], 0
        for i, c in enumerate(chunks, 1):
            block = (f"\n[{i}] {c.title} ({c.domain})\nURL: {c.url}\n"
                     f"Excerpt: {c.text}\n")
            if used + len(block) > MAX_CONTEXT_CHARS:
                # truncate this chunk to fit, then stop
                remaining = MAX_CONTEXT_CHARS - used
                if remaining > 400:
                    parts.append(truncate(block, remaining))
                break
            parts.append(block)
            used += len(block)
        sources_text = "".join(parts)

        prompt = SYNTHESIZER_PROMPT.format(
            query=query,
            summary=summary or "(none)",
            sources=sources_text,
        )

        if on_token:
            full = []
            for tok in self.llm.stream([{"role": "user", "content": prompt}],
                                       temperature=0.2, max_tokens=1500):
                full.append(tok)
                on_token(tok)
            return "".join(full)
        return self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=0.2, max_tokens=1500)

    def critique(self, query: str, answer: str, n_sources: int) -> dict:
        prompt = CRITIC_PROMPT.format(query=query, answer=answer, n_sources=n_sources)
        raw = self.llm.chat([{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=300)
        return extract_json(raw) or {}

    def maybe_update_summary(self, session_id: str):
        msgs = get_messages(session_id)
        total = sum(approx_tokens(m["content"]) for m in msgs)
        if total < SUMMARIZE_THRESHOLD_TOKENS:
            return
        conv = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
        prompt = SUMMARIZE_PROMPT.format(conversation=conv[-8000:])
        summary = self.llm.chat([{"role": "user", "content": prompt}],
                                temperature=0.2, max_tokens=300)
        update_summary(session_id, summary)

    # -- main entrypoint -------------------------------------------------

    def run(self, query: str, session_id: str,
            on_event: Optional[Callable] = None,
            on_token: Optional[Callable] = None) -> Dict:
        t0 = time.time()
        add_message(session_id, "user", query)
        summary = get_summary(session_id)

        # 1. PLAN
        self._emit(on_event, "plan_start", "Planning research strategy…")
        plan = self.plan(query, summary)
        self._emit(on_event, "plan_done",
                   f"Plan ready · {len(plan['queries'])} search queries",
                   {"plan": plan["plan"], "queries": plan["queries"]})

        # 2. SEARCH
        self._emit(on_event, "search_start", "Searching the web…")
        results = self.search_all(plan["queries"])
        self._emit(on_event, "search_done",
                   f"Found {len(results)} unique sources",
                   {"top_urls": [r.url for r in results[:8]]})

        # 3. FETCH
        self._emit(on_event, "fetch_start",
                   f"Reading top {min(self.k_fetch, len(results))} pages…")
        pages = self.fetch(results, top_n=self.k_fetch)
        self._emit(on_event, "fetch_done",
                   f"Extracted content from {len(pages)} pages",
                   {"fetched": [p.url for p in pages]})

        # 4. SELECT CONTEXT
        self._emit(on_event, "context_start", "Selecting relevant context…")
        chunks = self.build_context(query, pages)
        self._emit(on_event, "context_done",
                   f"Selected {len(chunks)} chunks from "
                   f"{len(set(c.domain for c in chunks))} domains")

        # 5. ANSWER
        self._emit(on_event, "answer_start",
                   "Synthesizing answer with citations…")
        if not chunks:
            answer = ("I couldn't find sufficient web evidence to answer this "
                      "confidently. Try rephrasing or adding context.")
        else:
            answer = self.synthesize(query, chunks, summary, on_token=on_token)

        # 6. CRITIC
        self._emit(on_event, "critic_start", "Self-checking grounding…")
        critique = self.critique(query, answer, len(chunks))
        self._emit(on_event, "critic_done", "Quality check complete",
                   {"critique": critique})

        # 7. PERSIST
        citations = [
            {"n": i + 1, "title": c.title, "url": c.url,
             "domain": c.domain, "relevance": c.relevance}
            for i, c in enumerate(chunks)
        ]
        elapsed = round(time.time() - t0, 2)
        save_turn(session_id, {
            "query": query,
            "search_queries": plan["queries"],
            "urls_opened": [p.url for p in pages],
            "context_snippets": [{"url": c.url, "text": c.text[:600]} for c in chunks],
            "answer": answer,
            "citations": citations,
            "critique": critique,
            "elapsed_s": elapsed,
        })
        add_message(session_id, "assistant", answer)
        self.maybe_update_summary(session_id)

        self._emit(on_event, "done", f"Done in {elapsed:.1f}s")

        return {
            "answer": answer,
            "citations": citations,
            "plan": plan,
            "critique": critique,
            "elapsed": elapsed,
            "n_sources": len(chunks),
            "n_pages_fetched": len(pages),
        }