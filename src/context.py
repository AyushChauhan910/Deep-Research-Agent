"""Chunking + lexical relevance scoring + domain-diversity selection.

Lexical is a deliberate choice: no sklearn/embedding dependency keeps the
framework-free spirit honest, and BM25-ish overlap is surprisingly competitive
when chunks are already topically pre-filtered by Tavily.
"""
import re
from typing import List

from .config import DEFAULT_MAX_CHUNK_CHARS
from .models import ContextChunk, FetchedPage


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "and", "or", "for", "with", "by", "as", "it", "this", "that",
    "be", "but", "not", "from", "have", "has", "had", "i", "you", "we",
    "they", "he", "she", "do", "does", "did", "what", "which", "who",
    "how", "why", "when", "where",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9']+", text.lower()) if t not in _STOP]


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Paragraph-aware chunking with a hard char cap."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > max_chars:
            # split overly long paragraph on sentence boundaries
            sents = re.split(r"(?<=[.!?])\s+", p)
            for s in sents:
                if len(cur) + len(s) + 1 < max_chars:
                    cur += (" " if cur else "") + s
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
        elif len(cur) + len(p) + 2 < max_chars:
            cur += ("\n\n" if cur else "") + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_pages(pages: List[FetchedPage]) -> List[ContextChunk]:
    out = []
    for page in pages:
        for i, ch in enumerate(chunk_text(page.text)):
            out.append(ContextChunk(
                text=ch,
                url=page.url,
                title=page.title or page.domain,
                domain=page.domain,
                chunk_idx=i,
            ))
    return out


def _score(query_tokens: set, chunk_text_: str) -> float:
    """BM25-style relevance: hit count divided by sqrt(chunk length) to sub-linearly penalize long chunks."""
    c_toks = _tokenize(chunk_text_)
    if not c_toks:
        return 0.0
    hits = sum(1 for t in c_toks if t in query_tokens)
    return hits / (len(c_toks) ** 0.5)


def select_context(query: str, chunks: List[ContextChunk],
                   max_chunks: int = 8, max_per_domain: int = 2) -> List[ContextChunk]:
    """Rank by lexical relevance, then enforce domain diversity in the final pick."""
    q_toks = set(_tokenize(query))
    scored = [(_score(q_toks, c.text), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)

    picked, per_domain = [], {}
    for s, c in scored:
        if s <= 0:
            continue
        if per_domain.get(c.domain, 0) >= max_per_domain:
            continue
        c.relevance = round(s, 4)
        picked.append(c)
        per_domain[c.domain] = per_domain.get(c.domain, 0) + 1
        if len(picked) >= max_chunks:
            break
    return picked