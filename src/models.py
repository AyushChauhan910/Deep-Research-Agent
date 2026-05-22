from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SearchResult:
    """A single result returned by the search client."""
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    raw_content: Optional[str] = None


@dataclass
class FetchedPage:
    """Full extracted text for one successfully fetched URL."""
    url: str
    title: str
    text: str
    domain: str
    fetched_at: str  # UTC ISO-8601, e.g. "2024-05-01T12:00:00Z"


@dataclass
class ContextChunk:
    """A paragraph-sized slice of a FetchedPage, scored for query relevance."""
    text: str
    url: str
    title: str
    domain: str
    chunk_idx: int
    relevance: float = 0.0  # BM25-style lexical score; higher is more relevant