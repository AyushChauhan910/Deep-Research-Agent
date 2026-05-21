from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: Optional[float] = None
    raw_content: Optional[str] = None


@dataclass
class FetchedPage:
    url: str
    title: str
    text: str
    domain: str
    fetched_at: str


@dataclass
class ContextChunk:
    text: str
    url: str
    title: str
    domain: str
    chunk_idx: int
    relevance: float = 0.0

    def to_dict(self):
        return asdict(self)