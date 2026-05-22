"""Tavily client. Could swap for Serper by changing this file alone."""
import os
from typing import List

import httpx

from .models import SearchResult


class TavilySearch:
    BASE = "https://api.tavily.com/search"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ["TAVILY_API_KEY"]

    def search(self, query: str, max_results: int = 6,
               include_raw: bool = False) -> List[SearchResult]:
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_raw_content": include_raw,
            "include_answer": False,
        }
        try:
            r = httpx.post(self.BASE, json=payload, timeout=30)
            r.raise_for_status()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as e:
            print(f"[search] error for '{query[:60]}': {e}")
            return []

        data = r.json()
        out = []
        for item in data.get("results", []):
            out.append(SearchResult(
                title=item.get("title", "") or "",
                url=item.get("url", ""),
                snippet=item.get("content", "") or "",
                score=item.get("score"),
                raw_content=item.get("raw_content") if include_raw else None,
            ))
        return out