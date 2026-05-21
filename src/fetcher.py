"""Async page fetching with readable-text extraction. Fails soft on every URL."""
import asyncio
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import httpx
import trafilatura

from .models import FetchedPage

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


async def _fetch_one(client: httpx.AsyncClient, url: str) -> Optional[FetchedPage]:
    try:
        r = await client.get(url, headers=HEADERS, timeout=15.0,
                             follow_redirects=True)
        if r.status_code != 200 or not r.text:
            return None
    except Exception:
        return None

    try:
        text = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    except Exception:
        text = None

    if not text or len(text) < 200:
        return None

    title = ""
    try:
        meta = trafilatura.extract_metadata(r.text)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    return FetchedPage(
        url=url,
        title=title,
        text=text,
        domain=urlparse(url).netloc,
        fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )


async def _fetch_many_async(urls: List[str], concurrency: int = 5) -> List[FetchedPage]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def bounded(u):
            async with sem:
                return await _fetch_one(client, u)
        results = await asyncio.gather(*[bounded(u) for u in urls],
                                       return_exceptions=False)
    return [r for r in results if r is not None]


def fetch_many(urls: List[str], concurrency: int = 5) -> List[FetchedPage]:
    """Sync wrapper. Streamlit runs in a thread so a fresh loop is fine."""
    try:
        return asyncio.run(_fetch_many_async(urls, concurrency))
    except RuntimeError:
        # In case we're already inside a loop (unlikely in Streamlit but safe)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_fetch_many_async(urls, concurrency))
        finally:
            loop.close()