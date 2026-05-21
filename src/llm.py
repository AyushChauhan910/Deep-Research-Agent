"""Thin Groq wrapper. Kept thin so swapping providers is one file."""
import os
from typing import Iterator, List, Dict

from groq import Groq

from .config import LLM_MODEL


class LLM:
    def __init__(self, model: str = None):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model or LLM_MODEL

    def chat(self, messages: List[Dict], temperature: float = 0.2,
             max_tokens: int = 2048) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: List[Dict], temperature: float = 0.3,
               max_tokens: int = 2048) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta