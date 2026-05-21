"""Custom metrics for research-answer quality.

Why these:
- citation_density: inline grounding is the whole point; sparse citations = unsupported claims
- unique_citations: detects "padding" with the same source repeatedly
- source_diversity: avoids mono-source answers
- keyword_coverage: cheap correctness proxy without expensive human gold answers
- uncertainty / conflict detection: regex heuristics on linguistic cues
- llm_judge: holistic, multi-dimensional check that's harder to game
"""
import re
from typing import Dict, List

from src.utils import extract_json


def citation_density(answer: str) -> float:
    words = max(len(answer.split()), 1)
    cites = len(re.findall(r"\[\d+\]", answer))
    return round((cites / words) * 100, 2)


def n_unique_citations(answer: str) -> int:
    return len(set(re.findall(r"\[\d+\]", answer)))


def source_diversity(citations: List[Dict]) -> float:
    if not citations:
        return 0.0
    domains = [c["domain"] for c in citations]
    return round(len(set(domains)) / len(domains), 2)


def keyword_coverage(answer: str, expected: List[str]) -> float:
    if not expected:
        return 1.0
    a = answer.lower()
    hits = sum(1 for k in expected if k.lower() in a)
    return round(hits / len(expected), 2)


_UNC_CUES = [
    "uncertain", "not enough", "insufficient", "can't confirm", "cannot confirm",
    "no clear", "unclear", "couldn't find", "speculative", "varies widely",
    "would require", "projection", "no reliable", "limited evidence",
]

_CONFLICT_CUES = [
    "disagree", "conflict", "however,", "on the other hand", "in contrast",
    "but [", "while [", "whereas",
]


def expresses_uncertainty(answer: str) -> bool:
    a = answer.lower()
    return any(c in a for c in _UNC_CUES)


def flags_conflict(answer: str) -> bool:
    a = answer.lower()
    return any(c in a for c in _CONFLICT_CUES)


# ---------- LLM-as-judge ----------

_JUDGE_PROMPT = """You are evaluating a research answer. Be strict.

Question: {question}

Answer:
{answer}

Citations available to the answer:
{citations}

Rate 1-5 (integer) for each. Output JSON only:
{{
  "grounded": <1-5>,
  "relevant": <1-5>,
  "well_cited": <1-5>,
  "clear": <1-5>,
  "uncertainty_calibration": <1-5>
}}

grounded: are claims actually traceable to a cited source?
relevant: does the answer address the question?
well_cited: are [n] markers used appropriately and consistently?
clear: is the answer well-structured and easy to read?
uncertainty_calibration: when evidence is weak, does the answer say so honestly?"""


def llm_judge(llm, question: str, answer: str, citations: List[Dict]) -> Dict:
    cite_str = "; ".join(
        f"[{c['n']}] {c['title']} ({c['domain']})" for c in citations
    ) or "(none)"
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer,
                                  citations=cite_str)
    raw = llm.chat([{"role": "user", "content": prompt}],
                   temperature=0.0, max_tokens=200)
    return extract_json(raw) or {}