"""Central place for tunables. Kept tiny on purpose — config sprawl is a smell."""
import os

LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")

# Agent defaults — override per-call when needed
DEFAULT_MAX_QUERIES = 4
DEFAULT_RESULTS_PER_QUERY = 6
DEFAULT_MAX_PAGES = 8
DEFAULT_MAX_CHUNKS = 8
DEFAULT_MAX_CHUNK_CHARS = 1500

# Context budget for synthesis
MAX_CONTEXT_CHARS = 14_000
SUMMARIZE_THRESHOLD_TOKENS = 2_000