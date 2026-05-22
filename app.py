"""Streamlit UI. Sessions in the sidebar, streaming events in st.status,
streaming tokens for the final answer, citations in an expander."""
import os

import streamlit as st
from dotenv import load_dotenv

from src.agent import DeepResearchAgent, Event
from src.llm import LLM
from src.memory import (create_session, delete_session, get_messages,
                        get_turns, init_db, list_sessions, rename_session)
from src.search import TavilySearch

load_dotenv()
init_db()

st.set_page_config(page_title="Deep Research Agent", layout="wide", page_icon="🔎")


@st.cache_resource
def get_clients():
    llm = LLM()
    search = TavilySearch(api_key=os.environ["TAVILY_API_KEY"])
    return llm, search


llm, search = get_clients()

# -- sidebar --------------------------------------------------------------

def _new_session_cb():
    st.session_state["session_id"] = create_session()


with st.sidebar:
    st.title("🔎 Deep Research")
    st.caption("Citation-grounded web research, with sessions.")

    st.button("➕  New session", on_click=_new_session_cb, use_container_width=True)

    st.divider()
    st.subheader("Sessions")
    for s in list_sessions(include_empty=False)[:25]:
        label = f"{s['title'][:28]}  ·  {s['id'][:6]}"
        col_sel, col_del = st.columns([5, 1])
        with col_sel:
            if st.button(label, key=f"sess-{s['id']}", use_container_width=True):
                st.session_state["session_id"] = s["id"]
                st.rerun()
        with col_del:
            if st.button("✕", key=f"del-{s['id']}", use_container_width=True):
                delete_session(s["id"])
                if st.session_state.get("session_id") == s["id"]:
                    st.session_state.pop("session_id", None)
                st.rerun()

    st.divider()
    with st.expander("⚙️ Settings"):
        k_queries = st.slider("Search queries per question", 1, 6, 4)
        k_fetch = st.slider("Pages to fetch", 3, 12, 8)
        k_chunks = st.slider("Context chunks", 3, 12, 8)
        st.session_state["k_queries"] = k_queries
        st.session_state["k_fetch"] = k_fetch
        st.session_state["k_chunks"] = k_chunks

# -- init session ---------------------------------------------------------

if "session_id" not in st.session_state:
    existing = list_sessions()
    if existing:
        st.session_state["session_id"] = existing[0]["id"]
    else:
        st.session_state["session_id"] = create_session()
sid = st.session_state["session_id"]

agent = DeepResearchAgent(
    llm, search,
    max_search_queries=st.session_state.get("k_queries", 4),
    max_pages_to_fetch=st.session_state.get("k_fetch", 8),
    max_context_chunks=st.session_state.get("k_chunks", 8),
)

# -- header ---------------------------------------------------------------

st.markdown("### Deep Research Agent")
st.caption(f"Session `{sid}`  ·  No agent framework  ·  Tavily + Groq Llama 3.3 70B")

# -- empty-state welcome --------------------------------------------------

if not get_messages(sid):
    st.markdown("")
    st.info(
        "Ask any research question — I'll plan a strategy, search the "
        "web, read the top sources, and answer with inline citations. "
        "Try one of these:"
    )
    cols = st.columns(3)
    examples = [
        "Who founded Sarvam AI and what is their flagship Indic LLM?",
        "Compare RAG vs fine-tuning for enterprise knowledge.",
        "What were India's major AI policy updates in 2025?",
    ]
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["_prefill"] = ex
            st.rerun()

# -- past conversation ----------------------------------------------------

past_turns = get_turns(sid)
turn_idx = 0
for msg in get_messages(sid):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and turn_idx < len(past_turns):
            t = past_turns[turn_idx]
            with st.expander(f"📚 Sources & trace ({len(t['citations'])} sources, "
                             f"{t.get('elapsed_s', 0):.1f}s)"):
                for c in t["citations"]:
                    st.markdown(
                        f"**[{c['n']}]** {c['title']} — *{c['domain']}*  \n{c['url']}"
                    )
                if t.get("search_queries"):
                    st.markdown("**Search queries issued:**")
                    for q in t["search_queries"]:
                        st.code(q, language=None)
                if t.get("critique"):
                    cri = t["critique"]
                    st.markdown(
                        f"**Self-critique** — grounding {cri.get('grounding', '—')}/5, "
                        f"completeness {cri.get('completeness', '—')}/5, "
                        f"conflict handling {cri.get('conflict_handling', '—')}/5"
                    )
            turn_idx += 1

# -- input ----------------------------------------------------------------

prefill = st.session_state.pop("_prefill", "")
user_query = st.chat_input("Ask a research question…")
if prefill and not user_query:
    user_query = prefill

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status = st.status("Starting research…", expanded=True)
        token_slot = st.empty()
        tokens = []

        def on_event(ev: Event):
            with status:
                st.markdown(f"**{ev.message}**")
                if ev.kind == "plan_done":
                    for q in ev.data.get("queries", []):
                        st.caption(f"🔍 `{q}`")
                elif ev.kind == "search_done":
                    for u in ev.data.get("top_urls", [])[:6]:
                        st.caption(f"📎 {u}")
                elif ev.kind == "fetch_done":
                    st.caption(f"✓ {len(ev.data.get('fetched', []))} pages parsed")
                elif ev.kind == "critic_done":
                    c = ev.data.get("critique", {})
                    if c:
                        st.caption(
                            f"Grounding {c.get('grounding', '—')}/5 · "
                            f"Completeness {c.get('completeness', '—')}/5 · "
                            f"Conflict {c.get('conflict_handling', '—')}/5"
                        )
                elif ev.kind == "refine_start":
                    st.caption(f"⚠️ {ev.data.get('issues', '')}")
                elif ev.kind == "refine_done":
                    c = ev.data.get("critique", {})
                    if c:
                        st.caption(
                            f"After refinement — grounding {c.get('grounding', '—')}/5"
                        )
                    for u in ev.data.get("extra_urls", []):
                        st.caption(f"➕ {u}")

        def on_token(tok: str):
            tokens.append(tok)
            token_slot.markdown("".join(tokens) + "▌")

        try:
            result = agent.run(user_query, sid,
                               on_event=on_event, on_token=on_token)
            token_slot.markdown(result["answer"])
            status.update(
                label=f"Done in {result['elapsed']:.1f}s · "
                      f"{result['n_sources']} sources cited",
                state="complete", expanded=False,
            )
            with st.expander(f"📚 Sources ({len(result['citations'])})"):
                for c in result["citations"]:
                    st.markdown(
                        f"**[{c['n']}]** {c['title']} — *{c['domain']}*  \n{c['url']}"
                    )

            # auto-rename session from first query
            existing = [s for s in list_sessions() if s["id"] == sid]
            if existing and existing[0]["title"] == "New session":
                rename_session(sid, user_query[:48])
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())