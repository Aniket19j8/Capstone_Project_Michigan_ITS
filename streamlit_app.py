"""
Streamlit was our first UI — dead simple, good for "does retrieval even work?"

Needs Ollama + chroma built (steps 03/04). `04_hybrid_retrieval.py` lives next to this file, not in scripts/.

Run:
  pip install streamlit
  streamlit run streamlit_app.py
"""

import streamlit as st
import time
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LLM_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Add scripts directory to path ──
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# ── Page Config ──
st.set_page_config(
    page_title="ITS — Intelligent Ticketing System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD PIPELINE
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    import importlib.util

    retriever_path = SCRIPTS_DIR / "04_hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("hybrid_retrieval", str(retriever_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ticket_retriever = mod.HybridRetriever("its_tickets", load_reranker=True)
    kb_retriever = mod.HybridRetriever("its_knowledge_base", load_reranker=True)

    return ticket_retriever, kb_retriever


@st.cache_resource
def load_llm():
    """Health check for Ollama model."""
    import requests
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": "Say ready",
                "stream": False,
                "options": {"num_predict": 5}
            },
            timeout=10
        )
        return resp.status_code == 200
    except:
        return False


def llm_generate(prompt, system="", temperature=0.1, max_tokens=2048):
    """Call Ollama LLM."""
    import requests
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=180
        )
        import re
        raw = resp.json().get("response", "").strip()
        # Strip thinking blocks emitted by reasoning models (e.g. qwen3)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    except Exception as e:
        return f"⚠️ LLM Error: {e}. Make sure Ollama is running."


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_ticket_context(results):
    parts = []
    for i, doc in enumerate(results):
        meta = doc.get("metadata", {})
        text = doc.get("text", "")[:500]
        score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
        parts.append(
            f"--- Ticket {i+1} [ID: {meta.get('ticket_id', doc['id'])}] "
            f"(relevance: {score:.3f}) ---\n"
            f"Title: {meta.get('title', 'N/A')}\n"
            f"Category: {meta.get('category', 'N/A')} | "
            f"Severity: {meta.get('severity', 'N/A')}\n"
            f"Content: {text}\n"
        )
    return "\n".join(parts) if parts else "No similar tickets found."


def format_kb_context(results):
    parts = []
    for i, doc in enumerate(results):
        meta = doc.get("metadata", {})
        text = doc.get("text", "")[:800]
        parts.append(
            f"--- KB Article {i+1} [Source: {meta.get('source_file', 'N/A')}] ---\n"
            f"Content:\n{text}\n"
        )
    return "\n".join(parts) if parts else "No relevant KB articles found."


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🎫 Intelligent Ticketing System")
st.caption("AI-powered ticket resolution using RAG (Retrieval-Augmented Generation)")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")

    try:
        ticket_retriever, kb_retriever = load_pipeline()
        st.success(
            f"Retriever Ready\n"
            f"Tickets: {ticket_retriever.collection.count()} | "
            f"KB: {kb_retriever.collection.count()}"
        )
    except Exception as e:
        st.error(f"Retriever failed: {e}")
        ticket_retriever = kb_retriever = None

    if load_llm():
        st.success(f"LLM Ready: {LLM_MODEL}")
    else:
        st.warning(f"LLM Offline — run `ollama serve`")

    st.divider()

    top_k_tickets = st.slider("Similar tickets", 1, 10, 3)
    top_k_kb = st.slider("KB articles", 1, 10, 3)
    use_reranking = st.checkbox("Use reranking", value=True)

# ─────────────────────────────────────────────
# MAIN TAB
# ─────────────────────────────────────────────
st.header("Submit a New Ticket")

ticket_input = st.text_area(
    "Describe your issue:",
    height=120,
    placeholder="My VPN disconnects every 10 minutes..."
)

if st.button("🚀 Submit & Analyze"):

    if not ticket_input.strip():
        st.warning("Please enter a description.")
    elif not ticket_retriever:
        st.error("Retriever not loaded.")
    else:
        with st.status("Running RAG pipeline...", expanded=True):

            # Step 1: Retrieve similar tickets
            t0 = time.time()
            similar_tickets = ticket_retriever.search(
                ticket_input,
                top_k=top_k_tickets,
                use_reranking=use_reranking
            )
            t1 = time.time()
            st.write(f"Found {len(similar_tickets)} similar tickets ({t1-t0:.2f}s)")

            # Step 2: Retrieve KB
            kb_results = kb_retriever.search(
                ticket_input,
                top_k=top_k_kb,
                use_reranking=use_reranking
            )
            t2 = time.time()
            st.write(f"Found {len(kb_results)} KB articles ({t2-t1:.2f}s)")

            # Step 3: Generate Resolution
            prompt = f"""You are an IT support agent. Based on the context below, give the user clear steps to resolve their issue.

User Ticket: {ticket_input}

Similar Tickets:
{format_ticket_context(similar_tickets)}

Knowledge Base:
{format_kb_context(kb_results)}

Reply with ONLY this — no intro, no reasoning, no commentary:

**Steps to resolve:**
1. <step>
2. <step>
3. <step>
(add more if needed)

**If unresolved:** <one line on who to contact>

Keep every step short and actionable. Use ONLY the provided context."""

            resolution = llm_generate(prompt)
            t3 = time.time()

            st.write(f"LLM generated response ({t3-t2:.2f}s)")
            st.success(f"Total time: {t3-t0:.1f}s")

        st.divider()
        st.subheader("📋 Resolution Blueprint")
        st.markdown(resolution)

        with st.expander("Similar Tickets"):
            for doc in similar_tickets:
                st.markdown(f"**{doc.get('metadata', {}).get('title','N/A')}**")
                st.caption(doc.get("text", "")[:300])

        with st.expander("Knowledge Base Articles"):
            for doc in kb_results:
                st.markdown(f"**{doc.get('metadata', {}).get('source_file','N/A')}**")
                st.caption(doc.get("text", "")[:300])