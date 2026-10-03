"""
Streamlit frontend for the LangGraph multi-tool chatbot
(token streaming + PDF upload + human-in-the-loop approval).

Backend file must be named `backend.py` in the same folder.

Run with:
    streamlit run frontend_v2.py
"""

import os
import tempfile
import time
import uuid

import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.types import Command

from backend import app, tools, rag_document


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Tool AI Chat", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Tool AI Assistant Created by Sailesh")
st.caption("Search the web, check weather, look up stocks, do math, "
           "buy stocks (with approval), and ask questions about your PDF.")

TOOL_ICONS = {
    "get_weather": "🌤️ Weather",
    "calculator": "🧮 Calculator",
    "stockprice_tool": "📈 Stock Price",
    "tavily_search": "🔎 Web Search",
    "rag_tool": "📄 PDF Search",
    "purchase_stock": "🛒 Purchase Stock",
}

RENDER_INTERVAL = 0.05  # seconds; redrawing markdown per token is slow


def pretty_tool_name(name: str) -> str:
    return TOOL_ICONS.get(name, f"🔧 {name}")


def extract_text(content) -> str:
    """Return only human-readable text from str / Gemini content-block lists.
    No .strip() here, so streamed chunks keep their spaces."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block \
                    and block.get("type", "text") == "text":
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_pdf" not in st.session_state:
    st.session_state.indexed_pdf = None
if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}


def get_pending_interrupt():
    """Return the interrupt message if the graph is paused, else None."""
    try:
        snapshot = app.get_state(CONFIG)
        for task in snapshot.tasks:
            for it in task.interrupts:
                return str(it.value)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📄 Chat with a PDF")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf is not None and st.session_state.indexed_pdf != pdf.name:
        with st.spinner("Indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getbuffer())
                tmp_path = tmp.name
            try:
                rag_document(tmp_path)  # builds + saves the FAISS index
                st.session_state.indexed_pdf = pdf.name
                st.success(f"Indexed: {pdf.name}")
            except Exception as e:
                st.error(f"Could not index PDF: {e}")
            finally:
                os.remove(tmp_path)
    elif st.session_state.indexed_pdf:
        st.caption(f"Active PDF: {st.session_state.indexed_pdf}")

    st.divider()
    st.subheader("Available tools")
    for t in tools:
        st.markdown(f"- {pretty_tool_name(getattr(t, 'name', str(t)))}")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_interrupt = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
def stream_response(payload):
    """Stream the graph output into the UI. `payload` is either new user
    input or a Command(resume=...) used to continue after an interrupt."""
    tools_used = []
    final_text = ""

    with st.chat_message("assistant"):
        status_box = st.empty()
        answer_box = st.empty()
        last_render = 0.0

        try:
            for chunk, meta in app.stream(payload, config=CONFIG, stream_mode="messages"):
                # Tokens from the LLM node only
                if isinstance(chunk, AIMessageChunk) \
                        and meta.get("langgraph_node") == "chat_node":
                    for tc in chunk.tool_call_chunks or []:
                        name = tc.get("name")
                        if name:
                            if name not in tools_used:
                                tools_used.append(name)
                            status_box.caption(f"⏳ Using {pretty_tool_name(name)}...")

                    text = extract_text(chunk.content)
                    if text:
                        final_text += text
                        now = time.monotonic()
                        if now - last_render >= RENDER_INTERVAL:
                            answer_box.markdown(final_text + "▌")
                            last_render = now

                # A tool finished -> discard pre-tool text, keep final answer only
                elif isinstance(chunk, ToolMessage):
                    status_box.caption(f"✅ {pretty_tool_name(chunk.name)} done")
                    final_text = ""
                    answer_box.empty()

        except Exception as e:
            final_text = f"⚠️ Something went wrong: {e}"

        if tools_used:
            names = ", ".join(pretty_tool_name(t) for t in tools_used)
            status_box.caption(f"Used: {names}")
        else:
            status_box.empty()
        answer_box.markdown(final_text.strip())

    final_text = final_text.strip()
    if final_text or tools_used:
        st.session_state.messages.append(
            {"role": "assistant", "content": final_text, "tools_used": tools_used}
        )

    # Did a tool pause the graph for human approval?
    pending = get_pending_interrupt()
    st.session_state.pending_interrupt = pending
    if pending:
        st.rerun()


# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("tools_used"):
            names = ", ".join(pretty_tool_name(t) for t in msg["tools_used"])
            st.caption(f"Used: {names}")
        if msg["content"]:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Human-in-the-loop approval
# ---------------------------------------------------------------------------
if st.session_state.pending_interrupt:
    with st.chat_message("assistant"):
        st.warning(f"⚠️ Approval needed: {st.session_state.pending_interrupt}")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approve", use_container_width=True)
        decline = col2.button("❌ Decline", use_container_width=True)

    if approve or decline:
        st.session_state.pending_interrupt = None
        stream_response(Command(resume="yes" if approve else "no"))


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    "Ask me anything...",
    disabled=bool(st.session_state.pending_interrupt),
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    stream_response({"messages": [HumanMessage(content=user_input)]})