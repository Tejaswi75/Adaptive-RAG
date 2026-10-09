"""
Landing page for the Adaptive RAG Streamlit app.
"""

import uuid

import streamlit as st

st.set_page_config(page_title="Adaptive RAG Assistant", page_icon="🤖")

# The chat page owns the session id. If this page was opened with a saved
# link (?sid=...), pass that id on so the chat and document are restored.
sid = st.query_params.get("sid", "")
try:
    if sid and str(uuid.UUID(sid)) == sid:
        st.session_state["session_id"] = sid
except ValueError:
    pass

st.title("🤖 Adaptive RAG Assistant")
st.write(
    "Upload a PDF or TXT document and ask questions about it. "
    "Each question is routed by a LangGraph agent to document retrieval, "
    "the LLM's general knowledge, or a live web search. Retrieved documents are "
    "graded for relevance, and the query is rewritten if they miss."
)

st.markdown(
    """
**Try it:**
1. Open the chat and upload a document from the sidebar (e.g. a resume or lecture notes).
2. Ask something about it, e.g. *"What projects are listed?"*
3. Ask a general or current-events question to see the other routes.
"""
)

if st.button("Start chatting →", type="primary"):
    st.switch_page("pages/chat.py")
