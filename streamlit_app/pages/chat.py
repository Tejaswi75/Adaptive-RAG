"""
Chat page for the Streamlit application.
"""

import uuid

import streamlit as st

from utils.api_client import query_backend, document_upload_rag, get_session, delete_document

# Configure page settings
st.set_page_config(
    page_title="Adaptive RAG Chat",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None
    }
)


def _is_session_id(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _start_session(session_id: str) -> None:
    """Load the saved document and chat history for this session id."""
    st.session_state["session_id"] = session_id
    saved = get_session(session_id)
    st.session_state.persistent = saved.get("persistent", False)
    st.session_state.saved_document = saved.get("document")
    st.session_state.chat_history = [(m["role"], m["content"]) for m in saved.get("history", [])]
    st.session_state.uploaded_files = {}
    # A new key empties the file picker, so a cleared chat doesn't re-upload the old file
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


# The session id lives in the page URL (?sid=...), so reopening the same link
# brings back the uploaded document and the conversation.
if "session_id" not in st.session_state or "saved_document" not in st.session_state:
    sid = st.session_state.get("session_id") or st.query_params.get("sid", "")
    _start_session(sid if _is_session_id(sid) else str(uuid.uuid4()))
st.query_params["sid"] = st.session_state["session_id"]

col1, col2 = st.columns([10, 2])
with col2:
    st.write("")  # Spacer
    if st.button("🔄 New chat", use_container_width=True):
        _start_session(str(uuid.uuid4()))  # a new chat starts without documents
        st.query_params["sid"] = st.session_state["session_id"]
        st.rerun()

st.title("💬 Adaptive RAG Chat")

# Document upload section
with st.sidebar:
    st.header("📂 Upload Documents")

    saved_doc = st.session_state.get("saved_document")
    if saved_doc:
        st.success(f"📄 Current document: {saved_doc.get('filename') or saved_doc.get('description')}")
        if st.session_state.get("persistent"):
            st.caption(
                "Saved with this chat for 7 days, then deleted automatically. "
                "Bookmark this page's link to come back to it."
            )
        if st.button("🗑️ Delete my document and chat", use_container_width=True):
            if delete_document(st.session_state["session_id"]):
                _start_session(st.session_state["session_id"])
                st.rerun()
            else:
                st.error("Could not delete the document. Please try again.")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file", type=["pdf", "txt"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    file_description = None
    if uploaded_file:
        file_description = st.text_input(
            "📄 Describe your document (required)",
            max_chars=300,
            placeholder="E.g. LangGraph tutorial with workflows and code examples"
        )

        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {}

        file_key = f"{uploaded_file.name}_{file_description}"

        if file_description:
            if file_key not in st.session_state.uploaded_files:
                # Upload file if not already uploaded
                success = document_upload_rag(uploaded_file, file_description, st.session_state["session_id"])
                if success:
                    st.session_state.uploaded_files[file_key] = True
                    st.session_state.saved_document = {
                        "filename": uploaded_file.name, "description": file_description,
                    }
                    st.rerun()
                else:
                    st.error(f"Document Upload Failed: {uploaded_file.name}")
            else:
                st.info(f"Uploaded: {uploaded_file.name}")
        else:
            st.warning("Please describe your document before uploading.")

# User input
user_input = st.chat_input("Ask a question...")

# Process user input and get response
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    response = query_backend(user_input, st.session_state["session_id"])
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()  # Rerun script to display updated messages

# Display chat history
for role, text in st.session_state.chat_history:
    # Models sometimes put <br> inside Markdown table cells; Streamlit shows it literally.
    st.chat_message(role).markdown(text.replace("<br>", " "))
