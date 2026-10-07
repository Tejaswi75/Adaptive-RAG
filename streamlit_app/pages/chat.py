"""
Chat page for the Streamlit application.
"""

import uuid

import streamlit as st

from utils.api_client import query_backend, document_upload_rag

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

# One chat history per browser session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

col1, col2 = st.columns([10, 2])
with col2:
    st.write("")  # Spacer
    if st.button("🔄 New chat", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

st.title("💬 Adaptive RAG Chat")

# Document upload section
with st.sidebar:
    st.header("📂 Upload Documents")

    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

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
                success = document_upload_rag(uploaded_file, file_description)
                if success:
                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.session_state.uploaded_files[file_key] = True
                else:
                    st.error(f"Document Upload Failed: {uploaded_file.name}")
            else:
                st.info(f"Uploaded: {uploaded_file.name}")
        else:
            st.warning("Please describe your document before uploading.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
