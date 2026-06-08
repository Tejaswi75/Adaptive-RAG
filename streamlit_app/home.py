"""
Home page for Streamlit authentication interface.
"""

import logging
import streamlit as st

# Hide sidebar for cleaner look
hide_sidebar_style = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LangGraph Chat")

st.title("🤖 Adaptive RAG Assistant")

# Bypass missing Rust authentication service
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "demo_session"
    st.session_state["jwt_token"] = "demo_jwt"
    st.session_state["username"] = "demo_user"

st.success("Authentication bypass enabled.")

with st.form("auth_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Enter Chat")

if submit:
    st.session_state["username"] = (
        username if username else "demo_user"
    )

    try:
        st.switch_page("pages/Chat.py")
    except Exception:
        try:
            st.switch_page("pages/chat.py")
        except Exception as e:
            st.error(f"Unable to open chat page: {e}")

# Debug logs section
with st.expander("📜 Debug Logs"):
    try:
        with open("app.log", "r") as log_file:
            st.text(log_file.read())
    except FileNotFoundError:
        st.warning("Log file not found yet.")