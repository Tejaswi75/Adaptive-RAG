"""
Client for the RAG backend.

By default the UI calls the FastAPI backend over HTTP. With EMBEDDED_BACKEND=true
it runs the same backend code inside the Streamlit process instead, so the whole
app deploys as a single Streamlit app (e.g. on Streamlit Community Cloud).
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import requests

logger = logging.getLogger(__name__)

# FastAPI backend URL (set BACKEND_URL when the backend runs elsewhere)
PYTHON_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
EMBEDDED = os.getenv("EMBEDDED_BACKEND", "false").lower() == "true"


_loop = None
_loop_lock = threading.Lock()


def _run(coro):
    """
    Run a backend coroutine on one long-lived event loop.

    Streamlit runs the script in a new thread on every interaction, and
    asyncio.run() would create (and close) a new loop each time. The async
    MongoDB client binds to the first loop it uses, so later calls on a new
    loop fail with "Event loop is closed". A single background loop avoids that.
    """
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="backend-loop", daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def _load_backend():
    """Make the repo root importable and return the backend route module."""
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.api import routes
    return routes


def query_backend(query: str, session_id: str) -> str:
    """
    Send a query to the RAG backend.

    Args:
        query: The user's query text.
        session_id: Session identifier for tracking conversation.

    Returns:
        Response text from the backend or error message.
    """
    if EMBEDDED:
        routes = _load_backend()
        from src.models.query_request import QueryRequest
        try:
            out = _run(routes.rag_query(QueryRequest(query=query, session_id=session_id)))
            return out["result"].content
        except Exception as e:
            logger.exception("Query failed")
            return f"Error: {e}"

    url = f"{PYTHON_BASE_URL}/rag/query"
    logger.info("Calling %s", url)

    response = requests.post(
        url,
        json={"query": query, "session_id": session_id},
        allow_redirects=False
    )

    if response.status_code == 200:
        return response.json()["result"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


def document_upload_rag(file, description: str, session_id: str) -> bool:
    """
    Upload a document to the RAG system.

    Args:
        file: File object to upload.
        description: Description of the document.
        session_id: Chat session the document belongs to.

    Returns:
        True if upload succeeds, False otherwise.
    """
    if EMBEDDED and file:
        _load_backend()
        from src.rag.document_upload import documents
        try:
            file.seek(0)
            return bool(documents(description, SimpleNamespace(filename=file.name, file=file), session_id))
        except Exception:
            logger.exception("Upload failed")
            return False

    headers = {
        "X-Description": description,
        "X-Session-Id": session_id,
    }
    url = f"{PYTHON_BASE_URL}/rag/documents/upload"

    if file:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(url, files=files, headers=headers)
        logger.info("Upload response: %s", response.status_code)

        # The backend returns 200 with {"status": false, "error": ...} on failure
        if response.status_code == 200 and response.json().get("status"):
            return True
        logger.error("Upload failed: %s", response.text)

    return False


def get_session(session_id: str) -> dict:
    """
    Fetch the saved document and chat history for a session.

    Returns:
        {"persistent": bool, "document": {"filename", "description"} or None,
         "history": [{"role", "content"}]}.
        On any error, an empty session.
    """
    empty = {"persistent": False, "document": None, "history": []}
    try:
        if EMBEDDED:
            routes = _load_backend()
            return _run(routes.get_session(session_id))
        response = requests.get(f"{PYTHON_BASE_URL}/rag/session", headers={"X-Session-Id": session_id})
        return response.json() if response.status_code == 200 else empty
    except Exception:
        logger.exception("Could not load session")
        return empty


def delete_document(session_id: str) -> bool:
    """Delete the session's saved document and chat history."""
    try:
        if EMBEDDED:
            routes = _load_backend()
            return bool(_run(routes.delete_document(session_id))["status"])
        response = requests.delete(f"{PYTHON_BASE_URL}/rag/documents", headers={"X-Session-Id": session_id})
        return response.status_code == 200
    except Exception:
        logger.exception("Could not delete document")
        return False
