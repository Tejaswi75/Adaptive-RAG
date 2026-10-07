"""
API client for communicating with backend services.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# FastAPI backend URL (set BACKEND_URL when the backend runs elsewhere)
PYTHON_BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def query_backend(query: str, session_id: str) -> str:
    """
    Send a query to the RAG backend.

    Args:
        query: The user's query text.
        session_id: Session identifier for tracking conversation.

    Returns:
        Response text from the backend or error message.
    """
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
