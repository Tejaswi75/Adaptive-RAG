"""
Per-session FAISS vector stores for uploaded documents.

Each chat session gets its own index, so one visitor's upload never answers
another visitor's questions. Only the most recent MAX_SESSIONS indexes are
kept in memory; indexes are lost when the backend restarts.
"""

import os
import threading
from collections import OrderedDict

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.core.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = None


def get_embeddings() -> FastEmbedEmbeddings:
    """
    Load the embedding model on first use.

    FastEmbed runs the model with ONNX Runtime, so there is no PyTorch
    dependency; the ~90 MB model is downloaded once and cached.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
NO_DOCUMENTS = "No documents have been uploaded in this session yet."

_stores: "OrderedDict[str, FAISS]" = OrderedDict()
_descriptions: dict = {}
_lock = threading.Lock()


def retriever_chain(chunks: list[Document], session_id: str, description: str = "") -> bool:
    """
    Embed document chunks into a new FAISS index for this session.

    Args:
        chunks: Document chunks to index.
        session_id: Chat session the document belongs to.
        description: The user's short description of the document.

    Returns:
        True on success, False otherwise.
    """
    try:
        store = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    except Exception as e:
        logger.error("Error storing documents in FAISS: %s", e)
        return False

    with _lock:
        _stores[session_id] = store
        _descriptions[session_id] = description
        _stores.move_to_end(session_id)
        while len(_stores) > MAX_SESSIONS:
            old, _ = _stores.popitem(last=False)
            _descriptions.pop(old, None)

    logger.info("Indexed %d chunks for session %s", len(chunks), session_id[:8])
    return True


class SessionRetriever:
    """Returns the most relevant chunks of a session's document as one string."""

    def __init__(self, session_id: str, k: int = 4):
        self.session_id = session_id
        self.k = k

    @property
    def has_document(self) -> bool:
        return self.session_id in _stores

    @property
    def description(self) -> str:
        return _descriptions.get(self.session_id, "")

    def invoke(self, query: str) -> str:
        with _lock:
            store = _stores.get(self.session_id)
            if store is not None:
                _stores.move_to_end(self.session_id)
        if store is None:
            return NO_DOCUMENTS
        docs = store.similarity_search(query, k=self.k)
        return "\n\n".join(d.page_content for d in docs)


def get_retriever(session_id: str = "default") -> SessionRetriever:
    """Return the retriever for a chat session."""
    return SessionRetriever(session_id)
