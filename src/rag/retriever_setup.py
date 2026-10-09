"""
Per-session FAISS vector stores for uploaded documents.

Each chat session gets its own index, so one visitor's upload never answers
another visitor's questions. Only the most recent MAX_SESSIONS indexes are
kept in memory. When MongoDB is configured, each session's chunks and
embeddings are also saved there (see document_store), so an evicted or
restarted session is rebuilt from the database instead of being lost.
"""

import os
import threading
import time
from collections import OrderedDict

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.core.logger import get_logger
from src.rag import document_store

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

# Sessions recently found to have no saved document, so every question in a
# document-less chat doesn't query MongoDB again: {session_id: time checked}
_missing: dict = {}
MISSING_CACHE_SECONDS = 60


def _cache(session_id: str, store: FAISS, description: str) -> None:
    with _lock:
        _stores[session_id] = store
        _descriptions[session_id] = description
        _stores.move_to_end(session_id)
        _missing.pop(session_id, None)
        while len(_stores) > MAX_SESSIONS:
            old, _ = _stores.popitem(last=False)
            _descriptions.pop(old, None)


def _restore(session_id: str) -> FAISS | None:
    """Rebuild a session's index from MongoDB, using the saved embeddings."""
    checked = _missing.get(session_id)
    if checked and time.monotonic() - checked < MISSING_CACHE_SECONDS:
        return None
    saved = document_store.load(session_id)
    if not saved or not saved["texts"]:
        if len(_missing) > 1000:  # keep the negative cache small
            _missing.clear()
        _missing[session_id] = time.monotonic()
        return None
    store = FAISS.from_embeddings(
        text_embeddings=list(zip(saved["texts"], saved["embeddings"])),
        embedding=get_embeddings(),
        metadatas=saved["metadatas"],
    )
    _cache(session_id, store, saved["description"])
    logger.info("Restored %d chunks for session %s from MongoDB", len(saved["texts"]), session_id[:8])
    return store


def _get_store(session_id: str) -> FAISS | None:
    with _lock:
        store = _stores.get(session_id)
        if store is not None:
            _stores.move_to_end(session_id)
            return store
    return _restore(session_id)


def forget(session_id: str) -> None:
    """Drop a session's document from memory and from MongoDB."""
    with _lock:
        _stores.pop(session_id, None)
        _descriptions.pop(session_id, None)
    _missing[session_id] = time.monotonic()
    document_store.delete(session_id)


def retriever_chain(chunks: list[Document], session_id: str, description: str = "",
                    filename: str = "") -> bool:
    """
    Embed document chunks into a new FAISS index for this session.

    Args:
        chunks: Document chunks to index.
        session_id: Chat session the document belongs to.
        description: The user's short description of the document.
        filename: Name of the uploaded file, shown when the session is restored.

    Returns:
        True on success, False otherwise.
    """
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    try:
        # Embed once and reuse the vectors for both FAISS and MongoDB
        vectors = get_embeddings().embed_documents(texts)
        store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, vectors)),
            embedding=get_embeddings(),
            metadatas=metadatas,
        )
    except Exception as e:
        logger.error("Error storing documents in FAISS: %s", e)
        return False

    _cache(session_id, store, description)
    logger.info("Indexed %d chunks for session %s", len(chunks), session_id[:8])

    # Saving is best effort: the document still works in memory if MongoDB is down
    if document_store.save(session_id, description, filename, texts, metadatas, vectors):
        logger.info("Saved session %s document to MongoDB", session_id[:8])
    return True


class SessionRetriever:
    """Returns the most relevant chunks of a session's document as one string."""

    def __init__(self, session_id: str, k: int = 4):
        self.session_id = session_id
        self.k = k

    @property
    def has_document(self) -> bool:
        return _get_store(self.session_id) is not None

    @property
    def description(self) -> str:
        _get_store(self.session_id)
        return _descriptions.get(self.session_id, "")

    def invoke(self, query: str) -> str:
        store = _get_store(self.session_id)
        if store is None:
            return NO_DOCUMENTS
        docs = store.similarity_search(query, k=self.k)
        return "\n\n".join(d.page_content for d in docs)


def get_retriever(session_id: str = "default") -> SessionRetriever:
    """Return the retriever for a chat session."""
    return SessionRetriever(session_id)


def document_info(session_id: str) -> dict | None:
    """Filename and description of the session's document, if it has one."""
    saved = document_store.info(session_id)
    if saved:
        return saved
    with _lock:
        if session_id in _stores:
            return {"filename": "", "description": _descriptions.get(session_id, "")}
    return None
