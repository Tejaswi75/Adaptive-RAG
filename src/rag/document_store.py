"""
Saves each session's uploaded document in MongoDB so it survives restarts.

What is stored: the text chunks, their embeddings and the user's description,
keyed by session id. The uploaded file itself is never stored. Everything is
deleted automatically after DOCUMENT_TTL_DAYS (a MongoDB TTL index), or at once
when the user presses "Delete my document".

Without MONGO_URI every function is a no-op and documents stay in memory only.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from src.core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DB_NAME = "adaptive_rag"
TTL_DAYS = int(os.getenv("DOCUMENT_TTL_DAYS", "7"))

_db = None
_indexes_ready = False


def enabled() -> bool:
    """True when a MongoDB connection string is configured."""
    return bool(os.getenv("MONGO_URI"))


def _get_db():
    """Synchronous MongoDB handle (the retriever runs in worker threads)."""
    global _db, _indexes_ready
    if _db is None:
        from pymongo import MongoClient
        _db = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)[DB_NAME]
    if not _indexes_ready:
        ttl = TTL_DAYS * 24 * 3600
        _db.document_sessions.create_index("created_at", expireAfterSeconds=ttl)
        _db.document_chunks.create_index("created_at", expireAfterSeconds=ttl)
        _db.document_chunks.create_index([("session_id", 1), ("position", 1)])
        # Chat history is kept for the same period as the document it talks about
        _db.chat_history.create_index("timestamp", expireAfterSeconds=ttl)
        _indexes_ready = True
    return _db


def save(session_id: str, description: str, filename: str,
         texts: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> bool:
    """Replace the session's saved document. Returns False if saving failed."""
    if not enabled():
        return False
    try:
        db = _get_db()
        now = datetime.now(timezone.utc)
        delete(session_id, include_chat=False)
        db.document_chunks.insert_many([
            {"session_id": session_id, "position": i, "text": t, "metadata": m,
             "embedding": [float(x) for x in e], "created_at": now}
            for i, (t, m, e) in enumerate(zip(texts, metadatas, embeddings))
        ])
        # Written last: a session only counts as saved once all its chunks are in
        db.document_sessions.replace_one(
            {"_id": session_id},
            {"_id": session_id, "description": description, "filename": filename,
             "chunk_count": len(texts), "created_at": now},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.error("Could not save document for session %s: %s", session_id[:8], e)
        return False


def load(session_id: str) -> Optional[dict]:
    """Return the saved document for a session, or None."""
    if not enabled():
        return None
    try:
        db = _get_db()
        info = db.document_sessions.find_one({"_id": session_id})
        if not info:
            return None
        chunks = list(db.document_chunks.find({"session_id": session_id}).sort("position", 1))
        if len(chunks) != info.get("chunk_count"):
            logger.warning("Saved document for session %s is incomplete; ignoring it", session_id[:8])
            return None
        return {
            "description": info.get("description", ""),
            "filename": info.get("filename", ""),
            "texts": [c["text"] for c in chunks],
            "metadatas": [c.get("metadata", {}) for c in chunks],
            "embeddings": [c["embedding"] for c in chunks],
        }
    except Exception as e:
        logger.error("Could not load document for session %s: %s", session_id[:8], e)
        return None


def info(session_id: str) -> Optional[dict]:
    """Filename and description of the saved document, without the chunks."""
    if not enabled():
        return None
    try:
        doc = _get_db().document_sessions.find_one({"_id": session_id}, {"_id": 0, "filename": 1, "description": 1})
        return doc
    except Exception as e:
        logger.error("Could not read document info for session %s: %s", session_id[:8], e)
        return None


def delete(session_id: str, include_chat: bool = True) -> None:
    """Remove the session's saved document (and, by default, its chat history)."""
    if not enabled():
        return
    db = _get_db()
    db.document_sessions.delete_one({"_id": session_id})
    db.document_chunks.delete_many({"session_id": session_id})
    if include_chat:
        db.chat_history.delete_many({"session_id": session_id})
