"""
Tests for saving uploaded documents in MongoDB (mongomock stands in for the
database, and a tiny hashing embedder stands in for the real model).

Run: pip install -r requirements-dev.txt && pytest
"""

import asyncio
import hashlib
import math

import mongomock
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag import document_store, retriever_setup as rs


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors: texts sharing words end up close together."""

    def _vec(self, text):
        v = [0.0] * 64
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def db(monkeypatch):
    """A fresh fake MongoDB, with MONGO_URI set so saving is on."""
    fake = mongomock.MongoClient()["adaptive_rag"]
    monkeypatch.setenv("MONGO_URI", "mongodb://fake")
    monkeypatch.setattr(document_store, "_db", fake)
    monkeypatch.setattr(document_store, "_indexes_ready", False)
    monkeypatch.setattr(rs, "_embeddings", HashEmbeddings())
    rs._stores.clear()
    rs._descriptions.clear()
    rs._missing.clear()
    yield fake
    rs._stores.clear()
    rs._descriptions.clear()
    rs._missing.clear()


def restart():
    """Simulate a backend restart: everything in memory is gone."""
    rs._stores.clear()
    rs._descriptions.clear()
    rs._missing.clear()


CHUNKS = [
    Document(page_content="Tejaswi studied at NIET Greater Noida with a CGPA of 7.3", metadata={"page": 0}),
    Document(page_content="Projects include a vector database written in C++", metadata={"page": 1}),
    Document(page_content="Hobbies are cricket and reading", metadata={"page": 1}),
]


def test_upload_is_saved_with_chunks_embeddings_and_description(db):
    assert rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    info = db.document_sessions.find_one({"_id": "s1"})
    assert info["filename"] == "resume.pdf" and info["description"] == "My resume"
    assert info["chunk_count"] == 3
    chunks = list(db.document_chunks.find({"session_id": "s1"}).sort("position", 1))
    assert [c["text"] for c in chunks] == [c.page_content for c in CHUNKS]
    assert len(chunks[0]["embedding"]) == 64


def test_document_comes_back_after_restart(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    restart()
    retriever = rs.get_retriever("s1")
    assert retriever.has_document
    assert retriever.description == "My resume"
    assert "NIET" in retriever.invoke("which college NIET CGPA")


def test_sessions_stay_separate(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    restart()
    assert not rs.get_retriever("s2").has_document
    assert rs.get_retriever("s2").invoke("college") == rs.NO_DOCUMENTS


def test_new_upload_replaces_the_old_one(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    rs.retriever_chain([Document(page_content="Notes about LangGraph nodes")], "s1", "Notes", "notes.txt")
    assert db.document_chunks.count_documents({"session_id": "s1"}) == 1
    restart()
    assert rs.get_retriever("s1").description == "Notes"


def test_delete_removes_document_and_chat_everywhere(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    db.chat_history.insert_one({"session_id": "s1", "type": "human", "content": "hi"})
    db.chat_history.insert_one({"session_id": "s2", "type": "human", "content": "keep me"})
    rs.forget("s1")
    assert not rs.get_retriever("s1").has_document
    assert db.document_sessions.count_documents({}) == 0
    assert db.document_chunks.count_documents({}) == 0
    assert db.chat_history.count_documents({"session_id": "s1"}) == 0
    assert db.chat_history.count_documents({"session_id": "s2"}) == 1


def test_saved_data_expires_after_seven_days(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    week = 7 * 24 * 3600
    for name, field in [("document_sessions", "created_at"), ("document_chunks", "created_at"),
                        ("chat_history", "timestamp")]:
        ttl = [i for i in db[name].index_information().values()
               if i.get("key") == [(field, 1)]]
        assert ttl and ttl[0].get("expireAfterSeconds") == week, name


def test_incomplete_save_is_ignored(db):
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    db.document_chunks.delete_one({"session_id": "s1", "position": 2})
    restart()
    assert not rs.get_retriever("s1").has_document


def test_without_mongo_documents_live_in_memory_only(db, monkeypatch):
    monkeypatch.delenv("MONGO_URI")
    assert rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    assert db.document_chunks.count_documents({}) == 0
    assert rs.get_retriever("s1").has_document
    restart()
    assert not rs.get_retriever("s1").has_document


def test_upload_still_works_if_mongo_is_down(db, monkeypatch):
    def broken():
        raise ConnectionError("cluster unreachable")
    monkeypatch.setattr(document_store, "_get_db", broken)
    assert rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    assert "NIET" in rs.get_retriever("s1").invoke("NIET college")


def test_session_endpoint_restores_document_and_history(db, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    from src.api import routes
    from src.memory.chathistory_in_memory import InMemoryChatMessageHistory

    monkeypatch.setattr(routes, "get_session_history", InMemoryChatMessageHistory)
    InMemoryChatMessageHistory._store.clear()
    history = InMemoryChatMessageHistory("s1")
    asyncio.run(history.add_message(HumanMessage(content="Where did I study?")))
    asyncio.run(history.add_message(AIMessage(content="NIET Greater Noida.")))
    rs.retriever_chain(CHUNKS, "s1", "My resume", "resume.pdf")
    restart()

    out = asyncio.run(routes.get_session("s1"))
    assert out["persistent"] is True
    assert out["document"] == {"filename": "resume.pdf", "description": "My resume"}
    assert out["history"] == [
        {"role": "user", "content": "Where did I study?"},
        {"role": "assistant", "content": "NIET Greater Noida."},
    ]

    asyncio.run(routes.delete_document("s1"))
    out = asyncio.run(routes.get_session("s1"))
    assert out["document"] is None and out["history"] == []
