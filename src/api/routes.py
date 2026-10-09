"""
API routes for RAG operations.
"""

import asyncio

from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage, AIMessage

from src.memory import get_session_history
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag import document_store
from src.rag.retriever_setup import document_info, forget
from src.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.
    """
    chat_history = get_session_history(req.session_id)

    await chat_history.add_message(
        HumanMessage(content=req.query)
    )

    messages = await chat_history.get_messages()

    # The graph is synchronous; run it in a worker thread so one slow query
    # doesn't block the server for everyone else.
    result = await asyncio.to_thread(
        builder.invoke, {"messages": messages, "session_id": req.session_id}
    )

    output_text = result["messages"][-1].content

    await chat_history.add_message(
        AIMessage(content=output_text)
    )

    return {
        "result": result["messages"][-1],
        "route": result.get("route"),
        "rewrites": result.get("rewrite_count") or 0,
    }


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description"),
    session_id: str = Header("default", alias="X-Session-Id"),
):
    """
    Upload a document for RAG processing.
    """
    try:
        status_upload = await asyncio.to_thread(documents, description, file, session_id)

        return {
            "status": status_upload
        }

    except Exception as e:
        logger.exception("Document upload failed")

        return {
            "status": False,
            "error": str(e)
        }


@router.get("/rag/session")
async def get_session(session_id: str = Header(..., alias="X-Session-Id")):
    """
    The saved document and chat history for a session, so a returning user
    picks up where they left off.
    """
    messages = await get_session_history(session_id).get_messages()
    roles = {"human": "user", "ai": "assistant"}
    return {
        "persistent": document_store.enabled(),
        "document": await asyncio.to_thread(document_info, session_id),
        "history": [
            {"role": roles[m.type], "content": m.content}
            for m in messages if m.type in roles
        ],
    }


@router.delete("/rag/documents")
async def delete_document(session_id: str = Header(..., alias="X-Session-Id")):
    """Delete the session's document and chat history, in memory and in MongoDB."""
    await asyncio.to_thread(forget, session_id)
    await get_session_history(session_id).clear()
    return {"status": True}
