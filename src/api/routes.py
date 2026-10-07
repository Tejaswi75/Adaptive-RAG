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
    result = await asyncio.to_thread(builder.invoke, {"messages": messages})

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
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.
    """
    try:
        status_upload = await asyncio.to_thread(documents, description, file)

        return {
            "status": status_upload
        }

    except Exception as e:
        logger.exception("Document upload failed")

        return {
            "status": False,
            "error": str(e)
        }