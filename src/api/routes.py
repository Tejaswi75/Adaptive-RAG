"""
API routes for RAG operations.
"""

from fastapi import APIRouter, UploadFile, File, Header
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.
    """
    chat_history = ChatHistory.get_session_history(req.session_id)

    await chat_history.add_message(
        HumanMessage(content=req.query)
    )

    messages = await chat_history.get_messages()

    result = builder.invoke({
        "messages": messages
    })

    output_text = result["messages"][-1].content

    await chat_history.add_message(
        AIMessage(content=output_text)
    )

    return {"result": result["messages"][-1]}


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.
    """
    try:
        status_upload = documents(description, file)

        return {
            "status": status_upload
        }

    except Exception as e:
        import traceback

        print("\n========== DOCUMENT UPLOAD ERROR ==========")
        traceback.print_exc()
        print("==========================================\n")

        return {
            "status": False,
            "error": str(e)
        }