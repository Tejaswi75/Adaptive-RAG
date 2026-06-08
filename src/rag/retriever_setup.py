"""
Retriever setup and vector store configuration.
"""

import os

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.core.config import settings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Global variable to store the FAISS vectorstore instance
_faiss_vectorstore = None


def retriever_chain(chunks: list[Document]):
    """
    Initialize and store documents in FAISS vector database.

    Args:
        chunks: List of document chunks to store.

    Returns:
        Boolean indicating success of the operation.
    """
    global _faiss_vectorstore

    try:
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        _faiss_vectorstore = vectorstore

        print("FAISS vector store initialized with documents")
        print(f"Vectorstore contains {len(chunks)} document chunks")

        return True

    except Exception as e:
        print(f"Error storing documents in FAISS: {e}")
        return False


def get_retriever():
    """
    Get a retriever tool connected to the FAISS vector store.

    Returns:
        A LangChain retriever tool configured for the vector store.
    """
    global _faiss_vectorstore

    try:
        # Use existing vectorstore if documents were uploaded
        if _faiss_vectorstore is not None:
            retriever = _faiss_vectorstore.as_retriever()
            print("Using existing FAISS vectorstore with uploaded documents")

        else:
            print("No documents uploaded yet, creating dummy vectorstore")

            from langchain_core.documents import Document as LangChainDocument

            dummy_doc = LangChainDocument(
                page_content="No documents have been uploaded yet. Please upload a document first.",
                metadata={"source": "initialization"}
            )

            _faiss_vectorstore = FAISS.from_documents(
                documents=[dummy_doc],
                embedding=embeddings
            )

            retriever = _faiss_vectorstore.as_retriever()

        # Load description if present
        if os.path.exists("description.txt"):
            with open("description.txt", "r", encoding="utf-8") as f:
                description = f.read()
        else:
            description = ""

        retriever_tool = create_retriever_tool(
            retriever,
            "retriever_customer_uploaded_documents",
            """
Use this tool whenever the user asks anything about the uploaded document.

Examples:
- What is my name?
- What is my education?
- What skills do I have?
- What projects are listed?
- Summarize my resume.
- Tell me about myself.

Always search the uploaded document before answering.
Do not rely on general knowledge for document-related questions.
"""
        )

        return retriever_tool

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise Exception(e)