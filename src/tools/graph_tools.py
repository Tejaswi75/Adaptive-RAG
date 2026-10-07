"""
Tools for graph routing and document grading.
"""

import os
from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.core.logger import get_logger
from src.llms.groq_llm import llm
from src.models.state import State
from src.models.verification_result import VerificationResult

config = Config()
logger = get_logger(__name__)

# Max query rewrites before falling back to web search (prevents infinite loops).
MAX_REWRITES = int(os.getenv("MAX_REWRITES", "2"))
# Set ENABLE_GRADER=false to skip grading/rewriting (used for evaluation ablation).
GRADER_ENABLED = os.getenv("ENABLE_GRADER", "true").lower() != "false"


def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """
    Route the graph to the appropriate node based on query classification.

    Args:
        state (State): The current state of the graph.

    Returns:
        The next node to execute: "retriever", "general_llm", or "web_search".
    """
    if state["route"] == "index":
        return "retriever"
    elif state["route"] == "general":
        return "general_llm"
    else:
        return "web_search"


def doc_tool(state: State) -> Literal["rewrite", "generate", "web_search"]:
    """
    Decide what to do after grading the retrieved documents.

    Args:
        state (State): The current state of the graph.

    Returns:
        "generate" if the documents are relevant, "rewrite" to retry retrieval
        with a reformulated query, or "web_search" once MAX_REWRITES is reached.
    """
    score = (state.get("binary_score") or "").strip().lower()
    rewrites = state.get("rewrite_count") or 0
    if score == "yes":
        return "generate"
    if rewrites >= MAX_REWRITES:
        logger.info("No relevant documents after %d rewrites; falling back to web search", rewrites)
        return "web_search"
    return "rewrite"


def verify_answer(state: State) -> Literal["__end__", "generate"]:
    """
    Verify whether the final answer is faithful to the retrieved context.

    Args:
        state (State): The current state of the graph.

    Returns:
        "__end__" if answer is faithful, otherwise "generate" to retry.
    """
    if state["route"] == "general":
        return "__end__"

    question = state["latest_query"]
    context = state["messages"][-1].content
    final_answer = state["messages"][-1].content

    verify_prompt = PromptTemplate(
        template=config.prompt("verify_prompt"),
        input_variables=["question", "context", "final_answer"]
    )
    llm_with_verification = llm.with_structured_output(VerificationResult)

    verify_chain = verify_prompt | llm_with_verification

    result = verify_chain.invoke({
        "question": question,
        "context": context,
        "final_answer": final_answer
    })

    if result.faithful:
        return "__end__"
    else:
        logger.info("Answer not faithful to context; regenerating")
        return "generate"
