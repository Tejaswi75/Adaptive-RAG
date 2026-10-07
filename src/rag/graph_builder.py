"""
Graph builder module for the adaptive RAG system.
"""

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph.state import StateGraph

from src.rag.reAct_agent import get_agent_executor
from src.rag.retriever_setup import get_retriever
from src.config.settings import Config
from src.core.logger import get_logger
from src.llms.groq_llm import llm
import re
from src.models.state import State
from src.tools.graph_tools import routing_tool, doc_tool, GRADER_ENABLED

config = Config()
logger = get_logger(__name__)


def first_label(text: str, labels: tuple, default: str) -> str:
    """
    Return whichever label appears first (as a whole word) in an LLM reply.

    Parsing plain text instead of forcing tool-calling structured output keeps
    the graph working with models that answer with a bare word.
    """
    found = [
        (m.start(), label)
        for label in labels
        if (m := re.search(rf"\b{label}\b", (text or "").lower()))
    ]
    return min(found)[1] if found else default


# Node implementations
def query_classifier(state: State):

    question = state["messages"][-1].content

    retriever = get_retriever()
    context = retriever.invoke(question)

    logger.debug("Question: %s", question)
    logger.debug("Retrieved context: %s", context)

    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"]
    )
    chain = classify_prompt | llm
    reply = chain.invoke({"question": question, "context": context}).content
    route = first_label(reply, ("index", "general", "search"), default="general")
    logger.info("Query routed to: %s", route)

    return {
        "messages": state["messages"],
        "route": route,
        "latest_query": question,
        "rewrite_count": 0,
    }


def general_llm(state: State):
    """
    Fetch general common knowledge result from the LLM.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages from LLM.
    """
    result = llm.invoke(state["messages"])
    logger.debug("General LLM answer: %s", result.content)
    return {"messages": result}


def retriever_node(state: State):
    """
    Retrieve results from vector stores using the reAct agent.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated messages with tool calls.
    """
    messages = state["latest_query"]
    agent_executor = get_agent_executor()
    result = agent_executor.invoke({"input": messages})

    # Extract tool calls
    intermediate_steps = result.get("intermediate_steps", [])
    tool_calls = []
    if intermediate_steps:
        for action, tool_result in intermediate_steps:
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
            })

    new_message = AIMessage(
        content=result["output"],
        additional_kwargs={"tool_calls": tool_calls},
    )

    return {
        "messages": [new_message]
    }


def grade(state: State):
    """
    Grade the results retrieved from vector stores.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Updated state with binary_score.
    """
    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"]
    )
    context = state["messages"][-1].content
    question = state["latest_query"]

    chain_graded = grading_prompt | llm
    reply = chain_graded.invoke({"question": question, "context": context}).content
    score = first_label(reply, ("yes", "no"), default="no")

    logger.info("Relevance grade: %s", score)
    return {"messages": state["messages"], "binary_score": score}


def rewrite_query(state: State):
    """
    Rewrite the query to get better retrieval results.

    Args:
        state (State): State of the question.

    Returns:
        dict: Updated latest_query.
    """
    query = state["latest_query"]
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"]
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"query": query})
    rewrites = (state.get("rewrite_count") or 0) + 1
    logger.info("Rewrite #%d: %s", rewrites, result.content)

    return {
        "latest_query": result.content,
        "rewrite_count": rewrites,
    }


def generate(state: State):
    """
    Generate the final answer for the user.

    Args:
        state (State): State of the question.

    Returns:
        dict: Generated response.
    """
    context = state["messages"][-1].content

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["question", "context"]
    )

    generate_chain = generate_prompt | llm
    result = generate_chain.invoke({"question": state["latest_query"], "context": context})

    return {"messages": [{"role": "assistant", "content": result.content}]}


def web_search(state: State):
    """
    Search the web for the rewritten query.

    Args:
        state (State): The current state of the graph.

    Returns:
        dict: Search results as messages.
    """
    # Initialize the Tavily tool
    search_tool = TavilySearchResults()

    # Search a query
    result = search_tool.invoke(state["latest_query"])

    contents = [item["content"] for item in result if "content" in item]
    logger.debug("Web search returned %d results", len(contents))

    return {
        "messages": [{"role": "assistant", "content": "\n\n".join(contents)}]
    }


# Build the graph
graph = StateGraph(State)

graph.add_node("query_analysis", query_classifier)
graph.add_node("retriever", retriever_node)
graph.add_node("generate", generate)
graph.add_node("web_search", web_search)
graph.add_node("general_llm", general_llm)

graph.add_edge(START, "query_analysis")
graph.add_edge("web_search", "generate")
graph.add_conditional_edges("query_analysis", routing_tool)

if GRADER_ENABLED:
    # Self-correcting loop: grade -> rewrite -> retrieve, falling back to web search
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite_query)
    graph.add_edge("retriever", "grade")
    graph.add_edge("rewrite", "retriever")
    graph.add_conditional_edges("grade", doc_tool)
else:
    # Ablation mode (ENABLE_GRADER=false): answer straight from the first retrieval
    graph.add_edge("retriever", "generate")
graph.add_edge("generate", END)
graph.add_edge("general_llm", END)

builder = graph.compile()

