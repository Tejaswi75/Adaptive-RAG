"""
Groq LLM initialization and configuration.

Set GROQ_MODEL in .env to switch models (see https://console.groq.com/docs/models).
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=GROQ_MODEL,
)
