"""
Core configuration and environment settings.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")


settings = Settings()

# Set env variables for LangChain integrations
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
