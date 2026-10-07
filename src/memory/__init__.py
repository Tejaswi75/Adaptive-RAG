"""
Chat history backends.

MongoDB is used when MONGO_URI is set; otherwise history is kept in memory.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_session_history(session_id: str):
    """Return the chat history object for a session."""
    if os.getenv("MONGO_URI"):
        # Imported lazily: creating the Mongo client resolves the cluster's DNS.
        from src.memory.chat_history_mongo import ChatHistory
        return ChatHistory.get_session_history(session_id)

    from src.memory.chathistory_in_memory import InMemoryChatMessageHistory
    return InMemoryChatMessageHistory(session_id)
