"""
In-memory chat history storage.

Used when MONGO_URI is not set, so the app runs locally without a database.
History is lost when the backend restarts.
"""

from typing import Dict, List

from langchain_core.messages import BaseMessage


class InMemoryChatMessageHistory:
    """Async chat history kept in process memory (same interface as the MongoDB one)."""

    _store: Dict[str, List[BaseMessage]] = {}

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def add_message(self, message: BaseMessage) -> None:
        self._store.setdefault(self.session_id, []).append(message)

    async def get_messages(self) -> List[BaseMessage]:
        return list(self._store.get(self.session_id, []))

    async def clear(self) -> None:
        self._store.pop(self.session_id, None)
