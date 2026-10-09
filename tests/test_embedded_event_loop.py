"""
In embedded mode (Streamlit Cloud) the UI calls the async backend directly.
Motor, the async MongoDB driver, binds to the event loop it first runs on, so
every backend call must run on the same long-lived loop. This test uses a
stand-in history object that fails the way Motor does when the loop changes.
"""

import asyncio
import importlib
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamlit_app"))


class LoopBoundHistory:
    """Remembers the first event loop it is used on, like a Motor client."""

    loop = None
    store = {}

    def __init__(self, session_id):
        self.session_id = session_id

    def _check(self):
        running = asyncio.get_running_loop()
        if LoopBoundHistory.loop is None:
            LoopBoundHistory.loop = running
        elif running is not LoopBoundHistory.loop:
            raise RuntimeError("Event loop is closed")

    async def add_message(self, message):
        self._check()
        self.store.setdefault(self.session_id, []).append(message)

    async def get_messages(self):
        self._check()
        return list(self.store.get(self.session_id, []))

    async def clear(self):
        self._check()
        self.store.pop(self.session_id, None)


def test_backend_calls_share_one_event_loop(monkeypatch):
    monkeypatch.setenv("EMBEDDED_BACKEND", "true")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("TAVILY_API_KEY", "x")
    import utils.api_client as api
    api = importlib.reload(api)
    routes = api._load_backend()

    monkeypatch.setattr(routes, "get_session_history", LoopBoundHistory)
    monkeypatch.setattr(routes, "forget", lambda sid: None)
    monkeypatch.setattr(routes.builder, "invoke",
                        lambda state: {"messages": state["messages"] + [AIMessage(content="ok")]})

    api.get_session("s1")                       # page load restores the chat
    assert api.query_backend("hello", "s1") == "ok"
    assert api.query_backend("again", "s1") == "ok"
    assert len(api.get_session("s1")["history"]) == 4
    assert api.delete_document("s1") is True
