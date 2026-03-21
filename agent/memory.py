"""
Conversation memory / session store.
"""


def get_history(session_id: str) -> list[dict]:
    raise NotImplementedError


def append(session_id: str, role: str, content: str) -> None:
    raise NotImplementedError
