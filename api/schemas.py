from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
