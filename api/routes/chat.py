from fastapi import APIRouter
from api.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    raise NotImplementedError
