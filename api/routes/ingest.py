from fastapi import APIRouter, UploadFile

router = APIRouter()


@router.post("/")
async def ingest_document(file: UploadFile) -> dict:
    raise NotImplementedError
