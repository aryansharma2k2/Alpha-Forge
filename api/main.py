"""
FastAPI entry point.
"""
from fastapi import FastAPI
from api.routes import chat, ingest, health

app = FastAPI(title="Financial Research RAG", version="0.1.0")

app.include_router(health.router)
app.include_router(chat.router, prefix="/chat")
app.include_router(ingest.router, prefix="/ingest")
