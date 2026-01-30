import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api import chunks as chunks_api
from app.api import docs as docs_api
from app.api import search as search_api
from app.config import settings
from app.core.embedder import QwenEmbedderClient
from app.core.reranker import QwenRerankerClient
from app.core.retriever import Retriever
from app.core.vector_store import ChromaVectorStore
from app.db.session import SessionLocal


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Evidence Retrieval", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    embedder = QwenEmbedderClient(
        base_url=settings.QWEN_EMBEDDING_BASE_URL,
        api_key=settings.QWEN_EMBEDDING_API_KEY,
        model=settings.QWEN_EMBEDDING_MODEL,
        timeout_seconds=settings.QWEN_EMBEDDING_TIMEOUT_SECONDS,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION,
    )
    reranker = QwenRerankerClient(
        base_url=settings.QWEN_RERANK_BASE_URL,
        api_key=settings.QWEN_EMBEDDING_API_KEY,
        model=settings.QWEN_RERANK_MODEL,
        timeout_seconds=settings.QWEN_RERANK_TIMEOUT_SECONDS,
    )
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        enable_rerank=settings.ENABLE_RERANK,
    )

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.reranker = reranker
    app.state.retriever = retriever


@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": "error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"message": "error", "detail": "Internal Server Error"},
    )


@app.get("/health")
def health():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="MySQL not available") from exc

    try:
        app.state.vector_store.ping()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Chroma not available") from exc

    missing = []
    if not settings.QWEN_EMBEDDING_BASE_URL:
        missing.append("QWEN_EMBEDDING_BASE_URL")
    if not settings.QWEN_EMBEDDING_MODEL:
        missing.append("QWEN_EMBEDDING_MODEL")
    if missing:
        raise HTTPException(
            status_code=500, detail=f"Missing Qwen config: {', '.join(missing)}"
        )

    return {"status": "ok"}


app.include_router(docs_api.router)
app.include_router(search_api.router)
app.include_router(chunks_api.router)
