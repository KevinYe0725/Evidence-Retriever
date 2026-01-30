import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.embedder import EmbeddingError
from app.core.retriever import Retriever
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request, db: Session = Depends(get_db)):
    retriever: Retriever = request.app.state.retriever
    try:
        results = retriever.search(
            db,
            query=payload.query,
            top_k=payload.top_k,
            return_n=payload.return_n,
            filters=payload.filters.dict() if payload.filters else None,
        )
    except EmbeddingError as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed") from exc

    return SearchResponse(query=payload.query, results=results)
