import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.schemas.chunk import ChunkContextResponse, ChunkOut


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chunks", tags=["chunks"])


@router.get("/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str, db: Session = Depends(get_db)):
    chunk = crud.get_chunk(db, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return ChunkOut.from_orm(chunk)


@router.get("/{chunk_id}/context", response_model=ChunkContextResponse)
def get_chunk_context(
    chunk_id: str, before: int = 1, after: int = 1, db: Session = Depends(get_db)
):
    if before < 0 or after < 0:
        raise HTTPException(status_code=400, detail="before/after must be >= 0")
    center = crud.get_chunk(db, chunk_id)
    if not center:
        raise HTTPException(status_code=404, detail="Chunk not found")

    para_start = max(1, center.para_index - before)
    para_end = center.para_index + after
    context_chunks = crud.get_chunk_context(db, center.doc_id, para_start, para_end)

    context = [chunk for chunk in context_chunks if chunk.chunk_id != center.chunk_id]

    return ChunkContextResponse(
        center=ChunkOut.from_orm(center),
        context=[ChunkOut.from_orm(chunk) for chunk in context],
    )
