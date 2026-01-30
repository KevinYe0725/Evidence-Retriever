from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChunkCreate(BaseModel):
    para_index: int
    char_start: int
    char_end: int
    text: str


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    para_index: int
    char_start: int
    char_end: int
    text: str
    is_deleted: int
    created_at: datetime

    class Config:
        orm_mode = True


class ChunkContextResponse(BaseModel):
    center: ChunkOut
    context: list[ChunkOut]


class CitationLocation(BaseModel):
    para_index: int
    char_start: int
    char_end: int


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    citation: str
    score: float
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    excerpt: str
    location: CitationLocation
