from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.chunk import SearchResult


class SearchFilters(BaseModel):
    doc_ids: Optional[List[str]] = None
    tags_any: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = 20
    return_n: int = 6
    filters: Optional[SearchFilters] = None
    include_context: bool = False


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
