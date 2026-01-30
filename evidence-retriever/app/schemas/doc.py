from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocCreate(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None


class DocCreateResponse(BaseModel):
    doc_id: str
    chunk_count: int


class DocOut(BaseModel):
    doc_id: str
    title: str
    is_deleted: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class DocListResponse(BaseModel):
    items: List[DocOut]
    limit: int
    offset: int
