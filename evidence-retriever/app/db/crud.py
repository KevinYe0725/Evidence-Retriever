from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Chunk, Doc, EmbeddingMeta


def insert_doc_and_chunks(db: Session, doc: Doc, chunks: Sequence[Chunk]) -> None:
    db.add(doc)
    db.add_all(list(chunks))


def insert_embedding_meta(
    db: Session,
    doc_id: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dim: int,
) -> None:
    meta = EmbeddingMeta(
        doc_id=doc_id,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    db.add(meta)


def get_doc(db: Session, doc_id: str) -> Optional[Doc]:
    return db.query(Doc).filter(Doc.doc_id == doc_id).first()


def get_doc_with_chunk_count(db: Session, doc_id: str) -> Optional[Tuple[Doc, int]]:
    row = (
        db.query(Doc, func.count(Chunk.chunk_id))
        .outerjoin(Chunk, Chunk.doc_id == Doc.doc_id)
        .filter(Doc.doc_id == doc_id)
        .group_by(Doc.doc_id)
        .first()
    )
    if not row:
        return None
    return row[0], int(row[1])


def list_docs(
    db: Session, limit: int = 20, offset: int = 0
) -> List[Tuple[Doc, int]]:
    rows = (
        db.query(Doc, func.count(Chunk.chunk_id))
        .outerjoin(Chunk, Chunk.doc_id == Doc.doc_id)
        .group_by(Doc.doc_id)
        .order_by(Doc.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [(row[0], int(row[1])) for row in rows]


def get_chunks_by_doc_id(db: Session, doc_id: str) -> List[Chunk]:
    return db.query(Chunk).filter(Chunk.doc_id == doc_id).all()


def get_chunk_ids_by_doc_id(db: Session, doc_id: str) -> List[str]:
    rows = db.query(Chunk.chunk_id).filter(Chunk.doc_id == doc_id).all()
    return [row[0] for row in rows]


def get_chunks_by_ids(db: Session, chunk_ids: Iterable[str]) -> List[Chunk]:
    if not chunk_ids:
        return []
    return db.query(Chunk).filter(Chunk.chunk_id.in_(list(chunk_ids))).all()


def get_chunks_with_doc_titles(
    db: Session, chunk_ids: Iterable[str]
) -> List[Tuple[Chunk, str]]:
    if not chunk_ids:
        return []
    return (
        db.query(Chunk, Doc.title)
        .join(Doc, Doc.doc_id == Chunk.doc_id)
        .filter(Chunk.chunk_id.in_(list(chunk_ids)))
        .filter(Chunk.is_deleted == 0)
        .filter(Doc.is_deleted == 0)
        .all()
    )


def get_chunk(db: Session, chunk_id: str) -> Optional[Chunk]:
    return db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()


def get_chunk_context(
    db: Session, doc_id: str, para_start: int, para_end: int
) -> List[Chunk]:
    return (
        db.query(Chunk)
        .filter(Chunk.doc_id == doc_id)
        .filter(Chunk.para_index >= para_start)
        .filter(Chunk.para_index <= para_end)
        .order_by(Chunk.para_index.asc(), Chunk.char_start.asc())
        .all()
    )


def mark_doc_deleted(db: Session, doc_id: str, is_deleted: int) -> None:
    now = datetime.utcnow()
    db.query(Doc).filter(Doc.doc_id == doc_id).update(
        {Doc.is_deleted: is_deleted, Doc.updated_at: now}, synchronize_session=False
    )
    db.query(Chunk).filter(Chunk.doc_id == doc_id).update(
        {Chunk.is_deleted: is_deleted}, synchronize_session=False
    )
