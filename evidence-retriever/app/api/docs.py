import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.chunker import chunk_text
from app.core.embedder import EmbeddingError
from app.core.vector_store import ChromaVectorStore
from app.db import crud
from app.db.models import Chunk, Doc
from app.db.session import get_db
from app.schemas.doc import DocCreate, DocCreateResponse, DocListResponse, DocOut


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/docs", tags=["docs"])


@router.post("", response_model=DocCreateResponse)
def create_doc(payload: DocCreate, request: Request, db: Session = Depends(get_db)):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    doc_id = str(uuid4())
    chunk_inputs = chunk_text(payload.text)
    if not chunk_inputs:
        raise HTTPException(status_code=400, detail="No chunks produced from text")

    chunk_models = []
    for chunk in chunk_inputs:
        chunk_models.append(
            Chunk(
                chunk_id=str(uuid4()),
                doc_id=doc_id,
                para_index=chunk.para_index,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
                is_deleted=0,
            )
        )

    doc_model = Doc(
        doc_id=doc_id,
        title=payload.title,
        tags_json=payload.tags,
        raw_text=payload.text,
        is_deleted=0,
    )

    embedder = request.app.state.embedder
    vector_store: ChromaVectorStore = request.app.state.vector_store

    try:
        with db.begin():
            crud.insert_doc_and_chunks(db, doc_model, chunk_models)
            db.flush()
            embeddings = embedder.embed_texts([chunk.text for chunk in chunk_models])
            if len(embeddings) != len(chunk_models):
                raise EmbeddingError("Embedding count mismatch")

            metadatas = [
                {
                    "doc_id": doc_id,
                    "para_index": chunk.para_index,
                    "is_deleted": 0,
                }
                for chunk in chunk_models
            ]
            vector_store.upsert_chunks(
                [chunk.chunk_id for chunk in chunk_models],
                embeddings,
                [chunk.text for chunk in chunk_models],
                metadatas,
            )
            embedding_dim = len(embeddings[0]) if embeddings else 0
            crud.insert_embedding_meta(
                db,
                doc_id,
                embedding_provider="qwen",
                embedding_model=embedder.model,
                embedding_dim=embedding_dim,
            )
    except EmbeddingError as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create document")
        raise HTTPException(status_code=500, detail="Failed to create document") from exc

    return DocCreateResponse(doc_id=doc_id, chunk_count=len(chunk_models))


@router.get("/{doc_id}", response_model=DocOut)
def get_doc(doc_id: str, db: Session = Depends(get_db)):
    result = crud.get_doc_with_chunk_count(db, doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    doc, chunk_count = result
    return DocOut(
        doc_id=doc.doc_id,
        title=doc.title,
        is_deleted=doc.is_deleted,
        chunk_count=chunk_count,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get("", response_model=DocListResponse)
def list_docs(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    rows = crud.list_docs(db, limit=limit, offset=offset)
    items = [
        DocOut(
            doc_id=doc.doc_id,
            title=doc.title,
            is_deleted=doc.is_deleted,
            chunk_count=chunk_count,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc, chunk_count in rows
    ]
    return DocListResponse(items=items, limit=limit, offset=offset)


@router.delete("/{doc_id}")
def delete_doc(doc_id: str, request: Request, db: Session = Depends(get_db)):
    doc = crud.get_doc(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store: ChromaVectorStore = request.app.state.vector_store
    chunk_ids = crud.get_chunk_ids_by_doc_id(db, doc_id)

    try:

            crud.mark_doc_deleted(db, doc_id, is_deleted=1)
            vector_store.mark_deleted(chunk_ids, is_deleted_flag=1)
            db.commit()
    except Exception as exc:
        logger.exception("Failed to delete document")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete document") from exc

    return {"doc_id": doc_id, "deleted": True}


@router.post("/{doc_id}/restore")
def restore_doc(doc_id: str, request: Request, db: Session = Depends(get_db)):
    doc = crud.get_doc(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store: ChromaVectorStore = request.app.state.vector_store
    chunk_ids = crud.get_chunk_ids_by_doc_id(db, doc_id)

    try:

            crud.mark_doc_deleted(db, doc_id, is_deleted=0)
            vector_store.mark_deleted(chunk_ids, is_deleted_flag=0)
            db.commit()
    except Exception as exc:
        logger.exception("Failed to restore document")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to restore document") from exc

    return {"doc_id": doc_id, "restored": True}
