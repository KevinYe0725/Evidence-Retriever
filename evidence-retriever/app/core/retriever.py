import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.citation import build_citation_bundle
from app.core.embedder import QwenEmbedderClient
from app.core.reranker import QwenRerankerClient, RerankError
from app.core.vector_store import ChromaVectorStore
from app.db import crud


logger = logging.getLogger(__name__)


def _distance_to_score(value: Any, is_distance: bool) -> float:
    # Chroma may return distances (smaller is better). Convert to higher-is-better score.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return -numeric if is_distance else numeric


class Retriever:
    def __init__(
        self,
        embedder: QwenEmbedderClient,
        vector_store: ChromaVectorStore,
        reranker: Optional[QwenRerankerClient] = None,
        enable_rerank: bool = False,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.enable_rerank = enable_rerank

    def search(
        self,
        db: Session,
        query: str,
        top_k: int,
        return_n: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query)
        where = {"is_deleted": 0}
        # TODO: extend where conditions for doc_ids/tags filters when needed.

        raw = self.vector_store.search(query_embedding, top_k, where=where)
        ids = raw.get("ids", [])
        if not ids:
            return []
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            id_list = ids[0]
        else:
            id_list = ids
        if not id_list:
            return []

        distance_values = raw.get("distances")
        score_values = raw.get("scores")
        distance_mode = bool(distance_values)
        if distance_values:
            raw_scores = distance_values
        else:
            raw_scores = score_values or []
        if isinstance(raw_scores, list) and raw_scores and isinstance(raw_scores[0], list):
            score_list = raw_scores[0]
        else:
            score_list = raw_scores

        rows = crud.get_chunks_with_doc_titles(db, id_list)
        chunk_map = {chunk.chunk_id: (chunk, title) for chunk, title in rows}

        candidates: List[Dict[str, Any]] = []
        seen = set()
        for idx, chunk_id in enumerate(id_list):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in chunk_map:
                continue
            chunk, title = chunk_map[chunk_id]
            vector_score = None
            if idx < len(score_list):
                vector_score = _distance_to_score(score_list[idx], distance_mode)
            candidates.append(
                {
                    "chunk": chunk,
                    "title": title,
                    "vector_score": vector_score,
                }
            )

        rerank_scores: Optional[List[float]] = None
        if self.enable_rerank and self.reranker and candidates:
            try:
                rerank_scores = self.reranker.rerank(
                    query, [item["chunk"].text for item in candidates]
                )
                if len(rerank_scores) != len(candidates):
                    raise RerankError("Rerank score count mismatch")
            except RerankError as exc:
                logger.warning("Rerank failed, fallback to vector order: %s", exc)
                rerank_scores = None

        if rerank_scores is not None:
            for idx, score in enumerate(rerank_scores):
                candidates[idx]["rerank_score"] = float(score)
            ordered = sorted(
                candidates,
                key=lambda item: item.get("rerank_score", 0.0),
                reverse=True,
            )
        else:
            ordered = candidates

        results: List[Dict[str, Any]] = []
        for item in ordered:
            chunk = item["chunk"]
            title = item["title"]
            citation, excerpt, location = build_citation_bundle(title, chunk)
            vector_score = item.get("vector_score")
            rerank_score = item.get("rerank_score")
            final_score = rerank_score if rerank_score is not None else vector_score
            if final_score is None:
                final_score = 0.0
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "citation": citation,
                    "score": final_score,
                    "vector_score": vector_score,
                    "rerank_score": rerank_score,
                    "excerpt": excerpt,
                    "location": location,
                }
            )
            if len(results) >= return_n:
                break

        return results
