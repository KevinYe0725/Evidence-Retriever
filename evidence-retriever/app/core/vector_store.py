from typing import Any, Dict, List, Optional

import chromadb


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        self.client = self._init_client(persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _init_client(self, persist_directory: str):
        try:
            return chromadb.PersistentClient(persist_directory=persist_directory)
        except TypeError:
            return chromadb.PersistentClient(path=persist_directory)

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not chunk_ids:
            return
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or {},
            include=[ "distances", "metadatas", "documents"],
        )

    def mark_deleted(self, chunk_ids: List[str], is_deleted_flag: int) -> None:
        if not chunk_ids:
            return
        existing = self.collection.get(
            ids=chunk_ids, include=["embeddings", "documents", "metadatas"]
        )
        ids = existing.get("ids", [])
        if not ids:
            return
        embeddings = existing.get("embeddings", [])
        documents = existing.get("documents", [])
        metadatas = existing.get("metadatas", [])
        updated_metadatas = []
        for meta in metadatas:
            new_meta = dict(meta or {})
            new_meta["is_deleted"] = int(is_deleted_flag)
            updated_metadatas.append(new_meta)
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=updated_metadatas,
        )

    def ping(self) -> bool:
        _ = self.collection.count()
        return True
