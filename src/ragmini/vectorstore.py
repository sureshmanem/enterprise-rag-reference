"""Chroma-backed vector store with embedding-model pinning.

Best practices applied:
- Cosine distance, the standard for normalised text embeddings.
- The embedding model ID is stamped on the collection; querying with a different
  model raises instead of silently returning garbage (the #1 RAG footgun).
- Deterministic chunk IDs + upsert = idempotent re-ingestion.
- Stale chunks from an updated source file are removed on re-ingest.
"""

from __future__ import annotations

from typing import Any

import chromadb

from .models import Chunk


class EmbeddingModelMismatch(RuntimeError):
    pass


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    # Chroma only accepts scalar metadata values
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


class VectorStore:
    def __init__(self, persist_dir: str | None, collection: str, embedding_model: str) -> None:
        self.client = (
            chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        )
        self.col = self.client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model}
        )
        stored = (self.col.metadata or {}).get("embedding_model")
        if stored and stored != embedding_model:
            raise EmbeddingModelMismatch(
                f"Collection '{collection}' was built with '{stored}' but the configured "
                f"model is '{embedding_model}'. Re-index into a new collection."
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[_clean_meta({**c.metadata, "source": c.source}) for c in chunks],
        )

    def delete_stale(self, source: str, keep_ids: set[str]) -> int:
        existing = self.col.get(where={"source": source}, include=[])["ids"]
        stale = [i for i in existing if i not in keep_ids]
        if stale:
            self.col.delete(ids=stale)
        return len(stale)

    def query(self, embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        res = self.col.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            meta = dict(meta or {})
            out.append((Chunk(id=cid, text=text, source=meta.pop("source", "?"), metadata=meta),
                        1.0 - dist))
        return out

    def all_chunks(self) -> list[Chunk]:
        res = self.col.get(include=["documents", "metadatas"])
        chunks = []
        for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True):
            meta = dict(meta or {})
            chunks.append(Chunk(id=cid, text=text, source=meta.pop("source", "?"), metadata=meta))
        return chunks

    def count(self) -> int:
        return self.col.count()
