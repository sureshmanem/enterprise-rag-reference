"""End-to-end orchestration: ingestion pipeline and question-answering service."""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from collections.abc import Iterable

from .chunking import chunk_documents
from .config import Settings
from .models import Answer, Chunk, Citation, Document
from .prompting import SYSTEM_PROMPT, build_user_prompt, format_context
from .providers import ChatModel, Embedder, build_chat, build_embedder
from .retrieval import HybridRetriever
from .vectorstore import VectorStore

log = logging.getLogger(__name__)
NO_ANSWER = "I don't know based on the provided documents."


class RAG:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        chat: ChatModel | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.s = settings
        self.embedder = embedder or build_embedder(settings)
        self.chat = chat or build_chat(settings)
        self.store = store or VectorStore(
            settings.persist_dir, settings.collection, self.embedder.model_id
        )
        self.retriever = HybridRetriever(
            self.store, self.embedder, settings.top_k, settings.candidate_k,
            settings.min_relevance, settings.hybrid,
        )

    # ------------------------------------------------------------- ingestion
    def ingest(self, docs: Iterable[Document]) -> dict[str, int]:
        t0 = time.perf_counter()
        chunks = list(chunk_documents(docs, self.s.chunk_size, self.s.chunk_overlap))
        unique: dict[str, Chunk] = {c.id: c for c in chunks}  # exact-duplicate removal
        chunks = list(unique.values())

        by_source: dict[str, set[str]] = defaultdict(set)
        for c in chunks:
            by_source[c.source].add(c.id)

        embeddings = self.embedder.embed([c.text for c in chunks]) if chunks else []
        self.store.upsert(chunks, embeddings)
        removed = sum(self.store.delete_stale(src, ids) for src, ids in by_source.items())

        stats = {
            "sources": len(by_source),
            "chunks": len(chunks),
            "stale_removed": removed,
            "total_in_store": self.store.count(),
        }
        log.info("Ingested %s in %.2fs", stats, time.perf_counter() - t0)
        return stats

    # ------------------------------------------------------------- answering
    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        t0 = time.perf_counter()
        results = self.retriever.retrieve(question)
        if not results:
            return Answer(question=question, answer=NO_ANSWER, citations=[], grounded=False)

        context, used = format_context(results, self.s.max_context_chars)
        t1 = time.perf_counter()
        text = self.chat.complete(SYSTEM_PROMPT, build_user_prompt(question, context)).strip()
        t2 = time.perf_counter()

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(used)})
        citations = [
            Citation(
                ref=n,
                source=used[n - 1].chunk.source,
                page=used[n - 1].chunk.metadata.get("page"),
                snippet=used[n - 1].chunk.text[:200],
            )
            for n in cited
        ]
        log.info("retrieval=%.0fms generation=%.0fms hits=%d cited=%s",
                 (t1 - t0) * 1000, (t2 - t1) * 1000, len(used), cited)
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            grounded=bool(citations) and NO_ANSWER not in text,
        )
