"""Hybrid retrieval: dense vectors + BM25 keywords, fused with Reciprocal Rank Fusion.

Why hybrid? Embeddings catch meaning ("time off" ~ "leave"), while BM25 catches exact
terms embeddings often miss (policy IDs, acronyms, product codes). RRF merges the two
ranked lists without needing to calibrate their very different score scales.
"""

from __future__ import annotations

import re

from langsmith import traceable
from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievedChunk
from .providers import Embedder
from .tracing import as_documents
from .vectorstore import VectorStore

RRF_K = 60  # standard constant from Cormack et al. (2009)

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Identifiers like HR-POL-021 or SKU-42: kept whole so an exact ID match outweighs shared parts.
IDENT_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from get how i in is it my of on or "
    "the to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Keyword tokens for BM25: lower-cased, stopwords dropped, crude plural fold, IDs kept whole."""
    text = text.lower()
    toks = [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS]
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks] + IDENT_RE.findall(text)


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
        candidate_k: int = 20,
        min_relevance: float = 0.25,
        hybrid: bool = True,
    ) -> None:
        self.store, self.embedder = store, embedder
        self.top_k, self.candidate_k = top_k, candidate_k
        self.min_relevance, self.hybrid = min_relevance, hybrid
        self._bm25: BM25Okapi | None = None
        self._corpus: list[Chunk] = []
        self._indexed_count = -1

    def _keyword_index(self) -> BM25Okapi | None:
        # Rebuild lazily when the store changed. For large corpora use a search engine
        # with native BM25 (e.g. Azure AI Search hybrid) instead of in-memory.
        if self.store.count() != self._indexed_count:
            self._corpus = self.store.all_chunks()
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self._corpus]) if self._corpus else None
            self._indexed_count = self.store.count()
        return self._bm25

    @traceable(
        run_type="retriever",
        name="retrieve",
        process_inputs=lambda i: {"query": i["query"]},
        process_outputs=as_documents,
    )
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = self.embedder.embed([query])[0]
        dense = [(c, s) for c, s in self.store.query(qvec, self.candidate_k) if s >= self.min_relevance]

        fused: dict[str, RetrievedChunk] = {}
        for rank, (chunk, sim) in enumerate(dense):
            fused[chunk.id] = RetrievedChunk(
                chunk=chunk, score=1 / (RRF_K + rank + 1), vector_similarity=sim
            )

        if self.hybrid and (bm25 := self._keyword_index()):
            scores = bm25.get_scores(tokenize(query))
            ranked = sorted(
                (i for i, s in enumerate(scores) if s > 0), key=lambda i: scores[i], reverse=True
            )[: self.candidate_k]
            for rank, idx in enumerate(ranked):
                chunk = self._corpus[idx]
                bonus = 1 / (RRF_K + rank + 1)
                if chunk.id in fused:
                    fused[chunk.id].score += bonus
                else:
                    fused[chunk.id] = RetrievedChunk(chunk=chunk, score=bonus)

        # A user who types an exact identifier wants that document: rank exact matches first.
        # (RRF alone can let near-identical IDs or dense noise outvote a perfect keyword hit.)
        idents = IDENT_RE.findall(query.lower())

        def exact(r: RetrievedChunk) -> bool:
            return bool(idents) and any(i in r.chunk.text.lower() for i in idents)

        results = sorted(fused.values(), key=lambda r: (exact(r), r.score), reverse=True)
        return self._dedupe(results)[: self.top_k]

    @staticmethod
    def _dedupe(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        out = []
        for r in results:
            key = r.chunk.text.strip().lower()[:200]
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out
