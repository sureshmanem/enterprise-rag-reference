"""LangSmith tracing.

Every question becomes one trace: `rag.ask` -> `retrieve` (embedding + hybrid search) -> `ChatOpenAI`,
with token usage, latency and the retrieved passages visible per step. Tracing is off unless
LANGSMITH_TRACING=true, and the SDK sends traces in a background thread, so it never blocks a request.
"""

from __future__ import annotations

import os

from .config import Settings


def configure_tracing(s: Settings) -> None:
    """Export LangSmith settings (possibly from .env) to the environment variables the SDK reads."""
    if not s.langsmith_tracing:
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key or "")
    os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
    if s.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", s.langsmith_endpoint)


def as_documents(results) -> dict:
    """Render retrieved chunks the way LangSmith's retriever view expects."""
    return {
        "documents": [
            {
                "type": "Document",
                "page_content": r.chunk.text,
                "metadata": {
                    "source": r.chunk.source,
                    "page": r.chunk.metadata.get("page"),
                    "score": round(r.score, 4),
                    "vector_similarity": r.vector_similarity,
                },
            }
            for r in (results or [])
        ]
    }
