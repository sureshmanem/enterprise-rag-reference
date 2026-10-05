"""Typed data contracts shared across the pipeline."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A parsed source unit (a whole file, or one page of a PDF)."""

    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(source: str, text: str, index: int) -> str:
        """Deterministic ID -> re-ingesting the same content is an idempotent upsert."""
        digest = hashlib.sha256(f"{source}|{index}|{text}".encode()).hexdigest()
        return digest[:32]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(description="Fused relevance score (higher is better)")
    vector_similarity: float | None = None


class Citation(BaseModel):
    ref: int
    source: str
    page: int | None = None
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool = Field(description="False when no relevant context was found")
