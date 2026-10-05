"""Typed configuration loaded from environment variables / .env.

Best practice: every tunable (chunk size, top-K, model names) lives in config,
never hard-coded, so experiments are reproducible and diffable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Providers: "openai" or "azure" (Azure OpenAI / AI Foundry)
    provider: Literal["azure", "openai"] = "openai"

    # Azure OpenAI / Azure AI Foundry
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_version: str = "2024-10-21"

    # OpenAI
    openai_api_key: str | None = None

    # Models (for Azure these are *deployment* names)
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_batch_size: int = Field(64, ge=1, le=2048)

    # Chunking
    chunk_size: int = Field(1000, ge=100, description="Max characters per chunk")
    chunk_overlap: int = Field(150, ge=0, description="Characters shared between chunks")

    # Retrieval
    top_k: int = Field(5, ge=1, le=50)
    candidate_k: int = Field(20, ge=1, description="Candidates per retriever before fusion")
    hybrid: bool = True
    min_relevance: float = Field(
        0.25, ge=0.0, le=1.0, description="Drop vector hits below this cosine similarity"
    )

    # Generation
    temperature: float = 0.0
    max_context_chars: int = 12000

    # Storage
    persist_dir: str = ".chroma"
    collection: str = "documents"

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if self.provider == "azure" and not (self.azure_endpoint and self.azure_api_key):
            raise ValueError("Azure provider needs RAG_AZURE_ENDPOINT and RAG_AZURE_API_KEY")
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI provider needs RAG_OPENAI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
