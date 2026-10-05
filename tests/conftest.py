import os
from pathlib import Path

import pytest

from ragmini.config import Settings
from ragmini.loaders import load_directory
from ragmini.pipeline import RAG

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def _has_credentials() -> bool:
    try:
        Settings()
    except ValueError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Integration tests call the real model API; skip them when no credentials are set."""
    if _has_credentials() and not os.getenv("RAG_SKIP_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="needs RAG_OPENAI_API_KEY (or Azure settings)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(persist_dir=str(tmp_path / "db"), chunk_size=400, chunk_overlap=60)


@pytest.fixture
def rag(settings) -> RAG:
    r = RAG(settings)
    r.ingest(load_directory(SAMPLE))
    return r
