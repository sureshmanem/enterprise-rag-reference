from pathlib import Path

import pytest

from ragmini.config import Settings
from ragmini.loaders import load_directory
from ragmini.pipeline import RAG

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(provider="fake", persist_dir=str(tmp_path / "db"), chunk_size=400,
                    chunk_overlap=60)


@pytest.fixture
def rag(settings) -> RAG:
    r = RAG(settings)
    r.ingest(load_directory(SAMPLE))
    return r
