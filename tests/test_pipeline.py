import pytest

from ragmini.config import Settings
from ragmini.loaders import load_directory
from ragmini.models import Document
from ragmini.pipeline import RAG
from ragmini.providers import FakeEmbedder
from ragmini.vectorstore import EmbeddingModelMismatch, VectorStore

from .conftest import SAMPLE


def test_answers_with_citation(rag):
    ans = rag.ask("How many days of annual leave do full-time employees get?")
    assert "20 days" in ans.answer
    assert ans.grounded
    assert ans.citations[0].source == "leave_policy.md"


def test_unknown_question_is_not_grounded(rag):
    ans = rag.ask("Who won the 1998 football world cup?")
    assert not ans.grounded


def test_reingest_is_idempotent(rag):
    before = rag.store.count()
    stats = rag.ingest(load_directory(SAMPLE))
    assert rag.store.count() == before
    assert stats["stale_removed"] == 0


def test_updated_source_removes_stale_chunks(settings):
    r = RAG(settings)
    r.ingest([Document(text="Old policy: 10 days of leave.", source="p.md")])
    r.ingest([Document(text="New policy: 25 days of leave.", source="p.md")])
    texts = [c.text for c in r.store.all_chunks()]
    assert texts == ["New policy: 25 days of leave."]


def test_embedding_model_mismatch_is_blocked(tmp_path):
    VectorStore(str(tmp_path), "docs", "model-a")
    with pytest.raises(EmbeddingModelMismatch):
        VectorStore(str(tmp_path), "docs", "model-b")


def test_keyword_search_finds_policy_ids(rag):
    results = rag.retriever.retrieve("HR-POL-021")
    assert results[0].chunk.source == "remote_work.md"


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValueError):
        Settings(provider="azure")


def test_empty_question_rejected(rag):
    with pytest.raises(ValueError):
        rag.ask("   ")


def test_fake_embedder_normalised():
    v = FakeEmbedder().embed(["hello world"])[0]
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6
