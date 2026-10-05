import pytest

from ragmini.config import Settings
from ragmini.loaders import load_directory
from ragmini.models import Document
from ragmini.pipeline import RAG
from ragmini.retrieval import tokenize
from ragmini.vectorstore import EmbeddingModelMismatch, VectorStore

from .conftest import SAMPLE


@pytest.mark.integration
def test_answers_with_citation(rag):
    ans = rag.ask("How many days of annual leave do full-time employees get?")
    assert "20 days" in ans.answer
    assert ans.grounded
    assert ans.citations[0].source == "leave_policy.md"


@pytest.mark.integration
def test_unknown_question_is_not_grounded(rag):
    ans = rag.ask("Who won the 1998 football world cup?")
    assert not ans.grounded


@pytest.mark.integration
def test_reingest_is_idempotent(rag):
    before = rag.store.count()
    stats = rag.ingest(load_directory(SAMPLE))
    assert rag.store.count() == before
    assert stats["stale_removed"] == 0


@pytest.mark.integration
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


@pytest.mark.integration
@pytest.mark.parametrize("policy_id, source", [
    ("HR-POL-021", "remote_work.md"),
    ("HR-POL-014", "leave_policy.md"),
    ("FIN-POL-012", "procurement_policy.md"),
    ("LEG-POL-001", "code_of_conduct.md"),
])
def test_keyword_search_finds_policy_ids(rag, policy_id, source):
    results = rag.retriever.retrieve(policy_id)
    assert results[0].chunk.source == source


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValueError):
        Settings(provider="azure")


@pytest.mark.integration
def test_empty_question_rejected(rag):
    with pytest.raises(ValueError):
        rag.ask("   ")


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert tokenize("How many leave days for HR-POL-021?") == [
        "many", "leave", "day", "hr", "pol", "021", "hr-pol-021"
    ]


def test_langsmith_tracing_requires_api_key():
    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        Settings(langsmith_tracing=True, langsmith_api_key=None)


@pytest.mark.integration
def test_ask_produces_trace_tree(rag):
    from unittest.mock import MagicMock

    import langsmith
    from langsmith import get_current_run_tree, traceable, tracing_context

    @traceable(name="probe")
    def probe():
        rag.ask("How many days of annual leave do full-time employees get?")
        return get_current_run_tree()

    with tracing_context(enabled=True, client=MagicMock(spec=langsmith.Client)):
        root = probe()

    ask = root.child_runs[0]
    assert ask.name == "rag.ask" and ask.metadata["chat_model"] == rag.s.chat_model
    assert [c.run_type for c in ask.child_runs] == ["retriever", "llm"]
    docs = ask.child_runs[0].outputs["documents"]
    assert docs[0]["metadata"]["source"] == "leave_policy.md"
