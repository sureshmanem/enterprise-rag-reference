import pytest

from ragmini.chunking import chunk_documents, chunk_text
from ragmini.models import Document


def test_chunks_respect_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_carries_context():
    text = " ".join(f"word{i}" for i in range(400))
    a, b = chunk_text(text, size=300, overlap=80)[:2]
    assert a.split()[-1] in b.split()[:20]


def test_prefers_paragraph_boundaries():
    text = ("Alpha " * 40).strip() + "\n\n" + ("Beta " * 40).strip()
    chunks = chunk_text(text, size=300, overlap=0)
    assert chunks[0].startswith("Alpha") and "Beta" not in chunks[0]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("x", size=100, overlap=100)


def test_ids_are_deterministic_and_metadata_kept():
    doc = Document(text="# Title\n\nSome body text.", source="a.md", metadata={"page": 2})
    c1 = list(chunk_documents([doc], 200, 20))
    c2 = list(chunk_documents([doc], 200, 20))
    assert [c.id for c in c1] == [c.id for c in c2]
    assert c1[0].metadata["page"] == 2
    assert c1[0].metadata["section"] == "Title"


def test_chunks_carry_document_title():
    body = "\n\n".join(f"## Section {i}\n\n" + "Some policy text here. " * 20 for i in range(5))
    doc = Document(text="# Travel Policy (FIN-POL-007)\n\n" + body, source="t.md")
    chunks = list(chunk_documents([doc], size=300, overlap=30))
    assert len(chunks) > 1
    assert {c.metadata["title"] for c in chunks} == {"Travel Policy (FIN-POL-007)"}
