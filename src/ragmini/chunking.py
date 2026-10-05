"""Recursive, structure-aware chunking with overlap.

Why not a fixed 1000-char slice? Cutting mid-sentence splits facts across chunks.
We try the largest natural boundary first (headings > paragraphs > lines >
sentences > words) and only fall back to a hard cut when nothing else fits.
Overlap keeps context that straddles a boundary retrievable from both sides.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from .models import Chunk, Document

SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep in text:
            pieces = text.split(sep)
            # re-attach the separator so headings / sentence ends are preserved
            pieces = [pieces[0]] + [sep.lstrip("\n") + p if sep.startswith("\n") else p
                                    for p in pieces[1:]]
            if sep == ". ":
                pieces = [p + "." if j < len(pieces) - 1 else p for j, p in enumerate(pieces)]
            out: list[str] = []
            for p in pieces:
                out.extend(_split(p, size, seps[i + 1:]) if len(p) > size else [p])
            return out
    return [text[j:j + size] for j in range(0, len(text), size)]  # last resort


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail} {piece}".strip() if tail else piece
            if len(current) > size:
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    return _merge(_split(text, size, SEPARATORS), size, overlap)


def _heading_for(text: str) -> str | None:
    m = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.M)
    return m.group(1).strip() if m else None


def chunk_documents(docs: Iterable[Document], size: int, overlap: int) -> Iterator[Chunk]:
    for doc in docs:
        last_heading = None
        for i, piece in enumerate(chunk_text(doc.text, size, overlap)):
            last_heading = _heading_for(piece) or last_heading
            meta = {**doc.metadata, "chunk_index": i}
            if last_heading:
                meta["section"] = last_heading
            yield Chunk(
                id=Chunk.make_id(doc.source, piece, i + 10_000 * meta.get("page", 0)),
                text=piece,
                source=doc.source,
                metadata=meta,
            )
