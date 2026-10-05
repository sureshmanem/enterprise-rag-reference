"""Embedding and chat-model providers behind small interfaces.

- Azure OpenAI (Azure AI Foundry deployments) and OpenAI are interchangeable.
- Batching + exponential-backoff retries handle rate limits (HTTP 429) gracefully.
- `Fake*` providers make the whole pipeline runnable and testable offline (CI, demos).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import Settings

TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model_id: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


# ------------------------------------------------------------------ OpenAI-family


def _client(s: Settings):
    if s.provider == "azure":
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=s.azure_endpoint, api_key=s.azure_api_key, api_version=s.azure_api_version
        )
    from openai import OpenAI

    return OpenAI(api_key=s.openai_api_key)


def _retryable():
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(6),
        reraise=True,
    )


class OpenAIEmbedder:
    def __init__(self, s: Settings) -> None:
        self.client, self.model_id, self.batch = _client(s), s.embedding_model, s.embedding_batch_size
        self._call = _retryable()(self._embed_batch)

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model_id, input=batch)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._call([t.replace("\n", " ") for t in texts[i:i + self.batch]]))
        return out


class OpenAIChat:
    def __init__(self, s: Settings) -> None:
        self.client, self.model, self.temperature = _client(s), s.chat_model, s.temperature
        self._call = _retryable()(self._complete)

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    def complete(self, system: str, user: str) -> str:
        return self._call(system, user)


# ------------------------------------------------------------------ Offline fakes


STOPWORDS = frozenset(
    "a an and are as at be by can do does for from get how i in is it my of on or "
    "the to was what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    toks = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]  # crude plural fold


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedding. Not semantic, but stable and offline."""

    def __init__(self, dim: int = 512) -> None:
        self.dim, self.model_id = dim, f"fake-hash-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class FakeChat:
    """Extractive stand-in for an LLM: returns the most query-overlapping sentence, cited."""

    def complete(self, system: str, user: str) -> str:
        q = set(tokenize(user.split("Question:")[-1]))
        best, best_ref, best_score = "", 0, 0
        for m in re.finditer(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[\d+\]|\nQuestion:|\Z)", user, re.S):
            for sent in re.split(r"(?<=[.!?])\s+|\n", m.group(2)):
                score = len(q & set(tokenize(sent)))
                if score > best_score:
                    best, best_ref, best_score = sent.strip(), int(m.group(1)), score
        if not best:
            return "I don't know based on the provided documents."
        return f"{best} [{best_ref}]"


def build_embedder(s: Settings) -> Embedder:
    return FakeEmbedder() if s.provider == "fake" else OpenAIEmbedder(s)


def build_chat(s: Settings) -> ChatModel:
    return FakeChat() if s.provider == "fake" else OpenAIChat(s)
