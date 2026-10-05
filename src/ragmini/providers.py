"""Embedding and chat-model providers behind small interfaces.

- Azure OpenAI (Azure AI Foundry deployments) and OpenAI are interchangeable.
- Batching + exponential-backoff retries handle rate limits (HTTP 429) gracefully.
"""

from __future__ import annotations

from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import Settings


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


def build_embedder(s: Settings) -> Embedder:
    return OpenAIEmbedder(s)


def build_chat(s: Settings) -> ChatModel:
    return OpenAIChat(s)
