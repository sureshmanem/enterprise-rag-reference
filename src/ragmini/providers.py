"""Embedding and chat-model providers behind small interfaces.

- Azure OpenAI (Azure AI Foundry deployments) and OpenAI are interchangeable.
- Batching + exponential-backoff retries handle rate limits (HTTP 429) gracefully.
- Clients are wrapped for LangSmith, so chat calls are traced with token usage when tracing is on.
"""

from __future__ import annotations

from typing import Protocol

from langsmith import get_current_run_tree, traceable
from langsmith.wrappers import wrap_openai
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

        return wrap_openai(AzureOpenAI(
            azure_endpoint=s.azure_endpoint, api_key=s.azure_api_key, api_version=s.azure_api_version
        ))
    from openai import OpenAI

    return wrap_openai(OpenAI(api_key=s.openai_api_key))


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

    @traceable(
        run_type="embedding",
        name="embed",
        process_inputs=lambda i: {"count": len(i["texts"])},
        process_outputs=lambda o: {"count": len(o), "dim": len(o[0]) if o else 0},
    )
    def embed(self, texts: list[str]) -> list[list[float]]:
        if run := get_current_run_tree():
            run.add_metadata({"ls_model_name": self.model_id})
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
