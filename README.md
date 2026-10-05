# rag-mini

A small, production-minded **Retrieval-Augmented Generation (RAG)** reference implementation in Python.
It ingests SharePoint, Word, PDF and Markdown documents, indexes them in a vector database, and answers
questions with an LLM that is grounded in, and cites, your own documents.

Runs fully **offline** out of the box (deterministic fake models) so anyone can clone, test and explore it.
Switch one environment variable to use **Azure OpenAI / Azure AI Foundry** or **OpenAI**.

## Architecture

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (one-time / scheduled)"]
        A[SharePoint · Word · PDF · MD] --> B[Parse & clean<br/>page-aware]
        B --> C[Recursive chunking<br/>size + overlap]
        C --> D[Embedding model]
    end
    subgraph Query["Query (every question)"]
        Q[User question] --> E[Embed query<br/>same model]
    end
    D --> V[(Vector DB<br/>Chroma, cosine)]
    E --> V
    Q --> K[BM25 keyword index]
    V --> F[Reciprocal Rank Fusion<br/>top-K]
    K --> F
    F --> P[Grounded prompt<br/>numbered context]
    P --> L[LLM]
    L --> R[Answer + citations]
```

## Best practices implemented

| Area | Practice | Where |
|---|---|---|
| Parsing | PDFs split per page so citations include page numbers; text normalised; empty/scanned pages reported | `loaders.py` |
| Parsing | SharePoint via Microsoft Graph with app-only auth, secrets from env, pagination handled | `loaders.py` |
| Chunking | Recursive, structure-aware splitting (headings → paragraphs → sentences → words), never mid-word | `chunking.py` |
| Chunking | Configurable overlap so facts on a boundary stay retrievable; section heading kept as metadata | `chunking.py` |
| Indexing | Deterministic chunk IDs → idempotent upserts; stale chunks deleted when a source changes | `vectorstore.py`, `pipeline.py` |
| Embeddings | **Embedding model pinned to the collection** – querying with a different model fails loudly | `vectorstore.py` |
| Embeddings | Batching + exponential backoff with jitter for rate limits (429) and transient errors | `providers.py` |
| Retrieval | **Hybrid search**: dense vectors + BM25 keywords fused with RRF (catches IDs like `HR-POL-021`) | `retrieval.py` |
| Retrieval | Minimum relevance threshold and near-duplicate removal before top-K | `retrieval.py` |
| Generation | Answer only from context, explicit "I don't know" fallback, mandatory `[n]` citations | `prompting.py` |
| Generation | Context treated as data (prompt-injection hygiene) and trimmed to a character budget | `prompting.py` |
| Output | Typed `Answer` with citations and a `grounded` flag (Pydantic contracts end to end) | `models.py` |
| Config | All tunables in validated settings (`RAG_*` env vars / `.env`) | `config.py` |
| Quality | Golden-set evaluation: hit-rate@K, MRR, answer accuracy, groundedness | `eval/` |
| Quality | Offline unit tests + GitHub Actions CI | `tests/`, `.github/` |
| Ops | Timing logs for retrieval vs generation; `/health` endpoint; Dockerfile | `pipeline.py`, `api.py` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag ingest data/sample
rag ask "How many leave days do I get per year?"
# Full-time employees receive 20 days of paid annual leave per calendar year. [1]
#   [1] leave_policy.md

rag serve            # chat UI at http://127.0.0.1:8000
pytest               # run the tests
python eval/run_eval.py              # hybrid retrieval
python eval/run_eval.py --no-hybrid  # vector-only, for comparison
```

### Use Azure OpenAI (Azure AI Foundry)

```bash
cp .env.example .env   # then fill in:
RAG_PROVIDER=azure
RAG_AZURE_ENDPOINT=https://<resource>.openai.azure.com/
RAG_AZURE_API_KEY=<key>
RAG_EMBEDDING_MODEL=<embedding-deployment-name>
RAG_CHAT_MODEL=<chat-deployment-name>
RAG_COLLECTION=documents_te3small   # name collections after the embedding model
```

Changing the embedding model requires re-indexing into a new collection; the store enforces this.

### Index SharePoint

Register an Entra ID app with the `Sites.Read.All` application permission, set `SP_TENANT_ID`,
`SP_CLIENT_ID`, `SP_CLIENT_SECRET`, then:

```bash
rag ingest-sharepoint --hostname contoso.sharepoint.com --site sites/HR --folder Policies
```

## Evaluation results (sample data, offline fake models)

| Mode | hit-rate@5 | MRR | answer accuracy |
|---|---|---|---|
| Hybrid (vector + BM25) | 1.0 | 1.0 | 0.8 |
| Vector only | 0.7 | 0.7 | 0.6 |

The fake embedder is a hashed bag-of-words, so these numbers demonstrate the *method*, not real model
quality. Re-run with your Azure deployments and your own golden set to tune chunk size, overlap and top-K.

## Project layout

```
src/ragmini/
  config.py      typed settings
  models.py      Document, Chunk, Answer contracts
  loaders.py     PDF / DOCX / MD / SharePoint
  chunking.py    recursive chunker with overlap
  providers.py   Azure OpenAI, OpenAI, offline fakes
  vectorstore.py Chroma with model pinning
  retrieval.py   hybrid retrieval + RRF
  prompting.py   grounded prompt + citations
  pipeline.py    ingest() and ask()
  api.py / cli.py
eval/            golden set + metrics
tests/           offline unit tests
```

## Production next steps

- Swap Chroma for **Azure AI Search** (native hybrid + semantic ranker) at scale.
- Add a **cross-encoder re-ranker** after fusion.
- Enforce **document-level permissions** (security trimming) using SharePoint ACLs as metadata filters.
- Incremental sync with Graph delta queries; OCR for scanned PDFs.
- Tracing (OpenTelemetry) and LLM-as-judge faithfulness evaluation.

## License

MIT
