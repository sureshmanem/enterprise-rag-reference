"""Retrieval + answer evaluation against a small golden set.

Metrics:
- hit_rate@k : the expected source appears in the top-k retrieved chunks
- MRR        : mean reciprocal rank of the first chunk from the expected source
- answer_acc : the generated answer contains the expected key fact
- grounded   : the answer cites at least one passage

Run:  python eval/run_eval.py --docs data/sample
      python eval/run_eval.py --langsmith   # also log an experiment to LangSmith
Use this to compare chunk sizes, top-K, hybrid on/off, or embedding models objectively.
With --langsmith the golden set is synced to a LangSmith dataset and each run becomes an
experiment, so configurations can be compared side by side in the LangSmith UI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragmini.config import Settings  # noqa: E402
from ragmini.loaders import load_directory  # noqa: E402
from ragmini.pipeline import RAG  # noqa: E402


def evaluate(rag: RAG, golden: list[dict]) -> dict[str, float]:
    hits = rr = correct = grounded = 0
    for case in golden:
        results = rag.retriever.retrieve(case["question"])
        sources = [r.chunk.source for r in results]
        if case["expected_source"] in sources:
            hits += 1
            rr += 1 / (sources.index(case["expected_source"]) + 1)
        ans = rag.ask(case["question"])
        correct += case["expected_answer_contains"].lower() in ans.answer.lower()
        grounded += ans.grounded
    n = len(golden)
    return {
        f"hit_rate@{rag.s.top_k}": round(hits / n, 3),
        "mrr": round(rr / n, 3),
        "answer_acc": round(correct / n, 3),
        "grounded": round(grounded / n, 3),
        "n": n,
    }


def sync_dataset(client, name: str, golden: list[dict]):
    """Create the LangSmith dataset from the golden set, adding any questions it lacks."""
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="rag-mini golden set")
    known = {ex.inputs["question"] for ex in client.list_examples(dataset_id=dataset.id)}
    new = [c for c in golden if c["question"] not in known]
    if new:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": c["question"]} for c in new],
            outputs=[{k: c[k] for k in ("expected_source", "expected_answer_contains")} for c in new],
        )
    return dataset


def git_context() -> dict[str, str]:
    """Commit, branch and origin of this run, so experiments can be traced back to code."""
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    ctx = {
        "commit": os.getenv("GITHUB_SHA") or git("rev-parse", "HEAD"),
        "branch": os.getenv("GITHUB_HEAD_REF") or os.getenv("GITHUB_REF_NAME")
        or git("rev-parse", "--abbrev-ref", "HEAD"),
        "source": "github-actions" if os.getenv("GITHUB_ACTIONS") else "local",
    }
    if os.getenv("GITHUB_RUN_ID"):
        ctx["ci_run_url"] = (f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
                             f"/actions/runs/{os.getenv('GITHUB_RUN_ID')}")
    return ctx


def run_langsmith(rag: RAG, golden: list[dict], dataset_name: str) -> None:
    from langsmith import Client, evaluate

    if not rag.s.langsmith_api_key:
        sys.exit("--langsmith needs LANGSMITH_API_KEY (in the environment or .env)")
    client = Client(api_key=rag.s.langsmith_api_key, api_url=rag.s.langsmith_endpoint)
    sync_dataset(client, dataset_name, golden)

    def target(inputs: dict) -> dict:
        ans = rag.ask(inputs["question"])
        sources = [r.chunk.source for r in rag.retriever.retrieve(inputs["question"])]
        return {"answer": ans.answer, "grounded": ans.grounded, "sources": sources}

    def hit(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": f"hit_rate@{rag.s.top_k}",
                "score": reference_outputs["expected_source"] in outputs["sources"]}

    def reciprocal_rank(outputs: dict, reference_outputs: dict) -> dict:
        src = reference_outputs["expected_source"]
        rr = 1 / (outputs["sources"].index(src) + 1) if src in outputs["sources"] else 0.0
        return {"key": "mrr", "score": rr}

    def answer_acc(outputs: dict, reference_outputs: dict) -> dict:
        return {"key": "answer_acc",
                "score": reference_outputs["expected_answer_contains"].lower() in outputs["answer"].lower()}

    def grounded(outputs: dict) -> dict:
        return {"key": "grounded", "score": outputs["grounded"]}

    s, git = rag.s, git_context()
    mode = "hybrid" if s.hybrid else "vector"
    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[hit, reciprocal_rank, answer_acc, grounded],
        experiment_prefix=f"{s.chat_model}-{mode}-k{s.top_k}-{git['commit'][:7]}",
        metadata={"embedding_model": s.embedding_model, "chat_model": s.chat_model, "top_k": s.top_k,
                  "hybrid": s.hybrid, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap, **git},
        client=client,
    )

    scores: dict[str, list[float]] = {}
    for row in results:
        for r in row["evaluation_results"]["results"]:
            scores.setdefault(r.key, []).append(float(r.score or 0))
    summary = {k: round(sum(v) / len(v), 3) for k, v in sorted(scores.items())}
    print(json.dumps({**summary, "n": len(golden)}, indent=2))
    print(f"LangSmith experiment: {results.experiment_name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/sample")
    p.add_argument("--golden", default="eval/golden.jsonl")
    p.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--langsmith", action="store_true", help="log an experiment to LangSmith")
    p.add_argument("--dataset", default="rag-mini-golden", help="LangSmith dataset name")
    args = p.parse_args()

    golden = [json.loads(line) for line in Path(args.golden).read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(persist_dir=tmp, hybrid=args.hybrid)
        rag = RAG(settings)
        rag.ingest(load_directory(args.docs))
        if args.langsmith:
            run_langsmith(rag, golden, args.dataset)
        else:
            print(json.dumps(evaluate(rag, golden), indent=2))


if __name__ == "__main__":
    main()
