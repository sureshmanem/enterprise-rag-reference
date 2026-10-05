"""Retrieval + answer evaluation against a small golden set.

Metrics:
- hit_rate@k : the expected source appears in the top-k retrieved chunks
- MRR        : mean reciprocal rank of the first chunk from the expected source
- answer_acc : the generated answer contains the expected key fact
- grounded   : the answer cites at least one passage

Run:  python eval/run_eval.py --docs data/sample
Use this to compare chunk sizes, top-K, hybrid on/off, or embedding models objectively.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/sample")
    p.add_argument("--golden", default="eval/golden.jsonl")
    p.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    golden = [json.loads(line) for line in Path(args.golden).read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(persist_dir=tmp, hybrid=args.hybrid)
        rag = RAG(settings)
        rag.ingest(load_directory(args.docs))
        print(json.dumps(evaluate(rag, golden), indent=2))


if __name__ == "__main__":
    main()
