"""
Evaluate the Adaptive RAG backend end to end.

Uploads eval/sample_doc.txt, sends every question in eval/questions.json to
/rag/query (each in a fresh session), and uses an LLM judge (Groq) to score
each answer against a reference. Reports answer accuracy, routing accuracy,
query rewrites and latency, and saves per-question results as JSON.

Usage (backend must be running, see README):
    python eval/run_eval.py --label with-grader

To measure what the grader + rewriter add, restart the backend with
ENABLE_GRADER=false and run again with another label:
    python eval/run_eval.py --label without-grader
"""

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

HERE = Path(__file__).parent
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


class Verdict(BaseModel):
    correct: bool = Field(description="True if the answer matches the reference in substance")
    reason: str = Field(description="One short sentence explaining the verdict")


JUDGE_PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Mark the answer correct if it contains the key facts of the reference answer and
does not contradict it. Extra correct detail is fine. Wording does not matter.
For questions whose reference says it is judged on using web results, mark it
correct if the system gives a concrete, plausible answer instead of refusing."""


def upload(doc_path: Path) -> None:
    with open(doc_path, "rb") as f:
        r = requests.post(
            f"{BACKEND}/rag/documents/upload",
            files={"file": (doc_path.name, f, "text/plain")},
            headers={"X-Description": "Employee handbook of Nimbus Robotics, a warehouse robotics company"},
            timeout=300,
        )
    r.raise_for_status()
    if not r.json().get("status"):
        raise RuntimeError(f"Upload failed: {r.text}")


def ask(question: str) -> dict:
    t0 = time.perf_counter()
    r = requests.post(
        f"{BACKEND}/rag/query",
        json={"query": question, "session_id": f"eval-{uuid.uuid4()}"},
        timeout=300,
    )
    latency = time.perf_counter() - t0
    r.raise_for_status()
    body = r.json()
    return {
        "answer": body["result"]["content"],
        "route": body.get("route"),
        "rewrites": body.get("rewrites", 0),
        "latency_s": round(latency, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="name for this run, e.g. with-grader")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    questions = json.loads((HERE / "questions.json").read_text())
    judge = ChatGroq(model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0).with_structured_output(Verdict)

    if not args.skip_upload:
        print("Uploading sample document...")
        upload(HERE / "sample_doc.txt")

    results = []
    for q in questions:
        try:
            out = ask(q["question"])
            verdict = judge.invoke(JUDGE_PROMPT.format(
                question=q["question"], reference=q["reference"], answer=out["answer"]))
            out.update(correct=verdict.correct, judge_reason=verdict.reason)
        except Exception as e:  # keep going; count as wrong
            out = {"answer": None, "route": None, "rewrites": 0, "latency_s": None,
                   "correct": False, "judge_reason": f"error: {e}"}
        out.update(id=q["id"], type=q["type"], question=q["question"])
        results.append(out)
        print(f"{'PASS' if out['correct'] else 'FAIL'}  {q['id']}  route={out['route']}  "
              f"rewrites={out['rewrites']}  {out['latency_s']}s")

    def acc(rows):
        return 100.0 * sum(r["correct"] for r in rows) / len(rows) if rows else 0.0

    doc_rows = [r for r in results if r["type"] == "index"]
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    summary = {
        "label": args.label,
        "questions": len(results),
        "overall_accuracy_pct": round(acc(results), 1),
        "document_qa_accuracy_pct": round(acc(doc_rows), 1),
        "routing_accuracy_pct": round(
            100.0 * sum(r["route"] == r["type"] for r in results) / len(results), 1),
        "doc_questions_needing_rewrite": sum(r["rewrites"] > 0 for r in doc_rows),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "mean_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
    }

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.label}.json").write_text(json.dumps(
        {"summary": summary, "results": results}, indent=2))

    print("\n| Metric | Value |\n|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")
    print(f"\nSaved to eval/results/{args.label}.json")


if __name__ == "__main__":
    main()
