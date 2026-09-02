"""Usage:
    python rag/evaluate.py --retrieval-only   # free, no API calls for generation
    python rag/evaluate.py                    # full run, 1 chat call per question
"""
import argparse
import re
import sys
from pathlib import Path

# Script inside a package: the project root must be importable before rag.src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from goldens import CASES  # noqa: E402
except ModuleNotFoundError:
    sys.exit(
        "rag/goldens.py not found.\n\n"
        "This script scores the pipeline against a document you provide:\n"
        "  1. put the document at rag/sample.md\n"
        "  2. write your questions in rag/goldens.py as CASES = [...]\n\n"
        "See 'Measuring accuracy' in README.md for the fields a case takes."
    )

from rag.src.rag_pipeline import (  # noqa: E402
    CANDIDATES_PER_RETRIEVER,
    CHAT_MODEL,
    FINAL_TOP_K,
    REFUSAL_MARKER,
    ask,
    build_retriever,
    check_citations,
    load_document,
)

DOCUMENT_PATH = str(Path(__file__).parent / "sample.md")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def rank_of_evidence(docs: list, evidence: str) -> int | None:
    needle = normalise(evidence)
    for i, doc in enumerate(docs, start=1):
        if needle in normalise(doc.page_content):
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip generation; measure retrieval with zero API calls")
    args = parser.parse_args()

    if not Path(DOCUMENT_PATH).exists():
        sys.exit(f"{DOCUMENT_PATH} not found -- put the document you want to score there.")

    print(f"Document : {DOCUMENT_PATH}")
    print(f"Retrieval: {CANDIDATES_PER_RETRIEVER} candidates each from BM25 + vector, "
          f"reranked to top {FINAL_TOP_K}")
    print(f"Generator: {'(skipped)' if args.retrieval_only else CHAT_MODEL}\n")

    retriever = build_retriever(load_document(DOCUMENT_PATH))

    graded = [c for c in CASES if c.get("evidence")]
    hits = reciprocal_ranks = 0.0
    answers_right = answers_scored = 0
    citation_problems = []
    failures = []

    for case in CASES:
        question = case["question"]
        docs = retriever.invoke(question)

        rank = None
        if case.get("evidence"):
            rank = rank_of_evidence(docs, case["evidence"])
            if rank:
                hits += 1
                reciprocal_ranks += 1 / rank
            else:
                failures.append(("RETRIEVAL", question, f"evidence never retrieved: {case['evidence']!r}"))

        if args.retrieval_only:
            mark = "ok " if (rank or not case.get("evidence")) else "MISS"
            print(f"  [{mark}] rank={rank or '-':<3} {question[:66]}")
            continue

        answer, docs = ask(retriever, question)
        answers_scored += 1
        refused = answer.lstrip().startswith(REFUSAL_MARKER)
        body = normalise(answer)

        if case.get("expect_refusal"):
            correct = refused
            detail = "answered anyway instead of refusing" if not correct else ""
        elif refused:
            correct, detail = False, "refused, but the document does cover this"
        else:
            wanted = case["expect"]
            matched = [w for w in wanted if normalise(w) in body]
            correct = bool(matched) if case.get("any_of") else len(matched) == len(wanted)
            missing = [w for w in wanted if normalise(w) not in body]
            detail = f"answer missing {missing}" if not correct else ""

        answers_right += correct
        if not correct:
            failures.append(("ANSWER", question, detail))

        problems = check_citations(answer, len(docs))
        if problems:
            citation_problems.append((question, problems))

        print(f"  [{'ok ' if correct else 'BAD'}] rank={rank or '-':<3} {question[:66]}")

    n = len(graded)
    print(f"\n{'='*70}\nRETRIEVAL  (no API calls)")
    if n:
        print(f"  hit@{FINAL_TOP_K}          {hits:.0f}/{n}  ({hits/n:.0%})")
        print(f"  MRR              {reciprocal_ranks/n:.3f}")
    else:
        print("  no cases carry an 'evidence' field -- nothing to score")

    if not args.retrieval_only:
        print(f"\nANSWERS    ({CHAT_MODEL})")
        if answers_scored:
            print(f"  correct          {answers_right}/{answers_scored}  ({answers_right/answers_scored:.0%})")
        else:
            print("  no cases scored")
        print(f"  citation issues  {len(citation_problems)}")
        for q, probs in citation_problems:
            print(f"    - {q[:56]}: {probs[0]}")

    if failures:
        print(f"\nFAILURES ({len(failures)})")
        for kind, q, detail in failures:
            print(f"  {kind:<9} {q[:60]}\n            {detail}")
    else:
        print("\nNo failures.")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
