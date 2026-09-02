from .chunking import CHUNK_OVERLAP, CHUNK_TOKENS, split_documents
from .document_loader import SUPPORTED_SUFFIXES, load_document
from .embeddings import EMBED_MODEL_LOCAL
from .llm import CHAT_MODEL, REFUSAL_MARKER, check_citations, generate_answer
from .retriever import (
    CANDIDATES_PER_RETRIEVER,
    FINAL_TOP_K,
    RERANK_MODEL,
    add_reranking,
    build_keyword_retriever,
    fuse,
)
from .vector_store import build_vector_retriever


def build_retriever(documents: list, progress=None):
    report = progress or (lambda *_: None)

    report("chunk", f"Splitting into ~{CHUNK_TOKENS}-token chunks ({CHUNK_OVERLAP} overlap)")
    chunks = split_documents(documents)

    report("embed", f"Embedding {len(chunks)} chunk(s) locally with {EMBED_MODEL_LOCAL}")
    vector_retriever = build_vector_retriever(chunks, k=CANDIDATES_PER_RETRIEVER)

    report("bm25", f"Building BM25 keyword index over {len(chunks)} chunk(s)")
    keyword_retriever = build_keyword_retriever(chunks, k=CANDIDATES_PER_RETRIEVER)

    report("rerank", f"Loading cross-encoder {RERANK_MODEL}")
    return add_reranking(fuse(vector_retriever, keyword_retriever))


def ask(retriever, question: str, progress=None) -> tuple[str, list]:
    report = progress or (lambda *_: None)

    report("retrieve", f"Hybrid search + reranking down to {FINAL_TOP_K} chunk(s)")
    docs = retriever.invoke(question)

    report("generate", f"Generating a grounded answer with {CHAT_MODEL}")
    return generate_answer(question, docs), docs


__all__ = [
    "CANDIDATES_PER_RETRIEVER",
    "CHAT_MODEL",
    "FINAL_TOP_K",
    "REFUSAL_MARKER",
    "SUPPORTED_SUFFIXES",
    "ask",
    "build_retriever",
    "check_citations",
    "load_document",
]
