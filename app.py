"""Run:  python app.py     then open http://localhost:8000

The whole web layer, on Python's standard library. It serves one HTML page and
two JSON endpoints, and it is deliberately thin: all the interesting work lives
in rag/, which knows nothing about HTTP.
"""
import os
import html
import json
import re
import tempfile
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import markdown

from rag.src.rag_pipeline import (
    SUPPORTED_SUFFIXES,
    ask,
    build_retriever,
    check_citations,
    load_document,
)

PORT = int(os.getenv("PORT", 8000))
PAGE = Path(__file__).parent / "index.html"

RETRIEVERS: dict[str, object] = {}

STAGE_LABELS = {
    "load": "Parse document",
    "chunk": "Split into chunks",
    "embed": "Embed and index",
    "bm25": "Build keyword index",
    "rerank": "Load reranker",
    "retrieve": "Retrieve and rerank",
    "generate": "Generate answer",
}

_markdown = markdown.Markdown(extensions=["tables", "sane_lists"])
_CITE_RE = re.compile(r"\[(\d+)\]")


def render_answer(text: str) -> str:
    """Markdown from the model into HTML for the page.

    Escaping first matters: the answer quotes the document back, so any HTML
    inside a PDF must not survive into the page. Markdown leaves the escaped
    entities alone, so **bold** still becomes <strong> afterwards.
    """
    body = _markdown.reset().convert(html.escape(text))
    return _CITE_RE.sub(r'<span class="cite">[\1]</span>', body)


def index_document(data: bytes, filename: str) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf, .txt or .md.")

    stages: list[dict] = []
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        stages.append({"label": STAGE_LABELS["load"], "detail": f"Reading {filename}"})
        documents = load_document(path)
        retriever = build_retriever(documents, progress=_collect(stages))
    finally:
        Path(path).unlink(missing_ok=True)

    doc_id = uuid.uuid4().hex
    RETRIEVERS[doc_id] = retriever
    return {"doc_id": doc_id, "filename": filename, "pages": len(documents), "stages": stages}


def answer_question(doc_id: str, question: str) -> dict:
    retriever = RETRIEVERS.get(doc_id)
    if retriever is None:
        raise LookupError("That document is no longer loaded -- please upload it again.")

    stages: list[dict] = []
    answer, docs = ask(retriever, question, progress=_collect(stages))
    return {
        "answer": render_answer(answer),
        "warnings": check_citations(answer, len(docs)),
        "sources": [doc.page_content for doc in docs],
        "stages": stages,
    }


def _collect(stages: list):
    return lambda name, detail: stages.append(
        {"label": STAGE_LABELS.get(name, name), "detail": detail}
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self):
        route = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            if route.path == "/upload":
                name = parse_qs(route.query).get("name", ["document"])[0]
                payload = index_document(body, name)
            elif route.path == "/ask":
                data = json.loads(body)
                payload = answer_question(data.get("doc_id", ""), data.get("question", ""))
            else:
                return self._send(404, "application/json", b'{"error": "Not found"}')
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(200, "application/json", json.dumps(payload).encode())

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"  {self.command} {self.path} -> {args[1]}")


if __name__ == "__main__":
    print(f"Ask My Docs running on http://localhost:{PORT}  (ctrl-c to stop)")
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
