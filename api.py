"""Run:  uvicorn api:app --port 8000"""
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from frontend import views
from rag.src.rag_pipeline import (
    SUPPORTED_SUFFIXES,
    ask,
    build_retriever,
    check_citations,
    load_document,
)

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Ask My Docs")
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")

# In process memory only, keyed by an unguessable id in a hidden form field.
RETRIEVERS: dict[str, object] = {}

STAGE_INFO = {
    "load":     ("Parse document", "PyPDFLoader / TextLoader"),
    "chunk":    ("Chunk", "RecursiveCharacterTextSplitter, tiktoken-counted"),
    "embed":    ("Embed & index", "bge-small-en-v1.5 locally, into a Chroma vector store"),
    "bm25":     ("Keyword index", "BM25Okapi from rank_bm25"),
    "rerank":   ("Load reranker", "BAAI/bge-reranker-base cross-encoder"),
    "retrieve": ("Retrieve & rerank", "BM25 + vector fused, then cross-encoder re-scored"),
    "generate": ("Generate answer", "Grounded prompt, citations required"),
}

HTML = "text/html; charset=utf-8"


def run_with_flow(work, subtitle: str):
    def generate():
        yield views.page_open(subtitle)
        yield views.flow_open()

        signals: queue.Queue = queue.Queue()
        outcome: dict = {}

        # A worker thread is required: the pipeline stages block, so without one
        # nothing could reach the browser until the whole run had finished.
        def runner():
            try:
                outcome["result"] = work(lambda s, d: signals.put((s, d)))
            except Exception as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                signals.put(None)

        threading.Thread(target=runner, daemon=True).start()
        started = time.monotonic()
        while (item := signals.get()) is not None:
            label, algorithm = STAGE_INFO.get(item[0], (item[0], ""))
            yield views.stage(label, algorithm, item[1])

        yield views.flow_close(ok="error" not in outcome, seconds=f"{time.monotonic()-started:.1f}")

        if "error" in outcome:
            yield views.error_card(outcome["error"])
        else:
            result = outcome["result"]
            if "answer" in result:
                yield views.answer_card(result["answer"], result["warnings"], result["sources"])
            else:
                yield views.indexed_card(result["filename"], result["pages"])
            yield views.ask_form(result["doc_id"])

        yield views.page_close()

    return StreamingResponse(generate(), media_type=HTML)


@app.get("/")
def home():
    page = views.page_open() + views.upload_form() + views.page_close()
    return StreamingResponse(iter([page]), media_type=HTML)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()

    # Spool to a temp file before streaming starts: load_document() takes a path.
    tmp_path = None
    if suffix in SUPPORTED_SUFFIXES:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

    def work(report):
        if tmp_path is None:
            raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf, .txt or .md.")
        try:
            report("load", f"Reading {filename}")
            documents = load_document(tmp_path)
            retriever = build_retriever(documents, progress=report)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        doc_id = uuid.uuid4().hex
        RETRIEVERS[doc_id] = retriever
        return {"doc_id": doc_id, "filename": filename, "pages": len(documents)}

    return run_with_flow(work, f"Indexing {filename}")


@app.post("/ask")
def ask_question(doc_id: str = Form(...), question: str = Form(...)):
    retriever = RETRIEVERS.get(doc_id)

    def work(report):
        if retriever is None:
            raise LookupError("That document is no longer loaded -- please upload it again.")
        answer, docs = ask(retriever, question, progress=report)
        return {
            "doc_id": doc_id,
            "answer": answer,
            "warnings": check_citations(answer, len(docs)),
            "sources": [doc.page_content for doc in docs],
        }

    return run_with_flow(work, question)
