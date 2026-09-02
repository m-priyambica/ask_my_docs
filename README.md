# Ask My Docs

Upload a PDF or text file and ask questions about it. Every answer cites the
exact source chunk it came from, so you can check it against the original text
yourself. When the document doesn't cover the question, the app says so instead
of guessing.

## Stack

- **LangChain** — document loading, chunking, retrievers, orchestration
- **BM25 + Chroma, fused via `EnsembleRetriever`** — hybrid keyword + semantic search
- **BGE embeddings** (`BAAI/bge-small-en-v1.5`, local) — no API key, no rate limit
- **BGE reranker** (`BAAI/bge-reranker-base`, local) — narrows candidates before generation
- **Google Gemini** (`gemini-3.5-flash-lite`) — the only hosted call, used for generation
- **Python-Markdown** — renders the model's Markdown answer into real HTML
- **`http.server`** — the standard library. No web framework, nothing to install.

Retrieval runs entirely on your machine. Only the final answer needs the network.

## Project layout

```
ask-my-docs/
├── rag/                    the pipeline — this is the project
│   ├── src/
│   │   ├── document_loader.py   1. PDF/text -> Document objects
│   │   ├── chunking.py          2. split into ~650-token chunks
│   │   ├── embeddings.py        3. chunks -> vectors, locally
│   │   ├── vector_store.py      4. vectors -> in-memory Chroma
│   │   ├── retriever.py         5. BM25 + vector, then reranking
│   │   ├── llm.py               6. the one hosted call, and its prompt
│   │   └── rag_pipeline.py      the two operations, wired together
│   └── evaluate.py         scores the pipeline; needs goldens.py + sample.md
│
├── app.py                  the whole web layer: one server, two endpoints
└── index.html              the whole UI: one page
```

`rag/` knows nothing about HTTP and `app.py` knows nothing about retrievers
beyond calling two functions. Callers reach the pipeline through
`rag.src.rag_pipeline`, which re-exports everything they need, so nothing
imports a stage directly.

## Running it

```
pip install -r requirements.txt
cp .env.example .env      # then put your real Gemini API key in .env
python app.py
```

Open <http://localhost:8000>. Get a free Gemini API key at
[aistudio.google.com](https://aistudio.google.com/apikey).

The first run downloads the two BGE models (~1.2 GB total) and caches them.

## How the web layer works

There isn't much of one, on purpose — the point of this project is the
retrieval pipeline, not the plumbing around it.

`app.py` subclasses `BaseHTTPRequestHandler` from the standard library and
answers three things:

| Route | Does |
|---|---|
| `GET /` | sends `index.html` |
| `POST /upload?name=…` | indexes the raw bytes in the body, returns a `doc_id` |
| `POST /ask` | takes `{doc_id, question}`, returns the answer as HTML |

The browser sends the file as the raw request body rather than a multipart
form, which is why no form-parsing library is needed. Both stages report their
progress through `rag_pipeline`'s optional `progress` callback; `app.py`
collects those into a list and returns them alongside the result, so the page
can show which steps ran.

The Markdown the model writes is converted to HTML in Python, in
`render_answer()`. It escapes the text *before* converting it, because the
answer quotes the document back — so HTML hidden inside a PDF can never reach
the page as live markup.

## Measuring accuracy

`rag/evaluate.py` scores the two halves of the pipeline separately, because
they fail separately: retrieval can miss the right chunk, or the model can
mishandle a chunk it was given.

It needs two files that are **not** in this repo, because the right ones depend
entirely on your own document:

| File | What it is |
|---|---|
| `rag/sample.md` | the document to score against |
| `rag/goldens.py` | a `CASES` list of questions with known-correct answers |

Each entry in `CASES` is a dict:

| Key | Type | Meaning |
|---|---|---|
| `question` | str | the question to ask |
| `evidence` | str | text that retrieval must surface, copied verbatim from the document — scores retrieval |
| `expect` | list[str] | strings the answer must contain — scores generation |
| `any_of` | bool | when `True`, one match in `expect` is enough |
| `expect_refusal` | bool | when `True`, a correct run answers `NOT IN SOURCES` |

```python
CASES = [
    {"question": "...", "evidence": "...", "expect": ["..."]},
    {"question": "...", "expect_refusal": True},
]
```

Then:

```
python rag/evaluate.py --retrieval-only   # free: no API calls at all
python rag/evaluate.py                    # adds one chat call per question
```

Run it from anywhere — it resolves both the project root and `sample.md`
against its own location, not the working directory. If either file is missing
it says so and exits rather than raising.

Retrieval is scored by string match against ground truth, so it needs no LLM
judge and no quota — you can run it as often as you like. A golden set earns
its keep when it contains deliberately confusable pairs (26 *days* of one thing
against 26 *weeks* of another) and questions the document genuinely doesn't
answer, so that inventing a plausible answer scores as a failure.

## Scope

This runs locally, for one person at a time. Retrievers are held in a plain
dict in process memory, so they vanish when you stop the server and a second
process would not see them. That is a deliberate trade: it keeps the storage
layer out of the way while you are learning the retrieval pipeline.

To make it multi-user you would move `RETRIEVERS` out of the process and give
Chroma a `persist_directory` — both are single-line changes, and everything
else would stay as it is.

## Notes on the model pins

Gemini's model lineup churns. `gemini-2.5-flash` — the original pin here — now
returns 404 for new users, and the rolling `gemini-flash-latest` alias resolves
to the newest model, which carries the *tightest* free quota (measured: 20
requests/day). The explicit lite pin in `rag/src/llm.py` is a deliberate choice
for free-tier headroom; the comment there explains how to trade it back for
stronger synthesis.

See the comment at the top of `requirements.txt` before loosening any pin.
