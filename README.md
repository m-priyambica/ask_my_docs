---
title: Ask My Docs
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

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
- **FastAPI + Jinja2** — server-rendered HTML. No JavaScript anywhere in the app.
- **Python-Markdown** — renders the model's Markdown answer into real HTML

Retrieval runs entirely on your machine. Only the final answer needs the network.

## Project layout

```
ask-my-docs/
├── frontend/               the web UI — knows nothing about RAG
│   ├── static/style.css
│   ├── templates/          one Jinja2 macro per HTML fragment
│   │   ├── base.html       page shell, streamed in two halves
│   │   ├── upload.html     step 1 — choose a document
│   │   ├── ask.html        step 2 — ask a question
│   │   └── run.html        pipeline flow + result cards
│   └── views.py            renders each macro as a plain string
│
├── rag/                    the pipeline — knows nothing about the web
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
└── api.py                  HTTP routes — the only seam between the two
```

The two halves never import each other; `api.py` is the only module that
imports both. Callers reach the pipeline through `rag.src.rag_pipeline`, which
re-exports everything they need, so no route ever imports a stage directly.

## Running it

```
pip install -r requirements.txt
cp .env.example .env      # then put your real Gemini API key in .env
uvicorn api:app --port 8000
```

Open <http://localhost:8000>. Get a free Gemini API key at
[aistudio.google.com](https://aistudio.google.com/apikey).

The first run downloads the two BGE models (~1.2 GB total) and caches them.

## How the UI works without JavaScript

The pipeline flow you see while a document indexes is a **streaming HTML
response**: `api.py` writes each stage to the browser the moment it starts, and
the browser renders it as it arrives. `rag_pipeline.py` reports its stages
through an optional `progress` callback. There is no JavaScript, no polling,
and no websocket — just HTML delivered in pieces, which browsers have always
done.

This is why `frontend/templates/` holds **macros rather than whole pages**.
Each macro is one fragment, and `views.py` exposes each as a function returning
a string, so `api.py` can emit them one at a time. Rendering a complete page at
the end of the request would lose the live flow entirely.

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

## Deploying to Hugging Face Spaces

This app needs a host that runs Python, so it goes on a **Docker** Space. A
Static Space serves files only — it cannot run `rag.py`.

1. Create a new Space, SDK = **Docker**.
2. Push this folder to the Space's git repo. The frontmatter at the top of this
   README is what tells Spaces to use Docker and to route to port 7860.
3. In Settings → *Variables and secrets*, add `GEMINI_API_KEY`. Do not commit
   `.env`.

The `Dockerfile` bakes the two BGE models into the image so the first visitor
isn't kept waiting on a 1.2 GB download. Expect a slow first build and a large
image; builds after that reuse the dependency layer.

It runs a single worker deliberately — retrievers are held in process memory, so
a second worker wouldn't see a document indexed by the first. If you ever need
more than one worker, move the retriever store out of the process first.

## Notes on the model pins

Gemini's model lineup churns. `gemini-2.5-flash` — the original pin here — now
returns 404 for new users, and the rolling `gemini-flash-latest` alias resolves
to the newest model, which carries the *tightest* free quota (measured: 20
requests/day). The explicit lite pin in `rag/src/llm.py` is a deliberate choice
for free-tier headroom; the comment there explains how to trade it back for
stronger synthesis.

See the comment at the top of `requirements.txt` before loosening any pin.
