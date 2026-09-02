# Ask My Docs

Upload a PDF or text file and ask questions about it. Every answer cites the
exact source chunk it came from, so you can check it against the original text
yourself. When the document doesn't cover the question, the app says so instead
of guessing.

## Stack

- **LangChain** — loading, chunking, retrievers, orchestration
- **Chroma** — in-memory vector store
- **BM25** (`rank_bm25`) — keyword index
- **BGE embeddings** (`BAAI/bge-small-en-v1.5`, local) — no key, no rate limit
- **BGE reranker** (`BAAI/bge-reranker-base`, local) — cross-encoder
- **Google Gemini** (`gemini-3.5-flash-lite`) — the only hosted call
- **Python-Markdown** — model's Markdown → HTML
- **`http.server`** — standard library. No web framework




## Layout

```
ask-my-docs/
├── rag/
│   ├── src/
│   │   ├── document_loader.py   1. PDF/text -> Documents
│   │   ├── chunking.py          2. split into ~650-token chunks
│   │   ├── embeddings.py        3. chunks -> vectors, locally
│   │   ├── vector_store.py      4. vectors -> in-memory Chroma
│   │   ├── retriever.py         5. BM25 + vector, then reranking
│   │   ├── llm.py               6. the one hosted call, and its prompt
│   │   └── rag_pipeline.py      the two operations, wired together
│   └── evaluate.py         scores the pipeline; needs goldens.py + sample.md
├── app.py                  routes, request handling, Markdown -> HTML
└── index.html              the UI
```

---



## What is RAG?

A language model only knows what it saw during training. It has never read your
handbook, your notes, or the PDF you downloaded this morning. Ask it about them
and it will invent a confident, fluent, wrong answer.

Pasting the whole document into the prompt doesn't scale: models have a size
limit, long prompts cost more and run slower, and accuracy *drops* when the one
useful sentence is buried in eighty pages.

**Retrieval-Augmented Generation** fixes this with a simple idea:

> Before answering, search the document for the few passages most likely to
> contain the answer. Put only those in the prompt. Tell the model to answer
> from them and nothing else.

So a RAG app is two systems joined together:

| | What it is | Where it runs here |
|---|---|---|
| **Retrieval** | a search engine over your document. No AI writing involved | your machine |
| **Generation** | one model call that reads what search found | Google Gemini |

They fail differently, so this project separates them and scores them
separately. When an answer is wrong, the first question is always *did
retrieval even find the right passage?*

---

## The pipeline

Two operations. **Indexing** runs once per document, **answering** once per
question.

```
INDEXING     file ──▶ load ──▶ chunk ──▶ ┬─▶ embed ──▶ Chroma  ┐
                                         └─▶ BM25 index ───────┴─▶ retriever

ANSWERING    question ──▶ retriever ──▶ hybrid search ──▶ rerank ──▶ top 8
                                                                       │
                          answer ◀── Gemini ◀── grounded prompt ◀──────┘
```

### 1. Document loading — `rag/src/document_loader.py`

A PDF is a layout format, not text. `PyPDFLoader` extracts the readable
characters, producing one LangChain `Document` per page. `.txt` and `.md` go
through `TextLoader` unchanged. `SUPPORTED_SUFFIXES` is the allow-list the web
layer checks against.

### 2. Chunking — `rag/src/chunking.py`

You can't search a whole document as one blob — you need pieces small enough
that each is *about* one thing. Text is split into ~650-token chunks.

A **token** is roughly ¾ of a word; it is how models count text. Sizes are
measured in tokens rather than characters so a chunk always fits the model's
budget predictably.

Two details matter:

- **Splitting is recursive.** `RecursiveCharacterTextSplitter` breaks at
  paragraphs first, then sentences, then words. A blind split every 650
  characters would cut sentences in half and destroy meaning.
- **Chunks overlap by 100 tokens.** If an answer straddles a boundary, the
  overlap guarantees at least one chunk still holds it whole.

Tuning: `CHUNK_TOKENS`, `CHUNK_OVERLAP`.

### 3. Embeddings — `rag/src/embeddings.py`

An **embedding** converts text into a vector — a list of numbers representing
its meaning. Texts meaning similar things land near each other in that space
even with no words in common, which is how a search for "time off" finds a
paragraph about "annual leave".

This uses `BAAI/bge-small-en-v1.5` running **locally**. No API key, no rate
limit, nothing leaving your machine. It downloads once (~130 MB) and caches.

### 4. Vector storage — `rag/src/vector_store.py`

The vectors go into **Chroma**, a vector database. Its job is one thing:
given a query vector, return the nearest stored vectors fast.

There is deliberately **no `persist_directory`**. The collection lives in memory
and dies with the object, so one user's document can never be retrieved by
another user's question.

### 5. Retrieval — `rag/src/retriever.py`

Three ideas stacked, and the most important file in the project.

**Keyword search.** `build_keyword_retriever()` builds a **BM25** index —
classic keyword ranking, no understanding of meaning. Why keep something so
old-fashioned? Because embeddings are weakest at exactly what BM25 is strongest
at: rare literal strings like product codes, names, `error_code_4021`. Semantic
search blurs those into uselessness; keyword search nails them.

**Hybrid fusion.** `fuse()` wraps both retrievers in an `EnsembleRetriever`,
weighted 50/50. Each returns `CANDIDATES_PER_RETRIEVER` (20) chunks, merged into
one ranked list. Good coverage, but noisy.

**Reranking.** `add_reranking()` puts a **cross-encoder** in front. Unlike
embeddings — which encode question and chunk separately and compare — a
cross-encoder reads the question *together with* each chunk and scores how well
it actually answers it. Far more accurate, far slower, which is exactly why it
only ever sees ~40 candidates rather than the whole document. `FINAL_TOP_K` (8)
survive.

This cheap-and-broad → expensive-and-precise shape is the standard pattern in
modern retrieval. Understand this file and you understand the project.

### 6. Generation — `rag/src/llm.py`

The only hosted call. Everything above ran locally.

- `format_sources()` numbers the chunks as `===== SOURCE 1 =====` blocks. The
  explicit delimiter matters: with a bare `[1] text` label, the model confuses
  numbered headings *inside* the document (`## 10. Referrals`) with citation
  markers and emits `[10]`.
- `SYSTEM_PROMPT` orders the model to use only those sources, cite every claim
  with `[n]`, answer in Markdown, and reply `NOT IN SOURCES:` rather than guess.
- `generate_answer()` makes the call via `chat_model()`.
- `check_citations()` verifies afterwards that cited numbers actually exist. A
  model citing `[10]` when given 2 sources is a real failure mode, and the UI
  surfaces the warning.

Tuning: `CHAT_MODEL`, `MAX_OUTPUT_TOKENS`, and `SYSTEM_PROMPT` — the
highest-leverage text in the repo.

### 7. Orchestration — `rag/src/rag_pipeline.py`

Forty lines, and the best entry point for reading the code. It exposes exactly
two functions and re-exports every constant callers need, so nothing outside
`rag/` ever imports a stage directly.

```python
build_retriever(documents, progress=None)   # stages 2-5, returns a retriever
ask(retriever, question, progress=None)     # stages 5-6, returns (answer, docs)
```

Both take an optional `progress` callback, called as `progress(stage, detail)`
after each step. That is how the UI reports what is happening without `rag/`
knowing a browser exists.

---

## How the files connect

```
app.py                        HTTP: routes, request/response, Markdown → HTML
  │  imports only
  ▼
rag/src/rag_pipeline.py       the public surface: build_retriever(), ask()
  │  imports
  ├── document_loader.py      file → Documents
  ├── chunking.py             Documents → chunks
  ├── embeddings.py ──┐
  ├── vector_store.py ┴────── chunks → Chroma vector retriever
  ├── retriever.py            BM25 + fusion + reranking
  └── llm.py                  prompt → Gemini → answer, + citation check

rag/evaluate.py               imports rag_pipeline directly, no web layer
```

The rule: **`rag/` knows nothing about HTTP, `app.py` knows nothing about
retrievers** beyond calling two functions. Neither half imports the other's
internals. You could delete `app.py` and drive the pipeline from a script.

**Read in this order:** `rag_pipeline.py` → `retriever.py` → `llm.py` →
everything else. Leave `app.py` for last; it's plumbing.

---

## `app.py` — the web layer

There is no framework and therefore **no router object**. `app.py` subclasses
`BaseHTTPRequestHandler` from the standard library; routing is a plain `if/elif`
on the request path inside two methods, `do_GET` and `do_POST`.

### Routes

| Method | Path | Body in | Returns |
|---|---|---|---|
| `GET` | `/`, `/index.html` | — | the page |
| `GET` | anything else | — | `404` |
| `POST` | `/upload?name=<filename>` | raw file bytes | `{doc_id, filename, pages, stages}` |
| `POST` | `/ask` | `{doc_id, question}` | `{answer, warnings, sources, stages}` |

The browser sends the file as the **raw request body**, not a multipart form —
which is why no form-parsing library is needed. The filename rides in the query
string.

### How a request reaches the pipeline

```
POST /upload?name=x.pdf
  do_POST reads Content-Length bytes
    → index_document(body, "x.pdf")
        checks the suffix against SUPPORTED_SUFFIXES
        spools the bytes to a temp file  (load_document needs a path)
        load_document()  → build_retriever(..., progress=...)
        deletes the temp file
        stores the retriever in RETRIEVERS[doc_id], returns the doc_id

POST /ask  {doc_id, question}
  do_POST parses JSON
    → answer_question(doc_id, question)
        looks up RETRIEVERS[doc_id]
        ask() → (answer, docs)
        render_answer(answer)   Markdown → HTML
        check_citations(answer, len(docs))
```

`RETRIEVERS` is a module-level dict keyed by a random `uuid4().hex`. The browser
holds that id and sends it back with each question. Nothing is persisted: stop
the server and the documents are gone.

### Errors

Any exception in `do_POST` is caught and returned as `{"error": "..."}` with
status **200**, not a 5xx. The page checks for that key and renders a failure
card. This keeps the client logic to one branch; the tradeoff is that HTTP
status alone won't tell you a request failed.

### Progress reporting

Both handlers pass a callback built by `_collect()` into the pipeline. Each
stage appends `{label, detail}` to a list, which is returned with the result so
the page can show which steps ran. `STAGE_LABELS` maps the pipeline's internal
names (`chunk`, `embed`, `bm25`, …) to display text.

### Markdown rendering

`render_answer()` converts the model's Markdown to HTML **in Python**. It
escapes the text *before* converting, because the answer quotes the document
back — so HTML hidden inside a PDF can never reach the page as live markup.
Citation markers `[n]` are then wrapped in `<span class="cite">` for styling.

---

## Quick start

```
pip install -r requirements.txt
cp .env.example .env      # then put your real Gemini API key in .env
python app.py
```

Open <http://localhost:8000>. Free Gemini key at
[aistudio.google.com](https://aistudio.google.com/apikey).

The first run downloads the two BGE models (~1.2 GB) and caches them — slow
once, fast after.

---


## Knobs worth turning

Change one number, re-run the evaluation, see what moves. That is the fastest
way to build intuition for RAG.

| Setting | File | Now | Effect |
|---|---|---|---|
| `CHUNK_TOKENS` | `chunking.py` | 650 | Smaller = precise but fragmented; larger = more context, more noise |
| `CHUNK_OVERLAP` | `chunking.py` | 100 | Guards answers split across a boundary |
| `CANDIDATES_PER_RETRIEVER` | `retriever.py` | 20 | Candidates each index proposes. More = better recall, slower |
| weights in `fuse()` | `retriever.py` | 0.5 / 0.5 | Semantic vs keyword balance |
| `FINAL_TOP_K` | `retriever.py` | 8 | Chunks reaching the prompt. More = fuller answers, more distraction |
| `MAX_OUTPUT_TOKENS` | `llm.py` | 4096 | Answer length ceiling |
| `SYSTEM_PROMPT` | `llm.py` | — | The rules the model follows |

---

## Measuring accuracy

`rag/evaluate.py` scores retrieval and generation separately, because they fail
separately. It needs two files **not** in this repo, since the right ones depend
on your document:

| File | What it is |
|---|---|
| `rag/sample.md` | the document to score against |
| `rag/goldens.py` | a `CASES` list of questions with known-correct answers |

| Key | Type | Meaning |
|---|---|---|
| `question` | str | the question to ask |
| `evidence` | str | text retrieval must surface, verbatim from the document |
| `expect` | list[str] | strings the answer must contain |
| `any_of` | bool | when `True`, one match in `expect` is enough |
| `expect_refusal` | bool | when `True`, a correct run answers `NOT IN SOURCES` |

```python
CASES = [
    {"question": "...", "evidence": "...", "expect": ["..."]},
    {"question": "...", "expect_refusal": True},
]
```

```
python rag/evaluate.py --retrieval-only   # free: no API calls
python rag/evaluate.py                    # one chat call per question
```

**hit@k** is how often the right passage was retrieved at all. **MRR** rewards
ranking it near the top — 1.0 means always first, 0.5 means typically second.

Retrieval scoring needs no quota, so run it constantly. A golden set earns its
keep when it holds confusable pairs (26 *days* of one thing vs 26 *weeks* of
another) and questions the document genuinely can't answer, so that inventing a
plausible answer scores as a failure.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Could not import module "api"` | Old command. No FastAPI here — run `python app.py` |
| First run hangs for minutes | Downloading the 1.2 GB BGE models. Once only |
| `429` / quota errors | Free Gemini tier is rate-limited. Wait a minute |
| "That document is no longer loaded" | Retrievers live in memory and die with the server. Re-upload |
| Answers `NOT IN SOURCES` but the doc covers it | A retrieval failure, not a model one. Check with `--retrieval-only` |

---

## Scope

Local, one person at a time. Retrievers sit in a plain dict in process memory,
so they vanish with the server and a second process wouldn't see them — a
deliberate trade that keeps the storage layer out of the way while you learn the
retrieval pipeline. To go multi-user, move `RETRIEVERS` out of the process and
give Chroma a `persist_directory`.

## Model pins

Gemini's lineup churns. `gemini-2.5-flash` now 404s for new users, and the
rolling `gemini-flash-latest` alias resolves to the newest model, which carries
the *tightest* free quota (measured: 20 requests/day). The explicit lite pin in
`rag/src/llm.py` buys free-tier headroom.

See the comment at the top of `requirements.txt` before loosening any pin.


See the comment at the top of `requirements.txt` before loosening any pin.
