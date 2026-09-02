import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# langchain-google-genai reads GOOGLE_API_KEY; this project documents GEMINI_API_KEY.
if "GEMINI_API_KEY" in os.environ:
    os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])

# Lite pin is deliberate: gemini-2.5-flash now 404s for new users, and the
# gemini-flash-latest alias resolves to a model with ~20 requests/day free.
CHAT_MODEL = "gemini-3.5-flash-lite"

# The default cap is a few hundred tokens, which truncates a structured answer
# halfway through a list. Raising it is what "more output" actually means.
MAX_OUTPUT_TOKENS = 4096

SYSTEM_PROMPT = """Answer the question using ONLY the numbered sources below.

Cite every claim with [n], where n is the number on the SOURCE block the claim
came from. Valid citations are [1] to [{count}] and nothing else. A number that
appears inside a source's own text -- a section number, a page number, a
numbered heading -- is part of the document, never a citation.

Write the answer as Markdown, structured so it can be skimmed:
- Open with one plain sentence that answers the question directly.
- Group the rest under `## ` headings, with `- ` bullets underneath.
- Write your own headings. Never copy a heading, section name or section
  number out of the document, and never number your headings.
- Bold a bullet's label: `- **Strengths:** simple and fast [2]`.
- Use a Markdown table when comparing three or more things on the same fields.
- Plain text only. No LaTeX and no `$...$` -- write K, not $K$.

Cover everything the sources support. Do not cut the answer short.

If the sources don't contain the answer, ignore the format above and reply with
exactly "NOT IN SOURCES:" followed by one sentence on what is missing. Never
guess, and never use outside knowledge to fill a gap.

Sources:
{context}"""

REFUSAL_MARKER = "NOT IN SOURCES:"

_CITATION_RE = re.compile(r"\[(\d+)\]")


def chat_model() -> ChatGoogleGenerativeAI:
    """The one hosted model in the app, configured in one place."""
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=0, max_output_tokens=MAX_OUTPUT_TOKENS
    )


def format_sources(docs: list) -> str:
    return "\n\n".join(
        f"===== SOURCE {i + 1} =====\n{doc.page_content}" for i, doc in enumerate(docs)
    )


def generate_answer(question: str, docs: list) -> str:
    prompt = SYSTEM_PROMPT.format(context=format_sources(docs), count=len(docs))
    response = chat_model().invoke([("system", prompt), ("human", question)])
    return response.content


def check_citations(answer: str, num_sources: int) -> list[str]:
    if answer.lstrip().startswith(REFUSAL_MARKER):
        return []

    cited = {int(n) for n in _CITATION_RE.findall(answer)}
    warnings = []
    invalid = sorted(n for n in cited if n < 1 or n > num_sources)
    if invalid:
        warnings.append(f"cites source(s) {invalid} that don't exist (only {num_sources} provided)")
    if not cited:
        warnings.append("no citations at all -- nothing here is verifiable against the sources")
    return warnings
