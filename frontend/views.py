import re
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

TEMPLATES_DIR = Path(__file__).parent / "templates"

# The model is asked for Markdown, so something has to turn it into HTML --
# otherwise `**bold**` and `- bullets` reach the page as literal asterisks.
# `tables` renders comparison tables; `sane_lists` stops a stray '*' mid-
# paragraph from being mistaken for a list.
_markdown = markdown.Markdown(extensions=["tables", "sane_lists"])

_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)

_base = _env.get_template("base.html").module
_upload = _env.get_template("upload.html").module
_ask = _env.get_template("ask.html").module
_run = _env.get_template("run.html").module


def page_open(subtitle: str = "") -> str:
    return _base.page_open(subtitle)


def page_close() -> str:
    return _base.page_close()


def upload_form() -> str:
    return _upload.upload_form()


def ask_form(doc_id: str) -> str:
    return _ask.ask_form(doc_id)


def flow_open() -> str:
    return _run.flow_open()


def stage(label: str, algorithm: str, detail: str = "") -> str:
    return _run.stage(label, algorithm, detail)


def flow_close(ok: bool, seconds: str) -> str:
    return _run.flow_close(ok, seconds)


def _render_answer(text: str) -> Markup:
    # Escape first, then render: the answer quotes the document back, so any
    # HTML inside a PDF must not survive into the page. Markdown leaves the
    # escaped entities alone, so `**bold**` still becomes <strong> afterwards.
    html = _markdown.reset().convert(str(escape(text)))
    return Markup(re.sub(r"\[(\d+)\]", r'<span class="cite">[\1]</span>', html))


def error_card(message: str) -> str:
    return _run.error_card(message)


def indexed_card(filename: str, pages: int) -> str:
    return _run.indexed_card(filename, pages)


def answer_card(answer: str, warnings: list, sources: list) -> str:
    return _run.answer_card(_render_answer(answer), warnings, sources)
