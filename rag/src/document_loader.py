from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_document(file_path: str) -> list:
    path = Path(file_path)
    loader = PyPDFLoader(file_path) if path.suffix.lower() == ".pdf" else TextLoader(file_path, encoding="utf-8")
    return loader.load()
