from langchain_chroma import Chroma

from .embeddings import get_embeddings


def build_vector_retriever(chunks: list, k: int):
    # No persist_directory: the collection dies with this object, so one user's
    # chunks can never be retrieved by another user's question.
    vectorstore = Chroma.from_documents(chunks, get_embeddings())
    return vectorstore.as_retriever(search_kwargs={"k": k})
