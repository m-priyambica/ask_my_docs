from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL_LOCAL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_LOCAL)
