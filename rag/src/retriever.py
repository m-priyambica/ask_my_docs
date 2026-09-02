from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

RERANK_MODEL = "BAAI/bge-reranker-base"
# A wider net for the reranker to sort through, and more of it kept: the answer
# can only be as complete as the chunks it is allowed to see. The cross-encoder
# is the expensive part, so this trades a little indexing time for coverage.
CANDIDATES_PER_RETRIEVER = 20
FINAL_TOP_K = 8


def build_keyword_retriever(chunks: list, k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def fuse(vector_retriever, keyword_retriever) -> EnsembleRetriever:
    return EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])


def add_reranking(base_retriever) -> ContextualCompressionRetriever:
    reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name=RERANK_MODEL), top_n=FINAL_TOP_K)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
