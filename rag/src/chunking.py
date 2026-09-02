from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_TOKENS = 650
CHUNK_OVERLAP = 100


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_TOKENS, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
