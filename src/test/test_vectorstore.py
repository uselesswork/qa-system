import asyncio
from pathlib import Path

import os
from dotenv import load_dotenv

# 加载 .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore

cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        chunks = splitter.split(doc)
        all_chunks.extend(chunks)
    return all_chunks


def txt_test():
    loader = TxtLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "basic_pipeline_examples" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_vector_store(chunks):
    # 1. 文档 → chunks
    print(f"Chunks: {len(chunks)}")

    # 2. chunks → embeddings
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 3. 向量库
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 4. 用第一个 chunk 向量当 query
    query_vector = embedded_chunks[0].vector
    results = store.search(query_vector, top_k=3)

    print(f"Search results: {len(results)}")
    for r in results:
        print("----")
        print(f"Score: {r.score:.4f}")
        print(r.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(test_vector_store(chunks))
