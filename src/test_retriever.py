import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever


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
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.txt")
    return build_chunks_from_docs(docs)


def md_test():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_retriever(chunks):
    # 1. 文档 → chunks
    print(f"Chunks: {len(chunks)}")

    # 2. chunks → embeddings
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 3. 向量库
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 4. Retriever
    retriever = SimpleRetriever(
        embedder=embedder,
        vectorstore=store
    )

    query = "什么是 RAG 系统？"
    results = await retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    for r in results:
        print("----")
        print(f"Score: {r.score:.4f}")
        print(r.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(test_retriever(chunks))
