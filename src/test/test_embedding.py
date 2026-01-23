from pathlib import Path
import asyncio
import os
from dotenv import load_dotenv

# 加载 .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.txt_loader import TxtLoader
from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.processors.cleaner import TextCleaner
from src.processors.splitter import TextSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding


cleaner = TextCleaner()
splitter = TextSplitter()


def build_chunks_from_docs(docs):
    """模块 1 + 模块 2 组合，仅用于测试"""
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


async def embedding_test(chunks):
    print(f"Total chunks: {len(chunks)}")

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    print(f"Embedded chunks: {len(embedded_chunks)}")

    for i, ec in enumerate(embedded_chunks):
        print("----")
        print(f"Chunk {i}")
        print(f"Vector shape: {ec.vector.shape}")
        print("Metadata:")
        for k, v in ec.metadata.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(embedding_test(chunks))
