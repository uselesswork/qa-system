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
from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore

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


async def test_rag(chunks):
    # 1. 构建文档 chunks
    print(f"Total chunks: {len(chunks)}")

    # 2. Embedding + VectorStore
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(vectorstore=store, embedder=embedder)

    # 3. Prompt + LLM
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        top_k=3,
    )

    # 4. 问答
    result = await rag.answer("什么是 RAG 系统？")

    print("Answer:")
    print(result.answer)
    print("----")
    print("Sources:")
    for s in result.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print("Content:")
        print(s.chunk.content)
        print("Metadata:")
        print(s.chunk.metadata)


if __name__ == "__main__":
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    asyncio.run(test_rag(chunks))
