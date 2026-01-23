import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader
from src.rerankers.embedding_reranker import EmbeddingReranker

BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
    docs = loader.load(BASE_DIR / "data" / "model9" / "test.md")
    return build_chunks_from_docs(docs)


def pdf_test():
    loader = PdfLoader()
    docs = loader.load(BASE_DIR / "data" / "model2" / "test.pdf")
    return build_chunks_from_docs(docs)


async def test_rag_compare(chunks,use_rerank: bool):
    # 1. 构建文档 chunks
    print(f"Total chunks: {len(chunks)}")

    # 2. Embedding + VectorStore
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(vectorstore=store, embedder=embedder)

    # 3. Prompt + LLM + Reranker
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()
    reranker = EmbeddingReranker() if use_rerank else None # 添加 Reranker

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=reranker, # 添加 Reranker
        top_k=3,
    )

    # 4. 问答
    # result = await rag.answer("什么是 RAG 系统？")
    result = await rag.answer("在 RAG 系统中，Reranking 的作用是什么？") # 修改问题

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

async def main():
    chunks = md_test()
    # chunks = txt_test()
    # chunks = pdf_test()

    print("不使用 Reranker")
    await test_rag_compare(chunks,use_rerank=False)
    print()
    print("---------------------------------------------------------------------------------------------------------")
    print("使用 Reranker")
    await test_rag_compare(chunks,use_rerank=True)

if __name__ == "__main__":

    asyncio.run(main())
