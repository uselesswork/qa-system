import asyncio
from pathlib import Path
from dotenv import load_dotenv

# ================== Path & Env ==================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

# ================== Loaders ==================

from src.loaders.md_loader import MarkdownLoader
from src.loaders.txt_loader import TxtLoader
from src.loaders.pdf_loader import PdfLoader

# ================== Processors ==================

from src.processors.cleaner import TextCleaner

# 模块 2（baseline）
from src.processors.splitter import TextSplitter

# 模块 10（semantic）
from src.processors.semantic.semantic_splitter import SemanticSplitter

# ================== Embedding / Store ==================

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore

# ================== Retriever / Reranker ==================

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker

# ================== Prompt / LLM / RAG ==================

from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore


# ================== Split Builders ==================

cleaner = TextCleaner()


def build_chunks_text_splitter(docs):
    """模块 2：字符 / 规则切分（baseline）"""
    splitter = TextSplitter()
    all_chunks = []

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        all_chunks.extend(splitter.split(doc))

    return all_chunks


def build_chunks_semantic_splitter(docs):
    """模块 10：语义切分"""
    splitter = SemanticSplitter()
    all_chunks = []

    for doc in docs:
        doc.content = cleaner.clean(doc.content)
        all_chunks.extend(splitter.split(doc))

    return all_chunks


# ================== Data Loaders ==================

def load_md_chunks(use_semantic: bool):
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model10" / "test.md")

    if use_semantic:
        print("[Semantic Splitter]")
        return build_chunks_semantic_splitter(docs)
    else:
        print("[Text Splitter]")
        return build_chunks_text_splitter(docs)


# ================== RAG Test ==================

async def test_rag(chunks, use_rerank: bool):
    print(f"Total chunks: {len(chunks)}")

    # 1. Embedding
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # 2. VectorStore
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # 3. Retriever
    retriever = SimpleRetriever(
        embedder=embedder,
        vectorstore=store
    )

    # 4. RAG
    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=SimplePromptBuilder(),
        llm=OpenAIChatLLM(),
        reranker=EmbeddingReranker() if use_rerank else None,
        top_k=5,
    )

    # 5. Ask
    result = await rag.answer(
        "在 RAG 系统中，Reranking 的作用是什么？"
    )

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


# ================== Main ==================

async def main():

    # 模块 2 + 无 Rerank
    print("\n==============================")
    print("TextSplitter + No Reranker")
    chunks = load_md_chunks(use_semantic=False)
    await test_rag(chunks, use_rerank=False)

    # 模块 2 + Rerank
    print("\n==============================")
    print("TextSplitter + Reranker")
    await test_rag(chunks, use_rerank=True)

    # 模块 10 + 无 Rerank
    print("\n==============================")
    print("SemanticSplitter + No Reranker")
    chunks = load_md_chunks(use_semantic=True)
    await test_rag(chunks, use_rerank=False)

    # 模块 10 + Rerank
    print("\n==============================")
    print("SemanticSplitter + Reranker")
    await test_rag(chunks, use_rerank=True)


if __name__ == "__main__":
    asyncio.run(main())
