import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.aggregators.answer_aware_aggregator import AnswerAwareAggregator

# ================== Path & Env ==================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# ================== Loaders ==================

from src.loaders.md_loader import MarkdownLoader

# ================== Processors ==================

from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

# ================== Embedding / Store ==================

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.vectorstores.memory_store import InMemoryVectorStore

# ================== Retriever / Reranker / Aggregator ==================

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.aggregators.simple_aggregator import AnswerAggregator

# ================== Prompt / LLM / RAG ==================

from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore


# ================== Chunk Builder ==================

cleaner = TextCleaner()
splitter = SemanticSplitter()


def load_md_chunks():
    loader = MarkdownLoader()
    docs = loader.load(BASE_DIR / "data" / "model10" / "test.md")

    all_chunks = []
    for doc in docs:
        doc.content = cleaner.clean(doc.content, preserve_newlines=True)
        all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")
    return all_chunks


# ================== RAG Test ==================

async def run_test(use_aggregator: bool):

    chunks = load_md_chunks()

    # Embedding
    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    # Vector store
    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    # Retriever
    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)

    # Prompt / LLM
    prompt_builder = SimplePromptBuilder()
    llm = OpenAIChatLLM()

    # Aggregator（关键差异）
    aggregator = (
        # AnswerAggregator(
        #     max_chunks=3
        # )
        AnswerAwareAggregator(
            llm=llm,
            prompt_builder=prompt_builder,
            embedder=embedder,
            max_chunks=3,
        )
        if use_aggregator
        else None
    )

    rag = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5,
    )

    result = await rag.answer(
        "在 RAG 系统中，Reranking 的作用是什么？"
    )

    # ================== Pretty Print ==================

    print("\n==============================")
    print("WITH Aggregator" if use_aggregator else "NO Aggregator")

    print("\n[Answer]")
    print(result.answer)

    print("\n[Evidence]")

    if use_aggregator:
        for e in result.evidence_chunks:
            print("----")
            print(f"Rank: {e.rank}")
            print(f"Score: {e.score:.4f}")
            print(f"Source: {e.source}")
            print(f"Chunk ID: {e.chunk_id}")
            print("Content:")
            print(e.content)
    else:
        for s in result.sources:
            print("----")
            print(f"Score: {s.score:.4f}")
            print("Content:")
            print(s.chunk.content)
            print("Metadata:")
            print(s.chunk.metadata)


# ================== Main ==================

async def main():
    await run_test(use_aggregator=False)
    await run_test(use_aggregator=True)


if __name__ == "__main__":
    asyncio.run(main())
