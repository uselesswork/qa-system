import asyncio
from pathlib import Path
from dotenv import load_dotenv

from src.aggregators.schemas import AggregatedAnswer
from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.prompts.simple_prompt import SimplePromptBuilder
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore

from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

cleaner = TextCleaner()
splitter = SemanticSplitter()

def load_md_chunks_multi():
    loader = MarkdownLoader()
    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):  # 遍历所有 md 文件
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")
    return all_chunks


async def run_comparison(query: str):
    chunks = load_md_chunks_multi()

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    # ------------------ Embedding-only RAG ------------------
    rag_embedding_only = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=None,
        top_k=5
    )

    result_embedding = await rag_embedding_only.answer(query)

    print("\n==============================")
    print("[Embedding-only top-K]")
    print("\n[Answer]")
    print(result_embedding.answer)
    print("\n[Evidence]")
    for s in result_embedding.sources:
        print("----")
        print(f"Score: {s.score:.4f}")
        print(f"Source: {s.chunk.metadata.get('source')}")
        print(f"Chunk ID: {s.chunk.metadata.get('chunk_id')}")
        print("Content:")
        print(s.chunk.content)

    # ------------------ Answer-aware RAG ------------------
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=5
    )

    rag_answer_aware = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5
    )

    result_answer_aware = await rag_answer_aware.answer(query)

    print("\n==============================")
    print("[Answer-aware top-K]")
    print("\n[Answer]")
    print(result_answer_aware.answer)
    print("\n[Evidence]")

    for e in result_answer_aware.evidence_chunks:
        print("----")
        print(f"Rank: {e.rank}")
        print(f"Score: {e.score:.4f}")
        print(f"Source: {e.source}")
        print(f"Chunk ID: {e.chunk_id}")
        print("Content:")
        print(e.content)


if __name__ == "__main__":
    asyncio.run(run_comparison("根据文档6的详细解释，对比传统重排和answer-aware重排在对抗性环境下的表现差异"))
