from pathlib import Path
from dotenv import load_dotenv

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore

from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.llms.openai_chat import OpenAIChatLLM
from src.rag.rag_core import RAGCore
from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def build_rag_core() -> RAGCore:
    # ---------- 文档加载 & 预处理 ----------
    cleaner = TextCleaner()
    splitter = SemanticSplitter()
    loader = MarkdownLoader()

    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Semantic Splitter] Total chunks: {len(all_chunks)}")

    # ---------- Embedding & VectorStore ----------
    embedder = OpenAIEmbedding()
    store = InMemoryVectorStore()

    # 这里是 async 的，但 FastAPI 启动时无法 await
    # 所以我们用一个 trick（见下）
    store._pending_chunks = all_chunks
    store._embedder = embedder

    # ---------- 组件 ----------
    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    reranker = EmbeddingReranker()
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=3
    )

    return RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=reranker,
        aggregator=aggregator,
        top_k=5
    )
