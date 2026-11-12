from pathlib import Path
import os
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
from src.aggregators.answer_multitude_aggregator import CrossDocumentAnswerAggregator

BASE_DIR = Path(__file__).resolve().parent.parent


def load_chunks(folder: Path):
    cleaner = TextCleaner()
    splitter = SemanticSplitter()
    loader = MarkdownLoader()
    all_chunks = []

    for file_path in sorted(folder.glob("*.md")):
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content, preserve_newlines=True)
            all_chunks.extend(splitter.split(doc))

    return all_chunks


async def build_rag_core() -> RAGCore:
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY：请在项目根目录 .env 中配置。")

    embedder = OpenAIEmbedding()
    store = InMemoryVectorStore()

    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    reranker = EmbeddingReranker()
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = CrossDocumentAnswerAggregator(
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
