import asyncio
import uuid
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / "LangSmith.env")

from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter

from src.embeddings.openai_embedding import OpenAIEmbedding
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.vectorstores.memory_store import InMemoryVectorStore
from src.retrievers.simple_retriever import SimpleRetriever
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.llms.openai_chat import OpenAIChatLLM
from src.aggregators.answer_multitude_aggregator import AnswerAwareAggregator
from src.rag.rag_core import RAGCore

from src.chat.service import ChatService, LLMQueryRewriter
from src.chat.memory import ChatMemory


cleaner = TextCleaner()
splitter = SemanticSplitter()


def load_md_chunks():
    loader = MarkdownLoader()
    folder = BASE_DIR / "data" / "model12"
    all_chunks = []

    for file_path in folder.glob("*.md"):
        docs = loader.load(file_path)
        for doc in docs:
            doc.content = cleaner.clean(doc.content)
            all_chunks.extend(splitter.split(doc))

    print(f"[Loader] total chunks = {len(all_chunks)}")
    return all_chunks


async def build_chat_service():
    #  文档 & 向量
    chunks = load_md_chunks()

    embedder = OpenAIEmbedding()
    embedded_chunks = await embedder.embed(chunks)

    store = InMemoryVectorStore()
    store.add(embedded_chunks)

    retriever = SimpleRetriever(embedder=embedder, vectorstore=store)
    llm = OpenAIChatLLM()
    prompt_builder = FailureAwarePromptBuilder()

    aggregator = AnswerAwareAggregator(
        llm=llm,
        prompt_builder=prompt_builder,
        embedder=embedder,
        max_chunks=3
    )

    rag_core = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        reranker=EmbeddingReranker(),
        aggregator=aggregator,
        top_k=5
    )

    memory = ChatMemory(max_turns=10)
    rewriter = LLMQueryRewriter(llm, max_turns=5)

    return ChatService(rag_core=rag_core, memory=memory, rewriter=rewriter)


async def run_multiturn_chat():
    chat_service = await build_chat_service()
    session_id = str(uuid.uuid4())

    questions = [
        "什么是 embedding-only rerank？",
        "它在什么情况下会失败？",
        "那 answer-aware 方法是怎么解决这个问题的？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n================ Turn {i} ================")
        print("[User]")
        print(q)

        result = await chat_service.chat(session_id, q)

        print("\n[Assistant]")
        print(result.assistant_message.content)

        if result.evidence:
            print("\n[Evidence]")
            for e in result.evidence:
                print(f"- Rank {e.rank} | Score {e.score:.4f} | Chunk {e.chunk_id}")


if __name__ == "__main__":
    asyncio.run(run_multiturn_chat())
