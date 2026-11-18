"""在独立固定语料上调用真实 RAG 链路并导出评测预测。会产生模型费用。"""

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag_app import load_chunks  # noqa: E402
from src.aggregators.answer_multitude_aggregator import CrossDocumentAnswerAggregator  # noqa: E402
from src.chat.schemas import ChatMessage  # noqa: E402
from src.chat.service import LLMQueryRewriter  # noqa: E402
from src.embeddings.openai_embedding import OpenAIEmbedding  # noqa: E402
from src.llms.openai_chat import OpenAIChatLLM  # noqa: E402
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder  # noqa: E402
from src.rag.rag_core import RAGCore  # noqa: E402
from src.rerankers.embedding_reranker import EmbeddingReranker  # noqa: E402
from src.retrievers.simple_retriever import SimpleRetriever  # noqa: E402
from src.vectorstores.memory_store import InMemoryVectorStore  # noqa: E402
from src.vectorstores.qdrant_store import QdrantVectorStore  # noqa: E402


class CountingLLM:
    def __init__(self, model: str):
        self.inner = OpenAIChatLLM(model=model)
        self.calls = 0

    async def generate(self, prompt: str):
        self.calls += 1
        return await self.inner.generate(prompt)


def source_name(value) -> str:
    return Path(str(value)).name


async def collect(dataset: Path, output: Path, model: str, corpus: Path | None = None,
                  vector_backend: str = "memory") -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY")
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    embedder = OpenAIEmbedding()
    with ExitStack() as stack:
        chunks = load_chunks(corpus or ROOT / "eval" / "corpus")
        if vector_backend == "qdrant-local":
            temp_dir = Path(stack.enter_context(TemporaryDirectory(prefix="qa-eval-")))
            store = QdrantVectorStore.local(temp_dir, collection="qa_eval")
            stack.callback(store.close)
            ready_ids = []
            for index, chunk in enumerate(chunks):
                document_id = str(uuid5(NAMESPACE_URL, f"eval:{source_name(chunk.metadata.get('source'))}"))
                chunk.metadata.update({
                    "document_id": document_id,
                    "knowledge_base_id": "eval",
                    "chunk_id": str(uuid5(NAMESPACE_URL, f"{document_id}:{index}")),
                })
                if document_id not in ready_ids:
                    ready_ids.append(document_id)
            store.ready_document_ids = ready_ids
            store.knowledge_base_id = "eval"
        elif vector_backend == "memory":
            store = InMemoryVectorStore()
        else:
            raise ValueError("vector_backend 只能是 memory 或 qdrant-local")
        store.add(await embedder.embed(chunks))
        llm = CountingLLM(model)
        prompt = FailureAwarePromptBuilder()
        core = RAGCore(embedder, store, SimpleRetriever(embedder, store), prompt, llm,
                       reranker=EmbeddingReranker(),
                       aggregator=CrossDocumentAnswerAggregator(llm, prompt, embedder, 3),
                       top_k=5)
        rewriter = LLMQueryRewriter(llm)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for case in cases:
                llm.calls = 0
                history = [ChatMessage(**item) for item in case.get("history", [])]
                started = time.perf_counter()
                query = await rewriter.rewrite(case["question"], history)
                result = await core.answer(query)
                row = {"id": case["id"],
                       "retrieved_sources": [source_name(item.chunk.metadata.get("source"))
                                             for item in result.sources],
                       "cited_sources": [source_name(item.source) for item in result.evidence_chunks],
                       "answer": result.answer,
                       "refused": not bool(result.evidence_chunks),
                       "latency_ms": round((time.perf_counter()-started)*1000, 3),
                       "model_calls": llm.calls, "vector_backend": vector_backend,
                       "model": model}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{case['id']}: {llm.calls} 次 LLM 调用")


def main() -> None:
    parser = argparse.ArgumentParser(description="调用真实 RAG 链路采集固定样本预测（会产生费用）")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "dataset.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "predictions.jsonl")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus")
    parser.add_argument("--vector-backend", choices=["memory", "qdrant-local"], default="memory")
    args = parser.parse_args()
    asyncio.run(collect(args.dataset, args.output, args.model, args.corpus, args.vector_backend))


if __name__ == "__main__":
    main()
