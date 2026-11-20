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
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from dotenv import load_dotenv
import httpx


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
    def __init__(self, model: str, call_sink=None, inner=None, price_version: str | None = None):
        self.inner = inner if inner is not None else OpenAIChatLLM(model=model)
        self.model = model
        self.call_sink = call_sink
        self.price_version = price_version
        self.case_id = None
        self.calls = 0

    async def generate(self, prompt: str):
        self.calls += 1
        started = time.perf_counter()
        answer = await self.inner.generate(prompt)
        metadata = answer.metadata or {}
        record = {"requestId": str(uuid4()), "caseId": self.case_id,
                  "modelId": metadata.get("model") or self.model,
                  "latencyMs": round((time.perf_counter() - started) * 1000, 3)}
        input_tokens = metadata.get("prompt_tokens")
        output_tokens = metadata.get("completion_tokens")
        if all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9
               for value in (input_tokens, output_tokens)):
            record.update(inputTokens=input_tokens, outputTokens=output_tokens)
        if self.price_version:
            record["priceVersion"] = self.price_version
        if self.call_sink is not None:
            self.call_sink(record)
        return answer


def calls_path_for(output: Path) -> Path:
    return output.with_name(f"{output.stem}.calls.jsonl")


def upload_calls(calls_path: Path, hub_url: str, hub_run_id: str,
                 transport: httpx.BaseTransport | None = None) -> tuple[int, int]:
    """重放本地明细；409 表示同一 run/requestId 已存在，不重复创建。"""
    run_url = f"{hub_url.rstrip('/')}/api/runs/{UUID(hub_run_id)}"
    created = duplicates = 0
    with httpx.Client(transport=transport, trust_env=False, timeout=10) as client:
        try:
            run_response = client.get(run_url)
        except httpx.RequestError as exc:
            raise RuntimeError("无法连接 Hub；本地调用明细未删除，可用 --retry-calls 重试。") from exc
        if run_response.status_code != 200:
            raise RuntimeError(f"Hub 运行检查返回 HTTP {run_response.status_code}；请先确认 runId 已存在。")
        for number, line in enumerate(calls_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            payload = {key: record[key] for key in ("requestId", "modelId", "latencyMs",
                       "inputTokens", "outputTokens", "priceVersion") if key in record}
            if not ("inputTokens" in payload and "outputTokens" in payload):
                payload.pop("inputTokens", None)
                payload.pop("outputTokens", None)
            try:
                response = client.post(f"{run_url}/calls", json=payload)
            except httpx.RequestError as exc:
                raise RuntimeError(f"Hub 上报在第 {number} 行中断；本地调用明细仍在 {calls_path}。可用 --retry-calls 重试。") from exc
            if response.status_code == 201:
                created += 1
            elif response.status_code == 409:
                duplicates += 1
            else:
                raise RuntimeError(f"Hub 上报第 {number} 行返回 HTTP {response.status_code}；本地调用明细仍在 {calls_path}。可用 --retry-calls 重试。")
    return created, duplicates


def publish_if_enabled(calls_path: Path, hub_url: str | None,
                       hub_run_id: str | None) -> tuple[int, int] | None:
    if not hub_url:
        return None
    return upload_calls(calls_path, hub_url, hub_run_id)


def source_name(value) -> str:
    return Path(str(value)).name


async def collect(dataset: Path, output: Path, model: str, corpus: Path | None = None,
                  vector_backend: str = "memory", calls_output: Path | None = None,
                  hub_url: str | None = None, hub_run_id: str | None = None,
                  price_version: str | None = None) -> None:
    if bool(hub_url) != bool(hub_run_id):
        raise ValueError("--hub-url 和 --hub-run-id 必须同时提供")
    calls_output = calls_output or calls_path_for(output)
    if calls_output == output:
        raise ValueError("调用明细文件不能与预测文件同名")
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
        calls_output.parent.mkdir(parents=True, exist_ok=True)
        with calls_output.open("w", encoding="utf-8") as call_handle:
            def save_call(record):
                call_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                call_handle.flush()

            llm = CountingLLM(model, save_call, price_version=price_version)
            await _collect_cases(cases, output, model, vector_backend, embedder, store, llm)
    published = publish_if_enabled(calls_output, hub_url, hub_run_id)
    if published is not None:
        created, duplicates = published
        print(f"Hub 上报完成：新增 {created} 条，已存在 {duplicates} 条。")


async def _collect_cases(cases, output, model, vector_backend, embedder, store, llm):
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
            llm.case_id = case["id"]
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
    parser.add_argument("--calls-output", type=Path, help="调用明细 JSONL；默认与预测文件同目录，文件名为 *.calls.jsonl")
    parser.add_argument("--hub-url", help="显式启用 Hub 上报；须与 --hub-run-id 一起使用")
    parser.add_argument("--hub-run-id", help="Hub 中已存在的运行 ID")
    parser.add_argument("--price-version", help="可选价格版本；未配置时不发送")
    parser.add_argument("--retry-calls", type=Path, help="只重传已有调用明细，不调用模型")
    args = parser.parse_args()
    if bool(args.hub_url) != bool(args.hub_run_id):
        parser.error("--hub-url 和 --hub-run-id 必须同时提供")
    if args.retry_calls:
        if not args.hub_url:
            parser.error("--retry-calls 需要 --hub-url 和 --hub-run-id")
        created, duplicates = upload_calls(args.retry_calls, args.hub_url, args.hub_run_id)
        print(f"Hub 重试完成：新增 {created} 条，已存在 {duplicates} 条。")
        return
    asyncio.run(collect(args.dataset, args.output, args.model, args.corpus,
                        args.vector_backend, args.calls_output, args.hub_url,
                        args.hub_run_id, args.price_version))


if __name__ == "__main__":
    main()
