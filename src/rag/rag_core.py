import time
from typing import AsyncIterator

from src.aggregators.base import BaseAnswerAggregator
from src.embeddings.base import BaseEmbedding
from src.llms.base import BaseLLM
from src.prompts.base import BasePromptBuilder
from src.rerankers.answer_aware_reranker import AnswerAwareReranker
from src.retrievers.base import BaseRetriever
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.search_result import SearchResult

from .result import RAGResult, RAGStreamEvent

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class RAGCore:
    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        reranker=None,
        aggregator: BaseAnswerAggregator | None = None,
        top_k: int = 5,
        enable_answer_aware: bool = False,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.reranker = reranker
        self.aggregator = aggregator
        self.top_k = top_k
        self.enable_answer_aware = enable_answer_aware

    async def _retrieve(self, query: str) -> list[SearchResult]:
        """普通问答与流式问答共用同一套召回和重排流程。"""
        batch = await self.retriever.retrieve_with_vector(
            query=query, top_k=self.top_k * 2
        )
        results = batch.results

        if self.reranker:
            query_vector = batch.query_vector
            if query_vector is None:
                query_embedding = await self.embedder.embed_texts([query])
                query_vector = query_embedding[0].vector
            results = self.reranker.rerank(
                query_vector=query_vector,
                results=results,
                top_k=self.top_k,
            )

        if self.enable_answer_aware:
            draft_prompt = self.prompt_builder.build(query=query, chunks=results)
            draft_answer = await self.llm.generate(draft_prompt)
            answer_reranker = AnswerAwareReranker(
                llm=self.llm,
                query_text=query,
                draft_answer=draft_answer.content,
            )
            results = await answer_reranker.rerank(
                results=results,
                top_k=self.top_k,
            )

        return results

    def _result(self, answer, sources, evidence, started, llm_metadata=None):
        return RAGResult(
            answer=answer,
            sources=sources,
            evidence_chunks=evidence,
            metadata={
                "latency": round(time.perf_counter() - started, 3),
                "top_k": self.top_k,
                "llm": llm_metadata or {},
            },
        )

    @traceable(name="rag_answer")
    async def answer(self, query: str) -> RAGResult:
        started = time.perf_counter()
        results = await self._retrieve(query)

        if self.aggregator:
            aggregated = await self.aggregator.aggregate(query=query, results=results)
            return self._result(
                aggregated.answer, results, aggregated.evidence_chunks, started
            )

        prompt = self.prompt_builder.build(query=query, chunks=results)
        llm_answer = await self.llm.generate(prompt)
        return self._result(
            llm_answer.content, results, [], started, llm_answer.metadata
        )

    @traceable(name="rag_answer_stream")
    async def stream_answer_with_result(
        self, query: str
    ) -> AsyncIterator[RAGStreamEvent]:
        """流式文本和最终证据来自同一次检索/生成，不再重复回答。"""
        started = time.perf_counter()
        results = await self._retrieve(query)

        if self.aggregator:
            aggregated = await self.aggregator.aggregate(query=query, results=results)
            # 聚合器当前只提供完整答案；这里只分块输出，不假装模型实时生成。
            for index in range(0, len(aggregated.answer), 64):
                yield RAGStreamEvent(answer_chunk=aggregated.answer[index:index + 64])
            yield RAGStreamEvent(
                result=self._result(
                    aggregated.answer, results, aggregated.evidence_chunks, started
                )
            )
            return

        prompt = self.prompt_builder.build(query=query, chunks=results)
        answer_parts = []
        async for chunk in self.llm.stream_generate(prompt):
            answer_parts.append(chunk)
            yield RAGStreamEvent(answer_chunk=chunk)
        yield RAGStreamEvent(
            result=self._result("".join(answer_parts), results, [], started)
        )

    async def stream_answer(self, query: str) -> AsyncIterator[str]:
        """保留原有纯文本流接口，供只关心答案文本的调用方使用。"""
        async for event in self.stream_answer_with_result(query):
            if event.answer_chunk is not None:
                yield event.answer_chunk
