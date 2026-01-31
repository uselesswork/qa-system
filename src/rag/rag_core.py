import time
import asyncio

# 添加条件导入以提高健壮性
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # 创建一个空装饰器作为替代
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.embeddings.base import BaseEmbedding
from src.vectorstores.base import BaseVectorStore
from src.retrievers.base import BaseRetriever
from src.prompts.base import BasePromptBuilder
from src.llms.base import BaseLLM
from .result import RAGResult


class RAGCore:

    def __init__(
        self,
        embedder: BaseEmbedding,
        vectorstore: BaseVectorStore,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        reranker=None, # 新增（可选）
        aggregator=None, # 新增（可选）
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.reranker = reranker # 新增（可选）
        self.aggregator = aggregator # 新增（可选）
        self.top_k = top_k

    @traceable(name="rag_answer")
    async def answer(self, query: str) -> RAGResult:

        start_time = time.time()

        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k * 2   # rerank 前多取
        )

        # 2. Rerank（新增）
        if self.reranker:
            query_embedding = await self.retriever.embedder.embed_texts([query])
            query_vector = query_embedding[0].vector

            retrieved_chunks = self.reranker.rerank(
                query_vector=query_vector,
                results=retrieved_chunks,
                top_k=self.top_k
            )

        # 3. Aggregator（新增）
        aggregated = None
        if self.aggregator:
            aggregated = await self.aggregator.aggregate(
                query=query,
                results=retrieved_chunks
            )

        # 4. 构建 Prompt
        prompt = self.prompt_builder.build(
            query=query,
            chunks=retrieved_chunks
        )

        # 5. LLM 生成回答
        llm_answer = await self.llm.generate(prompt)

        elapsed = round(time.time() - start_time, 3)

        if aggregated:
            final_answer = aggregated.answer
            evidence_chunks = aggregated.evidence_chunks
        else:
            final_answer = llm_answer.content
            evidence_chunks = []

        return RAGResult(
            answer=final_answer,
            sources=retrieved_chunks,
            evidence_chunks=evidence_chunks,  # 新增（模块11）
            metadata={
                "latency": elapsed,
                "top_k": self.top_k,
                "llm": llm_answer.metadata,
            }
        )

    @traceable(name="rag_answer_stream")
    async def stream_answer(self, query: str):
        """
        返回 async generator，每次 yield 增量文本
        """
        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k * 2
        )

        # 2. Rerank
        if self.reranker:
            query_embedding = await self.retriever.embedder.embed_texts([query])
            query_vector = query_embedding[0].vector

            retrieved_chunks = self.reranker.rerank(
                query_vector=query_vector,
                results=retrieved_chunks,
                top_k=self.top_k
            )

        # 3. Aggregator
        aggregated = None
        if self.aggregator:
            aggregated = await self.aggregator.aggregate(
                query=query,
                results=retrieved_chunks
            )

        # 4. 根据是否有聚合器决定生成方式
        if aggregated:
            # 流式返回聚合器生成的答案
            answer_text = aggregated.answer
            # 按字符流式返回
            for char in answer_text:
                yield char
                await asyncio.sleep(0.03)
        else:
            # 如果没有聚合器，使用原来流式生成方式
            # 构建 Prompt
            prompt = self.prompt_builder.build(
                query=query,
                chunks=retrieved_chunks
            )

            # LLM 流式生成
            async for chunk in self.llm.stream_generate(prompt):
                yield chunk  # 逐步返回文本

        # ⚠️ 注意：这里不返回最终 RAGResult，如果前端需要 Evidence / metadata，需要再调用 answer()