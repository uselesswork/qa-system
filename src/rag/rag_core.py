import time
from typing import List

# 添加条件导入以提高健壮性
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # 创建一个空的装饰器作为替代
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
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.top_k = top_k

    @traceable(name="rag_answer")
    async def answer(self, query: str) -> RAGResult:

        start_time = time.time()

        # 1. 检索相关 chunks
        retrieved_chunks = await self.retriever.retrieve(
            query=query,
            top_k=self.top_k
        )

        # 2. 构建 Prompt
        prompt = self.prompt_builder.build(
            query=query,
            chunks=retrieved_chunks
        )

        # 3. LLM 生成回答
        llm_answer = await self.llm.generate(prompt)

        elapsed = round(time.time() - start_time, 3)

        return RAGResult(
            answer=llm_answer.content,
            sources=retrieved_chunks,
            metadata={
                "latency": elapsed,
                "top_k": self.top_k,
                "llm": llm_answer.metadata,
            }
        )
