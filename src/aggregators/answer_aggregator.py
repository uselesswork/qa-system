from typing import List
from src.vectorstores.search_result import SearchResult
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from .base import BaseAnswerAggregator
from .schemas import AggregatedAnswer, EvidenceChunk


class AnswerAggregator(BaseAnswerAggregator):

    def __init__(
        self,
        llm: OpenAIChatLLM,
        prompt_builder: BasePromptBuilder,
        max_chunks: int = 3,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.max_chunks = max_chunks

    async def aggregate(
        self,
        query: str,
        results: List[SearchResult]
    ) -> AggregatedAnswer:

        # 1️⃣ 选择真正参与回答的 chunks（通常是 rerank 后的 top-k）
        top_results = results[: self.max_chunks]

        # 2️⃣ 使用 PromptBuilder 构建 prompt（关键变化）
        prompt = self.prompt_builder.build(
            query=query,
            chunks=top_results,
        )

        # 3️⃣ 调用 LLM
        answer = await self.llm.generate(prompt)

        # 4️⃣ 构造 evidence（这是 PromptBuilder 不该管的）
        evidence_chunks = []
        for i, r in enumerate(top_results):
            meta = r.chunk.metadata or {}
            evidence_chunks.append(
                EvidenceChunk(
                    rank=i + 1,
                    score=r.score,
                    content=r.chunk.content,
                    source=meta.get("source"),
                    chunk_id=meta.get("chunk_id"),
                )
            )

        return AggregatedAnswer(
            answer=answer,
            evidence_chunks=evidence_chunks
        )
