from typing import List

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import AggregatedAnswer, EvidenceChunk
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity


class AnswerAwareAggregator(BaseAnswerAggregator):
    """
    使用最终 Answer 作为 query，
    对 rerank 后的 chunks 再做一次 evidence 级别重排
    """

    def __init__(
        self,
        llm: OpenAIChatLLM,
        prompt_builder: BasePromptBuilder,
        embedder: OpenAIEmbedding,
        max_chunks: int = 3,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.embedder = embedder
        self.max_chunks = max_chunks

    async def aggregate(
        self,
        query: str,
        results: List[SearchResult],
    ) -> AggregatedAnswer:

        if not results:
            return AggregatedAnswer(answer="未检索到可用资料，无法依据当前知识库回答。", evidence_chunks=[])

        # 1. 先生成最终答案（这是“答案感知”的关键）
        prompt = self.prompt_builder.build(
            query=query,
            chunks=results[: self.max_chunks],
        )
        llm_answer = await self.llm.generate(prompt)
        answer_text = llm_answer.content

        # 2. 对 answer 做 embedding
        answer_embedding = await self.embedder.embed_texts([answer_text])
        answer_vector = answer_embedding[0].vector

        # 3. 对每个 chunk 计算 answer-aware score
        evidence_chunks: List[EvidenceChunk] = []

        for idx, r in enumerate(results):
            chunk_vector = r.chunk.vector
            score = cosine_similarity(answer_vector, chunk_vector)

            evidence_chunks.append(
                EvidenceChunk(
                    rank=0,  # 稍后统一排序后再填
                    score=score,
                    content=r.chunk.content,
                    source=r.chunk.metadata.get("source"),
                    chunk_id=r.chunk.metadata.get("chunk_id"),
                )
            )

        # 4. 按 answer-aware score 重排
        evidence_chunks.sort(key=lambda x: x.score, reverse=True)

        # 5. 重新标 rank，并截断
        final_chunks = []
        for i, e in enumerate(evidence_chunks[: self.max_chunks]):
            e.rank = i + 1
            final_chunks.append(e)

        return AggregatedAnswer(answer=answer_text, evidence_chunks=final_chunks)
