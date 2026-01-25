from typing import List

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk, AggregatedAnswer
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity


class AnswerAwareAggregator(BaseAnswerAggregator):
    """
    多文档 Answer-aware 聚合：
    - 先生成初步答案
    - 再基于答案 embedding 对所有文档 chunk 做 evidence-aware 重排
    """

    def __init__(
        self,
        llm: OpenAIChatLLM,
        prompt_builder: BasePromptBuilder,
        embedder: OpenAIEmbedding,
        max_chunks: int = 5,
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

        # 1. 生成初步答案
        top_results = results[: self.max_chunks]  # 先取前几个 chunks 构建 prompt
        prompt = self.prompt_builder.build(query=query, chunks=top_results)
        llm_answer = await self.llm.generate(prompt)
        answer_text = llm_answer.content

        # 2. 生成 answer embedding
        answer_embedding = await self.embedder.embed_texts([answer_text])
        answer_vector = answer_embedding[0].vector

        # 3. 对每个 chunk 计算 answer-aware score
        evidence_chunks: List[EvidenceChunk] = []
        for r in results:
            chunk_vector = r.chunk.vector
            score = cosine_similarity(answer_vector, chunk_vector)
            evidence_chunks.append(
                EvidenceChunk(
                    rank=0,
                    score=score,
                    content=r.chunk.content,
                    source=r.chunk.metadata.get("source"),
                    chunk_id=r.chunk.metadata.get("chunk_id"),
                )
            )

        # 4. 按 score 排序
        evidence_chunks.sort(key=lambda x: x.score, reverse=True)

        # 5. 标 rank，并截取前 max_chunks
        for i, e in enumerate(evidence_chunks[: self.max_chunks]):
            e.rank = i + 1

        return AggregatedAnswer(
            answer=answer_text,
            evidence_chunks=evidence_chunks[: self.max_chunks]
        )
