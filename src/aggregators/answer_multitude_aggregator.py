from typing import List
import math

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity


class AnswerAwareAggregator(BaseAnswerAggregator):
    """
    跨文档 Answer-aware Evidence Aggregator

    特点：
    - 使用【所有 chunks】生成一个全局答案
    - 用答案反向评估每个 chunk 的有效性
    - 显式惩罚 keyword-bomb / 空洞段落
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
    ) -> List[EvidenceChunk]:

        # ==========================================================
        # Phase 1: 用【全部 chunks】生成一个“全局答案”
        # ==========================================================

        global_prompt = self.prompt_builder.build(
            query=query,
            chunks=results,   # 🔥 不截断
        )

        llm_answer = await self.llm.generate(global_prompt)
        answer_text = llm_answer.content

        # ==========================================================
        # Phase 2: 计算 embedding
        # ==========================================================

        embeddings = await self.embedder.embed_texts(
            [query, answer_text]
        )
        query_vec = embeddings[0].vector
        answer_vec = embeddings[1].vector

        evidence_chunks: List[EvidenceChunk] = []

        # ==========================================================
        # Phase 3: 跨文档 Evidence 评分
        # ==========================================================

        for r in results:
            chunk_vec = r.chunk.vector
            content = r.chunk.content

            # 1️⃣ 与最终答案的一致性（最重要）
            answer_sim = cosine_similarity(answer_vec, chunk_vec)

            # 2️⃣ 与原始问题的相关性（防止跑题）
            query_sim = cosine_similarity(query_vec, chunk_vec)

            # 3️⃣ 信息密度惩罚（keyword bomb 杀手）
            token_count = max(len(content.split()), 1)
            unique_ratio = len(set(content.split())) / token_count
            info_density = math.log(1 + token_count) * unique_ratio

            final_score = (
                0.5 * answer_sim +
                0.3 * query_sim +
                0.2 * info_density
            )

            evidence_chunks.append(
                EvidenceChunk(
                    rank=0,
                    score=final_score,
                    content=content,
                    source=r.chunk.metadata.get("source"),
                    chunk_id=r.chunk.metadata.get("chunk_id"),
                )
            )

        # ==========================================================
        # Phase 4: 排序 & 截断
        # ==========================================================

        evidence_chunks.sort(key=lambda x: x.score, reverse=True)

        final_chunks = []
        for i, e in enumerate(evidence_chunks[: self.max_chunks]):
            e.rank = i + 1
            final_chunks.append(e)

        return final_chunks
