from typing import List
import math

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk, AggregatedAnswer
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
    ) -> AggregatedAnswer:

        # ==========================================================
        # Phase 1: 用【全部 chunks】生成一个“全局答案”
        # ==========================================================

        global_prompt = self.prompt_builder.build(
            query=query,
            chunks=results,  # 不截断
        )

        analysis_answer = await self.llm.generate(global_prompt)
        analysis_text = analysis_answer.content

        # ==========================================================
        # Phase 2: 计算 embedding
        # ==========================================================

        embeddings = await self.embedder.embed_texts(
            [query, analysis_text]
        )
        query_vec = embeddings[0].vector
        answer_vec = embeddings[1].vector

        # ✅ 新增：所有 chunk embedding
        chunk_vecs = [r.chunk.vector for r in results]

        # ==========================================================
        # Phase 3: 跨文档 Evidence 评分（关键改动）
        # ==========================================================

        evidence_chunks: List[EvidenceChunk] = []

        raw_consensus = [
            cosine_similarity(answer_vec, v) for v in chunk_vecs
        ]
        max_c = max(raw_consensus) or 1.0
        consensus_scores = [c / max_c for c in raw_consensus]

        for idx, r in enumerate(results):
            chunk_vec = r.chunk.vector
            content = r.chunk.content

            # 1️⃣ 与最终答案的一致性（局部）
            answer_sim = cosine_similarity(answer_vec, chunk_vec)

            # 2️⃣ 与原始问题的相关性
            query_sim = cosine_similarity(query_vec, chunk_vec)

            # 3️⃣ 信息密度惩罚（keyword bomb 杀手）
            tokens = content.split()
            token_count = max(len(tokens), 1)
            unique_ratio = len(set(tokens)) / token_count
            info_density = math.log(1 + token_count) * unique_ratio

            # ======================================================
            # ✅ 4️⃣ 跨文档“共识奖励”（真正拉开差距的地方）
            # ======================================================

            consensus_boost = consensus_scores[idx]

            final_score = (
                    0.45 * answer_sim +
                    0.25 * query_sim +
                    0.15 * info_density +
                    0.15 * consensus_boost  # 🔥 新增项
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

        return AggregatedAnswer(
            answer=analysis_text,       # 全局答案
            evidence_chunks=final_chunks
        )

