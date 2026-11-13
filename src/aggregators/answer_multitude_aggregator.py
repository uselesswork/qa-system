from typing import List
import math
import json

from src.aggregators.base import BaseAnswerAggregator
from src.aggregators.schemas import EvidenceChunk, AggregatedAnswer
from src.vectorstores.search_result import SearchResult
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.base import BasePromptBuilder
from src.utils.similarity import cosine_similarity
from src.utils.query_subject import mentioned_dynasties


class CrossDocumentAnswerAggregator(BaseAnswerAggregator):
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

        if not results:
            return AggregatedAnswer(answer="未检索到可用资料，无法依据当前知识库回答。", evidence_chunks=[])

        # 向量相似度只能召回候选片段，不能证明它们支持答案。先检查是否有
        # 可在原文中定位的直接证据；没有证据时不要让模型凭常识补全。
        excerpts = "\n\n".join(
            f"片段 {index}:\n{item.chunk.content}"
            for index, item in enumerate(results, 1)
        )
        check = await self.llm.generate(f"""
你是知识库证据审核员。只根据下面的片段判断能否回答问题，不能使用常识或外部知识。
问题：{query}

{excerpts}

若片段没有直接回答问题所需的信息，特别是问题中的人物、朝代、时间或对象与片段不符，返回
{{"supported": false, "quotes": []}}。
若可以回答，返回 {{"supported": true, "quotes": [{{"index": 1, "quote": "原文中连续出现的直接证据"}}]}}。
quote 必须是对应片段中至少 4 个字符的原文连续摘录。只输出 JSON，不要附加说明。
""".strip())
        try:
            verdict = json.loads(check.content.strip())
            supported = verdict.get("supported") is True
            selected = []
            if supported:
                for citation in verdict.get("quotes", []):
                    index = citation.get("index")
                    quote = citation.get("quote")
                    if (type(index) is int and 1 <= index <= len(results)
                            and isinstance(quote, str) and len(quote.strip()) >= 4
                            and quote in results[index - 1].chunk.content
                            and index not in selected):
                        selected.append(index)
        except (AttributeError, TypeError, ValueError):
            selected = []
        if not selected:
            return AggregatedAnswer(
                answer="当前知识库没有足以回答这个问题的资料。可补充文档，或主动开启联网搜索。",
                evidence_chunks=[],
            )
        results = [results[index - 1] for index in selected]
        required_dynasties = mentioned_dynasties(query)
        evidence_text = "\n".join(item.chunk.content for item in results)
        if any(name not in evidence_text for name in required_dynasties):
            return AggregatedAnswer(
                answer="当前知识库没有足以回答这个问题的资料。可补充文档，或主动开启联网搜索。",
                evidence_chunks=[],
            )

        # 只把通过检查的片段交给答案模型，避免分析文本引入额外事实。
        synthesis_prompt = self.prompt_builder.build(query=query, chunks=results)

        synthesized_answer = await self.llm.generate(synthesis_prompt)
        synthesized_text = synthesized_answer.content

        # ==========================================================
        # Phase 2: 计算 embedding
        # ==========================================================

        embeddings = await self.embedder.embed_texts(
            [query, synthesized_text]
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
            # ✅ 4️⃣ 跨文档"共识奖励"（真正拉开差距的地方）
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
            answer=synthesized_text,  # 经过跨文档分析的全局答案
            evidence_chunks=final_chunks
        )
