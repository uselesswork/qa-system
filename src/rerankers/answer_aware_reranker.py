from typing import List
import numpy as np

from src.rerankers.base import BaseReranker
from src.vectorstores.search_result import SearchResult


class AnswerAwareReranker(BaseReranker):
    """
    Answer-aware reranker:
    使用 LLM 判断 chunk 是否真正支持 draft answer
    """

    def __init__(
        self,
        llm,
        query_text: str,
        draft_answer: str
    ):
        self.llm = llm
        self.query_text = query_text
        self.draft_answer = draft_answer

    async def rerank(
        self,
        query_vector: np.ndarray,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:

        if not results:
            return []

        reranked = []

        for r in results:
            score = await self._score_chunk(r.chunk.content)

            reranked.append(
                SearchResult(
                    chunk=r.chunk,
                    score=score
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    async def _score_chunk(self, chunk_text: str) -> float:
        prompt = f"""
            You are evaluating whether a document chunk SUPPORTS a draft answer.

            Important:
            - Mere topical similarity or definition overlap is NOT support.
            - The chunk must provide:
              (a) causal explanation, OR
              (b) factual evidence, OR
              (c) reasoning that directly justifies the draft answer.
            
            [Question]
            {self.query_text}
            
            [Draft Answer]
            {self.draft_answer}
            
            [Document Chunk]
            {chunk_text}
            
            Task:
            Does this chunk provide concrete support for the draft answer?
            
            Scoring:
            - 1.0 = explicitly supports the answer with reasoning or evidence
            - 0.5 = partially relevant but weak or indirect support
            - 0.0 = only topically related or unrelated
            
            Only output a number between 0.0 and 1.0.

        """.strip()

        response = await self.llm.generate(prompt)


        try:
            return float(response.content.strip())
        except Exception:
            return 0.0
