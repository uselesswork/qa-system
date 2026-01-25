from typing import List
from .base import BasePromptBuilder
from src.vectorstores.search_result import SearchResult

class SimplePromptBuilder(BasePromptBuilder):

    def build(
        self,
        query: str,
        chunks: List[SearchResult]
    ) -> str:
        """
        为 AnswerAwareAggregator 构建 prompt：
        - 明确指示 LLM 聚焦真正答案
        - 忽略可能的高 embedding 干扰
        """

        # 为每个 chunk 标号
        context_text = "\n\n".join(
            f"[Chunk {i + 1}] {r.chunk.content}" for i, r in enumerate(chunks)
        )

        prompt = f"""
你是一个严谨的智能问答助手，请根据以下上下文回答问题。
注意：
- 请只基于上下文中的真正相关信息回答
- 忽略重复关键词或无用内容
- 即使某些段落看起来与问题相似但信息无效，也不要被误导
- 输出答案前，不要自己添加任何假设信息

【上下文】
{context_text}

【问题】
{query}

【回答要求】
- 只回答上下文中有的信息
- 如果上下文无法提供答案，请回答“未在文档中找到相关信息”
""".strip()

        return prompt
