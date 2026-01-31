from typing import List
from src.prompts.base import BasePromptBuilder
from src.vectorstores.search_result import SearchResult


class FailureAwarePromptBuilder(BasePromptBuilder):
    """
    基于检索 + 聚合 + 推理的失败机制识别
    """

    def build(
        self,
        query: str,
        chunks: List[SearchResult],
    ) -> str:

        context = "\n\n".join(
            f"[Chunk {i + 1} | Source: {r.chunk.metadata.get('source')}] \n"
            f"{r.chunk.content}"
            for i, r in enumerate(chunks)
        )

        prompt=f"""
            你是一个严谨的问答系统，只能基于给定的上下文回答问题。
    
            == == == == == == == == == ==
            【上下文信息】
            == == == == == == == == == ==
            {context}
    
            == == == == == == == == == ==
            【问题】
            == == == == == == == == == ==
            {query}
    
            == == == == == == == == == ==
            【回答要求（必须遵守）】
            == == == == == == == == == ==
    
            - 你的目标是给出 ** 问题的最终答案 **
            - 只有在上下文 ** 明确、一致地支持某个结论 ** 时，才能给出确定性回答
            - 如果上下文中存在：
            - 信息不完整
            - 关键证据缺失
            - 多个片段之间结论不一致
            请在答案中明确反映这种不确定性（例如使用条件化或范围性表述）
    
            - 不允许引入上下文之外的常识或推测
            - 不需要解释推理过程
            - 不需要评价上下文质量
            - 不需要比较不同方法
    
            == == == == == == == == == ==
            【输出格式】
            == == == == == == == == == ==
    
            直接给出问题的答案，不要包含分析说明。
        """.strip()

        return prompt
