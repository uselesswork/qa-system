from typing import List
from .base import BasePromptBuilder


class SimplePromptBuilder(BasePromptBuilder):
    def build(self, query: str, contexts: List[str]) -> str:
        context_text = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(contexts)
        )

        prompt = f"""
        你是一个严谨的智能问答助手，请基于给定的上下文回答问题。
    
        【上下文】
        {context_text}
    
        【问题】
        {query}
    
        【要求】
        - 仅基于上下文回答
        - 如果无法从上下文得到答案，请回答“未在文档中找到相关信息”
        """.strip()

        return prompt
