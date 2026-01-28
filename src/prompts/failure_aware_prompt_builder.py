from typing import List
from src.prompts.base import BasePromptBuilder
from src.vectorstores.search_result import SearchResult


class FailureAwarePromptBuilder(BasePromptBuilder):
    """
    Failure-aware Answer PromptBuilder

    核心目标：
    - 强制模型分析 embedding-only 在对抗性文档中的失败原因
    - 要求跨文档归因，而不是逐段复述
    - 明确禁止“未在文档中找到相关信息”作为答案
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

        prompt = f"""
你是一个研究级 RAG 系统分析助手。

下面给出的是一组 **可能包含对抗性噪声、误导性解释、以及碎片化真实信息** 的文档片段。

====================
【文档片段】
====================
{context}

====================
【问题】
====================
{query}

====================
【你的任务（必须全部完成）】
====================

1️⃣ **失败分析（核心）**  
解释：  
为什么在这些对抗性文档中，**仅依赖 embedding similarity 的重排方法会失败**？  
- 指出至少一种“看起来合理但实际上错误 / 误导”的模式  
- 说明 embedding-only 为什么容易被这种模式欺骗

2️⃣ **跨文档归因（关键）**  
即使没有单一文档完整说明失败原因，也请：
- 结合多个文档中的信息
- 抽象出“失败的共同机制”

3️⃣ **方法对比（结论）**  
说明：  
**Answer-aware 方法是如何克服上述失败问题的？**  
- 必须提到“基于生成答案反向评估证据”这一点

====================
【严格约束】
====================
- 不允许回答“未在文档中找到相关信息”
- 不要求逐字引用原文，允许抽象总结
- 回答必须是“机制解释”，而不是定义复述
""".strip()

        return prompt
