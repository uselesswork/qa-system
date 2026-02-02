from typing import List

from .memory import ChatMemory
from .schemas import ChatMessage, ChatTurn, Evidence
from .rewriter import BaseQueryRewriter
from src.rag.rag_core import RAGCore
from ..llms.openai_chat import OpenAIChatLLM


class LLMQueryRewriter(BaseQueryRewriter):

    def __init__(self, llm: OpenAIChatLLM, max_turns: int = 5):
        self.llm = llm
        self.max_turns = max_turns

    async def rewrite(self, query: str, history: List[ChatMessage]) -> str:
        history = history[-self.max_turns * 2:]

        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in history
        )

        prompt = f"""
            你是一个多轮 RAG 系统中的 Query Rewriter。

            系统目标：
            - 用户的问题是递进的
            - 你需要基于“对话历史”，判断当前问题处于什么阶段
            - 只生成【检索新增信息】所需的 query
            - 避免重复检索已解释的基础概念
            
            对话历史（按时间顺序）：
            {history_text}
            
            当前用户问题：
            {query}
            
            请遵循以下规则生成检索 query：
            1. 如果是第一轮：生成“定义 / 原理”型检索 query
            2. 如果是追问失败 / 局限：生成“失败机制 / 对抗性情况”型检索 query
            3. 如果是方法对比：生成“对比 / 解决机制”型检索 query
            4. 不要复述用户问题
            5. 不要给出答案，只输出一条检索 query
            
            输出格式（严格）：
            <REWRITTEN_QUERY>
        """

        result = await self.llm.generate(prompt)
        rewritten = result.content.strip()

        # 防御：空 rewrite 就 fallback
        if not rewritten:
            return query

        return rewritten

class ChatService:

    def __init__(
        self,
        rag_core: RAGCore,
        rewriter: BaseQueryRewriter,
        memory: ChatMemory
    ):
        self.rag_core = rag_core
        self.rewriter = rewriter
        self.memory = memory

    async def chat(self, session_id: str, user_input: str) -> ChatTurn:
        # 1️⃣ 取历史
        history = self.memory.get(session_id)

        # 2️⃣ rewrite（多轮 → 单轮）
        rewritten_query = await self.rewriter.rewrite(user_input, history)

        # 3️⃣ 记录用户输入
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.append(session_id, user_msg)

        # 4️⃣ 核心：直接用你现在的 RAGCore
        rag_result = await self.rag_core.answer(rewritten_query)

        # 5️⃣ 记录 assistant
        assistant_msg = ChatMessage(
            role="assistant",
            content=rag_result.answer
        )
        self.memory.append(session_id, assistant_msg)

        # 6️⃣ Evidence 结构化（支持 embedding-only / answer-aware）
        evidence = []
        for e in rag_result.evidence_chunks:
            evidence.append(
                Evidence(
                    rank=e.rank,
                    score=e.score,
                    source=e.source,
                    chunk_id=e.chunk_id,
                    content=e.content
                )
            )

        return ChatTurn(
            user_message=user_msg,
            assistant_message=assistant_msg,
            rewritten_query=rewritten_query,
            evidence=evidence,
            metadata=rag_result.metadata
        )
