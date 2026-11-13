from typing import List

from .memory import ChatMemory
from .schemas import ChatMessage, ChatTurn, Evidence
from .rewriter import BaseQueryRewriter
from src.rag.rag_core import RAGCore
from ..llms.openai_chat import OpenAIChatLLM
from src.utils.query_subject import mentioned_dynasties


class LLMQueryRewriter(BaseQueryRewriter):

    def __init__(self, llm: OpenAIChatLLM, max_turns: int = 5):
        self.llm = llm
        self.max_turns = max_turns

    async def rewrite(self, query: str, history: List[ChatMessage]) -> str:
        if not history:
            return query.strip()
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
            
            请将当前问题改写为一条可独立检索的问题。
            保留当前问题中的专有名词、人物、朝代、时间、限定条件和否定词。
            历史只能用来补全“它”“当时”等指代，不能改变当前问题的主题。
            不要添加用户未提及的书名、史料或事实。
            只输出问题正文，不要输出标签、占位符、解释或答案。
        """

        result = await self.llm.generate(prompt)
        rewritten = result.content.strip()

        # 模型有时把输出示例原样返回，不能将占位符送入检索。
        if (not rewritten or "<REWRITTEN_QUERY>" in rewritten
                or not mentioned_dynasties(query) <= mentioned_dynasties(rewritten)):
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

        user_msg = ChatMessage(role="user", content=user_input)

        # 只有 RAG 成功后才写入完整的一轮对话。
        rag_result = await self.rag_core.answer(rewritten_query)

        assistant_msg = ChatMessage(
            role="assistant",
            content=rag_result.answer
        )
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

        self.memory.append_turn(session_id, user_msg, assistant_msg)

        return ChatTurn(
            user_message=user_msg,
            assistant_message=assistant_msg,
            rewritten_query=rewritten_query,
            evidence=evidence,
            metadata=rag_result.metadata
        )
