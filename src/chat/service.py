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
            你是一个 RAG 查询重写器。
            根据对话历史，将用户问题改写为**独立、完整、可用于文档检索的问题**。
            不要回答问题。
            
            对话历史：
            {history_text}
            
            用户当前问题：
            {query}
            
            重写后的检索问题：
        """
        resp = await self.llm.generate(prompt)
        return resp.content.strip()

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
