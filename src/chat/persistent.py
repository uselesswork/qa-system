"""从数据库取历史并在问答成功后原子写入一整轮。"""

from fastapi.encoders import jsonable_encoder

from src.chat.repository import ChatRepository
from src.chat.service import LLMQueryRewriter
from src.rag.rag_core import RAGCore


class PersistentChatService:
    def __init__(self, repository: ChatRepository, rag_core: RAGCore,
                 rewriter: LLMQueryRewriter):
        self.repository = repository
        self.rag_core = rag_core
        self.rewriter = rewriter

    async def chat(self, user_id: str, session_id: str, message: str) -> dict:
        if not message.strip():
            raise ValueError("消息不能为空。")
        history = self.repository.history(user_id, session_id)
        rewritten = await self.rewriter.rewrite(message, history)
        result = await self.rag_core.answer(rewritten)
        evidence = [{"rank": e.rank, "score": round(e.score, 4),
                     "source": str(e.source) if e.source is not None else None,
                     "chunk_id": e.chunk_id, "content": e.content}
                    for e in result.evidence_chunks]
        metadata = jsonable_encoder(result.metadata or {})
        self.repository.append_turn(
            user_id, session_id, message, result.answer, rewritten,
            evidence, metadata,
        )
        return {"session_id": session_id, "user_message": message,
                "answer": result.answer, "rewritten_query": rewritten,
                "evidence": evidence, "metadata": metadata}
