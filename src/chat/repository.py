"""按用户隔离的 MySQL 聊天会话与完整轮次。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from src.chat.schemas import ChatMessage
from src.db import ChatMessageRecord, ChatSession, KnowledgeBase


class ChatNotFound(Exception):
    pass


class ChatRepository:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions

    @staticmethod
    def session_json(item: ChatSession) -> dict:
        return {"id": item.id, "title": item.title,
                "knowledge_base_id": item.knowledge_base_id,
                "created_at": item.created_at, "updated_at": item.updated_at}

    @staticmethod
    def message_json(item: ChatMessageRecord) -> dict:
        return {"role": item.role, "content": item.content,
                "rewritten_query": item.rewritten_query,
                "evidence": json.loads(item.evidence_json or "[]"),
                "metadata": json.loads(item.metadata_json or "{}")}

    def create(self, user_id: str, knowledge_base_id: str | None = None) -> dict:
        with self.sessions.begin() as session:
            if knowledge_base_id is not None:
                kb = session.get(KnowledgeBase, knowledge_base_id)
                if kb is None or kb.owner_id != user_id:
                    raise ChatNotFound
            item = ChatSession(user_id=user_id, knowledge_base_id=knowledge_base_id,
                               title="新对话")
            session.add(item)
            session.flush()
            return self.session_json(item)

    def scope(self, user_id: str, session_id: str) -> str | None:
        with self.sessions() as session:
            item = session.get(ChatSession, session_id)
            if item is None or item.user_id != user_id:
                raise ChatNotFound
            if item.knowledge_base_id is not None:
                kb = session.get(KnowledgeBase, item.knowledge_base_id)
                if kb is None or kb.owner_id != user_id:
                    raise ChatNotFound
            return item.knowledge_base_id

    def list(self, user_id: str) -> list[dict]:
        with self.sessions() as session:
            return [self.session_json(item) for item in session.scalars(
                select(ChatSession).where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
            )]

    def get(self, user_id: str, session_id: str) -> dict:
        with self.sessions() as session:
            item = session.get(ChatSession, session_id)
            if item is None or item.user_id != user_id:
                raise ChatNotFound
            messages = session.scalars(
                select(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id)
                .order_by(ChatMessageRecord.ordinal)
            ).all()
            return {**self.session_json(item),
                    "messages": [self.message_json(message) for message in messages]}

    def history(self, user_id: str, session_id: str, max_turns: int = 10) -> list[ChatMessage]:
        self.get(user_id, session_id)
        with self.sessions() as session:
            rows = session.scalars(
                select(ChatMessageRecord).where(ChatMessageRecord.session_id == session_id)
                .order_by(ChatMessageRecord.ordinal.desc()).limit(max_turns * 2)
            ).all()
            return [ChatMessage(role=row.role, content=row.content)
                    for row in reversed(rows)]

    def append_turn(self, user_id: str, session_id: str, user_text: str,
                    answer: str, rewritten_query: str, evidence: list[dict],
                    metadata: dict) -> None:
        with self.sessions.begin() as session:
            item = session.get(ChatSession, session_id, with_for_update=True)
            if item is None or item.user_id != user_id:
                raise ChatNotFound
            last = session.scalar(select(func.max(ChatMessageRecord.ordinal)).where(
                ChatMessageRecord.session_id == session_id
            ))
            ordinal = (last if last is not None else -1) + 1
            session.add(ChatMessageRecord(
                session_id=session_id, ordinal=ordinal, role="user", content=user_text,
            ))
            session.add(ChatMessageRecord(
                session_id=session_id, ordinal=ordinal + 1, role="assistant",
                content=answer, rewritten_query=rewritten_query,
                evidence_json=json.dumps(evidence, ensure_ascii=False),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            ))
            if item.title == "新对话":
                item.title = user_text[:80]
            item.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
