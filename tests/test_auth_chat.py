"""阶段 11：账号边界与持久化会话的无密钥回归。"""

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
from sqlalchemy import create_engine, event

from app.main import app
from app.api import chat as chat_api
from src.auth.service import AuthService
from src.chat.repository import ChatRepository
from src.db import Base, Document, KnowledgeBase, User
from src.db.session import create_session_factory
from src.embeddings.embedded_chunk import EmbeddedChunk
from src.knowledge.service import KnowledgeService
from src.vectorstores.qdrant_store import QdrantVectorStore


class FakeRAG:
    async def answer(self, query):
        evidence = SimpleNamespace(rank=1, score=0.9, source="owner.md",
                                   chunk_id="c1", content="仅归属用户可见")
        return SimpleNamespace(answer=f"回答：{query}", evidence_chunks=[evidence],
                               metadata={"model": "fake"})


class FakeRewriter:
    async def rewrite(self, query, history):
        return f"{history[-1].content} / {query}" if history else query


async def sign_in(client, username):
    credentials = {"username": username, "password": "long-test-password-123"}
    registered = await client.post("/auth/register", json=credentials)
    assert registered.status_code == 201, registered.text
    response = await client.post("/auth/login", json=credentials)
    assert response.status_code == 200, response.text
    client.headers["X-CSRF-Token"] = response.json()["csrf_token"]
    return response.json()


def test_auth_ownership_chat_persistence_and_csrf(tmp_path, monkeypatch):
    db_path = tmp_path / "stage11.sqlite3"
    url = f"sqlite+pysqlite:///{db_path.as_posix()}"

    def make_engine():
        engine = create_engine(url)
        @event.listens_for(engine, "connect")
        def foreign_keys(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")
        Base.metadata.create_all(engine)
        return engine

    async def scenario():
        engine = make_engine()
        factory = create_session_factory(engine)
        vectors = QdrantVectorStore.local(tmp_path / "vectors")
        knowledge = KnowledgeService(factory, vectors, None, tmp_path / "uploads")
        monkeypatch.setattr(app.state, "auth_service", AuthService(factory), raising=False)
        monkeypatch.setattr(app.state, "knowledge_service", knowledge, raising=False)
        monkeypatch.setattr(app.state, "query_rewriter", FakeRewriter(), raising=False)
        monkeypatch.setattr(chat_api, "session_core", lambda *_: FakeRAG())
        legacy = knowledge.create_kb("迁移前演示库", None)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as alice, \
                   httpx.AsyncClient(transport=transport, base_url="http://test") as bob:
            assert (await alice.get("/knowledge-bases")).status_code == 401
            alice_user = await sign_in(alice, "alice")
            await sign_in(bob, "bob")
            assert (await alice.get("/knowledge-bases")).json() == []
            assert (await bob.get(f"/knowledge-bases/{legacy['id']}")).status_code == 404
            assert (await alice.post(
                "/chat/sessions", json={"knowledge_base_id": legacy["id"]}
            )).status_code == 404
            own = await alice.post("/knowledge-bases", json={"name": "Alice 私有库"})
            assert own.status_code == 201, own.text
            kb_id = own.json()["id"]
            assert [item["id"] for item in (await alice.get("/knowledge-bases")).json()] == [kb_id]
            assert (await bob.get("/knowledge-bases")).json() == []
            assert (await bob.get(f"/knowledge-bases/{kb_id}/documents")).status_code == 404
            assert (await bob.post(f"/knowledge-bases/{kb_id}/ask", json={"query": "秘密"})).status_code == 404
            assert (await alice.post("/knowledge-bases", json={"name": "无 CSRF"},
                                     headers={"X-CSRF-Token": "wrong"})).status_code == 403
            created = await alice.post("/chat/sessions", json={"knowledge_base_id": kb_id})
            assert created.status_code == 201, created.text
            session_id = created.json()["id"]
            assert created.json()["knowledge_base_id"] == kb_id
            assert (await bob.post("/chat/sessions", json={"knowledge_base_id": kb_id})).status_code == 404
            first = await alice.post(f"/chat/sessions/{session_id}/messages",
                                     json={"message": "第一问"})
            assert first.status_code == 200, first.text
            assert first.json()["evidence"][0]["source"] == "owner.md"
            assert (await bob.get(f"/chat/sessions/{session_id}")).status_code == 404
            assert (await bob.post(f"/chat/sessions/{session_id}/messages",
                                   json={"message": "越权"})).status_code == 404
            saved = (await alice.get(f"/chat/sessions/{session_id}")).json()
            assert [message["role"] for message in saved["messages"]] == ["user", "assistant"]
            assert saved["messages"][1]["evidence"][0]["content"] == "仅归属用户可见"

            # 重建连接和仓储，模拟进程重启后从 MySQL/SQLite 重新读取。
            engine.dispose()
            engine = make_engine()
            app.state.auth_service = AuthService(create_session_factory(engine))
            assert (await alice.get("/auth/me")).json()["id"] == alice_user["id"]
            assert (await alice.get(f"/chat/sessions/{session_id}")).json()["messages"] == saved["messages"]
            second = await alice.post(f"/chat/sessions/{session_id}/messages",
                                      json={"message": "继续追问"})
            assert second.status_code == 200, second.text
            assert "回答：回答：第一问 / 继续追问" == second.json()["answer"]
            assert len((await alice.get(f"/chat/sessions/{session_id}")).json()["messages"]) == 4
            assert (await alice.post("/auth/logout")).status_code == 204
            assert (await alice.get(f"/chat/sessions/{session_id}")).status_code == 401
        vectors.close()
        engine.dispose()

    asyncio.run(scenario())


def test_failed_model_call_does_not_store_partial_turn(tmp_path):
    from src.chat.persistent import PersistentChatService
    from src.db import User

    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'chat.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    with factory.begin() as session:
        user = User(username="test", password_hash="not-used")
        session.add(user)
        session.flush()
        user_id = user.id
    repo = ChatRepository(factory)
    session_id = repo.create(user_id)["id"]

    class FailingRAG:
        async def answer(self, query):
            raise RuntimeError("model unavailable")

    async def run():
        try:
            await PersistentChatService(repo, FailingRAG(), FakeRewriter()).chat(
                user_id, session_id, "问题"
            )
        except RuntimeError:
            pass

    asyncio.run(run())
    assert repo.get(user_id, session_id)["messages"] == []
    engine.dispose()


def test_global_retrieval_filters_unowned_and_other_user_vectors(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'scope.sqlite3').as_posix()}")
    Base.metadata.create_all(engine)
    factory = create_session_factory(engine)
    vectors = QdrantVectorStore.local(tmp_path / "scope_vectors")
    service = KnowledgeService(factory, vectors, None, tmp_path / "uploads")
    with factory.begin() as session:
        owner = User(username="owner", password_hash="unused")
        other = User(username="other", password_hash="unused")
        session.add_all([owner, other])
        session.flush()
        owner_id = owner.id
        for index, kb_owner in enumerate((owner.id, other.id, None)):
            kb = KnowledgeBase(name=f"kb-{index}", owner_id=kb_owner)
            session.add(kb)
            session.flush()
            doc = Document(knowledge_base_id=kb.id, original_name=f"{index}.md",
                           file_type="md", sha256=str(index) * 64,
                           size_bytes=1, status="ready")
            session.add(doc)
            session.flush()
            vectors.add([EmbeddedChunk(
                f"content-{index}", np.array([1.0, 0.0], dtype=np.float32),
                {"chunk_id": "00000000-0000-0000-0000-00000000000" + str(index),
                 "document_id": doc.id, "knowledge_base_id": kb.id,
                 "source": f"{index}.md"},
            )])
    allowed = service.owner_ready_document_ids(owner_id)
    assert len(allowed) == 1
    scoped = QdrantVectorStore(vectors.client, vectors.collection,
                               ready_document_ids=allowed)
    results = scoped.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)
    assert [result.chunk.content for result in results] == ["content-0"]
    vectors.close()
    engine.dispose()
