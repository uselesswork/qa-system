import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
from sqlalchemy import create_engine, event, select

from app.main import app
from src.auth.service import AuthService
from src.aggregators.answer_multitude_aggregator import CrossDocumentAnswerAggregator
from src.db import Base, Chunk, Document
from src.db.session import create_session_factory
from src.embeddings.embedded_chunk import EmbeddedChunk
from src.knowledge.service import KnowledgeService
from src.rag.rag_core import RAGCore
from src.vectorstores.qdrant_store import QdrantVectorStore


class FakeEmbedding:
    async def embed(self, chunks):
        return [EmbeddedChunk(c.content, np.array([1.0, 0.0], dtype=np.float32),
                              dict(c.metadata)) for c in chunks]

    async def embed_texts(self, texts):
        return [EmbeddedChunk(text, np.array([1.0, 0.0], dtype=np.float32), {})
                for text in texts]


class FakeLLM:
    async def generate(self, prompt):
        return SimpleNamespace(content=prompt, metadata={})


class FailingEmbedding(FakeEmbedding):
    async def embed(self, chunks):
        raise RuntimeError("simulated embedding outage")


class ThreeDimensionEmbedding(FakeEmbedding):
    async def embed(self, chunks):
        return [EmbeddedChunk(c.content, np.array([1.0, 0.0, 0.0], dtype=np.float32),
                              dict(c.metadata)) for c in chunks]


def make_service(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{(tmp_path / 'db.sqlite').as_posix()}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    service = KnowledgeService(
        create_session_factory(engine), QdrantVectorStore.local(tmp_path / "vectors"),
        FakeEmbedding(), tmp_path / "uploads",
    )
    return engine, service


async def login_client(client, engine):
    app.state.auth_service = AuthService(create_session_factory(engine))
    registered = await client.post("/auth/register", json={
        "username": "tester", "password": "a-long-test-password"
    })
    assert registered.status_code == 201, registered.text
    logged_in = await client.post("/auth/login", json={
        "username": "tester", "password": "a-long-test-password"
    })
    assert logged_in.status_code == 200, logged_in.text
    client.headers["X-CSRF-Token"] = logged_in.json()["csrf_token"]


def test_knowledge_api_persists_filters_reindexes_and_deletes(tmp_path):
    async def scenario():
        engine, service = make_service(tmp_path)
        app.state.knowledge_service = service
        prompt_builder = SimpleNamespace(
            build=lambda query, chunks: "|".join(r.chunk.content for r in chunks)
        )
        app.state.rag_core = RAGCore(
            embedder=FakeEmbedding(), vectorstore=None, retriever=None,
            prompt_builder=prompt_builder, llm=FakeLLM(), top_k=5,
            aggregator=CrossDocumentAnswerAggregator(
                llm=FakeLLM(), prompt_builder=prompt_builder,
                embedder=FakeEmbedding(), max_chunks=3,
            ),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await login_client(client, engine)
            first = (await client.post("/knowledge-bases", json={"name": "甲库"})).json()["id"]
            second = (await client.post("/knowledge-bases", json={"name": "乙库"})).json()["id"]
            assert len((await client.get("/knowledge-bases")).json()) == 2

            first_doc = await client.post(
                f"/knowledge-bases/{first}/documents",
                files={"file": ("one.md", "# 标题\n甲库独有证据".encode(), "text/markdown")},
            )
            assert first_doc.status_code == 201, first_doc.text
            first_id = first_doc.json()["id"]
            assert first_doc.json()["status"] == "ready"
            other = await client.post(
                f"/knowledge-bases/{second}/documents",
                files={"file": ("two.txt", "乙库独有证据".encode(), "text/plain")},
            )
            assert other.status_code == 201, other.text
            duplicate = await client.post(
                f"/knowledge-bases/{first}/documents",
                files={"file": ("copy.md", "# 标题\n甲库独有证据".encode())},
            )
            assert duplicate.status_code == 409
            assert len((await client.get(f"/knowledge-bases/{first}/documents")).json()) == 1

            answer = await client.post(f"/knowledge-bases/{first}/ask", json={"query": "什么证据"})
            assert answer.status_code == 200, answer.text
            assert "甲库独有证据" in answer.json()["answer"]
            assert "乙库独有证据" not in answer.json()["answer"]
            assert answer.json()["evidence"]

            service.vectors.close()
            engine.dispose()
            engine, restarted = make_service(tmp_path)
            app.state.knowledge_service = restarted
            app.state.auth_service = AuthService(create_session_factory(engine))
            answer = await client.post(f"/knowledge-bases/{first}/ask", json={"query": "重启后"})
            assert answer.status_code == 200, answer.text
            assert "甲库独有证据" in answer.json()["answer"]

            rebuilt = await client.post(f"/knowledge-bases/{first}/documents/{first_id}/reindex")
            assert rebuilt.status_code == 200, rebuilt.text
            assert rebuilt.json()["status"] == "ready"
            assert (await client.delete(f"/knowledge-bases/{first}/documents/{first_id}")).status_code == 204
            assert (await client.post(f"/knowledge-bases/{first}/ask", json={"query": "删除后"})).status_code == 409
            assert (await client.get(f"/knowledge-bases/{first}/documents")).json() == []
            with create_session_factory(engine)() as session:
                assert session.scalars(select(Document).where(Document.knowledge_base_id == first)).all() == []
                assert session.scalars(select(Chunk)).all()  # 乙库片段仍在
            restarted.vectors.close()
        engine.dispose()

    asyncio.run(scenario())


def test_upload_validation_and_index_failure(tmp_path):
    async def scenario():
        engine, service = make_service(tmp_path)
        app.state.knowledge_service = service
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await login_client(client, engine)
            kb_id = (await client.post("/knowledge-bases", json={"name": "测试"})).json()["id"]
            bad = await client.post(f"/knowledge-bases/{kb_id}/documents",
                                    files={"file": ("evil.exe", b"abc")})
            assert bad.status_code == 400
            empty = await client.post(f"/knowledge-bases/{kb_id}/documents",
                                      files={"file": ("empty.md", b"")})
            assert empty.status_code == 413
            invalid = await client.post(f"/knowledge-bases/{kb_id}/documents",
                                        files={"file": ("broken.md", b"\xff\xfe")})
            assert invalid.status_code == 422
            missing = await client.post("/knowledge-bases/missing/documents",
                                        files={"file": ("a.md", b"# A")})
            assert missing.status_code == 404
            assert (await client.get(f"/knowledge-bases/{kb_id}/documents")).json() == []
            service.embedder = FailingEmbedding()
            failed = await client.post(f"/knowledge-bases/{kb_id}/documents",
                                       files={"file": ("fail.md", b"# failure\ntext")})
            assert failed.status_code == 503
            documents = (await client.get(f"/knowledge-bases/{kb_id}/documents")).json()
            assert len(documents) == 1
            assert documents[0]["status"] == "failed"
            assert (await client.post(f"/knowledge-bases/{kb_id}/ask",
                                      json={"query": "question"})).status_code == 409
            service.embedder = FakeEmbedding()
            retry = await client.post(
                f"/knowledge-bases/{kb_id}/documents/{documents[0]['id']}/reindex"
            )
            assert retry.status_code == 200, retry.text
            assert retry.json()["status"] == "ready"
            service.embedder = ThreeDimensionEmbedding()
            mismatch = await client.post(f"/knowledge-bases/{kb_id}/documents",
                                         files={"file": ("wide.md", b"# wide\ntext")})
            assert mismatch.status_code == 503
            assert "向量维度不匹配" in mismatch.json()["detail"]
            assert "集合为 2" in mismatch.json()["detail"]
            assert "新向量为 3" in mismatch.json()["detail"]
        service.vectors.close()
        engine.dispose()

    asyncio.run(scenario())
