"""真实 MySQL + Qdrant 本地模式的可清理冒烟测试。

默认使用假模型，不向外发送资料；--live 才调用外部模型端点。
仅显式运行，不作为默认 pytest 收集。需要 .env 中的 DATABASE_URL。
"""

import argparse
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

# 冒烟测试不验证追踪；先于 RAGCore/LangSmith 导入关闭，避免离线模式外发内容。
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from sqlalchemy import select

from app.api.knowledge import router
from src.db import Document, KnowledgeBase
from src.db.session import create_database_engine, create_session_factory
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.embeddings.embedded_chunk import EmbeddedChunk
from src.knowledge.service import KnowledgeService
from src.llms.openai_chat import OpenAIChatLLM
from src.prompts.failure_aware_prompt_builder import FailureAwarePromptBuilder
from src.rag.rag_core import RAGCore
from src.vectorstores.qdrant_store import QdrantVectorStore


ROOT = Path(__file__).resolve().parents[1]


class OfflineEmbedding:
    async def embed(self, chunks):
        return [EmbeddedChunk(chunk.content, np.array([1.0, 0.0], dtype=np.float32),
                              dict(chunk.metadata)) for chunk in chunks]

    async def embed_texts(self, texts):
        return [EmbeddedChunk(text, np.array([1.0, 0.0], dtype=np.float32), {})
                for text in texts]


class OfflineLLM:
    async def generate(self, prompt):
        return SimpleNamespace(content=prompt, metadata={})


async def main(live: bool = False) -> None:
    load_dotenv(ROOT / ".env")
    engine = create_database_engine()
    sessions = create_session_factory(engine)
    embedder = OpenAIEmbedding() if live else OfflineEmbedding()
    smoke_collection = f"qa_smoke_{uuid4().hex}"
    vectors = QdrantVectorStore.local(ROOT / "data" / "qdrant", smoke_collection)
    service = KnowledgeService(sessions, vectors, embedder, ROOT / "data" / "uploads")
    app = FastAPI()
    app.include_router(router)
    app.state.knowledge_service = service
    app.state.rag_core = RAGCore(
        embedder=embedder, vectorstore=None, retriever=None,
        prompt_builder=FailureAwarePromptBuilder(),
        llm=OpenAIChatLLM() if live else OfflineLLM(), top_k=3,
    )
    kb_id = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local-test"
        ) as client:
            response = await client.post("/knowledge-bases", json={
                "name": "阶段 10 临时联调库", "description": "冒烟测试结束后删除",
            })
            response.raise_for_status()
            kb_id = response.json()["id"]
            response = await client.post(
                f"/knowledge-bases/{kb_id}/documents",
                files={"file": ("smoke.md", "# 联调证据\n本项目的阶段十实现了按知识库隔离检索。".encode("utf-8"))},
            )
            response.raise_for_status()
            assert response.json()["status"] == "ready"
            response = await client.post(
                f"/knowledge-bases/{kb_id}/ask",
                json={"query": "阶段十实现了什么检索能力？"},
            )
            response.raise_for_status()
            assert "知识库" in response.json()["answer"]

            vectors.close()
            engine.dispose()
            engine = create_database_engine()
            sessions = create_session_factory(engine)
            vectors = QdrantVectorStore.local(ROOT / "data" / "qdrant", smoke_collection)
            service = KnowledgeService(sessions, vectors, embedder, ROOT / "data" / "uploads")
            app.state.knowledge_service = service
            response = await client.post(
                f"/knowledge-bases/{kb_id}/ask",
                json={"query": "重启后，阶段十实现了什么检索能力？"},
            )
            response.raise_for_status()
            assert "知识库" in response.json()["answer"]
            model_mode = "真实 Embedding/LLM" if live else "假 Embedding/LLM（未外发）"
            print(f"真实 MySQL、Qdrant、{model_mode}、重启后按库问答：通过")
    finally:
        try:
            if kb_id:
                with sessions() as session:
                    document_ids = list(session.scalars(select(Document.id).where(
                        Document.knowledge_base_id == kb_id
                    )))
                for document_id in document_ids:
                    service.delete_document(kb_id, document_id)
                with sessions.begin() as session:
                    kb = session.get(KnowledgeBase, kb_id)
                    if kb is not None:
                        session.delete(kb)
            if vectors.client.collection_exists(smoke_collection):
                vectors.client.delete_collection(smoke_collection)
        finally:
            vectors.close()
            engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="会向 .env 配置的模型端点发送演示文本与查询")
    asyncio.run(main(live=parser.parse_args().live))
