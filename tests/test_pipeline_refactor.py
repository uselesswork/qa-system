import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app import rag_app
from app.main import serialize_evidence, serialize_metadata
from src.embeddings.embedded_chunk import EmbeddedChunk
from src.loaders.document import Document
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter
from src.rag.rag_core import RAGCore
from src.rerankers.embedding_reranker import EmbeddingReranker
from src.retrievers.simple_retriever import SimpleRetriever
from src.retrievers.base import BaseRetriever
from src.vectorstores.search_result import SearchResult
from src.utils.similarity import cosine_similarity


def test_markdown_headings_preserve_preamble_and_offsets():
    text = "简介\r\n内容\r\n# 第一节\r\n正文\r\n## 第二节\r\n更多内容"
    cleaned = TextCleaner().clean(text, preserve_newlines=True)
    chunks = SemanticSplitter().split(
        Document(cleaned, {"file_type": "md", "source": "example.md"})
    )

    assert [chunk.metadata["heading"] for chunk in chunks] == [
        None, "第一节", "第二节"
    ]
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [0, 1, 2]
    assert "简介\n内容" == chunks[0].content
    for chunk in chunks:
        start = chunk.metadata["char_start"]
        end = chunk.metadata["char_end"]
        assert cleaned[start:end] == chunk.content


def test_markdown_without_heading_falls_back():
    chunks = SemanticSplitter().split(Document("普通正文", {"file_type": "md"}))
    assert len(chunks) == 1
    assert "semantic" not in chunks[0].metadata


def test_retrieval_and_rerank_embed_query_once():
    class FakeEmbedder:
        def __init__(self):
            self.calls = []

        async def embed_texts(self, texts):
            self.calls.append(texts)
            return [SimpleNamespace(vector=np.array([1.0, 0.0]))]

    class FakeStore:
        def search(self, query_vector, top_k):
            assert np.array_equal(query_vector, [1.0, 0.0])
            return [SearchResult(SimpleNamespace(vector=np.array([1.0, 0.0])), 0.9)]

    class FakeLLM:
        async def generate(self, prompt):
            return SimpleNamespace(content="回答", metadata={})

    embedder = FakeEmbedder()
    store = FakeStore()
    core = RAGCore(
        embedder=embedder,
        vectorstore=store,
        retriever=SimpleRetriever(embedder, store),
        prompt_builder=SimpleNamespace(build=lambda query, chunks: "prompt"),
        llm=FakeLLM(),
        reranker=EmbeddingReranker(),
    )

    assert asyncio.run(core.answer("问题")).answer == "回答"
    assert embedder.calls == [["问题"]]


def test_legacy_retriever_still_works_with_reranker():
    class LegacyRetriever(BaseRetriever):
        async def retrieve(self, query, top_k=5):
            return [SearchResult(SimpleNamespace(vector=np.array([1.0, 0.0])), 0.9)]

    class FakeEmbedder:
        async def embed_texts(self, texts):
            return [SimpleNamespace(vector=np.array([1.0, 0.0]))]

    class FakeLLM:
        async def generate(self, prompt):
            return SimpleNamespace(content="回答", metadata={})

    core = RAGCore(
        embedder=FakeEmbedder(), vectorstore=None, retriever=LegacyRetriever(),
        prompt_builder=SimpleNamespace(build=lambda query, chunks: "prompt"),
        llm=FakeLLM(), reranker=EmbeddingReranker(),
    )
    assert asyncio.run(core.answer("问题")).answer == "回答"


def test_build_rag_core_indexes_without_temporary_store_fields(tmp_path, monkeypatch):
    folder = tmp_path / "data" / "model12"
    folder.mkdir(parents=True)
    (folder / "one.md").write_text("# 标题\n正文", encoding="utf-8")

    class FakeEmbedder:
        def __init__(self):
            self.indexed = []

        async def embed(self, chunks):
            self.indexed = chunks
            return [
                EmbeddedChunk(chunk.content, np.array([1.0, 0.0]), chunk.metadata)
                for chunk in chunks
            ]

    class FakeLLM:
        pass

    monkeypatch.setattr(rag_app, "BASE_DIR", tmp_path)
    monkeypatch.setattr(rag_app, "OpenAIEmbedding", FakeEmbedder)
    monkeypatch.setattr(rag_app, "OpenAIChatLLM", FakeLLM)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")

    core = asyncio.run(rag_app.build_rag_core())

    assert len(core.vectorstore._chunks) == 1
    assert core.vectorstore._chunks[0].metadata["heading"] == "标题"
    assert not hasattr(core.vectorstore, "_pending_chunks")
    assert not hasattr(core.vectorstore, "_embedder")


def test_build_rag_core_reports_missing_key(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_app, "BASE_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(rag_app.build_rag_core())


def test_api_serialization_handles_paths_consistently():
    evidence = [SimpleNamespace(
        rank=1, score=0.98765, source=Path("docs/example.md"),
        chunk_id=2, content="依据",
    )]
    assert serialize_evidence(evidence)[0]["score"] == 0.9877
    assert serialize_evidence(evidence)[0]["source"] == str(Path("docs/example.md"))
    assert serialize_metadata({"nested": {"path": Path("docs/example.md")}}) == {
        "nested": {"path": str(Path("docs/example.md"))}
    }


def test_cosine_similarity_zero_vector_is_finite():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
