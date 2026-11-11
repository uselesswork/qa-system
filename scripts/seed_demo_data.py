"""将六份仓库示例资料写入关系库，供阶段 10 开发和测试使用。

这一步不写 Qdrant；文档保持 pending，不能视为已经可检索。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import Engine

from src.db import Chunk, Document, KnowledgeBase
from src.db.session import create_database_engine, create_session_factory
from src.loaders.md_loader import MarkdownLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_KB_ID = str(uuid5(NAMESPACE_URL, "qa-system:demo-knowledge-base:v1"))
DEMO_KB_NAME = "RAG 演示资料（未建向量索引）"


@dataclass(frozen=True)
class DemoGroup:
    id: str
    name: str
    description: str
    filenames: tuple[str, ...]


DEMO_GROUPS = (
    # 保持第一组的 ID 和文件归属不变，避免重建已有数据。
    DemoGroup(
        DEMO_KB_ID,
        DEMO_KB_NAME,
        "收录两份 RAG 问答示例资料，说明初步检索后处理候选片段、优先使用相关信息与回答质量之间的关系；尚未建立向量索引。",
        ("doc1.md", "doc2.md"),
    ),
    DemoGroup(
        str(uuid5(NAMESPACE_URL, "qa-system:demo-knowledge-base:v1:candidates")),
        "候选片段差异（未建向量索引）",
        "收录候选片段相关性差异的示例，展示无关片段进入上下文后可能干扰模型理解；尚未建立向量索引。",
        ("doc3.md",),
    ),
    DemoGroup(
        str(uuid5(NAMESPACE_URL, "qa-system:demo-knowledge-base:v1:noise")),
        "关键词噪声案例（未建向量索引）",
        "收录重复 RAG、Embedding 等关键词却未解释重排真实目的的干扰资料，用于验证表面相关性与证据质量的区别；尚未建立向量索引。",
        ("doc4.md",),
    ),
    DemoGroup(
        str(uuid5(NAMESPACE_URL, "qa-system:demo-knowledge-base:v1:truth")),
        "重排目标案例（未建向量索引）",
        "收录重排目标的简短说明：在已有候选结果中优先选出真正相关的片段，为生成模型提供高质量上下文；尚未建立向量索引。",
        ("doc5.md",),
    ),
    DemoGroup(
        str(uuid5(NAMESPACE_URL, "qa-system:demo-knowledge-base:v1:failure")),
        "失败机制案例（未建向量索引）",
        "收录仅依靠 Embedding 相似度重排可能被表面语义误导的案例，以及基于答案核验候选证据的思路；尚未建立向量索引。",
        ("doc6.md",),
    ),
)


@dataclass(frozen=True)
class SeedResult:
    knowledge_bases_created: int = 0
    descriptions_filled: int = 0
    documents_created: int = 0
    chunks_created: int = 0


def seed_demo_data(
    engine: Engine, source_dir: Path | None = None
) -> SeedResult:
    source_dir = source_dir or PROJECT_ROOT / "data" / "model12"
    loader = MarkdownLoader()
    cleaner = TextCleaner()
    splitter = SemanticSplitter()
    factory = create_session_factory(engine)
    kb_created = descriptions_filled = docs_created = chunks_created = 0

    # 先读完输入，缺文件或解码失败时不会留下半套数据。
    inputs = []
    for group in DEMO_GROUPS:
        for filename in group.filenames:
            path = source_dir / filename
            raw = path.read_bytes()
            document = loader.load(path)[0]
            document.content = cleaner.clean(document.content, preserve_newlines=True)
            inputs.append((group, filename, raw, splitter.split(document)))

    with factory.begin() as session:
        for group in DEMO_GROUPS:
            kb = session.get(KnowledgeBase, group.id)
            if kb is None:
                session.add(KnowledgeBase(
                    id=group.id, name=group.name, description=group.description
                ))
                kb_created += 1
            elif kb.name != group.name:
                raise RuntimeError("演示知识库 ID 已被其他数据占用，已停止写入。")
            elif not kb.description or not kb.description.strip():
                kb.description = group.description
                descriptions_filled += 1

        for group, filename, raw, text_chunks in inputs:
            document_id = str(uuid5(NAMESPACE_URL, f"{group.id}:{filename}"))
            digest = sha256(raw).hexdigest()
            saved = session.get(Document, document_id)
            if saved is None:
                session.add(Document(
                    id=document_id,
                    knowledge_base_id=group.id,
                    original_name=filename,
                    file_type="md",
                    storage_key=f"data/model12/{filename}",
                    sha256=digest,
                    size_bytes=len(raw),
                    status="pending",
                ))
                docs_created += 1
            elif saved.knowledge_base_id != group.id or saved.sha256 != digest:
                raise RuntimeError(f"演示文档 {filename} 与现有记录冲突，已停止写入。")

            for ordinal, text_chunk in enumerate(text_chunks):
                chunk_id = str(uuid5(NAMESPACE_URL, f"{document_id}:{ordinal}"))
                existing = session.get(Chunk, chunk_id)
                if existing is not None:
                    if existing.document_id != document_id or existing.content != text_chunk.content:
                        raise RuntimeError(f"演示片段 {filename}#{ordinal} 与现有记录冲突。")
                    continue
                session.add(Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    ordinal=ordinal,
                    content=text_chunk.content,
                    heading=text_chunk.metadata.get("heading"),
                    page_number=text_chunk.metadata.get("page"),
                    char_start=text_chunk.metadata.get("char_start"),
                    char_end=text_chunk.metadata.get("char_end"),
                ))
                chunks_created += 1

    return SeedResult(kb_created, descriptions_filled, docs_created, chunks_created)


def main() -> None:
    engine = create_database_engine()
    try:
        result = seed_demo_data(engine)
        print(
            "演示数据完成："
            f"新增知识库 {result.knowledge_bases_created}、"
            f"补全描述 {result.descriptions_filled}、"
            f"文档 {result.documents_created}、"
            f"片段 {result.chunks_created}。"
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
