"""文件落盘、关系记录和向量索引的最小同步编排。"""

from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.db import Chunk, Document, KnowledgeBase
from src.embeddings.base import BaseEmbedding
from src.loaders.md_loader import MarkdownLoader
from src.loaders.pdf_loader import PdfLoader
from src.loaders.txt_loader import TxtLoader
from src.processors.cleaner import TextCleaner
from src.processors.semantic.semantic_splitter import SemanticSplitter
from src.vectorstores.qdrant_store import QdrantVectorStore, VectorDimensionError


MAX_FILE_BYTES = 10 * 1024 * 1024
LOADERS = {".md": MarkdownLoader, ".txt": TxtLoader, ".pdf": PdfLoader}


class KnowledgeError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class KnowledgeService:
    def __init__(self, sessions: sessionmaker, vectors: QdrantVectorStore,
                 embedder: BaseEmbedding, upload_dir: Path):
        self.sessions = sessions
        self.vectors = vectors
        self.embedder = embedder
        self.upload_dir = upload_dir

    @staticmethod
    def kb_json(kb: KnowledgeBase) -> dict:
        return {"id": kb.id, "name": kb.name, "description": kb.description}

    @staticmethod
    def doc_json(doc: Document) -> dict:
        return {"id": doc.id, "knowledge_base_id": doc.knowledge_base_id,
                "original_name": doc.original_name, "status": doc.status,
                "error_message": doc.error_message, "size_bytes": doc.size_bytes}

    def create_kb(self, name: str, description: str | None) -> dict:
        name = name.strip()
        if not name or len(name) > 128:
            raise KnowledgeError("知识库名称不能为空且不能超过 128 个字符。")
        with self.sessions.begin() as session:
            kb = KnowledgeBase(name=name, description=description)
            session.add(kb)
            session.flush()
            return self.kb_json(kb)

    def list_kbs(self) -> list[dict]:
        with self.sessions() as session:
            return [self.kb_json(kb) for kb in session.scalars(
                select(KnowledgeBase).order_by(KnowledgeBase.created_at, KnowledgeBase.id)
            )]

    def get_kb(self, kb_id: str) -> dict:
        with self.sessions() as session:
            kb = session.get(KnowledgeBase, kb_id)
            if kb is None:
                raise KnowledgeError("知识库不存在。", 404)
            return self.kb_json(kb)

    def list_documents(self, kb_id: str) -> list[dict]:
        self.get_kb(kb_id)
        with self.sessions() as session:
            return [self.doc_json(doc) for doc in session.scalars(
                select(Document).where(Document.knowledge_base_id == kb_id)
                .order_by(Document.created_at, Document.id)
            )]

    def ready_document_ids(self, kb_id: str) -> list[str]:
        self.get_kb(kb_id)
        with self.sessions() as session:
            return list(session.scalars(select(Document.id).where(
                Document.knowledge_base_id == kb_id, Document.status == "ready"
            )))

    async def upload(self, kb_id: str, filename: str, content: bytes) -> dict:
        self.get_kb(kb_id)
        safe_name = Path(filename.replace("\\", "/")).name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or len(safe_name) > 255 or suffix not in LOADERS:
            raise KnowledgeError("仅支持 .md、.txt、.pdf 文件，文件名不能超过 255 字符。")
        if not content or len(content) > MAX_FILE_BYTES:
            raise KnowledgeError("文件不能为空或超过 10 MiB。", 413)
        digest = sha256(content).hexdigest()
        with self.sessions() as session:
            existing = session.scalar(select(Document).where(
                Document.knowledge_base_id == kb_id, Document.sha256 == digest
            ))
            if existing:
                raise KnowledgeError(f"相同内容已存在，文档 ID：{existing.id}", 409)

        document_id = str(uuid4())
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / f"{document_id}{suffix}"
        path.write_bytes(content)
        try:
            try:
                loaded = LOADERS[suffix]().load(path)
            except Exception as exc:
                raise KnowledgeError("文件无法解析，请确认编码或 PDF 内容有效。", 422) from exc
            cleaner = TextCleaner()
            splitter = SemanticSplitter()
            chunks = []
            for item in loaded:
                item.content = cleaner.clean(item.content, preserve_newlines=True)
                item.metadata["source"] = safe_name
                chunks.extend(splitter.split(item))
            if not chunks:
                raise KnowledgeError("文件没有可提取的文字；扫描 PDF 暂不支持。", 422)
            with self.sessions.begin() as session:
                session.add(Document(
                    id=document_id, knowledge_base_id=kb_id,
                    original_name=safe_name, file_type=suffix[1:],
                    storage_key=path.name, sha256=digest,
                    size_bytes=len(content), status="pending",
                ))
        except IntegrityError as exc:
            path.unlink(missing_ok=True)
            raise KnowledgeError("相同内容已存在于该知识库。", 409) from exc
        except Exception:
            path.unlink(missing_ok=True)
            raise

        try:
            await self._index(document_id, kb_id, chunks)
        except Exception as exc:
            detail = str(exc) if isinstance(exc, VectorDimensionError) else type(exc).__name__
            with self.sessions.begin() as session:
                doc = session.get(Document, document_id)
                doc.status = "failed"
                doc.error_message = detail
            if isinstance(exc, VectorDimensionError):
                raise KnowledgeError(f"索引失败：{detail}文档已保留为 failed，可重试。", 503) from exc
            raise KnowledgeError("索引失败，文档已保留为 failed，可重试。", 503) from exc
        with self.sessions() as session:
            return self.doc_json(session.get(Document, document_id))

    async def _index(self, document_id: str, kb_id: str, chunks: list) -> None:
        for ordinal, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": str(uuid5(NAMESPACE_URL, f"{document_id}:{ordinal}")),
                "document_id": document_id,
                "knowledge_base_id": kb_id,
            })
        embedded = await self.embedder.embed(chunks)
        if len(embedded) != len(chunks):
            raise ValueError("Embedding 返回的片段数量不匹配。")
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        try:
            self.vectors.add(embedded)
            with self.sessions.begin() as session:
                doc = session.get(Document, document_id)
                for ordinal, chunk in enumerate(chunks):
                    session.add(Chunk(
                        id=ids[ordinal], document_id=document_id, ordinal=ordinal,
                        content=chunk.content, heading=chunk.metadata.get("heading"),
                        page_number=chunk.metadata.get("page"),
                        char_start=chunk.metadata.get("char_start"),
                        char_end=chunk.metadata.get("char_end"),
                    ))
                doc.status = "ready"
                doc.error_message = None
        except Exception:
            try:
                self.vectors.delete_points(ids)
            except Exception:
                pass  # ready 过滤仍会阻止孤儿向量进入检索结果
            raise

    def delete_document(self, kb_id: str, document_id: str) -> None:
        with self.sessions.begin() as session:
            doc = session.get(Document, document_id)
            if doc is None or doc.knowledge_base_id != kb_id:
                raise KnowledgeError("文档不存在。", 404)
            doc.status = "pending"  # 删除跨库资源前先撤销可检索状态
            ids = list(session.scalars(select(Chunk.id).where(Chunk.document_id == document_id)))
            storage_key = doc.storage_key
        try:
            self.vectors.delete_points(ids)
        except Exception as exc:
            raise KnowledgeError("删除向量失败；文档保留为 pending，可重试。", 503) from exc
        with self.sessions.begin() as session:
            session.delete(session.get(Document, document_id))
        if storage_key:
            (self.upload_dir / Path(storage_key).name).unlink(missing_ok=True)

    async def reindex_document(self, kb_id: str, document_id: str) -> dict:
        with self.sessions.begin() as session:
            doc = session.get(Document, document_id)
            if doc is None or doc.knowledge_base_id != kb_id:
                raise KnowledgeError("文档不存在。", 404)
            if not doc.storage_key:
                raise KnowledgeError("该文档没有可重建的已保存文件。", 409)
            path = self.upload_dir / Path(doc.storage_key).name
            if not path.is_file():
                raise KnowledgeError("原文件不存在，无法重建。", 409)
            doc.status = "pending"
            ids = list(session.scalars(select(Chunk.id).where(Chunk.document_id == document_id)))
            suffix = path.suffix.lower()
            safe_name = doc.original_name
        try:
            self.vectors.delete_points(ids)
        except Exception as exc:
            raise KnowledgeError("删除旧向量失败，文档保留为 pending，可重试。", 503) from exc
        with self.sessions.begin() as session:
            session.query(Chunk).filter(Chunk.document_id == document_id).delete()
        cleaner = TextCleaner()
        splitter = SemanticSplitter()
        chunks = []
        try:
            for item in LOADERS[suffix]().load(path):
                item.content = cleaner.clean(item.content, preserve_newlines=True)
                item.metadata["source"] = safe_name
                chunks.extend(splitter.split(item))
            if not chunks:
                raise ValueError("原文件没有可提取的文字。")
            await self._index(document_id, kb_id, chunks)
        except Exception as exc:
            with self.sessions.begin() as session:
                doc = session.get(Document, document_id)
                doc.status = "failed"
                doc.error_message = type(exc).__name__
            raise KnowledgeError("重建索引失败，文档保留为 failed。", 503) from exc
        with self.sessions() as session:
            return self.doc_json(session.get(Document, document_id))
