from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.exc import IntegrityError

from src.db import Base, Chunk, Document, KnowledgeBase
from src.db import session as db_session
from scripts.seed_demo_data import DEMO_GROUPS, DEMO_KB_ID, seed_demo_data


def test_database_url_requires_local_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(db_session, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        db_session.get_database_url()


def test_models_enforce_document_identity_and_cascade():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = db_session.create_session_factory(engine)

    with factory.begin() as session:
        kb = KnowledgeBase(name="项目资料")
        doc = Document(
            knowledge_base=kb,
            original_name="readme.md",
            file_type="md",
            sha256="a" * 64,
            size_bytes=20,
        )
        chunk = Chunk(document=doc, ordinal=0, content="第一段", heading="介绍")
        session.add(chunk)

    assert len(kb.id) == len(doc.id) == len(chunk.id) == 36

    with factory() as session:
        assert session.scalar(select(Document).where(Document.knowledge_base_id == kb.id))
        session.add(Document(
            knowledge_base_id=kb.id, original_name="copy.md", file_type="md",
            sha256="a" * 64, size_bytes=20,
        ))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    with factory.begin() as session:
        session.delete(session.get(KnowledgeBase, kb.id))

    with factory() as session:
        assert session.scalars(select(Document)).all() == []
        assert session.scalars(select(Chunk)).all() == []

    engine.dispose()


def test_initial_migration_upgrades_and_downgrades_temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "migration_test.sqlite3"
    url = f"sqlite+pysqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(config, "head")
    command.check(config)

    engine = create_engine(url)
    inspector = inspect(engine)
    assert {"knowledge_bases", "documents", "chunks", "alembic_version"} <= set(
        inspector.get_table_names()
    )
    assert {"knowledge_base_id", "sha256", "status"} <= {
        column["name"] for column in inspector.get_columns("documents")
    }
    engine.dispose()

    command.downgrade(config, "base")
    engine = create_engine(url)
    assert not {"knowledge_bases", "documents", "chunks"} & set(
        inspect(engine).get_table_names()
    )
    engine.dispose()


def test_demo_seed_is_idempotent_and_does_not_overwrite_changed_input(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    for number in range(1, 7):
        (sample_dir / f"doc{number}.md").write_text(
            f"# 第 {number} 份\n正文 {number}", encoding="utf-8"
        )

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    first = seed_demo_data(engine, sample_dir)
    second = seed_demo_data(engine, sample_dir)

    assert (first.knowledge_bases_created, first.documents_created, first.chunks_created) == (5, 6, 6)
    assert (second.knowledge_bases_created, second.documents_created, second.chunks_created) == (0, 0, 0)
    assert first.descriptions_filled == second.descriptions_filled == 0
    with db_session.create_session_factory(engine)() as session:
        knowledge_bases = session.scalars(select(KnowledgeBase)).all()
        documents = session.scalars(select(Document)).all()
        chunks = session.scalars(select(Chunk)).all()
        assert len(knowledge_bases) == 5
        assert len(documents) == 6
        assert len(chunks) == 6
        assert DEMO_KB_ID in {kb.id for kb in knowledge_bases}
        assert {kb.description for kb in knowledge_bases} == {
            group.description for group in DEMO_GROUPS
        }
        assert {doc.knowledge_base_id for doc in documents} == {
            kb.id for kb in knowledge_bases
        }
        assert {doc.status for doc in session.scalars(select(Document))} == {"pending"}

    with db_session.create_session_factory(engine).begin() as session:
        session.get(KnowledgeBase, DEMO_KB_ID).description = None
        session.get(KnowledgeBase, DEMO_GROUPS[1].id).description = "用户自定义描述"

    backfill = seed_demo_data(engine, sample_dir)
    assert (backfill.knowledge_bases_created, backfill.descriptions_filled,
            backfill.documents_created, backfill.chunks_created) == (0, 1, 0, 0)
    with db_session.create_session_factory(engine)() as session:
        assert session.get(KnowledgeBase, DEMO_KB_ID).description == DEMO_GROUPS[0].description
        assert session.get(KnowledgeBase, DEMO_GROUPS[1].id).description == "用户自定义描述"

    (sample_dir / "doc1.md").write_text("# 第一份\n已更改的正文", encoding="utf-8")
    with pytest.raises(RuntimeError, match="冲突"):
        seed_demo_data(engine, sample_dir)
    with db_session.create_session_factory(engine)() as session:
        assert len(session.scalars(select(Document)).all()) == 6

    engine.dispose()
