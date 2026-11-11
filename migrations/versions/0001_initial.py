"""创建知识库、文档和片段元数据表。"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("knowledge_base_id", sa.String(36), nullable=False),
        sa.Column("original_name", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(16), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(16), server_default="pending", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("knowledge_base_id", "sha256", name="uq_documents_kb_sha256"),
        sa.CheckConstraint("size_bytes >= 0", name="ck_documents_size_nonnegative"),
        sa.CheckConstraint("status IN ('pending', 'indexing', 'ready', 'failed')", name="ck_documents_status"),
    )
    op.create_index("ix_documents_kb_status", "documents", ["knowledge_base_id", "status"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("heading", sa.String(255), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunks_ordinal_nonnegative"),
    )


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_index("ix_documents_kb_status", table_name="documents")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
