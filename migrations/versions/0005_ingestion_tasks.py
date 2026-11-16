"""Persist asynchronous document indexing tasks."""

from alembic import op
import sqlalchemy as sa


revision = "0005_ingestion_tasks"
down_revision = "0004_web_only_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.current_timestamp()),
    )
    op.create_index("ix_ingestion_tasks_document_id", "ingestion_tasks", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_tasks_document_id", table_name="ingestion_tasks")
    op.drop_table("ingestion_tasks")
