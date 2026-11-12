"""账号、知识库归属、登录会话和持久化聊天。"""

from alembic import op
import sqlalchemy as sa


revision = "0002_auth_and_chat"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.add_column(sa.Column("owner_id", sa.String(36), nullable=True))
        batch.create_foreign_key("fk_knowledge_bases_owner_id", "users", ["owner_id"], ["id"])
        batch.create_index("ix_knowledge_bases_owner_id", ["owner_id"])
    op.create_table(
        "auth_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("csrf_token", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_auth_sessions_user_id", "auth_sessions", ["user_id"])
    op.create_index("ix_auth_sessions_expires_at", "auth_sessions", ["expires_at"])
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.UniqueConstraint("session_id", "ordinal", name="uq_chat_messages_session_ordinal"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_messages_role"),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("auth_sessions")
    with op.batch_alter_table("knowledge_bases") as batch:
        batch.drop_index("ix_knowledge_bases_owner_id")
        batch.drop_constraint("fk_knowledge_bases_owner_id", type_="foreignkey")
        batch.drop_column("owner_id")
    op.drop_table("users")
