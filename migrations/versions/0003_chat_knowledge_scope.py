"""将新对话绑定到知识库，旧会话保持原有全库范围。"""

from alembic import op
import sqlalchemy as sa


revision = "0003_chat_knowledge_scope"
down_revision = "0002_auth_and_chat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch:
        batch.add_column(sa.Column("knowledge_base_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_chat_sessions_knowledge_base_id", "knowledge_bases",
            ["knowledge_base_id"], ["id"],
        )
        batch.create_index("ix_chat_sessions_knowledge_base_id", ["knowledge_base_id"])


def downgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch:
        batch.drop_index("ix_chat_sessions_knowledge_base_id")
        batch.drop_constraint("fk_chat_sessions_knowledge_base_id", type_="foreignkey")
        batch.drop_column("knowledge_base_id")
