"""区分无知识库的联网会话与迁移前的全库会话。"""

from alembic import op
import sqlalchemy as sa


revision = "0004_web_only_sessions"
down_revision = "0003_chat_knowledge_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch:
        batch.add_column(sa.Column(
            "session_type", sa.String(16), nullable=False, server_default="knowledge"
        ))


def downgrade() -> None:
    with op.batch_alter_table("chat_sessions") as batch:
        batch.drop_column("session_type")
