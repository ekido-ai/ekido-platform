"""The assistant: which teams may use it, and what it was asked that Ekido cannot do yet (decisions A3 and U4, 24 Nov).

Revision ID: 0010
Revises: 0009
"""
from alembic import op
import sqlalchemy as sa

revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'assistant_teams',
        sa.Column('org_id', sa.String(32), sa.ForeignKey('orgs.id'), primary_key=True),
        sa.Column('enabled', sa.Boolean, nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        'assistant_unsupported',
        sa.Column('day', sa.Date, primary_key=True),
        sa.Column('kind', sa.String(32), primary_key=True),
        sa.Column('count', sa.Integer, nullable=False),
    )


def downgrade() -> None:
    op.drop_table('assistant_unsupported')
    op.drop_table('assistant_teams')
