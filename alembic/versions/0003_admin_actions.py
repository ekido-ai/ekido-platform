"""Audit log for the owner's admin area (ticket C26).

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('admin_actions',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('email', sa.String(320), nullable=False),
        sa.Column('action', sa.String(40), nullable=False),
        sa.Column('target', sa.String(120), nullable=False),
        sa.Column('outcome', sa.String(16), nullable=False),
        sa.Column('detail', sa.JSON(), nullable=False))
    op.create_index('ix_admin_actions_at', 'admin_actions', ['at'])


def downgrade() -> None:
    op.drop_index('ix_admin_actions_at', table_name='admin_actions')
    op.drop_table('admin_actions')
