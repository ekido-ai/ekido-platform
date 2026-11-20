"""Deletion requests and erasure tasks (ticket C17).

Revision ID: 0006
Revises: 0005
"""
from alembic import op
import sqlalchemy as sa

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('deletion_requests',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('due_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(12), nullable=False),
        sa.Column('summary', sa.JSON(), nullable=False))
    op.create_index('ix_deletion_requests_user_id', 'deletion_requests', ['user_id'])
    op.create_table('erasure_tasks',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deletion_id', sa.String(36), sa.ForeignKey('deletion_requests.id'), nullable=False),
        sa.Column('kind', sa.String(24), nullable=False),
        sa.Column('ref', sa.String(64), nullable=False),
        sa.Column('status', sa.String(12), nullable=False),
        sa.Column('done_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False))
    op.create_index('ix_erasure_tasks_deletion_id', 'erasure_tasks', ['deletion_id'])


def downgrade() -> None:
    op.drop_index('ix_erasure_tasks_deletion_id', table_name='erasure_tasks')
    op.drop_table('erasure_tasks')
    op.drop_index('ix_deletion_requests_user_id', table_name='deletion_requests')
    op.drop_table('deletion_requests')
