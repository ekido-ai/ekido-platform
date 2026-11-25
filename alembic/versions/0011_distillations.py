"""Distillations: generated examples under review before training (ticket 154, the Platform half).

Revision ID: 0011
Revises: 0010
"""
from alembic import op
import sqlalchemy as sa

revision = '0011'
down_revision = '0010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'distillations',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('org_id', sa.String(36), nullable=False),
        sa.Column('order_id', sa.String(36), nullable=False, unique=True),
        sa.Column('dataset_id', sa.String(36), nullable=False),
        sa.Column('state', sa.String(16), nullable=False),
        sa.Column('generation_job_id', sa.String(36), nullable=True),
        sa.Column('generating_since', sa.DateTime(timezone=True), nullable=True),
        sa.Column('generated', sa.LargeBinary, nullable=True),
        sa.Column('generated_rows', sa.Integer, nullable=False, server_default='0'),
        sa.Column('generated_sha256', sa.String(64), nullable=True),
        sa.Column('sample', sa.JSON, nullable=False),
        sa.Column('removed', sa.JSON, nullable=False),
        sa.Column('regenerations', sa.Integer, nullable=False, server_default='0'),
        sa.Column('training_dataset_id', sa.String(36), nullable=True),
        sa.Column('training_job_id', sa.String(36), nullable=True),
        sa.Column('reason', sa.String(200), nullable=True),
    )
    op.create_index('ix_distillations_org_id', 'distillations', ['org_id'])
    op.create_index('ix_distillations_state', 'distillations', ['state'])


def downgrade() -> None:
    op.drop_index('ix_distillations_state', 'distillations')
    op.drop_index('ix_distillations_org_id', 'distillations')
    op.drop_table('distillations')
