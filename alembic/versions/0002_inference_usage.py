"""Per-key inference reservations and usage.

Revision ID: 0002
Revises: 0001
"""
from alembic import op
import sqlalchemy as sa

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('inference_usage',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('key_id', sa.String(32), sa.ForeignKey('api_keys.id'), nullable=False),
        sa.Column('model_id', sa.String(120), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True))
    op.create_index('ix_inference_usage_key_id', 'inference_usage', ['key_id'])
    op.create_index('ix_inference_usage_created_at', 'inference_usage', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_inference_usage_created_at', table_name='inference_usage')
    op.drop_index('ix_inference_usage_key_id', table_name='inference_usage')
    op.drop_table('inference_usage')
