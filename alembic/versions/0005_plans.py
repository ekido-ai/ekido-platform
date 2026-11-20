"""Team plans and plan usage (ticket C24).

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('team_plans',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('org_id', sa.String(36), nullable=False, unique=True),
        sa.Column('plan', sa.String(16), nullable=False),
        sa.Column('status', sa.String(16), nullable=False),
        sa.Column('subscription_id', sa.String(64), nullable=True, unique=True),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cancel_at_period_end', sa.Integer(), nullable=False))
    op.create_table('plan_usage',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('org_id', sa.String(36), nullable=False),
        sa.Column('kind', sa.String(12), nullable=False),
        sa.Column('period', sa.String(24), nullable=False),
        sa.Column('used', sa.BigInteger(), nullable=False),
        sa.UniqueConstraint('org_id', 'kind', 'period'))
    op.create_index('ix_plan_usage_org_id', 'plan_usage', ['org_id'])


def downgrade() -> None:
    op.drop_index('ix_plan_usage_org_id', table_name='plan_usage')
    op.drop_table('plan_usage')
    op.drop_table('team_plans')
