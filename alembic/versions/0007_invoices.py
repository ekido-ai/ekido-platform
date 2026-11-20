"""Invoice numbers (ticket C8): the number and date given to an order's invoice, and a counter per financial year.

Revision ID: 0007
Revises: 0006
"""
from alembic import op
import sqlalchemy as sa

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('invoice_records',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('order_id', sa.String(36), nullable=False, unique=True),
        sa.Column('org_id', sa.String(32), nullable=False),
        sa.Column('number', sa.String(32), nullable=False, unique=True),
        sa.Column('issued_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('buyer_name', sa.String(80), nullable=False),
        sa.Column('buyer_address', sa.JSON(), nullable=False))
    op.create_index('ix_invoice_records_org_id', 'invoice_records', ['org_id'])
    op.create_table('invoice_counters',
        sa.Column('financial_year', sa.String(8), primary_key=True),
        sa.Column('last', sa.Integer(), nullable=False))


def downgrade() -> None:
    op.drop_table('invoice_counters')
    op.drop_index('ix_invoice_records_org_id', table_name='invoice_records')
    op.drop_table('invoice_records')
