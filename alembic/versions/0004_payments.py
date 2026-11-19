"""Payment orders, handled webhook events and the ledger (ticket C7).

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('payment_orders',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('org_id', sa.String(36), nullable=False),
        sa.Column('quote_id', sa.String(64), nullable=False),
        sa.Column('terms_hash', sa.String(64), nullable=False),
        sa.Column('amount_paise', sa.BigInteger(), nullable=False),
        sa.Column('currency', sa.String(3), nullable=False),
        sa.Column('gateway_order_id', sa.String(64), nullable=False, unique=True),
        sa.Column('gateway_payment_id', sa.String(64), nullable=True, unique=True),
        sa.Column('status', sa.String(24), nullable=False),
        sa.Column('failed_attempts', sa.Integer(), nullable=False),
        sa.Column('quote', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False))
    op.create_index('ix_payment_orders_org_id', 'payment_orders', ['org_id'])
    op.create_index('ix_payment_orders_terms_hash', 'payment_orders', ['terms_hash'])
    op.create_table('webhook_events',
        sa.Column('event_id', sa.String(80), primary_key=True),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event', sa.String(60), nullable=False),
        sa.Column('outcome', sa.String(40), nullable=False))
    op.create_table('ledger_entries',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('order_id', sa.String(36), sa.ForeignKey('payment_orders.id'), nullable=False),
        sa.Column('kind', sa.String(24), nullable=False),
        sa.Column('account', sa.String(80), nullable=False),
        sa.Column('amount_paise', sa.BigInteger(), nullable=False),
        sa.UniqueConstraint('order_id', 'kind', 'account'))
    op.create_index('ix_ledger_entries_order_id', 'ledger_entries', ['order_id'])


def downgrade() -> None:
    op.drop_index('ix_ledger_entries_order_id', table_name='ledger_entries')
    op.drop_table('ledger_entries')
    op.drop_table('webhook_events')
    op.drop_index('ix_payment_orders_terms_hash', table_name='payment_orders')
    op.drop_index('ix_payment_orders_org_id', table_name='payment_orders')
    op.drop_table('payment_orders')
