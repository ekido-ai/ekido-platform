"""Link a paid order to the Coordinator job started from it (internal API).

Revision ID: 0008
Revises: 0007
"""
from alembic import op
import sqlalchemy as sa

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('payment_orders') as batch:
        batch.add_column(sa.Column('job_id', sa.String(36), nullable=True))
        batch.create_unique_constraint('uq_payment_orders_job_id', ['job_id'])


def downgrade() -> None:
    with op.batch_alter_table('payment_orders') as batch:
        batch.drop_constraint('uq_payment_orders_job_id', type_='unique')
        batch.drop_column('job_id')
