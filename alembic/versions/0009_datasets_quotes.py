"""Datasets (cleaned rows as the exact bytes trained on) and quotes (ticket C9).

Revision ID: 0009
Revises: 0008
"""
from alembic import op
import sqlalchemy as sa

revision = '0009'
down_revision = '0008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'datasets',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('org_id', sa.String(36), sa.ForeignKey('orgs.id'), nullable=False),
        sa.Column('recipe_id', sa.String(64), nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False),
        sa.Column('bytes', sa.Integer, nullable=False),
        sa.Column('train_rows', sa.Integer, nullable=False),
        sa.Column('test_rows', sa.Integer, nullable=False),
        sa.Column('redactions', sa.JSON),
        sa.Column('content', sa.LargeBinary, nullable=False),
    )
    op.create_index('ix_datasets_org_id', 'datasets', ['org_id'])
    op.create_index('ix_datasets_sha256', 'datasets', ['sha256'])
    op.create_table(
        'quotes',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('org_id', sa.String(36), sa.ForeignKey('orgs.id'), nullable=False),
        sa.Column('dataset_id', sa.String(36), sa.ForeignKey('datasets.id'), nullable=False),
        sa.Column('terms_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('quote', sa.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True)),
    )
    op.create_index('ix_quotes_org_id', 'quotes', ['org_id'])


def downgrade() -> None:
    op.drop_index('ix_quotes_org_id', 'quotes')
    op.drop_table('quotes')
    op.drop_index('ix_datasets_sha256', 'datasets')
    op.drop_index('ix_datasets_org_id', 'datasets')
    op.drop_table('datasets')
