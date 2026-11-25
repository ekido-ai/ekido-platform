"""When a team member joined, so an invite shows as pending until the person signs in (ekido-ai/ekido-core#104).

Existing rows are backfilled as joined when the member is an owner or has signed in at least once (their email is
verified): before this column, adding someone made them a member at once, so those people are in the team. Anyone
added who has never signed in stays pending.

Revision ID: 0012
Revises: 0011
"""
from alembic import op
import sqlalchemy as sa

revision = '0012'
down_revision = '0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('org_members') as batch:
        batch.add_column(sa.Column('joined_at', sa.DateTime(timezone=True), nullable=True))
    op.execute(
        "UPDATE org_members SET joined_at = created_at "
        "WHERE role = 'owner' OR user_id IN (SELECT user_id FROM user_emails WHERE verified_at IS NOT NULL)"
    )


def downgrade() -> None:
    with op.batch_alter_table('org_members') as batch:
        batch.drop_column('joined_at')
