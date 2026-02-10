"""add run events table

Revision ID: 002_run_events
Revises: 001_initial
Create Date: 2026-02-10
"""

from alembic import op
import sqlalchemy as sa

revision = "002_run_events"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "run_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_name", sa.String(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade():
    op.drop_table("run_events")
