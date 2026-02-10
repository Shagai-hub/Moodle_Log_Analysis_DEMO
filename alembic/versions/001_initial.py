"""initial schema

Revision ID: 001_initial
Revises: None
Create Date: 2025-01-05
"""

from alembic import op
import sqlalchemy as sa

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("column_list", sa.JSON(), nullable=False),
    )
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_id", sa.String(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
    )
    op.create_table(
        "raw_posts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dataset_id", sa.String(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=True),
        sa.Column("discussion", sa.Integer(), nullable=True),
        sa.Column("parent", sa.Integer(), nullable=True),
        sa.Column("userid", sa.Integer(), nullable=True),
        sa.Column("userfullname", sa.String(), nullable=True),
        sa.Column("created", sa.DateTime(), nullable=True),
        sa.Column("modified", sa.DateTime(), nullable=True),
        sa.Column("subject", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("wordcount", sa.Integer(), nullable=True),
        sa.Column("charcount", sa.Integer(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
    )
    op.create_table(
        "run_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "metric_observations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("userid", sa.String(), nullable=True),
        sa.Column("userfullname", sa.String(), nullable=True),
        sa.Column("metric_name", sa.String(), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=True),
    )
    op.create_table(
        "ranking_observations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("userid", sa.String(), nullable=True),
        sa.Column("userfullname", sa.String(), nullable=True),
        sa.Column("rank_name", sa.String(), nullable=False),
        sa.Column("rank_value", sa.Float(), nullable=True),
    )


def downgrade():
    op.drop_table("ranking_observations")
    op.drop_table("metric_observations")
    op.drop_table("run_artifacts")
    op.drop_table("raw_posts")
    op.drop_table("analysis_runs")
    op.drop_table("datasets")
