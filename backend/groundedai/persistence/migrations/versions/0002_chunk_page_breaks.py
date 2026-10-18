"""Chunks record where each later page starts, so evidence can name its exact page.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("page_breaks", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Plain SQL: SQLite (3.35+) and PostgreSQL both support DROP COLUMN.
    op.execute("ALTER TABLE chunks DROP COLUMN page_breaks")
