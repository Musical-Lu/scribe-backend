# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add the Sunet Drive integration.

Per customer: whether Drive is offered, the organisation's instance and
what it calls it. Per user: their own choice of instance. And a table of
short-lived Drive connections (Login Flow v2), which hold an encrypted app
password for at most an idle hour.

Revision ID: b3d5f7a9c1e2
Revises: a1c4e7f2b9d3
Create Date: 2026-09-27 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "b3d5f7a9c1e2"
down_revision: Union[str, Sequence[str], None] = "a1c4e7f2b9d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CUSTOMER_COLUMNS = [
    sa.Column(
        "drive_enabled", sa.Boolean(), nullable=False, server_default=sa.false()
    ),
    sa.Column("drive_url", sa.String(), nullable=True),
    sa.Column("drive_display_name", sa.String(), nullable=True),
]


def upgrade() -> None:
    """Upgrade schema."""

    inspector = inspect(op.get_bind())
    tables = inspector.get_table_names()

    # A table that does not exist yet is created whole, new columns
    # included, by SQLModel.metadata.create_all at startup.
    if "customer" in tables:
        customer_columns = {c["name"] for c in inspector.get_columns("customer")}
        for column in CUSTOMER_COLUMNS:
            if column.name not in customer_columns:
                op.add_column("customer", column.copy())

    if "users" in tables:
        user_columns = {c["name"] for c in inspector.get_columns("users")}
        if "drive_url" not in user_columns:
            op.add_column("users", sa.Column("drive_url", sa.String(), nullable=True))

    if "drive_connection" not in tables:
        op.create_table(
            "drive_connection",
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("instance", sa.String(), nullable=False),
            sa.Column("poll_endpoint", sa.String(), nullable=True),
            sa.Column("poll_token", sa.String(), nullable=True),
            sa.Column("login_name", sa.String(), nullable=True),
            sa.Column("app_password", sa.String(), nullable=True),
            sa.Column("dav_user", sa.String(), nullable=True),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("user_id"),
        )
        op.create_index(
            "ix_drive_connection_expires_at",
            "drive_connection",
            ["expires_at"],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""

    inspector = inspect(op.get_bind())
    tables = inspector.get_table_names()

    if "drive_connection" in tables:
        op.drop_index("ix_drive_connection_expires_at", table_name="drive_connection")
        op.drop_table("drive_connection")

    if "users" in tables and "drive_url" in {
        c["name"] for c in inspector.get_columns("users")
    }:
        op.drop_column("users", "drive_url")

    if "customer" in tables:
        customer_columns = {c["name"] for c in inspector.get_columns("customer")}
        for column in CUSTOMER_COLUMNS:
            if column.name in customer_columns:
                op.drop_column("customer", column.name)
