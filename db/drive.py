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

"""
Users' connections to their Sunet Drive, and their choice of instance.

A connection is a DriveConnection row (see db/models.py): while the user is
signing in to Drive it holds the Login Flow v2 poll token, and once they
have granted access it holds the app password Drive issued. Both secrets
are encrypted here, under a key derived from API_SECRET_KEY, and decrypted
only on the way out to utils/drive.py. The table is in the database rather
than in memory because the backend runs several worker processes: the
request that starts a sign-in and the one that polls it seldom land in the
same one.

Rows expire DRIVE_SESSION_IDLE_SECONDS after last use. drive_take_expired()
removes them in one statement and hands them to the sweeper in app.py,
which revokes each app password on the Drive side as well.

Never log a poll token, an app password, or the contents of a row.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional

from sqlalchemy import delete, select

from db.models import DriveConnection, User
from db.session import get_async_session
from utils.crypto import decrypt_with_key, derive_key, encrypt_with_key
from utils.log import get_logger
from utils.settings import get_settings

log = get_logger()
settings = get_settings()

# Versioned: changing it strands every stored connection, which only means
# everybody connects to Drive again.
SEAL_INFO = b"sunet-scribe-drive-connection-v1"


def _key() -> bytes:
    return derive_key(settings.API_SECRET_KEY, SEAL_INFO)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _open(row: DriveConnection) -> dict:
    """
    A row as a dict, secrets decrypted. Only for handing to utils/drive.py.
    """

    key = _key()

    return {
        "user_id": row.user_id,
        "instance": row.instance,
        "poll_endpoint": row.poll_endpoint,
        "poll_token": decrypt_with_key(key, row.poll_token) if row.poll_token else None,
        "login_name": row.login_name,
        "app_password": (
            decrypt_with_key(key, row.app_password) if row.app_password else None
        ),
        "dav_user": row.dav_user,
        "connected": bool(row.app_password),
        "expires_at": row.expires_at,
    }


async def drive_get(user_id: str) -> Optional[dict]:
    """
    A user's connection, pending or connected, unless it has expired.

    Parameters:
        user_id (str): The Scribe user.

    Returns:
        Optional[dict]: See _open(), or None.
    """

    async with get_async_session() as session:
        result = await session.execute(
            select(DriveConnection).where(
                DriveConnection.user_id == user_id,
                DriveConnection.expires_at > _now(),
            )
        )
        if not (row := result.scalars().first()):
            return None

        try:
            return _open(row)
        except Exception:
            # Sealed under a key that has since changed. Nothing to be done
            # but connect again.
            log.warning("A Drive connection could not be opened; dropping it.")
            await session.delete(row)
            return None


async def drive_start(
    user_id: str, instance: str, poll_endpoint: str, poll_token: str
) -> Optional[dict]:
    """
    Remember a sign-in that has just begun, replacing whatever connection
    the user had. The replaced row is returned so the caller can revoke its
    app password.

    Returns:
        Optional[dict]: The replaced connection, if there was one.
    """

    replaced = None

    async with get_async_session() as session:
        result = await session.execute(
            select(DriveConnection).where(DriveConnection.user_id == user_id)
        )
        if row := result.scalars().first():
            try:
                replaced = _open(row)
            except Exception:
                replaced = None
            await session.delete(row)
            await session.flush()

        session.add(
            DriveConnection(
                user_id=user_id,
                instance=instance,
                poll_endpoint=poll_endpoint,
                poll_token=encrypt_with_key(_key(), poll_token),
                expires_at=_now() + timedelta(seconds=settings.DRIVE_LOGIN_TTL_SECONDS),
            )
        )

    return replaced if replaced and replaced["connected"] else None


async def drive_complete(
    user_id: str, login_name: str, app_password: str, dav_user: str
) -> bool:
    """
    Turn a pending sign-in into a connection. The poll token is dropped:
    it is spent.

    Returns:
        bool: False when there was no pending sign-in to complete.
    """

    async with get_async_session() as session:
        result = await session.execute(
            select(DriveConnection)
            .where(DriveConnection.user_id == user_id)
            .with_for_update()
        )
        if not (row := result.scalars().first()) or row.app_password:
            return False

        row.poll_endpoint = None
        row.poll_token = None
        row.login_name = login_name
        row.app_password = encrypt_with_key(_key(), app_password)
        row.dav_user = dav_user
        row.expires_at = _now() + timedelta(seconds=settings.DRIVE_SESSION_IDLE_SECONDS)

    return True


async def drive_touch(user_id: str) -> None:
    """
    Push a connection's expiry forward: it has just been used.
    """

    async with get_async_session() as session:
        result = await session.execute(
            select(DriveConnection).where(DriveConnection.user_id == user_id)
        )
        if row := result.scalars().first():
            row.expires_at = _now() + timedelta(
                seconds=settings.DRIVE_SESSION_IDLE_SECONDS
            )


async def drive_remove(user_id: str) -> Optional[dict]:
    """
    Forget a user's connection. Returns it, so the caller can revoke the
    app password on the Drive side.
    """

    async with get_async_session() as session:
        result = await session.execute(
            select(DriveConnection).where(DriveConnection.user_id == user_id)
        )
        if not (row := result.scalars().first()):
            return None

        try:
            removed = _open(row)
        except Exception:
            removed = None

        await session.delete(row)

    return removed


async def drive_take_expired() -> list[dict]:
    """
    Remove every expired connection in one statement and return the ones
    that held an app password, for the sweeper to revoke.
    """

    async with get_async_session() as session:
        result = await session.execute(
            delete(DriveConnection)
            .where(DriveConnection.expires_at <= _now())
            .returning(
                DriveConnection.instance,
                DriveConnection.login_name,
                DriveConnection.app_password,
            )
            .execution_options(synchronize_session=False)
        )
        rows = result.all()

    key = _key()
    expired = []

    for instance, login_name, sealed in rows:
        if not sealed:
            continue
        try:
            expired.append(
                {
                    "instance": instance,
                    "login_name": login_name,
                    "app_password": decrypt_with_key(key, sealed),
                }
            )
        except Exception:
            continue

    return expired


async def user_set_drive_url(user_id: str, drive_url: Optional[str]) -> None:
    """
    Store a user's own choice of Drive instance, or clear it (None) to fall
    back to their organisation's.
    """

    async with get_async_session() as session:
        result = await session.execute(select(User).where(User.user_id == user_id))
        if user := result.scalars().first():
            user.drive_url = drive_url
