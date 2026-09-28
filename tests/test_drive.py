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
Get from / Save to Sunet Drive (SUNET/scribe-backend#64).

Nextcloud is played by an httpx MockTransport answering the handful of
endpoints Scribe uses (Login Flow v2, the OCS user and app-password
endpoints, WebDAV). The database is in-memory SQLite and the router runs on
a FastAPI app of its own with the signed-in user replaced, so nothing here
needs a real Drive, database or identity provider.
"""

import os

os.environ.setdefault("API_DATABASE_URL", "sqlite://")

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio

from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import db.drive as drive_db
import routers.drive as drive_router
import utils.drive as drive

from auth.oidc import get_current_user
from db.models import DriveConnection
from utils.crypto import (
    decrypt_data_from_file,
    generate_rsa_keypair,
    serialize_public_key_to_pem,
)

INSTANCE = "https://su.drive.sunet.se"
USER_ID = "scribe-user-1"
DAV_USER = "uid1"
LOGIN_NAME = "someone@su.se"
APP_PASSWORD = "app-password-secret"
POLL_TOKEN = "poll-token-secret"


# ---------------------------------------------------------------------------
# A pretend Nextcloud
# ---------------------------------------------------------------------------


class FakeDrive:
    """
    Just enough of Nextcloud to connect, list, fetch and save.
    """

    def __init__(self, host: str = "su.drive.sunet.se") -> None:
        self.host = host
        self.granted = False
        self.revoked: list[str] = []
        self.files: dict[str, bytes] = {"Lectures/talk one.mp3": b"ID3" + b"a" * 5000}
        self.folders = {"", "Lectures", "Notes"}
        self.poll_server = f"https://{host}"
        self.poll_endpoint = f"https://{host}/login/v2/poll"
        self.login_url = f"https://{host}/login/v2/flow/abc"

    def authorised(self, request: httpx.Request) -> bool:
        expected = httpx.BasicAuth(LOGIN_NAME, APP_PASSWORD)
        header = next(expected.auth_flow(httpx.Request("GET", "https://x"))).headers[
            "Authorization"
        ]
        return (
            request.headers.get("Authorization") == header
            and APP_PASSWORD not in self.revoked
        )

    def listing(self, folder: str) -> bytes:
        from urllib.parse import quote

        def parent(path: str) -> str:
            return path.rsplit("/", 1)[0] if "/" in path else ""

        root = f"/remote.php/dav/files/{DAV_USER}/"
        # The folder itself comes first, as Nextcloud sends it.
        responses = [(folder, True)]
        responses += [(f, True) for f in sorted(self.folders) if f and parent(f) == folder]
        responses += [(p, False) for p in self.files if parent(p) == folder]

        parts = []
        for path, is_dir in responses:
            href = root + quote(path)
            if is_dir:
                href += "/"
                prop = "<d:resourcetype><d:collection/></d:resourcetype><oc:size>42</oc:size>"
            else:
                prop = (
                    "<d:resourcetype/>"
                    f"<d:getcontentlength>{len(self.files[path])}</d:getcontentlength>"
                    "<d:getcontenttype>audio/mpeg</d:getcontenttype>"
                )
            parts.append(
                f"<d:response><d:href>{href}</d:href><d:propstat><d:prop>{prop}"
                "<d:getlastmodified>Mon, 21 Sep 2026 08:00:00 GMT</d:getlastmodified>"
                "</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response>"
            )

        return (
            '<?xml version="1.0"?><d:multistatus xmlns:d="DAV:" '
            'xmlns:oc="http://owncloud.org/ns">' + "".join(parts) + "</d:multistatus>"
        ).encode()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == self.host, "Scribe talked to a host it should not"
        path = request.url.path

        if request.method == "POST" and path == "/index.php/login/v2":
            return httpx.Response(
                200,
                json={
                    "poll": {"token": POLL_TOKEN, "endpoint": self.poll_endpoint},
                    "login": self.login_url,
                },
            )

        if request.method == "POST" and path == "/login/v2/poll":
            assert f"token={POLL_TOKEN}" in request.content.decode()
            if not self.granted:
                return httpx.Response(404, json=[])
            return httpx.Response(
                200,
                json={
                    "server": self.poll_server,
                    "loginName": LOGIN_NAME,
                    "appPassword": APP_PASSWORD,
                },
            )

        if path == "/ocs/v2.php/cloud/user":
            assert request.headers.get("OCS-APIRequest") == "true"
            if not self.authorised(request):
                return httpx.Response(401)
            return httpx.Response(200, json={"ocs": {"data": {"id": DAV_USER}}})

        if request.method == "DELETE" and path == "/ocs/v2.php/core/apppassword":
            if self.authorised(request):
                self.revoked.append(APP_PASSWORD)
            return httpx.Response(200, json={})

        prefix = f"/remote.php/dav/files/{DAV_USER}"
        if path.startswith(prefix):
            if not self.authorised(request):
                return httpx.Response(401)

            target = httpx.URL(request.url).path[len(prefix):].strip("/")
            from urllib.parse import unquote

            target = unquote(target)

            if request.method == "PROPFIND":
                assert request.headers.get("Depth") == "1"
                if target not in self.folders:
                    return httpx.Response(404)
                return httpx.Response(207, content=self.listing(target))

            if request.method == "GET":
                if target not in self.files:
                    return httpx.Response(404)
                return httpx.Response(200, content=self.files[target])

            if request.method == "PUT":
                if target in self.files and request.headers.get("If-None-Match") == "*":
                    return httpx.Response(412)
                self.files[target] = request.read()
                return httpx.Response(201)

        return httpx.Response(500)


@pytest.fixture()
def fake(monkeypatch):
    fake = FakeDrive()

    def client():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(fake), follow_redirects=False
        )

    monkeypatch.setattr(drive, "http_client", client)
    return fake


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = factory()
    yield session
    await session.close()
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _patch_session(db_session):
    @asynccontextmanager
    async def _get_async_session():
        try:
            yield db_session
            await db_session.commit()
        except Exception:
            await db_session.rollback()
            raise

    with patch("db.drive.get_async_session", _get_async_session):
        yield


# ---------------------------------------------------------------------------
# Instances: only Sunet Drive, only https
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entered, expected",
    [
        ("https://su.drive.sunet.se", INSTANCE),
        ("https://SU.Drive.Sunet.SE/", INSTANCE),
        ("su.drive.sunet.se", INSTANCE),
        ("https://su.drive.sunet.se:443", INSTANCE),
        ("  https://kau.drive.sunet.se  ", "https://kau.drive.sunet.se"),
    ],
)
def test_a_drive_instance_is_normalised(entered, expected):
    assert drive.normalise_instance(entered) == expected


@pytest.mark.parametrize(
    "entered",
    [
        "",
        None,
        "http://su.drive.sunet.se",
        "https://drive.sunet.se.evil.example",
        "https://evildrive.sunet.se",
        "https://drive.sunet.se",
        "https://example.com",
        "https://localhost",
        "https://127.0.0.1",
        "https://su.drive.sunet.se:8443",
        "https://user:pw@su.drive.sunet.se",
        "https://su.drive.sunet.se/index.php",
        "https://su.drive.sunet.se?x=1",
        "file:///etc/passwd",
    ],
)
def test_anything_but_a_sunet_drive_host_is_refused(entered):
    with pytest.raises(drive.DriveError):
        drive.normalise_instance(entered)


def test_the_allowed_hosts_come_from_settings(monkeypatch):
    monkeypatch.setattr(
        drive.settings, "DRIVE_ALLOWED_HOST_SUFFIXES", ["drive.example.org"]
    )

    assert drive.normalise_instance("x.drive.example.org") == "https://x.drive.example.org"
    with pytest.raises(drive.DriveError):
        drive.normalise_instance("su.drive.sunet.se")


def test_a_url_drive_hands_back_must_be_on_the_same_host():
    assert drive.same_host(INSTANCE, INSTANCE + "/login/v2/poll")
    assert not drive.same_host(INSTANCE, "https://kau.drive.sunet.se/login/v2/poll")
    assert not drive.same_host(INSTANCE, "http://su.drive.sunet.se/login/v2/poll")
    assert not drive.same_host(INSTANCE, "https://su.drive.sunet.se:8443/x")
    assert not drive.same_host(INSTANCE, "https://a@su.drive.sunet.se/x")


# ---------------------------------------------------------------------------
# Paths inside a Drive
# ---------------------------------------------------------------------------


def test_paths_are_cleaned():
    assert drive.clean_path("/Lectures//2026/") == "Lectures/2026"
    assert drive.clean_path("") == ""
    assert drive.clean_path(None) == ""
    assert drive.clean_path("a\\b") == "a/b"


@pytest.mark.parametrize("path", ["../other", "a/../../b", "./a", "a/\x00b", "a/\nb"])
def test_a_path_cannot_step_outside(path):
    with pytest.raises(drive.DriveError):
        drive.clean_path(path)


def test_every_segment_is_quoted_in_the_dav_url():
    url = drive.dav_url(INSTANCE, "a user", "Min mapp/fil #1?.mp3")

    assert url == (
        "https://su.drive.sunet.se/remote.php/dav/files/a%20user/"
        "Min%20mapp/fil%20%231%3F.mp3"
    )


def test_a_name_to_save_as_is_one_segment():
    assert drive.clean_name(" talk.srt ") == "talk.srt"
    for name in ("", "a/b.srt", ".."):
        with pytest.raises(drive.DriveError):
            drive.clean_name(name)


def test_only_media_can_be_brought_in():
    assert drive.is_media("Talk.MP4")
    assert not drive.is_media("notes.docx")
    assert not drive.is_media("mp3")


# ---------------------------------------------------------------------------
# The client against a pretend Nextcloud
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signing_in_waits_until_access_is_granted(fake):
    flow = await drive.login_start(INSTANCE)

    assert flow["login"] == fake.login_url
    assert await drive.login_poll(INSTANCE, flow["endpoint"], flow["token"]) is None

    fake.granted = True
    granted = await drive.login_poll(INSTANCE, flow["endpoint"], flow["token"])

    assert granted == {"login_name": LOGIN_NAME, "app_password": APP_PASSWORD}
    assert await drive.user_id(INSTANCE, LOGIN_NAME, APP_PASSWORD) == DAV_USER


@pytest.mark.asyncio
async def test_a_poll_endpoint_on_another_host_is_not_followed(fake):
    fake.poll_endpoint = "https://internal.example/poll"

    with pytest.raises(drive.DriveUnavailable):
        await drive.login_start(INSTANCE)


@pytest.mark.asyncio
async def test_a_grant_for_another_drive_is_refused(fake):
    fake.granted = True
    fake.poll_server = "https://kau.drive.sunet.se"

    with pytest.raises(drive.DriveUnavailable):
        await drive.login_poll(INSTANCE, fake.poll_endpoint, POLL_TOKEN)


def connection() -> dict:
    return {
        "instance": INSTANCE,
        "login_name": LOGIN_NAME,
        "app_password": APP_PASSWORD,
        "dav_user": DAV_USER,
    }


@pytest.mark.asyncio
async def test_a_folder_is_listed_folders_first(fake):
    entries = await drive.list_folder(connection(), "")

    assert [(e["name"], e["is_dir"]) for e in entries] == [
        ("Lectures", True),
        ("Notes", True),
    ]

    entries = await drive.list_folder(connection(), "Lectures")

    assert entries == [
        {
            "name": "talk one.mp3",
            "path": "Lectures/talk one.mp3",
            "is_dir": False,
            "size": 5003,
            "mime": "audio/mpeg",
            "modified": "2026-09-21T08:00:00+00:00",
        }
    ]


def test_a_listing_with_entities_is_refused():
    bomb = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]>
<d:multistatus xmlns:d="DAV:"><d:response><d:href>&lol2;</d:href></d:response></d:multistatus>"""

    with pytest.raises(drive.DriveUnavailable):
        drive.parse_listing(bomb, "/remote.php/dav/files/uid1/", "")


@pytest.mark.asyncio
async def test_a_revoked_password_is_unauthorised(fake):
    fake.revoked.append(APP_PASSWORD)

    with pytest.raises(drive.DriveUnauthorized):
        await drive.list_folder(connection(), "")


@pytest.mark.asyncio
async def test_saving_does_not_overwrite_unless_asked(fake):
    async def body(data):
        yield data

    await drive.upload(connection(), "Notes/a.srt", body(b"one"))
    assert fake.files["Notes/a.srt"] == b"one"

    with pytest.raises(drive.DriveConflict):
        await drive.upload(connection(), "Notes/a.srt", body(b"two"))
    assert fake.files["Notes/a.srt"] == b"one"

    await drive.upload(connection(), "Notes/a.srt", body(b"two"), overwrite=True)
    assert fake.files["Notes/a.srt"] == b"two"


@pytest.mark.asyncio
async def test_a_download_past_the_limit_is_stopped(fake):
    async def sink(reader):
        while await reader.read(1000):
            pass

    with pytest.raises(drive.TooLarge):
        await drive.download(connection(), "Lectures/talk one.mp3", sink, 1000)


# ---------------------------------------------------------------------------
# Connections at rest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_secrets_are_never_stored_in_the_clear(db_session):
    await drive_db.drive_start(USER_ID, INSTANCE, INSTANCE + "/login/v2/poll", POLL_TOKEN)

    row = (await db_session.execute(select(DriveConnection))).scalars().one()
    assert POLL_TOKEN not in row.poll_token

    await drive_db.drive_complete(USER_ID, LOGIN_NAME, APP_PASSWORD, DAV_USER)

    row = (await db_session.execute(select(DriveConnection))).scalars().one()
    await db_session.refresh(row)
    assert APP_PASSWORD not in row.app_password
    assert row.poll_token is None

    opened = await drive_db.drive_get(USER_ID)
    assert opened["app_password"] == APP_PASSWORD
    assert opened["connected"]


@pytest.mark.asyncio
async def test_an_expired_connection_is_gone_and_handed_over_for_revoking(db_session):
    await drive_db.drive_start(USER_ID, INSTANCE, INSTANCE + "/login/v2/poll", POLL_TOKEN)
    await drive_db.drive_complete(USER_ID, LOGIN_NAME, APP_PASSWORD, DAV_USER)

    row = (await db_session.execute(select(DriveConnection))).scalars().one()
    row.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)
    await db_session.commit()

    assert await drive_db.drive_get(USER_ID) is None
    assert await drive_db.drive_take_expired() == [
        {"instance": INSTANCE, "login_name": LOGIN_NAME, "app_password": APP_PASSWORD}
    ]
    assert (await db_session.execute(select(DriveConnection))).scalars().all() == []


@pytest.mark.asyncio
async def test_connecting_again_hands_back_the_old_grant_to_revoke():
    await drive_db.drive_start(USER_ID, INSTANCE, INSTANCE + "/p", POLL_TOKEN)
    await drive_db.drive_complete(USER_ID, LOGIN_NAME, APP_PASSWORD, DAV_USER)

    replaced = await drive_db.drive_start(USER_ID, INSTANCE, INSTANCE + "/p", "new")

    assert replaced["app_password"] == APP_PASSWORD
    assert not (await drive_db.drive_get(USER_ID))["connected"]


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keys():
    return {"api": generate_rsa_keypair(2048), "user": generate_rsa_keypair(2048)}


@pytest.fixture()
def customer():
    return {
        "drive_enabled": True,
        "drive_url": INSTANCE,
        "drive_display_name": "SU Box",
    }


@pytest.fixture()
def user():
    return {"user_id": USER_ID}


@pytest.fixture()
def api(monkeypatch, customer, user, keys, tmp_path, fake):
    app = FastAPI()
    app.include_router(drive_router.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: user

    async def customer_for(_user_id):
        return customer

    jobs = {}

    async def job_create(user_id, job_type, filename):
        jobs["job"] = {"uuid": "job-1", "status": "pending", "job_type": "transcription"}
        return jobs["job"]

    async def job_update(uuid, **kwargs):
        jobs["job"]["status"] = str(kwargs.get("status"))
        return jobs["job"]

    async def job_remove(uuid):
        jobs.pop("job", None)
        return True

    async def user_get(username):
        return {"user_id": "api"}

    async def public_key(user_id):
        pair = keys["api"] if user_id == "api" else keys["user"]
        return serialize_public_key_to_pem(pair[1])

    monkeypatch.setattr(drive_router, "customer_get_from_user_id", customer_for)
    monkeypatch.setattr(drive_router, "job_create", job_create)
    monkeypatch.setattr(drive_router, "job_update", job_update)
    monkeypatch.setattr(drive_router, "job_remove", job_remove)
    monkeypatch.setattr(drive_router, "user_get", user_get)
    monkeypatch.setattr(drive_router, "user_get_public_key", public_key)
    monkeypatch.setattr(drive_router.settings, "API_FILE_STORAGE_DIR", str(tmp_path))

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://scribe"
    )
    client.jobs = jobs
    return client


async def connect(api, fake) -> None:
    started = await api.post("/api/v1/drive/connect")
    assert started.status_code == 200
    assert started.json()["result"]["login_url"] == fake.login_url

    fake.granted = True
    polled = await api.get("/api/v1/drive/connect")
    assert polled.json()["result"]["state"] == "connected"


@pytest.mark.asyncio
async def test_drive_is_hidden_from_an_organisation_without_it(api, customer):
    customer["drive_enabled"] = False

    status = await api.get("/api/v1/drive")
    assert status.json()["result"] == {"enabled": False, "display_name": "Sunet Drive"}

    for method, url in [
        ("POST", "/api/v1/drive/connect"),
        ("GET", "/api/v1/drive/connect"),
        ("GET", "/api/v1/drive/files"),
    ]:
        response = await api.request(method, url)
        assert response.status_code == 403
        assert response.json()["reason"] == "disabled"


@pytest.mark.parametrize("drive_url", [None, "", "https://example.com"])
@pytest.mark.asyncio
async def test_an_organisation_without_a_sunet_drive_instance_has_no_drive(
    api, customer, drive_url
):
    customer["drive_url"] = drive_url

    status = await api.get("/api/v1/drive")
    assert status.json()["result"]["enabled"] is False

    response = await api.post("/api/v1/drive/connect")
    assert response.json()["reason"] == "disabled"


@pytest.mark.asyncio
async def test_users_cannot_choose_an_instance(api):
    response = await api.put("/api/v1/drive/instance", json={"url": INSTANCE})

    assert response.status_code in (404, 405)


@pytest.mark.asyncio
async def test_the_organisation_names_its_own_drive(api, customer):
    status = (await api.get("/api/v1/drive")).json()["result"]
    assert status["display_name"] == "SU Box"
    assert status["instance"] == INSTANCE
    assert not status["connected"]

    customer["drive_display_name"] = None
    status = (await api.get("/api/v1/drive")).json()["result"]
    assert status["display_name"] == "Sunet Drive"


@pytest.mark.asyncio
async def test_nothing_is_reachable_before_connecting(api):
    response = await api.get("/api/v1/drive/files")

    assert response.status_code == 409
    assert response.json()["reason"] == "not_connected"


@pytest.mark.asyncio
async def test_connect_list_and_disconnect(api, fake):
    status = await api.get("/api/v1/drive/connect")
    assert status.json()["result"]["state"] == "none"

    await api.post("/api/v1/drive/connect")
    assert (await api.get("/api/v1/drive/connect")).json()["result"]["state"] == "pending"
    assert (await api.get("/api/v1/drive")).json()["result"]["pending"]

    fake.granted = True
    assert (await api.get("/api/v1/drive/connect")).json()["result"]["state"] == "connected"

    listing = (await api.get("/api/v1/drive/files", params={"path": "Lectures"})).json()
    assert listing["result"]["entries"][0]["media"] is True

    await api.delete("/api/v1/drive/connect")
    assert APP_PASSWORD in fake.revoked
    assert (await api.get("/api/v1/drive/files")).status_code == 409


@pytest.mark.asyncio
async def test_a_grant_revoked_in_drive_ends_the_connection(api, fake):
    await connect(api, fake)
    fake.revoked.append(APP_PASSWORD)

    response = await api.get("/api/v1/drive/files")

    assert response.status_code == 409
    assert response.json()["reason"] == "not_connected"
    assert not (await api.get("/api/v1/drive")).json()["result"]["connected"]


@pytest.mark.asyncio
async def test_a_connection_to_an_instance_the_organisation_left_does_not_count(
    api, fake, customer
):
    await connect(api, fake)

    customer["drive_url"] = "https://kau.drive.sunet.se"

    assert (await api.get("/api/v1/drive/files")).json()["reason"] == "not_connected"
    assert not (await api.get("/api/v1/drive")).json()["result"]["connected"]


@pytest.mark.asyncio
async def test_logging_out_revokes_the_grant_in_drive(api, fake):
    await connect(api, fake)

    response = await api.delete("/api/v1/drive/connect")

    assert response.json()["result"]["state"] == "none"
    assert APP_PASSWORD in fake.revoked
    assert not (await api.get("/api/v1/drive")).json()["result"]["connected"]


@pytest.mark.asyncio
async def test_a_file_is_brought_in_encrypted(api, fake, keys, tmp_path):
    await connect(api, fake)

    response = await api.post(
        "/api/v1/drive/import", json={"path": "Lectures/talk one.mp3"}
    )

    assert response.status_code == 200
    assert response.json()["result"]["filename"] == "talk one.mp3"

    stored = tmp_path / USER_ID / "job-1"
    audio = fake.files["Lectures/talk one.mp3"]
    assert audio[:64] not in stored.read_bytes()
    assert b"".join(decrypt_data_from_file(keys["api"][0], str(stored))) == audio
    assert "UPLOADED" in api.jobs["job"]["status"].upper()


@pytest.mark.asyncio
async def test_only_media_is_brought_in(api, fake):
    await connect(api, fake)
    fake.files["Notes/notes.docx"] = b"PK"

    response = await api.post("/api/v1/drive/import", json={"path": "Notes/notes.docx"})

    assert response.status_code == 400
    assert "job" not in api.jobs


@pytest.mark.asyncio
async def test_a_missing_file_leaves_no_job_behind(api, fake):
    await connect(api, fake)

    response = await api.post("/api/v1/drive/import", json={"path": "Lectures/gone.mp3"})

    assert response.status_code == 404
    assert "job" not in api.jobs


@pytest.mark.asyncio
async def test_a_file_is_saved_and_not_overwritten_unless_asked(api, fake):
    await connect(api, fake)
    params = {"path": "Notes", "name": "talk.srt"}

    saved = await api.put("/api/v1/drive/files", params=params, content=b"1\n")
    assert saved.status_code == 200
    assert fake.files["Notes/talk.srt"] == b"1\n"

    again = await api.put("/api/v1/drive/files", params=params, content=b"2\n")
    assert again.status_code == 409
    assert again.json()["reason"] == "exists"

    forced = await api.put(
        "/api/v1/drive/files", params={**params, "overwrite": "true"}, content=b"2\n"
    )
    assert forced.status_code == 200
    assert fake.files["Notes/talk.srt"] == b"2\n"


@pytest.mark.asyncio
async def test_a_save_past_the_limit_is_refused(api, fake, monkeypatch):
    await connect(api, fake)
    monkeypatch.setattr(drive_router.settings, "DRIVE_MAX_SAVE_BYTES", 10)

    response = await api.put(
        "/api/v1/drive/files", params={"name": "big.txt"}, content=b"x" * 11
    )

    assert response.status_code == 413
    assert "big.txt" not in fake.files


@pytest.mark.asyncio
async def test_a_save_cannot_climb_out_of_the_drive(api, fake):
    await connect(api, fake)

    response = await api.put(
        "/api/v1/drive/files", params={"path": "../x", "name": "a.txt"}, content=b"a"
    )

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Customer settings
# ---------------------------------------------------------------------------


@pytest.fixture()
def admin_api(monkeypatch):
    import routers.customers as customers_router
    from auth.oidc import get_current_admin_user

    app = FastAPI()
    app.include_router(customers_router.router, prefix="/api/v1")
    app.dependency_overrides[get_current_admin_user] = lambda: {
        "user_id": "bofh", "bofh": True, "admin": True
    }

    stored = {"drive_enabled": False, "drive_url": None}
    saved = {}

    async def create(**kwargs):
        saved.update(kwargs)
        return kwargs

    async def update(customer_id, **kwargs):
        saved.update(kwargs)
        return kwargs

    async def get(customer_id):
        return stored

    monkeypatch.setattr(customers_router, "customer_create", create)
    monkeypatch.setattr(customers_router, "customer_update", update)
    monkeypatch.setattr(customers_router, "customer_get", get)

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://scribe"
    )
    client.stored, client.saved = stored, saved
    return client


@pytest.mark.asyncio
async def test_a_customer_drive_instance_must_be_sunet_drive(admin_api):
    response = await admin_api.put(
        "/api/v1/admin/customers/1",
        json={"drive_enabled": True, "drive_url": "https://example.com"},
    )

    assert response.status_code == 400
    assert admin_api.saved == {}


@pytest.mark.asyncio
async def test_drive_cannot_be_offered_without_an_instance(admin_api):
    created = await admin_api.post(
        "/api/v1/admin/customers",
        json={"partner_id": "N/A", "name": "Org", "drive_enabled": True},
    )
    assert created.status_code == 400

    updated = await admin_api.put(
        "/api/v1/admin/customers/1", json={"drive_enabled": True}
    )
    assert updated.status_code == 400

    admin_api.stored.update(drive_enabled=True, drive_url=INSTANCE)
    cleared = await admin_api.put("/api/v1/admin/customers/1", json={"drive_url": ""})
    assert cleared.status_code == 400
    assert admin_api.saved == {}


@pytest.mark.asyncio
async def test_a_customer_drive_instance_is_saved_normalised(admin_api):
    response = await admin_api.put(
        "/api/v1/admin/customers/1",
        json={"drive_enabled": True, "drive_url": "SU.drive.sunet.se"},
    )

    assert response.status_code == 200
    assert admin_api.saved["drive_url"] == INSTANCE
    assert admin_api.saved["drive_enabled"] is True


# ---------------------------------------------------------------------------
# A recording's original, saved to Drive
# ---------------------------------------------------------------------------


@pytest.fixture()
def original(monkeypatch, keys, tmp_path):
    """
    A recording's original on disk, encrypted for its owner as
    utils/recordings.py keeps it, and the job that names it.
    """

    import asyncio

    from utils.crypto import (
        encrypt_stream_to_file,
        encrypt_string,
        serialize_private_key_to_pem,
    )
    from utils.recordings import original_path

    private, public = keys["user"]
    audio = b"OggS" + os.urandom(3000)
    path = original_path(USER_ID, "job-9")
    path.parent.mkdir(parents=True, exist_ok=True)

    class Reader:
        def __init__(self, data):
            self.data = data

        async def read(self, size=-1):
            chunk, self.data = self.data[:size], self.data[size:]
            return chunk

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        encrypt_stream_to_file(public, Reader(audio), str(path))
    )

    job = {"uuid": "job-9", "filename": encrypt_string(public, "Seminar.webm")}

    async def job_get(uuid, user_id):
        return job if uuid == "job-9" and user_id == USER_ID else None

    async def private_key(user_id):
        return serialize_private_key_to_pem(private, b"secret")

    monkeypatch.setattr(drive_router, "job_get", job_get)
    monkeypatch.setattr(drive_router, "user_get_private_key", private_key)

    return audio


@pytest.mark.asyncio
async def test_an_original_is_saved_decrypted_under_its_own_name(api, fake, original):
    await connect(api, fake)

    response = await api.post(
        "/api/v1/drive/save-original",
        json={"job_id": "job-9", "encryption_password": "secret", "path": "Notes"},
    )

    assert response.status_code == 200
    assert response.json()["result"]["path"] == "Notes/Seminar.webm"
    assert fake.files["Notes/Seminar.webm"] == original


@pytest.mark.asyncio
async def test_an_original_is_not_overwritten_unless_asked(api, fake, original):
    await connect(api, fake)
    fake.files["Seminar.webm"] = b"older"
    body = {"job_id": "job-9", "encryption_password": "secret"}

    refused = await api.post("/api/v1/drive/save-original", json=body)
    assert refused.status_code == 409
    assert refused.json()["reason"] == "exists"
    assert fake.files["Seminar.webm"] == b"older"

    renamed = await api.post(
        "/api/v1/drive/save-original", json={**body, "name": "Seminar (2).webm"}
    )
    assert renamed.status_code == 200
    assert fake.files["Seminar (2).webm"] == original


@pytest.mark.asyncio
async def test_an_original_needs_the_encryption_password(api, fake, original):
    await connect(api, fake)

    response = await api.post(
        "/api/v1/drive/save-original",
        json={"job_id": "job-9", "encryption_password": "wrong"},
    )

    assert response.status_code == 403
    assert "Seminar.webm" not in fake.files


@pytest.mark.asyncio
async def test_only_a_job_with_an_original_has_one_to_save(api, fake, original):
    await connect(api, fake)

    response = await api.post(
        "/api/v1/drive/save-original",
        json={"job_id": "someone-elses", "encryption_password": "secret"},
    )

    assert response.status_code == 404
