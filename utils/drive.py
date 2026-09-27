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
Talking to a user's Sunet Drive, which is Nextcloud.

Scribe never holds standing access to anybody's Drive. A user connects with
Nextcloud's Login Flow v2: they sign in to their Drive in a tab of its own
and grant access there, Drive hands Scribe an app password for that grant,
and the password is revoked again after an idle hour (routers/drive.py,
db/drive.py). Everything here then acts as that Drive user, so Scribe sees
exactly what they are allowed to see and nothing else -- no Drive users,
groups or permissions are copied anywhere.

Files move between the two services server to server over WebDAV: GET to
bring one into Scribe, PUT to save one to Drive. Neither passes through the
user's own device.

Two rules hold for every request made from here:

- The instance is always one `normalise_instance()` accepted: https, on a
  host under DRIVE_ALLOWED_HOST_SUFFIXES. A user can choose their own
  instance, so without this the backend would fetch whatever URL it was
  given -- including addresses inside its own network.
- Redirects are never followed, and every URL Drive hands back (the login
  page, the poll endpoint, the server a login finished on) is checked to be
  on that same host before it is used.
"""

import posixpath

from collections.abc import AsyncIterator
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote, unquote, urlsplit

import defusedxml.ElementTree as ElementTree
import httpx

from utils.settings import get_settings

settings = get_settings()

USER_AGENT = "Sunet Scribe"

# The same list the upload dialog accepts. Only these can be brought in
# from Drive, since only these can be transcribed.
MEDIA_EXTENSIONS = frozenset(
    {
        ".mp3", ".wav", ".flac", ".mp4", ".mkv", ".avi", ".m4a", ".aiff",
        ".aif", ".mov", ".ogg", ".opus", ".webm", ".wma", ".mpg", ".mpeg",
    }
)

MAX_NAME_LENGTH = 255
MAX_PATH_LENGTH = 4096

PROPFIND_BODY = b"""<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns">
  <d:prop>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:getcontenttype/>
    <d:getlastmodified/>
    <oc:size/>
  </d:prop>
</d:propfind>
"""

DAV = "{DAV:}"
OC = "{http://owncloud.org/ns}"


class DriveError(Exception):
    """A request that cannot be carried out as asked. The message is safe
    to show the user."""


class DriveUnauthorized(DriveError):
    """Drive refused the app password: revoked in Drive, or expired."""


class DriveNotFound(DriveError):
    """No such file or folder in the user's Drive."""


class DriveConflict(DriveError):
    """The target already exists and was not to be overwritten."""


class DriveUnavailable(DriveError):
    """Drive could not be reached, or answered with something unusable."""


class TooLarge(DriveError):
    """More bytes than the limit allows."""


def http_client() -> httpx.AsyncClient:
    """
    The client every request to Drive goes through. Never follows a
    redirect: a redirect is Drive (or something pretending to be it)
    sending the request somewhere this module has not checked.

    Tests replace this to answer from a mock transport.
    """

    return httpx.AsyncClient(
        timeout=settings.DRIVE_HTTP_TIMEOUT_SECONDS,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )


def normalise_instance(url: Optional[str]) -> str:
    """
    The canonical form of a Drive instance URL, `https://<host>`, or
    DriveError when it is not one Scribe will talk to.

    Accepts a bare host as well ("su.drive.sunet.se"), since that is how a
    person is likely to type it. The host has to end in one of
    DRIVE_ALLOWED_HOST_SUFFIXES *as a whole label*: ".drive.sunet.se"
    admits "su.drive.sunet.se" and not "evildrive.sunet.se".

    Parameters:
        url (Optional[str]): What was entered.

    Returns:
        str: https://<host>, lower-cased, no port, path, query or fragment.
    """

    if not url or not (url := url.strip()):
        raise DriveError("No Drive address was given.")

    if "://" not in url:
        url = "https://" + url

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise DriveError("That is not a valid Drive address.")

    host = (parts.hostname or "").lower().rstrip(".")

    if parts.scheme.lower() != "https":
        raise DriveError("The Drive address must use https.")
    if parts.username or parts.password:
        raise DriveError("The Drive address must not contain a user name.")
    if port not in (None, 443):
        raise DriveError("The Drive address must not name a port.")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise DriveError("Give only the Drive address, e.g. https://su.drive.sunet.se.")
    if not host or not allowed_host(host):
        raise DriveError("That address is not a Sunet Drive instance.")

    return f"https://{host}"


def allowed_host(host: str) -> bool:
    """
    Whether a host is under one of DRIVE_ALLOWED_HOST_SUFFIXES.

    Parameters:
        host (str): A lower-case host name.

    Returns:
        bool: True when Scribe may talk to it.
    """

    for suffix in settings.DRIVE_ALLOWED_HOST_SUFFIXES:
        suffix = suffix.lower()
        if not suffix.startswith("."):
            suffix = "." + suffix
        if host.endswith(suffix) and len(host) > len(suffix):
            return True

    return False


def same_host(instance: str, url: str) -> bool:
    """
    Whether `url` is https on the instance's own host.

    Parameters:
        instance (str): A normalised instance, https://<host>.
        url (str): A URL Drive handed back.

    Returns:
        bool: True when it is safe to use.
    """

    try:
        parts = urlsplit(url)
        port = parts.port
    except (ValueError, TypeError):
        return False

    return (
        parts.scheme.lower() == "https"
        and port in (None, 443)
        and not parts.username
        and (parts.hostname or "").lower().rstrip(".") == urlsplit(instance).hostname
    )


def clean_path(path: Optional[str]) -> str:
    """
    A path inside the user's Drive, as "folder/sub/file" with no leading or
    trailing slash ("" is the top). Refuses anything that could step
    outside the user's own files: "..", ".", control characters.

    Parameters:
        path (Optional[str]): The path as the frontend sent it.

    Returns:
        str: The cleaned path.
    """

    path = path or ""

    if len(path) > MAX_PATH_LENGTH:
        raise DriveError("That path is too long.")

    parts = []

    for part in path.replace("\\", "/").split("/"):
        if not part:
            continue
        if part in (".", ".."):
            raise DriveError("That path is not allowed.")
        if any(ord(c) < 32 or ord(c) == 127 for c in part):
            raise DriveError("That path is not allowed.")
        if len(part) > MAX_NAME_LENGTH:
            raise DriveError("A name in that path is too long.")
        parts.append(part)

    return "/".join(parts)


def clean_name(name: Optional[str]) -> str:
    """
    A single file name to save as: one path segment, not empty.

    Parameters:
        name (Optional[str]): The name as the frontend sent it.

    Returns:
        str: The name.
    """

    name = (name or "").strip()
    cleaned = clean_path(name)

    if not cleaned or "/" in cleaned:
        raise DriveError("That is not a valid file name.")

    return cleaned


def is_media(name: str) -> bool:
    """
    Whether a file can be brought in to be transcribed.

    Parameters:
        name (str): A file name.

    Returns:
        bool: True for the formats the upload dialog accepts.
    """

    return posixpath.splitext(name.lower())[1] in MEDIA_EXTENSIONS


def dav_root(instance: str, dav_user: str) -> str:
    """
    The WebDAV URL of a Drive user's own files.
    """

    return f"{instance}/remote.php/dav/files/{quote(dav_user, safe='')}/"


def dav_url(instance: str, dav_user: str, path: str) -> str:
    """
    The WebDAV URL of a cleaned path in a Drive user's files, every segment
    quoted on its own so a name can hold anything a name may.
    """

    return dav_root(instance, dav_user) + "/".join(
        quote(part, safe="") for part in path.split("/") if part
    )


def _raise_for(response: httpx.Response) -> None:
    """
    Turn a Drive answer that is not a success into the error it means.
    """

    status = response.status_code

    if status < 300:
        return
    if status == 401:
        raise DriveUnauthorized("Drive no longer accepts Scribe's access.")
    if status == 403:
        raise DriveError("Drive does not allow that.")
    if status == 404:
        raise DriveNotFound("That file or folder is not in your Drive.")
    if status in (409, 412):
        raise DriveConflict("A file with that name already exists.")
    if status == 507:
        raise DriveError("Your Drive is full.")
    if 300 <= status < 400:
        raise DriveUnavailable("Drive answered with a redirect Scribe will not follow.")

    raise DriveUnavailable(f"Drive answered {status}.")


# ---------------------------------------------------------------------------
# Login Flow v2
# ---------------------------------------------------------------------------


async def login_start(instance: str) -> dict:
    """
    Begin a Login Flow v2 sign-in.

    Parameters:
        instance (str): A normalised instance.

    Returns:
        dict: {"login": <url the user opens>, "endpoint": <poll url>,
            "token": <poll token>}. The token is a secret.
    """

    try:
        async with http_client() as client:
            response = await client.post(f"{instance}/index.php/login/v2")
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")

    _raise_for(response)

    try:
        data = response.json()
        login = data["login"]
        endpoint = data["poll"]["endpoint"]
        token = data["poll"]["token"]
    except (ValueError, KeyError, TypeError):
        raise DriveUnavailable("Drive did not start a sign-in.")

    if not (
        isinstance(token, str)
        and token
        and same_host(instance, login)
        and same_host(instance, endpoint)
    ):
        raise DriveUnavailable("Drive did not start a sign-in.")

    return {"login": login, "endpoint": endpoint, "token": token}


async def login_poll(instance: str, endpoint: str, token: str) -> Optional[dict]:
    """
    Ask whether the user has granted access yet.

    Parameters:
        instance (str): The instance the sign-in began on.
        endpoint (str): The poll URL login_start() returned.
        token (str): The poll token.

    Returns:
        Optional[dict]: None while the user has not finished; otherwise
            {"login_name": ..., "app_password": ...}.
    """

    if not same_host(instance, endpoint):
        raise DriveUnavailable("The sign-in is not for this Drive.")

    try:
        async with http_client() as client:
            response = await client.post(endpoint, data={"token": token})
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")

    # 404 is Nextcloud's "not yet": the user has not granted access, or
    # the token has already been used or has expired.
    if response.status_code == 404:
        return None

    _raise_for(response)

    try:
        data = response.json()
        server = data["server"]
        login_name = data["loginName"]
        app_password = data["appPassword"]
    except (ValueError, KeyError, TypeError):
        raise DriveUnavailable("Drive did not finish the sign-in.")

    # The grant has to be for the instance the user chose, not for
    # wherever Drive says it is.
    if not same_host(instance, server) or not login_name or not app_password:
        raise DriveUnavailable("The sign-in finished on a different Drive.")

    return {"login_name": login_name, "app_password": app_password}


def _ocs_headers() -> dict:
    return {"OCS-APIRequest": "true", "Accept": "application/json"}


async def user_id(instance: str, login_name: str, app_password: str) -> str:
    """
    The Drive user id WebDAV paths are built from. Not always the login
    name: with federated sign-in the two differ.
    """

    try:
        async with http_client() as client:
            response = await client.get(
                f"{instance}/ocs/v2.php/cloud/user",
                params={"format": "json"},
                auth=(login_name, app_password),
                headers=_ocs_headers(),
            )
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")

    _raise_for(response)

    try:
        uid = response.json()["ocs"]["data"]["id"]
    except (ValueError, KeyError, TypeError):
        raise DriveUnavailable("Drive did not say who you are.")

    if not isinstance(uid, str) or not uid:
        raise DriveUnavailable("Drive did not say who you are.")

    return uid


async def revoke(instance: str, login_name: str, app_password: str) -> bool:
    """
    Revoke an app password, so the grant ends on the Drive side too.

    Returns:
        bool: True when Drive confirmed, or the password was already gone.
    """

    try:
        async with http_client() as client:
            response = await client.delete(
                f"{instance}/ocs/v2.php/core/apppassword",
                auth=(login_name, app_password),
                headers=_ocs_headers(),
            )
    except httpx.HTTPError:
        return False

    return response.status_code < 300 or response.status_code == 401


# ---------------------------------------------------------------------------
# WebDAV
# ---------------------------------------------------------------------------


def parse_listing(body: bytes, root_path: str, path: str) -> list[dict]:
    """
    The entries of a Depth: 1 PROPFIND, without the folder itself.

    Parameters:
        body (bytes): The multistatus XML.
        root_path (str): The URL path of the user's WebDAV root, as
            dav_root() builds it, to strip from every href.
        path (str): The cleaned path that was listed.

    Returns:
        list[dict]: {"name", "path", "is_dir", "size", "mime", "modified"},
            folders first, then by name.
    """

    try:
        tree = ElementTree.fromstring(body)
    except Exception:
        raise DriveUnavailable("Drive answered with a listing Scribe cannot read.")

    root_path = unquote(root_path)
    entries = []

    for response in tree.findall(f"{DAV}response"):
        href = response.findtext(f"{DAV}href") or ""
        href = unquote(urlsplit(href).path)

        if not href.startswith(root_path):
            continue

        entry_path = href[len(root_path):].strip("/")

        if entry_path == path:
            continue

        # Only the direct children of the folder asked for.
        if posixpath.dirname(entry_path) != path:
            continue

        prop = None
        for propstat in response.findall(f"{DAV}propstat"):
            if "200" in (propstat.findtext(f"{DAV}status") or ""):
                prop = propstat.find(f"{DAV}prop")
                break

        if prop is None:
            continue

        resourcetype = prop.find(f"{DAV}resourcetype")
        is_dir = resourcetype is not None and resourcetype.find(f"{DAV}collection") is not None

        size_text = prop.findtext(f"{OC}size") if is_dir else prop.findtext(
            f"{DAV}getcontentlength"
        )
        try:
            size = int(size_text) if size_text else None
        except ValueError:
            size = None

        modified = prop.findtext(f"{DAV}getlastmodified")
        try:
            modified = parsedate_to_datetime(modified).isoformat() if modified else None
        except (TypeError, ValueError):
            modified = None

        entries.append(
            {
                "name": posixpath.basename(entry_path),
                "path": entry_path,
                "is_dir": is_dir,
                "size": size,
                "mime": None if is_dir else prop.findtext(f"{DAV}getcontenttype"),
                "modified": modified,
            }
        )

        if len(entries) >= settings.DRIVE_MAX_LIST_ENTRIES:
            break

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return entries


async def list_folder(connection: dict, path: str) -> list[dict]:
    """
    The files and folders directly inside a folder of the user's Drive.

    Parameters:
        connection (dict): A connected row from db/drive.py.
        path (str): A cleaned path.
    """

    root = dav_root(connection["instance"], connection["dav_user"])

    try:
        async with http_client() as client:
            response = await client.request(
                "PROPFIND",
                dav_url(connection["instance"], connection["dav_user"], path),
                content=PROPFIND_BODY,
                headers={"Depth": "1", "Content-Type": "application/xml"},
                auth=(connection["login_name"], connection["app_password"]),
            )
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")

    _raise_for(response)

    if response.status_code != 207:
        raise DriveUnavailable("Drive answered with something other than a listing.")

    return parse_listing(response.content, urlsplit(root).path, path)


class _Reader:
    """
    An `async read(size)` over a streamed response, which is what
    encrypt_stream_to_file() takes, refusing to go past a limit so a file
    larger than Scribe accepts is stopped while it arrives rather than after.
    """

    def __init__(self, chunks: AsyncIterator[bytes], limit: int) -> None:
        self._chunks = chunks
        self._buffer = bytearray()
        self._limit = limit
        self._total = 0
        self._done = False

    async def read(self, size: int = -1) -> bytes:
        while not self._done and (size < 0 or len(self._buffer) < size):
            try:
                chunk = await self._chunks.__anext__()
            except StopAsyncIteration:
                self._done = True
                break

            self._total += len(chunk)
            if self._total > self._limit:
                raise TooLarge("That file is larger than Scribe accepts.")
            self._buffer.extend(chunk)

        if size < 0:
            size = len(self._buffer)

        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data


async def download(connection: dict, path: str, sink, limit: int) -> None:
    """
    Stream a file from the user's Drive into `sink`, an async callable
    given a reader (see _Reader), without holding the file in memory.

    Parameters:
        connection (dict): A connected row from db/drive.py.
        path (str): A cleaned path to a file.
        sink: async (reader) -> None, e.g. encrypting to disk.
        limit (int): Most bytes accepted.
    """

    try:
        async with http_client() as client:
            async with client.stream(
                "GET",
                dav_url(connection["instance"], connection["dav_user"], path),
                auth=(connection["login_name"], connection["app_password"]),
            ) as response:
                if response.status_code >= 300:
                    await response.aread()
                    _raise_for(response)

                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > limit:
                    raise TooLarge("That file is larger than Scribe accepts.")

                await sink(_Reader(response.aiter_bytes(), limit))
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")


async def upload(
    connection: dict,
    path: str,
    content: AsyncIterator[bytes],
    overwrite: bool = False,
) -> None:
    """
    Stream a file into the user's Drive.

    Parameters:
        connection (dict): A connected row from db/drive.py.
        path (str): A cleaned path, folder and name.
        content: The file, as chunks.
        overwrite (bool): When False, an existing file is left alone and
            DriveConflict raised instead.
    """

    headers = {"Content-Type": "application/octet-stream"}

    if not overwrite:
        # Makes the PUT conditional on nothing being there: one request, so
        # no gap between checking and writing for another file to land in.
        headers["If-None-Match"] = "*"

    try:
        async with http_client() as client:
            response = await client.put(
                dav_url(connection["instance"], connection["dav_user"], path),
                content=content,
                headers=headers,
                auth=(connection["login_name"], connection["app_password"]),
            )
    except httpx.HTTPError:
        raise DriveUnavailable("Drive could not be reached.")

    _raise_for(response)
