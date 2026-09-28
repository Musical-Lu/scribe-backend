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

Offered only to users whose organisation has it enabled (Customer.
drive_enabled) *and* has said which Drive is theirs (Customer.drive_url, a
Sunet Drive host -- see utils/drive.py). Users do not choose an instance:
an organisation without one simply does not get Drive.

The flow, as the frontend drives it:

- GET  /drive                  what is offered: enabled, display name,
                               instance, connected.
- POST /drive/connect          start signing in to Drive; answers the URL the
                               user opens in a tab of its own.
- GET  /drive/connect          poll: has the user granted access yet?
- DELETE /drive/connect        log out of Drive, revoking the grant there.
- GET  /drive/files?path=      list a folder.
- POST /drive/import           bring a file in as a new job.
- PUT  /drive/files?path=&name=&overwrite=
                               save the request body as a file in Drive.
- POST /drive/save-original    save a recording's original in Drive,
                               decrypted on the way, never via the
                               frontend.

Errors carry {"error": <message fit to show>, "reason": <code>}. The
reasons the frontend acts on: "disabled" (hide Drive), "not_connected"
(connect again), "exists" (ask to overwrite).
"not_connected" is 409 rather than 401 on purpose: a 401 from this API
means the *Scribe* session is over, and the frontend treats it that way.
"""

import asyncio
import posixpath

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from auth.oidc import get_current_user
from db.customer import customer_get_from_user_id
from db.drive import (
    drive_complete,
    drive_get,
    drive_remove,
    drive_start,
    drive_touch,
)
from db.job import job_create, job_get, job_remove, job_update
from db.models import JobStatusEnum, JobType
from db.user import user_get, user_get_private_key, user_get_public_key
from utils import drive
from utils.crypto import (
    decrypt_data_from_file,
    load_private_key,
    deserialize_public_key_from_pem,
    encrypt_stream_to_file,
    encrypt_string,
)
from utils.log import get_logger
from utils.settings import get_settings
from routers.transcriber import decrypt_filename
from utils.recordings import original_path
from utils.validators import DriveImportRequest, DriveSaveOriginalRequest

router = APIRouter(tags=["drive"])
settings = get_settings()
log = get_logger()


def _error(message: str, status: int, reason: Optional[str] = None) -> JSONResponse:
    content = {"error": message}
    if reason:
        content["reason"] = reason
    return JSONResponse(content=content, status_code=status)


def _drive_error(error: drive.DriveError) -> JSONResponse:
    match error:
        case drive.DriveUnauthorized():
            return _error(
                "Your Drive connection has ended. Connect again.", 409, "not_connected"
            )
        case drive.DriveNotFound():
            return _error(str(error), 404, "not_found")
        case drive.DriveConflict():
            return _error(str(error), 409, "exists")
        case drive.TooLarge():
            return _error(str(error), 413, "too_large")
        case drive.DriveUnavailable():
            return _error(str(error), 502, "unavailable")
        case _:
            return _error(str(error), 400, "invalid")


async def _context(user: dict) -> dict | JSONResponse:
    """
    What Drive means for this user: whether it is offered, what it is
    called, and which instance it is -- their organisation's.

    Returns:
        dict: {"enabled", "display_name", "instance"}, or a JSONResponse
            refusing the request when Drive is not offered to them.
    """

    customer = await customer_get_from_user_id(user["user_id"]) or {}

    try:
        instance = drive.normalise_instance(customer.get("drive_url"))
    except drive.DriveError:
        instance = None

    if not customer.get("drive_enabled") or not instance:
        return _error(
            "Drive is not available for your organisation.", 403, "disabled"
        )

    return {
        "enabled": True,
        "display_name": customer.get("drive_display_name")
        or settings.DRIVE_DEFAULT_DISPLAY_NAME,
        "instance": instance,
    }


async def _connection(user: dict) -> tuple[dict, dict] | JSONResponse:
    """
    The user's live connection to the instance they use now, touched so it
    lasts another idle period.

    Returns:
        tuple[dict, dict]: (context, connection), or a refusal.
    """

    context = await _context(user)
    if isinstance(context, JSONResponse):
        return context

    connection = await drive_get(user["user_id"])

    # A connection made to an instance the organisation has since moved
    # away from does not count.
    if (
        not connection
        or not connection["connected"]
        or connection["instance"] != context["instance"]
    ):
        return _error("Connect to Drive first.", 409, "not_connected")

    await drive_touch(user["user_id"])

    return context, connection


async def _forget(user_id: str) -> None:
    """
    Drop a connection Drive no longer accepts.
    """

    await drive_remove(user_id)


async def _revoke(connection: Optional[dict]) -> None:
    if connection and connection.get("app_password"):
        await drive.revoke(
            connection["instance"],
            connection["login_name"],
            connection["app_password"],
        )


@router.get("/drive")
async def drive_status(user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Whether Drive is offered to this user, and in what state.
    """

    context = await _context(user)

    if isinstance(context, JSONResponse):
        return JSONResponse(
            content={
                "result": {
                    "enabled": False,
                    "display_name": settings.DRIVE_DEFAULT_DISPLAY_NAME,
                }
            }
        )

    connection = await drive_get(user["user_id"])
    current = bool(connection) and connection["instance"] == context["instance"]

    return JSONResponse(
        content={
            "result": {
                **context,
                "connected": current and connection["connected"],
                "pending": current and not connection["connected"],
            }
        }
    )


@router.post("/drive/connect")
async def drive_connect(user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Start signing in to Drive. The user opens the returned URL themselves,
    in a tab of its own: this is where they visibly leave Scribe and grant
    access in Drive.
    """

    context = await _context(user)
    if isinstance(context, JSONResponse):
        return context

    try:
        flow = await drive.login_start(context["instance"])
    except drive.DriveError as error:
        return _drive_error(error)

    replaced = await drive_start(
        user["user_id"], context["instance"], flow["endpoint"], flow["token"]
    )
    await _revoke(replaced)

    return JSONResponse(content={"result": {"login_url": flow["login"]}})


@router.get("/drive/connect")
async def drive_connect_poll(user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Ask Drive once whether the user has granted access. The frontend calls
    this every few seconds while the user is signing in.

    Answers {"state": "connected" | "pending" | "none"}.
    """

    context = await _context(user)
    if isinstance(context, JSONResponse):
        return context

    connection = await drive_get(user["user_id"])

    if not connection or connection["instance"] != context["instance"]:
        return JSONResponse(content={"result": {"state": "none"}})

    if connection["connected"]:
        return JSONResponse(content={"result": {"state": "connected"}})

    try:
        granted = await drive.login_poll(
            connection["instance"], connection["poll_endpoint"], connection["poll_token"]
        )

        if granted is None:
            return JSONResponse(content={"result": {"state": "pending"}})

        dav_user = await drive.user_id(
            connection["instance"], granted["login_name"], granted["app_password"]
        )
    except drive.DriveError as error:
        return _drive_error(error)

    if not await drive_complete(
        user["user_id"], granted["login_name"], granted["app_password"], dav_user
    ):
        # Another request completed it first, or the sign-in was replaced:
        # this grant is not wanted.
        await drive.revoke(
            connection["instance"], granted["login_name"], granted["app_password"]
        )
        current = await drive_get(user["user_id"])
        state = "connected" if current and current["connected"] else "none"
        return JSONResponse(content={"result": {"state": state}})

    log.info(f"User {user['user_id']} connected to Drive.")

    return JSONResponse(content={"result": {"state": "connected"}})


@router.delete("/drive/connect")
async def drive_disconnect(user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Log out of Drive: Scribe's access is revoked there as well, and the
    next Get from / Save to asks the user to sign in again.
    """

    await _revoke(await drive_remove(user["user_id"]))

    return JSONResponse(content={"result": {"state": "none"}})


@router.get("/drive/files")
async def drive_list(
    path: str = "", user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    The files and folders in a folder of the user's Drive, as the user sees
    them there.
    """

    found = await _connection(user)
    if isinstance(found, JSONResponse):
        return found

    _, connection = found

    try:
        path = drive.clean_path(path)
        entries = await drive.list_folder(connection, path)
    except drive.DriveUnauthorized as error:
        await _forget(user["user_id"])
        return _drive_error(error)
    except drive.DriveError as error:
        return _drive_error(error)

    for entry in entries:
        entry["media"] = not entry["is_dir"] and drive.is_media(entry["name"])

    return JSONResponse(content={"result": {"path": path, "entries": entries}})


@router.post("/drive/import")
async def drive_import(
    item: DriveImportRequest, user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Bring a file from the user's Drive into Scribe as a new job, exactly as
    if it had been uploaded -- encrypted to disk as it arrives, never held
    whole in memory and never passing through the user's own device.
    """

    found = await _connection(user)
    if isinstance(found, JSONResponse):
        return found

    _, connection = found

    try:
        path = drive.clean_path(item.path)
    except drive.DriveError as error:
        return _drive_error(error)

    name = posixpath.basename(path)

    if not name or not drive.is_media(name):
        return _error("Only audio and video files can be transcribed.", 400, "invalid")

    if not (api_user := await user_get(username="api_user")):
        return _error("API user not found", 500)

    user_public_key = deserialize_public_key_from_pem(
        await user_get_public_key(user["user_id"])
    )
    api_public_key = deserialize_public_key_from_pem(
        await user_get_public_key(api_user["user_id"])
    )

    job = await job_create(
        user_id=user["user_id"],
        job_type=JobType.TRANSCRIPTION,
        filename=encrypt_string(user_public_key, name),
    )

    folder = Path(settings.API_FILE_STORAGE_DIR) / user["user_id"]
    destination = folder / job["uuid"]

    async def sink(reader) -> None:
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
        await encrypt_stream_to_file(
            api_public_key,
            reader,
            str(destination),
            chunk_size=settings.CRYPTO_CHUNK_SIZE,
        )

    try:
        await drive.download(connection, path, sink, settings.RECORDING_MAX_BYTES)
    except drive.DriveError as error:
        await job_remove(job["uuid"])
        if isinstance(error, drive.DriveUnauthorized):
            await _forget(user["user_id"])
        return _drive_error(error)
    except Exception:
        log.exception("Could not bring a file in from Drive")
        await job_remove(job["uuid"])
        return _error("The file could not be brought in from Drive.", 500)

    job = await job_update(job["uuid"], status=JobStatusEnum.UPLOADED)

    log.info(f"User {user['user_id']} brought job {job['uuid']} in from Drive.")

    return JSONResponse(
        content={
            "result": {
                "uuid": job["uuid"],
                "status": job["status"],
                "job_type": job["job_type"],
                "filename": name,
            }
        }
    )


@router.put("/drive/files")
async def drive_save(
    request: Request,
    name: str,
    path: str = "",
    overwrite: bool = False,
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """
    Save the request body as `name` in the folder `path` of the user's
    Drive. Streamed through, never held whole, and refused past
    DRIVE_MAX_SAVE_BYTES. Without `overwrite`, an existing file is left
    alone and the answer is 409 "exists".

    Nothing in Scribe is deleted by saving: retention stays as it always is.
    """

    found = await _connection(user)
    if isinstance(found, JSONResponse):
        return found

    _, connection = found

    try:
        folder = drive.clean_path(path)
        name = drive.clean_name(name)
    except drive.DriveError as error:
        return _drive_error(error)

    limit = settings.DRIVE_MAX_SAVE_BYTES
    declared = request.headers.get("content-length")

    if declared and declared.isdigit() and int(declared) > limit:
        return _drive_error(drive.TooLarge("That file is too large to save to Drive."))

    async def body():
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise drive.TooLarge("That file is too large to save to Drive.")
            yield chunk

    target = f"{folder}/{name}" if folder else name

    try:
        await drive.upload(connection, target, body(), overwrite=overwrite)
    except drive.DriveError as error:
        if isinstance(error, drive.DriveUnauthorized):
            await _forget(user["user_id"])
        return _drive_error(error)

    log.info(f"User {user['user_id']} saved a file to Drive.")

    return JSONResponse(content={"result": {"path": target}})


async def _chunks(iterator):
    """
    A sync iterator of bytes -- decrypt_data_from_file() -- as an async
    one, each chunk decrypted off the event loop.
    """

    done = object()

    while (chunk := await asyncio.to_thread(next, iterator, done)) is not done:
        yield chunk


@router.post("/drive/save-original")
async def drive_save_original(
    item: DriveSaveOriginalRequest, user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Save the original of a recording made in the browser to the user's
    Drive: decrypted with their encryption password, exactly as a download
    of it is (POST /transcriber/{job_id}/original), and streamed from disk
    straight into a WebDAV PUT. It never passes through the frontend or
    the user's own device, and is never on disk in the clear.

    Answers 409 "exists" unless `overwrite`, like PUT /drive/files.
    """

    found = await _connection(user)
    if isinstance(found, JSONResponse):
        return found

    _, connection = found

    job = await job_get(item.job_id, user["user_id"])
    file_path = original_path(user["user_id"], item.job_id)

    if not job or not await asyncio.to_thread(file_path.exists):
        return _error("That recording has no original.", 404, "not_found")

    try:
        private_key = await load_private_key(
            await user_get_private_key(user["user_id"]),
            item.encryption_password or "",
        )
    except Exception:
        return _error("Wrong encryption password.", 403, "invalid")

    try:
        name = item.name
        if not name:
            name = (
                await asyncio.to_thread(decrypt_filename, dict(job), private_key)
            ).get("filename") or "Recording"
        folder = drive.clean_path(item.path)
        name = drive.clean_name(name)
    except drive.DriveError as error:
        return _drive_error(error)

    target = f"{folder}/{name}" if folder else name
    content = _chunks(decrypt_data_from_file(private_key, str(file_path)))

    try:
        await drive.upload(connection, target, content, overwrite=item.overwrite)
    except drive.DriveError as error:
        if isinstance(error, drive.DriveUnauthorized):
            await _forget(user["user_id"])
        return _drive_error(error)

    log.info(f"User {user['user_id']} saved the original of {item.job_id} to Drive.")

    return JSONResponse(content={"result": {"path": target}})
