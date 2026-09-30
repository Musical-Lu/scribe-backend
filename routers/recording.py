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
Recordings from the browser, sent in parts while they are recorded and
made into a transcription job when they end.  See utils/recordings.py.

Answers are shaped for the recorder's retry logic in the frontend:

- 2xx: done, move on.
- 409: parts are missing; the body lists them and the browser sends them.
- 422: can never succeed as sent.  The browser stops retrying.
- 503: try again later -- the disk is full, or another request is finishing
  this same recording right now.
"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from auth.oidc import get_current_user
from db.job import job_create, job_remove, job_update
from db.models import JobStatusEnum, JobType
from db.user import user_get, user_get_private_key, user_get_public_key
from utils.crypto import (
    load_private_key,
    deserialize_public_key_from_pem,
    encrypt_string,
)
from utils.log import get_logger
from utils.recordings import (
    DEFAULT_MIME,
    RECOVERED_NAME,
    Busy,
    RecordingError,
    file_name,
    original_path,
    recordings,
)
from utils.settings import get_settings
from utils.validators import RecordingFinishRequest

router = APIRouter(tags=["recording"])
settings = get_settings()
logger = get_logger()


def _refused(error: RecordingError) -> JSONResponse:
    return JSONResponse({"result": {"error": str(error)}}, status_code=422)


def _unavailable(reason: str) -> JSONResponse:
    return JSONResponse({"result": {"error": reason}}, status_code=503)


async def _api_user() -> dict:
    api_user = await user_get(username="api_user")

    if not api_user:
        raise RuntimeError("API user not found")

    return api_user


async def _public_key(user_id: str):
    return deserialize_public_key_from_pem(await user_get_public_key(user_id))


@router.get("/recordings")
async def recordings_unfinished(user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    The signed-in user's recordings that were never finished and have had
    nothing sent to them for RECORDING_RECOVER_AFTER_MINUTES: for finishing
    one whose own browser lost track of it (its storage wiped, its device
    gone).  Only the parts before the first gap can be joined; `parts` says
    how many that is, `held` how many are here at all.
    """

    try:
        found = await asyncio.to_thread(
            recordings.unfinished,
            user["user_id"],
            settings.RECORDING_RECOVER_AFTER_MINUTES * 60,
        )
    except RecordingError as error:
        return _refused(error)

    return JSONResponse({"recordings": found})


@router.get("/recordings/{rid}")
async def recording_status(
    rid: str, user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Which parts are held, and which job the recording became if it already
    has.  Asked before sending, so a browser that reloaded half way sends
    only what is missing.
    """

    try:
        parts = await asyncio.to_thread(recordings.parts, user["user_id"], rid)
        done = await asyncio.to_thread(recordings.done, user["user_id"], rid)
    except RecordingError as error:
        return _refused(error)

    return JSONResponse({"parts": parts, "done": done})


@router.put("/recordings/{rid}/part/{seq}")
async def recording_part(
    rid: str,
    seq: int,
    request: Request,
    user: dict = Depends(get_current_user),
    mime: str | None = Query(default=None, alias="type", max_length=100),
) -> JSONResponse:
    """
    One part of a recording, encrypted to disk as it arrives.  `type` is the
    audio type the browser records in, kept for finishing the recording
    from elsewhere.
    """

    declared = request.headers.get("content-length")

    if declared and declared.isdigit() and int(declared) > settings.RECORDING_MAX_PART_BYTES:
        return _refused(RecordingError("part too large"))

    try:
        public_key = await _public_key((await _api_user())["user_id"])
        await recordings.write_part(
            user["user_id"], rid, seq, request.stream(), public_key, mime
        )
    except RecordingError as error:
        return _refused(error)
    except OSError:
        # A full or read-only disk.  The browser still holds the part, so
        # this is "later", not "never".
        logger.exception("Could not store a recording part")
        return _unavailable("storage unavailable")

    return JSONResponse({"ok": True})


@router.delete("/recordings/{rid}")
async def recording_discard(
    rid: str, user: dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Throw away a recording that was never finished.  A recording that has
    become a job is deleted as a job.
    """

    try:
        await asyncio.to_thread(recordings.discard, user["user_id"], rid)
    except RecordingError as error:
        return _refused(error)

    return JSONResponse({"ok": True})


def _already(done: dict, count: int) -> JSONResponse:
    """
    The answer to finishing a recording that already is a job.  The same
    job again for a repeated finish -- the answer was lost, a second tab
    asked too -- but refused when this finish holds more than that job
    does: it was finished from elsewhere while this browser went on
    recording, and "done" would tell it to throw the rest away.
    """

    finished = done.get("parts")

    if isinstance(finished, int) and count > finished:
        return _refused(RecordingError("already finished"))

    return JSONResponse({"done": done})


@router.post("/recordings/{rid}/finish")
async def recording_finish(
    rid: str,
    item: RecordingFinishRequest,
    user: dict = Depends(get_current_user),
) -> JSONResponse:
    """
    Make a recording whose parts are all here into a transcription job, and
    keep its original for the user to download.

    Idempotent: a finish repeated after it succeeded -- the answer was lost,
    a second tab asked too -- is answered with the job the first one made.
    """

    user_id = user["user_id"]
    count = item.parts

    try:
        mime = item.mime or await asyncio.to_thread(recordings.mime, user_id, rid)
        name = file_name(
            item.name if item.mime else (item.name or RECOVERED_NAME),
            mime or DEFAULT_MIME,
        )
    except RecordingError as error:
        return _refused(error)

    try:
        if done := await asyncio.to_thread(recordings.done, user_id, rid):
            return _already(done, count)

        missing = await asyncio.to_thread(recordings.missing, user_id, rid, count)

        if missing:
            return JSONResponse({"missing": missing}, status_code=409)

        await asyncio.to_thread(recordings.claim, user_id, rid)
    except RecordingError as error:
        return _refused(error)
    except Busy:
        return _unavailable("already finishing")

    job = None

    try:
        # Asked again under the claim: another request may have finished
        # it between the check above and taking the claim.
        if done := await asyncio.to_thread(recordings.done, user_id, rid):
            return _already(done, count)

        api_user = await _api_user()
        api_private_key = await load_private_key(
            await user_get_private_key(api_user["user_id"]),
            settings.API_PRIVATE_KEY_PASSWORD,
        )
        api_public_key = await _public_key(api_user["user_id"])
        user_public_key = await _public_key(user_id)

        job = await job_create(
            user_id=user_id,
            job_type=JobType.TRANSCRIPTION,
            filename=encrypt_string(user_public_key, name),
        )

        await recordings.assemble(
            user_id,
            rid,
            count,
            api_private_key,
            [
                (api_public_key, Path(settings.API_FILE_STORAGE_DIR) / user_id / job["uuid"]),
                (user_public_key, original_path(user_id, job["uuid"])),
            ],
        )

        await job_update(job["uuid"], status=JobStatusEnum.UPLOADED)

        done = {"uuid": job["uuid"], "filename": name, "parts": count}
        await asyncio.to_thread(recordings.mark_done, user_id, rid, done)
    except RecordingError as error:
        if job:
            await job_remove(job["uuid"])
        return _refused(error)
    except Exception:
        logger.exception("Could not finish a recording")
        if job:
            await job_remove(job["uuid"])
        return _unavailable("could not finish the recording")
    finally:
        await asyncio.to_thread(recordings.release, user_id, rid)

    return JSONResponse({"done": done})
