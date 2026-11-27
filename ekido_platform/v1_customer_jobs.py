"""What a customer sees of their own jobs and models (ticket C12, contract tags jobs and models).

The Coordinator knows everything about a job, including which devices trained it. A customer must never learn that
(decision 8), so these views are built from an allow-list: only the fields below leave this module, whatever the
Coordinator returns. Devices are never named, listed or counted; recoveries are reported only as "the work moved".

A job is the team's if the Platform started it from the team's paid order (`job_owners`). Anything else looks missing.
"""

from __future__ import annotations

import asyncio
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ekido_core.recipes import RECIPES

from . import accounts
from .accounts import AccountError, JobOwner, Principal
from .payments import PaymentOrder

CoordinatorCall = Callable[[str, str], tuple[int, Any]]
MAX_MODEL_BYTES = 1024 * 1024 * 1024

STAGES = {
    "OPEN": "starting", "ASSIGNED": "starting",
    "RUNNING": "training", "RECOVERING": "training",
    "VALIDATING": "checking", "COMPLETED": "checking", "DISPUTED": "checking",
    "SETTLED": "passed", "REFUNDED": "refunded", "FAILED": "refunded", "CANCELLED": "cancelled",
}
MODEL_READY = {"COMPLETED", "SETTLED"}
DONE_STAGES = {"passed", "refunded", "cancelled"}
# The live stream (ticket #156). The close codes tell the page whether trying again makes sense.
LIVE_INTERVAL = 5.0  # seconds between looks at the Coordinator, the same as the page's old polling
LIVE_MAX_SECONDS = 1800  # then closed; the page connects again, which checks the sign-in again from the start
LIVE_PER_PERSON = 5
CLOSE_DONE, CLOSE_BAD_ORIGIN, CLOSE_SIGN_IN, CLOSE_NOT_FOUND, CLOSE_TOO_MANY = 1000, 1008, 4401, 4404, 4429
SESSION_COOKIE = "ekido_session"


def customer_view(job: dict, events: list[dict], order: PaymentOrder | None) -> dict[str, Any]:
    """The allow-listed view of one job. Every value is computed here; nothing is passed through from the Coordinator."""
    spec = job.get("spec") if isinstance(job.get("spec"), dict) else {}
    state = str(job.get("state", ""))
    steps_done, recoveries, final = 0, 0, None
    for event in events if isinstance(events, list) else []:
        kind, data = event.get("type"), event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind == "training.progress" and isinstance(data.get("step"), int):
            steps_done = max(steps_done, data["step"])
        elif kind == "checkpoint.accepted" and isinstance(data.get("logical_step"), int):
            steps_done = max(steps_done, data["logical_step"])
        elif kind == "recovery.step" and data.get("phase") in (None, "resumed", "RESUMED"):
            recoveries += 1
        elif kind == "quality.result" and data.get("is_final") and isinstance(data.get("accuracy"), (int, float)):
            final = data
    recipe = RECIPES.get(spec.get("recipe_id", ""))
    total = spec.get("total_logical_steps") if isinstance(spec.get("total_logical_steps"), int) else None
    quality = None
    if recipe is not None:
        quality = {"metric": recipe.metric, "floor": recipe.floor, "score": None, "passed": None}
        if final is not None:
            quality["score"] = round(float(final["accuracy"]), 4)
            quality["passed"] = str(final.get("verdict", "")).upper() in ("PASS", "ACCEPT", "ACCEPTED")
    return {
        "job_id": str(job.get("job_id", "")),
        "state": state,
        "stage": STAGES.get(state, "starting"),
        "recipe_id": recipe.recipe_id if recipe else None,
        "recipe_title": recipe.title if recipe else None,
        "base_model": (spec.get("model") or {}).get("id") if isinstance(spec.get("model"), dict) else None,
        "steps_done": min(steps_done, total) if total else steps_done,
        "total_steps": total,
        "moved_devices": recoveries,  # how many times the work moved to another device; never which ones
        "quality": quality,
        "model_ready": state in MODEL_READY,
        "payment": {"order_id": order.id, "status": order.status, "amount_paise": order.amount_paise} if order else None,
    }


def build_customer_jobs_router(engine: Engine, principal: Callable, coordinator: Callable[[], CoordinatorCall],
                               coordinator_url: Callable[[], tuple[str, str]], *, allowed_origins: list[str] | None = None,
                               live_interval: Callable[[], float] = lambda: LIVE_INTERVAL) -> APIRouter:
    router = APIRouter(prefix="/v1")
    origins = set(allowed_origins or [])
    watching: dict[str, int] = {}
    watching_lock = threading.Lock()

    def db():
        with Session(engine) as session:
            yield session

    def own(session: Session, who: Principal, job_id: str) -> JobOwner:
        owner = session.get(JobOwner, job_id)
        if owner is None:
            raise AccountError("NOT_FOUND", "No such job.", 404)
        accounts.require_role(session, who, owner.org_id, "owner", "admin", "member")  # another team's job looks missing
        return owner

    def read(path: str) -> Any:
        try:
            status, body = coordinator()("GET", path)
        except OSError:
            raise AccountError("UNAVAILABLE", "Training status isn't reachable right now. Try again in a minute.", 503) from None
        if status == 404:
            raise AccountError("NOT_FOUND", "No such job.", 404)
        if status >= 400:
            raise AccountError("UNAVAILABLE", "Training status isn't reachable right now. Try again in a minute.", 502)
        return body

    def view(session: Session, job_id: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(job_id, safe="")
        job, events = read(f"/jobs/{quoted}"), read(f"/jobs/{quoted}/timeline")
        order = session.scalar(select(PaymentOrder).where(PaymentOrder.job_id == job_id))
        return customer_view(job if isinstance(job, dict) else {}, events if isinstance(events, list) else [], order)

    @router.get("/orgs/{org_id}/jobs", tags=["jobs"])
    def team_jobs(org_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> list[dict[str, Any]]:
        accounts.require_role(session, who, org_id, "owner", "admin", "member")
        ids = list(session.scalars(select(JobOwner.job_id).where(JobOwner.org_id == org_id).limit(50)))
        out = []
        for job_id in ids:
            try:
                out.append(view(session, job_id))
            except AccountError as exc:
                if exc.status != 404:
                    raise
        return out

    @router.get("/jobs/{job_id}", tags=["jobs"])
    def one_job(job_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        own(session, who, job_id)
        return view(session, job_id)

    def watcher(token: str | None, job_id: str) -> str:
        """The signed-in person, if they may see this job right now. Raises AccountError otherwise."""
        with Session(engine) as session:
            user_id = accounts.user_for_session(session, token)
            if user_id is None:
                raise AccountError("SIGN_IN_REQUIRED", "Sign in.", 401)
            own(session, Principal(via="session", user_id=user_id), job_id)
            return user_id

    def look(job_id: str) -> dict[str, Any]:
        with Session(engine) as session:
            return view(session, job_id)

    def close_code(exc: AccountError) -> int:
        return CLOSE_SIGN_IN if exc.status == 401 else CLOSE_NOT_FOUND  # another team's job looks missing here too

    async def wait(socket: WebSocket, seconds: float) -> bool:
        """Sleep, but notice at once if the page goes away. Whatever the page sends is read and ignored. False: it left."""
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            try:
                message = await asyncio.wait_for(socket.receive(), left)
            except asyncio.TimeoutError:
                break
            if message["type"] == "websocket.disconnect":
                return False
        return True

    @router.websocket("/jobs/{job_id}/live")
    async def live(socket: WebSocket, job_id: str) -> None:
        """The same allow-listed view as `GET /v1/jobs/{id}`, sent when it changes. Session cookie only.

        A browser sends the cookie with a WebSocket from any site and CORS does not apply, so the Origin must be one of ours."""
        if socket.headers.get("origin") not in origins:
            await socket.close(CLOSE_BAD_ORIGIN)
            return
        token = socket.cookies.get(SESSION_COOKIE)
        await socket.accept()  # accepted first, so the page can read why it is closed
        try:
            user_id = await run_in_threadpool(watcher, token, job_id)
        except AccountError as exc:
            await socket.close(close_code(exc))
            return
        with watching_lock:
            if watching.get(user_id, 0) >= LIVE_PER_PERSON:
                over = True
            else:
                over, watching[user_id] = False, watching.get(user_id, 0) + 1
        if over:
            await socket.close(CLOSE_TOO_MANY)
            return
        try:
            sent, ends = None, time.monotonic() + LIVE_MAX_SECONDS
            while time.monotonic() < ends:
                try:
                    await run_in_threadpool(watcher, token, job_id)  # every time: a sign-out or a removal ends the stream
                except AccountError as exc:
                    await socket.close(close_code(exc))
                    return
                try:
                    message = {"type": "job", "job": await run_in_threadpool(look, job_id)}
                except AccountError as exc:
                    if exc.status == 404:
                        await socket.close(CLOSE_NOT_FOUND)
                        return
                    message = {"type": "unavailable"}  # never the Coordinator's own words
                if message != sent:
                    await socket.send_json(message)
                    sent = message
                if message.get("job", {}).get("stage") in DONE_STAGES:
                    break
                if not await wait(socket, live_interval()):
                    return
            await socket.close(CLOSE_DONE)
        except (WebSocketDisconnect, RuntimeError):
            pass  # the page went away while we were sending
        finally:
            with watching_lock:
                watching[user_id] -= 1
                if not watching[user_id]:
                    del watching[user_id]

    @router.get("/orgs/{org_id}/models", tags=["models"])
    def team_models(org_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> list[dict[str, Any]]:
        """A model is a job whose final model passed its check. Its id is the job's id."""
        return [{"model_id": j["job_id"], **j} for j in team_jobs(org_id, who, session) if j["model_ready"]]

    @router.get("/models/{model_id}/download", tags=["models"])
    def download(model_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> Response:
        """The trained model as a zip (weights, manifest and a script to try it), only once it has passed."""
        own(session, who, model_id)
        if not view(session, model_id)["model_ready"]:
            raise AccountError("NOT_READY", "This model hasn't passed its check yet, so there's nothing to download.", 409)
        base, token = coordinator_url()
        request = urllib.request.Request(base.rstrip("/") + f"/jobs/{urllib.parse.quote(model_id, safe='')}/model")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as response:
                data = response.read(MAX_MODEL_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise AccountError("NOT_READY" if error.code == 404 else "UNAVAILABLE", "The model file isn't available right now.", 409 if error.code == 404 else 502) from None
        except OSError:
            raise AccountError("UNAVAILABLE", "The model file isn't reachable right now. Try again in a minute.", 503) from None
        if len(data) > MAX_MODEL_BYTES:
            raise AccountError("TOO_LARGE", "This model is too large to download here.", 502)
        name = f"ekido-model-{model_id[:8]}.zip"
        return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})

    return router
