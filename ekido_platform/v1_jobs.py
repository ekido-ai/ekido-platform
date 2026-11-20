"""`POST /v1/orgs/{org_id}/jobs`: start a training job from a paid order (tickets C7 and A14, tag jobs).

The customer has paid (the order is HELD). This route checks that the order is theirs, is paid and has no job yet, claims it so
two requests cannot both start one, asks the Coordinator to create and start the job, records that this team owns the job and
links the order to it. When the job ends the Coordinator settles or refunds the order (ekido-coordinator fiat_settlement).

The job's budget is the order's provider budget in rupees (paise / 100); the Coordinator only uses it as a ceiling and to
weigh providers against each other. If the Coordinator refuses, the claim is released and nothing is recorded, so the customer
can try again. A team and a member cannot start a job from another team's order: it looks missing, not forbidden.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from decimal import Decimal
from typing import Any, Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import accounts
from .accounts import AccountError, JobOwner, Principal
from .payments import HELD, PaymentOrder

# (method, path, JSON body) -> (status, parsed JSON); raises OSError when the Coordinator cannot be reached.
CoordinatorPost = Callable[[str, str, dict | None], tuple[int, Any]]


class StartJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=36)
    workload: dict[str, Any]
    policy: dict[str, Any]
    deadline_seconds: int = Field(default=86_400, gt=0, le=7 * 24 * 3600)


def coordinator_poster(base_url: str, token: str, timeout: float = 20.0) -> CoordinatorPost:
    """Calls the Coordinator's operator API with a JSON body. No redirects; replies are capped at 2 MB."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(method: str, path: str, body: dict | None) -> tuple[int, Any]:
        request = urllib.request.Request(base_url.rstrip("/") + path, method=method, data=json.dumps(body).encode() if body is not None else b"")
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with opener.open(request, timeout=timeout) as response:
                status, raw = response.status, response.read(2_000_001)
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read(2_000_001)
        if len(raw) > 2_000_000:
            raise OSError("Coordinator response too large")
        try:
            return status, json.loads(raw) if raw else None
        except ValueError:
            return status, None

    return call


def build_jobs_router(engine: Engine, principal: Callable, coordinator: CoordinatorPost) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def db():
        with Session(engine) as session:
            yield session

    def release(session: Session, order_id: str, claim: str) -> None:
        session.execute(update(PaymentOrder).where(PaymentOrder.id == order_id, PaymentOrder.job_id == claim).values(job_id=None))
        session.commit()

    @router.post("/orgs/{org_id}/jobs", status_code=201)
    def start_job(org_id: str, body: StartJob, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        order = session.get(PaymentOrder, body.order_id)
        if order is None or order.org_id != org_id:
            raise AccountError("NOT_FOUND", "No such order.", 404)
        if order.status != HELD:
            raise AccountError("NOT_PAID", f"This order is {order.status}, so no job can start from it.", 409)
        claim = "claim-" + uuid.uuid4().hex[:28]
        taken = session.execute(update(PaymentOrder).where(PaymentOrder.id == order.id, PaymentOrder.job_id.is_(None)).values(job_id=claim))
        session.commit()
        if taken.rowcount != 1:
            raise AccountError("ALREADY_STARTED", "A job has already been started from this order.", 409)
        budget = (Decimal(order.quote["provider_budget_paise"]) / 100).quantize(Decimal("0.000001"))
        try:
            status, created = coordinator("POST", "/jobs", {"workload": body.workload, "policy": body.policy, "budget_usdc": str(budget), "deadline_seconds": body.deadline_seconds})
            if status != 200 and status != 201 or not isinstance(created, dict) or not created.get("job_id"):
                raise AccountError("JOB_REFUSED", "The job could not be created: " + _reason(created), 422 if status in (400, 422) else 502)
            job_id = created["job_id"]
            status, started = coordinator("POST", f"/jobs/{job_id}/start", None)
            if status != 200:
                coordinator("POST", f"/jobs/{job_id}/cancel", None)  # best effort: do not leave a job that was never paid for to start later
                raise AccountError("JOB_REFUSED", "The job could not be started: " + _reason(started), 422 if status in (400, 409, 422) else 502)
        except OSError:
            release(session, order.id, claim)
            raise AccountError("UNAVAILABLE", "The training service is not reachable. Nothing was charged again; try again.", 503) from None
        except AccountError:
            release(session, order.id, claim)
            raise
        session.add(JobOwner(job_id=job_id, org_id=org_id))
        session.execute(update(PaymentOrder).where(PaymentOrder.id == order.id).values(job_id=job_id))
        session.commit()
        return {"job_id": job_id, "order_id": order.id, "state": (started or {}).get("state")}

    return router


def _reason(body: Any) -> str:
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail[:200] if isinstance(detail, str) else "the service refused it"
