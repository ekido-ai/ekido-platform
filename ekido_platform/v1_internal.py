"""Internal routes for the Coordinator (and nobody else): carry out "delete my data" and settle a paid order.

Fail-closed: with no `INTERNAL_TOKEN` configured every route answers 404, so they do not announce themselves; a wrong token gets 404 too. The
token is compared in constant time. The Platform must stay on a private host (the Coordinator reaches it over localhost), and these routes
are not part of the public API contract.

- `GET /internal/erasure/tasks`, `POST /internal/erasure/tasks/{id}/complete`, `POST /internal/erasure/tasks/{id}/failed`: the Coordinator's
  erasure worker lists what to destroy for customers who deleted their data, reports each one done, or reports a failure (counted, so a task
  that keeps failing is visible; the 30-day deadline is watched through `privacy.overdue`).
- `POST /internal/payments/orders/{order_id}/job`, `POST /internal/payments/jobs/{job_id}/settle`, `.../refund`: when a job's quality check
  has decided, the Coordinator tells the Platform to record the payout split or to refund the customer in full. Settling and refunding are
  idempotent in `payments.py`, so the Coordinator can safely retry.
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import privacy
from .payments import PaymentError, PaymentOrder, RazorpayRail
from . import v1_distill


class LinkJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1, max_length=36)


class Settle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    payouts: dict[str, int]


def build_internal_router(engine: Engine, internal_token: str, rail: RazorpayRail | None) -> APIRouter:
    router = APIRouter(prefix="/internal")

    def db():
        with Session(engine) as session:
            yield session

    def guard(x_internal_token: str | None = Header(default=None)) -> None:
        if not internal_token or not x_internal_token or not hmac.compare_digest(x_internal_token.encode(), internal_token.encode()):
            raise HTTPException(status_code=404)  # disabled or wrong: indistinguishable from a route that does not exist

    def task_view(task) -> dict[str, Any]:
        return {"id": task.id, "kind": task.kind, "ref": task.ref, "attempts": task.attempts, "deletion_id": task.deletion_id}

    @router.get("/erasure/tasks", dependencies=[Depends(guard)])
    def tasks(limit: int = 100, session: Session = Depends(db)) -> list[dict[str, Any]]:
        return [task_view(t) for t in privacy.pending_tasks(session, min(max(limit, 1), 500))]

    @router.post("/erasure/tasks/{task_id}/complete", dependencies=[Depends(guard)])
    def complete(task_id: str, session: Session = Depends(db)) -> dict[str, Any]:
        request = privacy.complete_task(session, task_id)
        return {"task": task_id, "request_status": request.status}

    @router.post("/erasure/tasks/{task_id}/failed", dependencies=[Depends(guard)])
    def failed(task_id: str, session: Session = Depends(db)) -> dict[str, Any]:
        task = session.get(privacy.ErasureTask, task_id)
        if task is None:
            raise HTTPException(status_code=404)
        task.attempts += 1
        session.commit()
        return {"task": task_id, "attempts": task.attempts}

    def need_rail() -> RazorpayRail:
        if rail is None:
            raise HTTPException(status_code=503, detail="payments are not configured")
        return rail

    def order_for_job(session: Session, job_id: str) -> PaymentOrder:
        order = session.scalar(select(PaymentOrder).where(PaymentOrder.job_id == job_id))
        if order is None:
            raise HTTPException(status_code=404, detail="no paid order is linked to this job")
        return order

    def refuse(exc: PaymentError) -> HTTPException:
        return HTTPException(status_code=exc.status, detail={"error": exc.code, "detail": exc.detail})

    @router.post("/payments/orders/{order_id}/job", dependencies=[Depends(guard)])
    def link_job(order_id: str, body: LinkJob, session: Session = Depends(db)) -> dict[str, Any]:
        """Tie a paid order to the Coordinator job that was started from it. One order, one job; the same link again is a no-op."""
        order = session.get(PaymentOrder, order_id)
        if order is None:
            raise HTTPException(status_code=404)
        if order.status not in ("HELD", "SETTLED"):
            raise HTTPException(status_code=409, detail={"error": "NOT_PAID", "detail": f"this order is {order.status}, so no job may start from it"})
        if order.job_id not in (None, body.job_id):
            raise HTTPException(status_code=409, detail={"error": "ALREADY_LINKED", "detail": "this order already has a job"})
        taken = session.scalar(select(PaymentOrder).where(PaymentOrder.job_id == body.job_id, PaymentOrder.id != order.id))
        if taken is not None:
            raise HTTPException(status_code=409, detail={"error": "JOB_TAKEN", "detail": "this job is already linked to another order"})
        order.job_id = body.job_id
        session.commit()
        return {"order_id": order.id, "job_id": order.job_id}

    @router.post("/distillations/{distillation_id}/generated", dependencies=[Depends(guard)])
    def distillation_generated(distillation_id: str, body: v1_distill.Generated, session: Session = Depends(db)) -> dict[str, Any]:
        """The Coordinator's report that example generation finished, with the rows and the verifier's verdict. Idempotent."""
        row = v1_distill.accept_rows(session, distillation_id, body, rail)
        return {"distillation_id": row.id, "state": row.state}

    @router.post("/distillations/{distillation_id}/failed", dependencies=[Depends(guard)])
    def distillation_failed(distillation_id: str, body: v1_distill.GenerationFailed, session: Session = Depends(db)) -> dict[str, Any]:
        """The generation job failed (no device, timed out): the customer is refunded in full. Idempotent."""
        row = v1_distill.fail_generation(session, distillation_id, body.reason, rail)
        return {"distillation_id": row.id, "state": row.state}

    @router.get("/payments/jobs/{job_id}", dependencies=[Depends(guard)])
    def job_order(job_id: str, session: Session = Depends(db)) -> dict[str, Any]:
        """What the Coordinator needs to split a paid order: the status and the whole-paise budget for providers."""
        order = order_for_job(session, job_id)
        return {"order_id": order.id, "status": order.status, "provider_budget_paise": order.quote["provider_budget_paise"]}

    @router.post("/payments/jobs/{job_id}/settle", dependencies=[Depends(guard)])
    def settle(job_id: str, body: Settle, session: Session = Depends(db)) -> dict[str, Any]:
        order = order_for_job(session, job_id)
        try:
            done = need_rail().settle(session, order.id, body.payouts)
        except PaymentError as exc:
            raise refuse(exc) from None
        return {"order_id": done.id, "status": done.status}

    @router.post("/payments/jobs/{job_id}/refund", dependencies=[Depends(guard)])
    def refund(job_id: str, session: Session = Depends(db)) -> dict[str, Any]:
        order = order_for_job(session, job_id)
        try:
            done = need_rail().refund(session, order.id, "quality check failed")
        except PaymentError as exc:
            raise refuse(exc) from None
        return {"order_id": done.id, "status": done.status}

    return router
