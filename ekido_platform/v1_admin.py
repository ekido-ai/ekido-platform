"""The owner's admin area (ticket C26): teams, jobs, payments and provider identities, plus cancel, with every change
written to an audit log.

Only a signed-in person whose email is in ADMIN_EMAILS gets in; everyone else, API keys included, gets 404, so the
routes don't announce themselves. Jobs, providers and settlements live in the Coordinator: this service reads them
server-side (same host, or with COORDINATOR_TOKEN), so the operator token never reaches a browser. Provider identities
are visible only through these routes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, DateTime, String, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from . import accounts
from .accounts import AccountError, ApiKey, Org, OrgMember, Principal, UserEmail
from .models import Base, utcnow

# (status, parsed JSON body); raises OSError when the Coordinator can't be reached.
CoordinatorCall = Callable[[str, str], tuple[int, Any]]


class AssistantSwitch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool


class AdminAction(Base):
    """One change made from the admin area: who, when, what and on what."""

    __tablename__ = "admin_actions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    user_id: Mapped[str] = mapped_column(String(36))
    email: Mapped[str] = mapped_column(String(320))
    action: Mapped[str] = mapped_column(String(40))
    target: Mapped[str] = mapped_column(String(120))
    outcome: Mapped[str] = mapped_column(String(16))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


def coordinator_client(base_url: str, token: str, timeout: float = 10.0) -> CoordinatorCall:
    """Calls the Coordinator's operator API. No redirects are followed; responses are capped at 2 MB."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def call(method: str, path: str) -> tuple[int, Any]:
        request = urllib.request.Request(base_url.rstrip("/") + path, method=method, data=b"" if method == "POST" else None)
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


def build_admin_router(engine: Engine, principal: Callable, admin_emails: list[str], coordinator: CoordinatorCall,
                       payment_rail: Callable[[], Any] = lambda: None) -> APIRouter:
    """`payment_rail` returns the Razorpay rail (or None when payments aren't configured); it's read per request."""
    router = APIRouter(prefix="/v1/admin")
    allowed = {accounts.normalise_email(e) for e in admin_emails if e.strip()}

    def db():
        with Session(engine) as session:
            yield session

    def admin(who: Principal = Depends(principal), session: Session = Depends(db)) -> tuple[str, str]:
        email = None
        if who.via == "session" and who.user_id:
            email = session.scalar(select(UserEmail.email).where(UserEmail.user_id == who.user_id))
        if not allowed or email is None or accounts.normalise_email(email) not in allowed:
            raise AccountError("NOT_FOUND", "Not found.", 404)
        return who.user_id, email

    def read(path: str) -> Any:
        try:
            status, body = coordinator("GET", path)
        except OSError:
            raise AccountError("COORDINATOR_UNAVAILABLE", "The Coordinator didn't answer.", 503) from None
        if status == 404:
            raise AccountError("NOT_FOUND", "Not found.", 404)
        if status >= 400:
            raise AccountError("COORDINATOR_ERROR", f"The Coordinator refused ({status}).", 502)
        return body

    def log(session: Session, who: tuple[str, str], action: str, target: str, outcome: str, detail: dict) -> None:
        session.add(AdminAction(id=str(uuid.uuid4()), at=utcnow(), user_id=who[0], email=who[1], action=action,
                                target=target, outcome=outcome, detail=detail))
        session.commit()

    @router.get("/orgs")
    def orgs(who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> list[dict[str, Any]]:
        members = dict(session.execute(select(OrgMember.org_id, func.count()).group_by(OrgMember.org_id)).all())
        keys = dict(session.execute(select(ApiKey.org_id, func.count()).where(ApiKey.revoked_at.is_(None)).group_by(ApiKey.org_id)).all())
        return [{"org_id": o.id, "name": o.name, "members": members.get(o.id, 0), "active_keys": keys.get(o.id, 0),
                 "created_at": o.created_at.isoformat() if o.created_at else None}
                for o in session.scalars(select(Org).order_by(Org.created_at.desc()))]

    @router.get("/orgs/{org_id}")
    def org(org_id: str, who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> dict[str, Any]:
        if session.get(Org, org_id) is None:
            raise AccountError("NOT_FOUND", "Not found.", 404)
        return {**accounts.org_view(session, org_id), "api_keys": accounts.list_api_keys(session, org_id)}

    @router.get("/jobs")
    def jobs(who: tuple[str, str] = Depends(admin)) -> Any:
        return read("/jobs?limit=100")

    @router.get("/jobs/{job_id}")
    def job(job_id: str, who: tuple[str, str] = Depends(admin)) -> Any:
        quoted = urllib.parse.quote(job_id, safe="")
        return {"job": read(f"/jobs/{quoted}"), "checkpoints": read(f"/jobs/{quoted}/checkpoints")}

    @router.get("/providers")
    def providers(who: tuple[str, str] = Depends(admin)) -> Any:
        return read("/providers")

    @router.get("/payments")
    def payments(who: tuple[str, str] = Depends(admin)) -> Any:
        return read("/settlements")

    @router.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str, who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> Any:
        quoted = urllib.parse.quote(job_id, safe="")
        try:
            status, body = coordinator("POST", f"/jobs/{quoted}/cancel")
        except OSError:
            log(session, who, "job.cancel", job_id, "unreachable", {})
            raise AccountError("COORDINATOR_UNAVAILABLE", "The Coordinator didn't answer; nothing was cancelled.", 503) from None
        outcome = "done" if status < 400 else "refused"
        log(session, who, "job.cancel", job_id, outcome, {"status": status, "state": body.get("state") if isinstance(body, dict) else None})
        if status == 404:
            raise AccountError("NOT_FOUND", "No such job.", 404)
        if status >= 400:
            detail = body.get("detail") if isinstance(body, dict) else None
            raise AccountError("CANCEL_REFUSED", str(detail or f"The Coordinator refused ({status})."), 409)
        return body

    @router.get("/orders")
    def orders(who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> list[dict[str, Any]]:
        """Paid and unpaid orders, newest first, with the team's name."""
        from .payments import PaymentOrder

        names = dict(session.execute(select(Org.id, Org.name)).all())
        rows = session.scalars(select(PaymentOrder).order_by(PaymentOrder.created_at.desc()).limit(200))
        return [{"order_id": o.id, "org_id": o.org_id, "team": names.get(o.org_id), "amount_paise": o.amount_paise, "currency": o.currency,
                 "status": o.status, "job_id": o.job_id, "created_at": o.created_at.isoformat() if o.created_at else None} for o in rows]

    @router.post("/orders/{order_id}/refund")
    def refund(order_id: str, who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> dict[str, Any]:
        """Refund a paid order in full (test mode: the rail refuses live keys). Only an order still held can be refunded;
        every attempt, refused or not, is written to the audit log."""
        from .payments import PaymentError, PaymentOrder

        rail = payment_rail()
        if rail is None:
            log(session, who, "order.refund", order_id, "unavailable", {})
            raise AccountError("PAYMENTS_UNAVAILABLE", "Payments aren't set up on this server, so nothing was refunded.", 503)
        if session.get(PaymentOrder, order_id) is None:
            raise AccountError("NOT_FOUND", "No such order.", 404)
        try:
            order = rail.refund(session, order_id, "refunded from the admin area")
        except PaymentError as exc:
            session.rollback()
            if exc.status >= 500:  # Razorpay unreachable or answered strangely: the order stays held, try again
                log(session, who, "order.refund", order_id, "unreachable", {"code": exc.code})
                raise AccountError("GATEWAY_UNAVAILABLE", "Razorpay didn't answer; nothing was refunded. Try again.", 503) from None
            log(session, who, "order.refund", order_id, "refused", {"code": exc.code})
            raise AccountError("REFUND_REFUSED", exc.detail, 409) from None
        except OSError:
            session.rollback()
            log(session, who, "order.refund", order_id, "unreachable", {})
            raise AccountError("GATEWAY_UNAVAILABLE", "Razorpay didn't answer; nothing was refunded. Try again.", 503) from None
        log(session, who, "order.refund", order_id, "done", {"amount_paise": order.amount_paise, "status": order.status})
        return {"order_id": order.id, "status": order.status, "amount_paise": order.amount_paise}

    @router.get("/assistant/teams")
    def assistant_teams(who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> list[dict[str, Any]]:
        """Every team and whether the assistant is switched on for it (off by default)."""
        from .v1_assistant import AssistantTeam

        enabled = dict(session.execute(select(AssistantTeam.org_id, AssistantTeam.enabled)).all())
        return [{"org_id": o.id, "team": o.name, "enabled": bool(enabled.get(o.id, False))}
                for o in session.scalars(select(Org).order_by(Org.created_at.desc()).limit(500))]

    @router.put("/assistant/teams/{org_id}")
    def assistant_switch(org_id: str, body: AssistantSwitch, who: tuple[str, str] = Depends(admin),
                         session: Session = Depends(db)) -> dict[str, Any]:
        """Switch the assistant on or off for one team. Every change is written to the audit log."""
        from .v1_assistant import AssistantTeam

        if session.get(Org, org_id) is None:
            raise AccountError("NOT_FOUND", "No such team.", 404)
        row = session.get(AssistantTeam, org_id)
        before = bool(row.enabled) if row is not None else False
        if row is None:
            session.add(AssistantTeam(org_id=org_id, enabled=body.enabled, updated_at=utcnow()))
        else:
            row.enabled, row.updated_at = body.enabled, utcnow()
        session.commit()
        log(session, who, "assistant.switch", org_id, "done", {"from": before, "to": body.enabled})
        return {"org_id": org_id, "enabled": body.enabled}

    @router.get("/assistant/unsupported")
    def assistant_unsupported(who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> list[dict[str, Any]]:
        """What people asked for that Ekido cannot train yet: the day and the kind of request and a count, never the text."""
        from .v1_assistant import AssistantUnsupported

        rows = session.scalars(select(AssistantUnsupported).order_by(AssistantUnsupported.day.desc(), AssistantUnsupported.kind).limit(200))
        return [{"day": r.day.isoformat(), "kind": r.kind, "count": r.count} for r in rows]

    @router.get("/audit")
    def audit(who: tuple[str, str] = Depends(admin), session: Session = Depends(db)) -> list[dict[str, Any]]:
        rows = session.scalars(select(AdminAction).order_by(AdminAction.at.desc()).limit(200))
        return [{"id": r.id, "at": r.at.isoformat(), "email": r.email, "action": r.action, "target": r.target,
                 "outcome": r.outcome, "detail": r.detail} for r in rows]

    return router
