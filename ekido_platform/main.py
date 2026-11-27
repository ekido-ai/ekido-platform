"""The Platform API service: everything a customer touches. Runs on 127.0.0.1:8010 behind Caddy, which sends it
`/sepolia/v1/*` (the public contract in docs/api/v1.yaml). The Coordinator keeps training, devices and settlement;
this service calls it over localhost when it needs a job started or read.

    uvicorn ekido_platform.main:create_app --factory --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from .accounts import AccountError
from .models import Base
from .v1_accounts import LogEmail, NoEmail, SesEmail, SmtpEmail, build_router, error

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite://"
    # Where sign-in links point, and how they're sent: "none", "log" (staging only), "ses" or "smtp".
    site_url: str = "http://localhost:3100"
    email_sender: str = "none"
    email_from: str = ""
    email_region: str = ""
    # EMAIL_SENDER=smtp: a STARTTLS server such as smtp.gmail.com:587. SMTP_PASSWORD comes from the environment only.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    session_cookie_secure: bool = True
    cors_origins: list[str] = ["http://localhost:3100", "http://127.0.0.1:3100"]
    # The Coordinator, reached over localhost. Its operator token is only needed when the two run on different hosts.
    coordinator_url: str = "http://127.0.0.1:8003"
    coordinator_token: str = ""
    inference_models_root: str = ""
    inference_internal_token: str = ""
    inference_rate_per_minute: int = 60
    # Operator-managed evidence and per-model manifests; empty disables savings reports.
    savings_evidence_root: str = ""
    assistant_engine_url: str = ""
    # Qwen3Guard's engine (loopback). Required: without it the assistant answers 503.
    assistant_guard_url: str = ""
    assistant_state_secret: str = ""
    # Messages one person may send the assistant per window (seconds). The defaults are the product's limit; staging raises them only to
    # run the evaluation sets, which send more than 20 messages in ten minutes.
    assistant_rate_limit: int = Field(default=20, ge=1, le=100_000)
    assistant_rate_window: int = Field(default=600, ge=1, le=86_400)
    assistant_refusal_limit: int = Field(default=5, ge=1, le=100_000)  # refusals per person per hour before the assistant pauses
    # Emails allowed into the owner's admin area (ticket C26). Empty keeps it closed to everyone.
    admin_emails: list[str] = []
    # Razorpay, test mode only (ticket C7). The key id is public; the secret and the webhook secret come from the environment
    # (AWS SSM on the servers) and are never in the repo. With any of the three empty, the webhook answers 503.
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""
    # The Razorpay plan ids for Starter and Business (created in the test-mode dashboard; ids are not secrets).
    razorpay_plan_starter: str = ""
    razorpay_plan_business: str = ""
    # Invoices (C8): who issues them, and whether they are still "TEST INVOICE: not a tax invoice" (until the company has a GSTIN).
    seller_name: str = "Ekido (test)"
    seller_gstin: str = ""
    invoice_test_mode: bool = True
    # Shared secret for the Coordinator-only routes under /internal (erasure tasks, settle and refund). Empty keeps them closed (404).
    internal_token: str = ""
    # Optional Valkey (or Redis) on this machine, for example redis://127.0.0.1:6379/0 (ticket #142). Empty: no cache. Never the
    # source of truth: if it is down, everything still works from Postgres.
    valkey_url: str = ""


def create_engine_for(database_url: str) -> Engine:
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(database_url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    return create_engine(database_url, pool_pre_ping=True)


def create_app(settings: Settings | None = None, engine: Engine | None = None, *, assistant_transport=None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or create_engine_for(settings.database_url)
    # SQLite (tests): create the tables. Postgres: the schema belongs to the migrations in apps/platform/alembic,
    # which the service unit runs before it starts.
    from . import payments, plans, privacy, v1_admin, v1_assistant, v1_datasets, v1_distill, v1_invoices  # noqa: F401  (register their tables before create_all)

    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)

    app = FastAPI(title="Ekido Platform API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
    app.state.engine = engine
    app.state.settings = settings

    if settings.email_sender == "ses":
        sender = SesEmail(settings.email_from, region=settings.email_region or None)
    elif settings.email_sender == "smtp":
        sender = SmtpEmail(settings.smtp_host, settings.smtp_port, settings.smtp_user, settings.smtp_password, settings.email_from)
    else:
        sender = {"log": LogEmail}.get(settings.email_sender, NoEmail)()
    from .cache import Cache

    cache = app.state.cache = Cache.from_url(settings.valkey_url)
    router, principal = build_router(engine, sender=sender, site_url=settings.site_url, allowed_origins=settings.cors_origins, cookie_secure=settings.session_cookie_secure, cache=cache)
    app.include_router(router)
    app.state.principal = principal

    from .v1_inference_authority import build_inference_authority
    app.include_router(build_inference_authority(engine, settings.inference_models_root,
                                                  settings.inference_internal_token, settings.inference_rate_per_minute))
    from .v1_savings import build_savings_router
    app.include_router(build_savings_router(engine, principal, settings.savings_evidence_root))
    app.include_router(v1_admin.build_admin_router(engine, principal, settings.admin_emails,
                                                   v1_admin.coordinator_client(settings.coordinator_url, settings.coordinator_token),
                                                   payment_rail=lambda: getattr(app.state, "payment_rail", None)))

    from .v1_assistant import SharedRateLimit, SharedRefusalBreaker, build_assistant_router
    app.include_router(build_assistant_router(engine, principal, settings.assistant_engine_url,
        settings.assistant_state_secret, guard_url=settings.assistant_guard_url, transport=assistant_transport,
        limiter=SharedRateLimit(cache, settings.assistant_rate_limit, settings.assistant_rate_window),
        breaker=SharedRefusalBreaker(cache, settings.assistant_refusal_limit)))

    from .payments import RazorpayHttpGateway, RazorpayRail
    from .v1_payments import build_payments_router
    rail = None
    if settings.razorpay_key_id and settings.razorpay_key_secret and settings.razorpay_webhook_secret:
        rail = RazorpayRail(RazorpayHttpGateway(settings.razorpay_key_id, settings.razorpay_key_secret), settings.razorpay_webhook_secret)
    if rail is not None:
        rail.cache = cache
    app.state.payment_rail = rail
    app.state.plans = None
    if rail is not None and settings.razorpay_plan_starter and settings.razorpay_plan_business:
        from .plans import PlansService

        app.state.plans = PlansService(rail.gateway, {"starter": settings.razorpay_plan_starter, "business": settings.razorpay_plan_business})
        rail.on_subscription_event = app.state.plans.sync_subscription
    app.include_router(build_payments_router(engine, rail))

    from .invoices import Party
    from .v1_invoices import build_invoices_router
    from .v1_internal import build_internal_router
    from .v1_privacy import build_privacy_router

    app.include_router(build_internal_router(engine, settings.internal_token, rail))
    from .v1_jobs import build_jobs_router, coordinator_poster

    app.state.coordinator_post = coordinator_poster(settings.coordinator_url, settings.coordinator_token)  # tests replace this
    app.include_router(build_jobs_router(engine, principal, lambda *a: app.state.coordinator_post(*a)))
    from .v1_distill import build_distill_router

    # Ticket 154 (Platform half): few examples, generated rows, the customer approves before training.
    app.include_router(build_distill_router(engine, principal, lambda *a: app.state.coordinator_post(*a), lambda: getattr(app.state, "payment_rail", None)))
    from .v1_customer_jobs import build_customer_jobs_router

    # Ticket C12: the customer's own jobs and models, allow-listed so no device identity ever reaches them.
    app.state.coordinator_get = v1_admin.coordinator_client(settings.coordinator_url, settings.coordinator_token)  # tests replace this
    app.include_router(build_customer_jobs_router(engine, principal, lambda: app.state.coordinator_get,
                                                  lambda: (settings.coordinator_url, settings.coordinator_token),
                                                  allowed_origins=settings.cors_origins, live_interval=lambda: getattr(app.state, "live_interval", 5.0)))
    from .v1_datasets import build_datasets_router

    # Ticket C9: recipes, datasets, quotes and orders. The rail and key id are read per request (tests set the rail).
    app.include_router(build_datasets_router(engine, principal, payment_rail=lambda: getattr(app.state, "payment_rail", None),
                                             razorpay_key_id=lambda: settings.razorpay_key_id if getattr(app.state, "payment_rail", None) else ""))

    app.include_router(build_invoices_router(engine, principal, seller=Party(settings.seller_name, (), settings.seller_gstin or None), test_mode=settings.invoice_test_mode))

    def export_payments(session, user_id, org_ids):
        from sqlalchemy import select

        rows = session.scalars(select(payments.PaymentOrder).where(payments.PaymentOrder.org_id.in_(org_ids or [""])).order_by(payments.PaymentOrder.created_at))
        return [{"order_id": o.id, "team": o.org_id, "amount_paise": o.amount_paise, "currency": o.currency, "status": o.status, "terms_hash": o.terms_hash, "created_at": o.created_at.isoformat()} for o in rows]

    def export_plans(session, user_id, org_ids):
        from sqlalchemy import select

        rows = session.scalars(select(plans.TeamPlan).where(plans.TeamPlan.org_id.in_(org_ids or [""])))
        return [{"team": r.org_id, "plan": r.plan, "status": r.status, "period_end": r.period_end.isoformat() if r.period_end else None} for r in rows]

    def erase_plans(session, org_id):
        """An erased team stops being billed: its subscription is cancelled at the end of the paid period, and its plan rows go.
        Payment orders and invoices stay (the law requires it) and hold no names or emails."""
        from sqlalchemy import delete, select

        row = session.scalar(select(plans.TeamPlan).where(plans.TeamPlan.org_id == org_id))
        if row is not None and row.status == plans.ACTIVE and row.subscription_id and app.state.plans is not None:
            app.state.plans.gateway.cancel_subscription(row.subscription_id, at_cycle_end=True)
        session.execute(delete(plans.PlanUsage).where(plans.PlanUsage.org_id == org_id))
        session.execute(delete(plans.TeamPlan).where(plans.TeamPlan.org_id == org_id))

    app.include_router(build_privacy_router(engine, principal, on_erased=cache.forget_callers, exporters={"payments": export_payments, "plans": export_plans}, org_erasers=[erase_plans, v1_distill.erase_distillations],
                                            job_checks=[privacy.refuse_while_jobs_run(lambda: app.state.coordinator_get)]))
    @app.exception_handler(AccountError)
    async def account_refused(_: Request, exc: AccountError):
        return error(exc)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "platform"}

    @app.get("/readyz")
    def readyz() -> Any:
        from fastapi.responses import JSONResponse

        try:
            with engine.connect() as connection:
                connection.execute(text("select 1"))
        except Exception:
            log.exception("database check failed")
            return JSONResponse({"ok": False, "checks": {"database": {"ok": False, "detail": "unreachable"}}}, status_code=503)
        return {"ok": True, "checks": {"database": {"ok": True, "detail": engine.dialect.name}}}

    return app
