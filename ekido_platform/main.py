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
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from .accounts import AccountError
from .models import Base
from .v1_accounts import LogEmail, NoEmail, SesEmail, build_router, error

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite://"
    # Where sign-in links point, and how they're sent: "none", "log" (staging only) or "ses".
    site_url: str = "http://localhost:3100"
    email_sender: str = "none"
    email_from: str = ""
    email_region: str = ""
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
    assistant_state_secret: str = ""
    assistant_datasets_root: str = ""
    # Emails allowed into the owner's admin area (ticket C26). Empty keeps it closed to everyone.
    admin_emails: list[str] = []


def create_engine_for(database_url: str) -> Engine:
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        return create_engine(database_url, poolclass=StaticPool, connect_args={"check_same_thread": False})
    return create_engine(database_url, pool_pre_ping=True)


def create_app(settings: Settings | None = None, engine: Engine | None = None, *, assistant_transport=None) -> FastAPI:
    settings = settings or Settings()
    engine = engine or create_engine_for(settings.database_url)
    # SQLite (tests): create the tables. Postgres: the schema belongs to the migrations in apps/platform/alembic,
    # which the service unit runs before it starts.
    from . import v1_admin  # noqa: F401  (registers the admin_actions table before create_all)

    if engine.dialect.name == "sqlite":
        Base.metadata.create_all(engine)

    app = FastAPI(title="Ekido Platform API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
    app.state.engine = engine
    app.state.settings = settings

    if settings.email_sender == "ses":
        sender = SesEmail(settings.email_from, region=settings.email_region or None)
    else:
        sender = {"log": LogEmail}.get(settings.email_sender, NoEmail)()
    router, principal = build_router(engine, sender=sender, site_url=settings.site_url, allowed_origins=settings.cors_origins, cookie_secure=settings.session_cookie_secure)
    app.include_router(router)
    app.state.principal = principal

    from .v1_inference_authority import build_inference_authority
    app.include_router(build_inference_authority(engine, settings.inference_models_root,
                                                  settings.inference_internal_token, settings.inference_rate_per_minute))
    from .v1_savings import build_savings_router
    app.include_router(build_savings_router(engine, principal, settings.savings_evidence_root))
    app.include_router(v1_admin.build_admin_router(engine, principal, settings.admin_emails,
                                                   v1_admin.coordinator_client(settings.coordinator_url, settings.coordinator_token)))

    from .v1_assistant import build_assistant_router
    app.include_router(build_assistant_router(engine, principal, settings.assistant_engine_url,
        settings.assistant_state_secret, settings.assistant_datasets_root, transport=assistant_transport))

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
