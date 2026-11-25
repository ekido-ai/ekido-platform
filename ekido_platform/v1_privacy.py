"""The privacy routes from docs/api/v1.yaml (tag privacy): `GET /v1/me/data-export`, `DELETE /v1/me`, and `GET` and `POST`
`/v1/me/consents`. Signed-in people only (an API key cannot erase or export a person). Business rules are in privacy.py."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import accounts, privacy
from .accounts import AccountError, Principal

COOKIE = "ekido_session"


class ConsentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: str
    granted: StrictBool  # "yes" or 1 must not count as a grant: consent is evidence
    policy_version: str = ""


def build_privacy_router(engine: Engine, principal: Callable, *, exporters: dict | None = None, org_erasers: list | None = None, job_checks: list | None = None) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def db():
        with Session(engine) as session:
            yield session

    def signed_in(who: Principal = Depends(principal)) -> Principal:
        if who.via != "session":
            raise AccountError("SIGN_IN_REQUIRED", "This needs a signed-in person, not an API key.", 403)
        return who

    @router.get("/me/consents")
    def consents(who: Principal = Depends(signed_in), session: Session = Depends(db)) -> list[dict[str, Any]]:
        return privacy.list_consents(session, who.user_id)

    @router.post("/me/consents", status_code=201)
    def add_consent(body: ConsentBody, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> dict[str, Any]:
        return privacy.record_consent(session, who.user_id, body.purpose, body.granted, body.policy_version)

    @router.get("/me/data-export")
    def data_export(who: Principal = Depends(signed_in), session: Session = Depends(db)) -> Response:
        archive = privacy.export_zip(privacy.export_data(session, who.user_id, exporters))
        return Response(archive, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="ekido-my-data.zip"', "Cache-Control": "no-store"})

    @router.delete("/me", status_code=202)
    def delete_me(request: Request, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> JSONResponse:
        request_row = privacy.request_deletion(session, who.user_id, org_erasers, job_checks=job_checks)
        response = JSONResponse({"status": request_row.status, "due_by": privacy._iso(request_row.due_at)}, status_code=202)
        response.delete_cookie(COOKIE, path="/")  # every session of theirs is already gone
        return response

    return router
