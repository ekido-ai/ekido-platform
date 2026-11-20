"""The Razorpay webhook route (`POST /v1/payments/razorpay/webhook`, docs/api/v1.yaml, tag billing).

Razorpay calls it, so it has no cookie or API key: the HMAC signature on the raw body is the authentication. A bad
signature is refused with 401. A repeat of an event we already handled is acknowledged with 200 and does nothing. If
Razorpay can't be reached while we re-check a payment, we answer 502 and the event is not recorded, so Razorpay
retries it. Not configured (no secret) answers 503, so nothing is ever accepted unsigned.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .payments import PaymentError, RazorpayRail


def build_payments_router(engine: Engine, rail: RazorpayRail | None) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/payments/razorpay/webhook", tags=["billing"])
    async def razorpay_webhook(request: Request):
        if rail is None:
            return JSONResponse({"error": "NOT_CONFIGURED", "detail": "Payments are not set up on this server."}, status_code=503)
        raw = await request.body()
        with Session(engine) as session:
            try:
                outcome = rail.handle_webhook(session, raw, request.headers.get("X-Razorpay-Signature"), request.headers.get("X-Razorpay-Event-Id"))
            except PaymentError as exc:
                session.rollback()
                return JSONResponse({"error": exc.code, "detail": exc.detail}, status_code=exc.status)
        return {"ok": True, "outcome": outcome}

    return router
