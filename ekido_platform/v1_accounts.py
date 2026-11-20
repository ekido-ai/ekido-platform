"""The /v1 account routes: email sign-in, the session cookie, /v1/me, teams, members and API keys.

Contract: docs/api/v1.yaml (tags auth and orgs). These routes check identity themselves, so ApiGuard lets /v1/
through. A signed-in browser carries the `ekido_session` cookie (httpOnly, SameSite=Lax). The web app calls these
routes on its own host (Caddy forwards /sepolia/* there), so the cookie stays first-party. Developers send an API
key as `Authorization: Bearer ek_...`. A cookie-authenticated change must come from an allowed Origin, which stops
other sites from acting with a visitor's cookie.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import accounts
from .accounts import AccountError, Principal

COOKIE = "ekido_session"
log = logging.getLogger(__name__)


class EmailSender(Protocol):
    def send_login_link(self, email: str, link: str) -> None:
        """Delivers a one-time sign-in link to the address."""


class NoEmail:
    """No sender configured: sign-in links can't be delivered, so starting one fails clearly."""

    def send_login_link(self, email: str, link: str) -> None:
        raise AccountError("EMAIL_UNAVAILABLE", "Email sign-in isn't set up on this server yet.", 503)


class LogEmail:
    """Staging and development only: writes the link to the server log instead of sending it."""

    def send_login_link(self, email: str, link: str) -> None:
        log.warning("sign-in link for %s (EMAIL_SENDER=log, not for production): %s", email, link)


class SesEmail:
    """Amazon SES, using the instance role. The From address must be verified in SES."""

    def __init__(self, sender: str, region: str | None = None, client=None):
        if client is None:
            import boto3

            client = boto3.client("ses", region_name=region)
        self.client, self.sender = client, sender

    def send_login_link(self, email: str, link: str) -> None:
        try:
            self.client.send_email(
                Source=self.sender,
                Destination={"ToAddresses": [email]},
                Message={"Subject": {"Data": SUBJECT}, "Body": {"Text": {"Data": login_text(link)}}},
            )
        except Exception as error:  # botocore errors, e.g. MessageRejected for an unverified address in the SES sandbox
            raise not_sent(error) from None


class SmtpEmail:
    """Any SMTP server with STARTTLS (for example Gmail with an app password). The password comes from the environment."""

    def __init__(self, host: str, port: int, user: str, password: str, sender: str, smtp_class=None, timeout: float = 15.0):
        import smtplib

        self.smtp_class = smtp_class or smtplib.SMTP
        self.host, self.port, self.user, self.password, self.sender, self.timeout = host, port, user, password, sender, timeout

    def send_login_link(self, email: str, link: str) -> None:
        from email.message import EmailMessage

        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self.sender, email, SUBJECT
        message.set_content(login_text(link))
        try:
            with self.smtp_class(self.host, self.port, timeout=self.timeout) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(message)
        except Exception as error:  # smtplib and socket errors
            raise not_sent(error) from None


SUBJECT = "Your Ekido sign-in link"


def login_text(link: str) -> str:
    return f"Sign in to Ekido:\n\n{link}\n\nThe link works once, for 15 minutes. If you didn't ask for it, ignore this email."


def not_sent(error: Exception) -> AccountError:
    """A clear answer instead of a 500. Only the error's type is logged: never the link or the password."""
    log.warning("sign-in email not sent: %s", error.__class__.__name__)
    return AccountError("EMAIL_NOT_SENT", "We couldn't send the email. Check the address, or try again in a minute.", 503)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmailStart(Strict):
    email: str = Field(max_length=320)


class EmailVerify(Strict):
    token: str = Field(max_length=200)


class NewOrg(Strict):
    name: str = Field(max_length=80)


class NewMember(Strict):
    email: str = Field(max_length=320)
    role: str


class NewKey(Strict):
    name: str = Field(max_length=60)
    mode: str
    scopes: list[str] | None = None


def error(exc: AccountError) -> JSONResponse:
    return JSONResponse({"error": exc.code, "detail": exc.detail}, status_code=exc.status)


def build_router(engine: Engine, *, sender: EmailSender, site_url: str, allowed_origins: list[str], cookie_secure: bool = True) -> tuple[APIRouter, Callable]:
    """Returns the router and the `principal` dependency, so other /v1 routes can require a caller."""
    router = APIRouter(prefix="/v1")
    origins = set(allowed_origins)

    def db():
        with Session(engine) as session:
            yield session

    def principal(request: Request, session: Session = Depends(db)) -> Principal:
        scheme, _, presented = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer" and presented.startswith("ek_"):
            found = accounts.principal_for_key(session, presented)
            if found is None:
                raise AccountError("INVALID_KEY", "This API key is invalid or revoked.", 401)
            return found
        user_id = accounts.user_for_session(session, request.cookies.get(COOKIE))
        if user_id is None:
            raise AccountError("SIGN_IN_REQUIRED", "Sign in, or send an API key.", 401)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin is not None and origin not in origins:
            raise AccountError("BAD_ORIGIN", "This request came from a site that isn't allowed to act for you.", 403)
        return Principal(via="session", user_id=user_id)

    def signed_in(who: Principal = Depends(principal)) -> Principal:
        if who.via != "session":
            raise AccountError("SIGN_IN_REQUIRED", "This needs a signed-in person, not an API key.", 403)
        return who

    @router.post("/auth/email/start", status_code=202)
    def email_start(body: EmailStart, session: Session = Depends(db)) -> dict[str, Any]:
        token = accounts.start_login(session, body.email)
        sender.send_login_link(accounts.normalise_email(body.email), f"{site_url.rstrip('/')}/login/verify?token={token}")
        # Same answer whether or not an account exists, so the endpoint can't be used to probe for users.
        return {"sent": True}

    @router.post("/auth/email/verify")
    def email_verify(body: EmailVerify, response: Response, session: Session = Depends(db)) -> dict[str, Any]:
        user, token = accounts.finish_login(session, body.token)
        response.set_cookie(COOKIE, token, max_age=int(accounts.SESSION_TTL.total_seconds()), httponly=True, secure=cookie_secure, samesite="lax", path="/")
        return accounts.me(session, user.id)

    @router.post("/auth/logout", status_code=204)
    def logout(request: Request, response: Response, session: Session = Depends(db)) -> Response:
        accounts.end_session(session, request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        response.status_code = 204
        return response

    @router.get("/me")
    def get_me(who: Principal = Depends(signed_in), session: Session = Depends(db)) -> dict[str, Any]:
        return accounts.me(session, who.user_id)

    @router.post("/orgs", status_code=201)
    def new_org(body: NewOrg, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> dict[str, Any]:
        org = accounts.create_org(session, who.user_id, body.name)
        return accounts.org_view(session, org.id)

    @router.get("/orgs/{org_id}")
    def get_org(org_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, *accounts.ROLES)
        return accounts.org_view(session, org_id)

    @router.post("/orgs/{org_id}/members", status_code=202)
    def invite(org_id: str, body: NewMember, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        if body.role == "owner" and accounts.role_in(session, org_id, who.user_id) != "owner":
            raise AccountError("FORBIDDEN", "Only an owner can make someone an owner.", 403)
        user_id = accounts.add_member(session, org_id, body.email, body.role)
        token = accounts.start_login(session, body.email)
        sender.send_login_link(accounts.normalise_email(body.email), f"{site_url.rstrip('/')}/login/verify?token={token}")
        return {"user_id": user_id, "invited": True}

    @router.delete("/orgs/{org_id}/members/{user_id}", status_code=204)
    def remove(org_id: str, user_id: str, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> Response:
        accounts.require_role(session, who, org_id, "owner", "admin")
        accounts.remove_member(session, org_id, user_id)
        return Response(status_code=204)

    @router.get("/orgs/{org_id}/api-keys")
    def keys(org_id: str, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> list[dict[str, Any]]:
        accounts.require_role(session, who, org_id, *accounts.ROLES)
        return accounts.list_api_keys(session, org_id)

    @router.post("/orgs/{org_id}/api-keys", status_code=201)
    def new_key(org_id: str, body: NewKey, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        key, secret = accounts.create_api_key(session, org_id, who.user_id, body.name, body.mode, body.scopes)
        return {**accounts.key_view(key), "secret": secret}

    @router.delete("/orgs/{org_id}/api-keys/{key_id}", status_code=204)
    def revoke(org_id: str, key_id: str, who: Principal = Depends(signed_in), session: Session = Depends(db)) -> Response:
        accounts.require_role(session, who, org_id, "owner", "admin")
        accounts.revoke_api_key(session, org_id, key_id)
        return Response(status_code=204)

    return router, principal
