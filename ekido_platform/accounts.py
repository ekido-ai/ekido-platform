"""Accounts for people who don't use a wallet: email sign-in, web sessions, teams (orgs) and API keys.

Part of the Platform API, in its own schema. A user's email lives in `user_emails`; a Coordinator job is linked to
a team through `job_owners`, which keeps the job's ID as a plain string (the job lives in the Coordinator).

Secrets are never stored: login links, sessions and API keys are random tokens kept only as SHA-256 hashes, and
looked up by that hash. They have 256 bits of entropy, so a fast hash is the right tool (no password stretching).
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from .models import Base, Timestamped, User, utcnow

LOGIN_TOKEN_TTL = timedelta(minutes=15)
SESSION_TTL = timedelta(days=30)
LOGIN_LINKS_PER_HOUR = 5
ROLES = ("owner", "admin", "member")
KEY_SCOPES = ("jobs", "models.predict", "models.read")
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,253}\.[^@\s]{2,}$")


class AccountError(ValueError):
    """A request that can't be done. `code` is the API error code, `status` the HTTP status, and `extra` any
    machine-readable fields the answer carries next to `error` and `detail` (for example `retry_after`)."""

    def __init__(self, code: str, detail: str, status: int = 400, extra: dict | None = None):
        super().__init__(detail)
        self.code, self.detail, self.status, self.extra = code, detail, status, extra or {}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _aware(moment: datetime) -> datetime:
    # SQLite returns naive datetimes even for timezone=True columns.
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _prefixed(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class UserEmail(Timestamped, Base):
    __tablename__ = "user_emails"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)  # Stored lowercase.
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LoginToken(Timestamped, Base):
    __tablename__ = "login_tokens"
    email: Mapped[str] = mapped_column(String(320), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WebSession(Timestamped, Base):
    __tablename__ = "web_sessions"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _prefixed("org"))
    name: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OrgMember(Timestamped, Base):
    __tablename__ = "org_members"
    __table_args__ = (UniqueConstraint("org_id", "user_id"),)
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    # When the person joined: at once for the team's creator, on their next sign-in for someone invited
    # (the invite email's link is a sign-in link). Null while the invite is pending (ekido-ai/ekido-core#104).
    joined_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _prefixed("key"))
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)
    name: Mapped[str] = mapped_column(String(60))
    mode: Mapped[str] = mapped_column(String(8))  # test or live
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    scopes: Mapped[list] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Consent(Timestamped, Base):
    __tablename__ = "consents"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(40))
    granted: Mapped[bool] = mapped_column(Boolean)
    policy_version: Mapped[str] = mapped_column(String(40), default="")


class JobOwner(Base):
    __tablename__ = "job_owners"
    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)  # The Coordinator's job ID.
    org_id: Mapped[str] = mapped_column(ForeignKey("orgs.id"), index=True)


class InferenceUsage(Base):
    """An invocation reservation; never contains prompts or responses."""
    __tablename__ = "inference_usage"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    key_id: Mapped[str] = mapped_column(ForeignKey("api_keys.id"), index=True)
    model_id: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


@dataclass(frozen=True)
class Principal:
    """Who is calling: a signed-in person (user_id set) or an API key (key_id and org_id set)."""

    via: str  # "session" or "api_key"
    user_id: str | None = None
    org_id: str | None = None
    key_id: str | None = None
    mode: str | None = None
    scopes: tuple[str, ...] = ()


def normalise_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL.match(email):
        raise AccountError("INVALID_EMAIL", "That doesn't look like an email address.", 422)
    return email


# ------------------------------------------------------------------ sign-in


def check_link_quota(session: Session, email: str, now: datetime | None = None, purpose: str = "sign_in") -> None:
    """Refuses when the address already had LOGIN_LINKS_PER_HOUR links in the last hour. The answer says how long to
    wait (`retry_after`, whole seconds, and `retry_at`), and, for an invite, talks about invites rather than sign-in
    links, since the person on the invite form never asked for a sign-in link."""
    now = now or utcnow()
    email = normalise_email(email)
    window = now - timedelta(hours=1)
    sent = session.scalars(
        select(LoginToken.created_at).where(LoginToken.email == email, LoginToken.created_at > window).order_by(LoginToken.created_at)
    ).all()
    if len(sent) < LOGIN_LINKS_PER_HOUR:
        return
    # The oldest link in the window drops out of it first; that's when one more can go.
    free_at = _aware(sent[len(sent) - LOGIN_LINKS_PER_HOUR]) + timedelta(hours=1)
    wait = max(60, int((free_at - now).total_seconds() + 59) // 60 * 60)
    minutes = wait // 60
    when = "in a minute" if minutes == 1 else f"in {minutes} minutes"
    extra = {"retry_after": wait, "retry_at": (now + timedelta(seconds=wait)).isoformat()}
    if purpose == "invite":
        raise AccountError(
            "TOO_MANY_INVITES",
            f"{email} has had {LOGIN_LINKS_PER_HOUR} emails from us in the last hour, so we're pausing for a bit to keep their inbox calm. "
            f"You can send the invite again {when}.",
            429,
            extra,
        )
    raise AccountError("TOO_MANY_LINKS", f"That's a lot of sign-in links for one hour. You can ask for a new one {when}.", 429, extra)


def start_login(session: Session, email: str, now: datetime | None = None, purpose: str = "sign_in") -> str:
    """Creates a one-time sign-in token for the address and returns it (to be emailed, never shown)."""
    now = now or utcnow()
    email = normalise_email(email)
    check_link_quota(session, email, now, purpose)
    token = secrets.token_urlsafe(32)
    session.add(LoginToken(email=email, token_hash=_hash(token), expires_at=now + LOGIN_TOKEN_TTL, created_at=now))
    session.commit()
    return token


def finish_login(session: Session, token: str, now: datetime | None = None) -> tuple[User, str]:
    """Uses a sign-in token once. Creates the user on first sign-in. Returns the user and a new session token."""
    now = now or utcnow()
    row = session.scalar(select(LoginToken).where(LoginToken.token_hash == _hash(token or "")))
    if row is None or row.used_at is not None or _aware(row.expires_at) <= now:
        raise AccountError("LINK_INVALID", "This sign-in link has expired or was already used. Ask for a new one.", 401)
    row.used_at = now
    address = session.scalar(select(UserEmail).where(UserEmail.email == row.email))
    if address is None:
        user = User()
        session.add(user)
        session.flush()
        address = UserEmail(user_id=user.id, email=row.email)
        session.add(address)
    user = session.get(User, address.user_id)
    address.verified_at = address.verified_at or now
    # Signing in is how an invited person joins: every pending invite of theirs becomes a membership now.
    for member in session.scalars(select(OrgMember).where(OrgMember.user_id == user.id, OrgMember.joined_at.is_(None))):
        member.joined_at = now
    session_token = secrets.token_urlsafe(32)
    session.add(WebSession(user_id=user.id, token_hash=_hash(session_token), expires_at=now + SESSION_TTL, created_at=now))
    session.commit()
    return user, session_token


def user_for_session(session: Session, token: str | None, now: datetime | None = None) -> str | None:
    if not token:
        return None
    now = now or utcnow()
    row = session.scalar(select(WebSession).where(WebSession.token_hash == _hash(token)))
    if row is None or row.revoked_at is not None or _aware(row.expires_at) <= now:
        return None
    return row.user_id


def end_session(session: Session, token: str | None) -> None:
    row = session.scalar(select(WebSession).where(WebSession.token_hash == _hash(token or "")))
    if row is not None and row.revoked_at is None:
        row.revoked_at = utcnow()
        session.commit()


# ------------------------------------------------------------------ teams


def role_in(session: Session, org_id: str, user_id: str) -> str | None:
    return session.scalar(select(OrgMember.role).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id))


def require_role(session: Session, principal: Principal, org_id: str, *allowed: str) -> None:
    """Members see their team; only the listed roles may change it. API keys act only within their own team."""
    if principal.via == "api_key":
        if principal.org_id != org_id:
            raise AccountError("NOT_FOUND", "No such team.", 404)
        if allowed and set(allowed) != set(ROLES):
            raise AccountError("FORBIDDEN", "API keys can't manage the team. Sign in instead.", 403)
        return
    role = role_in(session, org_id, principal.user_id or "")
    if role is None:
        raise AccountError("NOT_FOUND", "No such team.", 404)  # Don't reveal that it exists.
    if allowed and role not in allowed:
        raise AccountError("FORBIDDEN", f"Only {' or '.join(allowed)} can do this.", 403)


def create_org(session: Session, user_id: str, name: str) -> Org:
    name = (name or "").strip()
    if not 1 <= len(name) <= 80:
        raise AccountError("INVALID_NAME", "A team name needs 1 to 80 characters.", 422)
    org = Org(name=name)
    session.add(org)
    session.flush()
    session.add(OrgMember(org_id=org.id, user_id=user_id, role="owner", joined_at=utcnow()))
    session.commit()
    return org


def add_member(session: Session, org_id: str, email: str, role: str) -> str:
    """Adds a person by email, creating their (unverified) account if needed. Returns their user id.

    A new member is pending (`joined_at` empty) until they next sign in; inviting someone already in the team only
    changes their role."""
    if role not in ROLES:
        raise AccountError("INVALID_ROLE", f"Role must be one of {', '.join(ROLES)}.", 422)
    email = normalise_email(email)
    address = session.scalar(select(UserEmail).where(UserEmail.email == email))
    if address is None:
        user = User()
        session.add(user)
        session.flush()
        address = UserEmail(user_id=user.id, email=email)
        session.add(address)
    existing = session.scalar(select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == address.user_id))
    if existing is None:
        session.add(OrgMember(org_id=org_id, user_id=address.user_id, role=role))
    else:
        existing.role = role
    session.commit()
    return address.user_id


def remove_member(session: Session, org_id: str, user_id: str) -> None:
    member = session.scalar(select(OrgMember).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id))
    if member is None:
        raise AccountError("NOT_FOUND", "That person isn't in this team.", 404)
    if member.role == "owner":
        owners = session.scalar(select(func.count()).select_from(OrgMember).where(OrgMember.org_id == org_id, OrgMember.role == "owner"))
        if owners <= 1:
            raise AccountError("LAST_OWNER", "A team needs at least one owner. Make someone else an owner first.", 409)
    session.delete(member)
    session.commit()


def me(session: Session, user_id: str) -> dict:
    email = session.scalar(select(UserEmail.email).where(UserEmail.user_id == user_id))
    user = session.get(User, user_id)
    rows = session.execute(select(Org.id, Org.name, OrgMember.role).join(OrgMember, OrgMember.org_id == Org.id).where(OrgMember.user_id == user_id).order_by(Org.created_at))
    return {
        "user_id": user_id,
        "email": email,
        "name": user.display_name if user else None,
        "orgs": [{"org_id": org_id, "name": name, "role": role} for org_id, name, role in rows],
    }


def org_view(session: Session, org_id: str) -> dict:
    """The team and its members. `status` is "joined", or "invited" while the person hasn't signed in since the
    invite; `invited_at` and `joined_at` are ISO 8601 (or null)."""
    org = session.get(Org, org_id)
    members = session.execute(
        select(OrgMember.user_id, UserEmail.email, OrgMember.role, OrgMember.created_at, OrgMember.joined_at)
        .join(UserEmail, UserEmail.user_id == OrgMember.user_id, isouter=True)
        .where(OrgMember.org_id == org_id)
        .order_by(OrgMember.created_at)
    )
    return {
        "org_id": org.id,
        "name": org.name,
        "members": [
            {
                "user_id": u,
                "email": e,
                "role": r,
                "status": "joined" if joined else "invited",
                "invited_at": _aware(invited).isoformat() if invited else None,
                "joined_at": _aware(joined).isoformat() if joined else None,
            }
            for u, e, r, invited, joined in members
        ],
    }


def inviter_label(session: Session, user_id: str) -> str:
    """How an invite names the person who sent it: their display name, else their email."""
    user = session.get(User, user_id)
    if user is not None and user.display_name:
        return user.display_name
    return session.scalar(select(UserEmail.email).where(UserEmail.user_id == user_id)) or "Someone on your team"


# ------------------------------------------------------------------ API keys


def create_api_key(session: Session, org_id: str, user_id: str, name: str, mode: str, scopes: list[str] | None = None) -> tuple[ApiKey, str]:
    """Returns the stored key and its secret. The secret is shown once; only its hash is kept."""
    name = (name or "").strip()
    if not 1 <= len(name) <= 60:
        raise AccountError("INVALID_NAME", "A key name needs 1 to 60 characters.", 422)
    if mode not in ("test", "live"):
        raise AccountError("INVALID_MODE", "Mode must be test or live.", 422)
    scopes = list(dict.fromkeys(scopes or KEY_SCOPES))
    if not set(scopes) <= set(KEY_SCOPES):
        raise AccountError("INVALID_SCOPE", f"Scopes must be from {', '.join(KEY_SCOPES)}.", 422)
    secret = f"ek_{mode}_{secrets.token_urlsafe(30)}"
    key = ApiKey(org_id=org_id, name=name, mode=mode, prefix=secret[:12], key_hash=_hash(secret), scopes=scopes, created_by=user_id)
    session.add(key)
    session.commit()
    return key, secret


def key_view(key: ApiKey) -> dict:
    return {
        "key_id": key.id,
        "name": key.name,
        "mode": key.mode,
        "prefix": key.prefix,
        "scopes": list(key.scopes),
        "created_at": _aware(key.created_at).isoformat(),
        "last_used_at": _aware(key.last_used_at).isoformat() if key.last_used_at else None,
    }


def list_api_keys(session: Session, org_id: str) -> list[dict]:
    keys = session.scalars(select(ApiKey).where(ApiKey.org_id == org_id, ApiKey.revoked_at.is_(None)).order_by(ApiKey.created_at))
    return [key_view(k) for k in keys]


def revoke_api_key(session: Session, org_id: str, key_id: str) -> None:
    key = session.get(ApiKey, key_id)
    if key is None or key.org_id != org_id:
        raise AccountError("NOT_FOUND", "No such key.", 404)
    key.revoked_at = key.revoked_at or utcnow()
    session.commit()


def principal_for_key(session: Session, secret: str, now: datetime | None = None) -> Principal | None:
    if not secret.startswith(("ek_test_", "ek_live_")):
        return None
    key = session.scalar(select(ApiKey).where(ApiKey.key_hash == _hash(secret)))
    if key is None or key.revoked_at is not None:
        return None
    key.last_used_at = now or utcnow()
    session.commit()
    return Principal(via="api_key", org_id=key.org_id, key_id=key.id, mode=key.mode, scopes=tuple(key.scopes))
