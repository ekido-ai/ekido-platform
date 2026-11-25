"""India's DPDP Act, in the Platform: consent records, export my data, delete my data (ticket C17).

- **Consent** is an append-only history: every grant or withdrawal is a new row (purpose, yes or no, policy version, when).
  What a person agreed to *now* is the latest row per purpose.
- **Export** builds a ZIP of JSON files with everything this service holds about the person: profile, email, teams and roles,
  the API keys they made (never a key or its hash), consents, and the ids of their teams' jobs. It never lists other
  people's emails. Other modules can add files through `exporters`.
- **Delete** erases the person at once and records it. Their emails, sessions, sign-in links, consents and memberships are
  deleted. A team they were the only member of is erased too: its keys are removed and every one of its jobs gets an
  *erasure task* so the Coordinator and the storage can destroy the datasets, checkpoints, models and the job key (A14). A person
  who is the last owner of a team that still has other members must hand it over first (409). Payment and invoice records are
  kept as the law requires and are not touched here; they hold no name or email.
  Other modules clear their own rows for an erased team through `org_erasers`.

The user row stays as an empty tombstone (no name, no email), because other rows, such as the API keys a person made in
someone else's team, point at its id. It can no longer be signed in to or linked to a person.

The platform part of a deletion is immediate; the request stays SCHEDULED until every erasure task is done, and is due
within 30 days of the request (`overdue` lists any that are late).
"""

from __future__ import annotations

import io
import json
import urllib.parse
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, delete, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from .accounts import AccountError, ApiKey, Consent, InferenceUsage, JobOwner, LoginToken, Org, OrgMember, UserEmail, WebSession
from .models import Base, Timestamped, User, new_id, utcnow

PURPOSES = ("account", "training-data-processing", "marketing-email")
ERASURE_WINDOW = timedelta(days=30)
SCHEDULED, DONE = "SCHEDULED", "DONE"
PENDING = "PENDING"


class DeletionRequest(Timestamped, Base):
    """The record that a deletion was asked for and when it was finished. Holds only the opaque user id, no personal data."""

    __tablename__ = "deletion_requests"
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(12), default=SCHEDULED)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class ErasureTask(Timestamped, Base):
    """One thing another service must destroy: for a job of an erased team, its datasets, checkpoints, models and job key."""

    __tablename__ = "erasure_tasks"
    deletion_id: Mapped[str] = mapped_column(ForeignKey("deletion_requests.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24), default="job_data")
    ref: Mapped[str] = mapped_column(String(64))  # the job id
    status: Mapped[str] = mapped_column(String(12), default=PENDING)
    done_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, default=0)


Exporter = Callable[[Session, str, list[str]], Any]  # (session, user_id, the user's org ids) -> JSON-able
OrgEraser = Callable[[Session, str], None]  # (session, org_id): clear this module's rows for an erased team


def _aware(moment: datetime | None) -> datetime | None:
    return moment if moment is None or moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _iso(moment: datetime | None) -> str | None:
    moment = _aware(moment)
    return moment.isoformat() if moment else None


# ---------------------------------------------------------------------------------------------------- consent


def record_consent(session: Session, user_id: str, purpose: str, granted: bool, policy_version: str = "", now: datetime | None = None) -> dict:
    if purpose not in PURPOSES:
        raise AccountError("INVALID_PURPOSE", f"purpose must be one of {', '.join(PURPOSES)}.", 422)
    if type(granted) is not bool:
        raise AccountError("INVALID_CONSENT", "granted must be true or false.", 422)
    if not isinstance(policy_version, str) or len(policy_version) > 40:
        raise AccountError("INVALID_VERSION", "policy_version must be text of up to 40 characters.", 422)
    stamp = now or utcnow()
    last = _aware(session.scalar(select(func.max(Consent.created_at)).where(Consent.user_id == user_id)))
    if last is not None and stamp <= last:
        stamp = last + timedelta(microseconds=1)  # the history stays in the order things happened, even if the clock repeats
    row = Consent(user_id=user_id, purpose=purpose, granted=granted, policy_version=policy_version, created_at=stamp)
    session.add(row)
    session.commit()
    return _consent_view(row)


def _consent_view(row: Consent) -> dict:
    return {"purpose": row.purpose, "granted": row.granted, "policy_version": row.policy_version, "recorded_at": _iso(row.created_at)}


def list_consents(session: Session, user_id: str) -> list[dict]:
    """The whole history, oldest first, so what was agreed and when can always be shown."""
    rows = session.scalars(select(Consent).where(Consent.user_id == user_id).order_by(Consent.created_at, Consent.id))
    return [_consent_view(r) for r in rows]


def current_consents(session: Session, user_id: str) -> dict[str, bool]:
    """The latest answer per purpose. A purpose never answered is absent, which means no consent."""
    latest: dict[str, bool] = {}
    for row in session.scalars(select(Consent).where(Consent.user_id == user_id).order_by(Consent.created_at, Consent.id)):
        latest[row.purpose] = row.granted
    return latest


# ---------------------------------------------------------------------------------------------------- export


def _org_ids(session: Session, user_id: str) -> list[str]:
    return list(session.scalars(select(OrgMember.org_id).where(OrgMember.user_id == user_id).order_by(OrgMember.created_at)))


def export_data(session: Session, user_id: str, exporters: dict[str, Exporter] | None = None) -> dict[str, Any]:
    """File name to JSON-able content. No secrets (keys, hashes, tokens) and no other person's email or name."""
    user = session.get(User, user_id)
    if user is None:
        raise AccountError("NOT_FOUND", "No such account.", 404)
    org_ids = _org_ids(session, user_id)
    orgs = []
    for org_id in org_ids:
        org = session.get(Org, org_id)
        role = session.scalar(select(OrgMember.role).where(OrgMember.org_id == org_id, OrgMember.user_id == user_id))
        members = session.scalar(select(func.count()).select_from(OrgMember).where(OrgMember.org_id == org_id))
        orgs.append({"org_id": org_id, "name": org.name, "your_role": role, "members": members, "created_at": _iso(org.created_at)})
    keys = session.scalars(select(ApiKey).where(ApiKey.created_by == user_id).order_by(ApiKey.created_at))
    files: dict[str, Any] = {
        "profile.json": {"user_id": user.id, "display_name": user.display_name, "created_at": _iso(user.created_at)},
        "emails.json": [{"email": e.email, "verified_at": _iso(e.verified_at)} for e in session.scalars(select(UserEmail).where(UserEmail.user_id == user_id))],
        "teams.json": orgs,
        "api_keys.json": [{"key_id": k.id, "team": k.org_id, "name": k.name, "mode": k.mode, "prefix": k.prefix, "scopes": k.scopes, "created_at": _iso(k.created_at),
                           "last_used_at": _iso(k.last_used_at), "revoked_at": _iso(k.revoked_at)} for k in keys],
        "sessions.json": [{"created_at": _iso(s.created_at), "expires_at": _iso(s.expires_at), "revoked_at": _iso(s.revoked_at)} for s in session.scalars(select(WebSession).where(WebSession.user_id == user_id).order_by(WebSession.created_at))],
        "consents.json": list_consents(session, user_id),
        "jobs.json": [{"job_id": j.job_id, "team": j.org_id} for j in session.scalars(select(JobOwner).where(JobOwner.org_id.in_(org_ids or [""])).order_by(JobOwner.job_id))],
        "deletion_requests.json": [{"requested_at": _iso(d.created_at), "due_at": _iso(d.due_at), "status": d.status} for d in session.scalars(select(DeletionRequest).where(DeletionRequest.user_id == user_id))],
    }
    for name, exporter in (exporters or {}).items():
        files[name if name.endswith(".json") else f"{name}.json"] = exporter(session, user_id, org_ids)
    return files


README = """This is everything the Ekido Platform holds about you (India's DPDP Act, the right to access).

Each .json file is one topic. Your datasets, checkpoints and trained models are held by Ekido's training and storage services,
not by the Platform: jobs.json lists the ids of your teams' jobs, and a model can be downloaded from its job page.
Keys, passwords, sign-in links and other people's email addresses are never included.
Questions: see the privacy policy for the grievance officer's contact.
"""


def export_zip(files: dict[str, Any]) -> bytes:
    """The export as a ZIP of pretty-printed JSON files plus a README. Fixed timestamps, so the same data gives the same bytes."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in {"README.txt": README, **{n: json.dumps(c, indent=2, ensure_ascii=False) + "\n" for n, c in files.items()}}.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content.encode("utf-8"))
    return out.getvalue()


# ---------------------------------------------------------------------------------------------------- delete


ENDED_JOB_STATES = frozenset({"SETTLED", "FAILED", "CANCELLED", "REFUNDED"})  # the Coordinator's own list of jobs that are over


def refuse_while_jobs_run(coordinator: Callable[[], Any]) -> Callable[[list[str]], None]:
    """A check for `request_deletion`: a team whose job is still running or being checked cannot be erased yet.

    Erasing the team schedules its data for destruction, and the Coordinator destroys a job's key when told to, which would break the run
    (the worker waits for the key; the verifier needs it). So the person is told to cancel the job or wait until it ends, and nothing is
    deleted. Every job counts as running unless the Coordinator says it ended; a job the Coordinator does not know (404) has nothing to
    protect. If the Coordinator cannot be asked, the deletion is refused for now (503), because destroying data mid-job cannot be undone
    and asking again later is cheap.
    """

    def check(job_ids: list[str]) -> None:
        running = 0
        for job_id in job_ids:
            try:
                status, body = coordinator()("GET", f"/jobs/{urllib.parse.quote(job_id, safe='')}")
            except OSError:
                raise AccountError("JOB_CHECK_UNAVAILABLE", "We could not check whether your team has a job running. Nothing was deleted; please try again in a few minutes.", 503) from None
            if status == 404:
                continue
            if status != 200 or not isinstance(body, dict):
                raise AccountError("JOB_CHECK_UNAVAILABLE", "We could not check whether your team has a job running. Nothing was deleted; please try again in a few minutes.", 503)
            if str(body.get("state", "")) not in ENDED_JOB_STATES:
                running += 1
        if running:
            raise AccountError("JOB_RUNNING", f"Your team has {running} job{'s' if running != 1 else ''} still running or being checked. Cancel {'them' if running != 1 else 'it'} or wait until {'they end' if running != 1 else 'it ends'}, then ask again. Nothing was deleted.", 409)

    return check


def request_deletion(session: Session, user_id: str, org_erasers: list[OrgEraser] | None = None, now: datetime | None = None,
                     job_checks: list[Callable[[list[str]], None]] | None = None) -> DeletionRequest:
    """Erase a person now and schedule the rest. All or nothing: if anything refuses or fails, nothing is deleted.

    `job_checks` run first, with the ids of the jobs of every team that would be erased; one of them raises to refuse (see `refuse_while_jobs_run`)."""
    now = now or utcnow()
    user = session.get(User, user_id)
    if user is None:
        raise AccountError("NOT_FOUND", "No such account.", 404)
    already = session.scalar(select(DeletionRequest).where(DeletionRequest.user_id == user_id))
    if already is not None:
        return already  # asking twice does not erase twice
    memberships = list(session.scalars(select(OrgMember).where(OrgMember.user_id == user_id)))
    erase_orgs: list[str] = []
    for member in memberships:
        others = session.scalar(select(func.count()).select_from(OrgMember).where(OrgMember.org_id == member.org_id, OrgMember.user_id != user_id))
        if others == 0:
            erase_orgs.append(member.org_id)  # their team has nobody else: it goes with them
        elif member.role == "owner":
            other_owners = session.scalar(select(func.count()).select_from(OrgMember).where(OrgMember.org_id == member.org_id, OrgMember.role == "owner", OrgMember.user_id != user_id))
            if other_owners == 0:
                raise AccountError("OWNER_OF_SHARED_TEAM", "You are the only owner of a team that has other people. Make someone else an owner, or remove them, first.", 409)
    if erase_orgs and job_checks:
        job_ids = list(session.scalars(select(JobOwner.job_id).where(JobOwner.org_id.in_(erase_orgs)).order_by(JobOwner.job_id)))
        for check in job_checks:
            check(job_ids)  # raises AccountError to refuse; nothing has been changed yet
    try:
        request = DeletionRequest(user_id=user_id, due_at=now + ERASURE_WINDOW, status=SCHEDULED, created_at=now)
        session.add(request)
        session.flush()
        tasks = 0
        for org_id in erase_orgs:
            for owner in session.scalars(select(JobOwner).where(JobOwner.org_id == org_id).order_by(JobOwner.job_id)):
                session.add(ErasureTask(deletion_id=request.id, kind="job_data", ref=owner.job_id, created_at=now))
                tasks += 1
            session.execute(delete(JobOwner).where(JobOwner.org_id == org_id))
            key_ids = list(session.scalars(select(ApiKey.id).where(ApiKey.org_id == org_id)))
            if key_ids:
                session.execute(delete(InferenceUsage).where(InferenceUsage.key_id.in_(key_ids)))
            session.execute(delete(ApiKey).where(ApiKey.org_id == org_id))
            session.execute(delete(OrgMember).where(OrgMember.org_id == org_id))
            for eraser in org_erasers or []:
                eraser(session, org_id)  # other modules clear their rows (plans, usage...) or raise to stop everything
            session.execute(delete(Org).where(Org.id == org_id))
        session.execute(delete(OrgMember).where(OrgMember.user_id == user_id))  # the shared teams they leave
        emails = list(session.scalars(select(UserEmail.email).where(UserEmail.user_id == user_id)))
        if emails:
            session.execute(delete(LoginToken).where(LoginToken.email.in_(emails)))
        session.execute(delete(UserEmail).where(UserEmail.user_id == user_id))
        session.execute(delete(WebSession).where(WebSession.user_id == user_id))
        session.execute(delete(Consent).where(Consent.user_id == user_id))
        user.display_name = None
        request.summary = {"teams_erased": len(erase_orgs), "teams_left": len(memberships) - len(erase_orgs), "erasure_tasks": tasks}
        if tasks == 0:
            request.status, request.completed_at = DONE, now
        session.commit()
    except Exception:
        session.rollback()
        raise
    return request


def pending_tasks(session: Session, limit: int = 100) -> list[ErasureTask]:
    return list(session.scalars(select(ErasureTask).where(ErasureTask.status == PENDING).order_by(ErasureTask.created_at, ErasureTask.id).limit(limit)))


def complete_task(session: Session, task_id: str, now: datetime | None = None) -> DeletionRequest:
    """The other service has destroyed what the task named. When the last task of a request is done, the request is DONE."""
    now = now or utcnow()
    task = session.get(ErasureTask, task_id)
    if task is None:
        raise AccountError("NOT_FOUND", "No such erasure task.", 404)
    if task.status != DONE:
        task.status, task.done_at = DONE, now
        session.flush()
    left = session.scalar(select(func.count()).select_from(ErasureTask).where(ErasureTask.deletion_id == task.deletion_id, ErasureTask.status != DONE))
    request = session.get(DeletionRequest, task.deletion_id)
    if left == 0 and request.status != DONE:
        request.status, request.completed_at = DONE, now
    session.commit()
    return request


def overdue(session: Session, now: datetime | None = None) -> list[DeletionRequest]:
    """Deletions past their 30 days and still not done: a legal deadline that monitoring must watch."""
    now = now or utcnow()
    return [r for r in session.scalars(select(DeletionRequest).where(DeletionRequest.status != DONE)) if _aware(r.due_at) < now]


def deletion_status(session: Session, user_id: str) -> dict | None:
    request = session.scalar(select(DeletionRequest).where(DeletionRequest.user_id == user_id))
    if request is None:
        return None
    return {"status": request.status, "requested_at": _iso(request.created_at), "due_by": _iso(request.due_at), "completed_at": _iso(request.completed_at)}
