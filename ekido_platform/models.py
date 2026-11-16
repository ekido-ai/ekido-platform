"""The Platform API's own tables. They live in the `platform` schema on Postgres; the Coordinator's tables are in
another schema this service's database user can't read. Anything the Coordinator owns (jobs, providers, checkpoints)
is referred to by its ID as a plain string and fetched over the Coordinator's API, never joined."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Timestamped:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Timestamped, Base):
    """A person who signs in. Their email lives in user_emails; teams in org_members."""

    __tablename__ = "users"
    display_name: Mapped[str | None] = mapped_column(String(120))
