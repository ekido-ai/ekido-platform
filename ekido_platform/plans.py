"""Plans, the free tier, job tiers and usage limits (ticket C24).

Three plans: **Free** (no subscription), **Starter** and **Business** (Razorpay subscriptions, test mode). Paying for one
job at a time is available on every plan, so a plan decides what is *included*; it never forbids buying a job.

- Free: one small model per team, ever, and a monthly allowance of hosted requests.
- Starter and Business: a number of included jobs and hosted requests per billing period; Business adds the large
  model size, more seats and priority.
- A job is Small, Medium or Large by its rows and base model (`tier_for`), with a fixed per-job price for each.

Every rupee amount here is a **provisional placeholder** (`PRICES_PROVISIONAL`) until the measured GPU costs set the real
ones. Amounts are whole paise and exclude GST, which is added with the same rounding as quotes.

The subscription state is never taken from a webhook's own fields: on any `subscription.*` event we fetch the subscription
from Razorpay and sync from that. A team has the paid plan's benefits only while Razorpay says it is active and the paid
period has not ended; a failed renewal (pending or halted), a cancelled or expired subscription, and an unactivated one
all fall back to Free. Usage limits are checked and consumed in one atomic SQL update, so two requests at the limit
cannot both pass.

Seat limits and the routes that expose all this (`/v1/orgs/{id}/plan` and so on) are not part of this module: the API contract
has no plan endpoints yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, UniqueConstraint, select, update
from sqlalchemy.orm import Mapped, Session, mapped_column

from ekido_core.quotes import LLM_BASES, gst

from .models import Base, Timestamped, utcnow
from .payments import Gateway, PaymentError

PRICES_PROVISIONAL = True
TIERS = ("small", "medium", "large")
# Fixed price for one job by tier, in paise, before GST. Placeholders.
JOB_TIER_PRICES_PAISE = {"small": 29_900, "medium": 79_900, "large": 199_900}

ACTIVE, PENDING, PAST_DUE, HALTED, CANCELLED = "ACTIVE", "PENDING", "PAST_DUE", "HALTED", "CANCELLED"


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    monthly_price_paise: int  # before GST; 0 for Free
    jobs_included: int
    jobs_scope: str  # "lifetime" (Free) or "period" (paid)
    requests_included: int
    requests_scope: str  # "month" (Free: calendar month) or "period" (paid: the billing period)
    tiers: tuple[str, ...]
    seats: int
    priority: bool

    @property
    def monthly_total_paise(self) -> int:
        return self.monthly_price_paise + gst(self.monthly_price_paise)


PLANS: dict[str, Plan] = {
    "free": Plan("free", "Free", 0, 1, "lifetime", 1_000, "month", ("small",), 1, False),
    "starter": Plan("starter", "Starter", 99_900, 3, "period", 50_000, "period", ("small", "medium"), 3, False),
    "business": Plan("business", "Business", 499_900, 15, "period", 500_000, "period", TIERS, 10, True),
}
PAID = ("starter", "business")
_RANK = {"free": 0, "starter": 1, "business": 2}


def tier_for(train_rows: int, base_model: str) -> str:
    """Small, Medium or Large by rows and base model: small is the 0.5B model up to 500 rows; medium is the 0.5B up to
    2,000 rows or the 1.5B up to 500; everything bigger is large."""
    if base_model not in LLM_BASES:
        raise PaymentError("BAD_MODEL", f"base_model must be one of {', '.join(LLM_BASES)}", 422)
    if type(train_rows) is not int or train_rows < 1:
        raise PaymentError("BAD_ROWS", "train_rows must be a whole number from 1", 422)
    bigger = base_model != LLM_BASES[0]
    if not bigger and train_rows <= 500:
        return "small"
    if (not bigger and train_rows <= 2000) or (bigger and train_rows <= 500):
        return "medium"
    return "large"


def tier_price_paise(tier: str) -> dict:
    """The fixed per-job price for a tier, GST shown separately."""
    base = JOB_TIER_PRICES_PAISE[tier]
    return {"tier": tier, "taxable_paise": base, "gst_paise": gst(base), "total_paise": base + gst(base), "prices_provisional": PRICES_PROVISIONAL}


class PlanLimit(Exception):
    """A limit was reached. `upgrade_to` names the next plan that would allow it, or None."""

    def __init__(self, code: str, detail: str, upgrade_to: str | None = None, status: int = 429):
        super().__init__(detail)
        self.code, self.detail, self.upgrade_to, self.status = code, detail, upgrade_to, status


@dataclass(frozen=True)
class JobAllowance:
    included: bool  # True: this job is covered by the plan; False: the team must pay the fixed price for it
    plan: str
    reason: str


# ---------------------------------------------------------------------------------------------------- tables


class TeamPlan(Timestamped, Base):
    __tablename__ = "team_plans"
    org_id: Mapped[str] = mapped_column(String(36), unique=True)
    plan: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default=PENDING)
    subscription_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_at_period_end: Mapped[int] = mapped_column(Integer, default=0)


class PlanUsage(Timestamped, Base):
    """What a team has used in one period. `period` is "all" (lifetime), "m:YYYY-MM" (a calendar month) or "p:<epoch>"
    (a paid billing period, keyed by its start)."""

    __tablename__ = "plan_usage"
    __table_args__ = (UniqueConstraint("org_id", "kind", "period"),)
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(12))  # "jobs" or "requests"
    period: Mapped[str] = mapped_column(String(24))
    used: Mapped[int] = mapped_column(BigInteger, default=0)


def _aware(moment: datetime | None) -> datetime | None:
    return moment if moment is None or moment.tzinfo else moment.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------------------------- the service


class PlansService:
    def __init__(self, gateway: Gateway, razorpay_plan_ids: dict[str, str], now: Callable[[], datetime] | None = None):
        self.gateway, self.razorpay_plan_ids = gateway, dict(razorpay_plan_ids)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._plan_by_remote = {v: k for k, v in self.razorpay_plan_ids.items()}

    # -- what plan a team has right now
    def effective_plan(self, session: Session, org_id: str) -> Plan:
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        if row is None or row.plan not in PLANS:
            return PLANS["free"]
        end = _aware(row.period_end)
        if row.status == ACTIVE and end is not None and self._now() < end:
            return PLANS[row.plan]
        return PLANS["free"]  # pending, past due, halted, cancelled, expired or period over

    def status(self, session: Session, org_id: str) -> dict:
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        plan = self.effective_plan(session, org_id)
        return {"plan": plan.id, "subscription_status": row.status if row else None, "period_end": row.period_end.isoformat() if row and row.period_end else None,
                "cancel_at_period_end": bool(row and row.cancel_at_period_end), "prices_provisional": PRICES_PROVISIONAL}

    # -- subscribing, upgrading, cancelling
    def subscribe(self, session: Session, org_id: str, plan_id: str) -> dict:
        """Start a paid plan. It takes effect only when Razorpay says the first charge went through."""
        if plan_id not in PAID or plan_id not in self.razorpay_plan_ids:
            raise PaymentError("BAD_PLAN", "choose starter or business", 422)
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        if row is not None and row.status in (ACTIVE, PAST_DUE, PENDING) and row.subscription_id:
            raise PaymentError("ALREADY_SUBSCRIBED", "this team already has a subscription; change or cancel it instead", 409)
        remote = self.gateway.create_subscription(plan_id=self.razorpay_plan_ids[plan_id], total_count=120, notes={"org_id": org_id, "plan": plan_id})
        if not remote.get("id"):
            raise PaymentError("GATEWAY_MISMATCH", "Razorpay did not return a subscription", 502)
        if row is None:
            row = TeamPlan(org_id=org_id, plan=plan_id, status=PENDING, subscription_id=remote["id"])
            session.add(row)
        else:
            row.plan, row.status, row.subscription_id, row.period_start, row.period_end, row.cancel_at_period_end = plan_id, PENDING, remote["id"], None, None, 0
        session.commit()
        return {"subscription_id": remote["id"], "plan": plan_id, "status": PENDING, "checkout_url": remote.get("short_url")}

    def change_plan(self, session: Session, org_id: str, plan_id: str) -> dict:
        """Move an active subscription to the other paid plan, from now. Usage so far this period is kept."""
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        if plan_id not in PAID or plan_id not in self.razorpay_plan_ids:
            raise PaymentError("BAD_PLAN", "choose starter or business", 422)
        if row is None or row.status != ACTIVE or not row.subscription_id:
            raise PaymentError("NO_SUBSCRIPTION", "this team has no active subscription to change", 409)
        if row.plan == plan_id:
            return self.status(session, org_id)
        self.gateway.update_subscription(row.subscription_id, plan_id=self.razorpay_plan_ids[plan_id])
        row.plan = plan_id
        session.commit()
        return self.status(session, org_id)

    def cancel(self, session: Session, org_id: str) -> dict:
        """Stop renewing. The team keeps the plan until the paid period ends, then drops to Free."""
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        if row is None or row.status != ACTIVE or not row.subscription_id:
            raise PaymentError("NO_SUBSCRIPTION", "this team has no active subscription to cancel", 409)
        if not row.cancel_at_period_end:
            self.gateway.cancel_subscription(row.subscription_id, at_cycle_end=True)
            row.cancel_at_period_end = 1
            session.commit()
        return self.status(session, org_id)

    # -- Razorpay tells us something changed (renewal, failed renewal, cancel...): sync from Razorpay itself
    def sync_subscription(self, session: Session, subscription_id: str) -> str:
        row = session.scalar(select(TeamPlan).where(TeamPlan.subscription_id == subscription_id))
        if row is None:
            return "unknown_subscription"
        remote = self.gateway.fetch_subscription(subscription_id)  # raises: nothing is recorded, Razorpay retries
        if remote.get("id") != subscription_id:
            return "needs_review"
        known = self._plan_by_remote.get(remote.get("plan_id"))
        if known is None:
            return "needs_review"  # a plan we don't know: change nothing
        status = remote.get("status")
        if status == "active":
            row.status, row.plan = ACTIVE, known
            if remote.get("current_start") and remote.get("current_end"):
                row.period_start = datetime.fromtimestamp(remote["current_start"], timezone.utc)
                row.period_end = datetime.fromtimestamp(remote["current_end"], timezone.utc)
        elif status == "pending":
            row.status = PAST_DUE
        elif status == "halted":
            row.status = HALTED
        elif status in ("cancelled", "completed", "expired"):
            row.status = CANCELLED
        elif status in ("created", "authenticated"):
            row.status = PENDING
        else:
            return "ignored"
        session.commit()
        return f"synced_{row.status.lower()}"

    # -- usage and limits
    def _period_key(self, session: Session, org_id: str, scope: str) -> str:
        if scope == "lifetime":
            return "all"
        if scope == "month":
            now = self._now()
            return f"m:{now.year:04d}-{now.month:02d}"
        row = session.scalar(select(TeamPlan).where(TeamPlan.org_id == org_id))
        return f"p:{int(_aware(row.period_start).timestamp())}" if row and row.period_start else "p:none"

    def _used(self, session: Session, org_id: str, kind: str, period: str) -> int:
        return session.scalar(select(PlanUsage.used).where(PlanUsage.org_id == org_id, PlanUsage.kind == kind, PlanUsage.period == period)) or 0

    def job_allowance(self, session: Session, org_id: str, tier: str) -> JobAllowance:
        """Is the next job on this tier covered by the plan, or does the team pay the fixed price for it?"""
        if tier not in TIERS:
            raise PaymentError("BAD_TIER", "tier must be small, medium or large", 422)
        plan = self.effective_plan(session, org_id)
        if tier not in plan.tiers:
            return JobAllowance(False, plan.id, f"{tier} jobs are not included in the {plan.name} plan")
        used = self._used(session, org_id, "jobs", self._period_key(session, org_id, plan.jobs_scope))
        if used >= plan.jobs_included:
            return JobAllowance(False, plan.id, f"the {plan.name} plan's {plan.jobs_included} included job(s) are used")
        return JobAllowance(True, plan.id, "included in the plan")

    def record_job(self, session: Session, org_id: str, tier: str) -> JobAllowance:
        """Call when a job that the plan covers starts. Counts it, atomically; refuses if it is no longer covered."""
        plan = self.effective_plan(session, org_id)
        if self.job_allowance(session, org_id, tier).included is False:
            raise PlanLimit("JOB_NOT_INCLUDED", "this job is not covered by the plan; pay the fixed price instead", self._upgrade_for(plan, tier), 402)
        if not self._consume(session, org_id, "jobs", self._period_key(session, org_id, plan.jobs_scope), 1, plan.jobs_included):
            raise PlanLimit("JOB_NOT_INCLUDED", "the plan's included jobs are used; pay the fixed price instead", self._upgrade_for(plan, tier), 402)
        return JobAllowance(True, plan.id, "included in the plan")

    def consume_requests(self, session: Session, org_id: str, count: int = 1) -> int:
        """Count hosted requests, all or nothing. Returns how many are left; raises `PlanLimit` at the limit."""
        if type(count) is not int or count < 1:
            raise PaymentError("BAD_COUNT", "count must be a whole number from 1", 422)
        plan = self.effective_plan(session, org_id)
        period = self._period_key(session, org_id, plan.requests_scope)
        if not self._consume(session, org_id, "requests", period, count, plan.requests_included):
            raise PlanLimit("REQUEST_LIMIT", f"the {plan.name} plan's {plan.requests_included:,} hosted requests are used for this period", self._upgrade_for(plan, None))
        return plan.requests_included - self._used(session, org_id, "requests", period)

    def _consume(self, session: Session, org_id: str, kind: str, period: str, count: int, limit: int) -> bool:
        """Add `count` only if it fits under `limit`, in one statement, so concurrent requests cannot both pass."""
        if count > limit:
            return False
        if session.scalar(select(PlanUsage.id).where(PlanUsage.org_id == org_id, PlanUsage.kind == kind, PlanUsage.period == period)) is None:
            session.add(PlanUsage(org_id=org_id, kind=kind, period=period, used=0))
            session.flush()
        result = session.execute(update(PlanUsage).where(PlanUsage.org_id == org_id, PlanUsage.kind == kind, PlanUsage.period == period, PlanUsage.used + count <= limit).values(used=PlanUsage.used + count))
        session.commit()
        return result.rowcount == 1

    def _upgrade_for(self, plan: Plan, tier: str | None) -> str | None:
        for candidate in PAID:
            if _RANK[candidate] > _RANK[plan.id] and (tier is None or tier in PLANS[candidate].tiers):
                return candidate
        return None
