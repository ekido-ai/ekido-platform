"""Few examples, more rows, and the customer approves before training (ticket #154, the Platform half; contract: distillation, 25 Nov).

A customer with only a few examples (the plan: 30 to learn from, 20 to test with) asks for generated examples. One quote covers both steps. This
module is the Platform's side of that flow; the Coordinator's side (placing the generation job on a T4-class device, storing the rows) is separate.

    paid order (HELD) whose quote asked for distillation
      -> POST   /v1/orgs/{org}/distillations                  starts the generation job              GENERATING
      -> (the Coordinator posts the rows to /internal/distillations/{id}/generated)
           rows pass the checks                                                                     REVIEW (20 random generated rows to look at)
           rows fail the checks, or the job fails                full refund                        FAILED
      -> GET    .../distillations/{id}                          the state and, in REVIEW, the sample
      -> POST   .../distillations/{id}/remove                   mark sample rows wrong
      -> POST   .../distillations/{id}/regenerate               once: a new generation               GENERATING
      -> POST   .../distillations/{id}/approve                  builds the training set, starts it   APPROVED, then TRAINING
      -> POST   .../distillations/{id}/reject                   full refund                          REJECTED

Rules kept here:
- The sample is 20 rows drawn at random from the **generated** rows with the system's random source, never from the customer's test rows (those are
  not in the generated file at all, and a generated row equal to a test input fails the whole file).
- Rows the customer removes are left out of the training set. If more than a quarter of the sample is removed, the answer says so and suggests
  regenerating; it does not force it.
- Regenerating is allowed once. Rejecting, or a failed generation, refunds the order in full (`RazorpayRail.refund`, which is idempotent).
- Training is started on the paid order exactly like a normal job (same claim, same budget, same link), so the Coordinator settles or refunds it
  when it ends. **The generation job is not linked to the order**, so settlement ignores it: what the generation provider is paid is an open
  question (see the PR).
- Everything is the team's own: another team's distillation looks missing (404). The generated rows are customer data and are erased with the team.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, DateTime, Integer, LargeBinary, String, delete, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from ekido_core import distill_contract
from ekido_core.recipes import recipe

from . import accounts
from .accounts import AccountError, JobOwner, Principal
from .models import Base, Timestamped, utcnow
from .payments import HELD, PaymentError, PaymentOrder
from .v1_datasets import Dataset, Quote, dataset_bytes, recipe_job, split_rows
from .v1_jobs import cancel_quietly

GENERATING, REVIEW, APPROVED, TRAINING, REJECTED, FAILED = "GENERATING", "REVIEW", "APPROVED", "TRAINING", "REJECTED", "FAILED"
SAMPLE_SIZE = 20
MAX_REGENERATIONS = 1
SUGGEST_REGENERATE_ABOVE = 0.25
MAX_GENERATED_ROWS = 20_000

# (method, path, JSON body) -> (status, parsed JSON); raises OSError when the Coordinator cannot be reached.
CoordinatorPost = Callable[[str, str, "dict | None"], "tuple[int, Any]"]


class Distillation(Timestamped, Base):
    """One distillation of one paid order. The generated rows are customer data: they are erased with the team."""

    __tablename__ = "distillations"
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    order_id: Mapped[str] = mapped_column(String(36), unique=True)
    dataset_id: Mapped[str] = mapped_column(String(36))
    state: Mapped[str] = mapped_column(String(16), index=True)
    generation_job_id: Mapped[str | None] = mapped_column(String(36))
    generating_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # when the current generation started; see `sweep_stuck_generations`
    generated: Mapped[bytes | None] = mapped_column(LargeBinary)  # generated.jsonl, exactly as the Coordinator stored it
    generated_rows: Mapped[int] = mapped_column(Integer, default=0)
    generated_sha256: Mapped[str | None] = mapped_column(String(64))
    sample: Mapped[list] = mapped_column(JSON, default=list)  # indexes into the generated rows
    removed: Mapped[list] = mapped_column(JSON, default=list)  # indexes the customer marked wrong (all inside the sample)
    regenerations: Mapped[int] = mapped_column(Integer, default=0)
    training_dataset_id: Mapped[str | None] = mapped_column(String(36))
    training_job_id: Mapped[str | None] = mapped_column(String(36))
    reason: Mapped[str | None] = mapped_column(String(200))


class StartBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(min_length=1, max_length=36)


class RemoveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indices: list[int] = Field(min_length=1, max_length=SAMPLE_SIZE)


class GenerationFailed(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="the generation job failed", max_length=200)


class SweepBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    older_than_minutes: int = Field(default=360, ge=15, le=7 * 24 * 60)  # 6 hours unless told otherwise


class Generated(BaseModel):
    """What the Coordinator posts when generation finished: the rows file and the verifier's verdict on it."""

    model_config = ConfigDict(extra="forbid")
    rows_jsonl: str = Field(max_length=40_000_000)
    passed: bool
    reasons: list[str] = Field(default_factory=list, max_length=40)


def normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def generation_request(quote: dict, dataset: Dataset, *, attempt: int) -> tuple[dict, str]:
    """The generation job for a paid quote: the `distill-gen/v1` spec and `seeds.jsonl`, both as docs/api/distill-gen-v1.md in ekido-core says.
    The test rows never leave the Platform: the spec carries only hashes of their inputs."""
    r = recipe(dataset.recipe_id)
    seeds = [{column: row[column] for column in r.columns} for row in split_rows(dataset.content, "train")]
    seeds_bytes = distill_contract.jsonl_bytes(seeds)
    wanted = quote["plan"]["generated_rows"]
    spec = {
        "template": distill_contract.TEMPLATE,
        "recipe_id": dataset.recipe_id,
        "teacher": {"id": distill_contract.TEACHER_ID, "revision": distill_contract.TEACHER_REVISION,
                    "quantization": distill_contract.TEACHER_QUANTIZATION},
        "seeds": {"sha256": hashlib.sha256(seeds_bytes).hexdigest(), "bytes": len(seeds_bytes), "rows": len(seeds)},
        "test_input_sha256": distill_contract.test_input_hashes(dataset.recipe_id, split_rows(dataset.content, "test")),
        "per_seed": min(20, max(1, -(-wanted // max(1, len(seeds))))),
        "generation_seed": int(dataset.sha256[:8], 16) % 1_000_000 + attempt,
        "max_new_tokens": 96,
        "attempts": 3,
        "filter_version": distill_contract.FILTER_VERSION,
        "attempt": attempt,
    }
    problems = distill_contract.spec_problems(spec)
    if problems:
        raise AccountError("GENERATION_REFUSED", "These examples cannot be used to generate more: " + "; ".join(problems[:3]), 422)
    return spec, seeds_bytes.decode("utf-8")


def own(session: Session, org_id: str, distillation_id: str) -> Distillation:
    row = session.get(Distillation, distillation_id)
    if row is None or row.org_id != org_id:
        raise AccountError("NOT_FOUND", "No such distillation.", 404)
    return row


def generated_rows(row: Distillation) -> list[dict]:
    return [json.loads(line) for line in (row.generated or b"").decode("utf-8").splitlines() if line.strip()]


def view(row: Distillation, *, with_sample: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"distillation_id": row.id, "order_id": row.order_id, "state": row.state, "generated_rows": row.generated_rows,
                           "removed": len(row.removed or []), "regenerations_left": max(0, MAX_REGENERATIONS - row.regenerations),
                           "training_job_id": row.training_job_id, "reason": row.reason}
    if row.state == REVIEW:
        sample = row.sample or []
        out["suggest_regenerate"] = bool(sample) and len(row.removed or []) / len(sample) > SUGGEST_REGENERATE_ABOVE
        if with_sample:
            rows = generated_rows(row)
            out["sample"] = [{"index": i, "removed": i in (row.removed or []), **rows[i]} for i in sample]
    return out


# ------------------------------------------------------------------------------------------------------------- what happens, step by step

def _refund(rail, session: Session, order_id: str, reason: str) -> None:
    if rail is None:
        raise AccountError("PAYMENTS_UNAVAILABLE", "Payments aren't set up on this server yet.", 503)
    try:
        rail.refund(session, order_id, reason)
    except PaymentError as exc:
        raise AccountError(exc.code, exc.detail, exc.status) from None


def _ensure_refunded(session: Session, row: Distillation, rail) -> None:
    """A failed or rejected distillation must end with the money back. If the order is still HELD (the refund failed earlier), try again."""
    order = session.get(PaymentOrder, row.order_id)
    if order is not None and order.status == HELD and not (order.job_id and not order.job_id.startswith("claim-")):
        _refund(rail, session, row.order_id, row.reason or "distillation ended without a model")


def _start_generation(session: Session, row: Distillation, quote: dict, dataset: Dataset, coordinator: CoordinatorPost) -> None:
    """Ask the Coordinator for the generation job. On any refusal nothing changes and the caller sees why."""
    spec, seeds_jsonl = generation_request(quote, dataset, attempt=row.regenerations)
    try:
        status, created = coordinator("POST", "/distillations", {"spec": spec, "distillation_id": row.id, "seeds_jsonl": seeds_jsonl})
    except OSError:
        raise AccountError("UNAVAILABLE", "The training service is not reachable. Nothing was charged again; try again.", 503) from None
    if status not in (200, 201) or not isinstance(created, dict) or not created.get("job_id"):
        detail = created.get("detail") if isinstance(created, dict) and isinstance(created.get("detail"), str) else "the service refused it"
        raise AccountError("GENERATION_REFUSED", "The example generation could not start: " + detail[:200], 422 if status in (400, 409, 422) else 502)
    row.generation_job_id = created["job_id"]
    row.generating_since = utcnow()


def sweep_stuck_generations(session: Session, rail, *, older_than: timedelta, now: datetime | None = None) -> dict[str, int]:
    """For a timer. (1) Fail and refund every distillation that has been GENERATING for longer than `older_than`: the Coordinator never
    reported, so the customer's money must not stay held for ever. (2) Retry the refund of any FAILED or REJECTED one whose order is still
    held (an earlier refund failed). One refund that fails does not stop the others; it is counted in `refund_errors` and the next run retries it.
    Safe to run twice."""
    now = now or utcnow()
    cutoff = now - older_than
    failed = errors = 0
    handled: set[str] = set()
    for row in list(session.scalars(select(Distillation).where(Distillation.state == GENERATING))):
        since = row.generating_since or row.created_at
        since = since if since.tzinfo else since.replace(tzinfo=cutoff.tzinfo)
        if since > cutoff:
            continue
        handled.add(row.id)
        try:
            fail_generation(session, row.id, "example generation took too long", rail)
            failed += 1
        except AccountError:
            session.rollback()
            failed += 1  # it is FAILED now (the state was saved before the refund); only the refund is outstanding
            errors += 1
    for row in list(session.scalars(select(Distillation).where(Distillation.state.in_((FAILED, REJECTED))))):
        if row.id in handled:
            continue  # just failed above; its refund was tried once in this run
        try:
            _ensure_refunded(session, row, rail)
        except AccountError:
            session.rollback()
            errors += 1
    return {"failed": failed, "refund_errors": errors}


def accept_rows(session: Session, distillation_id: str, body: Generated, rail) -> Distillation:
    """The Coordinator's report that generation finished. Idempotent: a repeat after the state moved on changes nothing."""
    row = session.get(Distillation, distillation_id)
    if row is None:
        raise AccountError("NOT_FOUND", "No such distillation.", 404)
    if row.state != GENERATING:
        if row.state == FAILED:
            _ensure_refunded(session, row, rail)  # a repeat of the same report retries a refund that failed the first time
        return row
    dataset = session.get(Dataset, row.dataset_id)
    problem = None if body.passed else "the rows failed the verifier's checks: " + ", ".join(body.reasons[:8])
    rows: list[dict] = []
    if problem is None:
        try:
            rows = [json.loads(line) for line in body.rows_jsonl.splitlines() if line.strip()]
        except ValueError:
            problem = "the generated file is not JSON lines"
    if problem is None:
        r = recipe(dataset.recipe_id)
        columns = set(r.columns)
        source = r.columns[0]
        test_inputs = {normalise(x[source]) for x in split_rows(dataset.content, "test")}
        if not rows or len(rows) > MAX_GENERATED_ROWS:
            problem = "the generated file has no rows or too many"
        elif any(not isinstance(x, dict) or set(x) != columns or not all(isinstance(x[c], str) and x[c].strip() for c in columns) for x in rows):
            problem = "the generated rows do not have the recipe's columns"
        elif any(normalise(x[source]) in test_inputs for x in rows):
            problem = "a generated row equals one of the customer's test examples"  # the Platform checks this itself and does not rely on the verifier
    if problem is not None:
        row.state, row.reason = FAILED, problem[:200]
        session.commit()
        _refund(rail, session, row.order_id, "example generation failed")
        return row
    row.generated = ("\n".join(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in rows) + "\n").encode("utf-8")
    row.generated_rows = len(rows)
    row.generated_sha256 = hashlib.sha256(row.generated).hexdigest()
    row.sample = sorted(secrets.SystemRandom().sample(range(len(rows)), min(SAMPLE_SIZE, len(rows))))
    row.removed, row.state, row.reason = [], REVIEW, None
    session.commit()
    return row


def fail_generation(session: Session, distillation_id: str, reason: str, rail) -> Distillation:
    row = session.get(Distillation, distillation_id)
    if row is None:
        raise AccountError("NOT_FOUND", "No such distillation.", 404)
    if row.state != GENERATING:
        if row.state == FAILED:
            _ensure_refunded(session, row, rail)
        return row
    row.state, row.reason = FAILED, reason[:200]
    session.commit()
    _refund(rail, session, row.order_id, "example generation failed")
    return row


def erase_distillations(session: Session, org_id: str) -> None:
    """Part of "delete my data": the generated rows, the datasets and quotes of an erased team."""
    session.execute(delete(Distillation).where(Distillation.org_id == org_id))
    session.execute(delete(Quote).where(Quote.org_id == org_id))
    session.execute(delete(Dataset).where(Dataset.org_id == org_id))


# ----------------------------------------------------------------------------------------------------------------------- the customer routes

def build_distill_router(engine: Engine, principal: Callable, coordinator: CoordinatorPost, rail: Callable[[], Any] = lambda: None) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def db():
        with Session(engine) as session:
            yield session

    @router.post("/orgs/{org_id}/distillations", status_code=201, tags=["jobs"])
    def start(org_id: str, body: StartBody, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        order = session.get(PaymentOrder, body.order_id)
        if order is None or order.org_id != org_id:
            raise AccountError("NOT_FOUND", "No such order.", 404)
        if order.status != HELD:
            raise AccountError("NOT_PAID", f"This order is {order.status}, so no examples can be generated for it.", 409)
        quote = order.quote
        if not quote.get("distill") or not quote["plan"]["generated_rows"]:
            raise AccountError("NOT_A_DISTILLATION", "This order's quote did not ask for generated examples.", 422)
        quote_row = session.scalar(select(Quote).where(Quote.terms_hash == order.terms_hash, Quote.org_id == org_id))
        dataset = session.get(Dataset, quote_row.dataset_id) if quote_row is not None else None
        if dataset is None or dataset.sha256 != quote["dataset_sha256"]:
            raise AccountError("NOT_FOUND", "The examples this quote was made for are not here any more.", 404)
        if order.job_id:
            raise AccountError("ALREADY_STARTED", "A training job has already been started from this order.", 409)
        row = Distillation(org_id=org_id, order_id=order.id, dataset_id=dataset.id, state=GENERATING, regenerations=0)
        session.add(row)
        try:
            session.flush()  # the unique order id is the claim: a second request for this order fails here
        except Exception:
            session.rollback()
            raise AccountError("ALREADY_STARTED", "Example generation has already been started for this order.", 409) from None
        try:
            _start_generation(session, row, quote, dataset, coordinator)
        except AccountError:
            session.rollback()
            raise
        session.commit()
        return view(row)

    @router.get("/orgs/{org_id}/distillations/{distillation_id}", tags=["jobs"])
    def show(org_id: str, distillation_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin", "member")
        return view(own(session, org_id, distillation_id), with_sample=True)

    @router.post("/orgs/{org_id}/distillations/{distillation_id}/remove", tags=["jobs"])
    def remove(org_id: str, distillation_id: str, body: RemoveBody, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        row = own(session, org_id, distillation_id)
        if row.state != REVIEW:
            raise AccountError("NOT_IN_REVIEW", f"Rows can only be marked wrong while the examples are being reviewed; this is {row.state}.", 409)
        if any(i not in (row.sample or []) for i in body.indices):
            raise AccountError("NOT_IN_SAMPLE", "Only the rows shown for review can be marked wrong.", 422)
        row.removed = sorted(set(row.removed or []) | set(body.indices))
        session.commit()
        return view(row, with_sample=True)

    @router.post("/orgs/{org_id}/distillations/{distillation_id}/regenerate", tags=["jobs"])
    def regenerate(org_id: str, distillation_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        row = own(session, org_id, distillation_id)
        if row.state != REVIEW:
            raise AccountError("NOT_IN_REVIEW", f"Examples can only be regenerated from the review step; this is {row.state}.", 409)
        if row.regenerations >= MAX_REGENERATIONS:
            raise AccountError("NO_REGENERATIONS_LEFT", "Examples can be regenerated once. Approve this set or reject it for a full refund.", 409)
        order = session.get(PaymentOrder, row.order_id)
        dataset = session.get(Dataset, row.dataset_id)
        row.regenerations += 1
        try:
            _start_generation(session, row, order.quote, dataset, coordinator)
        except AccountError:
            session.rollback()
            raise
        row.state, row.generated, row.generated_rows, row.generated_sha256, row.sample, row.removed = GENERATING, None, 0, None, [], []
        row.generating_since = utcnow()
        session.commit()
        return view(row)

    @router.post("/orgs/{org_id}/distillations/{distillation_id}/approve", status_code=201, tags=["jobs"])
    def approve(org_id: str, distillation_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        row = own(session, org_id, distillation_id)
        if row.state not in (REVIEW, APPROVED):
            raise AccountError("NOT_IN_REVIEW", f"Only a set under review can be approved; this is {row.state}.", 409)
        order = session.get(PaymentOrder, row.order_id)
        seeds = session.get(Dataset, row.dataset_id)
        if row.state == REVIEW:
            train = split_rows(seeds.content, "train") + [x for i, x in enumerate(generated_rows(row)) if i not in set(row.removed or [])]
            test = split_rows(seeds.content, "test")
            content = dataset_bytes(seeds.recipe_id, train, test)
            combined = Dataset(org_id=org_id, recipe_id=seeds.recipe_id, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), train_rows=len(train),
                               test_rows=len(test), redactions=seeds.redactions or {}, content=content)
            session.add(combined)
            session.flush()
            row.training_dataset_id, row.state = combined.id, APPROVED
            session.commit()
        combined = session.get(Dataset, row.training_dataset_id)
        started = _start_training(session, row, order, combined, coordinator)
        return {**view(row), "job_id": started}

    @router.post("/orgs/{org_id}/distillations/{distillation_id}/reject", tags=["jobs"])
    def reject(org_id: str, distillation_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin")
        row = own(session, org_id, distillation_id)
        if row.state == REJECTED:
            _ensure_refunded(session, row, rail())  # asking again retries a refund that failed the first time
            return view(row)
        if row.state not in (REVIEW, APPROVED):
            raise AccountError("CANNOT_REJECT", f"A set can only be rejected while it is under review; this is {row.state}.", 409)
        row.state, row.reason = REJECTED, "rejected by the customer"
        session.commit()
        _refund(rail(), session, row.order_id, "customer rejected the generated examples")
        return view(row)

    def _start_training(session: Session, row: Distillation, order: PaymentOrder, combined: Dataset, post: CoordinatorPost) -> str:
        """The same claim, budget and link as a normal paid job (`v1_jobs`), on the approved training set."""
        claim = "claim-" + uuid.uuid4().hex[:28]  # the order must still have no job: one paid order, one job
        taken = session.execute(update(PaymentOrder).where(PaymentOrder.id == order.id, PaymentOrder.job_id.is_(None)).values(job_id=claim))
        session.commit()
        if taken.rowcount != 1:
            raise AccountError("ALREADY_STARTED", "A training job has already been started from this order.", 409)

        def release() -> None:
            session.execute(update(PaymentOrder).where(PaymentOrder.id == order.id, PaymentOrder.job_id == claim).values(job_id=None))
            session.commit()

        job_id = None
        try:
            spec, policy = _training_job(order.quote, combined)
            budget = (Decimal(order.quote["provider_budget_paise"]) / 100).quantize(Decimal("0.000001"))
            status, created = post("POST", "/jobs", {"workload": spec, "policy": policy, "budget_usdc": str(budget), "deadline_seconds": 86_400})
            if status not in (200, 201) or not isinstance(created, dict) or not created.get("job_id"):
                raise AccountError("JOB_REFUSED", "The training job could not be created.", 422 if status in (400, 422) else 502)
            job_id = created["job_id"]
            status, _ = post("POST", f"/jobs/{job_id}/activate", None)
            if status != 200:
                post("POST", f"/jobs/{job_id}/cancel", None)
                raise AccountError("JOB_REFUSED", "The training job could not be started.", 422 if status in (400, 409, 422) else 502)
        except OSError:
            if job_id is not None:
                cancel_quietly(post, job_id)
            release()
            raise AccountError("UNAVAILABLE", "The training service is not reachable. Nothing was charged again; try again.", 503) from None
        except AccountError:
            release()
            raise
        session.add(JobOwner(job_id=job_id, org_id=order.org_id))
        session.execute(update(PaymentOrder).where(PaymentOrder.id == order.id).values(job_id=job_id))
        row.training_job_id, row.state = job_id, TRAINING
        session.commit()
        return job_id

    return router


def _training_job(quote: dict, combined: Dataset) -> tuple[dict, dict]:
    """The `llm-lora/v1` spec and policy for the approved set: `recipe_job` with the plan's generated rows already inside the dataset."""
    adjusted = {**quote, "plan": {**quote["plan"], "generated_rows": 0}, "dataset_sha256": combined.sha256}
    return recipe_job(adjusted, combined)
