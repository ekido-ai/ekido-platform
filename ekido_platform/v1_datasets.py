"""The first half of the customer journey (ticket C9): recipes, datasets, quotes and orders (contract tags recipes and billing).

1. `GET /v1/recipes` lists the ready-made recipes from `ekido_core.recipes`, with their columns and quality floor in plain words.
2. `POST /v1/orgs/{org_id}/datasets` takes rows the browser has already cleaned of personal details. It checks them against the
   recipe, writes one canonical `dataset.jsonl` (every row carries its `split`), and keeps it with its SHA-256. Only counts of
   what the browser removed are stored, never the values. The bytes go to the Coordinator only when a paid job starts.
3. `POST /v1/orgs/{org_id}/quotes` prices a stored dataset with `ekido_core.quotes.make_quote` and keeps the quote.
4. `POST /v1/quotes/{quote_id}/orders` makes the Razorpay order (test mode) the browser pays with Checkout.
5. `recipe_job` turns a paid quote and its dataset into the exact `llm-lora/v1` spec and quality policy, so a customer never
   sends a training spec: what runs is what the quote priced.

A team sees only its own datasets and quotes; anything else looks missing (404), not forbidden.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Callable

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, LargeBinary, String, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column

from ekido_core.canonical import document_hash
from ekido_core.job_spec import LLM_BASES, MODEL_REVISIONS, spec_problems
from ekido_core.quotes import QuoteError, make_quote, verify_quote
from ekido_core.recipes import RECIPES, eval_set_hash, recipe

from . import accounts
from .accounts import AccountError, Principal
from .models import Base, Timestamped, utcnow

MAX_TRAIN_ROWS, MIN_TEST_ROWS, MAX_TEST_ROWS = 20_000, 20, 5_000
MAX_CELL_CHARS = 8_000
MAX_DATASET_BYTES = 20 * 1024 * 1024
PROBLEMS_SHOWN = 50


class Dataset(Timestamped, Base):
    """Cleaned rows for one recipe, as the exact bytes a provider trains on."""

    __tablename__ = "datasets"
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("orgs.id"), index=True)
    recipe_id: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    bytes: Mapped[int] = mapped_column(Integer)
    train_rows: Mapped[int] = mapped_column(Integer)
    test_rows: Mapped[int] = mapped_column(Integer)
    redactions: Mapped[dict] = mapped_column(JSON, default=dict)  # counts per kind, never values
    content: Mapped[bytes] = mapped_column(LargeBinary)


class Quote(Base):
    """A priced, hashed offer for one dataset. The terms hash is what gets paid for and later anchored."""

    __tablename__ = "quotes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("orgs.id"), index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"))
    terms_hash: Mapped[str] = mapped_column(String(64), unique=True)
    quote: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NewDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe_id: str = Field(max_length=64)
    train: list[dict[str, str]] = Field(max_length=MAX_TRAIN_ROWS)
    test: list[dict[str, str]] = Field(max_length=MAX_TEST_ROWS)
    redactions: dict[str, int] = Field(default_factory=dict, max_length=32)


class NewQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(max_length=36)
    base_model: str = Field(default=LLM_BASES[0], max_length=100)
    hosting_requests_per_month: int = Field(default=0, ge=0, le=10_000_000)
    distill: bool = False  # ask for generated examples as well (the one quote then covers both steps)


def recipe_view(r) -> dict[str, Any]:
    source, target = r.columns
    return {"recipe_id": r.recipe_id, "title": r.title, "task": r.task, "columns": {"input": source, "answer": target},
            "labels": list(r.labels), "metric": r.metric, "floor": r.floor, "quality_terms": r.plain_words(100).replace("100 ", "")}


# An estimate asks for no sign-in, so it is cheap and answers every caller the same: it prices two row counts, stores nothing and signs
# nothing. It names only what a customer is shown on a real quote's face; the fee split and the provider budget stay out.
ESTIMATE_FIELDS = ("currency", "training_paise", "gst_paise", "total_paise", "eta_minutes", "quality_terms", "refund_policy", "prices_provisional")
ESTIMATE_LIMIT, ESTIMATE_WINDOW = 60, 600  # requests per address per ten minutes


def client_address(request: Request) -> str:
    """The caller's address. The Platform listens on loopback behind Caddy, which sets X-Forwarded-For to the real address, so the header
    is believed only when the connection itself comes from loopback; a direct caller cannot choose its own bucket."""
    peer = request.client.host if request.client else "unknown"
    if peer in ("127.0.0.1", "::1"):
        forwarded = request.headers.get("x-forwarded-for", "")
        last = forwarded.split(",")[-1].strip()
        if last:
            return last[:64]
    return peer


def estimate_view(recipe_id: str, train_rows: int, test_rows: int) -> dict[str, Any]:
    """What a quote would roughly cost for this many examples, for a visitor who has not signed in. Raises QuoteError for a bad count."""
    quote = make_quote(recipe_id=recipe_id, base_model=LLM_BASES[0], train_rows=train_rows, test_rows=test_rows, dataset_sha256="0" * 64)
    return {"recipe_id": recipe_id, "train_rows": train_rows, "test_rows": test_rows, "estimate": True,
            **{field: quote[field] for field in ESTIMATE_FIELDS}}


def row_problems(recipe_id: str, train: list[dict], test: list[dict]) -> list[str]:
    """Every reason the rows don't fit the recipe, so the customer can fix them all at once."""
    r = recipe(recipe_id)
    columns = set(r.columns)
    problems: list[str] = []
    if not train:
        problems.append("train needs at least 1 row")
    if len(test) < MIN_TEST_ROWS:
        problems.append(f"test needs at least {MIN_TEST_ROWS} rows, so the quality check means something")
    for split, rows in (("train", train), ("test", test)):
        for index, row in enumerate(rows, start=1):
            if set(row) != columns:
                problems.append(f"{split} row {index}: needs exactly the columns {', '.join(r.columns)}")
                continue
            for column in r.columns:
                value = row[column].strip()
                if not value:
                    problems.append(f"{split} row {index}: {column} is empty")
                elif len(value) > MAX_CELL_CHARS:
                    problems.append(f"{split} row {index}: {column} is longer than {MAX_CELL_CHARS} characters")
            if r.labels and row[r.columns[1]].strip().lower() not in r.labels:
                problems.append(f"{split} row {index}: {r.columns[1]} must be one of {', '.join(r.labels)}")
    seen = {row[r.columns[0]].strip().casefold() for row in train if set(row) == columns}
    leaked = sum(1 for row in test if set(row) == columns and row[r.columns[0]].strip().casefold() in seen)
    if leaked:
        problems.append(f"{leaked} test rows also appear in train; the check must use examples the model never saw")
    return problems


def dataset_bytes(recipe_id: str, train: list[dict], test: list[dict]) -> bytes:
    """One JSONL line per row with its split, keys sorted and values trimmed: the same rows always give the same bytes."""
    r = recipe(recipe_id)
    clean = lambda row: {column: row[column].strip() for column in r.columns}
    lines = [json.dumps({"split": split, **clean(row)}, ensure_ascii=False, sort_keys=True) for split, rows in (("train", train), ("test", test)) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def split_rows(content: bytes, split: str) -> list[dict]:
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("split") == split]


def recipe_job(quote: dict, dataset: Dataset) -> tuple[dict, dict]:
    """The llm-lora/v1 spec and quality policy for a paid quote: exactly the plan that was priced, never more."""
    if dataset.sha256 != quote["dataset_sha256"] or dataset.recipe_id != quote["recipe_id"]:
        raise AccountError("QUOTE_MISMATCH", "This quote is for different data.", 409)
    plan = quote["plan"]  # exactly what was priced
    if plan["generated_rows"]:
        raise AccountError("DISTILL_UNAVAILABLE", "Generated examples are not available yet; quote without them.", 422)
    r = recipe(quote["recipe_id"])
    steps = plan["total_logical_steps"]
    spec = {
        "schema_version": 1,
        "template": "llm-lora/v1",
        "recipe_id": r.recipe_id,
        "model": {"id": quote["base_model"], "revision": MODEL_REVISIONS[quote["base_model"]]},
        "dataset": {"sha256": dataset.sha256, "bytes": dataset.bytes, "train_rows": dataset.train_rows, "test_rows": dataset.test_rows},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "max_seq_length": plan["max_seq_length"],
        "logical_batch_size": plan["logical_batch_size"],
        "total_logical_steps": steps,
        "checkpoint_every_steps": max(10, steps // 4),
        "seed": int(dataset.sha256[:8], 16) % 1_000_000,
        "optimizer": {"name": "adamw", "lr": 2e-4, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0},
        "lr_schedule": {"name": "linear_warmup_linear_decay", "warmup_steps": min(10, max(1, steps // 10))},
    }
    problems = spec_problems(spec)
    if problems:
        raise AccountError("SPEC_INVALID", "The training job could not be built: " + "; ".join(problems), 500)
    policy = {
        "schema_version": 1,
        "metric": "accuracy",
        "min_absolute": r.floor,
        "non_inferiority_margin": 0.02,
        "confidence": 0.95,
        "bootstrap_resamples": 2000,
        "eval_set_hash": eval_set_hash(r.recipe_id, split_rows(dataset.content, "test")),
        # Which scoring rule judged it: the recipe's own definition, hashed.
        "eval_version_hash": document_hash({"recipe_id": r.recipe_id, "metric": r.metric, "floor": r.floor, "pass_score": r.pass_score,
                                            "labels": list(r.labels), "instruction": r.instruction}),
    }
    return spec, policy


def build_datasets_router(engine: Engine, principal: Callable, payment_rail: Callable[[], Any] = lambda: None,
                          razorpay_key_id: Callable[[], str] = lambda: "") -> APIRouter:
    router = APIRouter(prefix="/v1")

    def db():
        with Session(engine) as session:
            yield session

    @router.get("/recipes", tags=["recipes"])
    def recipes() -> list[dict[str, Any]]:
        return [recipe_view(r) for r in RECIPES.values()]

    @router.get("/recipes/{recipe_id}", tags=["recipes"])
    def one_recipe(recipe_id: str) -> dict[str, Any]:
        if recipe_id not in RECIPES:
            raise AccountError("NOT_FOUND", "No such recipe.", 404)
        return recipe_view(RECIPES[recipe_id])

    from .v1_assistant import RateLimit  # here, not at the top: v1_assistant imports Dataset from this module

    estimate_limiter = RateLimit(ESTIMATE_LIMIT, ESTIMATE_WINDOW)

    @router.get("/recipes/{recipe_id}/estimate", tags=["recipes"])
    def recipe_estimate(recipe_id: str, request: Request, train_rows: int = Query(ge=1, le=MAX_TRAIN_ROWS),
                        test_rows: int = Query(ge=MIN_TEST_ROWS, le=MAX_TEST_ROWS)) -> dict[str, Any]:
        """A rough price and the quality terms for a recipe, before anyone signs in. Not a quote: nothing is stored, signed or reserved,
        and the real price is the quote made after the examples are uploaded."""
        if recipe_id not in RECIPES:
            raise AccountError("NOT_FOUND", "No such recipe.", 404)
        if not estimate_limiter.allow(client_address(request)):
            raise AccountError("RATE_LIMITED", "Too many price estimates. Try again in a few minutes.", 429)
        try:
            return estimate_view(recipe_id, train_rows, test_rows)
        except QuoteError as exc:
            raise AccountError("ESTIMATE_REFUSED", str(exc), 422) from None

    @router.post("/orgs/{org_id}/datasets", status_code=201, tags=["recipes"])
    def new_dataset(org_id: str, body: NewDataset, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin", "member")
        if body.recipe_id not in RECIPES:
            raise AccountError("NOT_FOUND", "No such recipe.", 404)
        problems = row_problems(body.recipe_id, body.train, body.test)
        if problems:
            shown = problems[:PROBLEMS_SHOWN]
            more = f" (and {len(problems) - len(shown)} more)" if len(problems) > len(shown) else ""
            raise AccountError("ROWS_INVALID", f"{len(problems)} problems with these rows{more}: " + "; ".join(shown), 422)
        content = dataset_bytes(body.recipe_id, body.train, body.test)
        if len(content) > MAX_DATASET_BYTES:
            raise AccountError("TOO_LARGE", "These rows are larger than 20 MB; use fewer or shorter examples.", 413)
        redactions = {kind[:32]: count for kind, count in body.redactions.items() if 0 <= count <= 10_000_000}
        dataset = Dataset(org_id=org_id, recipe_id=body.recipe_id, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content),
                          train_rows=len(body.train), test_rows=len(body.test), redactions=redactions, content=content)
        session.add(dataset)
        session.commit()
        return {"dataset_id": dataset.id, "recipe_id": dataset.recipe_id, "sha256": dataset.sha256, "bytes": dataset.bytes,
                "train_rows": dataset.train_rows, "test_rows": dataset.test_rows, "redactions": dataset.redactions}

    @router.post("/orgs/{org_id}/quotes", status_code=201, tags=["billing"])
    def new_quote(org_id: str, body: NewQuote, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        accounts.require_role(session, who, org_id, "owner", "admin", "member")
        dataset = session.get(Dataset, body.dataset_id)
        if dataset is None or dataset.org_id != org_id:
            raise AccountError("NOT_FOUND", "No such dataset.", 404)
        try:
            quote = make_quote(recipe_id=dataset.recipe_id, base_model=body.base_model, train_rows=dataset.train_rows, test_rows=dataset.test_rows,
                               dataset_sha256=dataset.sha256, distill=body.distill, hosting_requests_per_month=body.hosting_requests_per_month)
        except QuoteError as exc:
            raise AccountError("QUOTE_REFUSED", str(exc), 422) from None
        # The terms hash is unique and a quote is stamped to the second, so asking twice for the same data in one second (a double click)
        # gives the same terms: return the quote that already exists instead of failing on the unique constraint.
        existing = session.scalar(select(Quote).where(Quote.terms_hash == quote["terms_hash"]))
        if existing is not None:
            if existing.org_id != org_id:
                raise AccountError("TRY_AGAIN", "Please ask for the quote again in a moment.", 409)  # never hand back another team's quote
            return {"quote_id": existing.id, "dataset_id": existing.dataset_id, **existing.quote}
        row = Quote(id=str(uuid.uuid4()), org_id=org_id, dataset_id=dataset.id, terms_hash=quote["terms_hash"], quote=quote)
        session.add(row)
        session.commit()
        return {"quote_id": row.id, "dataset_id": dataset.id, **quote}

    @router.post("/quotes/{quote_id}/orders", status_code=201, tags=["billing"])
    def new_order(quote_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        row = session.get(Quote, quote_id)
        if row is None:
            raise AccountError("NOT_FOUND", "No such quote.", 404)
        accounts.require_role(session, who, row.org_id, "owner", "admin")  # another team's quote looks missing
        try:
            verify_quote(row.quote)
        except QuoteError as exc:
            raise AccountError("QUOTE_EXPIRED", f"{exc}. Ask for a new quote.", 410) from None
        rail, key_id = payment_rail(), razorpay_key_id()
        if rail is None or not key_id:
            raise AccountError("PAYMENTS_UNAVAILABLE", "Payments aren't set up on this server yet.", 503)
        from .payments import PaymentError

        try:
            order = rail.create_order(session, org_id=row.org_id, quote_id=row.id, quote=row.quote)
        except PaymentError as exc:
            raise AccountError(exc.code, exc.detail, exc.status) from None
        return {"order_id": order.id, "gateway_order_id": order.gateway_order_id, "key_id": key_id,
                "amount_paise": order.amount_paise, "currency": order.currency, "status": order.status}

    @router.get("/orders/{order_id}", tags=["billing"])
    def order_status(order_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> dict[str, Any]:
        """Polled by the browser after Checkout until Razorpay's webhook marks the order paid (HELD)."""
        from .payments import PaymentOrder

        order = session.get(PaymentOrder, order_id)
        if order is None:
            raise AccountError("NOT_FOUND", "No such order.", 404)
        accounts.require_role(session, who, order.org_id, "owner", "admin", "member")
        return {"order_id": order.id, "status": order.status, "job_id": order.job_id if order.job_id and not order.job_id.startswith("claim-") else None}

    return router
