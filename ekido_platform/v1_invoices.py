"""`GET /v1/orgs/{org_id}/invoices/{order_id}`: a paid order's GST invoice as a PDF (ticket C8, docs/api/v1.yaml, tag billing).

The first request for an order gives it an invoice number and a date and stores them, so asking again returns the same invoice. Numbers
run in order within an Indian financial year (EKI/2026-27/000001...) and are assigned in one SQL step, so two requests never get the
same number. The PDF is built from the order's own quote (`invoices.build_invoice` checks it first) and is marked as a test invoice
until `INVOICE_TEST_MODE=false` and the company has a GSTIN. A refunded order gets no invoice; an unpaid one neither.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from fastapi import APIRouter, Depends, Response
from sqlalchemy import JSON, DateTime, Integer, String, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column

from . import accounts
from .accounts import AccountError, Org, Principal
from .invoices import InvoiceError, Party, build_invoice, financial_year, invoice_number, render_pdf
from .models import Base, Timestamped, utcnow
from .payments import HELD, SETTLED, PaymentOrder


class InvoiceRecord(Timestamped, Base):
    """The number and date given to an order's invoice, so the same order always gets the same invoice."""

    __tablename__ = "invoice_records"
    order_id: Mapped[str] = mapped_column(String(36), unique=True)
    org_id: Mapped[str] = mapped_column(String(32), index=True)
    number: Mapped[str] = mapped_column(String(32), unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    buyer_name: Mapped[str] = mapped_column(String(80))
    buyer_address: Mapped[list] = mapped_column(JSON, default=list)


class InvoiceCounter(Base):
    __tablename__ = "invoice_counters"
    financial_year: Mapped[str] = mapped_column(String(8), primary_key=True)
    last: Mapped[int] = mapped_column(Integer, default=0)


def next_sequence(session: Session, fy: str) -> int:
    """The next number of this financial year, taken in one UPDATE so concurrent callers cannot share one."""
    if session.get(InvoiceCounter, fy) is None:
        try:
            with session.begin_nested():
                session.add(InvoiceCounter(financial_year=fy, last=0))
                session.flush()
        except IntegrityError:
            pass  # someone else created it first
    return session.execute(update(InvoiceCounter).where(InvoiceCounter.financial_year == fy).values(last=InvoiceCounter.last + 1).returning(InvoiceCounter.last)).scalar_one()


def issue(session: Session, order: PaymentOrder, org: Org, now: datetime | None = None) -> InvoiceRecord:
    existing = session.scalar(select(InvoiceRecord).where(InvoiceRecord.order_id == order.id))
    if existing is not None:
        return existing
    now = now or utcnow()
    number = invoice_number(next_sequence(session, financial_year(now)), now)
    record = InvoiceRecord(order_id=order.id, org_id=org.id, number=number, issued_at=now, buyer_name=org.name, buyer_address=[])
    session.add(record)
    session.commit()
    return record


def build_invoices_router(engine: Engine, principal: Callable, *, seller: Party, test_mode: bool = True) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def db():
        with Session(engine) as session:
            yield session

    @router.get("/orgs/{org_id}/invoices/{order_id}")
    def get_invoice(org_id: str, order_id: str, who: Principal = Depends(principal), session: Session = Depends(db)) -> Response:
        accounts.require_role(session, who, org_id, *accounts.ROLES)
        order = session.get(PaymentOrder, order_id)
        org = session.get(Org, org_id)
        if order is None or order.org_id != org_id or org is None:
            raise AccountError("NOT_FOUND", "No such invoice.", 404)  # another team's order looks missing, not forbidden
        if order.status not in (HELD, SETTLED):
            raise AccountError("NOT_INVOICEABLE", "Only a paid order has an invoice." if order.status != "REFUNDED" else "This order was refunded, so there is no invoice.", 409)
        record = issue(session, order, org)
        try:
            invoice = build_invoice(order.quote, number=record.number, issued_at=record.issued_at, buyer=Party(record.buyer_name, tuple(record.buyer_address)), seller=seller, test_mode=test_mode)
            pdf = render_pdf(invoice)
        except InvoiceError as exc:
            raise AccountError("INVOICE_UNAVAILABLE", str(exc), 422) from None
        filename = record.number.replace("/", "-")
        return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="invoice-{filename}.pdf"', "Cache-Control": "no-store"})

    return router
