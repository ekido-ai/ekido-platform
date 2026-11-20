"""Paying in rupees: a payment port, a Razorpay rail, signed webhooks, a ledger and reconciliation (ticket C7).

The flow for one job's quote, in whole paise:

1. `create_order`: the quote is verified and a Razorpay order is made for its total (taxable plus GST).
2. The customer pays on Razorpay Checkout (UPI or card). Razorpay calls our webhook.
3. `handle_webhook`: the signature is checked, a repeated event is ignored, and for a capture the payment is
   fetched again from Razorpay. Only if its order, amount and currency match our order is the money marked HELD.
   The webhook body alone is never trusted.
4. When the quality check is done: `settle` (it passed) records the split in the ledger, or `refund` (it failed)
   returns the full amount to the customer through Razorpay.
5. `reconcile` compares Razorpay's list of payments with our orders and ledger and reports every difference.

Provider earnings are recorded in the ledger and paid out later, when Razorpay Route and the company are ready, so no
transfer is made here. The escrow contract and the coordinator's settlement code are not touched; the crypto rail
(`CryptoEscrowRail`) is a separate adapter in the coordinator and is not part of this module.

Test mode only: `RazorpayHttpGateway` refuses any key that doesn't start with `rzp_test_`. Keys and the webhook secret
come from the environment (AWS SSM on the servers) and are never logged or stored.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from ekido_core.quotes import QuoteError, verify_quote

from .models import Base, Timestamped, new_id, utcnow

log = logging.getLogger(__name__)

CREATED, HELD, SETTLED, REFUNDED, NEEDS_REVIEW = "CREATED", "HELD", "SETTLED", "REFUNDED", "NEEDS_REVIEW"


class PaymentError(ValueError):
    """A payment action that can't be done. `code` is a fixed string the API can return; the message is safe to show."""

    def __init__(self, code: str, detail: str, status: int = 400):
        super().__init__(detail)
        self.code, self.detail, self.status = code, detail, status


# ---------------------------------------------------------------------------------------------------- tables


class PaymentOrder(Timestamped, Base):
    __tablename__ = "payment_orders"
    org_id: Mapped[str] = mapped_column(String(36), index=True)
    quote_id: Mapped[str] = mapped_column(String(64))
    terms_hash: Mapped[str] = mapped_column(String(64), index=True)
    amount_paise: Mapped[int] = mapped_column(BigInteger)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    gateway_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    gateway_payment_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    job_id: Mapped[str | None] = mapped_column(String(36), unique=True)  # the Coordinator job started from this paid order
    status: Mapped[str] = mapped_column(String(24), default=CREATED)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    quote: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class WebhookEvent(Base):
    """Every webhook delivery we have handled, so a repeat is acknowledged and ignored."""

    __tablename__ = "webhook_events"
    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event: Mapped[str] = mapped_column(String(60))
    outcome: Mapped[str] = mapped_column(String(40))


class LedgerEntry(Timestamped, Base):
    """Money moves as signed whole paise: CAPTURE and REFUND on the customer's side, PROVIDER_EARNING, EKIDO_FEE and GST
    when a job passes. One entry per (order, kind, account), so settling twice cannot record anything twice."""

    __tablename__ = "ledger_entries"
    __table_args__ = (UniqueConstraint("order_id", "kind", "account"),)
    order_id: Mapped[str] = mapped_column(ForeignKey("payment_orders.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    account: Mapped[str] = mapped_column(String(80), default="")  # the payee for PROVIDER_EARNING, else ""
    amount_paise: Mapped[int] = mapped_column(BigInteger)


# ---------------------------------------------------------------------------------------------------- gateway


class Gateway(Protocol):
    """What we use from Razorpay. Tests pass a fake; production passes `RazorpayHttpGateway`."""

    def create_order(self, *, amount_paise: int, currency: str, receipt: str, notes: dict) -> dict:
        """Make a Razorpay order; returns its id, amount and currency."""

    def fetch_payment(self, payment_id: str) -> dict:
        """One payment as Razorpay has it: id, order_id, amount, currency, status."""

    def refund_payment(self, payment_id: str, amount_paise: int, *, receipt: str) -> dict:
        """Refund a captured payment; returns the refund with its amount."""

    def list_payments(self, *, from_ts: int, to_ts: int) -> list[dict]:
        """Every payment in the time window."""

    def create_subscription(self, *, plan_id: str, total_count: int, notes: dict) -> dict:
        """Start a monthly subscription on a Razorpay plan; returns its id and status."""

    def fetch_subscription(self, subscription_id: str) -> dict:
        """A subscription as Razorpay has it: id, plan_id, status, current_start, current_end."""

    def cancel_subscription(self, subscription_id: str, *, at_cycle_end: bool) -> dict:
        """Cancel now, or when the paid period ends."""

    def update_subscription(self, subscription_id: str, *, plan_id: str) -> dict:
        """Move a subscription to another Razorpay plan, from now."""


class RazorpayHttpGateway:
    """Razorpay's REST API with HTTP basic auth. Refuses live keys: this build is test mode only."""

    def __init__(self, key_id: str, key_secret: str, *, base_url: str = "https://api.razorpay.com/v1", timeout: float = 20.0,
                 opener: Callable[[urllib.request.Request, float], Any] | None = None):
        if not key_id.startswith("rzp_test_"):
            raise PaymentError("LIVE_KEY_REFUSED", "only Razorpay test keys (rzp_test_...) are accepted in this build", 500)
        if not key_secret:
            raise PaymentError("NOT_CONFIGURED", "the Razorpay key secret is missing", 500)
        self._auth = "Basic " + base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
        self._base, self._timeout = base_url.rstrip("/"), timeout
        self._open = opener or (lambda request, timeout: urllib.request.urlopen(request, timeout=timeout))  # noqa: S310 (fixed https base)

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self._base + path, data=data, method=method, headers={"Authorization": self._auth, "Content-Type": "application/json"})
        try:
            with self._open(request, self._timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            # Razorpay's error text can name fields; the status is enough, and nothing from the request is echoed.
            raise PaymentError("GATEWAY_REFUSED", f"Razorpay refused the request ({exc.code})", 502) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise PaymentError("GATEWAY_UNREACHABLE", "Razorpay could not be reached", 502) from None

    def create_order(self, *, amount_paise: int, currency: str, receipt: str, notes: dict) -> dict:
        return self._call("POST", "/orders", {"amount": amount_paise, "currency": currency, "receipt": receipt, "notes": notes})

    def fetch_payment(self, payment_id: str) -> dict:
        return self._call("GET", f"/payments/{payment_id}")

    def refund_payment(self, payment_id: str, amount_paise: int, *, receipt: str) -> dict:
        return self._call("POST", f"/payments/{payment_id}/refund", {"amount": amount_paise, "receipt": receipt})

    def create_subscription(self, *, plan_id: str, total_count: int, notes: dict) -> dict:
        return self._call("POST", "/subscriptions", {"plan_id": plan_id, "total_count": total_count, "customer_notify": 1, "notes": notes})

    def fetch_subscription(self, subscription_id: str) -> dict:
        return self._call("GET", f"/subscriptions/{subscription_id}")

    def cancel_subscription(self, subscription_id: str, *, at_cycle_end: bool) -> dict:
        return self._call("POST", f"/subscriptions/{subscription_id}/cancel", {"cancel_at_cycle_end": 1 if at_cycle_end else 0})

    def update_subscription(self, subscription_id: str, *, plan_id: str) -> dict:
        return self._call("PATCH", f"/subscriptions/{subscription_id}", {"plan_id": plan_id, "schedule_change_at": "now"})

    def list_payments(self, *, from_ts: int, to_ts: int) -> list[dict]:
        found, skip = [], 0
        while True:
            page = self._call("GET", f"/payments?from={from_ts}&to={to_ts}&count=100&skip={skip}").get("items", [])
            found += page
            if len(page) < 100:
                return found
            skip += 100


# ---------------------------------------------------------------------------------------------------- the rail


def verify_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    """Razorpay signs the exact raw body with HMAC-SHA256 and the webhook secret, hex encoded."""
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip())


@dataclass
class Reconciliation:
    checked: int = 0
    unknown_payments: list[str] = field(default_factory=list)  # captured at Razorpay, no order here
    not_held: list[str] = field(default_factory=list)  # captured at Razorpay, our order isn't HELD or SETTLED
    amount_mismatch: list[str] = field(default_factory=list)
    refund_mismatch: list[str] = field(default_factory=list)  # we say REFUNDED, Razorpay doesn't show a refund (or the reverse)
    missing_at_gateway: list[str] = field(default_factory=list)  # we hold it, Razorpay's list doesn't have it
    ledger_unbalanced: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any((self.unknown_payments, self.not_held, self.amount_mismatch, self.refund_mismatch, self.missing_at_gateway, self.ledger_unbalanced))

    def as_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked, "unknown_payments": self.unknown_payments, "not_held": self.not_held,
                "amount_mismatch": self.amount_mismatch, "refund_mismatch": self.refund_mismatch,
                "missing_at_gateway": self.missing_at_gateway, "ledger_unbalanced": self.ledger_unbalanced}


class RazorpayRail:
    def __init__(self, gateway: Gateway, webhook_secret: str, on_subscription_event=None):
        self.gateway, self._webhook_secret = gateway, webhook_secret
        # Called with (session, subscription_id) for every subscription.* event; the plans service syncs from Razorpay.
        self.on_subscription_event = on_subscription_event

    # -- 1. order
    def create_order(self, session: Session, *, org_id: str, quote_id: str, quote: dict) -> PaymentOrder:
        try:
            verify_quote(quote)  # integrity and not expired: a new order needs a live quote
        except (QuoteError, KeyError, ValueError) as exc:
            raise PaymentError("QUOTE_INVALID", f"this quote can't be paid: {exc}", 410 if "expired" in str(exc) else 422) from None
        if quote.get("currency") != "INR" or type(quote.get("total_paise")) is not int or quote["total_paise"] <= 0:
            raise PaymentError("QUOTE_INVALID", "only a rupee quote with a positive total can be paid", 422)
        existing = session.scalar(select(PaymentOrder).where(PaymentOrder.org_id == org_id, PaymentOrder.terms_hash == quote["terms_hash"], PaymentOrder.status.in_((CREATED, HELD, SETTLED))))
        if existing is not None:
            return existing  # the same terms, the same team: one order, so a double click can't charge twice
        order = PaymentOrder(org_id=org_id, quote_id=quote_id, terms_hash=quote["terms_hash"], amount_paise=quote["total_paise"], currency="INR", gateway_order_id=f"pending:{new_id()}", quote=quote)
        session.add(order)
        session.flush()
        remote = self.gateway.create_order(amount_paise=order.amount_paise, currency="INR", receipt=order.id, notes={"terms_hash": quote["terms_hash"]})
        if remote.get("amount") != order.amount_paise or remote.get("currency") != "INR" or not remote.get("id"):
            session.rollback()
            raise PaymentError("GATEWAY_MISMATCH", "Razorpay returned an order that doesn't match the quote", 502)
        order.gateway_order_id = remote["id"]
        session.commit()
        return order

    # -- 2 and 3. webhook
    def handle_webhook(self, session: Session, raw_body: bytes, signature: str | None, event_id: str | None = None) -> str:
        """Returns what happened. Raises PaymentError(BAD_SIGNATURE) for an unsigned or forged call."""
        if not verify_signature(raw_body, signature, self._webhook_secret):
            raise PaymentError("BAD_SIGNATURE", "the webhook signature is not valid", 401)
        try:
            body = json.loads(raw_body)
            event = str(body["event"])
        except (ValueError, KeyError, TypeError):
            raise PaymentError("BAD_EVENT", "the webhook body is not a Razorpay event", 400) from None
        key = (event_id or "").strip() or "sha256:" + hashlib.sha256(raw_body).hexdigest()
        if session.get(WebhookEvent, key) is not None:
            return "duplicate"
        entity = (((body.get("payload") or {}).get("payment") or {}).get("entity")) or {}
        if event == "payment.captured":
            outcome = self._on_captured(session, entity)
        elif event == "payment.failed":
            outcome = self._on_failed(session, entity)
        elif event.startswith("subscription."):
            sub = (((body.get("payload") or {}).get("subscription") or {}).get("entity")) or {}
            outcome = self.on_subscription_event(session, sub.get("id")) if self.on_subscription_event and sub.get("id") else "ignored"
        else:
            outcome = "ignored"
        session.add(WebhookEvent(event_id=key, event=event[:60], outcome=outcome[:40]))
        session.commit()
        return outcome

    def _order_for(self, session: Session, gateway_order_id: str | None) -> PaymentOrder | None:
        if not gateway_order_id:
            return None
        return session.scalar(select(PaymentOrder).where(PaymentOrder.gateway_order_id == gateway_order_id))

    def _on_failed(self, session: Session, entity: dict) -> str:
        order = self._order_for(session, entity.get("order_id"))
        if order is None:
            return "unknown_order"
        if order.status == CREATED:
            order.failed_attempts += 1  # the customer can try again on the same order
        return "failed_recorded"

    def _on_captured(self, session: Session, entity: dict) -> str:
        order = self._order_for(session, entity.get("order_id"))
        payment_id = entity.get("id")
        if order is None or not payment_id:
            return "unknown_order"
        if order.gateway_payment_id == payment_id and order.status in (HELD, SETTLED, REFUNDED):
            return "already_held"
        remote = self.gateway.fetch_payment(payment_id)  # never trust the webhook's own numbers
        if not (remote.get("id") == payment_id and remote.get("order_id") == order.gateway_order_id
                and remote.get("amount") == order.amount_paise and remote.get("currency") == order.currency and remote.get("status") == "captured"):
            order.status = NEEDS_REVIEW
            log.warning("payment %s did not match order %s; flagged for review", payment_id, order.id)
            return "needs_review"
        if order.gateway_payment_id not in (None, payment_id):
            order.status = NEEDS_REVIEW  # a second, different payment for one order
            return "needs_review"
        order.gateway_payment_id, order.status = payment_id, HELD
        self._add(session, order, "CAPTURE", "", order.amount_paise)
        return "held"

    # -- 4. the quality check is done
    def settle(self, session: Session, order_id: str, payouts: dict[str, int]) -> PaymentOrder:
        """The job passed. `payouts` maps each payee to whole paise; together they must equal the provider budget."""
        order = self._get(session, order_id)
        budget, fee, gst = order.quote["provider_budget_paise"], order.quote["ekido_fee_paise"], order.quote["gst_paise"]
        if order.status == SETTLED:
            if self._earnings(session, order) == dict(payouts):
                return order  # the same settlement again: nothing to do
            raise PaymentError("ALREADY_SETTLED", "this order was already settled with a different split", 409)
        if order.status != HELD:
            raise PaymentError("NOT_HELD", f"only a paid order can be settled; this one is {order.status}", 409)
        if any(type(v) is not int or v <= 0 for v in payouts.values()) or not payouts or sum(payouts.values()) != budget:
            raise PaymentError("PAYOUT_MISMATCH", "the payouts must be positive whole paise that add up to the provider budget", 422)
        if budget + fee + gst != order.amount_paise:
            raise PaymentError("QUOTE_INVALID", "the quote's parts don't add up to what was paid", 422)
        for payee, amount in sorted(payouts.items()):
            self._add(session, order, "PROVIDER_EARNING", payee, amount)
        self._add(session, order, "EKIDO_FEE", "", fee)
        self._add(session, order, "GST", "", gst)
        order.status = SETTLED
        session.commit()
        return order

    def refund(self, session: Session, order_id: str, reason: str = "quality check failed") -> PaymentOrder:
        """The job failed (or was cancelled): the full amount goes back to the customer. Safe to call again."""
        order = self._get(session, order_id)
        if order.status == REFUNDED:
            return order
        if order.status != HELD or not order.gateway_payment_id:
            raise PaymentError("NOT_HELD", f"only a paid order can be refunded; this one is {order.status}", 409)
        remote = self.gateway.refund_payment(order.gateway_payment_id, order.amount_paise, receipt=f"refund-{order.id}")  # raises: stays HELD, retry later
        if remote.get("amount") != order.amount_paise:
            raise PaymentError("GATEWAY_MISMATCH", "Razorpay refunded a different amount than was asked", 502)
        self._add(session, order, "REFUND", "", -order.amount_paise)
        order.status = REFUNDED
        log.info("order %s refunded: %s", order.id, reason)
        session.commit()
        return order

    # -- 5. reconcile
    def reconcile(self, session: Session, *, from_ts: int, to_ts: int) -> Reconciliation:
        report = Reconciliation()
        seen_orders = set()
        for payment in self.gateway.list_payments(from_ts=from_ts, to_ts=to_ts):
            if payment.get("status") not in ("captured", "refunded"):
                continue
            report.checked += 1
            order = self._order_for(session, payment.get("order_id"))
            if order is None:
                report.unknown_payments.append(payment.get("id", "?"))
                continue
            seen_orders.add(order.id)
            if payment.get("amount") != order.amount_paise:
                report.amount_mismatch.append(order.id)
            if order.status in (CREATED, NEEDS_REVIEW):
                report.not_held.append(order.id)
            refunded_remotely = payment.get("status") == "refunded" or bool(payment.get("amount_refunded"))
            if refunded_remotely != (order.status == REFUNDED):
                report.refund_mismatch.append(order.id)
        window = (datetime.fromtimestamp(from_ts, timezone.utc), datetime.fromtimestamp(to_ts, timezone.utc))
        for order in session.scalars(select(PaymentOrder).where(PaymentOrder.status.in_((HELD, SETTLED, REFUNDED)))):
            created = order.created_at if order.created_at.tzinfo else order.created_at.replace(tzinfo=timezone.utc)
            if window[0] <= created <= window[1] and order.id not in seen_orders:
                report.missing_at_gateway.append(order.id)
            if not self._balanced(session, order):
                report.ledger_unbalanced.append(order.id)
        return report

    # -- helpers
    def _get(self, session: Session, order_id: str) -> PaymentOrder:
        order = session.get(PaymentOrder, order_id)
        if order is None:
            raise PaymentError("NOT_FOUND", "no such order", 404)
        return order

    def _add(self, session: Session, order: PaymentOrder, kind: str, account: str, amount: int) -> None:
        session.add(LedgerEntry(order_id=order.id, kind=kind, account=account, amount_paise=amount))
        session.flush()

    def _earnings(self, session: Session, order: PaymentOrder) -> dict[str, int]:
        rows = session.scalars(select(LedgerEntry).where(LedgerEntry.order_id == order.id, LedgerEntry.kind == "PROVIDER_EARNING"))
        return {row.account: row.amount_paise for row in rows}

    def _balanced(self, session: Session, order: PaymentOrder) -> bool:
        rows = list(session.scalars(select(LedgerEntry).where(LedgerEntry.order_id == order.id)))
        kinds = {}
        for row in rows:
            kinds[row.kind] = kinds.get(row.kind, 0) + row.amount_paise
        if order.status == HELD:
            return kinds == {"CAPTURE": order.amount_paise}
        if order.status == REFUNDED:
            return kinds == {"CAPTURE": order.amount_paise, "REFUND": -order.amount_paise}
        if order.status == SETTLED:
            return kinds.get("CAPTURE") == order.amount_paise and kinds.get("PROVIDER_EARNING", 0) + kinds.get("EKIDO_FEE", 0) + kinds.get("GST", 0) == order.amount_paise
        return True
