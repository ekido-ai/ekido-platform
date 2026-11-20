"""C7: the Razorpay rail against a fake gateway. The cases follow the A5 matrix (UPI, card, failed payment, a pass with a
payout split, a fail with a refund, a duplicate webhook, a bad signature, reconciliation) plus the edges around them."""

import hashlib
import hmac
import itertools
import json
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core.canonical import document_hash
from ekido_core.quotes import make_quote

from ekido_platform import payments
from ekido_platform.main import Settings, create_app
from ekido_platform.payments import (CREATED, HELD, NEEDS_REVIEW, REFUNDED, SETTLED, LedgerEntry, PaymentError, PaymentOrder,
                                      RazorpayHttpGateway, RazorpayRail, WebhookEvent, verify_signature)

SECRET = "test-webhook-secret"
ORG = "org-1"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"


class FakeGateway:
    """Behaves like the parts of Razorpay we use, and records every call."""

    def __init__(self):
        self.orders, self.payments, self.refunds, self.calls = {}, {}, {}, []
        self.counter = itertools.count(1)
        self.refund_fails = False
        self.unreachable = False

    def create_order(self, *, amount_paise, currency, receipt, notes):
        self.calls.append("create_order")
        oid = f"order_{next(self.counter):04d}"
        self.orders[oid] = {"id": oid, "amount": amount_paise, "currency": currency, "receipt": receipt, "notes": notes, "status": "created"}
        return dict(self.orders[oid])

    def pay(self, order_id, method="upi", amount=None, status="captured"):
        pid = f"pay_{next(self.counter):04d}"
        order = self.orders[order_id]
        self.payments[pid] = {"id": pid, "order_id": order_id, "amount": order["amount"] if amount is None else amount, "currency": "INR",
                              "status": status, "method": method, "amount_refunded": 0, "created_at": int(datetime.now(timezone.utc).timestamp())}
        return pid

    def fetch_payment(self, payment_id):
        self.calls.append("fetch_payment")
        if self.unreachable:
            raise PaymentError("GATEWAY_UNREACHABLE", "Razorpay could not be reached", 502)
        return dict(self.payments[payment_id])

    def refund_payment(self, payment_id, amount_paise, *, receipt):
        self.calls.append("refund_payment")
        if self.refund_fails:
            raise PaymentError("GATEWAY_UNREACHABLE", "Razorpay could not be reached", 502)
        if receipt in self.refunds:  # Razorpay-style idempotency on the receipt
            return dict(self.refunds[receipt])
        self.payments[payment_id].update(status="refunded", amount_refunded=amount_paise)
        self.refunds[receipt] = {"id": f"rfnd_{next(self.counter):04d}", "payment_id": payment_id, "amount": amount_paise}
        return dict(self.refunds[receipt])

    def list_payments(self, *, from_ts, to_ts):
        return [dict(p) for p in self.payments.values()]


def event(name, payment):
    return json.dumps({"event": name, "payload": {"payment": {"entity": payment}}}, separators=(",", ":")).encode()


def sign(raw, secret=SECRET):
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def quote(rows=300, distill=True):
    return make_quote(recipe_id="ticket-sorter-hinglish", base_model=BASE, train_rows=rows, test_rows=40, dataset_sha256="a" * 64, distill=distill)


class Base(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.engine = self.app.state.engine
        self.gateway = FakeGateway()
        self.rail = RazorpayRail(self.gateway, SECRET)
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)
        self.quote = quote()

    def order(self, q=None):
        return self.rail.create_order(self.session, org_id=ORG, quote_id="q-1", quote=q or self.quote)

    def webhook(self, name, payment, event_id=None):
        raw = event(name, payment)
        return self.rail.handle_webhook(self.session, raw, sign(raw), event_id)

    def paid(self, method="upi"):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id, method)
        self.webhook("payment.captured", self.gateway.payments[pid], f"evt_{pid}")
        self.session.refresh(order)
        return order, pid

    def entries(self, order):
        return [(e.kind, e.account, e.amount_paise) for e in self.session.scalars(select(LedgerEntry).where(LedgerEntry.order_id == order.id).order_by(LedgerEntry.kind, LedgerEntry.account))]

    def split(self, q=None):
        budget = (q or self.quote)["provider_budget_paise"]
        first = budget // 3
        return {"provider-a": first, "provider-b": budget - first}


class OrderTests(Base):
    def test_an_order_is_made_for_the_quote_total_and_carries_its_hash(self):
        order = self.order()
        self.assertEqual((order.status, order.amount_paise, order.terms_hash), (CREATED, self.quote["total_paise"], self.quote["terms_hash"]))
        remote = self.gateway.orders[order.gateway_order_id]
        self.assertEqual((remote["amount"], remote["currency"], remote["notes"]), (self.quote["total_paise"], "INR", {"terms_hash": self.quote["terms_hash"]}))
        self.assertEqual(remote["receipt"], order.id)

    def test_the_same_quote_for_the_same_team_gives_one_order(self):
        self.assertEqual(self.order().id, self.order().id)
        self.assertEqual(self.gateway.calls.count("create_order"), 1)

    def test_a_tampered_expired_or_non_rupee_quote_cannot_be_paid(self):
        bad = dict(self.quote, total_paise=self.quote["total_paise"] - 1)
        with self.assertRaises(PaymentError) as caught:
            self.order(bad)
        self.assertEqual(caught.exception.code, "QUOTE_INVALID")
        old = make_quote(recipe_id="ticket-sorter-hinglish", base_model=BASE, train_rows=100, test_rows=40, dataset_sha256="b" * 64, now=datetime.now(timezone.utc) - timedelta(days=400))
        with self.assertRaises(PaymentError) as caught:
            self.order(old)
        self.assertEqual(caught.exception.status, 410)
        usd = dict(self.quote, currency="USD")
        usd["terms_hash"] = document_hash({k: v for k, v in usd.items() if k != "terms_hash"})
        with self.assertRaises(PaymentError):
            self.order(usd)
        self.assertEqual(self.gateway.calls.count("create_order"), 0)

    def test_a_gateway_order_that_does_not_match_is_refused(self):
        original = self.gateway.create_order
        self.gateway.create_order = lambda **kw: dict(original(**kw), amount=1)
        with self.assertRaises(PaymentError) as caught:
            self.order()
        self.assertEqual(caught.exception.code, "GATEWAY_MISMATCH")
        self.assertEqual(self.session.scalars(select(PaymentOrder)).all(), [])


class PayTests(Base):
    def test_upi_and_card_payments_are_held_and_recorded(self):
        for method in ("upi", "card"):
            with self.subTest(method=method):
                self.setUp()
                order, pid = self.paid(method)
                self.assertEqual((order.status, order.gateway_payment_id), (HELD, pid))
                self.assertEqual(self.entries(order), [("CAPTURE", "", order.amount_paise)])

    def test_a_failed_payment_leaves_the_order_open_for_another_try(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id, "card", status="failed")
        self.assertEqual(self.webhook("payment.failed", self.gateway.payments[pid], "evt_f1"), "failed_recorded")
        self.session.refresh(order)
        self.assertEqual((order.status, order.failed_attempts), (CREATED, 1))
        self.assertEqual(self.entries(order), [])
        retry = self.gateway.pay(order.gateway_order_id, "upi")  # a second attempt on the same order succeeds
        self.webhook("payment.captured", self.gateway.payments[retry], "evt_ok")
        self.session.refresh(order)
        self.assertEqual((order.status, order.gateway_payment_id), (HELD, retry))

    def test_the_webhook_is_not_trusted_the_payment_is_fetched_again(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id)
        lying = dict(self.gateway.payments[pid], amount=1)  # the webhook says 1 paisa, Razorpay says the full amount
        self.webhook("payment.captured", lying, "evt_1")
        self.session.refresh(order)
        self.assertEqual(order.status, HELD)
        self.assertIn("fetch_payment", self.gateway.calls)

    def test_a_payment_that_does_not_match_the_order_needs_review_and_holds_nothing(self):
        for label, tweak in (("amount", lambda p: p.update(amount=p["amount"] - 1)), ("status", lambda p: p.update(status="authorized")),
                             ("currency", lambda p: p.update(currency="USD")), ("order", lambda p: p.update(order_id="order_other"))):
            with self.subTest(label=label):
                self.setUp()
                order = self.order()
                pid = self.gateway.pay(order.gateway_order_id)
                tweak(self.gateway.payments[pid])
                outcome = self.webhook("payment.captured", {"id": pid, "order_id": order.gateway_order_id}, f"evt_{label}")
                self.session.refresh(order)
                self.assertEqual(outcome, "needs_review")
                self.assertEqual(order.status, NEEDS_REVIEW)
                self.assertEqual(self.entries(order), [])

    def test_a_second_different_payment_for_one_order_needs_review(self):
        order, _ = self.paid()
        second = self.gateway.pay(order.gateway_order_id, "card")
        self.assertEqual(self.webhook("payment.captured", self.gateway.payments[second], "evt_two"), "needs_review")
        self.session.refresh(order)
        self.assertEqual(order.status, NEEDS_REVIEW)

    def test_a_payment_for_an_unknown_order_is_recorded_and_ignored(self):
        self.assertEqual(self.webhook("payment.captured", {"id": "pay_x", "order_id": "order_nope"}, "evt_u"), "unknown_order")
        self.assertEqual(self.webhook("order.paid", {}, "evt_o"), "ignored")

    def test_if_razorpay_cannot_be_reached_nothing_is_recorded_so_it_is_retried(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id)
        self.gateway.unreachable = True
        raw = event("payment.captured", self.gateway.payments[pid])
        with self.assertRaises(PaymentError) as caught:
            self.rail.handle_webhook(self.session, raw, sign(raw), "evt_retry")
        self.assertEqual(caught.exception.status, 502)
        self.assertIsNone(self.session.get(WebhookEvent, "evt_retry"))
        self.gateway.unreachable = False
        self.assertEqual(self.rail.handle_webhook(self.session, raw, sign(raw), "evt_retry"), "held")  # the retry works


class WebhookSecurityTests(Base):
    def test_a_duplicate_event_changes_nothing(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id)
        raw = event("payment.captured", self.gateway.payments[pid])
        self.assertEqual(self.rail.handle_webhook(self.session, raw, sign(raw), "evt_1"), "held")
        fetches = self.gateway.calls.count("fetch_payment")
        self.assertEqual(self.rail.handle_webhook(self.session, raw, sign(raw), "evt_1"), "duplicate")
        self.assertEqual(self.gateway.calls.count("fetch_payment"), fetches)
        self.assertEqual(len(self.entries(order)), 1)

    def test_a_replay_without_an_event_id_is_still_deduplicated_by_its_body(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id)
        raw = event("payment.captured", self.gateway.payments[pid])
        self.assertEqual(self.rail.handle_webhook(self.session, raw, sign(raw)), "held")
        self.assertEqual(self.rail.handle_webhook(self.session, raw, sign(raw)), "duplicate")

    def test_the_same_payment_in_a_new_event_does_not_double_count(self):
        order, pid = self.paid()
        self.assertEqual(self.webhook("payment.captured", self.gateway.payments[pid], "evt_other_id"), "already_held")
        self.assertEqual(len(self.entries(order)), 1)

    def test_a_bad_missing_or_wrong_secret_signature_is_refused_and_changes_nothing(self):
        order = self.order()
        pid = self.gateway.pay(order.gateway_order_id)
        raw = event("payment.captured", self.gateway.payments[pid])
        for signature in ("0" * 64, "", None, sign(raw, "another-secret"), sign(raw)[:-1]):
            with self.subTest(signature=signature), self.assertRaises(PaymentError) as caught:
                self.rail.handle_webhook(self.session, raw, signature, "evt_bad")
            self.assertEqual(caught.exception.status, 401)
        with self.assertRaises(PaymentError):
            self.rail.handle_webhook(self.session, raw + b" ", sign(raw), "evt_bad")  # a signed body that was altered
        self.session.refresh(order)
        self.assertEqual(order.status, CREATED)
        self.assertIsNone(self.session.get(WebhookEvent, "evt_bad"))

    def test_no_secret_configured_accepts_nothing(self):
        self.assertFalse(verify_signature(b"x", sign(b"x", ""), ""))
        self.assertFalse(verify_signature(b"x", None, SECRET))

    def test_a_signed_but_malformed_body_is_a_bad_event(self):
        raw = b"not json"
        with self.assertRaises(PaymentError) as caught:
            self.rail.handle_webhook(self.session, raw, sign(raw), "e")
        self.assertEqual(caught.exception.code, "BAD_EVENT")


class SettleAndRefundTests(Base):
    def test_a_job_that_passes_records_the_payout_split_and_the_books_balance(self):
        order, _ = self.paid()
        split = self.split()
        self.rail.settle(self.session, order.id, split)
        self.session.refresh(order)
        self.assertEqual(order.status, SETTLED)
        rows = {(k, a): v for k, a, v in self.entries(order)}
        self.assertEqual(rows[("PROVIDER_EARNING", "provider-a")] + rows[("PROVIDER_EARNING", "provider-b")], self.quote["provider_budget_paise"])
        self.assertEqual(rows[("EKIDO_FEE", "")], self.quote["ekido_fee_paise"])
        self.assertEqual(rows[("GST", "")], self.quote["gst_paise"])
        earned = sum(v for (k, _), v in rows.items() if k != "CAPTURE")
        self.assertEqual(earned, rows[("CAPTURE", "")], "what the customer paid equals what was allocated, to the paisa")
        self.assertNotIn("refund_payment", self.gateway.calls)

    def test_settling_again_is_safe_but_a_different_split_is_refused(self):
        order, _ = self.paid()
        split = self.split()
        self.rail.settle(self.session, order.id, split)
        before = self.entries(order)
        self.rail.settle(self.session, order.id, dict(split))
        self.assertEqual(self.entries(order), before)
        other = {"provider-c": self.quote["provider_budget_paise"]}
        with self.assertRaises(PaymentError) as caught:
            self.rail.settle(self.session, order.id, other)
        self.assertEqual(caught.exception.code, "ALREADY_SETTLED")

    def test_a_split_that_does_not_add_up_is_refused_and_leaves_the_order_held(self):
        order, _ = self.paid()
        budget = self.quote["provider_budget_paise"]
        for bad in ({}, {"a": budget - 1}, {"a": budget + 1}, {"a": 0, "b": budget}, {"a": -5, "b": budget + 5}, {"a": budget / 2, "b": budget / 2}):
            with self.subTest(bad=bad), self.assertRaises(PaymentError) as caught:
                self.rail.settle(self.session, order.id, bad)
            self.assertEqual(caught.exception.code, "PAYOUT_MISMATCH")
        self.session.refresh(order)
        self.assertEqual((order.status, len(self.entries(order))), (HELD, 1))

    def test_only_a_paid_order_can_be_settled_or_refunded(self):
        order = self.order()
        for action in (lambda: self.rail.settle(self.session, order.id, self.split()), lambda: self.rail.refund(self.session, order.id)):
            with self.assertRaises(PaymentError) as caught:
                action()
            self.assertEqual(caught.exception.code, "NOT_HELD")
        with self.assertRaises(PaymentError) as caught:
            self.rail.refund(self.session, "missing")
        self.assertEqual(caught.exception.status, 404)

    def test_a_job_that_fails_gets_the_full_amount_back_automatically(self):
        order, pid = self.paid("card")
        self.rail.refund(self.session, order.id, "quality check failed")
        self.session.refresh(order)
        self.assertEqual(order.status, REFUNDED)
        refund = list(self.gateway.refunds.values())[0]
        self.assertEqual((refund["payment_id"], refund["amount"]), (pid, order.amount_paise))
        self.assertEqual(sum(v for _, _, v in self.entries(order)), 0)

    def test_refunding_twice_refunds_once_and_a_failed_gateway_call_can_be_retried(self):
        order, _ = self.paid()
        self.gateway.refund_fails = True
        with self.assertRaises(PaymentError):
            self.rail.refund(self.session, order.id)
        self.session.refresh(order)
        self.assertEqual((order.status, len(self.entries(order))), (HELD, 1))  # nothing was recorded as refunded
        self.gateway.refund_fails = False
        self.rail.refund(self.session, order.id)
        self.rail.refund(self.session, order.id)
        self.assertEqual(len(self.gateway.refunds), 1)
        self.assertEqual(self.gateway.calls.count("refund_payment"), 2)  # one failed, one done; the third call did nothing
        with self.assertRaises(PaymentError):
            self.rail.settle(self.session, order.id, self.split())  # a refunded order can never be settled

    def test_a_settled_order_cannot_be_refunded(self):
        order, _ = self.paid()
        self.rail.settle(self.session, order.id, self.split())
        with self.assertRaises(PaymentError):
            self.rail.refund(self.session, order.id)


class ReconcileTests(Base):
    def run_report(self):
        now = int(datetime.now(timezone.utc).timestamp())
        return self.rail.reconcile(self.session, from_ts=now - 3600, to_ts=now + 3600)

    def test_everything_matches_after_the_full_matrix(self):
        held, _ = self.paid()
        settled, _ = self.paid_other(rows=200)
        self.rail.settle(self.session, settled.id, self.split(settled.quote))
        refunded, _ = self.paid_other(rows=100)
        self.rail.refund(self.session, refunded.id)
        report = self.run_report()
        self.assertTrue(report.ok, report.as_dict())
        self.assertEqual(report.checked, 3)

    def paid_other(self, rows):
        q = quote(rows, distill=False)
        order = self.rail.create_order(self.session, org_id=ORG, quote_id=f"q-{rows}", quote=q)
        pid = self.gateway.pay(order.gateway_order_id, "card")
        self.webhook("payment.captured", self.gateway.payments[pid], f"evt_{pid}")
        self.session.refresh(order)
        return order, pid

    def test_every_kind_of_difference_is_reported(self):
        self.gateway.payments["pay_ghost"] = {"id": "pay_ghost", "order_id": "order_unknown", "amount": 500, "status": "captured", "currency": "INR"}
        missed = self.order()  # captured at Razorpay, but the webhook never reached us
        self.gateway.pay(missed.gateway_order_id)
        wrong, _ = self.paid_other(rows=150)
        next(p for p in self.gateway.payments.values() if p["order_id"] == wrong.gateway_order_id).update(amount=wrong.amount_paise + 1)
        refunded_here, pid = self.paid_other(rows=120)
        self.rail.refund(self.session, refunded_here.id)
        self.gateway.payments[pid].update(status="captured", amount_refunded=0)  # Razorpay doesn't show that refund
        vanished, vpid = self.paid_other(rows=90)
        del self.gateway.payments[vpid]
        report = self.run_report()
        self.assertFalse(report.ok)
        self.assertEqual(report.unknown_payments, ["pay_ghost"])
        self.assertEqual(report.not_held, [missed.id])
        self.assertEqual(report.amount_mismatch, [wrong.id])
        self.assertEqual(report.refund_mismatch, [refunded_here.id])
        self.assertEqual(report.missing_at_gateway, [vanished.id])

    def test_an_unbalanced_ledger_is_reported(self):
        order, _ = self.paid()
        entry = self.session.scalar(select(LedgerEntry).where(LedgerEntry.order_id == order.id))
        entry.amount_paise -= 1
        self.session.commit()
        self.assertEqual(self.run_report().ledger_unbalanced, [order.id])


class HttpGatewayTests(unittest.TestCase):
    """The real HTTP gateway, with the network replaced: requests have the right shape and secrets stay out of errors."""

    def make(self, reply=None, fail=None):
        self.seen = []

        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(self.body).encode()

        def opener(request, timeout):
            self.seen.append(request)
            if fail:
                raise fail
            return Response(reply if reply is not None else {"id": "order_1", "amount": 100, "currency": "INR"})

        return RazorpayHttpGateway("rzp_test_abc", "s3cret", opener=opener)

    def test_a_live_key_or_empty_secret_is_refused(self):
        for key_id, secret in (("rzp_live_abc", "x"), ("abc", "x"), ("rzp_test_abc", "")):
            with self.subTest(key_id=key_id), self.assertRaises(PaymentError):
                RazorpayHttpGateway(key_id, secret)

    def test_requests_use_basic_auth_json_and_the_right_paths(self):
        gateway = self.make()
        gateway.create_order(amount_paise=15222, currency="INR", receipt="r1", notes={"terms_hash": "h"})
        gateway.fetch_payment("pay_1")
        gateway.refund_payment("pay_1", 15222, receipt="refund-r1")
        create, fetch, refund = self.seen
        self.assertEqual((create.method, create.full_url), ("POST", "https://api.razorpay.com/v1/orders"))
        self.assertEqual(json.loads(create.data), {"amount": 15222, "currency": "INR", "receipt": "r1", "notes": {"terms_hash": "h"}})
        self.assertEqual((fetch.method, fetch.full_url), ("GET", "https://api.razorpay.com/v1/payments/pay_1"))
        self.assertEqual((refund.method, refund.full_url), ("POST", "https://api.razorpay.com/v1/payments/pay_1/refund"))
        self.assertEqual(json.loads(refund.data), {"amount": 15222, "receipt": "refund-r1"})
        import base64
        self.assertEqual(create.get_header("Authorization"), "Basic " + base64.b64encode(b"rzp_test_abc:s3cret").decode())

    def test_list_payments_pages_through_results(self):
        pages = iter([{"items": [{"id": f"p{i}"} for i in range(100)]}, {"items": [{"id": "last"}]}])
        gateway = self.make()
        gateway._call = lambda method, path, body=None: next(pages)
        self.assertEqual(len(gateway.list_payments(from_ts=1, to_ts=2)), 101)

    def test_errors_never_echo_the_secret_or_the_request(self):
        for failure in (urllib.error.HTTPError("u", 400, "bad", {}, None), urllib.error.URLError("down"), TimeoutError()):
            with self.subTest(failure=type(failure).__name__), self.assertRaises(PaymentError) as caught:
                self.make(fail=failure).create_order(amount_paise=1, currency="INR", receipt="r", notes={})
            self.assertNotIn("s3cret", str(caught.exception))
            self.assertNotIn("rzp_test_abc", str(caught.exception))
            self.assertEqual(caught.exception.status, 502)


class WebhookRouteTests(unittest.TestCase):
    def client(self, configured=True):
        settings = Settings(database_url="sqlite://", _env_file=None, razorpay_key_id="rzp_test_abc" if configured else "",
                            razorpay_key_secret="s3cret" if configured else "", razorpay_webhook_secret=SECRET if configured else "")
        app = create_app(settings)
        if configured:
            self.gateway = FakeGateway()
            app.state.payment_rail.gateway = self.gateway  # the HTTP gateway is replaced by the fake
        return TestClient(app), app

    def test_not_configured_answers_503_and_accepts_nothing(self):
        client, _ = self.client(configured=False)
        response = client.post("/v1/payments/razorpay/webhook", content=b"{}", headers={"X-Razorpay-Signature": sign(b"{}")})
        self.assertEqual((response.status_code, response.json()["error"]), (503, "NOT_CONFIGURED"))

    def test_bad_signature_401_good_signature_200_and_duplicate_200(self):
        client, app = self.client()
        with Session(app.state.engine) as session:
            order = app.state.payment_rail.create_order(session, org_id=ORG, quote_id="q", quote=quote())
            oid = order.gateway_order_id
        pid = self.gateway.pay(oid, "upi")
        raw = event("payment.captured", self.gateway.payments[pid])
        url = "/v1/payments/razorpay/webhook"
        bad = client.post(url, content=raw, headers={"X-Razorpay-Signature": "0" * 64})
        self.assertEqual((bad.status_code, bad.json()["error"]), (401, "BAD_SIGNATURE"))
        missing = client.post(url, content=raw)
        self.assertEqual(missing.status_code, 401)
        good = client.post(url, content=raw, headers={"X-Razorpay-Signature": sign(raw), "X-Razorpay-Event-Id": "evt_9"})
        self.assertEqual((good.status_code, good.json()), (200, {"ok": True, "outcome": "held"}))
        again = client.post(url, content=raw, headers={"X-Razorpay-Signature": sign(raw), "X-Razorpay-Event-Id": "evt_9"})
        self.assertEqual((again.status_code, again.json()["outcome"]), (200, "duplicate"))
        with Session(app.state.engine) as session:
            self.assertEqual(session.scalar(select(PaymentOrder)).status, HELD)

    def test_live_keys_stop_the_app_from_starting_payments(self):
        with self.assertRaises(PaymentError):
            create_app(Settings(database_url="sqlite://", _env_file=None, razorpay_key_id="rzp_live_abc", razorpay_key_secret="x", razorpay_webhook_secret="y"))


if __name__ == "__main__":
    unittest.main()
