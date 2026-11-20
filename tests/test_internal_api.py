"""The Coordinator-only routes: erasure tasks and settling or refunding a paid order."""

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core.quotes import make_quote

from ekido_platform import privacy
from ekido_platform.accounts import JobOwner, Org, OrgMember
from ekido_platform.main import Settings, create_app
from ekido_platform.models import User
from ekido_platform.payments import PaymentOrder

TOKEN = "internal-secret-token"
H = {"X-Internal-Token": TOKEN}


class FakeRefunds:
    def __init__(self):
        self.refunds = []

    def refund_payment(self, payment_id, amount_paise, *, receipt):
        self.refunds.append((payment_id, amount_paise, receipt))
        return {"id": "rfnd_1", "amount": amount_paise}


def quote(rows=300):
    return make_quote(recipe_id="ticket-sorter-hinglish", base_model="Qwen/Qwen2.5-0.5B-Instruct", train_rows=rows, test_rows=40, dataset_sha256="a" * 64)


class Base(unittest.TestCase):
    def build(self, token=TOKEN, payments=True):
        extra = dict(razorpay_key_id="rzp_test_abc", razorpay_key_secret="s", razorpay_webhook_secret="w") if payments else {}
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, internal_token=token, **extra))
        self.engine = self.app.state.engine
        if payments:
            self.gateway = FakeRefunds()
            self.app.state.payment_rail.gateway = self.gateway
        self.client = TestClient(self.app)

    def setUp(self):
        self.build()

    def order(self, status="HELD", rows=300, job_id=None):
        q = quote(rows)
        with Session(self.engine) as s:
            row = PaymentOrder(org_id="org-1", quote_id="q", terms_hash=q["terms_hash"], amount_paise=q["total_paise"], currency="INR", gateway_order_id=f"order_{rows}_{status}",
                               gateway_payment_id=f"pay_{rows}_{status}", status=status, quote=q, job_id=job_id)
            s.add(row)
            s.commit()
            return row.id, q


class Guard(Base):
    def test_every_route_is_a_404_without_the_right_token_and_when_no_token_is_set(self):
        order_id, _ = self.order()
        calls = [("get", "/internal/erasure/tasks"), ("post", "/internal/erasure/tasks/x/complete"), ("post", "/internal/erasure/tasks/x/failed"),
                 ("post", f"/internal/payments/orders/{order_id}/job"), ("post", "/internal/payments/jobs/j/settle"), ("post", "/internal/payments/jobs/j/refund")]
        for headers in ({}, {"X-Internal-Token": "wrong"}, {"X-Internal-Token": TOKEN + "x"}):
            for method, path in calls:
                with self.subTest(path=path, headers=headers):
                    kwargs = {"json": {"job_id": "j", "payouts": {"a": 1}}} if method == "post" else {}
                    self.assertEqual(getattr(self.client, method)(path, headers=headers, **kwargs).status_code, 404)
        self.build(token="")  # no token configured: closed for everyone, even with a header
        for method, path in calls:
            kwargs = {"json": {"job_id": "j", "payouts": {"a": 1}}} if method == "post" else {}
            self.assertEqual(getattr(self.client, method)(path, headers={"X-Internal-Token": ""}, **kwargs).status_code, 404)
            self.assertEqual(getattr(self.client, method)(path, headers=H, **kwargs).status_code, 404)


class Erasure(Base):
    def make_task(self, ref="job-1"):
        with Session(self.engine) as s:
            user = User()
            s.add(user)
            s.flush()
            org = Org(name="T")
            s.add(org)
            s.flush()
            s.add(OrgMember(org_id=org.id, user_id=user.id, role="owner"))
            s.add(JobOwner(job_id=ref, org_id=org.id))
            s.commit()
            request = privacy.request_deletion(s, user.id)
            return request.id

    def test_tasks_are_listed_failures_are_counted_and_completing_finishes_the_request(self):
        self.make_task("job-1")
        listed = self.client.get("/internal/erasure/tasks", headers=H).json()
        self.assertEqual([(t["kind"], t["ref"], t["attempts"]) for t in listed], [("job_data", "job-1", 0)])
        task = listed[0]["id"]
        self.assertEqual(self.client.post(f"/internal/erasure/tasks/{task}/failed", headers=H).json()["attempts"], 1)
        self.assertEqual(self.client.post(f"/internal/erasure/tasks/{task}/failed", headers=H).json()["attempts"], 2)
        done = self.client.post(f"/internal/erasure/tasks/{task}/complete", headers=H).json()
        self.assertEqual(done["request_status"], "DONE")
        self.assertEqual(self.client.get("/internal/erasure/tasks", headers=H).json(), [])
        self.assertEqual(self.client.post(f"/internal/erasure/tasks/{task}/complete", headers=H).json()["request_status"], "DONE")  # repeating is safe

    def test_unknown_tasks_are_404(self):
        self.assertEqual(self.client.post("/internal/erasure/tasks/nope/complete", headers=H).status_code, 404)
        self.assertEqual(self.client.post("/internal/erasure/tasks/nope/failed", headers=H).status_code, 404)

    def test_the_list_is_bounded(self):
        self.assertEqual(self.client.get("/internal/erasure/tasks?limit=0", headers=H).status_code, 200)
        self.assertEqual(self.client.get("/internal/erasure/tasks?limit=100000", headers=H).status_code, 200)


class Payments(Base):
    def test_a_paid_order_is_linked_to_its_job_once(self):
        order_id, _ = self.order()
        self.assertEqual(self.client.post(f"/internal/payments/orders/{order_id}/job", headers=H, json={"job_id": "job-1"}).json(), {"order_id": order_id, "job_id": "job-1"})
        self.assertEqual(self.client.post(f"/internal/payments/orders/{order_id}/job", headers=H, json={"job_id": "job-1"}).status_code, 200)  # the same link again
        other = self.client.post(f"/internal/payments/orders/{order_id}/job", headers=H, json={"job_id": "job-2"})
        self.assertEqual((other.status_code, other.json()["detail"]["error"]), (409, "ALREADY_LINKED"))
        second, _ = self.order(rows=200)
        taken = self.client.post(f"/internal/payments/orders/{second}/job", headers=H, json={"job_id": "job-1"})
        self.assertEqual((taken.status_code, taken.json()["detail"]["error"]), (409, "JOB_TAKEN"))

    def test_an_unpaid_or_unknown_order_cannot_start_a_job(self):
        created, _ = self.order(status="CREATED", rows=110)
        response = self.client.post(f"/internal/payments/orders/{created}/job", headers=H, json={"job_id": "j"})
        self.assertEqual((response.status_code, response.json()["detail"]["error"]), (409, "NOT_PAID"))
        self.assertEqual(self.client.post("/internal/payments/orders/missing/job", headers=H, json={"job_id": "j"}).status_code, 404)
        self.assertEqual(self.client.post(f"/internal/payments/orders/{created}/job", headers=H, json={"job_id": "j", "x": 1}).status_code, 422)

    def test_a_job_that_passes_settles_its_order_and_repeating_is_safe(self):
        order_id, q = self.order(job_id="job-1")
        split = {"provider-a": q["provider_budget_paise"] // 3, "provider-b": q["provider_budget_paise"] - q["provider_budget_paise"] // 3}
        first = self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": split})
        self.assertEqual((first.status_code, first.json()["status"]), (200, "SETTLED"))
        self.assertEqual(self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": split}).status_code, 200)
        different = self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": {"x": q["provider_budget_paise"]}})
        self.assertEqual((different.status_code, different.json()["detail"]["error"]), (409, "ALREADY_SETTLED"))
        self.assertEqual(self.gateway.refunds, [])

    def test_a_split_that_does_not_add_up_is_refused_and_the_order_stays_held(self):
        order_id, q = self.order(job_id="job-1")
        for bad in ({"a": q["provider_budget_paise"] - 1}, {}, {"a": 0, "b": q["provider_budget_paise"]}):
            response = self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": bad})
            self.assertEqual((response.status_code, response.json()["detail"]["error"]), (422, "PAYOUT_MISMATCH"))
        self.assertEqual(self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": {"a": 1.5}}).status_code, 422)  # strict: whole paise
        with Session(self.engine) as s:
            self.assertEqual(s.scalar(select(PaymentOrder.status)), "HELD")

    def test_a_job_that_fails_refunds_the_customer_in_full_once(self):
        order_id, q = self.order(job_id="job-1")
        for _ in range(2):
            response = self.client.post("/internal/payments/jobs/job-1/refund", headers=H)
            self.assertEqual((response.status_code, response.json()["status"]), (200, "REFUNDED"))
        self.assertEqual(self.gateway.refunds, [("pay_300_HELD", q["total_paise"], f"refund-{order_id}")])

    def test_a_job_with_no_linked_order_is_404_and_a_settled_order_cannot_be_refunded(self):
        self.assertEqual(self.client.post("/internal/payments/jobs/nobody/refund", headers=H).status_code, 404)
        self.assertEqual(self.client.post("/internal/payments/jobs/nobody/settle", headers=H, json={"payouts": {"a": 1}}).status_code, 404)
        order_id, q = self.order(job_id="job-9")
        self.client.post("/internal/payments/jobs/job-9/settle", headers=H, json={"payouts": {"a": q["provider_budget_paise"]}})
        response = self.client.post("/internal/payments/jobs/job-9/refund", headers=H)
        self.assertEqual((response.status_code, response.json()["detail"]["error"]), (409, "NOT_HELD"))

    def test_without_payments_configured_the_payment_routes_say_so(self):
        self.build(payments=False)
        order_id, _ = self.order(job_id="job-1")
        self.assertEqual(self.client.post("/internal/payments/jobs/job-1/refund", headers=H).status_code, 503)
        self.assertEqual(self.client.post("/internal/payments/jobs/job-1/settle", headers=H, json={"payouts": {"a": 1}}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
