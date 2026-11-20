"""POST /v1/orgs/{org_id}/jobs: a paid order starts a job, once, for its own team's owners and admins."""

import os
import re
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core.quotes import make_quote

from ekido_platform.accounts import JobOwner
from ekido_platform.main import Settings, create_app
from ekido_platform.payments import PaymentOrder

SITE = "https://ekido.example"
WORKLOAD = {"template": "llm-lora/v1", "recipe_id": "ticket-sorter-hinglish", "model": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "r1"},
            "dataset": {"sha256": "a" * 64, "bytes": 1000, "train_rows": 300, "test_rows": 40}, "total_logical_steps": 57, "logical_batch_size": 16}
BODY = {"workload": WORKLOAD, "policy": {"min": 1}}


class FakeCoordinator:
    def __init__(self):
        self.calls, self.create_status, self.start_status, self.down = [], 200, 200, False

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if self.down:
            raise OSError("down")
        if path == "/jobs":
            return self.create_status, ({"job_id": "job-1", "state": "OPEN"} if self.create_status == 200 else {"detail": "bad spec"})
        if path.endswith("/start"):
            return self.start_status, ({"job_id": "job-1", "state": "ASSIGNED"} if self.start_status == 200 else {"detail": "not funded"})
        return 200, {}


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()  # a file database: threads in the race test each get their own connection, as on Postgres
        self.app = create_app(Settings(database_url=f"sqlite:///{os.path.join(tmp, 'p.db')}", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE], session_cookie_secure=False))
        self.engine = self.app.state.engine
        self.coordinator = FakeCoordinator()
        self.app.state.coordinator_post = self.coordinator
        self.clients = {}

    def client(self, email="asha@example.com"):
        if email not in self.clients:
            c = TestClient(self.app)
            with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
                c.post("/v1/auth/email/start", json={"email": email})
            token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
            self.assertEqual(c.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)
            self.clients[email] = c
        return self.clients[email]

    def org(self, email="asha@example.com"):
        return self.client(email).post("/v1/orgs", json={"name": "Asha Traders"}).json()["org_id"]

    def order(self, org_id, status="HELD", n=[0]):
        n[0] += 1
        q = make_quote(recipe_id="ticket-sorter-hinglish", base_model="Qwen/Qwen2.5-0.5B-Instruct", train_rows=300, test_rows=40, dataset_sha256="a" * 64)
        with Session(self.engine) as s:
            row = PaymentOrder(org_id=org_id, quote_id="q", terms_hash=q["terms_hash"], amount_paise=q["total_paise"], currency="INR", gateway_order_id=f"o{n[0]}", status=status, quote=q)
            s.add(row)
            s.commit()
            return row.id, q

    def start(self, email, org, order_id, **extra):
        return self.client(email).post(f"/v1/orgs/{org}/jobs", json=dict(BODY, order_id=order_id, **extra))

    def order_row(self, order_id):
        with Session(self.engine) as s:
            return s.get(PaymentOrder, order_id).job_id


class Start(Base):
    def test_a_paid_order_starts_a_job_that_the_team_owns_and_the_order_points_to(self):
        org = self.org()
        order_id, q = self.order(org)
        r = self.start("asha@example.com", org, order_id)
        self.assertEqual((r.status_code, r.json()["job_id"], r.json()["state"]), (201, "job-1", "ASSIGNED"))
        self.assertEqual(self.order_row(order_id), "job-1")
        with Session(self.engine) as s:
            self.assertEqual(s.scalar(select(JobOwner.org_id).where(JobOwner.job_id == "job-1")), org)
        (m1, p1, b1), (m2, p2, _) = self.coordinator.calls
        self.assertEqual((m1, p1, p2), ("POST", "/jobs", "/jobs/job-1/start"))
        self.assertEqual(b1["budget_usdc"], f"{q['provider_budget_paise'] / 100:.6f}")
        self.assertEqual(b1["workload"], BODY["workload"])

    def test_a_second_start_from_the_same_order_is_refused_and_calls_nothing(self):
        org = self.org()
        order_id, _ = self.order(org)
        self.start("asha@example.com", org, order_id)
        before = len(self.coordinator.calls)
        r = self.start("asha@example.com", org, order_id)
        self.assertEqual((r.status_code, (r.json().get("error") or r.json().get("code") or r.json().get("detail"))), (409, "ALREADY_STARTED"))
        self.assertEqual(len(self.coordinator.calls), before)

    def test_two_requests_at_once_start_exactly_one_job(self):
        org = self.org()
        order_id, _ = self.order(org)
        results = []
        barrier = threading.Barrier(2)
        clients = [self.client("asha@example.com"), TestClient(self.app)]
        clients[1].cookies.update(clients[0].cookies)  # a second connection for the same signed-in person

        def go(client):
            barrier.wait()
            results.append(client.post(f"/v1/orgs/{org}/jobs", json=dict(BODY, order_id=order_id)).status_code)

        threads = [threading.Thread(target=go, args=(c,)) for c in clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(results), [201, 409])
        self.assertEqual(len([c for c in self.coordinator.calls if c[1] == "/jobs"]), 1)

    def test_an_unpaid_refunded_or_settled_order_cannot_start_a_job(self):
        org = self.org()
        for status in ("CREATED", "REFUNDED", "SETTLED", "FAILED"):
            order_id, _ = self.order(org, status)
            self.assertEqual(self.start("asha@example.com", org, order_id).status_code, 409, status)
        self.assertEqual(self.coordinator.calls, [])

    def test_another_teams_order_looks_missing_and_a_stranger_is_refused(self):
        asha, ravi = self.org("asha@example.com"), self.client("ravi@example.com").post("/v1/orgs", json={"name": "Ravi"}).json()["org_id"]
        order_id, _ = self.order(asha)
        self.assertEqual(self.start("ravi@example.com", ravi, order_id).status_code, 404)  # his own team, her order
        self.assertIn(self.start("ravi@example.com", asha, order_id).status_code, (403, 404))  # her team, not a member
        self.assertEqual(self.start("asha@example.com", asha, "nope").status_code, 404)
        self.assertEqual(TestClient(self.app).post(f"/v1/orgs/{asha}/jobs", json=dict(BODY, order_id=order_id)).status_code, 401)
        self.assertEqual(self.coordinator.calls, [])

    def test_unknown_fields_and_bad_deadlines_are_422(self):
        org = self.org()
        order_id, _ = self.order(org)
        self.assertEqual(self.start("asha@example.com", org, order_id, budget_usdc="9999").status_code, 422)
        self.assertEqual(self.start("asha@example.com", org, order_id, deadline_seconds=0).status_code, 422)


class Workload(Base):
    def attempt(self, **changes):
        org = self.org()
        order_id, _ = self.order(org)
        workload = {**WORKLOAD, **changes}
        r = self.client("asha@example.com").post(f"/v1/orgs/{org}/jobs", json={"order_id": order_id, "workload": workload, "policy": {}})
        return r, order_id

    def test_a_workload_that_differs_from_the_quote_is_refused_before_the_coordinator_is_called(self):
        wrong = {
            "template": {"template": "other"}, "recipe_id": {"recipe_id": "something-else"},
            "model": {"model": {"id": "Big/Model-70B", "revision": "r1"}},
            "dataset sha": {"dataset": {**WORKLOAD["dataset"], "sha256": "b" * 64}},
            "test rows": {"dataset": {**WORKLOAD["dataset"], "test_rows": 41}},
            "more rows": {"dataset": {**WORKLOAD["dataset"], "train_rows": 301}},
            "more steps": {"total_logical_steps": 58},
            "batch": {"logical_batch_size": 64},
            "steps as text": {"total_logical_steps": "57"},
        }
        for name, change in wrong.items():
            with self.subTest(name):
                r, order_id = self.attempt(**change)
                self.assertEqual((r.status_code, r.json()["error"]), (422, "WORKLOAD_MISMATCH"))
                self.assertIsNone(self.order_row(order_id))
        self.assertEqual(self.coordinator.calls, [])

    def test_less_work_than_quoted_is_allowed(self):
        r, _ = self.attempt(total_logical_steps=30, dataset={**WORKLOAD["dataset"], "train_rows": 200})
        self.assertEqual(r.status_code, 201)

    def test_a_missing_field_is_a_mismatch_not_a_crash(self):
        org = self.org()
        order_id, _ = self.order(org)
        r = self.client("asha@example.com").post(f"/v1/orgs/{org}/jobs", json={"order_id": order_id, "workload": {}, "policy": {}})
        self.assertEqual((r.status_code, r.json()["error"]), (422, "WORKLOAD_MISMATCH"))


class StaleClaims(Base):
    def test_a_claim_left_by_a_crashed_request_is_released_after_ten_minutes_and_not_before(self):
        from ekido_platform import v1_jobs
        import time
        org = self.org()
        order_id, _ = self.order(org)
        for age, expected in ((60, 409), (601, 201)):
            with Session(self.engine) as s:
                s.get(PaymentOrder, order_id).job_id = v1_jobs.claim_marker(time.time() - age)
                s.commit()
            self.assertEqual(self.start("asha@example.com", org, order_id).status_code, expected, age)

    def test_markers_fit_the_column_and_unreadable_claims_count_as_stale(self):
        from ekido_platform import v1_jobs
        self.assertLessEqual(len(v1_jobs.claim_marker()), 36)
        self.assertTrue(v1_jobs.is_stale_claim("claim-garbage"))
        self.assertFalse(v1_jobs.is_stale_claim("a-real-job-id"))
        self.assertFalse(v1_jobs.is_stale_claim(None))
        self.assertFalse(v1_jobs.is_stale_claim(v1_jobs.claim_marker()))


class Failures(Base):
    def test_a_refused_create_releases_the_order_so_the_customer_can_retry(self):
        org = self.org()
        order_id, _ = self.order(org)
        self.coordinator.create_status = 422
        r = self.start("asha@example.com", org, order_id)
        self.assertEqual(r.status_code, 422)
        self.assertIsNone(self.order_row(order_id))
        self.coordinator.create_status = 200
        self.assertEqual(self.start("asha@example.com", org, order_id).status_code, 201)

    def test_a_refused_start_cancels_the_job_releases_the_order_and_records_no_owner(self):
        org = self.org()
        order_id, _ = self.order(org)
        self.coordinator.start_status = 409
        self.assertEqual(self.start("asha@example.com", org, order_id).status_code, 422)
        self.assertIn(("POST", "/jobs/job-1/cancel", None), self.coordinator.calls)
        self.assertIsNone(self.order_row(order_id))
        with Session(self.engine) as s:
            self.assertEqual(s.scalars(select(JobOwner)).all(), [])

    def test_an_unreachable_coordinator_is_503_and_the_order_is_released(self):
        org = self.org()
        order_id, _ = self.order(org)
        self.coordinator.down = True
        self.assertEqual(self.start("asha@example.com", org, order_id).status_code, 503)
        self.assertIsNone(self.order_row(order_id))


if __name__ == "__main__":
    unittest.main()
