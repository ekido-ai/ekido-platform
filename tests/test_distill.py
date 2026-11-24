"""Ticket #154 (Platform half): generated examples, reviewed by the customer, then training. A fake Coordinator and a fake Razorpay."""

import json
import os
import re
import tempfile
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_platform import v1_distill
from ekido_platform.accounts import JobOwner
from ekido_platform.main import Settings, create_app
from ekido_platform.payments import PaymentOrder
from ekido_platform.v1_datasets import Dataset, Quote, split_rows
from ekido_platform.v1_distill import Distillation

TOKEN = "internal-token"
H = {"X-Internal-Token": TOKEN}
LABELS = ("complaint", "order", "question", "refund", "other")


class FakeRazorpay:
    def __init__(self):
        self.payments, self.refunds, self.refund_fails = {}, [], False

    def fetch_payment(self, payment_id):
        return dict(self.payments[payment_id])

    def refund_payment(self, payment_id, amount_paise, *, receipt):
        if self.refund_fails:
            from ekido_platform.payments import PaymentError

            raise PaymentError("GATEWAY_UNREACHABLE", "Razorpay could not be reached", 502)
        self.refunds.append((payment_id, amount_paise))
        self.payments[payment_id]["amount_refunded"] = amount_paise
        return {"id": f"rfnd_{len(self.refunds)}", "amount": amount_paise}


class FakeCoordinator:
    def __init__(self):
        self.calls, self.down, self.refuse = [], False, set()
        self.n = 0

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if self.down:
            raise OSError("down")
        if path in self.refuse:
            return 422, {"detail": "no device can run this"}
        self.n += 1
        if path in ("/distillations", "/jobs"):
            return 201, {"job_id": f"{'gen' if path == '/distillations' else 'train'}-{self.n}"}
        return 200, {"state": "ASSIGNED"}

    def posted(self, path):
        return [c for c in self.calls if c[1] == path]


class FakeJobs:
    """Answers GET /jobs/<id> for the deletion check."""

    def __call__(self, method, path):
        return 404, None


def seeds(n=30):
    return [{"text": f"seed ticket number {i} parcel nahi aaya", "label": LABELS[i % 5]} for i in range(n)]


def held_out(n=20):
    return [{"text": f"held back test ticket {i} paisa wapas", "label": LABELS[i % 5]} for i in range(n)]


def generated(n=970, offset=0):
    return [{"text": f"generated ticket {offset + i} delivery late hai", "label": LABELS[i % 5]} for i in range(n)]


def jsonl(rows):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.app = create_app(Settings(database_url=f"sqlite:///{os.path.join(tmp, 'p.db')}", _env_file=None, email_sender="log", site_url="https://ekido.example",
                                       cors_origins=["https://ekido.example"], session_cookie_secure=False, internal_token=TOKEN,
                                       razorpay_key_id="rzp_test_abc", razorpay_key_secret="s", razorpay_webhook_secret="w"))
        self.engine = self.app.state.engine
        self.razorpay = FakeRazorpay()
        self.app.state.payment_rail.gateway = self.razorpay
        self.coordinator = FakeCoordinator()
        self.app.state.coordinator_post = self.coordinator
        self.app.state.coordinator_get = FakeJobs()
        self.clients = {}
        self.org = self.org_for("asha@example.com")

    def client(self, email):
        if email not in self.clients:
            c = TestClient(self.app)
            with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
                c.post("/v1/auth/email/start", json={"email": email})
            token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
            self.assertEqual(c.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)
            self.clients[email] = c
        return self.clients[email]

    def org_for(self, email, name="Asha Traders"):
        return self.client(email).post("/v1/orgs", json={"name": name}).json()["org_id"]

    def paid_order(self, org=None, distill=True, email="asha@example.com", n_seeds=30, variant=""):
        """A dataset of seeds and tests, a quote (with or without generated examples) and a paid (HELD) order for it."""
        org = org or self.org
        c = self.client(email)
        dataset = c.post(f"/v1/orgs/{org}/datasets", json={"recipe_id": "ticket-sorter-hinglish", "train": [{**s, "text": s["text"] + variant} for s in seeds(n_seeds)], "test": held_out()})
        self.assertEqual(dataset.status_code, 201, dataset.text)
        quote = c.post(f"/v1/orgs/{org}/quotes", json={"dataset_id": dataset.json()["dataset_id"], "distill": distill})
        self.assertEqual(quote.status_code, 201, quote.text)
        with Session(self.engine) as s:
            q = s.get(Quote, quote.json()["quote_id"])
            n = s.scalar(select(PaymentOrder.id).limit(1))
            pid = f"pay_{q.id[:8]}"
            order = PaymentOrder(org_id=org, quote_id=q.id, terms_hash=q.terms_hash, amount_paise=q.quote["total_paise"], currency="INR", gateway_order_id=f"order_{q.id[:8]}",
                                 gateway_payment_id=pid, status="HELD", quote=q.quote)
            s.add(order)
            s.commit()
            self.razorpay.payments[pid] = {"id": pid, "order_id": order.gateway_order_id, "amount": order.amount_paise, "currency": "INR", "amount_refunded": 0}
            return order.id, q.quote

    def start(self, order_id, org=None, email="asha@example.com"):
        return self.client(email).post(f"/v1/orgs/{org or self.org}/distillations", json={"order_id": order_id})

    def report(self, did, rows=None, passed=True, reasons=()):
        rows = generated() if rows is None else rows
        return self.client("asha@example.com").post(f"/internal/distillations/{did}/generated", headers=H,
                                                     json={"rows_jsonl": jsonl(rows), "passed": passed, "reasons": list(reasons)})

    def row(self, did):
        with Session(self.engine) as s:
            r = s.get(Distillation, did)
            s.expunge(r)
            return r

    def order_status(self, order_id):
        with Session(self.engine) as s:
            return s.get(PaymentOrder, order_id).status

    def in_review(self, **kw):
        order_id, quote = self.paid_order(**kw)
        did = self.start(order_id).json()["distillation_id"]
        self.assertEqual(self.report(did).status_code, 200)
        self.assertEqual(self.row(did).state, "REVIEW")
        return order_id, did, quote

    def url(self, did, tail=""):
        return f"/v1/orgs/{self.org}/distillations/{did}{tail}"


class Starting(Base):
    def test_a_paid_distillation_order_starts_the_generation_job_with_the_spec_the_plan_names(self):
        order_id, quote = self.paid_order()
        r = self.start(order_id)
        self.assertEqual((r.status_code, r.json()["state"]), (201, "GENERATING"))
        (method, path, body), = self.coordinator.posted("/distillations")
        spec = body["spec"]
        self.assertEqual((spec["template"], spec["recipe_id"], spec["seed_rows"], spec["target_rows"], spec["attempt"]), ("distill-gen/v1", "ticket-sorter-hinglish", 30, quote["plan"]["generated_rows"], 0))
        self.assertEqual(len(spec["test_inputs_sha256"]), 20)
        self.assertNotIn("held back test ticket", json.dumps(body))  # only hashes of the test inputs leave, never the text
        self.assertEqual(body["distillation_id"], r.json()["distillation_id"])

    def test_one_quote_covers_both_steps(self):
        _, quote = self.paid_order()
        self.assertTrue(quote["distill"])
        self.assertGreater(quote["distillation_paise"], 0)
        self.assertGreater(quote["training_paise"], 0)
        self.assertEqual(quote["plan"]["generated_rows"], 970)

    def test_asking_for_the_same_quote_twice_in_one_second_returns_the_same_quote_and_never_another_teams(self):
        from unittest.mock import patch

        c = self.client("asha@example.com")
        dataset = c.post(f"/v1/orgs/{self.org}/datasets", json={"recipe_id": "ticket-sorter-hinglish", "train": seeds(), "test": held_out()}).json()["dataset_id"]
        first = c.post(f"/v1/orgs/{self.org}/quotes", json={"dataset_id": dataset, "distill": True})
        self.assertEqual(first.status_code, 201)
        fixed = {k: v for k, v in first.json().items() if k not in ("quote_id", "dataset_id")}
        with patch("ekido_platform.v1_datasets.make_quote", return_value=fixed):  # the same second: the same terms
            again = c.post(f"/v1/orgs/{self.org}/quotes", json={"dataset_id": dataset, "distill": True})
            self.assertEqual((again.status_code, again.json()["quote_id"]), (201, first.json()["quote_id"]))
            other = self.org_for("ravi@example.com", "Ravi")
            theirs = self.client("ravi@example.com").post(f"/v1/orgs/{other}/datasets", json={"recipe_id": "ticket-sorter-hinglish", "train": seeds(), "test": held_out()}).json()["dataset_id"]
            refused = self.client("ravi@example.com").post(f"/v1/orgs/{other}/quotes", json={"dataset_id": theirs, "distill": True})
            self.assertEqual((refused.status_code, refused.json()["error"]), (409, "TRY_AGAIN"))

    def test_a_second_start_for_the_same_order_is_refused(self):
        order_id, _ = self.paid_order()
        self.assertEqual(self.start(order_id).status_code, 201)
        again = self.start(order_id)
        self.assertEqual((again.status_code, again.json()["error"]), (409, "ALREADY_STARTED"))
        self.assertEqual(len(self.coordinator.posted("/distillations")), 1)

    def test_only_a_paid_order_whose_quote_asked_for_it_and_that_is_the_teams_own_can_start(self):
        plain, _ = self.paid_order(distill=False, variant=" a")
        self.assertEqual(self.start(plain).json()["error"], "NOT_A_DISTILLATION")
        order_id, _ = self.paid_order(variant=" b")
        with Session(self.engine) as s:
            s.get(PaymentOrder, order_id).status = "CREATED"
            s.commit()
        self.assertEqual((self.start(order_id).status_code, self.start(order_id).json()["error"]), (409, "NOT_PAID"))
        other = self.org_for("ravi@example.com", "Ravi")
        paid, _ = self.paid_order(variant=" c")
        self.assertEqual(self.start(paid, org=other, email="ravi@example.com").status_code, 404)  # his team, her order
        self.assertEqual(self.coordinator.calls, [])

    def test_a_coordinator_that_refuses_or_is_down_leaves_nothing_behind_and_the_order_can_try_again(self):
        order_id, _ = self.paid_order()
        for arrange, status in ((lambda: self.coordinator.refuse.add("/distillations"), 422), (lambda: setattr(self.coordinator, "down", True), 503)):
            arrange()
            self.assertEqual(self.start(order_id).status_code, status)
            with Session(self.engine) as s:
                self.assertEqual(s.scalars(select(Distillation)).all(), [])
        self.coordinator.refuse.clear()
        self.coordinator.down = False
        self.assertEqual(self.start(order_id).status_code, 201)

    def test_only_owners_and_admins_start(self):
        order_id, _ = self.paid_order()
        self.assertEqual(TestClient(self.app).post(f"/v1/orgs/{self.org}/distillations", json={"order_id": order_id}).status_code, 401)


class WhenGenerationEnds(Base):
    def test_rows_that_pass_go_to_review_with_twenty_random_generated_rows_and_never_a_test_row(self):
        order_id, did, _ = self.in_review()
        row = self.row(did)
        self.assertEqual((row.generated_rows, len(row.sample), len(set(row.sample))), (970, 20, 20))
        self.assertTrue(all(0 <= i < 970 for i in row.sample))
        shown = self.client("asha@example.com").get(self.url(did)).json()
        self.assertEqual(len(shown["sample"]), 20)
        self.assertTrue(all(x["text"].startswith("generated ticket") for x in shown["sample"]))  # never a seed or a test row
        self.assertFalse(any("held back test ticket" in x["text"] for x in shown["sample"]))
        self.assertEqual(self.order_status(order_id), "HELD")  # nothing refunded

    def test_two_samples_differ(self):
        samples = set()
        for _ in range(4):
            self.setUp()
            _, did, _ = self.in_review()
            samples.add(tuple(self.row(did).sample))
        self.assertGreater(len(samples), 1)

    def test_a_failed_verdict_refunds_the_order_in_full_once_even_when_reported_twice(self):
        order_id, quote = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        for _ in range(2):
            self.assertEqual(self.report(did, passed=False, reasons=["TEST_OVERLAP", "BAD_LABEL"]).status_code, 200)
        self.assertEqual(self.row(did).state, "FAILED")
        self.assertIn("TEST_OVERLAP", self.row(did).reason)
        with Session(self.engine) as s:
            paid_with = s.get(PaymentOrder, order_id).gateway_payment_id
        self.assertEqual((self.order_status(order_id), self.razorpay.refunds), ("REFUNDED", [(paid_with, quote["total_paise"])]))  # the full amount, once

    def test_the_platform_checks_the_rows_itself_and_does_not_rely_on_the_verdict(self):
        bad = {
            "a generated row equal to a test example": generated(5) + [{"text": "HELD BACK   test ticket 3 paisa wapas", "label": "refund"}],
            "wrong columns": [{"text": "only a text column"}],
            "an empty label": [{"text": "something new here", "label": " "}],
            "no rows": [],
        }
        for label, rows in bad.items():
            with self.subTest(label):
                self.setUp()
                order_id, _ = self.paid_order()
                did = self.start(order_id).json()["distillation_id"]
                self.assertEqual(self.report(did, rows=rows, passed=True).status_code, 200)  # the verdict said pass
                self.assertEqual(self.row(did).state, "FAILED")
                self.assertEqual(self.order_status(order_id), "REFUNDED")

    def test_a_file_that_is_not_json_lines_fails_and_refunds(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        r = self.client("asha@example.com").post(f"/internal/distillations/{did}/generated", headers=H, json={"rows_jsonl": "{oops", "passed": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("FAILED", "REFUNDED"))

    def test_a_job_that_failed_refunds_too(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        r = self.client("asha@example.com").post(f"/internal/distillations/{did}/failed", headers=H, json={"reason": "no device in time"})
        self.assertEqual((r.status_code, r.json()["state"]), (200, "FAILED"))
        self.assertEqual(self.order_status(order_id), "REFUNDED")

    def test_a_refund_that_failed_is_retried_by_the_next_report_and_never_doubled(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        self.razorpay.refund_fails = True
        self.assertGreaterEqual(self.report(did, passed=False).status_code, 500)
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("FAILED", "HELD"))  # recorded as failed, money not yet back
        self.razorpay.refund_fails = False
        self.assertEqual(self.report(did, passed=False).status_code, 200)
        self.assertEqual(self.order_status(order_id), "REFUNDED")
        self.assertEqual(len(self.razorpay.refunds), 1)

    def test_a_late_report_after_the_state_moved_on_changes_nothing(self):
        _, did, _ = self.in_review()
        before = self.row(did).sample
        self.assertEqual(self.report(did, rows=generated(50, offset=5000)).status_code, 200)
        self.assertEqual((self.row(did).state, self.row(did).generated_rows, self.row(did).sample), ("REVIEW", 970, before))

    def test_the_internal_routes_are_closed_without_the_token(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        c = self.client("asha@example.com")
        self.assertEqual(c.post(f"/internal/distillations/{did}/generated", json={"rows_jsonl": "", "passed": True}).status_code, 404)
        self.assertEqual(c.post(f"/internal/distillations/{did}/failed", json={}, headers={"X-Internal-Token": "wrong"}).status_code, 404)


class Sweeping(Base):
    def age(self, did, minutes):
        from datetime import datetime, timedelta, timezone

        with Session(self.engine) as s:
            s.get(Distillation, did).generating_since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            s.commit()

    def sweep(self, body=None, headers=H):
        return self.client("asha@example.com").post("/internal/distillations/sweep", headers=headers, **({"json": body} if body is not None else {}))

    def test_a_generation_that_never_reported_is_failed_and_refunded_after_six_hours_and_not_before(self):
        order_id, quote = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        self.age(did, 300)  # 5 hours: still within the default
        self.assertEqual(self.sweep().json(), {"failed": 0, "refund_errors": 0})
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("GENERATING", "HELD"))
        self.age(did, 361)
        self.assertEqual(self.sweep().json(), {"failed": 1, "refund_errors": 0})
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("FAILED", "REFUNDED"))
        self.assertIn("too long", self.row(did).reason)
        self.assertEqual([amount for _, amount in self.razorpay.refunds], [quote["total_paise"]])
        self.assertEqual(self.sweep().json(), {"failed": 0, "refund_errors": 0})  # running it again does nothing and refunds nothing more
        self.assertEqual(len(self.razorpay.refunds), 1)

    def test_only_generating_ones_are_touched_and_the_window_can_be_chosen(self):
        _, review, _ = self.in_review()  # waiting for the customer, not for the Coordinator
        order_id, _ = self.paid_order(variant=" other")
        slow = self.start(order_id).json()["distillation_id"]
        self.age(slow, 120)
        self.assertEqual(self.sweep({"older_than_minutes": 60}).json(), {"failed": 1, "refund_errors": 0})
        self.assertEqual((self.row(review).state, self.row(slow).state), ("REVIEW", "FAILED"))

    def test_a_regeneration_restarts_the_clock(self):
        _, did, _ = self.in_review()
        self.client("asha@example.com").post(self.url(did, "/regenerate"))
        self.assertEqual(self.sweep().json(), {"failed": 0, "refund_errors": 0})  # just restarted, however old the first generation was

    def test_a_refund_that_failed_is_retried_by_the_next_sweep(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        self.age(did, 400)
        self.razorpay.refund_fails = True
        first = self.sweep()
        self.assertEqual((first.status_code, first.json()), (200, {"failed": 1, "refund_errors": 1}))  # the timer is told, and carries on
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("FAILED", "HELD"))
        self.razorpay.refund_fails = False
        self.assertEqual(self.sweep().json(), {"failed": 0, "refund_errors": 0})
        self.assertEqual((self.order_status(order_id), len(self.razorpay.refunds)), ("REFUNDED", 1))

    def test_one_refund_that_fails_does_not_stop_the_others(self):
        ids = []
        for variant in (" a", " b"):
            order_id, _ = self.paid_order(variant=variant)
            did = self.start(order_id).json()["distillation_id"]
            self.age(did, 400)
            ids.append((order_id, did))
        self.razorpay.refund_fails = True
        self.assertEqual(self.sweep().json(), {"failed": 2, "refund_errors": 2})
        self.razorpay.refund_fails = False
        self.assertEqual(self.sweep().json(), {"failed": 0, "refund_errors": 0})
        self.assertEqual([self.order_status(o) for o, _ in ids], ["REFUNDED", "REFUNDED"])

    def test_the_sweep_is_closed_without_the_token_and_refuses_absurd_windows(self):
        self.assertEqual(self.sweep(headers={}).status_code, 404)
        self.assertEqual(self.sweep({"older_than_minutes": 1}).status_code, 422)
        self.assertEqual(self.sweep({"older_than_minutes": 10**7}).status_code, 422)
        self.assertEqual(self.sweep({"surprise": 1}).status_code, 422)


class Review(Base):
    def test_only_rows_in_the_sample_can_be_marked_wrong_and_only_while_in_review(self):
        _, did, _ = self.in_review()
        c = self.client("asha@example.com")
        outside = next(i for i in range(970) if i not in self.row(did).sample)
        self.assertEqual(c.post(self.url(did, "/remove"), json={"indices": [outside]}).status_code, 422)
        inside = self.row(did).sample[:2]
        r = c.post(self.url(did, "/remove"), json={"indices": inside})
        self.assertEqual((r.status_code, r.json()["removed"]), (200, 2))
        self.assertEqual([x["index"] for x in r.json()["sample"] if x["removed"]], sorted(inside))
        self.assertEqual(c.post(self.url(did, "/remove"), json={"indices": []}).status_code, 422)

    def test_a_quarter_of_the_sample_removed_is_fine_more_than_that_suggests_regenerating(self):
        _, did, _ = self.in_review()
        c = self.client("asha@example.com")
        sample = self.row(did).sample
        five = c.post(self.url(did, "/remove"), json={"indices": sample[:5]}).json()
        self.assertEqual((five["removed"], five["suggest_regenerate"]), (5, False))  # exactly a quarter
        six = c.post(self.url(did, "/remove"), json={"indices": sample[:6]}).json()
        self.assertEqual((six["removed"], six["suggest_regenerate"]), (6, True))

    def test_regenerating_is_allowed_once_and_makes_a_new_generation_with_a_new_seed(self):
        _, did, _ = self.in_review()
        c = self.client("asha@example.com")
        r = c.post(self.url(did, "/regenerate"))
        self.assertEqual((r.status_code, r.json()["state"], r.json()["regenerations_left"]), (200, "GENERATING", 0))
        first, second = [call[2]["spec"] for call in self.coordinator.posted("/distillations")]
        self.assertEqual((first["attempt"], second["attempt"]), (0, 1))
        self.assertNotEqual(first["generation_seed"], second["generation_seed"])
        self.assertIsNone(self.row(did).generated)  # the old rows are gone
        self.assertEqual(self.report(did, rows=generated(970, offset=9000)).status_code, 200)
        again = c.post(self.url(did, "/regenerate"))
        self.assertEqual((again.status_code, again.json()["error"]), (409, "NO_REGENERATIONS_LEFT"))

    def test_a_regeneration_the_coordinator_refuses_changes_nothing(self):
        _, did, _ = self.in_review()
        self.coordinator.refuse.add("/distillations")
        r = self.client("asha@example.com").post(self.url(did, "/regenerate"))
        self.assertEqual(r.status_code, 422)
        row = self.row(did)
        self.assertEqual((row.state, row.regenerations, row.generated_rows), ("REVIEW", 0, 970))

    def test_nothing_can_be_reviewed_before_the_rows_arrive(self):
        order_id, _ = self.paid_order()
        did = self.start(order_id).json()["distillation_id"]
        c = self.client("asha@example.com")
        for tail in ("/remove", "/regenerate", "/approve", "/reject"):
            body = {"indices": [1]} if tail == "/remove" else None
            r = c.post(self.url(did, tail), json=body) if body else c.post(self.url(did, tail))
            self.assertEqual((tail, r.status_code), (tail, 409))

    def test_another_team_sees_nothing(self):
        _, did, _ = self.in_review()
        other = self.org_for("ravi@example.com", "Ravi")
        c = self.client("ravi@example.com")
        for call in (c.get(f"/v1/orgs/{other}/distillations/{did}"), c.post(f"/v1/orgs/{other}/distillations/{did}/approve"), c.post(f"/v1/orgs/{other}/distillations/{did}/reject")):
            self.assertEqual(call.status_code, 404)
        self.assertEqual(c.get(f"/v1/orgs/{self.org}/distillations/{did}").status_code, 404)  # not a member of her team: it looks missing, not forbidden


class Approving(Base):
    def test_approving_builds_the_training_set_from_the_seeds_and_the_kept_rows_and_starts_training_on_the_paid_order(self):
        order_id, did, quote = self.in_review()
        c = self.client("asha@example.com")
        removed = self.row(did).sample[:3]
        c.post(self.url(did, "/remove"), json={"indices": removed})
        r = c.post(self.url(did, "/approve"))
        self.assertEqual(r.status_code, 201, r.text)
        job_id = r.json()["job_id"]
        self.assertEqual((r.json()["state"], job_id), ("TRAINING", self.row(did).training_job_id))
        with Session(self.engine) as s:
            combined = s.get(Dataset, self.row(did).training_dataset_id)
            train, test = split_rows(combined.content, "train"), split_rows(combined.content, "test")
            self.assertEqual((len(train), len(test)), (30 + 970 - 3, 20))
            texts = {x["text"] for x in train}
            gen = self.row(did)
            all_rows = [json.loads(line) for line in gen.generated.decode().splitlines()]
            self.assertTrue(all(all_rows[i]["text"] not in texts for i in removed))  # the removed rows are out
            self.assertTrue(all(x["text"] in texts for x in seeds()))  # the seeds are in
            self.assertEqual([x["text"] for x in test], [x["text"] for x in held_out()])  # the test rows are exactly the customer's own
            self.assertEqual(s.get(PaymentOrder, order_id).job_id, job_id)
            self.assertEqual(s.scalar(select(JobOwner.org_id).where(JobOwner.job_id == job_id)), self.org)
        (_, _, created), = self.coordinator.posted("/jobs")
        self.assertEqual(created["workload"]["template"], "llm-lora/v1")
        self.assertEqual(created["workload"]["dataset"]["train_rows"], 30 + 970 - 3)
        self.assertEqual(created["workload"]["dataset"]["test_rows"], 20)
        self.assertEqual(created["workload"]["dataset"]["sha256"], combined.sha256)
        self.assertEqual(created["budget_usdc"], f"{quote['provider_budget_paise'] / 100:.6f}")
        self.assertEqual(self.coordinator.posted(f"/jobs/{job_id}/start")[0][0], "POST")

    def test_approving_twice_starts_one_job(self):
        _, did, _ = self.in_review()
        c = self.client("asha@example.com")
        first = c.post(self.url(did, "/approve")).json()["job_id"]
        second = c.post(self.url(did, "/approve"))
        self.assertEqual(second.status_code, 409)  # TRAINING is not a state that can be approved again
        self.assertEqual(len(self.coordinator.posted("/jobs")), 1)
        self.assertEqual(self.row(did).training_job_id, first)

    def test_a_coordinator_that_is_down_leaves_the_set_approved_and_a_retry_starts_training(self):
        order_id, did, _ = self.in_review()
        c = self.client("asha@example.com")
        self.coordinator.down = True
        self.assertEqual(c.post(self.url(did, "/approve")).status_code, 503)
        row = self.row(did)
        self.assertEqual((row.state, row.training_job_id), ("APPROVED", None))
        with Session(self.engine) as s:
            self.assertIsNone(s.get(PaymentOrder, order_id).job_id)  # the claim was released
            self.assertEqual(len(s.scalars(select(Dataset).where(Dataset.id == row.training_dataset_id)).all()), 1)
        self.coordinator.down = False
        retry = c.post(self.url(did, "/approve"))
        self.assertEqual((retry.status_code, retry.json()["state"]), (201, "TRAINING"))
        self.assertEqual(len([d for d in self.coordinator.calls if d[1] == "/jobs"]), 2)  # one refused by the outage, one real, no duplicate data set
        with Session(self.engine) as s:
            self.assertEqual(len(s.scalars(select(Dataset).where(Dataset.id == self.row(did).training_dataset_id)).all()), 1)

    def test_an_order_that_already_has_a_job_cannot_also_train_here(self):
        order_id, did, _ = self.in_review()
        with Session(self.engine) as s:
            s.get(PaymentOrder, order_id).job_id = "some-other-job"  # a normal job was started from the same order meanwhile
            s.commit()
        r = self.client("asha@example.com").post(self.url(did, "/approve"))
        self.assertEqual((r.status_code, r.json()["error"]), (409, "ALREADY_STARTED"))
        self.assertEqual(self.coordinator.posted("/jobs"), [])
        self.assertNotEqual(self.row(did).state, "TRAINING")

    def test_a_job_the_coordinator_refuses_to_start_is_cancelled_and_the_order_is_released(self):
        order_id, did, _ = self.in_review()
        self.coordinator.refuse.add(f"/jobs/train-{self.coordinator.n + 1}/start")
        r = self.client("asha@example.com").post(self.url(did, "/approve"))
        self.assertGreaterEqual(r.status_code, 400)
        self.assertTrue(any(c[1].endswith("/cancel") for c in self.coordinator.calls))
        with Session(self.engine) as s:
            self.assertIsNone(s.get(PaymentOrder, order_id).job_id)
            self.assertEqual(s.scalars(select(JobOwner)).all(), [])


class Rejecting(Base):
    def test_rejecting_refunds_the_whole_order_once_and_no_training_starts(self):
        order_id, did, quote = self.in_review()
        c = self.client("asha@example.com")
        for _ in range(2):
            r = c.post(self.url(did, "/reject"))
            self.assertEqual((r.status_code, r.json()["state"]), (200, "REJECTED"))
        self.assertEqual(self.order_status(order_id), "REFUNDED")
        self.assertEqual([amount for _, amount in self.razorpay.refunds], [quote["total_paise"]])
        self.assertEqual(self.coordinator.posted("/jobs"), [])

    def test_a_refund_that_failed_is_retried_when_the_customer_asks_again(self):
        order_id, did, _ = self.in_review()
        c = self.client("asha@example.com")
        self.razorpay.refund_fails = True
        self.assertGreaterEqual(c.post(self.url(did, "/reject")).status_code, 500)
        self.assertEqual((self.row(did).state, self.order_status(order_id)), ("REJECTED", "HELD"))
        self.razorpay.refund_fails = False
        self.assertEqual(c.post(self.url(did, "/reject")).status_code, 200)
        self.assertEqual((self.order_status(order_id), len(self.razorpay.refunds)), ("REFUNDED", 1))

    def test_once_training_has_started_it_cannot_be_rejected(self):
        order_id, did, _ = self.in_review()
        c = self.client("asha@example.com")
        c.post(self.url(did, "/approve"))
        r = c.post(self.url(did, "/reject"))
        self.assertEqual((r.status_code, r.json()["error"]), (409, "CANNOT_REJECT"))
        self.assertEqual((self.order_status(order_id), self.razorpay.refunds), ("HELD", []))


class DeleteMyData(Base):
    def test_deleting_the_account_erases_the_datasets_quotes_and_generated_rows_of_the_team(self):
        order_id, did, _ = self.in_review()
        self.assertEqual(self.client("asha@example.com").delete("/v1/me").status_code, 202)
        with Session(self.engine) as s:
            self.assertEqual((len(s.scalars(select(Distillation)).all()), len(s.scalars(select(Dataset)).all()), len(s.scalars(select(Quote)).all())), (0, 0, 0))
            self.assertEqual(len(s.scalars(select(PaymentOrder)).all()), 1)  # payment records stay, as the law requires (they hold no names)


if __name__ == "__main__":
    unittest.main()
