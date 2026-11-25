"""Ticket C9: recipes → dataset → quote → order → job, with the job built on the server from what was quoted and paid for."""
import hashlib
import json
import re
import unittest

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_core.job_spec import spec_problems
from ekido_core.quotes import training_plan
from ekido_core.recipes import eval_set_hash
from ekido_platform.main import Settings, create_app
from ekido_platform.payments import HELD, PaymentOrder, RazorpayRail
from tests.test_payments import SECRET, FakeGateway, event, sign

SITE = "https://ekido.example"
LABELS = ("complaint", "order", "question", "refund", "other")


def rows(n, offset=0):
    return [{"text": f"Ticket number {offset + i}: mera order {offset + i} abhi tak nahi aaya", "label": LABELS[(offset + i) % 5]} for i in range(n)]


class Flow(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE],
                                       session_cookie_secure=False, razorpay_key_id="rzp_test_public"))
        self.gateway = FakeGateway()
        self.app.state.payment_rail = RazorpayRail(self.gateway, SECRET)
        self.calls = []

        def coordinator(method, path, body):
            self.calls.append((method, path, body))
            if method == "PUT" and path.startswith("/datasets/"):
                return 201, {"sha256": path.rsplit("/", 1)[1], "bytes": len(body)}
            if method == "POST" and path == "/jobs":
                return 201, {"job_id": "00000000-0000-4000-8000-0000000000aa", "state": "OPEN"}
            if method == "POST" and path.endswith("/activate"):
                return 200, {"state": "ASSIGNED"}
            return 404, {"detail": "not found"}

        self.app.state.coordinator_post = coordinator
        self.client = TestClient(self.app)
        self.org = self.team("asha@example.com", "Sharma Electronics")

    def sign_in(self, email):
        self.client.post("/v1/auth/logout")
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.client.post("/v1/auth/email/start", json={"email": email})
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.assertEqual(self.client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)

    def team(self, email, name):
        self.sign_in(email)
        return self.client.post("/v1/orgs", json={"name": name}, headers={"Origin": SITE}).json()["org_id"]

    def post(self, path, body=None):
        return self.client.post(path, json=body, headers={"Origin": SITE})

    def dataset(self, train=None, test=None, org=None, **extra):
        body = {"recipe_id": "ticket-sorter-hinglish", "train": rows(40) if train is None else train, "test": rows(20, 1000) if test is None else test, **extra}
        return self.post(f"/v1/orgs/{org or self.org}/datasets", body)

    def paid_quote(self):
        dataset = self.dataset(redactions={"phone": 3}).json()
        quote = self.post(f"/v1/orgs/{self.org}/quotes", {"dataset_id": dataset["dataset_id"]}).json()
        order = self.post(f"/v1/quotes/{quote['quote_id']}/orders").json()
        pid = self.gateway.pay(order["gateway_order_id"])
        raw = event("payment.captured", self.gateway.payments[pid])
        with Session(self.app.state.engine) as session:
            self.app.state.payment_rail.handle_webhook(session, raw, sign(raw), f"evt_{pid}")
        return dataset, quote, order

    def test_recipes_are_public_with_their_columns_and_floor(self):
        recipes = {r["recipe_id"]: r for r in self.client.get("/v1/recipes").json()}
        self.assertEqual(len(recipes), 5)
        sorter = recipes["ticket-sorter-hinglish"]
        self.assertEqual((sorter["columns"], sorter["floor"], sorter["labels"][0]), ({"input": "text", "answer": "label"}, 0.8, "complaint"))
        self.assertIn("80%", sorter["quality_terms"])
        self.assertEqual(self.client.get("/v1/recipes/nope").status_code, 404)

    def test_a_dataset_is_stored_as_canonical_bytes_with_counts_only(self):
        response = self.dataset(redactions={"phone": 3, "aadhaar": 1})
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertEqual((body["train_rows"], body["test_rows"], body["redactions"]), (40, 20, {"phone": 3, "aadhaar": 1}))
        again = self.dataset(redactions={"phone": 3, "aadhaar": 1}).json()
        self.assertEqual(again["sha256"], body["sha256"])  # the same rows always give the same bytes

    def test_rows_that_dont_fit_are_refused_with_every_problem(self):
        bad = rows(5) + [{"text": "  ", "label": "refund"}, {"text": "x", "label": "spam"}, {"text": "y"}]
        response = self.dataset(train=bad, test=rows(10, 1000) + [rows(1)[0]])
        self.assertEqual((response.status_code, response.json()["error"]), (422, "ROWS_INVALID"))
        detail = response.json()["detail"]
        for expected in ("text is empty", "label must be one of", "needs exactly the columns", "test needs at least 20", "also appear in train"):
            self.assertIn(expected, detail)

    def test_another_team_cant_see_or_quote_my_data(self):
        dataset = self.dataset().json()
        self.team("ravi@example.com", "Ravi Traders")
        self.assertEqual(self.post(f"/v1/orgs/{self.org}/datasets", {"recipe_id": "ticket-sorter-hinglish", "train": rows(40), "test": rows(20, 1000)}).status_code, 404)
        other = self.client.get("/v1/me").json()["orgs"][0]["org_id"]
        self.assertEqual(self.post(f"/v1/orgs/{other}/quotes", {"dataset_id": dataset["dataset_id"]}).status_code, 404)

    def test_a_quote_prices_the_dataset_and_an_order_is_made_for_it(self):
        dataset = self.dataset().json()
        quote = self.post(f"/v1/orgs/{self.org}/quotes", {"dataset_id": dataset["dataset_id"]})
        self.assertEqual(quote.status_code, 201, quote.text)
        q = quote.json()
        self.assertEqual((q["dataset_sha256"], q["plan"]["train_rows"], q["currency"]), (dataset["sha256"], 40, "INR"))
        self.assertEqual(q["total_paise"], q["training_paise"] + q["gst_paise"])
        order = self.post(f"/v1/quotes/{q['quote_id']}/orders")
        self.assertEqual(order.status_code, 201, order.text)
        self.assertEqual((order.json()["key_id"], order.json()["amount_paise"]), ("rzp_test_public", q["total_paise"]))
        self.assertEqual(self.client.get(f"/v1/orders/{order.json()['order_id']}").json()["status"], "CREATED")

    def test_without_payments_configured_no_order_is_made(self):
        dataset = self.dataset().json()
        quote = self.post(f"/v1/orgs/{self.org}/quotes", {"dataset_id": dataset["dataset_id"]}).json()
        self.app.state.payment_rail = None
        self.assertEqual(self.post(f"/v1/quotes/{quote['quote_id']}/orders").status_code, 503)

    def test_an_unpaid_quote_cant_start_a_job(self):
        dataset = self.dataset().json()
        quote = self.post(f"/v1/orgs/{self.org}/quotes", {"dataset_id": dataset["dataset_id"]}).json()
        response = self.post(f"/v1/orgs/{self.org}/jobs", {"quote_id": quote["quote_id"]})
        self.assertEqual(response.status_code, 402)
        self.assertEqual(self.calls, [])

    def test_a_paid_quote_starts_exactly_the_job_that_was_priced(self):
        dataset, quote, order = self.paid_quote()
        self.assertEqual(self.client.get(f"/v1/orders/{order['order_id']}").json()["status"], HELD)
        response = self.post(f"/v1/orgs/{self.org}/jobs", {"quote_id": quote["quote_id"]})
        self.assertEqual(response.status_code, 201, response.text)
        (put_method, put_path, blob), (_, jobs_path, job), _start = self.calls
        self.assertEqual((put_method, put_path, jobs_path), ("PUT", f"/datasets/{dataset['sha256']}", "/jobs"))
        self.assertEqual(hashlib.sha256(blob).hexdigest(), dataset["sha256"])  # the exact bytes that were quoted
        spec, policy = job["workload"], job["policy"]
        self.assertEqual(spec_problems(spec), [])
        plan = training_plan("ticket-sorter-hinglish", 40)
        self.assertEqual((spec["template"], spec["total_logical_steps"], spec["logical_batch_size"]), ("llm-lora/v1", plan["total_logical_steps"], plan["logical_batch_size"]))
        test_rows = [json.loads(line) for line in blob.decode().splitlines() if json.loads(line)["split"] == "test"]
        self.assertEqual(policy["eval_set_hash"], eval_set_hash("ticket-sorter-hinglish", test_rows))
        self.assertEqual(policy["min_absolute"], 0.8)
        with Session(self.app.state.engine) as session:
            self.assertEqual(session.get(PaymentOrder, order["order_id"]).job_id, "00000000-0000-4000-8000-0000000000aa")
        again = self.post(f"/v1/orgs/{self.org}/jobs", {"quote_id": quote["quote_id"]})
        self.assertEqual(again.status_code, 409)  # one paid quote, one job

    def test_quote_and_order_can_not_both_be_sent(self):
        self.assertEqual(self.post(f"/v1/orgs/{self.org}/jobs", {"quote_id": "q", "order_id": "o"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
