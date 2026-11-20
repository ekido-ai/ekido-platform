"""C8: the invoice route: a paid order's PDF, numbered once, for its own team only."""

import os
import re
import tempfile
import threading
import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ekido_core.quotes import make_quote, rupees

from ekido_platform import v1_invoices
from ekido_platform.main import Settings, create_app
from ekido_platform.models import Base
from ekido_platform.payments import PaymentOrder
from ekido_platform.v1_invoices import InvoiceRecord

SITE = "https://ekido.example"


def quote(rows=300):
    return make_quote(recipe_id="ticket-sorter-hinglish", base_model="Qwen/Qwen2.5-0.5B-Instruct", train_rows=rows, test_rows=40, dataset_sha256="a" * 64)


class Base_(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE], session_cookie_secure=False))
        self.engine = self.app.state.engine
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

    def org(self, email, name="Asha Traders"):
        return self.client(email).post("/v1/orgs", json={"name": name}).json()["org_id"]

    def order(self, org_id, status="SETTLED", rows=300):
        q = quote(rows)
        with Session(self.engine) as s:
            row = PaymentOrder(org_id=org_id, quote_id="q", terms_hash=q["terms_hash"], amount_paise=q["total_paise"], currency="INR", gateway_order_id=f"order_{rows}_{status}_{org_id[-4:]}", status=status, quote=q)
            s.add(row)
            s.commit()
            return row.id, q


class Route(Base_):
    def test_a_paid_order_gets_a_pdf_with_its_figures_and_a_test_marking(self):
        org = self.org("asha@example.com")
        order_id, q = self.order(org)
        response = self.client().get(f"/v1/orgs/{org}/invoices/{order_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertIn("invoice-EKI-", response.headers["content-disposition"])
        text = response.content.decode("latin-1")
        self.assertTrue(response.content.startswith(b"%PDF-"))
        for needle in ("TEST INVOICE: not a tax invoice", "Asha Traders", "Rs " + rupees(q["total_paise"])[1:], q["terms_hash"], "EKI/"):
            self.assertIn(needle, text)

    def test_held_orders_are_invoiceable_too(self):
        org = self.org("asha@example.com")
        order_id, _ = self.order(org, status="HELD")
        self.assertEqual(self.client().get(f"/v1/orgs/{org}/invoices/{order_id}").status_code, 200)

    def test_the_same_order_always_gets_the_same_number_and_new_orders_get_the_next_one(self):
        org = self.org("asha@example.com")
        first, _ = self.order(org, rows=100)
        second, _ = self.order(org, rows=200)
        numbers = []
        for order_id in (first, first, second, first):
            body = self.client().get(f"/v1/orgs/{org}/invoices/{order_id}").content.decode("latin-1")
            numbers.append(re.search(r"Invoice no: (EKI/\d{4}-\d{2}/\d{6})", body).group(1))
        self.assertEqual(numbers[0], numbers[1])
        self.assertEqual(numbers[0], numbers[3])
        year = numbers[0].split("/")[1]
        self.assertEqual((numbers[0], numbers[2]), (f"EKI/{year}/000001", f"EKI/{year}/000002"))

    def test_unpaid_refunded_and_unknown_orders_have_no_invoice(self):
        org = self.org("asha@example.com")
        created, _ = self.order(org, status="CREATED", rows=110)
        refunded, _ = self.order(org, status="REFUNDED", rows=120)
        for order_id, code in ((created, 409), (refunded, 409), ("no-such-order", 404)):
            self.assertEqual(self.client().get(f"/v1/orgs/{org}/invoices/{order_id}").status_code, code, order_id)
        self.assertIn("refunded", self.client().get(f"/v1/orgs/{org}/invoices/{refunded}").text)
        with Session(self.engine) as s:
            self.assertEqual(s.scalars(select(InvoiceRecord)).all(), [])  # none of them consumed a number

    def test_another_teams_order_looks_missing_and_signed_out_callers_are_refused(self):
        mine = self.org("asha@example.com")
        theirs = self.org("ravi@example.com", "Ravi Co")
        theirs_order, _ = self.order(theirs)
        self.assertEqual(self.client("asha@example.com").get(f"/v1/orgs/{mine}/invoices/{theirs_order}").status_code, 404)  # asking through my own team
        self.assertIn(self.client("asha@example.com").get(f"/v1/orgs/{theirs}/invoices/{theirs_order}").status_code, (403, 404))  # asking through theirs
        self.assertEqual(TestClient(self.app).get(f"/v1/orgs/{theirs}/invoices/{theirs_order}").status_code, 401)

    def test_a_tampered_quote_gets_no_invoice(self):
        org = self.org("asha@example.com")
        order_id, _ = self.order(org)
        with Session(self.engine) as s:
            row = s.get(PaymentOrder, order_id)
            row.quote = dict(row.quote, gst_paise=row.quote["gst_paise"] - 100)
            s.commit()
        response = self.client().get(f"/v1/orgs/{org}/invoices/{order_id}")
        self.assertEqual((response.status_code, response.json()["error"]), (422, "INVOICE_UNAVAILABLE"))

    def test_the_real_mode_says_tax_invoice_and_shows_the_sellers_gstin(self):
        app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE], session_cookie_secure=False,
                                  invoice_test_mode=False, seller_name="Ekido Pvt Ltd"))
        self.app, self.engine, self.clients = app, app.state.engine, {}
        org = self.org("asha@example.com")
        order_id, _ = self.order(org)
        text = self.client().get(f"/v1/orgs/{org}/invoices/{order_id}").content.decode("latin-1")
        self.assertIn("TAX INVOICE", text)
        self.assertNotIn("TEST INVOICE", text)
        self.assertIn("Ekido Pvt Ltd", text)


class Numbering(unittest.TestCase):
    def test_the_financial_year_counter_runs_in_order_and_starts_again_each_year(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        from sqlalchemy.pool import StaticPool

        engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        with Session(engine) as s:
            self.assertEqual([v1_invoices.next_sequence(s, "2026-27") for _ in range(3)], [1, 2, 3])
            self.assertEqual(v1_invoices.next_sequence(s, "2027-28"), 1)
            self.assertEqual(v1_invoices.next_sequence(s, "2026-27"), 4)
            s.commit()

    def test_concurrent_callers_never_share_a_number(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        engine = create_engine(f"sqlite:///{os.path.join(folder.name, 'n.db')}", connect_args={"timeout": 20})
        self.addCleanup(engine.dispose)
        Base.metadata.create_all(engine)
        with Session(engine) as s:
            v1_invoices.next_sequence(s, "2026-27")
            s.commit()
        got, barrier = [], threading.Barrier(5)

        def take():
            with Session(engine) as s:
                barrier.wait()
                got.append(v1_invoices.next_sequence(s, "2026-27"))
                s.commit()

        threads = [threading.Thread(target=take) for _ in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(got), [2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
