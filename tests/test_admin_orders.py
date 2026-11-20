"""The owner's admin area (C26): listing orders and refunding a paid one, with every attempt in the audit log."""
import re
import unittest

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_platform.main import Settings, create_app
from ekido_platform.payments import HELD, REFUNDED, PaymentOrder, RazorpayRail
from tests.test_payments import SECRET, FakeGateway, event, quote, sign

SITE = "https://ekido.example"
OWNER = "piyush@example.com"


class AdminOrders(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE],
                                       session_cookie_secure=False, admin_emails=[OWNER]))
        self.gateway = FakeGateway()
        self.app.state.payment_rail = RazorpayRail(self.gateway, SECRET)
        self.client = TestClient(self.app)

    def sign_in(self, email):
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.client.post("/v1/auth/email/start", json={"email": email})
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.assertEqual(self.client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)

    def paid_order(self):
        rail = self.app.state.payment_rail
        with Session(self.app.state.engine) as session:
            order = rail.create_order(session, org_id="org-1", quote_id="q-1", quote=quote())
            pid = self.gateway.pay(order.gateway_order_id)
            raw = event("payment.captured", self.gateway.payments[pid])
            rail.handle_webhook(session, raw, sign(raw), f"evt_{pid}")
            session.refresh(order)
            self.assertEqual(order.status, HELD)
            return order.id

    def refund(self, order_id):
        return self.client.post(f"/v1/admin/orders/{order_id}/refund", headers={"Origin": SITE})

    def audit(self):
        return [(a["action"], a["target"], a["outcome"]) for a in self.client.get("/v1/admin/audit").json()]

    def test_only_the_owner_sees_and_refunds_orders(self):
        order_id = self.paid_order()
        self.assertEqual(self.client.get("/v1/admin/orders").status_code, 401)
        self.sign_in("someone@example.com")
        self.assertEqual(self.client.get("/v1/admin/orders").status_code, 404)
        self.assertEqual(self.refund(order_id).status_code, 404)
        self.assertEqual(self.gateway.calls.count("refund_payment"), 0)

    def test_the_owner_lists_orders_and_refunds_a_paid_one_once(self):
        order_id = self.paid_order()
        self.sign_in(OWNER)
        rows = self.client.get("/v1/admin/orders").json()
        self.assertEqual([(r["order_id"], r["status"]) for r in rows], [(order_id, HELD)])
        done = self.refund(order_id)
        self.assertEqual((done.status_code, done.json()["status"]), (200, REFUNDED))
        again = self.refund(order_id)  # already refunded: the rail returns it unchanged, no second refund at Razorpay
        self.assertEqual(again.json()["status"], REFUNDED)
        self.assertEqual(self.gateway.calls.count("refund_payment"), 1)
        self.assertEqual(self.audit(), [("order.refund", order_id, "done"), ("order.refund", order_id, "done")])

    def test_an_unpaid_order_is_refused_and_logged(self):
        with Session(self.app.state.engine) as session:
            order_id = self.app.state.payment_rail.create_order(session, org_id="org-1", quote_id="q-1", quote=quote()).id
        self.sign_in(OWNER)
        refused = self.refund(order_id)
        self.assertEqual((refused.status_code, refused.json()["error"]), (409, "REFUND_REFUSED"))
        self.assertEqual(self.audit(), [("order.refund", order_id, "refused")])

    def test_razorpay_down_keeps_the_order_held_and_says_try_again(self):
        order_id = self.paid_order()
        self.gateway.refund_fails = True
        self.sign_in(OWNER)
        response = self.refund(order_id)
        self.assertEqual((response.status_code, response.json()["error"]), (503, "GATEWAY_UNAVAILABLE"))
        with Session(self.app.state.engine) as session:
            self.assertEqual(session.get(PaymentOrder, order_id).status, HELD)
        self.assertEqual(self.audit(), [("order.refund", order_id, "unreachable")])

    def test_without_payments_configured_nothing_is_refunded(self):
        order_id = self.paid_order()
        self.app.state.payment_rail = None
        self.sign_in(OWNER)
        self.assertEqual(self.refund(order_id).status_code, 503)

    def test_a_refund_needs_an_allowed_origin(self):
        order_id = self.paid_order()
        self.sign_in(OWNER)
        response = self.client.post(f"/v1/admin/orders/{order_id}/refund", headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.gateway.calls.count("refund_payment"), 0)


if __name__ == "__main__":
    unittest.main()
