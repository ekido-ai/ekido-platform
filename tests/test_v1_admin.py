"""The owner's admin area (C26): who gets in, what it reads from the Coordinator, and the audit log."""
import json
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fastapi.testclient import TestClient

from ekido_platform.main import Settings, create_app

SITE = "https://ekido.example"
OWNER = "piyush@example.com"


class StubCoordinator(BaseHTTPRequestHandler):
    """Answers like the Coordinator's operator API, and records what it was asked."""

    calls: list[tuple[str, str, str | None]] = []

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.calls.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path.startswith("/jobs?"):
            return self.reply(200, [{"job_id": "job-1", "state": "RUNNING"}])
        if self.path == "/jobs/job-1":
            return self.reply(200, {"job_id": "job-1", "state": "RUNNING"})
        if self.path == "/jobs/job-1/checkpoints":
            return self.reply(200, [])
        if self.path == "/providers":
            return self.reply(200, [{"provider_id": "p1", "name": "Asha's laptop", "wallet_address": "0xabc"}])
        if self.path == "/settlements":
            return self.reply(200, [])
        return self.reply(404, {"detail": "not found"})

    def do_POST(self):
        self.calls.append(("POST", self.path, self.headers.get("Authorization")))
        if self.path == "/jobs/job-1/cancel":
            return self.reply(200, {"job_id": "job-1", "state": "CANCELLED"})
        if self.path == "/jobs/job-2/cancel":
            return self.reply(409, {"detail": "illegal transition SETTLED -> CANCELLED"})
        return self.reply(404, {"detail": "not found"})

    def log_message(self, *args):
        pass


class AdminArea(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StubCoordinator)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        StubCoordinator.calls = []
        settings = Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE],
                            session_cookie_secure=False, admin_emails=[OWNER.upper()],
                            coordinator_url=f"http://127.0.0.1:{self.server.server_port}", coordinator_token="operator-secret")
        self.client = TestClient(create_app(settings))

    def sign_in(self, email):
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.client.post("/v1/auth/email/start", json={"email": email})
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.assertEqual(self.client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)

    def test_only_the_listed_owner_gets_in(self):
        self.assertEqual(self.client.get("/v1/admin/orgs").status_code, 401)  # not signed in
        self.sign_in("someone@example.com")
        self.assertEqual(self.client.get("/v1/admin/orgs").status_code, 404)  # signed in, not an admin
        self.client.post("/v1/auth/logout")
        self.sign_in(OWNER)
        self.assertEqual(self.client.get("/v1/admin/orgs").status_code, 200)

    def test_an_api_key_never_opens_the_admin_area(self):
        self.sign_in(OWNER)
        org = self.client.post("/v1/orgs", json={"name": "Ekido"}, headers={"Origin": SITE}).json()
        secret = self.client.post(f"/v1/orgs/{org['org_id']}/api-keys", json={"name": "k", "mode": "test"}, headers={"Origin": SITE}).json()["secret"]
        self.client.post("/v1/auth/logout")
        self.assertEqual(self.client.get("/v1/admin/orgs", headers={"Authorization": f"Bearer {secret}"}).status_code, 404)

    def test_teams_and_coordinator_data_are_listed_with_the_operator_token(self):
        self.sign_in(OWNER)
        self.client.post("/v1/orgs", json={"name": "Sharma Electronics"}, headers={"Origin": SITE})
        orgs = self.client.get("/v1/admin/orgs").json()
        self.assertEqual([(o["name"], o["members"], o["active_keys"]) for o in orgs], [("Sharma Electronics", 1, 0)])
        self.assertEqual(self.client.get("/v1/admin/jobs").json()[0]["job_id"], "job-1")
        self.assertEqual(self.client.get("/v1/admin/jobs/job-1").json()["job"]["state"], "RUNNING")
        self.assertEqual(self.client.get("/v1/admin/providers").json()[0]["name"], "Asha's laptop")
        self.assertEqual(self.client.get("/v1/admin/payments").json(), [])
        self.assertEqual(self.client.get("/v1/admin/jobs/missing").status_code, 404)
        self.assertTrue(all(auth == "Bearer operator-secret" for _, _, auth in StubCoordinator.calls))

    def test_cancel_is_done_by_the_coordinator_and_every_attempt_is_logged(self):
        self.sign_in(OWNER)
        self.assertEqual(self.client.post("/v1/admin/jobs/job-1/cancel", headers={"Origin": SITE}).json()["state"], "CANCELLED")
        refused = self.client.post("/v1/admin/jobs/job-2/cancel", headers={"Origin": SITE})
        self.assertEqual(refused.status_code, 409)
        self.assertIn("illegal transition", refused.json()["detail"])
        audit = self.client.get("/v1/admin/audit").json()
        self.assertEqual([(a["action"], a["target"], a["outcome"], a["email"]) for a in audit],
                         [("job.cancel", "job-2", "refused", OWNER), ("job.cancel", "job-1", "done", OWNER)])

    def test_admin_changes_need_an_allowed_origin(self):
        self.sign_in(OWNER)
        self.assertEqual(self.client.post("/v1/admin/jobs/job-1/cancel", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(StubCoordinator.calls, [])


if __name__ == "__main__":
    unittest.main()
