"""Ticket C12: a customer sees their own job's progress and result, and never which devices trained it."""
import json
import re
import unittest

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from ekido_platform.accounts import JobOwner
from ekido_platform.main import Settings, create_app
from ekido_platform.v1_customer_jobs import customer_view

SITE = "https://ekido.example"
JOB = "00000000-0000-4000-8000-0000000000aa"
SPEC = {"template": "llm-lora/v1", "recipe_id": "ticket-sorter-hinglish", "model": {"id": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "abc"},
        "total_logical_steps": 60, "dataset": {"sha256": "a" * 64}}
DEVICE = {"provider_id": "prov-secret-1", "name": "Ravi's RTX 4090", "wallet_address": "0x" + "ab" * 20, "gpu_model": "RTX 4090"}


def timeline(final_verdict=None):
    events = [
        {"type": "training.progress", "data": {"step": 20, "total_steps": 60, **DEVICE}},
        {"type": "failure.detected", "data": {"cause": "heartbeat lost", **DEVICE}},
        {"type": "recovery.step", "data": {"phase": "resumed", **DEVICE}},
        {"type": "checkpoint.accepted", "data": {"logical_step": 45, "signer": DEVICE["wallet_address"]}},
    ]
    if final_verdict:
        events.append({"type": "quality.result", "data": {"is_final": True, "accuracy": 0.86, "verdict": final_verdict, "verifier": "0x" + "cd" * 20}})
    return events


class CustomerJobs(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE], session_cookie_secure=False))
        self.state = "RUNNING"
        self.events = timeline()

        def coordinator(method, path):
            if path == f"/jobs/{JOB}":
                return 200, {"job_id": JOB, "state": self.state, "spec": SPEC, "assigned_provider": DEVICE}
            if path == f"/jobs/{JOB}/timeline":
                return 200, self.events
            return 404, {"detail": "not found"}

        self.app.state.coordinator_get = coordinator
        self.client = TestClient(self.app)
        self.org = self.team("asha@example.com", "Sharma Electronics")
        with Session(self.app.state.engine) as session:
            session.add(JobOwner(job_id=JOB, org_id=self.org))
            session.commit()

    def sign_in(self, email):
        self.client.post("/v1/auth/logout")
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.client.post("/v1/auth/email/start", json={"email": email})
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.assertEqual(self.client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)

    def team(self, email, name):
        self.sign_in(email)
        return self.client.post("/v1/orgs", json={"name": name}, headers={"Origin": SITE}).json()["org_id"]

    def assert_no_device(self, payload):
        text = json.dumps(payload)
        for secret in (DEVICE["provider_id"], DEVICE["name"], DEVICE["wallet_address"], DEVICE["gpu_model"], "cd" * 20, "heartbeat"):
            self.assertNotIn(secret, text)

    def test_progress_is_shown_without_any_device_identity(self):
        job = self.client.get(f"/v1/jobs/{JOB}")
        self.assertEqual(job.status_code, 200, job.text)
        body = job.json()
        self.assertEqual((body["stage"], body["steps_done"], body["total_steps"], body["moved_devices"]), ("training", 45, 60, 1))
        self.assertEqual((body["recipe_id"], body["quality"]["floor"], body["quality"]["score"], body["model_ready"]), ("ticket-sorter-hinglish", 0.8, None, False))
        self.assert_no_device(body)
        self.assert_no_device(self.client.get(f"/v1/orgs/{self.org}/jobs").json())

    def test_a_passed_job_is_a_model_with_its_score(self):
        self.state, self.events = "SETTLED", timeline("PASS")
        body = self.client.get(f"/v1/jobs/{JOB}").json()
        self.assertEqual((body["stage"], body["quality"]["score"], body["quality"]["passed"], body["model_ready"]), ("passed", 0.86, True, True))
        models = self.client.get(f"/v1/orgs/{self.org}/models").json()
        self.assertEqual([m["model_id"] for m in models], [JOB])
        self.assert_no_device(models)

    def test_a_refunded_job_says_so_and_has_no_model(self):
        self.state, self.events = "REFUNDED", timeline("FAIL")
        body = self.client.get(f"/v1/jobs/{JOB}").json()
        self.assertEqual((body["stage"], body["quality"]["passed"], body["model_ready"]), ("refunded", False, False))
        self.assertEqual(self.client.get(f"/v1/orgs/{self.org}/models").json(), [])
        self.assertEqual(self.client.get(f"/v1/models/{JOB}/download").status_code, 409)

    def test_another_team_cant_see_the_job(self):
        self.team("ravi@example.com", "Ravi Traders")
        self.assertEqual(self.client.get(f"/v1/jobs/{JOB}").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/models/{JOB}/download").status_code, 404)
        self.assertEqual(self.client.get(f"/v1/jobs/{'0' * 8}-0000-4000-8000-000000000000").status_code, 404)

    def test_signed_out_gets_401(self):
        self.client.post("/v1/auth/logout")
        self.assertEqual(self.client.get(f"/v1/jobs/{JOB}").status_code, 401)

    def test_the_coordinator_down_is_a_clear_503(self):
        def down(method, path):
            raise OSError("refused")

        self.app.state.coordinator_get = down
        response = self.client.get(f"/v1/jobs/{JOB}")
        self.assertEqual((response.status_code, response.json()["error"]), (503, "UNAVAILABLE"))

    def test_the_view_is_an_allow_list(self):
        view = customer_view({"job_id": JOB, "state": "RUNNING", "spec": SPEC, "anything": DEVICE}, timeline(), None)
        self.assertEqual(set(view), {"job_id", "state", "stage", "recipe_id", "recipe_title", "base_model", "steps_done", "total_steps",
                                     "moved_devices", "quality", "model_ready", "payment"})


if __name__ == "__main__":
    unittest.main()
