"""Ticket #156: live job progress over a WebSocket. Only the allow-listed view, only to the job's own team, only from our site."""
import json
import time
import unittest
from unittest import mock

from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from ekido_platform import v1_customer_jobs as module
from ekido_platform.accounts import OrgMember
from tests import test_customer_jobs as fixture

ALLOWED = {"job_id", "state", "stage", "recipe_id", "recipe_title", "base_model", "steps_done", "total_steps", "moved_devices", "quality",
           "model_ready", "payment"}
DEVICE, JOB, SITE, timeline = fixture.DEVICE, fixture.JOB, fixture.SITE, fixture.timeline
PATH = f"/v1/jobs/{JOB}/live"


class Live(unittest.TestCase):
    def setUp(self):
        base = fixture.CustomerJobs("test_signed_out_gets_401")
        base.setUp()
        self.base, self.app, self.client = base, base.app, base.client
        self.app.state.live_interval = 0.02
        self.base.sign_in("asha@example.com")

    def connect(self, origin=SITE, path=PATH):
        return self.client.websocket_connect(path, headers={"origin": origin} if origin else {})

    def code(self, socket):
        with self.assertRaises(WebSocketDisconnect) as closed:
            while True:
                socket.receive_json()
        return closed.exception.code

    def test_it_sends_only_the_allowed_fields_and_no_device(self):
        with self.connect() as socket:
            message = socket.receive_json()
        self.assertEqual((message["type"], set(message["job"])), ("job", ALLOWED))
        self.assertEqual((message["job"]["stage"], message["job"]["steps_done"], message["job"]["moved_devices"]), ("training", 45, 1))
        text = json.dumps(message)
        for secret in (DEVICE["provider_id"], DEVICE["name"], DEVICE["wallet_address"], DEVICE["gpu_model"], "heartbeat"):
            self.assertNotIn(secret, text)

    def test_a_change_is_sent_and_a_finished_job_closes_the_stream(self):
        with self.connect() as socket:
            self.assertEqual(socket.receive_json()["job"]["stage"], "training")
            self.base.state, self.base.events = "SETTLED", timeline("PASS")
            self.assertEqual(socket.receive_json()["job"]["stage"], "passed")
            self.assertEqual(self.code(socket), 1000)

    def test_nothing_is_sent_again_while_nothing_changes(self):
        with self.connect() as socket:
            socket.receive_json()
            time.sleep(0.2)
            self.base.state = "VALIDATING"
            self.assertEqual(socket.receive_json()["job"]["stage"], "checking")

    def test_another_site_is_refused_even_with_the_cookie(self):
        for origin in ("https://evil.example", None, SITE + ".evil.example"):
            with self.subTest(origin=origin), self.assertRaises(WebSocketDisconnect) as refused:
                with self.connect(origin=origin):
                    pass
            self.assertEqual(refused.exception.code, 1008)

    def test_signed_out_and_other_teams_get_nothing(self):
        self.client.post("/v1/auth/logout")
        with self.connect() as socket:
            self.assertEqual(self.code(socket), 4401)
        self.base.team("ravi@example.com", "Ravi Traders")
        with self.connect() as socket:
            self.assertEqual(self.code(socket), 4404)
        with self.connect(path=f"/v1/jobs/{'0' * 8}-0000-4000-8000-000000000000/live") as socket:
            self.assertEqual(self.code(socket), 4404)

    def test_signing_out_ends_an_open_stream(self):
        with self.connect() as socket:
            socket.receive_json()
            self.client.post("/v1/auth/logout")
            self.assertEqual(self.code(socket), 4401)

    def test_leaving_the_team_ends_an_open_stream(self):
        with self.connect() as socket:
            socket.receive_json()
            with Session(self.app.state.engine) as session:
                session.query(OrgMember).filter_by(org_id=self.base.org).delete()
                session.commit()
            self.assertEqual(self.code(socket), 4404)

    def test_a_coordinator_that_is_down_is_reported_without_its_words(self):
        def down(method, path):
            raise OSError("connection to 10.0.0.5 refused")

        with self.connect() as socket:
            socket.receive_json()
            self.app.state.coordinator_get = down
            self.assertEqual(socket.receive_json(), {"type": "unavailable"})

    def test_one_person_cannot_hold_unlimited_streams_and_closed_ones_are_given_back(self):
        with mock.patch.object(module, "LIVE_PER_PERSON", 1):
            with self.connect() as first:
                first.receive_json()
                with self.connect() as second:
                    self.assertEqual(self.code(second), 4429)
            for _ in range(50):
                with self.connect() as again:
                    message = None
                    try:
                        message = again.receive_json()
                    except WebSocketDisconnect:
                        time.sleep(0.05)
                if message:
                    break
            self.assertEqual(message["type"], "job")


if __name__ == "__main__":
    unittest.main()
