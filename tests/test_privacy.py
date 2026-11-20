"""C17: consent records, export my data, delete my data (DPDP), through the real /v1 routes."""

import io
import json
import re
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_platform import accounts, privacy
from ekido_platform.accounts import (AccountError, ApiKey, Consent, InferenceUsage, JobOwner, LoginToken, Org, OrgMember, UserEmail, WebSession)
from ekido_platform.main import Settings, create_app
from ekido_platform.models import User
from ekido_platform.payments import PaymentOrder
from ekido_platform.plans import ACTIVE, PlanUsage, TeamPlan

SITE = "https://ekido.example"


class FakeSubscriptions:
    def __init__(self):
        self.cancelled = []

    def cancel_subscription(self, subscription_id, *, at_cycle_end):
        self.cancelled.append((subscription_id, at_cycle_end))
        return {}


class Base(unittest.TestCase):
    def setUp(self):
        settings = Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE], session_cookie_secure=False,
                            razorpay_key_id="rzp_test_abc", razorpay_key_secret="s", razorpay_webhook_secret="w", razorpay_plan_starter="plan_s", razorpay_plan_business="plan_b")
        self.app = create_app(settings)
        self.gateway = FakeSubscriptions()
        self.app.state.plans.gateway = self.gateway
        self.engine = self.app.state.engine
        self.clients = {}

    def client(self, email="asha@example.com") -> TestClient:
        """A browser signed in as this person (its own cookie jar)."""
        if email not in self.clients:
            client = TestClient(self.app)
            with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
                client.post("/v1/auth/email/start", json={"email": email})
            token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
            self.assertEqual(client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)
            self.clients[email] = client
        return self.clients[email]

    def user_id(self, email):
        with Session(self.engine) as s:
            return s.scalar(select(UserEmail.user_id).where(UserEmail.email == email))

    def make_org(self, email, name="Asha Traders"):
        return self.client(email).post("/v1/orgs", json={"name": name}).json()["org_id"]

    def count(self, model, *where):
        with Session(self.engine) as s:
            return len(s.scalars(select(model).where(*where)).all())


class ConsentTests(Base):
    def test_a_grant_and_a_withdrawal_are_both_kept_in_order(self):
        c = self.client()
        self.assertEqual(c.get("/v1/me/consents").json(), [])
        first = c.post("/v1/me/consents", json={"purpose": "marketing-email", "granted": True, "policy_version": "2026-11-19"})
        self.assertEqual(first.status_code, 201)
        self.assertEqual((first.json()["purpose"], first.json()["granted"], first.json()["policy_version"]), ("marketing-email", True, "2026-11-19"))
        self.assertTrue(first.json()["recorded_at"])
        c.post("/v1/me/consents", json={"purpose": "training-data-processing", "granted": True})
        c.post("/v1/me/consents", json={"purpose": "marketing-email", "granted": False})
        history = c.get("/v1/me/consents").json()
        self.assertEqual([(h["purpose"], h["granted"]) for h in history], [("marketing-email", True), ("training-data-processing", True), ("marketing-email", False)])
        with Session(self.engine) as s:
            self.assertEqual(privacy.current_consents(s, self.user_id("asha@example.com")), {"marketing-email": False, "training-data-processing": True})

    def test_history_keeps_its_order_even_when_the_clock_repeats(self):
        self.client()
        uid = self.user_id("asha@example.com")
        same = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
        with Session(self.engine) as s:
            for granted in (True, False, True, False, True):  # five answers at one instant
                privacy.record_consent(s, uid, "marketing-email", granted, now=same)
            self.assertEqual([c["granted"] for c in privacy.list_consents(s, uid)], [True, False, True, False, True])
            self.assertEqual(privacy.current_consents(s, uid), {"marketing-email": True})

    def test_bad_consents_are_refused_and_not_stored(self):
        c = self.client()
        for body in ({"purpose": "newsletter", "granted": True}, {"purpose": "account", "granted": "yes"}, {"purpose": "account"}, {"granted": True},
                     {"purpose": "account", "granted": True, "extra": 1}, {"purpose": "account", "granted": True, "policy_version": "x" * 41}):
            with self.subTest(body=body):
                self.assertIn(c.post("/v1/me/consents", json=body).status_code, (422,))
        self.assertEqual(self.count(Consent), 0)

    def test_people_only_see_their_own_consents(self):
        self.client("a@example.com").post("/v1/me/consents", json={"purpose": "account", "granted": True})
        self.assertEqual(self.client("b@example.com").get("/v1/me/consents").json(), [])

    def test_signed_out_and_api_key_callers_are_refused(self):
        org = self.make_org("asha@example.com")
        secret = self.client().post(f"/v1/orgs/{org}/api-keys", json={"name": "k", "mode": "test"}).json()
        key = secret.get("secret") or secret.get("key")
        anonymous = TestClient(self.app)
        for call in (lambda c, h: c.get("/v1/me/consents", headers=h), lambda c, h: c.get("/v1/me/data-export", headers=h), lambda c, h: c.delete("/v1/me", headers=h)):
            self.assertEqual(call(anonymous, {}).status_code, 401)
            if key:
                self.assertEqual(call(anonymous, {"Authorization": f"Bearer {key}"}).status_code, 403)


class ExportTests(Base):
    def export(self, email="asha@example.com"):
        response = self.client(email).get("/v1/me/data-export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")
        self.assertIn("attachment", response.headers["content-disposition"])
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        return {n: archive.read(n).decode() for n in archive.namelist()}

    def test_the_export_lists_everything_held_about_the_person(self):
        c = self.client()
        org = self.make_org("asha@example.com")
        c.post(f"/v1/orgs/{org}/api-keys", json={"name": "my key", "mode": "test"})
        c.post("/v1/me/consents", json={"purpose": "account", "granted": True, "policy_version": "v1"})
        with Session(self.engine) as s:
            s.add(JobOwner(job_id="job-1", org_id=org))
            s.add(PaymentOrder(org_id=org, quote_id="q", terms_hash="a" * 64, amount_paise=15222, currency="INR", gateway_order_id="order_1", status="HELD", quote={}))
            s.commit()
        files = self.export()
        self.assertEqual(set(files), {"README.txt", "profile.json", "emails.json", "teams.json", "api_keys.json", "sessions.json", "consents.json", "jobs.json",
                                      "deletion_requests.json", "payments.json", "plans.json"})
        self.assertEqual(json.loads(files["emails.json"])[0]["email"], "asha@example.com")
        self.assertEqual(json.loads(files["teams.json"])[0]["name"], "Asha Traders")
        self.assertEqual(json.loads(files["teams.json"])[0]["your_role"], "owner")
        self.assertEqual(json.loads(files["api_keys.json"])[0]["name"], "my key")
        self.assertEqual(json.loads(files["consents.json"])[0]["policy_version"], "v1")
        self.assertEqual(json.loads(files["jobs.json"]), [{"job_id": "job-1", "team": org}])
        self.assertEqual(json.loads(files["payments.json"])[0]["amount_paise"], 15222)

    def test_the_export_holds_no_secret_and_no_other_persons_email(self):
        org = self.make_org("asha@example.com")
        self.client().post(f"/v1/orgs/{org}/members", json={"email": "friend@example.com", "role": "member"})
        key = self.client().post(f"/v1/orgs/{org}/api-keys", json={"name": "k", "mode": "test"}).json()
        everything = "\n".join(v for name, v in self.export().items() if name != "README.txt")  # the README only mentions the words
        self.assertNotIn("friend@example.com", everything)
        self.assertEqual(json.loads(self.export()["teams.json"])[0]["members"], 2)  # a count, not a list of people
        with Session(self.engine) as s:
            for stored in s.scalars(select(ApiKey.key_hash)):
                self.assertNotIn(stored, everything)
        secrets = [v for v in key.values() if isinstance(v, str) and v.startswith("ek_") and len(v) > 20]
        self.assertTrue(secrets, "the key response should hold the one-time secret")
        for value in secrets:
            self.assertNotIn(value, everything)  # the short public prefix may appear; the secret may not
        for forbidden in ("token_hash", "key_hash", "password"):
            self.assertNotIn(forbidden, everything)

    def test_each_person_gets_only_their_own_data_and_the_same_data_gives_the_same_bytes(self):
        self.make_org("a@example.com", "Team A")
        self.make_org("b@example.com", "Team B")
        self.assertNotIn("Team B", "".join(self.export("a@example.com").values()))
        with Session(self.engine) as s:
            uid = self.user_id("a@example.com")
            self.assertEqual(privacy.export_zip(privacy.export_data(s, uid)), privacy.export_zip(privacy.export_data(s, uid)))


class DeleteTests(Base):
    def setup_asha(self):
        """Asha is the only member of her team: it has a key, two jobs, a subscription and a payment."""
        c = self.client()
        org = self.make_org("asha@example.com")
        c.post(f"/v1/orgs/{org}/api-keys", json={"name": "k", "mode": "test"})
        c.post("/v1/me/consents", json={"purpose": "account", "granted": True})
        with Session(self.engine) as s:
            s.add_all([JobOwner(job_id="job-1", org_id=org), JobOwner(job_id="job-2", org_id=org)])
            key_id = s.scalar(select(ApiKey.id).where(ApiKey.org_id == org))
            s.add(InferenceUsage(id="u1", key_id=key_id, model_id="m", created_at=datetime.now(timezone.utc)))
            s.add(TeamPlan(org_id=org, plan="starter", status=ACTIVE, subscription_id="sub_1", period_end=datetime.now(timezone.utc) + timedelta(days=10)))
            s.add(PlanUsage(org_id=org, kind="jobs", period="p:1", used=2))
            s.add(PaymentOrder(org_id=org, quote_id="q", terms_hash="a" * 64, amount_paise=15222, currency="INR", gateway_order_id="order_1", status="SETTLED", quote={"x": 1}))
            s.commit()
        return org

    def test_deleting_erases_the_person_their_empty_team_and_schedules_the_rest(self):
        org = self.setup_asha()
        uid = self.user_id("asha@example.com")
        response = self.client().delete("/v1/me")
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual(body["status"], "SCHEDULED")
        due = datetime.fromisoformat(body["due_by"])
        self.assertAlmostEqual((due - datetime.now(timezone.utc)).total_seconds(), 30 * 86400, delta=60)
        # gone at once:
        self.assertEqual((self.count(UserEmail), self.count(WebSession), self.count(Consent), self.count(OrgMember), self.count(Org), self.count(ApiKey), self.count(JobOwner),
                          self.count(InferenceUsage), self.count(TeamPlan), self.count(PlanUsage)), (0,) * 10)
        with Session(self.engine) as s:
            self.assertIsNone(s.get(User, uid).display_name)  # an empty tombstone, linked to no email
        # billing stops, and payment records stay:
        self.assertEqual(self.gateway.cancelled, [("sub_1", True)])
        self.assertEqual(self.count(PaymentOrder), 1)
        # the other services are told what to destroy:
        with Session(self.engine) as s:
            self.assertEqual(sorted((t.kind, t.ref) for t in privacy.pending_tasks(s)), [("job_data", "job-1"), ("job_data", "job-2")])
            self.assertEqual(privacy.deletion_status(s, uid)["status"], "SCHEDULED")
        # the session no longer works:
        self.assertEqual(self.client().get("/v1/me").status_code, 401)

    def test_the_request_is_done_when_every_task_is_done_and_late_ones_are_listed(self):
        self.setup_asha()
        uid = self.user_id("asha@example.com")
        self.client().delete("/v1/me")
        with Session(self.engine) as s:
            tasks = privacy.pending_tasks(s)
            self.assertEqual(privacy.overdue(s), [])
            self.assertEqual(len(privacy.overdue(s, now=datetime.now(timezone.utc) + timedelta(days=31))), 1)  # past 30 days and not done
            first = privacy.complete_task(s, tasks[0].id)
            self.assertEqual(first.status, "SCHEDULED")
            privacy.complete_task(s, tasks[0].id)  # repeating a finished task changes nothing
            done = privacy.complete_task(s, tasks[1].id)
            self.assertEqual(done.status, "DONE")
            self.assertIsNotNone(done.completed_at)
            self.assertEqual(privacy.overdue(s, now=datetime.now(timezone.utc) + timedelta(days=60)), [])
            self.assertEqual(privacy.pending_tasks(s), [])
            self.assertEqual(privacy.deletion_status(s, uid)["status"], "DONE")

    def test_a_person_with_nothing_to_destroy_elsewhere_is_done_at_once(self):
        self.client("solo@example.com")
        self.client("solo@example.com").delete("/v1/me")
        with Session(self.engine) as s:
            self.assertEqual(privacy.deletion_status(s, self.user_id_after("solo"))["status"], "DONE")

    def user_id_after(self, name):  # the email is erased, so find the tombstone through its deletion request
        with Session(self.engine) as s:
            return s.scalar(select(privacy.DeletionRequest.user_id).order_by(privacy.DeletionRequest.created_at.desc()))

    def test_leaving_a_shared_team_does_not_erase_it_or_touch_the_other_members(self):
        org = self.make_org("asha@example.com")
        self.client().post(f"/v1/orgs/{org}/members", json={"email": "ravi@example.com", "role": "owner"})
        self.client("ravi@example.com")  # ravi signs in (his account existed unverified)
        with Session(self.engine) as s:
            s.add(JobOwner(job_id="job-1", org_id=org))
            s.commit()
        self.client().delete("/v1/me")
        with Session(self.engine) as s:
            self.assertEqual([m.role for m in s.scalars(select(OrgMember).where(OrgMember.org_id == org))], ["owner"])
            self.assertIsNotNone(s.get(Org, org))
            self.assertEqual(privacy.pending_tasks(s), [])  # the team's jobs stay: it still has an owner
        self.assertEqual(self.client("ravi@example.com").get("/v1/me").json()["email"], "ravi@example.com")

    def test_the_last_owner_of_a_team_with_other_people_must_hand_it_over_and_nothing_is_deleted(self):
        org = self.make_org("asha@example.com")
        self.client().post(f"/v1/orgs/{org}/members", json={"email": "ravi@example.com", "role": "member"})
        response = self.client().delete("/v1/me")
        self.assertEqual((response.status_code, response.json()["error"]), (409, "OWNER_OF_SHARED_TEAM"))
        self.assertEqual((self.count(UserEmail, UserEmail.email == "asha@example.com"), self.count(privacy.DeletionRequest)), (1, 0))
        self.assertEqual(self.client().get("/v1/me").status_code, 200)  # still signed in

    def test_if_anything_fails_midway_nothing_at_all_is_deleted(self):
        org = self.setup_asha()

        def broken(session, org_id):
            raise RuntimeError("a module could not clear its rows")

        with Session(self.engine) as s, self.assertRaises(RuntimeError):
            privacy.request_deletion(s, self.user_id("asha@example.com"), org_erasers=[broken])
        self.assertEqual((self.count(UserEmail), self.count(Org), self.count(ApiKey), self.count(JobOwner), self.count(Consent), self.count(privacy.DeletionRequest),
                          self.count(privacy.ErasureTask)), (1, 1, 1, 2, 1, 0, 0))
        self.assertEqual(self.client().get("/v1/me").status_code, 200)

    def test_asking_twice_does_not_erase_twice(self):
        self.setup_asha()
        uid = self.user_id("asha@example.com")
        with Session(self.engine) as s:
            first = privacy.request_deletion(s, uid)
            again = privacy.request_deletion(s, uid)
            self.assertEqual(first.id, again.id)
            self.assertEqual(len(privacy.pending_tasks(s)), 2)

    def test_the_same_email_can_sign_up_again_as_a_new_empty_account(self):
        self.setup_asha()
        old_id = self.user_id("asha@example.com")
        self.client().delete("/v1/me")
        self.clients.pop("asha@example.com")
        fresh = self.client("asha@example.com")
        me = fresh.get("/v1/me").json()
        self.assertEqual((me["email"], me["orgs"]), ("asha@example.com", []))
        self.assertNotEqual(self.user_id("asha@example.com"), old_id)

    def test_an_unused_sign_in_link_dies_with_the_account(self):
        self.client()
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            TestClient(self.app).post("/v1/auth/email/start", json={"email": "asha@example.com"})
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.client().delete("/v1/me")
        self.assertEqual(self.count(LoginToken), 0)
        self.assertEqual(TestClient(self.app).post("/v1/auth/email/verify", json={"token": token}).status_code, 401)

    def test_other_people_and_teams_are_untouched(self):
        self.setup_asha()
        other = self.make_org("ravi@example.com", "Ravi Co")
        with Session(self.engine) as s:
            s.add(JobOwner(job_id="job-9", org_id=other))
            s.commit()
        self.client().delete("/v1/me")
        self.assertEqual(self.client("ravi@example.com").get("/v1/me").json()["orgs"][0]["name"], "Ravi Co")
        self.assertEqual(self.count(JobOwner, JobOwner.org_id == other), 1)
        self.assertEqual(self.count(ApiKey), 0)

    def test_a_keys_made_in_someone_elses_team_stay_but_no_longer_lead_to_a_person(self):
        org = self.make_org("ravi@example.com", "Ravi Co")
        self.client("ravi@example.com").post(f"/v1/orgs/{org}/members", json={"email": "asha@example.com", "role": "admin"})
        asha = self.client("asha@example.com")
        asha.post(f"/v1/orgs/{org}/api-keys", json={"name": "asha's key", "mode": "test"})
        asha.delete("/v1/me")
        self.assertEqual(self.count(ApiKey), 1)  # the team still owns its key
        with Session(self.engine) as s:
            self.assertEqual(s.scalar(select(func_count())), 1)

    def test_the_deletion_record_holds_no_personal_data(self):
        self.setup_asha()
        self.client().delete("/v1/me")
        with Session(self.engine) as s:
            row = s.scalar(select(privacy.DeletionRequest))
            dumped = json.dumps({"user": row.user_id, "summary": row.summary, "status": row.status})
        self.assertNotIn("asha", dumped)
        self.assertNotIn("@", dumped)
        self.assertEqual(row.summary, {"teams_erased": 1, "teams_left": 0, "erasure_tasks": 2})


def func_count():
    from sqlalchemy import func

    return func.count(ApiKey.id)


if __name__ == "__main__":
    unittest.main()
