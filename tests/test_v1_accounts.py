"""The /v1 account routes end to end: email sign-in, the session cookie, teams and API keys."""
import re
import unittest

from fastapi.testclient import TestClient

from ekido_platform.main import Settings, create_app

SITE = "https://ekido.example"


def app(**overrides):
    settings = dict(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE,
                    cors_origins=[SITE], session_cookie_secure=False)
    settings.update(overrides)
    return create_app(Settings(**settings))


class V1Accounts(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app())

    def link_token(self, email):
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.assertEqual(self.client.post("/v1/auth/email/start", json={"email": email}).status_code, 202)
        return re.search(r"token=([\w-]+)", logs.output[-1]).group(1)

    def sign_in(self, email="asha@example.com"):
        response = self.client.post("/v1/auth/email/verify", json={"token": self.link_token(email)})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_sign_in_sets_a_safe_cookie_and_me_works(self):
        response = self.sign_in()
        cookie = response.headers["set-cookie"].lower()
        self.assertIn("ekido_session=", cookie)
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=lax", cookie)
        me = self.client.get("/v1/me").json()
        self.assertEqual((me["email"], me["orgs"]), ("asha@example.com", []))

    def test_errors_have_one_shape(self):
        response = self.client.get("/v1/me")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "SIGN_IN_REQUIRED")
        bad = self.client.post("/v1/auth/email/verify", json={"token": "nope"})
        self.assertEqual((bad.status_code, bad.json()["error"]), (401, "LINK_INVALID"))

    def test_teams_and_api_keys(self):
        self.sign_in()
        org = self.client.post("/v1/orgs", json={"name": "Asha's Clinic"}, headers={"origin": SITE}).json()
        made = self.client.post(f"/v1/orgs/{org['org_id']}/api-keys", json={"name": "server", "mode": "test"}).json()
        secret = made["secret"]
        self.assertTrue(secret.startswith("ek_test_"))
        listed = self.client.get(f"/v1/orgs/{org['org_id']}/api-keys").json()
        self.assertEqual([k["key_id"] for k in listed], [made["key_id"]])
        self.assertNotIn("secret", listed[0])

        # A key can read its own team, but can't manage it or reach /v1/me.
        keyed = TestClient(self.client.app)
        auth = {"authorization": f"Bearer {secret}"}
        self.assertEqual(keyed.get(f"/v1/orgs/{org['org_id']}", headers=auth).status_code, 200)
        self.assertEqual(keyed.post(f"/v1/orgs/{org['org_id']}/api-keys", json={"name": "x", "mode": "test"}, headers=auth).status_code, 403)
        self.assertEqual(keyed.get("/v1/me", headers=auth).status_code, 403)

        self.assertEqual(self.client.delete(f"/v1/orgs/{org['org_id']}/api-keys/{made['key_id']}").status_code, 204)
        self.assertEqual(keyed.get(f"/v1/orgs/{org['org_id']}", headers=auth).status_code, 401)

    def test_other_sites_cant_act_with_the_cookie(self):
        self.sign_in()
        refused = self.client.post("/v1/orgs", json={"name": "x"}, headers={"origin": "https://evil.example"})
        self.assertEqual((refused.status_code, refused.json()["error"]), (403, "BAD_ORIGIN"))

    def test_members_and_roles(self):
        self.sign_in("owner@x.co")
        org = self.client.post("/v1/orgs", json={"name": "Shop"}).json()["org_id"]
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING"):
            invited = self.client.post(f"/v1/orgs/{org}/members", json={"email": "staff@x.co", "role": "member"})
        self.assertEqual(invited.status_code, 202)
        self.assertEqual(self.client.delete(f"/v1/orgs/{org}/members/{invited.json()['user_id']}").status_code, 204)
        owner_id = self.client.get("/v1/me").json()["user_id"]
        last = self.client.delete(f"/v1/orgs/{org}/members/{owner_id}")
        self.assertEqual((last.status_code, last.json()["error"]), (409, "LAST_OWNER"))

    def test_someone_elses_team_looks_missing(self):
        self.sign_in("a@x.co")
        org = self.client.post("/v1/orgs", json={"name": "A"}).json()["org_id"]
        other = TestClient(self.client.app)
        token = self.link_token("b@x.co")
        other.post("/v1/auth/email/verify", json={"token": token})
        self.assertEqual(other.get(f"/v1/orgs/{org}").status_code, 404)

    def test_logout_revokes_the_session(self):
        self.sign_in()
        self.assertEqual(self.client.post("/v1/auth/logout").status_code, 204)
        self.assertEqual(self.client.get("/v1/me").status_code, 401)

    def test_without_an_email_sender_sign_in_says_so(self):
        client = TestClient(app(email_sender="none"))
        response = client.post("/v1/auth/email/start", json={"email": "a@x.co"})
        self.assertEqual((response.status_code, response.json()["error"]), (503, "EMAIL_UNAVAILABLE"))


if __name__ == "__main__":
    unittest.main()
