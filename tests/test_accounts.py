"""Accounts logic: sign-in links, sessions, teams, roles and API keys."""
import unittest
from datetime import timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ekido_platform import accounts
from ekido_platform.accounts import AccountError, ApiKey, LoginToken, Principal, WebSession
from ekido_platform.models import Base, utcnow


class AccountsTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.session = Session(engine)

    def tearDown(self):
        self.session.close()

    def sign_in(self, email="Asha@Example.com"):
        token = accounts.start_login(self.session, email)
        return accounts.finish_login(self.session, token)

    def test_first_sign_in_creates_the_user_and_a_session(self):
        user, session_token = self.sign_in()
        self.assertEqual(accounts.user_for_session(self.session, session_token), user.id)
        self.assertEqual(accounts.me(self.session, user.id)["email"], "asha@example.com")
        again, _ = self.sign_in("asha@example.com")
        self.assertEqual(again.id, user.id)

    def test_tokens_are_stored_only_as_hashes(self):
        token = accounts.start_login(self.session, "a@b.co")
        _, session_token = accounts.finish_login(self.session, token)
        stored = [r.token_hash for r in self.session.scalars(select(LoginToken))] + [r.token_hash for r in self.session.scalars(select(WebSession))]
        self.assertNotIn(token, stored)
        self.assertNotIn(session_token, stored)

    def test_a_link_works_once_and_expires(self):
        token = accounts.start_login(self.session, "a@b.co")
        accounts.finish_login(self.session, token)
        with self.assertRaises(AccountError) as reused:
            accounts.finish_login(self.session, token)
        self.assertEqual(reused.exception.status, 401)
        old = accounts.start_login(self.session, "a@b.co")
        with self.assertRaises(AccountError):
            accounts.finish_login(self.session, old, now=utcnow() + accounts.LOGIN_TOKEN_TTL + timedelta(seconds=1))

    def test_links_are_rate_limited_per_address(self):
        for _ in range(accounts.LOGIN_LINKS_PER_HOUR):
            accounts.start_login(self.session, "a@b.co")
        with self.assertRaises(AccountError) as limited:
            accounts.start_login(self.session, "A@B.co")
        self.assertEqual(limited.exception.status, 429)

    def test_bad_email_is_refused(self):
        with self.assertRaises(AccountError):
            accounts.start_login(self.session, "not-an-email")

    def test_logout_and_expiry_end_the_session(self):
        _, token = self.sign_in()
        accounts.end_session(self.session, token)
        self.assertIsNone(accounts.user_for_session(self.session, token))
        _, token = self.sign_in()
        self.assertIsNone(accounts.user_for_session(self.session, token, now=utcnow() + accounts.SESSION_TTL + timedelta(seconds=1)))

    def test_teams_roles_and_the_last_owner(self):
        owner, _ = self.sign_in("owner@x.co")
        org = accounts.create_org(self.session, owner.id, "Asha's Clinic")
        member_id = accounts.add_member(self.session, org.id, "staff@x.co", "member")
        outsider, _ = self.sign_in("other@x.co")
        accounts.require_role(self.session, Principal(via="session", user_id=member_id), org.id, *accounts.ROLES)
        with self.assertRaises(AccountError) as forbidden:
            accounts.require_role(self.session, Principal(via="session", user_id=member_id), org.id, "owner", "admin")
        self.assertEqual(forbidden.exception.status, 403)
        with self.assertRaises(AccountError) as hidden:
            accounts.require_role(self.session, Principal(via="session", user_id=outsider.id), org.id)
        self.assertEqual(hidden.exception.status, 404)
        with self.assertRaises(AccountError) as last:
            accounts.remove_member(self.session, org.id, owner.id)
        self.assertEqual(last.exception.code, "LAST_OWNER")
        accounts.remove_member(self.session, org.id, member_id)
        self.assertEqual([m["email"] for m in accounts.org_view(self.session, org.id)["members"]], ["owner@x.co"])

    def test_api_keys(self):
        owner, _ = self.sign_in()
        org = accounts.create_org(self.session, owner.id, "Shop")
        key, secret = accounts.create_api_key(self.session, org.id, owner.id, "server", "test")
        self.assertTrue(secret.startswith("ek_test_"))
        self.assertNotIn(secret, [k.key_hash for k in self.session.scalars(select(ApiKey))])
        found = accounts.principal_for_key(self.session, secret)
        self.assertEqual((found.via, found.org_id, found.mode), ("api_key", org.id, "test"))
        self.assertEqual(accounts.list_api_keys(self.session, org.id)[0]["prefix"], secret[:12])
        self.assertNotIn("secret", accounts.list_api_keys(self.session, org.id)[0])
        accounts.revoke_api_key(self.session, org.id, key.id)
        self.assertIsNone(accounts.principal_for_key(self.session, secret))
        self.assertIsNone(accounts.principal_for_key(self.session, "ek_test_madeup"))

    def test_api_keys_stay_inside_their_team_and_cant_manage_it(self):
        owner, _ = self.sign_in()
        mine = accounts.create_org(self.session, owner.id, "Mine")
        theirs = accounts.create_org(self.session, owner.id, "Theirs")
        _, secret = accounts.create_api_key(self.session, mine.id, owner.id, "k", "live")
        key = accounts.principal_for_key(self.session, secret)
        accounts.require_role(self.session, key, mine.id, *accounts.ROLES)
        with self.assertRaises(AccountError):
            accounts.require_role(self.session, key, theirs.id, *accounts.ROLES)
        with self.assertRaises(AccountError):
            accounts.require_role(self.session, key, mine.id, "owner", "admin")

    def test_key_validation(self):
        owner, _ = self.sign_in()
        org = accounts.create_org(self.session, owner.id, "Shop")
        for bad in (("", "test", None), ("k", "prod", None), ("k", "test", ["admin"])):
            with self.assertRaises(AccountError):
                accounts.create_api_key(self.session, org.id, owner.id, *bad)


if __name__ == "__main__":
    unittest.main()
