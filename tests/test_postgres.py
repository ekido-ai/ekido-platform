"""Runs against a real Postgres when TEST_DATABASE_URL is set (CI's Postgres job, after `alembic upgrade head`).

The URL connects as the platform's own user, whose search_path is the `platform` schema. Each test runs inside a
transaction that is rolled back. One test proves the separation between services: this user cannot read the
Coordinator's tables at all.
"""
import os
import unittest

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from ekido_platform import accounts
from ekido_platform.accounts import AccountError
from ekido_platform.models import Base

URL = os.environ.get("TEST_DATABASE_URL", "")


@unittest.skipUnless(URL.startswith("postgresql"), "set TEST_DATABASE_URL to a migrated Postgres database")
class OnPostgres(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(URL)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def setUp(self):
        self.connection = self.engine.connect()
        self.outer = self.connection.begin()
        self.session = Session(bind=self.connection, join_transaction_mode="create_savepoint")

    def tearDown(self):
        self.session.close()
        self.outer.rollback()
        self.connection.close()

    def test_migrations_created_every_table_the_models_define(self):
        existing = set(inspect(self.connection).get_table_names())
        self.assertEqual(set(Base.metadata.tables) - existing, set())
        self.assertEqual(self.connection.execute(text("select current_schema()")).scalar(), "platform")

    def test_accounts_flow(self):
        token = accounts.start_login(self.session, "Asha@Example.com")
        user, session_token = accounts.finish_login(self.session, token)
        self.assertEqual(accounts.user_for_session(self.session, session_token), user.id)
        org = accounts.create_org(self.session, user.id, "Asha's Clinic")
        accounts.add_member(self.session, org.id, "staff@example.com", "member")
        key, secret = accounts.create_api_key(self.session, org.id, user.id, "server", "live")
        self.assertEqual(accounts.principal_for_key(self.session, secret).org_id, org.id)
        self.assertEqual(len(accounts.org_view(self.session, org.id)["members"]), 2)
        with self.assertRaises(AccountError):
            accounts.finish_login(self.session, token)  # used once already

    def test_unique_email_is_enforced(self):
        token = accounts.start_login(self.session, "x@example.com")
        accounts.finish_login(self.session, token)
        again = accounts.start_login(self.session, "X@example.com")
        user, _ = accounts.finish_login(self.session, again)
        rows = self.session.scalars(select(accounts.UserEmail).where(accounts.UserEmail.email == "x@example.com")).all()
        self.assertEqual([r.user_id for r in rows], [user.id])

    def test_the_coordinators_tables_are_out_of_reach(self):
        # Separate users per service: Postgres itself refuses, so no code path can read another service's data.
        # The savepoint must be inside assertRaises, so it rolls back the failed statement before the check ends.
        with self.assertRaises(ProgrammingError) as refused, self.connection.begin_nested():
            self.connection.execute(text("select count(*) from coordinator.jobs"))
        self.assertIn("permission denied", str(refused.exception).lower())


if __name__ == "__main__":
    unittest.main()
