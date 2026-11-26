"""Ticket #142: the optional Valkey cache. The platform must behave the same with it, without it and with it broken.

By default the tests use fakeredis (in memory, with Lua). Set TEST_VALKEY_URL to also run them against a real server, as CI does."""
import hashlib
import hmac
import json
import os
import re
import unittest
from unittest import mock

import fakeredis
from fastapi.testclient import TestClient

from ekido_platform import cache as cache_module
from ekido_platform.cache import Cache, digest
from ekido_platform.main import Settings, create_app
from ekido_platform.v1_assistant import SharedRateLimit, SharedRefusalBreaker

SITE = "https://ekido.example"
REAL = os.environ.get("TEST_VALKEY_URL", "")


class Broken:
    """A server that is down: every command fails."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise ConnectionError("refused")

        return fail


def fake():
    return fakeredis.FakeRedis(decode_responses=True)


def real():
    import redis

    client = redis.Redis.from_url(REAL, decode_responses=True)
    client.flushdb()
    return client


class CacheContract:
    """The same checks for the fake and for a real server."""

    def make(self):
        raise NotImplementedError

    def setUp(self):
        self.client = self.make()
        self.cache = Cache(self.client)

    def test_a_caller_is_loaded_once_and_no_secret_is_stored(self):
        loads = []
        for _ in range(3):
            self.assertEqual(self.cache.caller("key", "ek_test_secret", lambda: loads.append(1) or {"org_id": "o"}), {"org_id": "o"})
        self.assertEqual(len(loads), 1)
        stored = " ".join(self.client.keys("*")) + " ".join(str(self.client.get(k)) for k in self.client.keys("*"))
        self.assertNotIn("ek_test_secret", stored)
        self.assertLessEqual(self.client.ttl(f"ekido:auth:key:{digest('ek_test_secret')}"), cache_module.AUTH_TTL)

    def test_a_refusal_is_not_remembered(self):
        self.assertIsNone(self.cache.caller("key", "s", lambda: None))
        self.assertEqual(self.cache.caller("key", "s", lambda: {"a": 1}), {"a": 1})

    def test_forgetting_drops_everyone(self):
        self.cache.caller("key", "s", lambda: {"v": 1})
        self.cache.forget_callers()
        self.assertEqual(self.cache.caller("key", "s", lambda: {"v": 2}), {"v": 2})

    def test_a_revoke_during_the_database_read_is_not_overwritten(self):
        def load():
            self.cache.forget_callers()  # someone revokes while we are still reading
            return {"v": "stale"}

        self.cache.caller("key", "s", load)
        self.assertEqual(self.cache.caller("key", "s", lambda: {"v": "fresh"}), {"v": "fresh"})

    def test_losing_the_generation_never_revives_old_entries(self):
        self.cache.caller("key", "s", lambda: {"v": 1})
        self.client.delete("ekido:auth:generation")  # evicted by the memory cap
        self.assertEqual(self.cache.caller("key", "s", lambda: {"v": 2}), {"v": 2})

    def test_the_rate_window_is_shared_and_bounded(self):
        other = Cache(self.client)
        results = [(self.cache if i % 2 else other).allow("b", "person", 3, 60) for i in range(5)]
        self.assertEqual(results, [True, True, True, False, False])
        self.assertEqual(self.cache.count("b", "person", 60), 3)
        self.assertTrue(self.cache.allow("b", "someone else", 3, 60))

    def test_the_breaker_opens_for_every_process(self):
        first, second = SharedRefusalBreaker(self.cache, limit=2, window=60), SharedRefusalBreaker(Cache(self.client), limit=2, window=60)
        first.record("p")
        self.assertFalse(second.open("p"))
        first.record("p")
        self.assertTrue(second.open("p"))


class WithFake(CacheContract, unittest.TestCase):
    def make(self):
        return fake()


@unittest.skipUnless(REAL, "set TEST_VALKEY_URL to run against a real Valkey")
class WithRealValkey(CacheContract, unittest.TestCase):
    def make(self):
        return real()


class WhenValkeyIsMissingOrDown(unittest.TestCase):
    def test_no_url_means_no_cache(self):
        cache = Cache.from_url("")
        self.assertFalse(cache.enabled)
        self.assertEqual(cache.caller("key", "s", lambda: {"v": 1}), {"v": 1})
        self.assertIsNone(cache.allow("b", "p", 1, 60))
        self.assertFalse(cache.webhook_seen("e"))

    def test_a_dead_server_never_raises_and_is_not_asked_again_at_once(self):
        now = [0.0]
        client = mock.Mock(wraps=Broken())
        client.get.side_effect = ConnectionError("redis://user:password@host refused")
        cache = Cache(client, clock=lambda: now[0])
        with self.assertLogs("ekido_platform.cache", "WARNING") as logs:
            self.assertEqual(cache.caller("key", "s", lambda: {"v": 1}), {"v": 1})
        self.assertNotIn("password", "\n".join(logs.output))
        for _ in range(5):
            cache.caller("key", "s", lambda: {"v": 1})
        self.assertEqual(client.get.call_count, 1)
        now[0] = cache_module.PAUSE_AFTER_ERROR + 1
        cache.caller("key", "s", lambda: {"v": 1})
        self.assertEqual(client.get.call_count, 2)

    def test_the_limits_still_hold_in_this_process(self):
        limiter = SharedRateLimit(Cache(Broken()), limit=2, window=60)
        self.assertEqual([limiter.allow("p") for _ in range(3)], [True, True, False])
        breaker = SharedRefusalBreaker(Cache(Broken()), limit=2, window=60)
        breaker.record("p")
        breaker.record("p")
        self.assertTrue(breaker.open("p"))


def build(client):
    """The whole app, with its cache replaced by one on `client`."""
    settings = Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE],
                        session_cookie_secure=False, valkey_url="redis://unused", razorpay_key_id="rzp_test_x",
                        razorpay_key_secret="s", razorpay_webhook_secret="hook-secret")
    with mock.patch.object(Cache, "from_url", classmethod(lambda cls, url: cls(client))):
        return create_app(settings)


class ThroughTheApp:
    """Sign in, keys, revocation, erasure and webhooks, run with a working cache and with a dead one."""

    def make(self):
        raise NotImplementedError

    def setUp(self):
        self.app = build(self.make())
        self.me = TestClient(self.app)

    def sign_in(self, client, email="asha@example.com"):
        with self.assertLogs("ekido_platform.v1_accounts", level="WARNING") as logs:
            self.assertEqual(client.post("/v1/auth/email/start", json={"email": email}).status_code, 202)
        token = re.search(r"token=([\w-]+)", logs.output[-1]).group(1)
        self.assertEqual(client.post("/v1/auth/email/verify", json={"token": token}).status_code, 200)

    def team_and_key(self):
        self.sign_in(self.me)
        org = self.me.post("/v1/orgs", json={"name": "Clinic"}, headers={"origin": SITE}).json()["org_id"]
        made = self.me.post(f"/v1/orgs/{org}/api-keys", json={"name": "k", "mode": "test"}).json()
        return org, made

    def test_a_revoked_key_stops_working_at_once(self):
        org, made = self.team_and_key()
        keyed, auth = TestClient(self.app), {"authorization": f"Bearer {made['secret']}"}
        for _ in range(2):
            self.assertEqual(keyed.get(f"/v1/orgs/{org}", headers=auth).status_code, 200)
        self.assertEqual(self.me.delete(f"/v1/orgs/{org}/api-keys/{made['key_id']}").status_code, 204)
        self.assertEqual(keyed.get(f"/v1/orgs/{org}", headers=auth).status_code, 401)

    def test_a_signed_out_cookie_stops_working_at_once(self):
        self.sign_in(self.me)
        cookie = self.me.cookies.get("ekido_session")
        self.assertEqual(self.me.get("/v1/me").status_code, 200)
        self.assertLess(self.me.post("/v1/auth/logout").status_code, 300)
        thief = TestClient(self.app)
        self.assertEqual(thief.get("/v1/me", headers={"cookie": f"ekido_session={cookie}"}).status_code, 401)

    def test_delete_my_data_stops_the_session_and_the_key_at_once(self):
        org, made = self.team_and_key()
        keyed, auth = TestClient(self.app), {"authorization": f"Bearer {made['secret']}"}
        cookie = self.me.cookies.get("ekido_session")
        self.assertEqual(keyed.get(f"/v1/orgs/{org}", headers=auth).status_code, 200)
        self.assertEqual(self.me.get("/v1/me").status_code, 200)
        self.assertEqual(self.me.delete("/v1/me").status_code, 202)
        self.assertEqual(keyed.get(f"/v1/orgs/{org}", headers=auth).status_code, 401)
        self.assertEqual(TestClient(self.app).get("/v1/me", headers={"cookie": f"ekido_session={cookie}"}).status_code, 401)

    def webhook(self, body, signature=None, event_id="evt_1"):
        signature = signature or hmac.new(b"hook-secret", body, hashlib.sha256).hexdigest()
        return self.me.post("/v1/payments/razorpay/webhook", content=body,
                            headers={"X-Razorpay-Signature": signature, "X-Razorpay-Event-Id": event_id})

    def test_a_repeated_webhook_is_a_duplicate_and_a_forged_repeat_is_still_refused(self):
        body = json.dumps({"event": "order.paid", "payload": {}}).encode()
        self.assertEqual(self.webhook(body).json()["outcome"], "ignored")
        self.assertEqual(self.webhook(body).json()["outcome"], "duplicate")
        self.assertEqual(self.webhook(body, signature="0" * 64).status_code, 401)


class AppWithCache(ThroughTheApp, unittest.TestCase):
    def make(self):
        return fake()


class AppWithDeadCache(ThroughTheApp, unittest.TestCase):
    def make(self):
        return Broken()


if __name__ == "__main__":
    unittest.main()
