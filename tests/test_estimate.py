"""A visitor who has not signed in can ask what a job would roughly cost: GET /v1/recipes/{recipe_id}/estimate."""
import unittest

from fastapi.testclient import TestClient

from ekido_core.quotes import make_quote
from ekido_platform.main import Settings, create_app
from ekido_platform.v1_datasets import ESTIMATE_FIELDS, EstimateLimiter, bucket_of, client_address

SITE = "https://ekido.example"
RECIPE = "review-sentiment"


def build():
    return create_app(Settings(database_url="sqlite://", _env_file=None, email_sender="log", site_url=SITE, cors_origins=[SITE],
                               session_cookie_secure=False))


class Estimate(unittest.TestCase):
    def setUp(self):
        # Caddy forwards from loopback; the test client would otherwise appear as the host "testclient", whose header is rightly ignored.
        self.client = TestClient(build(), client=("127.0.0.1", 50000))

    def get(self, recipe=RECIPE, train=500, test=100, **kw):
        return self.client.get(f"/v1/recipes/{recipe}/estimate", params={"train_rows": train, "test_rows": test}, **kw)

    def test_needs_no_sign_in_and_matches_the_real_quote(self):
        r = self.get()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        real = make_quote(recipe_id=RECIPE, base_model="Qwen/Qwen2.5-0.5B-Instruct", train_rows=500, test_rows=100, dataset_sha256="a" * 64)
        self.assertEqual(body["total_paise"], real["total_paise"])
        self.assertEqual((body["training_paise"], body["gst_paise"], body["eta_minutes"]), (real["training_paise"], real["gst_paise"], real["eta_minutes"]))
        self.assertTrue(body["estimate"])
        self.assertTrue(body["prices_provisional"])
        self.assertIn("Full refund", body["refund_policy"])
        self.assertEqual(body["quality_terms"]["floor"], real["quality_terms"]["floor"])

    def test_the_sample_prices_to_rs_116_82(self):
        self.assertEqual(self.get().json()["total_paise"], 11682)

    def test_shows_only_what_a_customer_sees_on_a_quote_not_the_fee_split(self):
        body = self.get().json()
        for private in ("ekido_fee_paise", "provider_budget_paise", "terms_hash", "dataset_sha256", "expires_at", "base_model", "plan"):
            self.assertNotIn(private, body)
        self.assertEqual(set(body) - set(ESTIMATE_FIELDS), {"recipe_id", "train_rows", "test_rows", "estimate", "distill", "generated_rows"})

    def test_stores_nothing_and_signs_nothing(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session
        from ekido_platform.v1_datasets import Quote

        self.get()
        self.assertNotIn("terms_hash", self.get().json())
        with Session(self.client.app.state.engine) as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Quote)), 0)

    def test_a_direct_caller_cannot_choose_its_own_bucket(self):
        direct = TestClient(build())  # peer "testclient", not loopback: the header is ignored
        for _ in range(60):
            direct.get(f"/v1/recipes/{RECIPE}/estimate", params={"train_rows": 500, "test_rows": 100}, headers={"x-forwarded-for": "198.51.100.1"})
        again = direct.get(f"/v1/recipes/{RECIPE}/estimate", params={"train_rows": 500, "test_rows": 100}, headers={"x-forwarded-for": "198.51.100.2"})
        self.assertEqual(again.status_code, 429)

    def test_an_unknown_recipe_is_404(self):
        self.assertEqual(self.get(recipe="no-such-recipe").status_code, 404)

    def test_too_few_test_rows_is_refused_with_a_reason(self):
        r = self.get(test=5)
        self.assertEqual(r.status_code, 422)  # the framework's bounds check: at least 20 test rows

    def test_rows_outside_the_bounds_are_refused(self):
        self.assertEqual(self.get(train=0).status_code, 422)
        self.assertEqual(self.get(train=20_001).status_code, 422)
        self.assertEqual(self.get(test=5_001).status_code, 422)
        self.assertEqual(self.client.get(f"/v1/recipes/{RECIPE}/estimate").status_code, 422)

    def test_each_address_has_its_own_limit(self):
        for _ in range(60):
            self.assertEqual(self.get(headers={"x-forwarded-for": "203.0.113.7"}).status_code, 200)
        limited = self.get(headers={"x-forwarded-for": "203.0.113.7"})
        self.assertEqual((limited.status_code, limited.json()["error"]), (429, "RATE_LIMITED"))
        self.assertEqual(self.get(headers={"x-forwarded-for": "203.0.113.8"}).status_code, 200)  # another visitor is not affected


class Distill(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(build(), client=("127.0.0.1", 50000))

    def get(self, **extra):
        return self.client.get(f"/v1/recipes/{RECIPE}/estimate", params={"train_rows": 100, "test_rows": 40, **extra}).json()

    def test_by_default_there_are_no_generated_examples(self):
        body = self.get()
        self.assertEqual((body["distill"], body["generated_rows"]), (False, 0))

    def test_asking_for_generated_examples_is_priced_like_the_real_quote(self):
        plain, topped_up = self.get(), self.get(distill=True)
        real = make_quote(recipe_id=RECIPE, base_model="Qwen/Qwen2.5-0.5B-Instruct", train_rows=100, test_rows=40, dataset_sha256="a" * 64, distill=True)
        self.assertTrue(topped_up["distill"])
        self.assertGreater(topped_up["generated_rows"], 0)
        self.assertGreater(topped_up["total_paise"], plain["total_paise"])
        self.assertEqual((topped_up["total_paise"], topped_up["generated_rows"]), (real["total_paise"], real["plan"]["generated_rows"]))


class Buckets(unittest.TestCase):
    def test_an_ipv6_address_counts_as_its_slash_64(self):
        self.assertEqual(bucket_of("2001:db8:1:2:aaaa:bbbb:cccc:dddd"), bucket_of("2001:db8:1:2::1"))
        self.assertNotEqual(bucket_of("2001:db8:1:2::1"), bucket_of("2001:db8:1:3::1"))
        self.assertEqual(bucket_of("203.0.113.7"), "203.0.113.7")
        self.assertEqual(bucket_of("not an address"), "not an address")

    def test_rotating_inside_one_slash_64_does_not_give_a_fresh_allowance(self):
        limiter = EstimateLimiter(limit=3, window=600)
        results = [limiter.allow(bucket_of(f"2001:db8:1:2::{i:x}")) for i in range(1, 6)]
        self.assertEqual(results, [True, True, True, False, False])


class Limiter(unittest.TestCase):
    def test_a_full_table_never_refuses_a_new_visitor(self):
        limiter = EstimateLimiter(limit=2, window=600, max_buckets=3)
        for n in range(3):
            self.assertTrue(limiter.allow(f"10.0.0.{n}", now=1.0))
        self.assertTrue(limiter.allow("198.51.100.9", now=2.0))  # the table was full of recent addresses: still served
        self.assertLessEqual(len(limiter.seen), 3)

    def test_the_oldest_address_is_the_one_dropped(self):
        limiter = EstimateLimiter(limit=2, window=600, max_buckets=2)
        limiter.allow("a", now=1.0)
        limiter.allow("b", now=2.0)
        limiter.allow("a", now=3.0)  # a is now the most recent
        limiter.allow("c", now=4.0)  # drops b
        self.assertEqual(list(limiter.seen), ["a", "c"])

    def test_the_limit_and_the_window_still_work(self):
        limiter = EstimateLimiter(limit=2, window=10)
        self.assertEqual([limiter.allow("x", now=t) for t in (0, 1, 2)], [True, True, False])
        self.assertTrue(limiter.allow("x", now=11.5))


class ClientAddress(unittest.TestCase):
    class Req:
        def __init__(self, peer, forwarded=None):
            self.client = type("C", (), {"host": peer})() if peer else None
            self.headers = {"x-forwarded-for": forwarded} if forwarded else {}

    def test_believes_the_header_only_from_loopback(self):
        self.assertEqual(client_address(self.Req("127.0.0.1", "198.51.100.4")), "198.51.100.4")
        self.assertEqual(client_address(self.Req("::1", "198.51.100.4")), "198.51.100.4")
        self.assertEqual(client_address(self.Req("203.0.113.9", "198.51.100.4")), "203.0.113.9")  # a direct caller can't pick its bucket

    def test_takes_the_address_the_proxy_appended(self):
        self.assertEqual(client_address(self.Req("127.0.0.1", "10.0.0.1, 198.51.100.4")), "198.51.100.4")

    def test_without_a_header_or_peer(self):
        self.assertEqual(client_address(self.Req("127.0.0.1")), "127.0.0.1")
        self.assertEqual(client_address(self.Req(None)), "unknown")


if __name__ == "__main__":
    unittest.main()
