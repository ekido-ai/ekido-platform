"""C24: plans, the free tier, tiers and limits, against a fake Razorpay. Cases follow the A9 matrix: subscribe to each plan,
renewal, failed renewal, cancel, the free-tier limit, upgrade."""

import hashlib
import hmac
import json
import threading
import unittest
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ekido_core.quotes import GST_BPS, LLM_BASES, gst

from ekido_platform import plans
from ekido_platform.main import Settings, create_app
from ekido_platform.payments import PaymentError, RazorpayRail
from ekido_platform.plans import (ACTIVE, CANCELLED, HALTED, PAID, PAST_DUE, PENDING, PLANS, PlanLimit, PlansService, PlanUsage, TeamPlan,
                                  tier_for, tier_price_paise)

ORG = "org-1"
START = datetime(2026, 11, 1, tzinfo=timezone.utc)
SECRET = "wh-secret"
SMALL_MODEL, BIG_MODEL = LLM_BASES[0], LLM_BASES[1]


class FakeSubscriptions:
    def __init__(self):
        self.subs, self.calls, self.fail = {}, [], False

    def create_subscription(self, *, plan_id, total_count, notes):
        self.calls.append(("create", plan_id))
        sid = f"sub_{len(self.subs) + 1:03d}"
        self.subs[sid] = {"id": sid, "plan_id": plan_id, "status": "created", "current_start": None, "current_end": None, "short_url": f"https://rzp.io/{sid}"}
        return dict(self.subs[sid])

    def fetch_subscription(self, subscription_id):
        self.calls.append(("fetch", subscription_id))
        if self.fail:
            raise PaymentError("GATEWAY_UNREACHABLE", "unreachable", 502)
        return dict(self.subs[subscription_id])

    def cancel_subscription(self, subscription_id, *, at_cycle_end):
        self.calls.append(("cancel", subscription_id, at_cycle_end))
        return dict(self.subs[subscription_id])

    def update_subscription(self, subscription_id, *, plan_id):
        self.calls.append(("update", subscription_id, plan_id))
        self.subs[subscription_id]["plan_id"] = plan_id
        return dict(self.subs[subscription_id])

    def charge(self, sid, start, days=30):
        """Razorpay charged the customer: the subscription is active for a new period."""
        self.subs[sid].update(status="active", current_start=int(start.timestamp()), current_end=int((start + timedelta(days=days)).timestamp()))


class Base(unittest.TestCase):
    def setUp(self):
        self.app = create_app(Settings(database_url="sqlite://", _env_file=None))
        self.session = Session(self.app.state.engine)
        self.addCleanup(self.session.close)
        self.gateway = FakeSubscriptions()
        self.clock = START
        self.service = PlansService(self.gateway, {"starter": "plan_rzp_starter", "business": "plan_rzp_business"}, now=lambda: self.clock)

    def subscribe(self, plan="starter", org=ORG, start=START):
        out = self.service.subscribe(self.session, org, plan)
        self.gateway.charge(out["subscription_id"], start)
        self.assertEqual(self.service.sync_subscription(self.session, out["subscription_id"]), "synced_active")
        return out["subscription_id"]

    def row(self, org=ORG):
        return self.session.scalar(select(TeamPlan).where(TeamPlan.org_id == org))

    def plan(self, org=ORG):
        return self.service.effective_plan(self.session, org).id


class CatalogueTests(unittest.TestCase):
    def test_three_plans_with_gst_on_top_and_provisional_prices(self):
        self.assertEqual(set(PLANS), {"free", "starter", "business"})
        self.assertTrue(plans.PRICES_PROVISIONAL)
        self.assertEqual(PLANS["free"].monthly_price_paise, 0)
        for plan_id in PAID:
            plan = PLANS[plan_id]
            self.assertEqual(plan.monthly_total_paise, plan.monthly_price_paise + gst(plan.monthly_price_paise))
            self.assertEqual(GST_BPS, 1800)

    def test_each_paid_plan_includes_more_than_the_one_below(self):
        free, starter, business = (PLANS[p] for p in ("free", "starter", "business"))
        self.assertLess(free.jobs_included, starter.jobs_included)
        self.assertLess(starter.jobs_included, business.jobs_included)
        self.assertLess(free.requests_included, starter.requests_included)
        self.assertLess(starter.requests_included, business.requests_included)
        self.assertEqual(free.tiers, ("small",))
        self.assertIn("large", business.tiers)
        self.assertNotIn("large", starter.tiers)
        self.assertTrue(business.priority and not starter.priority)
        self.assertLess(starter.seats, business.seats)

    def test_job_tiers_by_rows_and_model(self):
        for rows, model, tier in ((1, SMALL_MODEL, "small"), (500, SMALL_MODEL, "small"), (501, SMALL_MODEL, "medium"), (2000, SMALL_MODEL, "medium"),
                                  (2001, SMALL_MODEL, "large"), (1, BIG_MODEL, "medium"), (500, BIG_MODEL, "medium"), (501, BIG_MODEL, "large")):
            with self.subTest(rows=rows, model=model):
                self.assertEqual(tier_for(rows, model), tier)
        for bad in ((0, SMALL_MODEL), (-1, SMALL_MODEL), (5.5, SMALL_MODEL), (10, "gpt-x")):
            with self.subTest(bad=bad), self.assertRaises(PaymentError):
                tier_for(*bad)

    def test_fixed_job_prices_show_gst_separately_and_rise_with_the_tier(self):
        prices = [tier_price_paise(t) for t in plans.TIERS]
        for price in prices:
            self.assertEqual(price["total_paise"], price["taxable_paise"] + price["gst_paise"])
            self.assertEqual(price["gst_paise"], gst(price["taxable_paise"]))
            self.assertTrue(price["prices_provisional"])
        self.assertEqual([p["total_paise"] for p in prices], sorted(p["total_paise"] for p in prices))


class SubscribeTests(Base):
    def test_a_team_with_no_subscription_is_on_free(self):
        self.assertEqual(self.plan(), "free")
        self.assertIsNone(self.service.status(self.session, ORG)["subscription_status"])

    def test_subscribing_to_each_plan_takes_effect_only_after_razorpay_confirms_the_charge(self):
        for plan in PAID:
            with self.subTest(plan=plan):
                org = f"org-{plan}"
                out = self.service.subscribe(self.session, org, plan)
                self.assertEqual((out["status"], out["plan"]), (PENDING, plan))
                self.assertTrue(out["checkout_url"].startswith("https://"))
                self.assertEqual(self.plan(org), "free")  # not paid yet
                self.gateway.charge(out["subscription_id"], START)
                self.service.sync_subscription(self.session, out["subscription_id"])
                self.assertEqual(self.plan(org), plan)
                self.assertEqual(self.gateway.subs[out["subscription_id"]]["plan_id"], f"plan_rzp_{plan}")

    def test_bad_plans_and_double_subscriptions_are_refused(self):
        for bad in ("free", "gold", "", None):
            with self.subTest(bad=bad), self.assertRaises(PaymentError) as caught:
                self.service.subscribe(self.session, ORG, bad)
            self.assertEqual(caught.exception.code, "BAD_PLAN")
        self.subscribe()
        with self.assertRaises(PaymentError) as caught:
            self.service.subscribe(self.session, ORG, "business")
        self.assertEqual(caught.exception.code, "ALREADY_SUBSCRIBED")
        self.assertEqual(sum(1 for c in self.gateway.calls if c[0] == "create"), 1)


class RenewalTests(Base):
    def test_a_renewal_starts_a_new_period_with_fresh_usage(self):
        sid = self.subscribe()
        for _ in range(PLANS["starter"].jobs_included):
            self.service.record_job(self.session, ORG, "small")
        self.assertFalse(self.service.job_allowance(self.session, ORG, "small").included)
        self.clock = START + timedelta(days=30, hours=1)  # the paid period has ended; Razorpay renews
        self.assertEqual(self.plan(), "free")  # until the renewal is confirmed, no benefits
        self.gateway.charge(sid, START + timedelta(days=30))
        self.assertEqual(self.service.sync_subscription(self.session, sid), "synced_active")
        self.assertEqual(self.plan(), "starter")
        self.assertTrue(self.service.job_allowance(self.session, ORG, "small").included)  # a new period, a new allowance

    def test_a_failed_renewal_drops_the_team_to_free_and_a_later_payment_restores_it(self):
        sid = self.subscribe()
        self.gateway.subs[sid]["status"] = "pending"  # Razorpay could not charge the card and will retry
        self.assertEqual(self.service.sync_subscription(self.session, sid), "synced_past_due")
        self.assertEqual((self.row().status, self.plan()), (PAST_DUE, "free"))
        self.gateway.subs[sid]["status"] = "halted"  # retries ran out
        self.assertEqual(self.service.sync_subscription(self.session, sid), "synced_halted")
        self.assertEqual((self.row().status, self.plan()), (HALTED, "free"))
        self.gateway.charge(sid, START + timedelta(days=3))  # the customer fixed the card
        self.service.sync_subscription(self.session, sid)
        self.assertEqual(self.plan(), "starter")

    def test_a_webhook_cannot_grant_a_plan_by_lying_the_state_comes_from_razorpay(self):
        out = self.service.subscribe(self.session, ORG, "business")
        # Razorpay still says "created" (no charge). Syncing from it keeps the team on free, whatever a webhook claimed.
        self.assertEqual(self.service.sync_subscription(self.session, out["subscription_id"]), "synced_pending")
        self.assertEqual(self.plan(), "free")

    def test_a_razorpay_answer_for_a_different_subscription_changes_nothing(self):
        sid = self.subscribe()
        self.gateway.fetch_subscription = lambda subscription_id: {"id": "sub_other", "plan_id": "plan_rzp_business", "status": "halted"}
        self.assertEqual(self.service.sync_subscription(self.session, sid), "needs_review")
        self.assertEqual((self.row().status, self.plan()), (ACTIVE, "starter"))

    def test_unknown_subscriptions_and_unknown_razorpay_plans_change_nothing(self):
        self.assertEqual(self.service.sync_subscription(self.session, "sub_nope"), "unknown_subscription")
        sid = self.subscribe()
        self.gateway.subs[sid]["plan_id"] = "plan_rzp_mystery"
        self.assertEqual(self.service.sync_subscription(self.session, sid), "needs_review")
        self.assertEqual(self.plan(), "starter")


class CancelAndUpgradeTests(Base):
    def test_cancel_keeps_the_plan_until_the_period_ends_then_drops_to_free(self):
        sid = self.subscribe()
        out = self.service.cancel(self.session, ORG)
        self.assertTrue(out["cancel_at_period_end"])
        self.assertIn(("cancel", sid, True), self.gateway.calls)
        self.assertEqual(self.plan(), "starter")  # paid for, so still theirs
        self.service.cancel(self.session, ORG)  # again: nothing more is sent
        self.assertEqual(sum(1 for c in self.gateway.calls if c[0] == "cancel"), 1)
        self.gateway.subs[sid]["status"] = "cancelled"  # Razorpay ends it at the period's end
        self.service.sync_subscription(self.session, sid)
        self.assertEqual((self.row().status, self.plan()), (CANCELLED, "free"))

    def test_cancelling_without_a_subscription_is_refused(self):
        with self.assertRaises(PaymentError) as caught:
            self.service.cancel(self.session, ORG)
        self.assertEqual(caught.exception.code, "NO_SUBSCRIPTION")

    def test_a_cancelled_team_can_subscribe_again(self):
        sid = self.subscribe()
        self.gateway.subs[sid]["status"] = "cancelled"
        self.service.sync_subscription(self.session, sid)
        new = self.subscribe("business", start=START + timedelta(days=40))
        self.assertNotEqual(new, sid)
        self.assertEqual(self.plan(), "business")

    def test_upgrading_switches_the_plan_now_and_keeps_this_periods_usage(self):
        self.subscribe("starter")
        self.service.record_job(self.session, ORG, "small")
        self.service.consume_requests(self.session, ORG, 100)
        out = self.service.change_plan(self.session, ORG, "business")
        self.assertEqual(out["plan"], "business")
        self.assertTrue(self.service.job_allowance(self.session, ORG, "large").included)  # business adds the large size
        used = {u.kind: u.used for u in self.session.scalars(select(PlanUsage))}
        self.assertEqual(used, {"jobs": 1, "requests": 100})

    def test_changing_plan_needs_an_active_subscription_and_a_paid_target(self):
        for target in ("business", "free"):
            with self.subTest(target=target), self.assertRaises(PaymentError):
                self.service.change_plan(self.session, ORG, target)
        self.subscribe("starter")
        self.assertEqual(self.service.change_plan(self.session, ORG, "starter")["plan"], "starter")  # same plan: nothing sent
        self.assertEqual(sum(1 for c in self.gateway.calls if c[0] == "update"), 0)


class LimitTests(Base):
    def test_the_free_tier_includes_one_small_job_ever_and_then_asks_for_payment(self):
        self.assertTrue(self.service.job_allowance(self.session, ORG, "small").included)
        self.service.record_job(self.session, ORG, "small")
        again = self.service.job_allowance(self.session, ORG, "small")
        self.assertFalse(again.included)  # pay the fixed price instead; paying per job is always available
        with self.assertRaises(PlanLimit) as caught:
            self.service.record_job(self.session, ORG, "small")
        self.assertEqual((caught.exception.code, caught.exception.status, caught.exception.upgrade_to), ("JOB_NOT_INCLUDED", 402, "starter"))
        self.clock = START + timedelta(days=90)  # "one per team" is for life, not per month
        self.assertFalse(self.service.job_allowance(self.session, ORG, "small").included)

    def test_free_does_not_include_medium_or_large_jobs(self):
        for tier in ("medium", "large"):
            self.assertFalse(self.service.job_allowance(self.session, ORG, tier).included)
        with self.assertRaises(PlanLimit) as caught:
            self.service.record_job(self.session, ORG, "medium")
        self.assertEqual(caught.exception.upgrade_to, "starter")
        with self.assertRaises(PlanLimit) as caught:
            self.service.record_job(self.session, ORG, "large")
        self.assertEqual(caught.exception.upgrade_to, "business")
        with self.assertRaises(PaymentError):
            self.service.job_allowance(self.session, ORG, "huge")

    def test_the_free_tier_hosted_request_limit_is_enforced_and_resets_each_month(self):
        free = PLANS["free"].requests_included
        self.assertEqual(self.service.consume_requests(self.session, ORG, free - 1), 1)
        self.assertEqual(self.service.consume_requests(self.session, ORG), 0)
        with self.assertRaises(PlanLimit) as caught:
            self.service.consume_requests(self.session, ORG)
        self.assertEqual((caught.exception.code, caught.exception.status, caught.exception.upgrade_to), ("REQUEST_LIMIT", 429, "starter"))
        self.clock = datetime(2026, 12, 1, tzinfo=timezone.utc)
        self.assertEqual(self.service.consume_requests(self.session, ORG), free - 1)  # a new calendar month

    def test_a_batch_that_does_not_fit_is_refused_whole_and_counts_nothing(self):
        free = PLANS["free"].requests_included
        self.service.consume_requests(self.session, ORG, free - 5)
        with self.assertRaises(PlanLimit):
            self.service.consume_requests(self.session, ORG, 10)
        self.assertEqual(self.service.consume_requests(self.session, ORG, 5), 0)  # the 5 that fit still can
        for bad in (0, -1, 2.5, "3"):
            with self.subTest(bad=bad), self.assertRaises(PaymentError):
                self.service.consume_requests(self.session, ORG, bad)

    def test_a_paid_plan_counts_jobs_and_requests_per_billing_period(self):
        self.subscribe("starter")
        for _ in range(PLANS["starter"].jobs_included):
            self.service.record_job(self.session, ORG, "medium")
        self.assertFalse(self.service.job_allowance(self.session, ORG, "small").included)
        with self.assertRaises(PlanLimit) as caught:
            self.service.record_job(self.session, ORG, "small")
        self.assertEqual(caught.exception.upgrade_to, "business")
        self.assertEqual(self.service.consume_requests(self.session, ORG, PLANS["starter"].requests_included), 0)
        with self.assertRaises(PlanLimit):
            self.service.consume_requests(self.session, ORG)

    def test_teams_do_not_share_usage(self):
        self.service.record_job(self.session, "org-a", "small")
        self.assertTrue(self.service.job_allowance(self.session, "org-b", "small").included)

    def test_two_requests_at_the_limit_cannot_both_pass(self):
        # A file database, so each thread has its own connection (the in-memory test database shares one).
        import os, tempfile
        from sqlalchemy import create_engine
        from ekido_platform.models import Base as Tables

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        engine = create_engine(f"sqlite:///{os.path.join(folder.name, 'usage.db')}", connect_args={"timeout": 15})
        self.addCleanup(engine.dispose)
        Tables.metadata.create_all(engine)
        free = PLANS["free"].requests_included
        with Session(engine) as session:
            self.service.consume_requests(session, ORG, free - 1)  # exactly one request is left
        results, barrier = [], threading.Barrier(4)

        def attempt():
            with Session(engine) as session:
                barrier.wait()
                try:
                    self.service.consume_requests(session, ORG)
                    results.append("ok")
                except PlanLimit:
                    results.append("refused")

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(sorted(results), ["ok", "refused", "refused", "refused"])
        with Session(engine) as session:
            self.assertEqual(session.scalar(select(PlanUsage.used)), free)

class WiringTests(unittest.TestCase):
    def test_the_app_builds_the_plans_service_only_when_payments_and_both_plan_ids_are_set(self):
        common = dict(database_url="sqlite://", _env_file=None, razorpay_key_id="rzp_test_abc", razorpay_key_secret="s", razorpay_webhook_secret="w")
        self.assertIsNone(create_app(Settings(**common)).state.plans)
        self.assertIsNone(create_app(Settings(**common, razorpay_plan_starter="plan_s")).state.plans)
        app = create_app(Settings(**common, razorpay_plan_starter="plan_s", razorpay_plan_business="plan_b"))
        self.assertEqual(app.state.plans.razorpay_plan_ids, {"starter": "plan_s", "business": "plan_b"})
        self.assertEqual(app.state.payment_rail.on_subscription_event, app.state.plans.sync_subscription)
        self.assertIsNone(create_app(Settings(database_url="sqlite://", _env_file=None)).state.plans)


class WebhookTests(Base):
    def setUp(self):
        super().setUp()
        self.rail = RazorpayRail(self.gateway, SECRET, on_subscription_event=self.service.sync_subscription)

    def send(self, event, sid, event_id):
        raw = json.dumps({"event": event, "payload": {"subscription": {"entity": {"id": sid}}}}, separators=(",", ":")).encode()
        return self.rail.handle_webhook(self.session, raw, hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest(), event_id)

    def test_subscription_events_sync_from_razorpay_and_repeats_are_ignored(self):
        out = self.service.subscribe(self.session, ORG, "starter")
        sid = out["subscription_id"]
        self.gateway.charge(sid, START)
        self.assertEqual(self.send("subscription.activated", sid, "evt_1"), "synced_active")
        self.assertEqual(self.send("subscription.activated", sid, "evt_1"), "duplicate")
        self.assertEqual(self.plan(), "starter")
        self.gateway.subs[sid]["status"] = "halted"
        self.assertEqual(self.send("subscription.halted", sid, "evt_2"), "synced_halted")
        self.assertEqual(self.plan(), "free")

    def test_if_razorpay_cannot_be_reached_the_event_is_retried(self):
        sid = self.service.subscribe(self.session, ORG, "starter")["subscription_id"]
        self.gateway.charge(sid, START)
        self.gateway.fail = True
        with self.assertRaises(PaymentError):
            self.send("subscription.charged", sid, "evt_3")
        self.gateway.fail = False
        self.assertEqual(self.send("subscription.charged", sid, "evt_3"), "synced_active")

    def test_without_a_plans_service_subscription_events_are_ignored(self):
        bare = RazorpayRail(self.gateway, SECRET)
        raw = b'{"event":"subscription.charged","payload":{"subscription":{"entity":{"id":"sub_001"}}}}'
        self.assertEqual(bare.handle_webhook(self.session, raw, hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest(), "e"), "ignored")


if __name__ == "__main__":
    unittest.main()
