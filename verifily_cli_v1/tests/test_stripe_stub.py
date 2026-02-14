"""Tests for Stripe stub integration -- checkout, webhooks, subscriptions, gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.billing.stripe import MockStripeClient, StripeConfig
from verifily_cli_v1.core.billing.subscriptions import (
    SubscriptionRecord,
    SubscriptionStatus,
    SubscriptionsStore,
    subscriptions_store,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRILL_DIR = REPO_ROOT / "examples" / "customer_drill"


def _make_client(monkeypatch, **kwargs):
    """Create a TestClient with given settings overrides."""
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_STRIPE_ENABLED", raising=False)
    monkeypatch.delenv("VERIFILY_BILLING_ENFORCE", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("STRIPE_PRICE_ID_PRO", raising=False)
    settings = load_settings(**kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    # Install mock stripe client
    app.state._stripe_client = MockStripeClient()
    return TestClient(app)


def _make_stripe_client(monkeypatch, **kwargs):
    """Create TestClient with Stripe enabled + mock env vars."""
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_mock_key")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_mock_secret")
    monkeypatch.setenv("STRIPE_PRICE_ID_PRO", "price_mock_pro")
    return _make_client(monkeypatch, stripe_enabled=True, enable_billing=True, **kwargs)


def _webhook_event(event_type: str, org_id: str = "default", project_id: str = "default",
                   customer: str = "cus_123", subscription: str = "sub_123") -> bytes:
    """Build a mock webhook event payload."""
    return json.dumps({
        "type": event_type,
        "data": {
            "object": {
                "customer": customer,
                "subscription": subscription,
                "metadata": {"org_id": org_id, "project_id": project_id},
            }
        },
    }).encode()


# ── StripeConfig Tests ─────────────────────────────────────────


class TestStripeConfig:
    def test_from_env_disabled(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_STRIPE_ENABLED", raising=False)
        cfg = StripeConfig.from_env()
        assert cfg.enabled is False

    def test_from_env_enabled(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_STRIPE_ENABLED", "1")
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
        monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
        monkeypatch.setenv("STRIPE_PRICE_ID_PRO", "price_test")
        cfg = StripeConfig.from_env()
        assert cfg.enabled is True
        assert cfg.secret_key == "sk_test"

    def test_validate_missing_keys(self, monkeypatch):
        monkeypatch.setenv("VERIFILY_STRIPE_ENABLED", "1")
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("STRIPE_PRICE_ID_PRO", raising=False)
        cfg = StripeConfig.from_env()
        err = cfg.validate()
        assert err is not None
        assert "STRIPE_SECRET_KEY" in err

    def test_validate_ok(self):
        cfg = StripeConfig(
            enabled=True,
            secret_key="sk_test",
            webhook_secret="whsec_test",
            price_id_pro="price_test",
        )
        assert cfg.validate() is None


# ── MockStripeClient Tests ─────────────────────────────────────


class TestMockStripeClient:
    def test_create_customer(self):
        client = MockStripeClient()
        cid = client.create_customer(name="test")
        assert cid.startswith("cus_mock_")

    def test_create_checkout_session(self):
        client = MockStripeClient()
        session = client.create_checkout_session(
            price_id="price_test",
            success_url="http://ok",
            cancel_url="http://cancel",
        )
        assert "url" in session
        assert session["url"].startswith("https://checkout.stripe.com/mock/")

    def test_construct_event_valid(self):
        client = MockStripeClient()
        payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {"customer": "cus_1"}}}).encode()
        event = client.construct_event(payload, "valid_sig")
        assert event["type"] == "checkout.session.completed"

    def test_construct_event_invalid_sig(self):
        client = MockStripeClient()
        with pytest.raises(ValueError, match="Invalid signature"):
            client.construct_event(b"{}", "invalid")


# ── SubscriptionsStore Tests ───────────────────────────────────


class TestSubscriptionsStore:
    def test_set_and_get(self):
        store = SubscriptionsStore()
        store.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.ACTIVE,
            stripe_customer_id="cus_1",
            plan="pro",
        )
        rec = store.get("org-1", "proj-1")
        assert rec is not None
        assert rec.status == SubscriptionStatus.ACTIVE
        assert rec.plan == "pro"

    def test_get_missing_returns_none(self):
        store = SubscriptionsStore()
        assert store.get("org-x", "proj-x") is None

    def test_require_active_true(self):
        store = SubscriptionsStore()
        store.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.ACTIVE,
            plan="pro",
        )
        assert store.require_active("org-1", "proj-1") is True

    def test_require_active_false_incomplete(self):
        store = SubscriptionsStore()
        store.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.INCOMPLETE,
        )
        assert store.require_active("org-1", "proj-1") is False

    def test_require_active_false_missing(self):
        store = SubscriptionsStore()
        assert store.require_active("org-x", "proj-x") is False

    def test_update_preserves_created_at(self):
        store = SubscriptionsStore()
        rec1 = store.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.INCOMPLETE,
            stripe_customer_id="cus_1",
        )
        rec2 = store.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.ACTIVE,
        )
        assert rec2.created_at == rec1.created_at
        assert rec2.updated_at >= rec1.updated_at
        assert rec2.stripe_customer_id == "cus_1"

    def test_reset_clears(self):
        store = SubscriptionsStore()
        store.set_status(org_id="org-1", project_id="proj-1", status=SubscriptionStatus.ACTIVE)
        store.reset()
        assert store.get("org-1", "proj-1") is None

    def test_persistence_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        store1 = SubscriptionsStore()
        store1.configure_persistence(path)
        store1.set_status(
            org_id="org-1", project_id="proj-1",
            status=SubscriptionStatus.ACTIVE,
            stripe_customer_id="cus_1",
            plan="pro",
        )

        store2 = SubscriptionsStore()
        store2.configure_persistence(path)
        rec = store2.get("org-1", "proj-1")
        assert rec is not None
        assert rec.status == SubscriptionStatus.ACTIVE
        assert rec.plan == "pro"

    def test_to_dict_roundtrip(self):
        rec = SubscriptionRecord(
            org_id="org-1", project_id="proj-1",
            stripe_customer_id="cus_1",
            status=SubscriptionStatus.ACTIVE,
            plan="pro",
            created_at=100.0,
            updated_at=200.0,
        )
        d = rec.to_dict()
        rec2 = SubscriptionRecord.from_dict(d)
        assert rec2.org_id == rec.org_id
        assert rec2.status == rec.status


# ── API Tests: Stripe Disabled ─────────────────────────────────


class TestStripeDisabled:
    def test_checkout_501_when_disabled(self, monkeypatch):
        client = _make_client(monkeypatch, stripe_enabled=False, enable_billing=True)
        resp = client.post("/v1/billing/checkout", json={"plan": "pro"})
        assert resp.status_code == 501
        body = resp.json()
        msg = body.get("error", {}).get("message", "") or body.get("detail", "")
        assert "not enabled" in msg.lower()

    def test_subscription_501_when_disabled(self, monkeypatch):
        client = _make_client(monkeypatch, stripe_enabled=False, enable_billing=True)
        resp = client.get("/v1/billing/subscription", params={"project_id": "default"})
        assert resp.status_code == 501

    def test_webhook_501_when_disabled(self, monkeypatch):
        client = _make_client(monkeypatch, stripe_enabled=False, enable_billing=True)
        resp = client.post("/v1/billing/webhook", content=b"{}")
        assert resp.status_code == 501


# ── API Tests: Stripe Enabled ──────────────────────────────────


class TestStripeEnabled:
    def test_checkout_creates_session(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        resp = client.post("/v1/billing/checkout", json={"plan": "pro"})
        assert resp.status_code == 200
        data = resp.json()
        assert "checkout_url" in data
        assert data["plan"] == "pro"
        assert data["stripe_customer_id"].startswith("cus_mock_")

    def test_checkout_invalid_plan(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        resp = client.post("/v1/billing/checkout", json={"plan": "enterprise"})
        assert resp.status_code == 422

    def test_webhook_checkout_completed_sets_active(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        payload = _webhook_event("checkout.session.completed", org_id="default", project_id="proj-1")
        resp = client.post(
            "/v1/billing/webhook",
            content=payload,
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )
        assert resp.status_code == 200

        sub_resp = client.get("/v1/billing/subscription", params={"project_id": "proj-1"})
        data = sub_resp.json()
        assert data["status"] == "active"
        assert data["project_id"] == "proj-1"

    def test_webhook_subscription_deleted_sets_canceled(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        # First activate
        client.post(
            "/v1/billing/webhook",
            content=_webhook_event("checkout.session.completed", org_id="default", project_id="proj-1"),
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )
        # Then cancel
        client.post(
            "/v1/billing/webhook",
            content=_webhook_event("customer.subscription.deleted", org_id="default", project_id="proj-1"),
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )

        sub_resp = client.get("/v1/billing/subscription", params={"project_id": "proj-1"})
        assert sub_resp.json()["status"] == "canceled"

    def test_webhook_invalid_signature_400(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        resp = client.post(
            "/v1/billing/webhook",
            content=b'{"type":"test"}',
            headers={"stripe-signature": "invalid", "content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_webhook_payment_failed_sets_incomplete(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        payload = _webhook_event("invoice.payment_failed", org_id="default", project_id="proj-1")
        resp = client.post(
            "/v1/billing/webhook",
            content=payload,
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )
        assert resp.status_code == 200

        sub_resp = client.get("/v1/billing/subscription", params={"project_id": "proj-1"})
        assert sub_resp.json()["status"] == "incomplete"

    def test_subscription_none_returns_free(self, monkeypatch):
        client = _make_stripe_client(monkeypatch)
        resp = client.get("/v1/billing/subscription", params={"project_id": "nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "none"
        assert data["plan"] == "free"


# ── Soft Gating Tests ──────────────────────────────────────────


class TestBillingGating:
    def test_usage_export_402_when_enforce_no_subscription(self, monkeypatch):
        client = _make_stripe_client(monkeypatch, billing_enforce=True)
        resp = client.get("/v1/billing/usage_export", params={"format": "csv", "period_days": 30})
        assert resp.status_code == 402
        body = resp.json()
        msg = body.get("error", {}).get("message", "") or str(body.get("detail", ""))
        assert "pro feature" in msg.lower() or "payment" in msg.lower()

    def test_usage_export_ok_when_enforce_and_active(self, monkeypatch):
        client = _make_stripe_client(monkeypatch, billing_enforce=True)
        # Activate subscription via webhook
        payload = _webhook_event("checkout.session.completed", org_id="default", project_id="default")
        client.post(
            "/v1/billing/webhook",
            content=payload,
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )
        resp = client.get("/v1/billing/usage_export", params={"format": "csv", "period_days": 30})
        assert resp.status_code == 200
        assert "date" in resp.text

    def test_usage_export_ok_when_enforce_disabled(self, monkeypatch):
        client = _make_stripe_client(monkeypatch, billing_enforce=False)
        resp = client.get("/v1/billing/usage_export", params={"format": "csv", "period_days": 30})
        assert resp.status_code == 200

    def test_usage_export_ok_without_stripe(self, monkeypatch):
        """Dev mode: billing enabled but stripe/enforce disabled -- no gating."""
        client = _make_client(monkeypatch, enable_billing=True, stripe_enabled=False, billing_enforce=False)
        resp = client.get("/v1/billing/usage_export", params={"format": "csv", "period_days": 30})
        assert resp.status_code == 200
