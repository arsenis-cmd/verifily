"""SDK tests for Stripe checkout + subscription methods."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from verifily_cli_v1.core.api.jobs import jobs_store
from verifily_cli_v1.core.api.server import create_app
from verifily_cli_v1.core.api.settings import load_settings
from verifily_cli_v1.core.billing.stripe import MockStripeClient
from verifily_sdk import VerifilyClient


def _make_sdk(monkeypatch, **kwargs):
    monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
    monkeypatch.delenv("VERIFILY_STRIPE_ENABLED", raising=False)
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_mock")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_mock")
    monkeypatch.setenv("STRIPE_PRICE_ID_PRO", "price_mock")
    settings = load_settings(stripe_enabled=True, enable_billing=True, **kwargs)
    app = create_app(settings)
    jobs_store.stop_worker()
    app.state._stripe_client = MockStripeClient()
    transport = TestClient(app, base_url="http://testserver")
    sdk = VerifilyClient.__new__(VerifilyClient)
    sdk._base_url = "http://testserver"
    sdk._api_key = None
    sdk._timeout = 60.0
    sdk._retries = 0
    sdk._client = transport
    return sdk, transport


class TestSDKCheckout:
    def test_checkout_returns_url(self, monkeypatch):
        sdk, _ = _make_sdk(monkeypatch)
        resp = sdk.checkout(plan="pro")
        assert resp.checkout_url.startswith("https://checkout.stripe.com/mock/")
        assert resp.plan == "pro"

    def test_checkout_with_project(self, monkeypatch):
        sdk, _ = _make_sdk(monkeypatch)
        resp = sdk.checkout(plan="pro", project_id="my-proj")
        assert resp.stripe_customer_id.startswith("cus_mock_")


class TestSDKSubscription:
    def test_subscription_none(self, monkeypatch):
        sdk, _ = _make_sdk(monkeypatch)
        resp = sdk.subscription(project_id="default")
        assert resp.status == "none"
        assert resp.plan == "free"

    def test_subscription_after_activation(self, monkeypatch):
        sdk, transport = _make_sdk(monkeypatch)

        # Simulate webhook activation
        payload = json.dumps({
            "type": "checkout.session.completed",
            "data": {"object": {
                "customer": "cus_test",
                "subscription": "sub_test",
                "metadata": {"org_id": "default", "project_id": "my-proj"},
            }},
        }).encode()
        transport.post(
            "/v1/billing/webhook",
            content=payload,
            headers={"stripe-signature": "valid", "content-type": "application/json"},
        )

        resp = sdk.subscription(project_id="my-proj")
        assert resp.status == "active"
        assert resp.plan == "pro"

    def test_checkout_invalid_plan(self, monkeypatch):
        sdk, _ = _make_sdk(monkeypatch)
        with pytest.raises(Exception):
            sdk.checkout(plan="enterprise")

    def test_subscription_disabled_501(self, monkeypatch):
        monkeypatch.delenv("VERIFILY_API_KEY", raising=False)
        monkeypatch.delenv("VERIFILY_STRIPE_ENABLED", raising=False)
        settings = load_settings(stripe_enabled=False, enable_billing=True)
        app = create_app(settings)
        jobs_store.stop_worker()
        transport = TestClient(app, base_url="http://testserver")
        sdk = VerifilyClient.__new__(VerifilyClient)
        sdk._base_url = "http://testserver"
        sdk._api_key = None
        sdk._timeout = 60.0
        sdk._retries = 0
        sdk._client = transport
        with pytest.raises(Exception):
            sdk.subscription(project_id="default")
