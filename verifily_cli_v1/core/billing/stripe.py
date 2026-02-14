"""Stripe integration layer -- thin wrapper around the stripe SDK.

Design principles:
  - Never import `stripe` if VERIFILY_STRIPE_ENABLED != 1
  - All Stripe calls go through StripeClient (mockable in tests)
  - Config loaded from env vars
  - Returns 501 if Stripe disabled, 500 if misconfigured
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("verifily.api")


@dataclass(frozen=True)
class StripeConfig:
    """Stripe configuration from environment."""

    enabled: bool = False
    secret_key: str = ""
    webhook_secret: str = ""
    price_id_pro: str = ""
    success_url: str = "http://localhost:8000/billing/success"
    cancel_url: str = "http://localhost:8000/billing/cancel"

    @classmethod
    def from_env(cls) -> "StripeConfig":
        from verifily_cli_v1.core.api.settings import _bool_env

        return cls(
            enabled=_bool_env("VERIFILY_STRIPE_ENABLED", False),
            secret_key=os.environ.get("STRIPE_SECRET_KEY", ""),
            webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", ""),
            price_id_pro=os.environ.get("STRIPE_PRICE_ID_PRO", ""),
            success_url=os.environ.get(
                "STRIPE_SUCCESS_URL", "http://localhost:8000/billing/success"
            ),
            cancel_url=os.environ.get(
                "STRIPE_CANCEL_URL", "http://localhost:8000/billing/cancel"
            ),
        )

    def validate(self) -> Optional[str]:
        """Return error message if config is incomplete, else None."""
        if not self.enabled:
            return None
        missing = []
        if not self.secret_key:
            missing.append("STRIPE_SECRET_KEY")
        if not self.webhook_secret:
            missing.append("STRIPE_WEBHOOK_SECRET")
        if not self.price_id_pro:
            missing.append("STRIPE_PRICE_ID_PRO")
        if missing:
            return f"Stripe enabled but missing env vars: {', '.join(missing)}"
        return None


class StripeClient:
    """Thin wrapper around stripe SDK. Mockable for testing.

    In production, set VERIFILY_STRIPE_ENABLED=1 and provide real keys.
    In tests, replace this with a mock that returns canned responses.
    """

    def __init__(self, config: StripeConfig) -> None:
        self._config = config
        self._stripe: Any = None
        if config.enabled and config.secret_key:
            try:
                import stripe

                stripe.api_key = config.secret_key
                self._stripe = stripe
            except ImportError:
                logger.warning(
                    "stripe package not installed; Stripe calls will fail"
                )

    def create_checkout_session(
        self,
        *,
        customer_id: Optional[str] = None,
        price_id: str,
        success_url: str,
        cancel_url: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout Session. Returns dict with id + url."""
        if self._stripe is None:
            raise RuntimeError("Stripe SDK not available")

        params: Dict[str, Any] = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
        }
        if customer_id:
            params["customer"] = customer_id
        if metadata:
            params["metadata"] = metadata

        session = self._stripe.checkout.Session.create(**params)
        return {"id": session.id, "url": session.url, "customer": session.customer}

    def create_customer(
        self,
        *,
        name: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a Stripe Customer. Returns customer ID."""
        if self._stripe is None:
            raise RuntimeError("Stripe SDK not available")

        params: Dict[str, Any] = {}
        if name:
            params["name"] = name
        if metadata:
            params["metadata"] = metadata

        customer = self._stripe.Customer.create(**params)
        return customer.id

    def construct_event(
        self, payload: bytes, sig_header: str
    ) -> Dict[str, Any]:
        """Verify webhook signature and parse event."""
        if self._stripe is None:
            raise RuntimeError("Stripe SDK not available")

        event = self._stripe.Webhook.construct_event(
            payload, sig_header, self._config.webhook_secret
        )
        return {"type": event.type, "data": event.data.object}


class MockStripeClient:
    """Mock Stripe client for testing without real Stripe calls.

    Used when VERIFILY_STRIPE_ENABLED=1 but no real stripe package.
    Also usable in test fixtures via monkeypatch.
    """

    def __init__(self, config: Optional[StripeConfig] = None) -> None:
        self._config = config
        self._session_counter = 0
        self._customer_counter = 0

    def create_checkout_session(
        self,
        *,
        customer_id: Optional[str] = None,
        price_id: str,
        success_url: str,
        cancel_url: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        self._session_counter += 1
        cid = customer_id or f"cus_mock_{self._customer_counter + 1:04d}"
        return {
            "id": f"cs_mock_{self._session_counter:04d}",
            "url": f"https://checkout.stripe.com/mock/{self._session_counter:04d}",
            "customer": cid,
        }

    def create_customer(
        self,
        *,
        name: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        self._customer_counter += 1
        return f"cus_mock_{self._customer_counter:04d}"

    def construct_event(
        self, payload: bytes, sig_header: str
    ) -> Dict[str, Any]:
        """Parse event from raw JSON payload. Ignores signature in mock mode."""
        import json

        if sig_header == "invalid":
            raise ValueError("Invalid signature")
        data = json.loads(payload)
        return {"type": data.get("type", ""), "data": data.get("data", {}).get("object", {})}
