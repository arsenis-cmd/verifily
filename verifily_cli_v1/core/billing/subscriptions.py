"""Subscription state store -- local JSONL persistence.

Tracks which org/project has an active Stripe subscription.
No external DB. Thread-safe. Same persistence pattern as billing_store.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("verifily.api")


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    INCOMPLETE = "incomplete"
    CANCELED = "canceled"


@dataclass
class SubscriptionRecord:
    """A subscription tied to an org + project."""

    org_id: str
    project_id: str
    stripe_customer_id: str = ""
    stripe_subscription_id: str = ""
    plan: str = "free"
    status: SubscriptionStatus = SubscriptionStatus.INCOMPLETE
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "project_id": self.project_id,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "plan": self.plan,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubscriptionRecord":
        return cls(
            org_id=d.get("org_id", "default"),
            project_id=d.get("project_id", "default"),
            stripe_customer_id=d.get("stripe_customer_id", ""),
            stripe_subscription_id=d.get("stripe_subscription_id", ""),
            plan=d.get("plan", "free"),
            status=SubscriptionStatus(d.get("status", "incomplete")),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
        )


class SubscriptionsStore:
    """Thread-safe in-memory subscription store with optional JSONL persistence."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Key: (org_id, project_id)
        self._subs: Dict[tuple, SubscriptionRecord] = {}
        self._persist_path: Optional[str] = None

    def configure_persistence(self, path: Optional[str]) -> None:
        with self._lock:
            self._persist_path = path
        if path:
            self._replay(path)

    def _replay(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    rec = SubscriptionRecord.from_dict(d)
                    key = (rec.org_id, rec.project_id)
                    with self._lock:
                        self._subs[key] = rec
        except Exception:
            logger.warning("subscriptions_store: replay failed for %s", path, exc_info=True)

    def _persist(self, rec: SubscriptionRecord) -> None:
        path = self._persist_path
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(rec.to_dict(), separators=(",", ":")) + "\n")
                f.flush()
        except Exception:
            logger.warning("subscriptions_store: persist failed", exc_info=True)

    def set_status(
        self,
        *,
        org_id: str,
        project_id: str,
        status: SubscriptionStatus,
        stripe_customer_id: str = "",
        stripe_subscription_id: str = "",
        plan: str = "free",
    ) -> SubscriptionRecord:
        """Create or update a subscription record."""
        now = time.time()
        key = (org_id, project_id)
        with self._lock:
            existing = self._subs.get(key)
            if existing:
                rec = SubscriptionRecord(
                    org_id=org_id,
                    project_id=project_id,
                    stripe_customer_id=stripe_customer_id or existing.stripe_customer_id,
                    stripe_subscription_id=stripe_subscription_id or existing.stripe_subscription_id,
                    plan=plan or existing.plan,
                    status=status,
                    created_at=existing.created_at,
                    updated_at=now,
                )
            else:
                rec = SubscriptionRecord(
                    org_id=org_id,
                    project_id=project_id,
                    stripe_customer_id=stripe_customer_id,
                    stripe_subscription_id=stripe_subscription_id,
                    plan=plan,
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
            self._subs[key] = rec
        self._persist(rec)
        return rec

    def get(self, org_id: str, project_id: str) -> Optional[SubscriptionRecord]:
        """Get subscription for org/project, or None."""
        with self._lock:
            return self._subs.get((org_id, project_id))

    def require_active(self, org_id: str, project_id: str) -> bool:
        """Return True if subscription is ACTIVE for this org/project."""
        rec = self.get(org_id, project_id)
        if rec is None:
            return False
        return rec.status == SubscriptionStatus.ACTIVE

    def list_all(self) -> List[SubscriptionRecord]:
        """Return all subscription records."""
        with self._lock:
            return list(self._subs.values())

    def reset(self) -> None:
        with self._lock:
            self._subs.clear()
            self._persist_path = None


subscriptions_store = SubscriptionsStore()
