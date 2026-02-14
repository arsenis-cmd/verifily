"""Tests for enterprise policy engine."""

from __future__ import annotations

from verifily_cli_v1.core.security.policies import (
    PolicyConfig,
    PolicyResult,
    evaluate_policies,
)


def _result(**overrides):
    """Build a minimal pipeline result dict."""
    base = {
        "contamination": {"status": "PASS"},
        "contract": {"valid": True},
        "report_summary": {"pii_total_hits": 0},
        "decision": {"metrics": {"f1": 0.95}},
    }
    base.update(overrides)
    return base


class TestContaminationPolicy:
    def test_pass_ok(self):
        cfg = PolicyConfig(require_contamination_pass=True)
        r = evaluate_policies(cfg, _result())
        assert r.allowed
        assert r.violations == []

    def test_fail_violation(self):
        cfg = PolicyConfig(require_contamination_pass=True)
        r = evaluate_policies(cfg, _result(contamination={"status": "FAIL"}))
        assert not r.allowed
        assert any("contamination" in v for v in r.violations)


class TestReproducibilityPolicy:
    def test_valid_ok(self):
        cfg = PolicyConfig(require_reproducibility=True)
        r = evaluate_policies(cfg, _result())
        assert r.allowed

    def test_invalid_violation(self):
        cfg = PolicyConfig(require_reproducibility=True)
        r = evaluate_policies(cfg, _result(contract={"valid": False}))
        assert not r.allowed
        assert any("contract" in v for v in r.violations)


class TestPiiPolicy:
    def test_within_threshold(self):
        cfg = PolicyConfig(block_if_pii_hits=5)
        r = evaluate_policies(cfg, _result(report_summary={"pii_total_hits": 3}))
        assert r.allowed

    def test_exceed_threshold(self):
        cfg = PolicyConfig(block_if_pii_hits=5)
        r = evaluate_policies(cfg, _result(report_summary={"pii_total_hits": 10}))
        assert not r.allowed
        assert any("PII" in v for v in r.violations)


class TestF1Policy:
    def test_above_threshold(self):
        cfg = PolicyConfig(min_f1_threshold=0.8)
        r = evaluate_policies(cfg, _result())
        assert r.allowed

    def test_below_threshold(self):
        cfg = PolicyConfig(min_f1_threshold=0.99)
        r = evaluate_policies(cfg, _result())
        assert not r.allowed
        assert any("F1" in v for v in r.violations)

    def test_missing_f1(self):
        cfg = PolicyConfig(min_f1_threshold=0.8)
        r = evaluate_policies(cfg, _result(decision={"metrics": {}}))
        assert not r.allowed
        assert any("unavailable" in v for v in r.violations)


class TestCombined:
    def test_empty_policy_allows_all(self):
        cfg = PolicyConfig()
        r = evaluate_policies(cfg, _result())
        assert r.allowed
        assert r.violations == []

    def test_multiple_violations(self):
        cfg = PolicyConfig(
            require_contamination_pass=True,
            require_reproducibility=True,
            block_if_pii_hits=0,
        )
        r = evaluate_policies(
            cfg,
            _result(
                contamination={"status": "FAIL"},
                contract={"valid": False},
                report_summary={"pii_total_hits": 5},
            ),
        )
        assert not r.allowed
        assert len(r.violations) == 3
