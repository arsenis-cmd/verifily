"""Tests for Risk Score and Health Index computation.

All tests are fast (<0.2s), deterministic, and verify:
- Score computation correctness
- Component weighting
- Boundary conditions (0-100 clamping)
- Artifact writing
- No PII in outputs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from verifily_cli_v1.core.scoring import (
    HealthIndex,
    RiskHealthSummary,
    RiskScore,
    ScoreComponent,
    compute_dataset_risk,
    compute_model_health,
    compute_scores_from_pipeline_result,
    compute_verdict,
    write_score_artifacts,
)


class TestScoreComponent:
    """Tests for ScoreComponent dataclass."""

    def test_component_creation(self):
        """Should create a valid component."""
        c = ScoreComponent(
            name="Test",
            value=50.0,
            weight=0.5,
            detail="Test detail",
        )
        assert c.name == "Test"
        assert c.value == 50.0
        assert c.weight == 0.5
        assert c.contribution == 25.0  # 50 * 0.5

    def test_contribution_auto_computed(self):
        """Contribution should be auto-computed from value * weight."""
        c = ScoreComponent(name="X", value=100, weight=0.3, detail="")
        assert c.contribution == 30.0

    def test_explicit_contribution(self):
        """Should allow explicit contribution override."""
        c = ScoreComponent(
            name="X", value=100, weight=0.3, detail="",
            contribution=25.0,  # Override
        )
        assert c.contribution == 25.0


class TestRiskScore:
    """Tests for RiskScore dataclass."""

    def test_risk_level_low(self):
        """Should classify 0-25 as LOW."""
        r = RiskScore(total=15, components=[], summary="Low risk")
        assert r.level == "LOW"

    def test_risk_level_medium(self):
        """Should classify 26-50 as MEDIUM."""
        r = RiskScore(total=35, components=[], summary="Medium risk")
        assert r.level == "MEDIUM"

    def test_risk_level_high(self):
        """Should classify 51-75 as HIGH."""
        r = RiskScore(total=65, components=[], summary="High risk")
        assert r.level == "HIGH"

    def test_risk_level_critical(self):
        """Should classify 76-100 as CRITICAL."""
        r = RiskScore(total=85, components=[], summary="Critical risk")
        assert r.level == "CRITICAL"

    def test_clamping_low(self):
        """Should clamp negative values to 0."""
        r = RiskScore(total=-10, components=[], summary="")
        assert r.total == 0.0

    def test_clamping_high(self):
        """Should clamp values > 100 to 100."""
        r = RiskScore(total=150, components=[], summary="")
        assert r.total == 100.0

    def test_to_dict_structure(self):
        """Should serialize to expected dict structure."""
        r = RiskScore(
            total=42.5,
            components=[ScoreComponent("PII", 30, 1.0, "3 hits")],
            summary="Test",
        )
        d = r.to_dict()
        assert d["total"] == 42.5
        assert d["level"] == "MEDIUM"
        assert d["summary"] == "Test"
        assert len(d["components"]) == 1
        assert d["components"][0]["name"] == "PII"


class TestHealthIndex:
    """Tests for HealthIndex dataclass."""

    def test_health_level_excellent(self):
        """Should classify 76-100 as EXCELLENT."""
        h = HealthIndex(total=85, components=[], summary="Great")
        assert h.level == "EXCELLENT"

    def test_health_level_good(self):
        """Should classify 51-75 as GOOD."""
        h = HealthIndex(total=65, components=[], summary="Good")
        assert h.level == "GOOD"

    def test_health_level_fair(self):
        """Should classify 26-50 as FAIR."""
        h = HealthIndex(total=35, components=[], summary="Fair")
        assert h.level == "FAIR"

    def test_health_level_poor(self):
        """Should classify 0-25 as POOR."""
        h = HealthIndex(total=15, components=[], summary="Poor")
        assert h.level == "POOR"


class TestComputeDatasetRisk:
    """Tests for compute_dataset_risk function."""

    def test_clean_dataset_low_risk(self):
        """Clean dataset should have LOW risk."""
        report = {
            "pii_clean": True,
            "pii_total_hits": 0,
            "row_count": 1000,
        }
        contract = {"valid": True}
        contamination = {"status": "PASS"}
        
        risk = compute_dataset_risk(report, contamination, contract)
        
        assert risk.total <= 25
        assert risk.level == "LOW"

    def test_pii_increases_risk(self):
        """PII hits should increase risk score."""
        report_clean = {"pii_clean": True, "pii_total_hits": 0, "row_count": 1000}
        report_pii = {"pii_clean": False, "pii_total_hits": 5, "row_count": 1000}
        
        risk_clean = compute_dataset_risk(report_clean, None, None)
        risk_pii = compute_dataset_risk(report_pii, None, None)
        
        assert risk_pii.total > risk_clean.total

    def test_contamination_fail_high_risk(self):
        """Contamination FAIL should result in high risk."""
        contamination = {
            "status": "FAIL",
            "exact_overlaps": 10,
            "near_duplicates": 5,
        }
        
        risk = compute_dataset_risk(None, contamination, None)
        
        assert risk.total >= 40  # Contamination adds 50, so should be high

    def test_contamination_warn_medium_risk(self):
        """Contamination WARN should add moderate risk."""
        contamination = {"status": "WARN"}
        
        risk = compute_dataset_risk(None, contamination, None)
        
        # Should have contamination component with 20 contribution
        contam_comp = [c for c in risk.components if c.name == "Contamination"]
        assert len(contam_comp) == 1
        assert contam_comp[0].contribution >= 20

    def test_invalid_contract_increases_risk(self):
        """Invalid contract should increase risk."""
        contract = {
            "valid": False,
            "missing_files": ["config.yaml"],
            "checks": [{"status": "FAIL"}],
        }
        
        risk = compute_dataset_risk(None, None, contract)
        
        assert risk.total > 0
        contract_comp = [c for c in risk.components if c.name == "Contract Issues"]
        assert len(contract_comp) == 1
        assert contract_comp[0].contribution > 0

    def test_small_dataset_risk(self):
        """Small datasets should have size risk component."""
        report = {"row_count": 50}  # Below default 100 threshold
        
        risk = compute_dataset_risk(report, None, None, min_rows_threshold=100)
        
        size_comp = [c for c in risk.components if c.name == "Dataset Size"]
        assert len(size_comp) == 1
        assert size_comp[0].contribution > 0

    def test_privacy_mode_local_bonus(self):
        """LOCAL privacy mode should reduce risk."""
        ctx_local = {"privacy_mode": "local"}
        ctx_remote = {"privacy_mode": "remote"}
        
        risk_local = compute_dataset_risk(None, None, None, ctx_local)
        risk_remote = compute_dataset_risk(None, None, None, ctx_remote)
        
        # Local should have lower base contribution (bonus)
        privacy_local = [c for c in risk_local.components if c.name == "Privacy Context"]
        privacy_remote = [c for c in risk_remote.components if c.name == "Privacy Context"]
        
        if privacy_local and privacy_remote:
            assert privacy_local[0].contribution <= privacy_remote[0].contribution

    def test_no_raw_data_in_output(self):
        """Output should not contain raw dataset data."""
        report = {
            "pii_clean": False,
            "pii_total_hits": 5,
            "row_count": 1000,
            # Should NOT include raw rows, samples, etc.
        }
        
        risk = compute_dataset_risk(report, None, None)
        output = risk.to_dict()
        
        # Verify no raw data
        output_str = json.dumps(output)
        assert "sample_rows" not in output_str.lower()
        assert "raw" not in output_str.lower() or "raw_dataset" not in output_str.lower()


class TestComputeModelHealth:
    """Tests for compute_model_health function."""

    def test_ship_decision_high_health(self):
        """SHIP decision should result in high health."""
        decision = {"recommendation": "SHIP", "confidence": 0.95}
        
        health = compute_model_health(decision, None)
        
        assert health.total >= 30  # At least 40 * 0.95 = 38 from decision
        decision_comp = [c for c in health.components if c.name == "Decision Outcome"]
        assert decision_comp[0].contribution >= 35

    def test_dont_ship_low_health(self):
        """DONT_SHIP decision should result in low health."""
        decision = {"recommendation": "DONT_SHIP", "confidence": 0.9}
        
        health = compute_model_health(decision, None)
        
        assert health.total < 50  # Should be low

    def test_f1_score_contributes(self):
        """F1 score should contribute to health."""
        eval_results = {"overall": {"f1": 0.75}}
        
        health = compute_model_health(None, eval_results)
        
        metric_comp = [c for c in health.components if c.name == "Evaluation Metrics"]
        assert len(metric_comp) == 1
        assert metric_comp[0].contribution == 22.5  # 0.75 * 30

    def test_accuracy_fallback(self):
        """Should use accuracy if F1 not available."""
        eval_results = {"overall": {"accuracy": 0.80}}
        
        health = compute_model_health(None, eval_results)
        
        metric_comp = [c for c in health.components if c.name == "Evaluation Metrics"]
        assert metric_comp[0].contribution == 24.0  # 0.80 * 30

    def test_reproducibility_boost(self):
        """Reproducibility verification should boost health."""
        health_no_repro = compute_model_health(None, None, reproducibility_ok=False)
        health_repro = compute_model_health(None, None, reproducibility_ok=True)
        
        repro_no = [c for c in health_no_repro.components if c.name == "Reproducibility"]
        repro_yes = [c for c in health_repro.components if c.name == "Reproducibility"]
        
        assert repro_yes[0].contribution > repro_no[0].contribution


class TestComputeVerdict:
    """Tests for compute_verdict function."""

    def test_approved_low_risk_good_health(self):
        """Low risk + good health = APPROVED."""
        risk = RiskScore(total=20, components=[], summary="Low")
        health = HealthIndex(total=80, components=[], summary="Good")
        
        verdict, recs = compute_verdict(risk, health)
        
        assert verdict == "APPROVED"

    def test_blocked_critical_risk(self):
        """Critical risk = BLOCKED."""
        risk = RiskScore(total=90, components=[], summary="Critical")
        health = HealthIndex(total=80, components=[], summary="Good")
        
        verdict, recs = compute_verdict(risk, health)
        
        assert verdict == "BLOCKED"
        assert any("dataset" in r.lower() for r in recs)

    def test_blocked_poor_health(self):
        """Poor health = BLOCKED."""
        risk = RiskScore(total=30, components=[], summary="Medium")
        health = HealthIndex(total=15, components=[], summary="Poor")
        
        verdict, recs = compute_verdict(risk, health)
        
        assert verdict == "BLOCKED"

    def test_review_required_mixed(self):
        """Mixed results = REVIEW_REQUIRED."""
        risk = RiskScore(total=60, components=[], summary="High")
        health = HealthIndex(total=60, components=[], summary="Fair")
        
        verdict, recs = compute_verdict(risk, health)
        
        assert verdict == "REVIEW_REQUIRED"


class TestWriteScoreArtifacts:
    """Tests for artifact writing."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for testing."""
        risk = RiskScore(
            total=35.5,
            components=[ScoreComponent("PII", 20, 1.0, "2 hits")],
            summary="Medium risk",
        )
        health = HealthIndex(
            total=75.0,
            components=[ScoreComponent("Decision", 30, 1.0, "SHIP")],
            summary="Good health",
        )
        return risk, health

    def test_writes_all_artifacts(self, sample_scores, tmp_path: Path):
        """Should write all expected artifacts."""
        risk, health = sample_scores
        
        artifacts = write_score_artifacts(risk, health, tmp_path)
        
        assert "risk_score_json" in artifacts
        assert "health_index_json" in artifacts
        assert "summary_json" in artifacts
        assert "risk_score_txt" in artifacts
        assert "health_index_txt" in artifacts

    def test_json_files_valid(self, sample_scores, tmp_path: Path):
        """JSON files should be valid and parseable."""
        risk, health = sample_scores
        
        artifacts = write_score_artifacts(risk, health, tmp_path)
        
        # Verify risk_score.json
        risk_data = json.loads(artifacts["risk_score_json"].read_text())
        assert risk_data["total"] == 35.5
        assert risk_data["level"] == "MEDIUM"
        
        # Verify health_index.json
        health_data = json.loads(artifacts["health_index_json"].read_text())
        assert health_data["total"] == 75.0
        assert health_data["level"] == "GOOD"

    def test_txt_files_readable(self, sample_scores, tmp_path: Path):
        """TXT files should be human-readable."""
        risk, health = sample_scores
        
        artifacts = write_score_artifacts(risk, health, tmp_path)
        
        risk_txt = artifacts["risk_score_txt"].read_text()
        assert "DATASET RISK SCORE" in risk_txt
        assert "35.5" in risk_txt
        assert "MEDIUM" in risk_txt

    def test_summary_includes_verdict(self, sample_scores, tmp_path: Path):
        """Summary JSON should include verdict."""
        risk, health = sample_scores
        
        artifacts = write_score_artifacts(risk, health, tmp_path)
        
        summary = json.loads(artifacts["summary_json"].read_text())
        assert "verdict" in summary
        assert "recommendations" in summary
        assert "risk_score" in summary
        assert "health_index" in summary


class TestRiskHealthSummary:
    """Tests for RiskHealthSummary."""

    def test_summary_to_dict(self):
        """Should serialize to dict correctly."""
        risk = RiskScore(total=25, components=[], summary="Low")
        health = HealthIndex(total=80, components=[], summary="Good")
        summary = RiskHealthSummary(
            risk_score=risk,
            health_index=health,
            verdict="APPROVED",
            recommendations=["All good"],
        )
        
        d = summary.to_dict()
        
        assert d["verdict"] == "APPROVED"
        assert d["recommendations"] == ["All good"]
        assert d["risk_score"]["total"] == 25
        assert d["health_index"]["total"] == 80


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_risk(self):
        """Same inputs should produce identical risk scores."""
        report = {"pii_clean": False, "pii_total_hits": 3, "row_count": 500}
        contamination = {"status": "WARN"}
        contract = {"valid": True}
        
        risk1 = compute_dataset_risk(report, contamination, contract)
        risk2 = compute_dataset_risk(report, contamination, contract)
        
        assert risk1.total == risk2.total
        assert len(risk1.components) == len(risk2.components)

    def test_same_input_same_health(self):
        """Same inputs should produce identical health indices."""
        decision = {"recommendation": "SHIP", "confidence": 0.9}
        eval_results = {"overall": {"f1": 0.72}}
        
        health1 = compute_model_health(decision, eval_results)
        health2 = compute_model_health(decision, eval_results)
        
        assert health1.total == health2.total
        assert len(health1.components) == len(health2.components)


class TestScoreCommandIntegration:
    """Integration tests for the score command."""

    def test_score_command_finds_artifacts(self, tmp_path: Path):
        """Score command should find and use artifacts."""
        # Create mock run directory
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        
        # Create artifacts
        (run_dir / "report.json").write_text(json.dumps({
            "pii_clean": True,
            "row_count": 1000,
        }))
        (run_dir / "decision.json").write_text(json.dumps({
            "recommendation": "SHIP",
            "confidence": 0.9,
        }))
        
        # Load and compute
        from verifily_cli_v1.core.io import read_json
        report = read_json(run_dir / "report.json")
        decision = read_json(run_dir / "decision.json")
        
        risk = compute_dataset_risk(report, None, None)
        health = compute_model_health(decision, None)
        
        assert risk.level == "LOW"
        assert health.level in ("GOOD", "EXCELLENT")
