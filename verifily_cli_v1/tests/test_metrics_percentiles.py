"""Tests for metrics latency percentiles and enhanced metrics.

Target: ~10 tests, runtime <0.3s
"""

from __future__ import annotations

import pytest

from verifily_cli_v1.core.api.metrics import (
    LatencyHistogram,
    EndpointMetrics,
    MetricsStore,
)


class TestLatencyHistogram:
    """Test latency histogram calculations."""

    def test_record_and_count(self) -> None:
        """Can record samples and get count."""
        hist = LatencyHistogram()
        hist.record(100.0)
        hist.record(200.0)
        hist.record(300.0)
        
        assert hist.count() == 3

    def test_p50_calculation(self) -> None:
        """50th percentile is median."""
        hist = LatencyHistogram()
        for i in range(1, 101):
            hist.record(float(i))
        
        p50 = hist.p50()
        assert p50 is not None
        assert 48 <= p50 <= 52  # Approximate median

    def test_p95_calculation(self) -> None:
        """95th percentile calculation."""
        hist = LatencyHistogram()
        for i in range(1, 101):
            hist.record(float(i))
        
        p95 = hist.p95()
        assert p95 is not None
        assert 93 <= p95 <= 97  # 95th percentile

    def test_p99_calculation(self) -> None:
        """99th percentile calculation."""
        hist = LatencyHistogram()
        for i in range(1, 101):
            hist.record(float(i))
        
        p99 = hist.p99()
        assert p99 is not None
        assert 98 <= p99 <= 100  # 99th percentile

    def test_empty_histogram_returns_none(self) -> None:
        """Empty histogram returns None for percentiles."""
        hist = LatencyHistogram()
        
        assert hist.p50() is None
        assert hist.p95() is None
        assert hist.p99() is None

    def test_ring_buffer_limit(self) -> None:
        """Ring buffer respects max size."""
        hist = LatencyHistogram(max_samples=5)
        for i in range(10):
            hist.record(float(i))
        
        assert hist.count() == 5  # Only last 5 kept

    def test_reset_clears(self) -> None:
        """Reset clears all samples."""
        hist = LatencyHistogram()
        hist.record(100.0)
        hist.reset()
        
        assert hist.count() == 0
        assert hist.p50() is None


class TestEndpointMetrics:
    """Test per-endpoint metrics."""

    def test_record_request(self) -> None:
        """Request recording increments counter."""
        em = EndpointMetrics()
        em.record_request(100.0)
        em.record_request(200.0)
        
        assert em.request_count == 2
        assert em.latency.count() == 2

    def test_error_rate_calculation(self) -> None:
        """Error rate is correctly calculated."""
        em = EndpointMetrics()
        em.record_request(100.0, is_error=False)
        em.record_request(200.0, is_error=True)
        em.record_request(150.0, is_error=True)
        
        assert em.error_count == 2
        assert em.error_rate() == 2/3

    def test_zero_error_rate(self) -> None:
        """No errors means 0% error rate."""
        em = EndpointMetrics()
        em.record_request(100.0, is_error=False)
        
        assert em.error_rate() == 0.0

    def test_error_rate_no_requests(self) -> None:
        """No requests means 0% error rate."""
        em = EndpointMetrics()
        
        assert em.error_rate() == 0.0


class TestMetricsStore:
    """Test enhanced metrics store."""

    def test_record_endpoint_request(self) -> None:
        """Can record full endpoint metrics."""
        store = MetricsStore()
        store.record_endpoint_request("/v1/validate", "POST", 100.0, is_error=False)
        store.record_endpoint_request("/v1/validate", "POST", 200.0, is_error=False)
        
        assert store._requests_total == 2

    def test_record_decision_distribution(self) -> None:
        """Decision distribution is tracked separately."""
        store = MetricsStore()
        store.record_decision("PASS")
        store.record_decision("PASS")
        store.record_decision("FAIL")
        
        assert store._decision_distribution.get("PASS") == 2
        assert store._decision_distribution.get("FAIL") == 1

    def test_contamination_failure_tracking(self) -> None:
        """Contamination failures tracked."""
        store = MetricsStore()
        store.record_contamination_failure()
        store.record_contamination_failure()
        
        assert store._contamination_failures == 2

    def test_retrain_trigger_tracking(self) -> None:
        """Retrain triggers tracked."""
        store = MetricsStore()
        store.inc_retrain_trigger()
        
        assert store._retrain_triggers_total == 1

    def test_monitor_metrics(self) -> None:
        """Monitor metrics tracked."""
        store = MetricsStore()
        store.set_monitor_active(5)
        store.inc_monitor_regression()
        
        assert store._monitor_active == 5
        assert store._monitor_regressions_total == 1

    def test_latency_percentiles(self) -> None:
        """Global latency percentiles calculated."""
        store = MetricsStore()
        for i in range(1, 101):
            store.record_endpoint_request("/test", "GET", float(i))
        
        percentiles = store.get_latency_percentiles()
        assert percentiles["p50"] is not None
        assert percentiles["p95"] is not None
        assert percentiles["p99"] is not None

    def test_format_metrics_prometheus_style(self) -> None:
        """Output is Prometheus-style plaintext."""
        store = MetricsStore()
        store.inc_requests_total()
        store.record_decision("PASS")
        store.record_endpoint_request("/v1/validate", "POST", 100.0)
        
        output = store.format_metrics()
        
        # Check for Prometheus format elements
        assert "# HELP" in output
        assert "# TYPE" in output
        assert "verifily_requests_total" in output
        assert "verifily_decision_total" in output
        assert "verifily_endpoint_latency_ms" in output

    def test_format_metrics_includes_percentiles(self) -> None:
        """Output includes latency percentiles."""
        store = MetricsStore()
        for i in range(100, 201):
            store.record_endpoint_request("/test", "GET", float(i))
        
        output = store.format_metrics()
        
        assert 'quantile="0.5"' in output
        assert 'quantile="0.95"' in output
        assert 'quantile="0.99"' in output

    def test_format_metrics_includes_errors(self) -> None:
        """Output includes error counts."""
        store = MetricsStore()
        store.record_endpoint_request("/test", "GET", 100.0, is_error=True)
        
        output = store.format_metrics()
        
        assert "verifily_endpoint_errors_total" in output

    def test_reset_clears_all(self) -> None:
        """Reset clears all metrics."""
        store = MetricsStore()
        store.record_endpoint_request("/test", "GET", 100.0)
        store.record_decision("PASS")
        store.inc_retrain_trigger()
        
        store.reset()
        
        assert store._requests_total == 0
        assert len(store._decision_distribution) == 0
        assert store._retrain_triggers_total == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
