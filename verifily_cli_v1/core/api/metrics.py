"""In-memory metrics store for Verifily API.

Thread-safe counters and latency tracking — process lifetime only, no persistence.
Exposes Prometheus-style plaintext via format_metrics().
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


class LatencyHistogram:
    """Simple latency histogram with percentile calculation.
    
    Stores latencies in a ring buffer (fixed size) to limit memory.
    """
    
    def __init__(self, max_samples: int = 10000) -> None:
        self._max_samples = max_samples
        self._samples: Deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
    
    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        with self._lock:
            self._samples.append(latency_ms)
    
    def percentile(self, p: float) -> Optional[float]:
        """Calculate percentile (0-100)."""
        with self._lock:
            if not self._samples:
                return None
            sorted_samples = sorted(self._samples)
            idx = int(len(sorted_samples) * p / 100)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def p50(self) -> Optional[float]:
        """50th percentile (median)."""
        return self.percentile(50)
    
    def p95(self) -> Optional[float]:
        """95th percentile."""
        return self.percentile(95)
    
    def p99(self) -> Optional[float]:
        """99th percentile."""
        return self.percentile(99)
    
    def count(self) -> int:
        """Number of samples recorded."""
        with self._lock:
            return len(self._samples)
    
    def reset(self) -> None:
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


class EndpointMetrics:
    """Per-endpoint metrics including error rates and latency."""
    
    def __init__(self) -> None:
        self.request_count = 0
        self.error_count = 0
        self.latency = LatencyHistogram()
    
    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        self.request_count += 1
        self.latency.record(latency_ms)
        if is_error:
            self.error_count += 1
    
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def reset(self) -> None:
        self.request_count = 0
        self.error_count = 0
        self.latency.reset()


class MetricsStore:
    """Thread-safe in-memory counter store with latency tracking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # Request counters
        self._requests_total: int = 0
        self._requests_inflight: int = 0
        
        # Legacy endpoint tracking (keep for compatibility)
        self._endpoint_requests: Dict[Tuple[str, str], int] = {}
        
        # Enhanced per-endpoint metrics
        self._endpoint_metrics: Dict[Tuple[str, str], EndpointMetrics] = {}
        
        # Decision tracking
        self._decision_total: Dict[str, int] = {}
        self._decision_distribution: Dict[str, int] = {}  # PASS, FAIL, SKIP
        
        # Contamination tracking
        self._contamination_total: Dict[str, int] = {}
        self._contamination_failures: int = 0
        
        # Retrain triggers
        self._retrain_triggers_total: int = 0
        
        # Monitor metrics
        self._monitor_active: int = 0
        self._monitor_regressions_total: int = 0
        
        # Global latency histogram
        self._global_latency = LatencyHistogram()

    def inc_requests_total(self) -> None:
        with self._lock:
            self._requests_total += 1

    def inc_inflight(self) -> None:
        with self._lock:
            self._requests_inflight += 1

    def dec_inflight(self) -> None:
        with self._lock:
            self._requests_inflight = max(0, self._requests_inflight - 1)

    def inc_endpoint(self, path: str, method: str) -> None:
        """Legacy endpoint counter (increment only)."""
        key = (path, method)
        with self._lock:
            self._endpoint_requests[key] = self._endpoint_requests.get(key, 0) + 1

    def record_endpoint_request(
        self, 
        path: str, 
        method: str, 
        latency_ms: float,
        is_error: bool = False
    ) -> None:
        """Record full endpoint metrics including latency and errors."""
        key = (path, method)
        with self._lock:
            self._requests_total += 1
            
            if key not in self._endpoint_metrics:
                self._endpoint_metrics[key] = EndpointMetrics()
            self._endpoint_metrics[key].record_request(latency_ms, is_error)
            
            # Also update legacy counter for compatibility
            self._endpoint_requests[key] = self._endpoint_requests.get(key, 0) + 1
            
            # Update global latency
            self._global_latency.record(latency_ms)

    def inc_decision(self, decision: str) -> None:
        with self._lock:
            self._decision_total[decision] = self._decision_total.get(decision, 0) + 1

    def record_decision(self, status: str) -> None:
        """Record decision with distribution tracking."""
        with self._lock:
            self._decision_total[status] = self._decision_total.get(status, 0) + 1
            self._decision_distribution[status] = self._decision_distribution.get(status, 0) + 1

    def inc_contamination(self, status: str) -> None:
        with self._lock:
            self._contamination_total[status] = self._contamination_total.get(status, 0) + 1

    def record_contamination_failure(self) -> None:
        """Record a contamination check failure."""
        with self._lock:
            self._contamination_failures += 1

    def inc_retrain_trigger(self) -> None:
        """Record a retrain trigger event."""
        with self._lock:
            self._retrain_triggers_total += 1

    def set_monitor_active(self, count: int) -> None:
        """Set the number of active monitors."""
        with self._lock:
            self._monitor_active = count

    def inc_monitor_regression(self) -> None:
        """Record a monitor regression detection."""
        with self._lock:
            self._monitor_regressions_total += 1

    def get_latency_percentiles(self) -> Dict[str, Optional[float]]:
        """Get global latency percentiles."""
        with self._lock:
            return {
                "p50": self._global_latency.p50(),
                "p95": self._global_latency.p95(),
                "p99": self._global_latency.p99(),
            }

    def format_metrics(self) -> str:
        """Return Prometheus-style plaintext metrics."""
        lines: List[str] = []
        
        with self._lock:
            # Basic counters
            lines.append(f"# HELP verifily_requests_total Total API requests")
            lines.append(f"# TYPE verifily_requests_total counter")
            lines.append(f"verifily_requests_total {self._requests_total}")
            
            lines.append(f"# HELP verifily_requests_inflight Inflight requests")
            lines.append(f"# TYPE verifily_requests_inflight gauge")
            lines.append(f"verifily_requests_inflight {self._requests_inflight}")
            
            # Legacy endpoint counters (for compatibility)
            lines.append(f"# HELP verifily_endpoint_requests_total Requests per endpoint")
            lines.append(f"# TYPE verifily_endpoint_requests_total counter")
            for (path, method), count in sorted(self._endpoint_requests.items()):
                lines.append(
                    f'verifily_endpoint_requests_total{{path="{path}",method="{method}"}} {count}'
                )
            
            # Enhanced endpoint metrics with latency and errors
            lines.append(f"# HELP verifily_endpoint_latency_ms Endpoint latency in milliseconds")
            lines.append(f"# TYPE verifily_endpoint_latency_ms summary")
            for (path, method), metrics in sorted(self._endpoint_metrics.items()):
                p50 = metrics.latency.p50()
                p95 = metrics.latency.p95()
                p99 = metrics.latency.p99()
                if p50 is not None:
                    lines.append(
                        f'verifily_endpoint_latency_ms{{path="{path}",method="{method}",quantile="0.5"}} {p50:.3f}'
                    )
                if p95 is not None:
                    lines.append(
                        f'verifily_endpoint_latency_ms{{path="{path}",method="{method}",quantile="0.95"}} {p95:.3f}'
                    )
                if p99 is not None:
                    lines.append(
                        f'verifily_endpoint_latency_ms{{path="{path}",method="{method}",quantile="0.99"}} {p99:.3f}'
                    )
            
            # Error rates per endpoint
            lines.append(f"# HELP verifily_endpoint_errors_total Errors per endpoint")
            lines.append(f"# TYPE verifily_endpoint_errors_total counter")
            for (path, method), metrics in sorted(self._endpoint_metrics.items()):
                if metrics.error_count > 0:
                    lines.append(
                        f'verifily_endpoint_errors_total{{path="{path}",method="{method}"}} {metrics.error_count}'
                    )
            
            # Decision metrics
            lines.append(f"# HELP verifily_decision_total Total decisions")
            lines.append(f"# TYPE verifily_decision_total counter")
            for decision, count in sorted(self._decision_total.items()):
                lines.append(
                    f'verifily_decision_total{{decision="{decision}"}} {count}'
                )
            
            lines.append(f"# HELP verifily_decision_distribution Decision distribution")
            lines.append(f"# TYPE verifily_decision_distribution counter")
            for status, count in sorted(self._decision_distribution.items()):
                lines.append(
                    f'verifily_decision_distribution{{status="{status}"}} {count}'
                )
            
            # Contamination metrics
            lines.append(f"# HELP verifily_contamination_total Total contamination checks")
            lines.append(f"# TYPE verifily_contamination_total counter")
            for status, count in sorted(self._contamination_total.items()):
                lines.append(
                    f'verifily_contamination_total{{status="{status}"}} {count}'
                )
            
            lines.append(f"# HELP verifily_contamination_failures_total Contamination failures")
            lines.append(f"# TYPE verifily_contamination_failures_total counter")
            lines.append(f"verifily_contamination_failures_total {self._contamination_failures}")
            
            # Retrain triggers
            lines.append(f"# HELP verifily_retrain_triggers_total Retrain triggers")
            lines.append(f"# TYPE verifily_retrain_triggers_total counter")
            lines.append(f"verifily_retrain_triggers_total {self._retrain_triggers_total}")
            
            # Monitor metrics
            lines.append(f"# HELP verifily_monitor_active Active monitors")
            lines.append(f"# TYPE verifily_monitor_active gauge")
            lines.append(f"verifily_monitor_active {self._monitor_active}")
            
            lines.append(f"# HELP verifily_monitor_regressions_total Monitor regressions")
            lines.append(f"# TYPE verifily_monitor_regressions_total counter")
            lines.append(f"verifily_monitor_regressions_total {self._monitor_regressions_total}")
            
            # Global latency percentiles
            lines.append(f"# HELP verifily_latency_ms Latency in milliseconds")
            lines.append(f"# TYPE verifily_latency_ms summary")
            p50 = self._global_latency.p50()
            p95 = self._global_latency.p95()
            p99 = self._global_latency.p99()
            if p50 is not None:
                lines.append(f'verifily_latency_ms{{quantile="0.5"}} {p50:.3f}')
            if p95 is not None:
                lines.append(f'verifily_latency_ms{{quantile="0.95"}} {p95:.3f}')
            if p99 is not None:
                lines.append(f'verifily_latency_ms{{quantile="0.99"}} {p99:.3f}')
        
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all counters (for testing)."""
        with self._lock:
            self._requests_total = 0
            self._requests_inflight = 0
            self._endpoint_requests.clear()
            self._endpoint_metrics.clear()
            self._decision_total.clear()
            self._decision_distribution.clear()
            self._contamination_total.clear()
            self._contamination_failures = 0
            self._retrain_triggers_total = 0
            self._monitor_active = 0
            self._monitor_regressions_total = 0
            self._global_latency.reset()


# Singleton instance — one per process.
metrics = MetricsStore()
