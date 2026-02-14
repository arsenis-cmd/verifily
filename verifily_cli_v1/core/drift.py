"""Dataset drift detection for Verifily.

Detects meaningful distribution changes between baseline and candidate datasets.
Uses MinHash similarity and tag distribution analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class DriftStatus(str, Enum):
    """Drift detection status."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class DriftCheckResult:
    """Result of drift detection check."""
    
    status: DriftStatus
    similarity_score: float  # 0-1, higher is more similar
    tag_shift: Dict[str, float] = field(default_factory=dict)
    length_shift: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    baseline_stats: Dict[str, Any] = field(default_factory=dict)
    candidate_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "similarity_score": round(self.similarity_score, 4),
            "tag_shift": self.tag_shift,
            "length_shift": self.length_shift,
            "reasons": self.reasons,
            "recommended_actions": self.recommended_actions,
            "baseline_stats": self.baseline_stats,
            "candidate_stats": self.candidate_stats,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class DriftError(Exception):
    """Drift detection failed."""
    pass


def _resolve_dataset_path(path: Union[str, Path]) -> Path:
    """Resolve dataset path from file or artifact directory."""
    path = Path(path)
    
    if path.is_file():
        return path
    
    # Try artifact directory structure
    dataset_path = path / "dataset.jsonl"
    if dataset_path.exists():
        return dataset_path
    
    # Try any JSONL file
    jsonl_files = list(path.glob("*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    
    raise DriftError(f"No dataset found in: {path}")


def _resolve_report_path(path: Union[str, Path]) -> Optional[Path]:
    """Resolve report.json path from artifact directory."""
    path = Path(path)
    
    if path.is_dir():
        report_path = path / "report.json"
        if report_path.exists():
            return report_path
    
    return None


def _compute_minhash_similarity(
    baseline_path: Path,
    candidate_path: Path,
    num_perm: int = 128,
    sample_size: Optional[int] = 5000,
) -> float:
    """Compute MinHash similarity between two datasets.
    
    Uses existing fingerprint logic from verifily.fingerprint.
    """
    try:
        # Import fingerprint module
        from verifily.fingerprint import compute_fingerprint
        
        baseline_fp = compute_fingerprint(str(baseline_path), num_perm=num_perm)
        candidate_fp = compute_fingerprint(str(candidate_path), num_perm=num_perm)
        
        similarity = baseline_fp.jaccard(candidate_fp)
        return float(similarity)
    except ImportError:
        # Fallback: simple n-gram similarity
        return _compute_ngram_similarity(baseline_path, candidate_path, sample_size)


def _compute_ngram_similarity(
    baseline_path: Path,
    candidate_path: Path,
    sample_size: Optional[int] = 5000,
) -> float:
    """Fallback n-gram similarity without MinHash."""
    def get_ngrams(text: str, n: int = 3) -> set:
        words = text.lower().split()
        return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))
    
    def sample_texts(path: Path, sample_size: int) -> set:
        ngrams = set()
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= sample_size:
                    break
                try:
                    obj = json.loads(line.strip())
                    if isinstance(obj, dict):
                        text = obj.get("text", "") or obj.get("instruction", "") or ""
                        if text:
                            ngrams.update(get_ngrams(str(text)))
                        count += 1
                except (json.JSONDecodeError, AttributeError):
                    continue
        return ngrams
    
    sample_size = sample_size or 5000
    baseline_ngrams = sample_texts(baseline_path, sample_size)
    candidate_ngrams = sample_texts(candidate_path, sample_size)
    
    if not baseline_ngrams or not candidate_ngrams:
        return 0.0
    
    intersection = len(baseline_ngrams & candidate_ngrams)
    union = len(baseline_ngrams | candidate_ngrams)
    
    return intersection / union if union > 0 else 0.0


def _extract_tag_distribution(
    dataset_path: Path,
    max_samples: int = 10000,
) -> Dict[str, Dict[str, int]]:
    """Extract tag distributions from dataset.
    
    Returns: {tag_key: {tag_value: count}}
    """
    tag_counts: Dict[str, Dict[str, int]] = {}
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            try:
                obj = json.loads(line.strip())
                if not isinstance(obj, dict):
                    continue
                
                # Look for common tag fields
                for key in ["category", "source", "difficulty", "tag", "tags", "label", "type"]:
                    value = obj.get(key)
                    if value is not None:
                        if isinstance(value, list):
                            for v in value:
                                tag_counts.setdefault(key, {}).setdefault(str(v), 0)
                                tag_counts[key][str(v)] += 1
                        else:
                            tag_counts.setdefault(key, {}).setdefault(str(value), 0)
                            tag_counts[key][str(value)] += 1
            except (json.JSONDecodeError, AttributeError):
                continue
    
    return tag_counts


def _compute_tag_shift(
    baseline_dist: Dict[str, Dict[str, int]],
    candidate_dist: Dict[str, Dict[str, int]],
) -> Dict[str, float]:
    """Compute L1 distance between tag distributions.
    
    Returns: {tag_key: shift_magnitude}
    """
    shifts = {}
    
    all_keys = set(baseline_dist.keys()) | set(candidate_dist.keys())
    
    for key in all_keys:
        baseline_counts = baseline_dist.get(key, {})
        candidate_counts = candidate_dist.get(key, {})
        
        if not baseline_counts or not candidate_counts:
            shifts[key] = 1.0  # Maximum shift if key missing
            continue
        
        # Normalize to probabilities
        baseline_total = sum(baseline_counts.values())
        candidate_total = sum(candidate_counts.values())
        
        if baseline_total == 0 or candidate_total == 0:
            shifts[key] = 1.0
            continue
        
        all_values = set(baseline_counts.keys()) | set(candidate_counts.keys())
        
        # Compute L1 distance
        l1_distance = 0.0
        for value in all_values:
            baseline_p = baseline_counts.get(value, 0) / baseline_total
            candidate_p = candidate_counts.get(value, 0) / candidate_total
            l1_distance += abs(baseline_p - candidate_p)
        
        # L1 distance is between 0 and 2, normalize to 0-1
        shifts[key] = min(1.0, l1_distance / 2)
    
    return shifts


def _extract_length_stats(dataset_path: Path, max_samples: int = 10000) -> Dict[str, float]:
    """Extract token/character length statistics from dataset."""
    lengths = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, dict):
                    # Get text length
                    text = obj.get("text", "") or obj.get("instruction", "") or obj.get("content", "")
                    if text:
                        lengths.append(len(str(text).split()))
            except (json.JSONDecodeError, AttributeError):
                continue
    
    if not lengths:
        return {}
    
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    
    return {
        "mean": sum(lengths) / n,
        "median": sorted_lengths[n // 2],
        "p50": sorted_lengths[n // 2],
        "p95": sorted_lengths[int(n * 0.95)],
        "p99": sorted_lengths[int(n * 0.99)] if n >= 100 else sorted_lengths[-1],
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
    }


def _compute_length_shift(
    baseline_stats: Dict[str, float],
    candidate_stats: Dict[str, float],
) -> Dict[str, float]:
    """Compute length statistic shifts."""
    if not baseline_stats or not candidate_stats:
        return {}
    
    shifts = {}
    for key in ["median", "p50", "p95"]:
        if key in baseline_stats and key in candidate_stats:
            baseline_val = baseline_stats[key]
            candidate_val = candidate_stats[key]
            if baseline_val > 0:
                # Relative change
                shifts[key] = (candidate_val - baseline_val) / baseline_val
            else:
                shifts[key] = 0.0
    
    return shifts


def detect_drift(
    baseline_dataset: Union[str, Path],
    candidate_dataset: Union[str, Path],
    *,
    schema: str = "auto",
    min_similarity_warn: float = 0.65,
    min_similarity_fail: float = 0.45,
    max_tag_shift_warn: float = 0.15,
    max_tag_shift_fail: float = 0.30,
    require_report_stats: bool = False,
) -> DriftCheckResult:
    """Detect drift between baseline and candidate datasets.
    
    Args:
        baseline_dataset: Path to baseline dataset or artifact directory
        candidate_dataset: Path to candidate dataset or artifact directory
        schema: Dataset schema type (auto-detected if "auto")
        min_similarity_warn: Similarity threshold for WARN (0-1)
        min_similarity_fail: Similarity threshold for FAIL (0-1)
        max_tag_shift_warn: Tag shift threshold for WARN (0-1)
        max_tag_shift_fail: Tag shift threshold for FAIL (0-1)
        require_report_stats: Fail if report.json stats unavailable
        
    Returns:
        DriftCheckResult with status, scores, and recommendations
        
    Example:
        >>> result = detect_drift(
        ...     "datasets/production_last/",
        ...     "datasets/candidate/",
        ...     min_similarity_warn=0.65,
        ...     min_similarity_fail=0.45,
        ... )
        >>> print(result.status)  # PASS, WARN, or FAIL
        >>> print(f"Similarity: {result.similarity_score:.2f}")
    """
    # Resolve paths
    baseline_path = _resolve_dataset_path(baseline_dataset)
    candidate_path = _resolve_dataset_path(candidate_dataset)
    
    if not baseline_path.exists():
        raise DriftError(f"Baseline dataset not found: {baseline_path}")
    if not candidate_path.exists():
        raise DriftError(f"Candidate dataset not found: {candidate_path}")
    
    # Compute similarity using fingerprint
    similarity = _compute_minhash_similarity(baseline_path, candidate_path)
    
    # Extract tag distributions
    baseline_tags = _extract_tag_distribution(baseline_path)
    candidate_tags = _extract_tag_distribution(candidate_path)
    tag_shift = _compute_tag_shift(baseline_tags, candidate_tags)
    
    # Extract length stats
    baseline_report_path = _resolve_report_path(baseline_dataset)
    candidate_report_path = _resolve_report_path(candidate_dataset)
    
    if baseline_report_path and candidate_report_path:
        # Use report.json stats if available
        try:
            with open(baseline_report_path) as f:
                baseline_report = json.load(f)
            with open(candidate_report_path) as f:
                candidate_report = json.load(f)
            
            baseline_stats = baseline_report.get("stats", {})
            candidate_stats = candidate_report.get("stats", {})
        except (json.JSONDecodeError, IOError):
            baseline_stats = _extract_length_stats(baseline_path)
            candidate_stats = _extract_length_stats(candidate_path)
    else:
        baseline_stats = _extract_length_stats(baseline_path)
        candidate_stats = _extract_length_stats(candidate_path)
        
        if require_report_stats and (not baseline_report_path or not candidate_report_path):
            return DriftCheckResult(
                status=DriftStatus.FAIL,
                similarity_score=similarity,
                reasons=["Report stats required but not available"],
                recommended_actions=["Generate report.json with --report flag"],
            )
    
    length_shift = _compute_length_shift(baseline_stats, candidate_stats)
    
    # Determine status
    reasons = []
    recommended_actions = []
    
    # Check similarity
    if similarity < min_similarity_fail:
        reasons.append(f"Similarity {similarity:.2f} below fail threshold {min_similarity_fail}")
        recommended_actions.append("Review dataset for major content changes")
        recommended_actions.append("Consider retraining with new baseline")
    elif similarity < min_similarity_warn:
        reasons.append(f"Similarity {similarity:.2f} below warn threshold {min_similarity_warn}")
        recommended_actions.append("Monitor for continued drift")
    
    # Check tag shifts
    max_tag_shift = max(tag_shift.values()) if tag_shift else 0.0
    if max_tag_shift > max_tag_shift_fail:
        reasons.append(f"Tag shift {max_tag_shift:.2f} exceeds fail threshold {max_tag_shift_fail}")
        recommended_actions.append("Review tag distribution changes")
    elif max_tag_shift > max_tag_shift_warn:
        reasons.append(f"Tag shift {max_tag_shift:.2f} exceeds warn threshold {max_tag_shift_warn}")
        recommended_actions.append("Investigate tag distribution changes")
    
    # Determine final status
    if similarity < min_similarity_fail or max_tag_shift > max_tag_shift_fail:
        status = DriftStatus.FAIL
    elif similarity < min_similarity_warn or max_tag_shift > max_tag_shift_warn:
        status = DriftStatus.WARN
    else:
        status = DriftStatus.PASS
    
    return DriftCheckResult(
        status=status,
        similarity_score=similarity,
        tag_shift=tag_shift,
        length_shift=length_shift,
        reasons=reasons,
        recommended_actions=recommended_actions,
        baseline_stats=baseline_stats,
        candidate_stats=candidate_stats,
    )


def format_drift_report(result: DriftCheckResult) -> str:
    """Format drift check result as a rich table."""
    lines = []
    
    # Status emoji
    status_emoji = {
        DriftStatus.PASS: "✅",
        DriftStatus.WARN: "⚠️",
        DriftStatus.FAIL: "❌",
    }
    
    lines.append("Drift Detection Report")
    lines.append("=" * 50)
    lines.append(f"Status: {status_emoji[result.status]} {result.status.value}")
    lines.append(f"Similarity Score: {result.similarity_score:.2%}")
    lines.append("")
    
    # Tag shifts
    if result.tag_shift:
        lines.append("Tag Distribution Shifts:")
        for tag, shift in sorted(result.tag_shift.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {tag}: {shift:.2%}")
        lines.append("")
    
    # Length shifts
    if result.length_shift:
        lines.append("Length Statistic Shifts:")
        for stat, shift in result.length_shift.items():
            lines.append(f"  {stat}: {shift:+.1%}")
        lines.append("")
    
    # Reasons
    if result.reasons:
        lines.append("Reasons:")
        for reason in result.reasons:
            lines.append(f"  • {reason}")
        lines.append("")
    
    # Recommendations
    if result.recommended_actions:
        lines.append("Recommended Actions:")
        for action in result.recommended_actions:
            lines.append(f"  • {action}")
        lines.append("")
    
    return "\n".join(lines)
