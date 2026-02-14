"""PII and secret leak detection for artifact auditing.

Scans artifacts for potential leaks of sensitive information.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Set, Union

class RedactionAuditError(Exception):
    """Audit process failed."""
    pass


@dataclass
class LeakFinding:
    """A detected potential leak."""
    type: str  # 'email', 'phone', 'api_key', 'token', 'password', etc.
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    file: str
    line: int
    column: int
    snippet: str
    context: str = ""  # Additional context about the finding
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "snippet": self.snippet,
            "context": self.context,
        }


@dataclass
class RedactionAuditReport:
    """Complete redaction audit report."""
    status: str  # 'PASS', 'FAIL', 'WARN'
    files_scanned: int = 0
    findings: List[LeakFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "files_scanned": self.files_scanned,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# Regex patterns for detecting sensitive data
SENSITIVE_PATTERNS: Dict[str, tuple] = {
    # Emails
    "email": (
        re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        "MEDIUM"
    ),
    
    # Phone numbers (US and international)
    "phone": (
        re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            re.IGNORECASE
        ),
        "MEDIUM"
    ),
    
    # API Keys (common patterns)
    "api_key": (
        re.compile(
            r'(?:api[_-]?key|apikey)["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_\-]{16,})',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # Bearer tokens
    "bearer_token": (
        re.compile(
            r'bearer\s+[a-zA-Z0-9_\-\.]+',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # Generic tokens
    "token": (
        re.compile(
            r'(?:token|auth_token|access_token)["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_\-\.]{8,})',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # Passwords
    "password": (
        re.compile(
            r'(?:password|passwd|pwd)["\'\s]*[:=]["\'\s]*([^\s"\'\n]{4,})',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # Secrets
    "secret": (
        re.compile(
            r'(?:secret|secret_key|client_secret)["\'\s]*[:=]["\'\s]*([a-zA-Z0-9_\-]{8,})',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # Private keys (various formats)
    "private_key": (
        re.compile(
            r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # AWS Access Keys
    "aws_access_key": (
        re.compile(r'AKIA[0-9A-Z]{16}'),
        "HIGH"
    ),
    
    # AWS Secret Keys
    "aws_secret_key": (
        re.compile(
            r'(?:aws_secret_access_key|aws_secret)["\'\s]*[:=]["\'\s]*([a-zA-Z0-9/+=]{40})',
            re.IGNORECASE
        ),
        "HIGH"
    ),
    
    # GitHub tokens
    "github_token": (
        re.compile(r'gh[pousr]_[A-Za-z0-9_]{36,}'),
        "HIGH"
    ),
    
    # Slack tokens
    "slack_token": (
        re.compile(r'xox[baprs]-[0-9a-zA-Z]{10,48}'),
        "HIGH"
    ),
    
    # Generic high-entropy strings that look like secrets
    "high_entropy_secret": (
        re.compile(
            r'["\']([a-zA-Z0-9+/=]{32,})["\']',
        ),
        "LOW"
    ),
}

# Whitelist patterns - things that look like secrets but aren't
WHITELIST_PATTERNS: List[Pattern] = [
    # Example/test values
    re.compile(r'example|sample|test|demo|fake|dummy', re.IGNORECASE),
    # Common placeholders
    re.compile(r'your_|my_|insert_|enter_|change_|replace_', re.IGNORECASE),
    # Hash values from our own system (already hashed)
    re.compile(r'^[a-f0-9]{64}$', re.IGNORECASE),  # SHA-256 hashes
    # UUIDs
    re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
]


def audit_run_directory(
    run_dir: Union[str, Path],
    fail_on_high: bool = True,
    warn_on_medium: bool = True,
) -> RedactionAuditReport:
    """Audit a run directory for PII/secrets leaks.
    
    Args:
        run_dir: Path to run directory
        fail_on_high: Whether HIGH severity findings cause FAIL status
        warn_on_medium: Whether MEDIUM severity findings cause WARN status
        
    Returns:
        RedactionAuditReport with findings
        
    Example:
        >>> report = audit_run_directory("/runs/run_001")
        >>> print(report.status)  # 'PASS', 'FAIL', or 'WARN'
        >>> if report.findings:
        ...     for finding in report.findings:
        ...         print(f"{finding.type}: {finding.snippet}")
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        raise RedactionAuditError(f"Run directory not found: {run_path}")
    
    report = RedactionAuditReport(status="PASS")
    
    # Files to audit
    audit_extensions = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
    files_to_audit = [
        f for f in run_path.iterdir()
        if f.is_file() and f.suffix.lower() in audit_extensions
    ]
    
    report.files_scanned = len(files_to_audit)
    
    # Audit each file
    for file_path in files_to_audit:
        findings = _audit_file(file_path)
        report.findings.extend(findings)
    
    # Determine status based on findings
    high_findings = [f for f in report.findings if f.severity == "HIGH"]
    medium_findings = [f for f in report.findings if f.severity == "MEDIUM"]
    
    if high_findings and fail_on_high:
        report.status = "FAIL"
    elif medium_findings and warn_on_medium:
        report.status = "WARN"
    
    # Build summary
    report.summary = {
        "high_severity": len(high_findings),
        "medium_severity": len(medium_findings),
        "low_severity": len([f for f in report.findings if f.severity == "LOW"]),
        "findings_by_type": _count_by_type(report.findings),
    }
    
    return report


def _audit_file(file_path: Path) -> List[LeakFinding]:
    """Audit a single file for leaks."""
    findings = []
    
    try:
        content = file_path.read_text(encoding='utf-8', errors='replace')
    except IOError as e:
        # Skip files we can't read
        return findings
    
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        for leak_type, (pattern, severity) in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(line):
                snippet = match.group(0)
                
                # Check whitelist
                if _is_whitelisted(snippet):
                    continue
                
                # Truncate snippet for reporting
                display_snippet = snippet[:50] + "..." if len(snippet) > 50 else snippet
                
                finding = LeakFinding(
                    type=leak_type,
                    severity=severity,
                    file=file_path.name,
                    line=line_num,
                    column=match.start() + 1,
                    snippet=display_snippet,
                )
                findings.append(finding)
    
    return findings


def _is_whitelisted(snippet: str) -> bool:
    """Check if a snippet is in the whitelist."""
    for pattern in WHITELIST_PATTERNS:
        if pattern.search(snippet):
            return True
    return False


def _count_by_type(findings: List[LeakFinding]) -> Dict[str, int]:
    """Count findings by type."""
    counts = {}
    for finding in findings:
        counts[finding.type] = counts.get(finding.type, 0) + 1
    return counts


def audit_single_file(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
) -> List[LeakFinding]:
    """Audit a single file for leaks.
    
    Args:
        file_path: Path to file
        file_type: Optional type hint ('json', 'jsonl', 'log')
        
    Returns:
        List of findings
    """
    return _audit_file(Path(file_path))


def write_audit_report(
    report: RedactionAuditReport,
    output_path: Union[str, Path],
) -> Path:
    """Write audit report to JSON file.
    
    Args:
        report: Audit report to write
        output_path: Path for redaction_audit.json
        
    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.write_text(report.to_json())
    return output_path


def quick_audit(run_dir: Union[str, Path]) -> str:
    """Quick audit returning just the status.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        'PASS', 'FAIL', or 'WARN'
    """
    report = audit_run_directory(run_dir)
    return report.status
