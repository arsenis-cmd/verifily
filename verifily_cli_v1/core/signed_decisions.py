"""Digital signatures for decision artifacts.

Provides optional signing of decision files using SHA-256 + local private key.
No external crypto services required.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

class SigningError(Exception):
    """Signing operation failed."""
    pass


class VerificationError(Exception):
    """Signature verification failed."""
    pass


class MissingKeyError(SigningError):
    """Signing key not found."""
    pass


# Default key location
DEFAULT_KEY_PATH = Path.home() / ".verifily" / "signing_key.pem"


def generate_signing_key(key_path: Optional[Union[str, Path]] = None) -> Path:
    """Generate a new Ed25519-style signing key pair.
    
    For this implementation, we use a simple SHA-256 HMAC-based
    signature scheme that requires no external crypto libraries.
    In production, replace with Ed25519 or ECDSA.
    
    Args:
        key_path: Where to store the key (default: ~/.verifily/signing_key.pem)
        
    Returns:
        Path to generated key file
        
    Example:
        >>> key_path = generate_signing_key()
        >>> print(f"Key stored at: {key_path}")
    """
    key_path = Path(key_path or DEFAULT_KEY_PATH)
    key_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    
    # Generate 256-bit random key
    secret_key = secrets.token_hex(32)
    
    # Store with basic metadata
    key_data = {
        "version": "1.0",
        "algorithm": "hmac-sha256",
        "key_id": secrets.token_hex(8),
        "secret_key": secret_key,
        "created": _now_iso(),
    }
    
    key_path.write_text(json.dumps(key_data, indent=2))
    key_path.chmod(0o600)  # Owner read/write only
    
    return key_path


def sign_decision(
    decision_path: Union[str, Path],
    key_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Sign a decision file.
    
    Creates a detached signature file (.sig) alongside the decision.
    
    Args:
        decision_path: Path to decision.json
        key_path: Path to signing key (generates if missing)
        output_path: Path for signature (default: decision.sig in same dir)
        
    Returns:
        Signature metadata including signature string
        
    Raises:
        SigningError: If signing fails
        MissingKeyError: If key not found and auto-generate disabled
        
    Example:
        >>> result = sign_decision("/runs/run_001/decision.json")
        >>> print(f"Signature: {result['signature'][:32]}...")
    """
    decision_path = Path(decision_path)
    
    if not decision_path.exists():
        raise SigningError(f"Decision file not found: {decision_path}")
    
    # Generate key if not exists
    key_path = Path(key_path or DEFAULT_KEY_PATH)
    if not key_path.exists():
        key_path = generate_signing_key(key_path)
    
    # Load key
    key_data = json.loads(key_path.read_text())
    secret_key = key_data.get("secret_key")
    key_id = key_data.get("key_id", "unknown")
    
    if not secret_key:
        raise SigningError(f"Invalid key file: {key_path}")
    
    # Load and canonicalize decision
    try:
        with open(decision_path) as f:
            decision = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise SigningError(f"Cannot read decision: {e}")
    
    # Create canonical form (excluding any existing signature)
    decision_copy = {k: v for k, v in decision.items() if k not in ("signature", "_signature")}
    canonical_json = json.dumps(decision_copy, sort_keys=True, separators=(',', ':'))
    
    # Sign using HMAC-SHA256
    signature = _hmac_sign(canonical_json.encode('utf-8'), secret_key)
    
    # Build signature metadata
    sig_metadata = {
        "version": "1.0",
        "algorithm": "hmac-sha256",
        "key_id": key_id,
        "timestamp": _now_iso(),
        "signature": signature,
        "decision_hash": hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()[:16],
    }
    
    # Determine output path
    if output_path is None:
        output_path = decision_path.parent / "decision.sig"
    else:
        output_path = Path(output_path)
    
    # Write signature file
    output_path.write_text(json.dumps(sig_metadata, indent=2))
    
    return sig_metadata


def verify_decision_signature(
    decision_path: Union[str, Path],
    signature_path: Optional[Union[str, Path]] = None,
    key_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Verify a decision signature.
    
    Args:
        decision_path: Path to decision.json
        signature_path: Path to decision.sig (auto-detected if None)
        key_path: Path to signing key (default: ~/.verifily/signing_key.pem)
        
    Returns:
        Verification result with status and details
        
    Raises:
        VerificationError: If signature invalid
        MissingKeyError: If key not found
        
    Example:
        >>> result = verify_decision_signature("/runs/run_001/decision.json")
        >>> print(f"Valid: {result['valid']}")
    """
    decision_path = Path(decision_path)
    
    # Determine signature path
    if signature_path is None:
        signature_path = decision_path.parent / "decision.sig"
    else:
        signature_path = Path(signature_path)
    
    # Load signature
    if not signature_path.exists():
        raise VerificationError(f"Signature file not found: {signature_path}")
    
    try:
        sig_metadata = json.loads(signature_path.read_text())
    except (json.JSONDecodeError, IOError) as e:
        raise VerificationError(f"Cannot read signature: {e}")
    
    # Load key
    key_path = Path(key_path or DEFAULT_KEY_PATH)
    if not key_path.exists():
        raise MissingKeyError(f"Signing key not found: {key_path}")
    
    key_data = json.loads(key_path.read_text())
    secret_key = key_data.get("secret_key")
    stored_key_id = key_data.get("key_id", "unknown")
    
    # Load decision
    try:
        with open(decision_path) as f:
            decision = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise VerificationError(f"Cannot read decision: {e}")
    
    # Create canonical form
    decision_copy = {k: v for k, v in decision.items() if k not in ("signature", "_signature")}
    canonical_json = json.dumps(decision_copy, sort_keys=True, separators=(',', ':'))
    
    # Extract signature
    signature = sig_metadata.get("signature")
    sig_key_id = sig_metadata.get("key_id", "unknown")
    
    if not signature:
        raise VerificationError("No signature found in metadata")
    
    # Verify key ID matches
    if sig_key_id != stored_key_id:
        return {
            "valid": False,
            "error": f"Key ID mismatch: signature={sig_key_id}, key={stored_key_id}",
            "key_id_match": False,
        }
    
    # Verify signature
    expected_signature = _hmac_sign(canonical_json.encode('utf-8'), secret_key)
    signature_valid = secrets.compare_digest(signature, expected_signature)
    
    # Verify decision hash if present
    decision_hash_valid = True
    stored_hash = sig_metadata.get("decision_hash")
    if stored_hash:
        computed_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()[:16]
        decision_hash_valid = stored_hash == computed_hash
    
    return {
        "valid": signature_valid and decision_hash_valid,
        "signature_valid": signature_valid,
        "hash_valid": decision_hash_valid,
        "algorithm": sig_metadata.get("algorithm", "unknown"),
        "key_id": sig_key_id,
        "timestamp": sig_metadata.get("timestamp"),
    }


def _hmac_sign(message: bytes, key: str) -> str:
    """Simple HMAC-SHA256 signature.
    
    Note: In production, use Python's hmac module or cryptography library.
    This is a minimal implementation for zero-dependency signing.
    """
    # Basic HMAC construction: H(K XOR opad || H(K XOR ipad || message))
    key_bytes = key.encode('utf-8')
    
    # Key padding to block size (64 bytes for SHA-256)
    if len(key_bytes) > 64:
        key_bytes = hashlib.sha256(key_bytes).digest()
    
    key_bytes = key_bytes.ljust(64, b'\x00')
    
    # Inner and outer pads
    ipad = bytes(b ^ 0x36 for b in key_bytes)
    opad = bytes(b ^ 0x5c for b in key_bytes)
    
    # HMAC computation
    inner_hash = hashlib.sha256(ipad + message).digest()
    outer_hash = hashlib.sha256(opad + inner_hash).digest()
    
    return base64.b64encode(outer_hash).decode('ascii')


def _now_iso() -> str:
    """Current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def has_signature(run_dir: Union[str, Path]) -> bool:
    """Check if a run directory has a decision signature.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        True if decision.sig exists
    """
    return (Path(run_dir) / "decision.sig").exists()


def batch_sign_runs(
    runs_dir: Union[str, Path],
    pattern: str = "run_*",
    key_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Sign all decisions in a runs directory.
    
    Args:
        runs_dir: Directory containing run subdirectories
        pattern: Glob pattern for run directories
        key_path: Path to signing key
        
    Returns:
        Summary of signing operations
    """
    runs_path = Path(runs_dir)
    results = {
        "signed": 0,
        "skipped": 0,
        "errors": [],
        "runs": [],
    }
    
    for run_dir in runs_path.glob(pattern):
        if not run_dir.is_dir():
            continue
        
        decision_path = run_dir / "decision.json"
        if not decision_path.exists():
            results["skipped"] += 1
            continue
        
        try:
            sig_result = sign_decision(decision_path, key_path)
            results["signed"] += 1
            results["runs"].append({
                "run_id": run_dir.name,
                "signature": sig_result["signature"][:16] + "...",
            })
        except SigningError as e:
            results["errors"].append({
                "run_id": run_dir.name,
                "error": str(e),
            })
    
    return results
