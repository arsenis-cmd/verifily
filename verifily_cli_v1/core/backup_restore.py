"""Backup and restore for Verifily enterprise deployment.

Creates compressed archives of operational metadata (not raw datasets/runs).
Supports JSONL stores, logs, and configuration.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from verifily_cli_v1.core.runtime_paths import get_runtime_paths

logger = logging.getLogger("verifily.backup")


class BackupError(Exception):
    """Backup operation failed."""
    pass


class RestoreError(Exception):
    """Restore operation failed."""
    pass


@dataclass
class BackupManifest:
    """Manifest describing backup contents."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    verifily_version: str = "1.0.0"
    files: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "verifily_version": self.verifily_version,
            "files": self.files,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BackupManifest":
        return cls(
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            verifily_version=data.get("verifily_version", "1.0.0"),
            files=data.get("files", []),
        )


def _sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_backup_files(runtime_paths) -> List[Tuple[Path, str]]:
    """Get list of files to backup with their archive names.
    
    Returns:
        List of (source_path, archive_name) tuples
    """
    files = []
    
    # Usage events
    usage_log = runtime_paths.get_usage_log()
    if usage_log.exists():
        files.append((usage_log, "store/usage_events.jsonl"))
    
    # Jobs events
    jobs_log = runtime_paths.get_jobs_log()
    if jobs_log.exists():
        files.append((jobs_log, "store/jobs_events.jsonl"))
    
    # Monitor events
    monitor_log = runtime_paths.get_monitor_log()
    if monitor_log.exists():
        files.append((monitor_log, "store/monitor_events.jsonl"))
    
    # Workspaces store
    workspaces_store = runtime_paths.get_workspaces_store()
    if workspaces_store.exists():
        files.append((workspaces_store, "store/workspaces.jsonl"))
    
    return files


def create_backup(
    output_path: Path,
    include_logs: bool = False,
) -> Dict[str, any]:
    """Create a backup archive.
    
    Args:
        output_path: Path to write backup archive (.tar.gz)
        include_logs: Whether to include log files (default: False)
        
    Returns:
        Dict with backup statistics and manifest
        
    Raises:
        BackupError: If backup fails
    """
    runtime_paths = get_runtime_paths()
    manifest = BackupManifest()
    files_backed_up = 0
    total_bytes = 0
    
    try:
        with tarfile.open(output_path, "w:gz") as tar:
            # Get files to backup
            files = _get_backup_files(runtime_paths)
            
            for source_path, archive_name in files:
                if not source_path.exists():
                    continue
                
                # Calculate hash
                file_hash = _sha256_file(source_path)
                file_size = source_path.stat().st_size
                
                # Add to manifest
                manifest.files.append({
                    "path": archive_name,
                    "sha256": file_hash,
                    "size": file_size,
                })
                
                # Add to archive
                tar.add(source_path, arcname=archive_name)
                
                files_backed_up += 1
                total_bytes += file_size
                logger.info(f"Backed up: {source_path} -> {archive_name}")
            
            # Add manifest
            manifest_json = json.dumps(manifest.to_dict(), indent=2).encode()
            manifest_info = tarfile.TarInfo(name="backup_manifest.json")
            manifest_info.size = len(manifest_json)
            tar.addfile(manifest_info, fileobj=io.BytesIO(manifest_json))
        
        backup_size = output_path.stat().st_size
        
        return {
            "success": True,
            "output_path": str(output_path),
            "files_backed_up": files_backed_up,
            "total_bytes": total_bytes,
            "backup_bytes": backup_size,
            "manifest": manifest.to_dict(),
        }
        
    except Exception as e:
        raise BackupError(f"Failed to create backup: {e}") from e


def verify_backup(backup_path: Path) -> Tuple[bool, Optional[Dict]]:
    """Verify a backup archive integrity.
    
    Returns:
        Tuple of (is_valid, manifest_dict or None)
    """
    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            # Extract and parse manifest
            try:
                manifest_member = tar.getmember("backup_manifest.json")
                manifest_file = tar.extractfile(manifest_member)
                manifest_data = json.loads(manifest_file.read().decode())
            except (KeyError, json.JSONDecodeError) as e:
                return False, None
            
            # Verify each file
            for file_info in manifest_data.get("files", []):
                archive_path = file_info["path"]
                expected_hash = file_info["sha256"]
                
                try:
                    member = tar.getmember(archive_path)
                    f = tar.extractfile(member)
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                    
                    if actual_hash != expected_hash:
                        logger.error(f"Hash mismatch for {archive_path}")
                        return False, manifest_data
                except KeyError:
                    logger.error(f"Missing file in archive: {archive_path}")
                    return False, manifest_data
            
            return True, manifest_data
            
    except Exception as e:
        logger.error(f"Backup verification failed: {e}")
        return False, None


def restore_backup(
    backup_path: Path,
    force: bool = False,
) -> Dict[str, any]:
    """Restore from a backup archive.
    
    Args:
        backup_path: Path to backup archive
        force: Overwrite existing files
        
    Returns:
        Dict with restore statistics
        
    Raises:
        RestoreError: If restore fails
    """
    runtime_paths = get_runtime_paths()
    
    # Verify backup first
    is_valid, manifest_data = verify_backup(backup_path)
    if not is_valid:
        raise RestoreError("Backup archive is invalid or corrupted")
    
    # Check for existing files
    files_to_restore = []
    for file_info in manifest_data.get("files", []):
        archive_path = file_info["path"]
        
        # Map archive path to runtime path
        if archive_path == "store/usage_events.jsonl":
            dest_path = runtime_paths.get_usage_log()
        elif archive_path == "store/jobs_events.jsonl":
            dest_path = runtime_paths.get_jobs_log()
        elif archive_path == "store/monitor_events.jsonl":
            dest_path = runtime_paths.get_monitor_log()
        elif archive_path == "store/workspaces.jsonl":
            dest_path = runtime_paths.get_workspaces_store()
        else:
            continue
        
        if dest_path.exists() and not force:
            raise RestoreError(
                f"File already exists: {dest_path}\n"
                "Use --force to overwrite"
            )
        
        files_to_restore.append((archive_path, dest_path, file_info["sha256"]))
    
    # Perform restore
    restored = []
    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            for archive_path, dest_path, expected_hash in files_to_restore:
                # Ensure parent directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract file
                member = tar.getmember(archive_path)
                f = tar.extractfile(member)
                content = f.read()
                
                # Verify hash
                actual_hash = hashlib.sha256(content).hexdigest()
                if actual_hash != expected_hash:
                    raise RestoreError(f"Hash mismatch for {archive_path}")
                
                # Write file
                dest_path.write_bytes(content)
                restored.append(str(dest_path))
                logger.info(f"Restored: {dest_path}")
    
    except Exception as e:
        raise RestoreError(f"Restore failed: {e}") from e
    
    # Write restore report
    report = {
        "restored_at": datetime.now(timezone.utc).isoformat(),
        "backup_path": str(backup_path),
        "files_restored": restored,
        "manifest": manifest_data,
    }
    report_path = runtime_paths.store / "restore_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    
    return {
        "success": True,
        "files_restored": len(restored),
        "restore_report": str(report_path),
    }


def list_backup_contents(backup_path: Path) -> List[Dict[str, str]]:
    """List contents of a backup archive."""
    is_valid, manifest_data = verify_backup(backup_path)
    if not is_valid:
        raise BackupError("Invalid backup archive")
    
    return manifest_data.get("files", [])
