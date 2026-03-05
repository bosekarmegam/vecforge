# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Backup and restore snapshots for VecForge vaults.

Creates complete vault snapshots preserving encryption state and
all data. Supports point-in-time recovery.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Vault backup and restore manager.

    Creates full snapshots of the vault database file. Encryption
    state is preserved — encrypted vaults remain encrypted in snapshots.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        vault_path: Path to the vault database file.

    Performance:
        Snapshot: O(file_size) — full file copy
        Restore: O(file_size) — full file copy

    Example:
        >>> snap = SnapshotManager("my_vault.db")
        >>> snap.create_snapshot("backups/")
        'backups/my_vault_20240101_120000.db'
        >>> snap.restore_snapshot("backups/my_vault_20240101_120000.db")
    """

    def __init__(self, vault_path: str) -> None:
        self._vault_path = Path(vault_path)

    def create_snapshot(self, backup_dir: str) -> str:
        """Create a timestamped snapshot of the vault.

        Args:
            backup_dir: Directory to store the snapshot file.

        Returns:
            Path to the created snapshot file.

        Raises:
            FileNotFoundError: If vault file does not exist.

        Performance:
            Time: O(file_size)
        """
        if not self._vault_path.exists():
            raise FileNotFoundError(
                f"Vault file not found: {self._vault_path}\n"
                f"VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = self._vault_path.stem
        suffix = self._vault_path.suffix
        snapshot_name = f"{stem}_{timestamp}{suffix}"
        snapshot_path = backup_path / snapshot_name

        shutil.copy2(str(self._vault_path), str(snapshot_path))
        logger.info("Snapshot created: %s", snapshot_path)

        return str(snapshot_path)

    def restore_snapshot(self, snapshot_path: str) -> None:
        """Restore vault from a snapshot.

        Args:
            snapshot_path: Path to the snapshot file to restore.

        Raises:
            FileNotFoundError: If snapshot file does not exist.

        Performance:
            Time: O(file_size)
        """
        source = Path(snapshot_path)
        if not source.exists():
            raise FileNotFoundError(
                f"Snapshot file not found: {snapshot_path}\n"
                f"VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        shutil.copy2(str(source), str(self._vault_path))
        logger.info("Vault restored from: %s", snapshot_path)

    def list_snapshots(self, backup_dir: str) -> list[str]:
        """List available snapshots in a backup directory.

        Args:
            backup_dir: Directory containing snapshots.

        Returns:
            Sorted list of snapshot file paths (newest first).

        Performance:
            Time: O(S log S) where S = number of snapshots
        """
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return []

        stem = self._vault_path.stem
        suffix = self._vault_path.suffix
        pattern = f"{stem}_*{suffix}"

        snapshots = sorted(
            [str(p) for p in backup_path.glob(pattern)],
            reverse=True,
        )
        return snapshots
