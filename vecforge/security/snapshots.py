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

Creates complete vault snapshots preserving encryption state and all
data. Snapshots of SQLite vaults use the SQLite online-backup API, so a
consistent single-file copy is produced even while the vault is open in
WAL mode (a plain file copy would miss everything still in the ``-wal``
file). Restoring removes stale ``-wal`` / ``-shm`` journals so the
restored file is what SQLite reads next.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_sqlite_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


class SnapshotManager:
    """Vault backup and restore manager.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        vault_path: Path to the vault database file.

    Performance:
        Snapshot: O(file_size) — SQLite online backup or file copy
        Restore: O(file_size) — full file copy

    Example:
        >>> snap = SnapshotManager("my_vault.db")
        >>> snap.create_snapshot("backups/")
        'backups/my_vault_20240101_120000.db'
        >>> snap.restore_snapshot("backups/my_vault_20240101_120000.db")
    """

    def __init__(self, vault_path: str) -> None:
        self._vault_path = Path(vault_path)

    def _journal_files(self, base: Path) -> list[Path]:
        return [base.with_name(base.name + "-wal"), base.with_name(base.name + "-shm")]

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
        snapshot_path = backup_path / f"{stem}_{timestamp}{suffix}"
        # why: avoid clobbering when two snapshots land in the same second
        counter = 1
        while snapshot_path.exists():
            snapshot_path = backup_path / f"{stem}_{timestamp}_{counter}{suffix}"
            counter += 1

        if _is_sqlite_file(self._vault_path):
            # security/why: online backup copies committed WAL content too
            # why: as_uri() percent-encodes '#', '?' and spaces in the path
            src_uri = self._vault_path.resolve().as_uri() + "?mode=ro"
            src = sqlite3.connect(src_uri, uri=True)
            dest = sqlite3.connect(str(snapshot_path))
            try:
                src.backup(dest)
            finally:
                dest.close()
                src.close()
        else:
            # Encrypted (SQLCipher) or foreign file: byte copy + journals
            shutil.copy2(str(self._vault_path), str(snapshot_path))
            for journal in self._journal_files(self._vault_path):
                if journal.exists():
                    shutil.copy2(
                        str(journal),
                        str(
                            snapshot_path.with_name(snapshot_path.name + journal.suffix)
                        ),
                    )

        logger.info("Snapshot created: %s", snapshot_path)
        return str(snapshot_path)

    def restore_snapshot(self, snapshot_path: str) -> None:
        """Restore vault from a snapshot.

        The vault must not be open in another process while restoring.

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

        # why: stale journals would override the restored main file
        for journal in self._journal_files(self._vault_path):
            if journal.exists():
                journal.unlink()

        shutil.copy2(str(source), str(self._vault_path))
        for journal in self._journal_files(source):
            if journal.exists():
                shutil.copy2(
                    str(journal),
                    str(
                        self._vault_path.with_name(
                            self._vault_path.name + journal.suffix
                        )
                    ),
                )
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
