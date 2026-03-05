# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Audit log writer + reader for VecForge.

Records all mutating operations as append-only JSONL audit events.
Every add, delete, update, and admin action is logged for compliance.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditLogger:
    """Append-only JSONL audit log for compliance and security.

    Every mutating operation emits an audit event with actor, operation,
    target, timestamp, and metadata. Logs are append-only and tamper-evident.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        log_path: Path to the JSONL audit log file. If None, audit
            logging is disabled (but a warning is emitted).

    Performance:
        Write: O(1) per event (append-only)
        Read: O(N) where N = total events

    Example:
        >>> audit = AuditLogger("audit.jsonl")
        >>> audit.log("admin", "add", doc_id="d123", namespace="default")
        >>> events = audit.read_log()
        >>> print(events[0]["operation"])
        'add'
    """

    def __init__(self, log_path: str | None = None) -> None:
        self._path = Path(log_path) if log_path else None
        self._enabled = log_path is not None

        if self._enabled and self._path is not None:
            # why: Ensure parent directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Audit logging enabled: %s", self._path)
        else:
            logger.debug("Audit logging disabled")

    @property
    def enabled(self) -> bool:
        """Return whether audit logging is active.

        Performance:
            Time: O(1)
        """
        return self._enabled

    def log(
        self,
        actor: str,
        operation: str,
        doc_id: str | None = None,
        namespace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write an audit event to the log.

        Args:
            actor: Identifier of the user/key performing the action.
            operation: Operation name (add, delete, update, search, etc.).
            doc_id: Target document ID, if applicable.
            namespace: Target namespace, if applicable.
            metadata: Additional event metadata.

        Performance:
            Time: O(1) — single file append

        Example:
            >>> audit.log(
            ...     actor="key-abc123",
            ...     operation="add",
            ...     doc_id="doc-xyz",
            ...     namespace="ward_7",
            ...     metadata={"chars": 1500}
            ... )
        """
        if not self._enabled or self._path is None:
            return

        event = {
            "timestamp": time.time(),
            "actor": actor,
            "operation": operation,
            "doc_id": doc_id,
            "namespace": namespace,
            "metadata": metadata or {},
        }

        # security: Append-only write — no modification of existing entries
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def read_log(
        self,
        since: float | None = None,
        until: float | None = None,
        actor: str | None = None,
        operation: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read audit events with optional filters.

        Args:
            since: Unix timestamp — events after this time.
            until: Unix timestamp — events before this time.
            actor: Filter by actor identity.
            operation: Filter by operation type.

        Returns:
            List of audit event dictionaries.

        Performance:
            Time: O(N) where N = total events in log
        """
        if not self._enabled or self._path is None or not self._path.exists():
            return []

        events = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit log entry")
                    continue

                # why: Apply filters
                ts = event.get("timestamp", 0)
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue
                if actor is not None and event.get("actor") != actor:
                    continue
                if operation is not None and event.get("operation") != operation:
                    continue

                events.append(event)

        return events
