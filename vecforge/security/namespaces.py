# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Multi-tenant namespace isolation for VecForge.

Ensures that data operations are always scoped to a namespace,
preventing cross-tenant data leaks.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging

from vecforge.core.storage import StorageBackend
from vecforge.exceptions import NamespaceNotFoundError

logger = logging.getLogger(__name__)


class NamespaceManager:
    """Multi-tenant namespace manager.

    Every data operation flows through the namespace manager to
    guarantee tenant isolation. The 'default' namespace always exists.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        storage: StorageBackend instance.

    Performance:
        Namespace check: O(1) with cached list
        Create/list: O(K) where K = number of namespaces

    Example:
        >>> nsm = NamespaceManager(storage)
        >>> nsm.create("ward_7")
        >>> nsm.validate("ward_7")  # OK
        >>> nsm.validate("ward_99")  # raises NamespaceNotFoundError
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage
        self._cache: set[str] | None = None

    def _refresh_cache(self) -> None:
        """Refresh the namespace cache from storage.

        Performance:
            Time: O(K) where K = number of namespaces
        """
        self._cache = set(self._storage.list_namespaces())

    def validate(self, namespace: str) -> None:
        """Validate that a namespace exists.

        Args:
            namespace: Namespace name to validate.

        Raises:
            NamespaceNotFoundError: If namespace does not exist.

        Performance:
            Time: O(1) with cache
        """
        if self._cache is None:
            self._refresh_cache()

        assert self._cache is not None
        if namespace not in self._cache:
            # why: Refresh cache in case namespace was created elsewhere
            self._refresh_cache()
            if namespace not in self._cache:
                raise NamespaceNotFoundError(namespace, sorted(self._cache))

    def create(self, name: str) -> None:
        """Create a new namespace.

        Args:
            name: Namespace name.

        Performance:
            Time: O(1)
        """
        self._storage.create_namespace(name)
        if self._cache is not None:
            self._cache.add(name)
        logger.info("Created namespace: %s", name)

    def list_all(self) -> list[str]:
        """List all namespace names.

        Returns:
            Sorted list of namespace names.

        Performance:
            Time: O(K log K)
        """
        self._refresh_cache()
        assert self._cache is not None
        return sorted(self._cache)

    def exists(self, namespace: str) -> bool:
        """Check if a namespace exists.

        Args:
            namespace: Namespace name.

        Returns:
            True if namespace exists.

        Performance:
            Time: O(1) with cache
        """
        if self._cache is None:
            self._refresh_cache()
        assert self._cache is not None
        return namespace in self._cache
