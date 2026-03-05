# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Role-based access control (RBAC) for VecForge.

Provides API key → role mapping with permission enforcement.
Every write operation must check permissions before executing.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging

from vecforge.exceptions import VecForgePermissionError

logger = logging.getLogger(__name__)

# security: Role hierarchy — higher roles inherit all lower permissions
_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"read", "write", "delete", "create_namespace", "manage_keys", "backup"},
    "read-write": {"read", "write", "delete"},
    "read-only": {"read"},
}

# why: Default role when no API key is provided — full access for local use
_DEFAULT_ROLE = "admin"


class RBACManager:
    """Role-based access control manager.

    Maps API keys to roles and enforces permission checks. When no
    API key is provided, defaults to admin role (local-first trust).

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Args:
        api_key: Current user's API key. None = local admin.
        key_roles: Mapping of API key → role name.

    Performance:
        Permission check: O(1)

    Example:
        >>> rbac = RBACManager(api_key="key123", key_roles={"key123": "read-only"})
        >>> rbac.require("read")   # OK
        >>> rbac.require("write")  # raises VecForgePermissionError
    """

    def __init__(
        self,
        api_key: str | None = None,
        key_roles: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._key_roles = key_roles or {}
        self._current_role = self._resolve_role()

    def _resolve_role(self) -> str:
        """Resolve the current user's role from their API key.

        Returns:
            Role name string.

        Performance:
            Time: O(1)
        """
        if self._api_key is None:
            # why: No API key = local use = admin privileges
            return _DEFAULT_ROLE

        role = self._key_roles.get(self._api_key, "read-only")
        if role not in _ROLE_PERMISSIONS:
            logger.warning(
                "Unknown role '%s' for key '%s' — defaulting to read-only",
                role,
                self._api_key[:8] + "...",
            )
            role = "read-only"

        return role

    @property
    def current_role(self) -> str:
        """Return the current user's role.

        Performance:
            Time: O(1)
        """
        return self._current_role

    @property
    def key_id(self) -> str:
        """Return a safe identifier for the current key (for audit logs).

        Performance:
            Time: O(1)
        """
        if self._api_key is None:
            return "local-admin"
        # security: Never log full API keys
        return self._api_key[:8] + "..." if len(self._api_key) > 8 else "***"

    def require(self, permission: str) -> None:
        """Check if current role has the required permission.

        Args:
            permission: Required permission (read, write, delete, etc.).

        Raises:
            VecForgePermissionError: If current role lacks the permission.

        Performance:
            Time: O(1)

        Example:
            >>> rbac.require("write")  # raises if read-only
        """
        permissions = _ROLE_PERMISSIONS.get(self._current_role, set())
        if permission not in permissions:
            raise VecForgePermissionError(permission, self._current_role)

    def has_permission(self, permission: str) -> bool:
        """Check if current role has a permission without raising.

        Args:
            permission: Permission to check.

        Returns:
            True if permission is granted, False otherwise.

        Performance:
            Time: O(1)
        """
        permissions = _ROLE_PERMISSIONS.get(self._current_role, set())
        return permission in permissions

    def register_key(self, api_key: str, role: str) -> None:
        """Register a new API key with a role.

        Args:
            api_key: The API key to register.
            role: Role to assign (admin, read-write, read-only).

        Raises:
            VecForgePermissionError: If current user is not admin.
            ValueError: If role is not valid.

        Performance:
            Time: O(1)
        """
        # security: Only admins can register new keys
        self.require("manage_keys")

        if role not in _ROLE_PERMISSIONS:
            raise ValueError(
                f"Invalid role '{role}'.\n"
                f"Valid roles: {', '.join(_ROLE_PERMISSIONS.keys())}\n"
                f"VecForge by Suneel Bose K · ArcGX TechLabs"
            )

        self._key_roles[api_key] = role
        logger.info("Registered key %s... with role '%s'", api_key[:8], role)
