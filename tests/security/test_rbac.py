# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Security tests for RBAC — role-based access control."""

from __future__ import annotations

import pytest

from vecforge.exceptions import VecForgePermissionError
from vecforge.security.rbac import RBACManager


class TestRBAC:
    """Tests for role-based access control."""

    def test_default_admin_access(self) -> None:
        """No API key = local admin with full access."""
        rbac = RBACManager()
        rbac.require("read")
        rbac.require("write")
        rbac.require("delete")
        rbac.require("create_namespace")
        rbac.require("manage_keys")

    def test_read_only_can_read(self) -> None:
        """Read-only key can read."""
        rbac = RBACManager(
            api_key="readonly-key",
            key_roles={"readonly-key": "read-only"},
        )
        rbac.require("read")  # should not raise

    def test_read_only_cannot_write(self) -> None:
        """Read-only key must be blocked from write operations."""
        rbac = RBACManager(
            api_key="readonly-key",
            key_roles={"readonly-key": "read-only"},
        )
        with pytest.raises(VecForgePermissionError):
            rbac.require("write")

    def test_read_only_cannot_delete(self) -> None:
        """Read-only key must be blocked from delete operations."""
        rbac = RBACManager(
            api_key="readonly-key",
            key_roles={"readonly-key": "read-only"},
        )
        with pytest.raises(VecForgePermissionError):
            rbac.require("delete")

    def test_read_write_can_write(self) -> None:
        """Read-write key can read and write."""
        rbac = RBACManager(
            api_key="rw-key",
            key_roles={"rw-key": "read-write"},
        )
        rbac.require("read")
        rbac.require("write")
        rbac.require("delete")

    def test_read_write_cannot_manage_keys(self) -> None:
        """Read-write key cannot manage keys."""
        rbac = RBACManager(
            api_key="rw-key",
            key_roles={"rw-key": "read-write"},
        )
        with pytest.raises(VecForgePermissionError):
            rbac.require("manage_keys")

    def test_unknown_key_gets_no_access(self) -> None:
        """VecForge 2.0: an unregistered API key is denied everything."""
        rbac = RBACManager(api_key="unknown-key")
        assert rbac.current_role == "none"
        with pytest.raises(VecForgePermissionError):
            rbac.require("read")
        with pytest.raises(VecForgePermissionError):
            rbac.require("write")

    def test_hashed_registry_and_revoke(self) -> None:
        """Keys are stored as digests and resolve on the next manager."""
        from vecforge.security.rbac import hash_api_key

        admin = RBACManager()
        admin.register_key("team-key", "read-write")
        registry = admin.key_roles
        assert "team-key" not in registry
        assert registry[hash_api_key("team-key")] == "read-write"
        member = RBACManager(api_key="team-key", key_roles=registry)
        member.require("write")
        assert admin.revoke_key("team-key") and not admin.revoke_key("team-key")
        with pytest.raises(ValueError):
            admin.register_key("k", "none")

    def test_has_permission(self) -> None:
        """has_permission returns bool without raising."""
        rbac = RBACManager(
            api_key="readonly-key",
            key_roles={"readonly-key": "read-only"},
        )
        assert rbac.has_permission("read")
        assert not rbac.has_permission("write")

    def test_key_id_truncated(self) -> None:
        """Key ID is truncated for security."""
        rbac = RBACManager(api_key="super-secret-key-12345")
        assert "..." in rbac.key_id
        assert len(rbac.key_id) < len("super-secret-key-12345")

    def test_register_key(self) -> None:
        """Admin can register new keys."""
        rbac = RBACManager()  # admin
        rbac.register_key("new-key", "read-write")

    def test_non_admin_cannot_register(self) -> None:
        """Non-admin cannot register keys."""
        rbac = RBACManager(
            api_key="rw-key",
            key_roles={"rw-key": "read-write"},
        )
        with pytest.raises(VecForgePermissionError):
            rbac.register_key("new-key", "read-only")
