# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Licensed under BSL 1.1 — see LICENSE for details.

"""Unit tests for VecForge exceptions."""

from __future__ import annotations

from vecforge.exceptions import (
    DeletionProtectedError,
    EncryptionKeyError,
    IngestError,
    InvalidAlphaError,
    NamespaceNotFoundError,
    VaultEmptyError,
    VecForgeError,
    VecForgePermissionError,
)


class TestExceptionHierarchy:
    """All exceptions must inherit from VecForgeError."""

    def test_vault_empty_inherits(self) -> None:
        assert issubclass(VaultEmptyError, VecForgeError)

    def test_namespace_not_found_inherits(self) -> None:
        assert issubclass(NamespaceNotFoundError, VecForgeError)

    def test_permission_error_inherits(self) -> None:
        assert issubclass(VecForgePermissionError, VecForgeError)

    def test_invalid_alpha_inherits(self) -> None:
        assert issubclass(InvalidAlphaError, VecForgeError)

    def test_encryption_key_inherits(self) -> None:
        assert issubclass(EncryptionKeyError, VecForgeError)

    def test_deletion_protected_inherits(self) -> None:
        assert issubclass(DeletionProtectedError, VecForgeError)

    def test_ingest_error_inherits(self) -> None:
        assert issubclass(IngestError, VecForgeError)


class TestExceptionMessages:
    """Error messages must tell the user what to do."""

    def test_vault_empty_has_action(self) -> None:
        err = VaultEmptyError("test_vault")
        msg = str(err)
        assert "test_vault" in msg
        assert "db.add" in msg or "db.ingest" in msg

    def test_namespace_not_found_shows_available(self) -> None:
        err = NamespaceNotFoundError("ward_99", ["default", "ward_7"])
        msg = str(err)
        assert "ward_99" in msg
        assert "default" in msg
        assert "create_namespace" in msg

    def test_permission_error_shows_role(self) -> None:
        err = VecForgePermissionError("write", "read-only")
        msg = str(err)
        assert "write" in msg
        assert "read-only" in msg

    def test_invalid_alpha_shows_range(self) -> None:
        err = InvalidAlphaError(1.5)
        msg = str(err)
        assert "1.5" in msg
        assert "0.0" in msg
        assert "1.0" in msg

    def test_encryption_key_wrong(self) -> None:
        err = EncryptionKeyError("wrong_key")
        msg = str(err)
        assert "VECFORGE_KEY" in msg

    def test_encryption_key_missing(self) -> None:
        err = EncryptionKeyError("missing")
        msg = str(err)
        assert "encryption_key" in msg

    def test_deletion_protected_shows_fix(self) -> None:
        err = DeletionProtectedError("doc_123")
        msg = str(err)
        assert "doc_123" in msg
        assert "deletion_protection" in msg

    def test_ingest_error_shows_formats(self) -> None:
        err = IngestError("report.xyz", "Unsupported format")
        msg = str(err)
        assert "report.xyz" in msg
        assert ".txt" in msg or ".pdf" in msg

    def test_all_exceptions_have_branding(self) -> None:
        """All exceptions must contain ArcGX branding."""
        exceptions = [
            VaultEmptyError("test"),
            NamespaceNotFoundError("test"),
            VecForgePermissionError("write", "read-only"),
            InvalidAlphaError(1.5),
            EncryptionKeyError("wrong_key"),
            DeletionProtectedError("doc"),
            IngestError("file", "reason"),
        ]
        for exc in exceptions:
            assert (
                "ArcGX" in str(exc) or "vecforge" in str(exc).lower()
            ), f"{type(exc).__name__} missing branding"
