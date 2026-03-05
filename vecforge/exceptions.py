# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
VecForge custom exceptions.

All exceptions inherit from VecForgeError for easy catching.
Error messages always tell the user what to do next.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations


class VecForgeError(Exception):
    """Base exception for all VecForge errors.

    All VecForge exceptions inherit from this class so callers
    can catch ``except VecForgeError`` for a blanket handler.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.
    """


class VaultEmptyError(VecForgeError):
    """Raised when searching an empty vault.

    Example:
        >>> raise VaultEmptyError("my_vault")
        VaultEmptyError: Vault 'my_vault' contains no documents.
        Add documents with: db.add("your text") or db.ingest("path/")
    """

    def __init__(self, vault_name: str) -> None:
        super().__init__(
            f"Vault '{vault_name}' contains no documents.\n"
            f'Add documents with: db.add("your text") or db.ingest("path/")\n'
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class NamespaceNotFoundError(VecForgeError):
    """Raised when referencing a namespace that does not exist.

    Example:
        >>> raise NamespaceNotFoundError("ward_7", ["default", "ward_5"])
        NamespaceNotFoundError: Namespace 'ward_7' does not exist ...
    """

    def __init__(self, namespace: str, available: list[str] | None = None) -> None:
        available_str = ", ".join(available) if available else "none"
        super().__init__(
            f"Namespace '{namespace}' does not exist in this vault.\n"
            f"Available namespaces: [{available_str}]\n"
            f"Create it with: db.create_namespace('{namespace}')\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class VecForgePermissionError(VecForgeError):
    """Raised when the current API key lacks required permission.

    Example:
        >>> raise VecForgePermissionError("write", "read-only")
        VecForgePermissionError: Permission denied: 'write' requires ...
    """

    def __init__(self, operation: str, current_role: str) -> None:
        super().__init__(
            f"Permission denied: '{operation}' requires a higher role.\n"
            f"Current role: '{current_role}'.\n"
            f"Request an upgraded key from your vault administrator.\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class InvalidAlphaError(VecForgeError):
    """Raised when alpha is outside [0.0, 1.0].

    Example:
        >>> raise InvalidAlphaError(1.5)
        InvalidAlphaError: alpha must be between 0.0 and 1.0, got 1.5.
    """

    def __init__(self, alpha: float) -> None:
        super().__init__(
            f"alpha must be between 0.0 and 1.0, got {alpha}.\n"
            f"Use alpha=0.0 for keyword-only, alpha=1.0 for semantic-only, "
            f"alpha=0.5 for balanced hybrid search.\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class EncryptionKeyError(VecForgeError):
    """Raised when encryption key is invalid or missing.

    Example:
        >>> raise EncryptionKeyError("wrong_key")
        EncryptionKeyError: Failed to decrypt vault with provided key.
    """

    def __init__(self, reason: str = "wrong_key") -> None:
        messages = {
            "wrong_key": (
                "Failed to decrypt vault with provided key.\n"
                "Ensure VECFORGE_KEY environment variable is set correctly.\n"
                "If you've lost the key, the vault data cannot be recovered."
            ),
            "missing": (
                "This vault is encrypted but no encryption_key was provided.\n"
                "Pass encryption_key=os.environ['VECFORGE_KEY'] when opening.\n"
                "Example: VecForge('vault', encryption_key=os.environ['VECFORGE_KEY'])"
            ),
            "sqlcipher_unavailable": (
                "SQLCipher is not installed on this system.\n"
                "Encryption requires the sqlcipher3 package.\n"
                "Install with: pip install sqlcipher3\n"
                "Falling back to unencrypted SQLite storage."
            ),
        }
        msg = messages.get(reason, f"Encryption error: {reason}")
        super().__init__(
            f"{msg}\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class DeletionProtectedError(VecForgeError):
    """Raised when attempting to delete from a deletion-protected vault.

    Example:
        >>> raise DeletionProtectedError("doc_123")
        DeletionProtectedError: Cannot delete doc 'doc_123' ...
    """

    def __init__(self, doc_id: str) -> None:
        super().__init__(
            f"Cannot delete doc '{doc_id}': vault has deletion_protection=True.\n"
            f"Disable with: VecForge('vault', deletion_protection=False)\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )


class IngestError(VecForgeError):
    """Raised when document ingestion fails.

    Example:
        >>> raise IngestError("report.xyz", "Unsupported file format")
        IngestError: Failed to ingest 'report.xyz': Unsupported file format.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Failed to ingest '{path}': {reason}.\n"
            f"Supported formats: .txt, .md, .pdf, .docx, .html\n"
            f"VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )
