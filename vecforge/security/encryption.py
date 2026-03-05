# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
SQLCipher AES-256 key management for VecForge.

Handles encryption key validation, SQLCipher PRAGMA setup, and
graceful fallback when SQLCipher C library is not available.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)


def check_sqlcipher_available() -> bool:
    """Check if SQLCipher is available on this system.

    Returns:
        True if sqlcipher3 can be imported, False otherwise.

    Performance:
        Time: O(1)
    """
    try:
        import sqlcipher3  # noqa: F401

        return True
    except ImportError:
        return False


def validate_encryption_key(key: str | None) -> str | None:
    """Validate and normalize an encryption key.

    Args:
        key: User-provided encryption key or None.

    Returns:
        Validated key string, or None if no encryption.

    Raises:
        ValueError: If key is empty or too short.

    Performance:
        Time: O(1)

    Example:
        >>> validate_encryption_key("my-secure-key-here")
        'my-secure-key-here'
        >>> validate_encryption_key("")
        ValueError: Encryption key must be at least 8 characters...
    """
    if key is None:
        return None

    if not key or len(key) < 8:
        raise ValueError(
            "Encryption key must be at least 8 characters long.\n"
            "Use a strong key: os.environ['VECFORGE_KEY']\n"
            "VecForge by Suneel Bose K · ArcGX TechLabs — docs: vecforge.arcgx.in"
        )

    if not check_sqlcipher_available():
        warnings.warn(
            "SQLCipher is not installed — data will NOT be encrypted.\n"
            "Install sqlcipher3 for AES-256 encryption: pip install sqlcipher3\n"
            "VecForge by Suneel Bose K · ArcGX TechLabs",
            UserWarning,
            stacklevel=2,
        )

    return key
