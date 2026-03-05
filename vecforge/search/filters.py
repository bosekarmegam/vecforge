# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
Metadata filtering logic for VecForge.

Supports equality, range (gte, lte, gt, lt), in/not_in operators
for flexible metadata-based result filtering.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

from typing import Any


class MetadataFilter:
    """Filter search results based on metadata conditions.

    Supports operators: equality, gte, lte, gt, lt, in, not_in, ne.
    Filters are specified as nested dictionaries.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.

    Performance:
        Time: O(N * F) where N = results, F = number of filter keys

    Example:
        >>> f = MetadataFilter({"type": "NDA", "year": {"gte": 2023}})
        >>> f.matches({"type": "NDA", "year": 2024})
        True
        >>> f.matches({"type": "NDA", "year": 2020})
        False
    """

    _OPERATORS = {"gte", "lte", "gt", "lt", "in", "not_in", "ne"}

    def __init__(self, filters: dict[str, Any]) -> None:
        self._filters = filters

    def matches(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata satisfies all filter conditions.

        Args:
            metadata: Document metadata dictionary.

        Returns:
            True if all filter conditions are satisfied.

        Performance:
            Time: O(F) where F = number of filter keys
        """
        for key, condition in self._filters.items():
            if key not in metadata:
                return False

            value = metadata[key]

            if isinstance(condition, dict):
                # why: Operator-based filtering
                if not self._check_operators(value, condition):
                    return False
            else:
                # why: Simple equality check
                if value != condition:
                    return False

        return True

    def _check_operators(self, value: Any, operators: dict[str, Any]) -> bool:
        """Check operator-based conditions on a value.

        Args:
            value: Metadata field value.
            operators: Dictionary of operator → threshold pairs.

        Returns:
            True if all operator conditions pass.

        Performance:
            Time: O(number of operators)
        """
        for op, threshold in operators.items():
            if op not in self._OPERATORS:
                # why: Treat unknown keys as nested equality
                if value != threshold:
                    return False
                continue

            if op == "gte" and not (value >= threshold):
                return False
            if op == "lte" and not (value <= threshold):
                return False
            if op == "gt" and not (value > threshold):
                return False
            if op == "lt" and not (value < threshold):
                return False
            if op == "in" and value not in threshold:
                return False
            if op == "not_in" and value in threshold:
                return False
            if op == "ne" and value == threshold:
                return False

        return True

    def filter_results(
        self, results: list[Any], metadata_getter: Any = None
    ) -> list[Any]:
        """Filter a list of results by metadata conditions.

        Args:
            results: List of result objects.
            metadata_getter: Callable or attribute name to extract metadata.
                If None, expects results to have a 'metadata' attribute.

        Returns:
            Filtered list of results.

        Performance:
            Time: O(N * F) where N = results, F = filter keys
        """
        if not self._filters:
            return results

        filtered = []
        for result in results:
            if metadata_getter is not None:
                if callable(metadata_getter):
                    meta = metadata_getter(result)
                else:
                    meta = getattr(result, metadata_getter)
            else:
                meta = getattr(result, "metadata", {})

            if self.matches(meta):
                filtered.append(result)

        return filtered
