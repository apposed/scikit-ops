"""Workflows that produce a collection of masks."""

from __future__ import annotations

from .detect_then_mask import Detected, detect_then_mask

__all__ = ["Detected", "detect_then_mask"]
