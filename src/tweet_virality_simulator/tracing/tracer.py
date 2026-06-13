"""Tracer implementations and the process-wide registry.

A ``Tracer`` receives ``(tweet, dna, params, prediction)`` records. The default
is :class:`NullTracer` (emits nothing). The private backend swaps in a remote
sink with :func:`set_tracer` at startup — no engine change, and outcomes the
sink later observes close the calibration loop.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tracer(ABC):
    """Sink for prediction records."""

    @abstractmethod
    def emit(self, record: Dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NullTracer(Tracer):
    """Default: record nothing. Keeps the open engine private by default."""

    def emit(self, record: Dict[str, Any]) -> None:
        return None


class JsonlTracer(Tracer):
    """Append each record as one JSON line. Local, dependency-free."""

    def __init__(self, path: str) -> None:
        self.path = path

    def emit(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Tracing must never break an analysis.
            return None


_TRACER: Optional[Tracer] = None


def set_tracer(tracer: Optional[Tracer]) -> None:
    """Register the process-wide tracer (the backend attaches its sink here)."""
    global _TRACER
    _TRACER = tracer


def get_tracer() -> Tracer:
    """Resolve the active tracer: explicit > ``TVS_TRACE_PATH`` > null."""
    if _TRACER is not None:
        return _TRACER
    path = os.getenv("TVS_TRACE_PATH")
    if path:
        return JsonlTracer(path)
    return NullTracer()
