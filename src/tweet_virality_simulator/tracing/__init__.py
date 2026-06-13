"""Tracing seam: emit every prediction record for the outcome-ingestion flywheel.

The open engine emits a structured record per analysis. By default this is a
no-op (privacy first). Set ``TVS_TRACE_PATH`` to append JSONL locally, or have
the private backend register a remote sink via :func:`set_tracer` — that sink is
what feeds retrain jobs. The engine never imports the closed sink directly.
"""

from __future__ import annotations

from .tracer import (
    JsonlTracer,
    NullTracer,
    Tracer,
    get_tracer,
    set_tracer,
)

__all__ = ["Tracer", "NullTracer", "JsonlTracer", "get_tracer", "set_tracer"]
