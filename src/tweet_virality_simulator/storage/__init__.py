"""Storage seam: where reports, datasets, and artifacts live.

The open repo ships only :class:`LocalStore` (JSON on disk). Proprietary
datasets (real cascades, persona pools) and the hosted API's Postgres adapter
implement the same :class:`Store` interface and are registered at runtime — the
engine reads through the interface and never ships the private data.
"""

from __future__ import annotations

from .base import Store
from .local import LocalStore, MemoryStore

__all__ = ["Store", "LocalStore", "MemoryStore", "get_store"]


def get_store(backend: str = "local", **kwargs) -> Store:
    """Return a store by name. Only ``local``/``memory`` ship in the open repo;
    the private backend registers e.g. ``postgres`` against the same interface.
    """
    backend = (backend or "local").lower()
    if backend == "memory":
        return MemoryStore()
    return LocalStore(**kwargs)
