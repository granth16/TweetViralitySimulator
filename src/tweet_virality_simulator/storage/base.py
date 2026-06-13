"""Abstract storage interface.

A ``Store`` is a minimal content-addressable + report store. The closed backend
implements the same methods against Postgres/S3; the engine only depends on
this interface, so private data never enters the open package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

from ..models import Report


class Store(ABC):
    # --- reports ---
    @abstractmethod
    def save_report(self, report: Report) -> str:
        """Persist a report; return its id."""
        raise NotImplementedError

    @abstractmethod
    def get_report(self, report_id: str) -> Optional[Report]:
        raise NotImplementedError

    # --- arbitrary keyed artifacts (datasets, persona pools, profiles) ---
    @abstractmethod
    def put_artifact(self, key: str, value: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_artifacts(self, prefix: str = "") -> Iterable[str]:
        raise NotImplementedError
