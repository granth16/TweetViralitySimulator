"""Local, dependency-free stores: JSON on disk and in-memory.

These are the open defaults. The private backend provides a Postgres/S3 store
implementing the same :class:`Store` interface.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, Optional

from ..models import Report
from .base import Store


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class MemoryStore(Store):
    """Ephemeral store, useful for tests and the hosted API's request scope."""

    def __init__(self) -> None:
        self._reports: Dict[str, Report] = {}
        self._artifacts: Dict[str, Dict[str, Any]] = {}

    def save_report(self, report: Report) -> str:
        rid = _hash_id(report.tweet)
        self._reports[rid] = report
        return rid

    def get_report(self, report_id: str) -> Optional[Report]:
        return self._reports.get(report_id)

    def put_artifact(self, key: str, value: Dict[str, Any]) -> None:
        self._artifacts[key] = value

    def get_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        return self._artifacts.get(key)

    def list_artifacts(self, prefix: str = "") -> Iterable[str]:
        return [k for k in self._artifacts if k.startswith(prefix)]


class LocalStore(Store):
    """JSON files under a root directory (default ``./.tvs_store``)."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.getenv("TVS_STORE_PATH", ".tvs_store")
        self._reports_dir = os.path.join(self.root, "reports")
        self._artifacts_dir = os.path.join(self.root, "artifacts")
        os.makedirs(self._reports_dir, exist_ok=True)
        os.makedirs(self._artifacts_dir, exist_ok=True)

    def _report_path(self, rid: str) -> str:
        return os.path.join(self._reports_dir, f"{rid}.json")

    def _artifact_path(self, key: str) -> str:
        safe = key.replace("/", "__")
        return os.path.join(self._artifacts_dir, f"{safe}.json")

    def save_report(self, report: Report) -> str:
        rid = _hash_id(report.tweet)
        with open(self._report_path(rid), "w", encoding="utf-8") as fh:
            fh.write(report.model_dump_json(indent=2))
        return rid

    def get_report(self, report_id: str) -> Optional[Report]:
        path = self._report_path(report_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return Report.model_validate_json(fh.read())

    def put_artifact(self, key: str, value: Dict[str, Any]) -> None:
        with open(self._artifact_path(key), "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, default=str)

    def get_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._artifact_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def list_artifacts(self, prefix: str = "") -> Iterable[str]:
        if not os.path.isdir(self._artifacts_dir):
            return []
        keys = [
            fn[:-5].replace("__", "/")
            for fn in os.listdir(self._artifacts_dir)
            if fn.endswith(".json")
        ]
        return [k for k in keys if k.startswith(prefix)]
