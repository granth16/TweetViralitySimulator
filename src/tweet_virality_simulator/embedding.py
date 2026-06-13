"""Tiny deterministic text embedding (hashing trick).

We avoid a heavy embedding model dependency for v0.1: tokens are hashed into a
fixed-width vector with signed buckets. Similar text → similar vectors, which is
enough to give the audience topical heterogeneity. Swappable later for a real
embedding model behind the same interface.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import EMB_DIM

_TOKEN_RE = re.compile(r"[a-z0-9#@']+")


def tokens(text: str):
    return _TOKEN_RE.findall(text.lower())


def _hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def embed_text(text: str, dim: int = EMB_DIM) -> np.ndarray:
    """Return a unit-norm embedding of ``text``."""
    vec = np.zeros(dim, dtype=np.float64)
    toks = tokens(text)
    if not toks:
        vec[0] = 1.0
        return vec
    for tok in toks:
        h = _hash(tok)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
