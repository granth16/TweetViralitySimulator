"""Validation & calibration harness.

Open *method*, swappable *data*: the metrics here run on the open face-validity
benchmark by default, and on real outcome data (loaded via the ``storage`` seam)
in the private backend. That's how "the fitted profile is more accurate" becomes
a reproducible number.
"""

from __future__ import annotations

from .harness import ValidationResult, evaluate
from .tune import TuneResult, objective, tune

__all__ = ["evaluate", "ValidationResult", "tune", "TuneResult", "objective"]
