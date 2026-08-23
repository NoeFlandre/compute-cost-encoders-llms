from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeGuard


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    """Return whether value is a finite, non-boolean number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def logsumexp(values: Sequence[float]) -> float:
    """Return a numerically stable log-sum-exp for non-empty values."""

    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))
