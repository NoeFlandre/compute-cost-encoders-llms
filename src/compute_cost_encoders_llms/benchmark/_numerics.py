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


def _number_value(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def logsumexp(values: Sequence[float]) -> float:
    """Return a numerically stable log-sum-exp for non-empty values."""

    maximum = max(values)
    exp = math.exp
    return maximum + math.log(sum(exp(value - maximum) for value in values))
