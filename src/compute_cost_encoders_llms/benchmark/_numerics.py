from __future__ import annotations

import math
from collections.abc import Sequence


def logsumexp(values: Sequence[float]) -> float:
    """Return a numerically stable log-sum-exp for non-empty values."""

    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))
