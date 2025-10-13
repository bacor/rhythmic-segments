"""General-purpose utilities for rhythmic segment processing."""

import numpy as np


def is_nan(value: object) -> bool:
    """Return ``True`` when *value* behaves like a numeric NaN.

    >>> is_nan(float("nan"))
    True
    >>> is_nan("nan")
    False
    >>> import numpy as np
    >>> is_nan(np.nan)
    True
    """

    try:
        return bool(np.isnan(value))  # type: ignore[arg-type]
    except TypeError:
        return False
