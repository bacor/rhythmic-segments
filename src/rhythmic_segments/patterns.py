"""Pattern generation utilities for rhythmic segments."""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Tuple, Union, Sequence

import numpy as np

PatternsMatrixLike = Union[np.ndarray, Sequence[Sequence[float]]]
PatternsLike = Union[Sequence[float], PatternsMatrixLike]


def product_patterns(
    factors: Iterable[int], length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return all possible patterns that can be generated using combinations
     of *factors* of given *length*.

    A pattern is a length-``length`` vector whose elements are normalized to sum
    to one. The accompanying ratios matrix stores the raw integer combinations.

    >>> patterns, ratios = product_patterns([1, 2], 2)
    >>> patterns
    array([[0.5       , 0.5       ],
           [0.33333333, 0.66666667],
           [0.66666667, 0.33333333]])
    >>> ratios
    array([[1, 1],
           [1, 2],
           [2, 1]])

    """

    patterns: List[List[float]] = []
    ratios: List[Tuple[int, ...]] = []
    for pattern in product(*[factors] * length):
        total = sum(pattern)
        ratio = [value / total for value in pattern]
        if ratio not in patterns:
            patterns.append(ratio)
            ratios.append(pattern)
    return np.array(patterns, dtype=float), np.array(ratios, dtype=int)


def integer_ratio_patterns(
    integers: Iterable[int], length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for :func:`product_patterns`.

    >>> integer_ratio_patterns([1, 2], 2)[0]
    array([[0.5       , 0.5       ],
           [0.33333333, 0.66666667],
           [0.66666667, 0.33333333]])
    """

    return product_patterns(integers, length)


def isochronous_pattern(length: int) -> np.ndarray:
    """Return the isochronous pattern of given length.

    >>> isochronous_pattern(4)
    array([[0.25, 0.25, 0.25, 0.25]])
    """

    if length <= 1:
        raise ValueError("Length must be greater than one.")
    return np.ones((1, length), dtype=float) / length


def total_variation_distance(
    pat1: PatternsLike,
    pat2: PatternsLike,
    check_normalized: bool = True,
    assume_normalized_tol: float = 1e-9,
) -> np.ndarray:
    """Return the pairwise total variation distance between patterns.

    The function accepts one-dimensional patterns (single observations) as well
    as two-dimensional arrays where each row represents a pattern. In either
    case, the return value is a distance matrix whose ``(i, j)`` entry contains
    the distance between ``pat1[i]`` and ``pat2[j]``.

    >>> total_variation_distance([0.5, 0.5], [0.5, 0.5])
    array([[0.]])
    >>> total_variation_distance([0.5, 0.5], [1.0, 0.0])
    array([[0.5]])
    >>> total_variation_distance([[0.25, 0.5, 0.25]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    array([[0.75, 0.5 ]])
    """

    pat1_arr = np.asarray(pat1, dtype=float)
    pat2_arr = np.asarray(pat2, dtype=float)

    if pat1_arr.ndim == 1:
        pat1_arr = pat1_arr[np.newaxis, :]
    if pat2_arr.ndim == 1:
        pat2_arr = pat2_arr[np.newaxis, :]

    if pat1_arr.shape[1] != pat2_arr.shape[1]:
        raise ValueError("Patterns must have the same length along axis 1.")

    if check_normalized:
        pat1_sums = pat1_arr.sum(axis=1)
        pat2_sums = pat2_arr.sum(axis=1)
        if not (
            np.allclose(pat1_sums, 1, atol=assume_normalized_tol)
            and np.allclose(pat2_sums, 1, atol=assume_normalized_tol)
        ):
            raise ValueError("Patterns must sum to one within tolerance.")

    distances = 0.5 * np.abs(
        pat1_arr[:, np.newaxis, :] - pat2_arr[np.newaxis, :, :]
    ).sum(axis=2)
    return distances


def anisochrony(
    patterns: PatternsLike,
    check_normalized=True,
) -> Union[float, np.ndarray]:
    """Return the anisochrony of a given pattern.

    The anisochrony is the normalized total variation distance between the
    given pattern and the isochronous pattern of the same length. The
    normalization constant ensures the anisochrony is in the range [0, 1].

    >>> anisochrony([0.5, 0.5])
    0.0
    >>> anisochrony([[0.25, 0.75], [0.2, 0.8]])
    array([0.5, 0.6])
    >>> anisochrony([0.25, 0.5, 0.25])
    0.25
    """

    patterns = np.asarray(patterns, dtype=float)
    if patterns.ndim == 1:
        patterns = patterns[np.newaxis, :]
    if patterns.ndim != 2:
        raise ValueError("Pattern must be one- or two-dimensional.")
    if check_normalized and not np.allclose(patterns.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Pattern must sum to one within tolerance.")

    n = patterns.shape[1]
    Cn = n / (n - 1)
    iso_pat = isochronous_pattern(n)
    anisochrony = Cn * total_variation_distance(
        patterns, iso_pat, check_normalized=False
    )
    values = anisochrony[:, 0] if anisochrony.ndim == 2 else anisochrony
    if values.size == 1:
        return float(values.item())
    return values


def isochrony(
    patterns: PatternsLike,
    check_normalized=True,
) -> Union[float, np.ndarray]:
    """Return the normalized isochrony score for the given pattern(s).

    Isochrony is defined as :math:`1 - \\text{anisochrony}`. Perfectly regular
    rhythms yield a score of ``1`` and increasingly irregular (un-isochronous)
    patterns approach ``0``. As with :func:`anisochrony`, a single pattern produces a
    scalar while multiple patterns return a NumPy array of scores.

    >>> isochrony([0.5, 0.5])
    1.0
    >>> isochrony([[0.25, 0.75], [0.5, 0.5]])
    array([0.5, 1. ])
    """
    return 1 - anisochrony(patterns, check_normalized=check_normalized)


def npvi(patterns: PatternsMatrixLike, check_normalized: bool = True) -> float:
    """Return the normalized pairwise variability index for length-two patterns.

    Each input pattern should contain exactly two values (e.g., consecutive
    intervals). The input must contain one pattern per row. When
    ``check_normalized`` is ``True`` (default) the patterns must sum to one,
    which is typical for normalized segment durations. The function returns the
    mean nPVI across all provided patterns.

    >>> npvi([[0.25, 0.75], [0.5, 0.5]])
    50.0
    >>> npvi([[1.0, 0.0], [0.0, 1.0]])
    200.0
    >>> npvi([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    0.0
    """

    patterns_arr = np.asarray(patterns, dtype=float)
    if patterns_arr.ndim != 2:
        raise ValueError("nPVI expects a 2D array of patterns.")
    if patterns_arr.shape[1] != 2:
        raise ValueError("nPVI is defined for length-two patterns.")

    if check_normalized and not np.allclose(patterns_arr.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Patterns must sum to one within tolerance.")

    values = np.asarray(anisochrony(patterns_arr, check_normalized=False), dtype=float)
    return float(200 * np.mean(values))
