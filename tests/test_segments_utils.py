"""Tests for standalone segment utility helpers."""

import numpy as np
import pytest

from rhythmic_segments.segments import extract_segments, normalize_segments


def test_extract_segments_basic():
    contiguous = extract_segments([1, 2, 3], 2)
    np.testing.assert_array_equal(contiguous, np.array([[1.0, 2.0], [2.0, 3.0]]))

    result = extract_segments(np.arange(1, 6, dtype=float), 3)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
    np.testing.assert_array_equal(result, expected)

    with pytest.warns(UserWarning):
        warn_result = extract_segments([1, 2], 3, warn_on_short=True)
    assert warn_result.shape == (0, 3)

    with pytest.raises(ValueError):
        extract_segments([1, 0, 2], 2)

    allowed_zero = extract_segments([0, 1, 2], 2, check_zero_intervals=False)
    np.testing.assert_array_equal(allowed_zero, np.array([[0.0, 1.0], [1.0, 2.0]]))

    with pytest.raises(ValueError):
        extract_segments([1.0, np.nan, 2.0], 2)

    allowed_nan = extract_segments(
        [1.0, np.nan, 2.0], 2, check_nan_intervals=False
    )
    expected_nan = np.array([[1.0, np.nan], [np.nan, 2.0]])
    np.testing.assert_allclose(allowed_nan, expected_nan, equal_nan=True)

    with pytest.warns(UserWarning):
        empty = extract_segments([1], 2)
    assert empty.shape == (0, 2)


def test_normalize_segments_basic():
    raw = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
    patterns, duration = normalize_segments(raw)
    np.testing.assert_allclose(duration, np.array([3.0, 5.0, 9.0]))
    np.testing.assert_allclose(patterns, raw / duration[:, np.newaxis])
