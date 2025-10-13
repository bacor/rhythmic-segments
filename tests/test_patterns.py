import numpy as np
import pytest

from rhythmic_segments.patterns import (
    anisochrony,
    isochronous_pattern,
    isochrony,
    npvi,
    total_variation_distance,
)


def test_total_variation_distance_returns_matrix():
    pat1 = np.array([[0.5, 0.5], [0.25, 0.75]])
    pat2 = np.array([[1.0, 0.0], [0.0, 1.0]])

    distances = total_variation_distance(pat1, pat2)

    expected = np.array(
        [
            [0.5, 0.5],
            [0.75, 0.25],
        ]
    )
    np.testing.assert_allclose(distances, expected)


def test_total_variation_distance_checks_normalization():
    pat = np.array([[0.5, 0.6]])
    with pytest.raises(ValueError):
        total_variation_distance(pat, pat)

    # Within tolerance should pass
    normalized = np.array([[0.5, 0.5 + 1e-10]])
    total_variation_distance(normalized, normalized)


def test_total_variation_distance_accepts_single_patterns():
    dist_single = total_variation_distance([0.25, 0.75], [0.25, 0.75])
    np.testing.assert_allclose(dist_single, np.array([[0.0]]))

    dist_against_many = total_variation_distance(
        [0.25, 0.75], [[0.25, 0.75], [0.5, 0.5]]
    )
    expected = np.array([[0.0, 0.25]])
    np.testing.assert_allclose(dist_against_many, expected)


def test_total_variation_distance_requires_matching_width():
    with pytest.raises(ValueError):
        total_variation_distance([0.5, 0.5], [[0.3, 0.3, 0.4]])


def test_isochronous_pattern_requires_length_greater_than_one():
    for invalid in (0, 1):
        with pytest.raises(ValueError):
            isochronous_pattern(invalid)


def test_anisochrony_returns_scalar_for_single_pattern():
    result = anisochrony([0.25, 0.75])
    assert isinstance(result, float)
    assert result == pytest.approx(0.5)


def test_anisochrony_returns_vector_for_multiple_patterns():
    result = anisochrony([[0.25, 0.75], [0.5, 0.5]])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([0.5, 0.0]))


def test_anisochrony_requires_two_dimensional_inputs():
    with pytest.raises(ValueError):
        anisochrony([[[0.5, 0.5]]])


def test_anisochrony_requires_normalized_patterns():
    with pytest.raises(ValueError):
        anisochrony([[0.6, 0.6]])


def test_isochrony_matches_complement_of_anisochrony():
    patterns = [[0.25, 0.75], [0.5, 0.5]]
    aniso_values = anisochrony(patterns)
    iso_values = isochrony(patterns)
    np.testing.assert_allclose(iso_values, 1 - aniso_values)
    assert isinstance(isochrony([0.5, 0.5]), float)


def test_npvi_computes_mean_over_pairs():
    value = npvi([[0.25, 0.75], [0.5, 0.5]])
    assert value == pytest.approx(50.0)


def test_npvi_rejects_non_normalized_patterns_when_requested():
    patterns = [[0.6, 0.6], [0.4, 0.5]]
    with pytest.raises(ValueError):
        npvi(patterns)

    # Disabling normalization check should succeed
    npvi(patterns, check_normalized=False)


def test_npvi_requires_two_dimensional_input():
    with pytest.raises(ValueError):
        npvi([0.5, 0.5])


def test_npvi_requires_length_two_patterns():
    three_part = [[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]]
    with pytest.raises(ValueError):
        npvi(three_part)
