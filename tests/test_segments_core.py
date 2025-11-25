"""High-level RhythmicSegments behaviour tests."""

import numpy as np
import pandas as pd
import pytest

from rhythmic_segments.segments import RhythmicSegments


def test_from_segments_basic():
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float)
    meta = pd.DataFrame({"label": ["s1", "s2"]})
    rs = RhythmicSegments.from_segments(matrix, meta=meta)
    assert rs.length == 2
    assert rs.count == 2
    np.testing.assert_array_equal(rs.segments, matrix.astype(np.float32))
    assert rs.meta["label"].tolist() == ["s1", "s2"]


def test_from_segments_rejects_reserved_columns():
    data = [[1.0, 2.0], [2.0, 3.0]]
    with pytest.raises(ValueError):
        RhythmicSegments.from_segments(data, meta={"step": [0, 1]})
    with pytest.raises(ValueError):
        RhythmicSegments.from_segments(data, meta={"start_time": [0.0, 1.0]})


def test_requires_min_length():
    with pytest.raises(ValueError):
        RhythmicSegments.from_segments([[1.0]], length=1)
    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals([0.5, 0.6], length=1)
    with pytest.raises(ValueError):
        RhythmicSegments.from_events([0.0, 0.5], length=1)


def test_meta_operations():
    rs = RhythmicSegments.from_intervals([0.5, 1.0, 0.75, 1.25], length=2)
    rs_with_meta = rs.with_meta(section=["a", "b", "c"])
    assert rs_with_meta.meta["section"].tolist() == ["a", "b", "c"]

    taken = rs_with_meta.take([1, 2])
    assert taken.count == 2
    assert taken.meta["section"].tolist() == ["b", "c"]

    filtered = rs_with_meta.filter([True, False, True])
    assert filtered.count == 2
    assert filtered.meta["section"].tolist() == ["a", "c"]

    queried = rs_with_meta.query("section == 'b'")
    assert queried.count == 1
    assert queried.meta.iloc[0]["section"] == "b"

    shuffled = rs_with_meta.shuffle(random_state=0)
    assert set(shuffled.meta["section"]) == {"a", "b", "c"}

    combined = RhythmicSegments.concat(rs_with_meta, rs_with_meta, source_col="source")
    assert combined.count == rs_with_meta.count * 2
    assert combined.meta["section"].tolist() == ["a", "b", "c", "a", "b", "c"]
    assert combined.meta["source"].tolist() == [0, 0, 0, 1, 1, 1]


def test_repr_includes_meta():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        meta={"label": ["a", "b", "c"]},
    )
    summary = repr(rs)
    assert "RhythmicSegments(" in summary
    assert "segment_length=2" in summary
    assert "n_segments=3" in summary
    assert "meta_columns=[label]" in summary


def test_repr_handles_empty_meta():
    matrix = np.arange(8, dtype=float).reshape(4, 2)
    rs = RhythmicSegments.from_segments(matrix)
    summary = repr(rs)
    assert "n_meta_cols=0" in summary
    assert ", ..." in summary


def test_filter_duration_value_bounds():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
    )
    filtered = rs.filter_by_duration(min_value=4.5, max_value=8.5)
    assert filtered.count == 1
    np.testing.assert_allclose(filtered.durations, np.array([5.0], dtype=np.float32))


def test_filter_duration_quantiles():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
    )
    filtered = rs.filter_by_duration(min_quantile=0.25, max_quantile=0.75)
    np.testing.assert_allclose(
        filtered.durations, np.array([5.0, 9.0], dtype=np.float32)
    )


def test_filter_duration_invalid_quantile():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration(min_quantile=-0.1)


def test_filter_duration_conflicting_bounds():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration(min_value=5.0, max_value=1.0)


def test_filter_duration_requires_bounds():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration()


def test_filter_duration_value_precedence():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 2.0], [3.0, 4.0]])
    result = rs.filter_by_duration(min_value=7.0, min_quantile=0.1)
    np.testing.assert_allclose(result.durations, np.array([7.0], dtype=np.float32))


def test_patdur_combines_pattern_and_duration():
    rs = RhythmicSegments.from_segments([[1.0, 2.0, 1.0], [2.0, 2.0, 4.0]])
    expected = np.column_stack((rs.patterns[:, :-1], rs.durations))
    np.testing.assert_allclose(rs.patdur, expected)


def test_pat_and_dur_shorthands():
    rs = RhythmicSegments.from_segments([[1.0, 2.0, 1.0], [2.0, 1.0, 3.0]])
    np.testing.assert_allclose(rs.pat, rs.patterns[:, :-1])
    np.testing.assert_allclose(rs.dur, rs.durations)


def test_ratio_only_for_length_two():
    rs = RhythmicSegments.from_segments([[1.0, 1.0], [2.0, 1.0]])
    np.testing.assert_allclose(rs.ratio, rs.patterns[:, 0])
    with pytest.raises(ValueError):
        RhythmicSegments.from_segments([[1.0, 1.0, 2.0]]).ratio
