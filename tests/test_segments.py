"""pytest-based coverage for rhythmic segment utilities."""

import numpy as np
import pandas as pd
import pytest

from rhythmic_segments.segments import (
    RhythmicSegments,
    extract_segments,
    split_blocks,
    normalize_segments,
)


def test_split_blocks():
    sections = split_blocks([1, 2, np.nan, 3])
    assert len(sections) == 2
    np.testing.assert_array_equal(sections[0], np.array([1.0, 2.0]))
    np.testing.assert_array_equal(sections[1], np.array([3.0]))

    single = split_blocks([1, 2, 3], separator=None)
    assert len(single) == 1
    np.testing.assert_array_equal(single[0], np.array([1.0, 2.0, 3.0]))


def test_extract_segments_behaviour():
    contiguous = extract_segments([1, 2, 3], 2)
    np.testing.assert_array_equal(contiguous, np.array([[1.0, 2.0], [2.0, 3.0]]))

    result = extract_segments(np.arange(1, 6, dtype=float), 3)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
    np.testing.assert_array_equal(result, expected)

    with pytest.warns(UserWarning):
        warn_result = extract_segments([1, 2], 3, warn_on_short=True)
    assert warn_result.shape == (0, 3)

    dropped = extract_segments([0, 1, 2, 3, 4], 2, drop_zeros=True)
    np.testing.assert_array_equal(
        dropped, np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    )

    with pytest.raises(ValueError):
        extract_segments([1, 0, 2], 2)

    allowed_zero = extract_segments([0, 1, 2], 2, allow_zero=True)
    np.testing.assert_array_equal(allowed_zero, np.array([[0.0, 1.0], [1.0, 2.0]]))

    with pytest.warns(UserWarning):
        empty = extract_segments([1], 2)
    assert empty.shape == (0, 2)


def test_normalize_segments():
    raw = np.array([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]])
    patterns, duration = normalize_segments(raw)
    np.testing.assert_allclose(duration, np.array([3.0, 5.0, 9.0]))
    np.testing.assert_allclose(patterns, raw / duration[:, np.newaxis])


def test_rs_from_intervals():
    base_meta = dict(label=["w", "x", "y", "z"])

    def aggregator(df):
        return dict(label=df.iloc[1]["label"])

    rs = RhythmicSegments.from_intervals(
        [0.5, 1.0, 0.75, 1.25],
        length=2,
        meta=base_meta,
        meta_aggregator=aggregator,
    )
    assert rs.count == 3
    assert rs.segments.shape == (3, 2)
    np.testing.assert_array_equal(
        rs.durations, np.array([1.5, 1.75, 2.0], dtype=np.float32)
    )
    assert list(rs.meta["label"]) == ["x", "y", "z"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            [0.5, 1.0, 0.75, 1.25],
            length=2,
            meta=pd.DataFrame({"m": [1, 2]}),
            meta_aggregator=aggregator,
        )


def test_rs_from_intervals_split_at_nan():
    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            [0.5, np.nan, 1.0], length=2, split_at_nan=False
        )


def test_rs_from_intervals_nan():
    intervals = [0.5, 1.0, 0.75, np.nan, 1.25, 0.9]
    meta = {"sec": ["a", "b", "c", "nan", "d", "e"]}
    rs_split = RhythmicSegments.from_intervals(
        intervals,
        length=2,
        meta=meta,
        meta_aggregator=lambda df: {"sec": "-".join(df["sec"])},
    )
    assert rs_split.count == 3
    assert list(rs_split.meta["sec"]) == ["a-b", "b-c", "d-e"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            intervals,
            length=2,
            meta=pd.DataFrame({"sec": ["only"]}),
            meta_aggregator=lambda df: {"sec": "-".join(df["sec"])},
        )


def test_rs_from_intervals_len1():
    intervals = [0.1, 0.2, np.nan, 0.3, 0.4]
    meta = dict(label=["x", "y", "nan", "z", "w"])
    rs = RhythmicSegments.from_intervals(
        intervals,
        length=1,
        meta=meta,
        meta_aggregator=lambda df: {"label": df.iloc[0]["label"]},
    )
    assert rs.count == 4
    assert list(rs.meta["label"]) == ["x", "y", "z", "w"]


def test_rs_from_events_matches_intervals():
    events = [0.0, 0.5, 1.25, 2.0]
    rs_events = RhythmicSegments.from_events(events, length=2)
    rs_intervals = RhythmicSegments.from_intervals(
        np.diff(np.asarray(events, dtype=float)), length=2
    )
    assert rs_events.count == rs_intervals.count
    np.testing.assert_array_equal(rs_events.segments, rs_intervals.segments)
    np.testing.assert_array_equal(rs_events.patterns, rs_intervals.patterns)
    np.testing.assert_array_equal(rs_events.durations, rs_intervals.durations)


def test_rs_from_events_meta_handling():
    events = [0.0, 0.5, 1.0]
    meta = {"label": ["a", "b"]}

    def agg(df):
        return {"label": df.iloc[-1]["label"]}

    rs = RhythmicSegments.from_events(
        events,
        length=2,
        meta=meta,
        meta_aggregator=agg,
    )
    assert rs.count == 1
    assert list(rs.meta["label"]) == ["b"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_events(
            events,
            length=2,
            meta={"label": ["a", "b", "c"]},
            meta_aggregator=agg,
        )


def test_rs_from_events_requires_increasing():
    with pytest.raises(ValueError):
        RhythmicSegments.from_events([0.0, 0.5, 0.4, 1.0], length=2)


def test_rs_from_events_zero_interval_handling():
    events = [0.0, 0.5, 0.5, 1.0]
    with pytest.raises(ValueError):
        RhythmicSegments.from_events(events, length=2)

    rs = RhythmicSegments.from_events(events, length=2, allow_zero_intervals=True)
    expected = RhythmicSegments.from_intervals(
        np.diff(np.asarray(events, dtype=float)), length=2, allow_zero=True
    )
    np.testing.assert_array_equal(rs.segments, expected.segments)
    np.testing.assert_array_equal(rs.patterns, expected.patterns)
    np.testing.assert_array_equal(rs.durations, expected.durations)


def test_rs_from_events_drop_na():
    events = [0.0, np.nan, 0.5, 1.0]
    with pytest.raises(ValueError):
        RhythmicSegments.from_events(events, length=2, split_at_nan=False)

    rs = RhythmicSegments.from_events(events, length=2, drop_na=True)
    expected = RhythmicSegments.from_intervals(
        np.diff(np.asarray([0.0, 0.5, 1.0], dtype=float)), length=2
    )
    np.testing.assert_array_equal(rs.segments, expected.segments)


def test_rs_from_events_preserves_nan_boundaries():
    events = [0.0, 0.5, 1.0, np.nan, 1.5, 2.5, 3.0, 4.0]
    rs_events = RhythmicSegments.from_events(events, length=2)
    intervals = [0.5, 0.5, np.nan, 1.0, 0.5, 1.0]
    rs_intervals = RhythmicSegments.from_intervals(intervals, length=2)
    np.testing.assert_array_equal(rs_events.segments, rs_intervals.segments)


def test_rs_from_segments():
    matrix = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float)
    meta = pd.DataFrame({"label": ["s1", "s2"]})
    rs = RhythmicSegments.from_segments(matrix, meta=meta)
    assert rs.length == 2
    assert rs.count == 2
    np.testing.assert_array_equal(rs.segments, matrix.astype(np.float32))
    assert list(rs.meta["label"]) == ["s1", "s2"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_segments(matrix, length=3)

    empty = RhythmicSegments.from_segments(np.empty((0, 2)), length=2)
    assert empty.count == 0
    assert empty.length == 2

    with pytest.raises(ValueError):
        RhythmicSegments.from_segments(matrix, meta=pd.DataFrame({"label": ["only"]}))


def test_rs_meta_operations():
    rs = RhythmicSegments.from_intervals([0.5, 1.0, 0.75, 1.25], length=2)
    rs_with_meta = rs.with_meta(section=["a", "b", "c"])
    assert list(rs_with_meta.meta["section"]) == ["a", "b", "c"]

    taken = rs_with_meta.take([1, 2])
    assert taken.count == 2
    assert list(taken.meta["section"]) == ["b", "c"]

    filtered = rs_with_meta.filter([True, False, True])
    assert filtered.count == 2
    assert list(filtered.meta["section"]) == ["a", "c"]

    queried = rs_with_meta.query("section == 'b'")
    assert queried.count == 1
    assert queried.meta.iloc[0]["section"] == "b"

    shuffled = rs_with_meta.shuffle(random_state=0)
    assert set(shuffled.meta["section"]) == {"a", "b", "c"}
    # ensure shuffle can produce a different order
    assert (
        list(shuffled.meta["section"]) != list(rs_with_meta.meta["section"])
        or rs_with_meta.count <= 1
    )

    combined = RhythmicSegments.concat(rs_with_meta, rs_with_meta, source_col="source")
    assert combined.count == rs_with_meta.count * 2
    assert list(combined.meta["section"]) == ["a", "b", "c", "a", "b", "c"]
    assert list(combined.meta["source"]) == [0, 0, 0, 1, 1, 1]


def test_rs_repr_includes_meta_columns():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        meta={"label": ["a", "b", "c"]},
    )
    summary = repr(rs)
    assert "RhythmicSegments(" in summary
    assert "segment_length=2" in summary
    assert "n_segments=3" in summary
    assert "meta_columns=[label]" in summary
    assert "[1." in summary and "[3." in summary


def test_rs_repr_handles_missing_meta_and_ellipsis():
    matrix = np.arange(8, dtype=float).reshape(4, 2)
    rs = RhythmicSegments.from_segments(matrix)
    summary = repr(rs)
    assert "n_meta_cols=0" in summary
    assert ", ..." in summary


def test_rs_filter_by_duration_value_bounds():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]],
    )
    filtered = rs.filter_by_duration(min_value=4.5, max_value=8.5)
    assert filtered.count == 1
    np.testing.assert_allclose(filtered.durations, np.array([5.0], dtype=np.float32))


def test_rs_filter_by_duration_quantiles():
    rs = RhythmicSegments.from_segments(
        [[1.0, 2.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
    )
    filtered = rs.filter_by_duration(min_quantile=0.25, max_quantile=0.75)
    np.testing.assert_allclose(
        filtered.durations, np.array([5.0, 9.0], dtype=np.float32)
    )


def test_rs_filter_by_duration_invalid_quantile():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration(min_quantile=-0.1)


def test_rs_filter_by_duration_conflicting_bounds():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration(min_value=5.0, max_value=1.0)


def test_rs_filter_by_duration_requires_bounds():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        rs.filter_by_duration()


def test_rs_filter_by_duration_value_precedence():
    rs = RhythmicSegments.from_segments([[1.0, 2.0], [2, 2.0], [3.0, 4.0]])
    result = rs.filter_by_duration(min_value=7.0, min_quantile=0.1)
    # min_value should take precedence, leaving only the last duration (7.0)
    np.testing.assert_allclose(result.durations, np.array([7.0], dtype=np.float32))
