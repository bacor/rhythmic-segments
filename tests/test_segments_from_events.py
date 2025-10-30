"""Tests focused on RhythmicSegments.from_events."""

import numpy as np
import pandas as pd
import pytest

from rhythmic_segments.segments import RhythmicSegments


def test_events_match_intervals():
    events = [0.0, 0.5, 1.25, 2.0]
    rs_events = RhythmicSegments.from_events(events, length=2)
    rs_intervals = RhythmicSegments.from_intervals(
        np.diff(np.asarray(events, dtype=float)), length=2
    )
    assert rs_events.count == rs_intervals.count
    np.testing.assert_array_equal(rs_events.segments, rs_intervals.segments)
    np.testing.assert_array_equal(rs_events.patterns, rs_intervals.patterns)
    np.testing.assert_array_equal(rs_events.durations, rs_intervals.durations)


def test_events_meta_roundtrip():
    events = [0.0, 0.5, 1.0, 1.5]
    meta = {"label": ["start", "mid", "almost", "end"]}

    def event_agg(df: pd.DataFrame) -> dict:
        return {"label": df.iloc[-1]["label"]}

    def segment_agg(df: pd.DataFrame) -> dict:
        return {"label": df.iloc[-1]["label"]}

    rs = RhythmicSegments.from_events(
        events,
        length=2,
        meta=meta,
        interval_meta_agg=event_agg,
        segment_meta_agg=segment_agg,
    )
    assert rs.count == 2
    assert list(rs.meta["label"]) == ["almost", "end"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_events(
            events,
            length=2,
            meta={"label": ["start", "mid"]},
            interval_meta_agg=event_agg,
            segment_meta_agg=segment_agg,
        )


def test_events_meta_with_nan():
    events = [0.0, 0.5, 1.0, np.nan, 2.0, 3.0, 4.0]
    meta = {"label": ["a", "b", "c", np.nan, "d", "e", "f"]}

    def event_agg_concat(df: pd.DataFrame) -> dict:
        if pd.isna(df["label"]).any():
            return {"label": np.nan}
        return {"label": "".join(df["label"])}

    def segment_concat(df: pd.DataFrame) -> dict:
        return {"label": "-".join(df["label"])}

    rs = RhythmicSegments.from_events(
        events,
        length=2,
        meta=meta,
        interval_meta_agg=event_agg_concat,
        segment_meta_agg=segment_concat,
    )
    assert list(rs.meta["label"]) == ["ab-bc", "de-ef"]


def test_events_require_monotonic():
    with pytest.raises(ValueError):
        RhythmicSegments.from_events([0.0, 0.5, 0.4, 1.0], length=2)


def test_events_zero_interval_handling():
    events = [0.0, 0.5, 0.5, 1.0]
    with pytest.raises(ValueError):
        RhythmicSegments.from_events(events, length=2)

    rs = RhythmicSegments.from_events(
        events, length=2, check_zero_intervals=False
    )
    expected = RhythmicSegments.from_intervals(
        np.diff(np.asarray(events, dtype=float)),
        length=2,
        check_zero_intervals=False,
    )
    np.testing.assert_array_equal(rs.segments, expected.segments)
    np.testing.assert_array_equal(rs.patterns, expected.patterns)
    np.testing.assert_array_equal(rs.durations, expected.durations)


def test_events_drop_nan():
    events = [0.0, np.nan, 0.5, 1.0, 1.5]
    with pytest.raises(ValueError):
        RhythmicSegments.from_events(events, length=2, split_at_nan=False)

    rs = RhythmicSegments.from_events(events, length=2, drop_nan=True)
    expected = RhythmicSegments.from_intervals(
        np.diff(np.asarray([0.0, 0.5, 1.0, 1.5], dtype=float)), length=2
    )
    np.testing.assert_array_equal(rs.segments, expected.segments)


def test_events_preserve_nan_boundaries():
    events = [0.0, 0.5, 1.0, np.nan, 1.5, 2.5, 3.0, 4.0]
    rs_events = RhythmicSegments.from_events(events, length=2)
    intervals = np.diff(np.asarray(events, dtype=float))
    assert np.isnan(intervals).any()
    rs_intervals = RhythmicSegments.from_intervals(intervals, length=2)
    assert rs_events.count == rs_intervals.count
    np.testing.assert_array_equal(rs_events.segments, rs_intervals.segments)


def test_events_add_start_time():
    events = [0.0, 0.5, 1.0, np.nan, 2.0, 2.5, 3.0]
    rs = RhythmicSegments.from_events(events, length=2)
    assert "start_time" in rs.meta
    np.testing.assert_allclose(rs.start_time.to_numpy(), np.array([0.0, 2.0]))
    np.testing.assert_array_equal(rs.meta["start_time"].to_numpy(), rs.start_time.to_numpy())


def test_events_reject_start_time_override():
    events = [0.0, 0.5, 1.0, 1.5]

    def segment_meta_agg(df: pd.DataFrame) -> dict:
        return {"start_time": float(df.index[0])}

    with pytest.raises(ValueError):
        RhythmicSegments.from_events(
            events,
            length=2,
            meta={"label": ["a", "b", "c", "d"]},
            segment_meta_agg=segment_meta_agg,
        )


def test_events_require_enough_samples():
    with pytest.raises(ValueError):
        RhythmicSegments.from_events([0.0, 0.5, 1.0], length=3)
