"""Tests focused on RhythmicSegments.from_intervals."""

import numpy as np
import pandas as pd
import pytest

from rhythmic_segments.segments import RhythmicSegments


def test_basic_intervals():
    base_meta = dict(label=["w", "x", "y", "z"])

    def aggregator(df: pd.DataFrame) -> dict:
        return dict(label=df.iloc[1]["label"])

    rs = RhythmicSegments.from_intervals(
        [0.5, 1.0, 0.75, 1.25],
        length=2,
        meta=base_meta,
        meta_agg=aggregator,
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
            meta_agg=aggregator,
        )


def test_split_at_nan_required_when_false():
    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            [0.5, np.nan, 1.0], length=2, split_at_nan=False
        )


def test_intervals_with_nan_blocks():
    intervals = [0.5, 1.0, 0.75, np.nan, 1.25, 0.9]
    meta = {"sec": ["a", "b", "c", "nan", "d", "e"]}
    rs_split = RhythmicSegments.from_intervals(
        intervals,
        length=2,
        meta=meta,
        meta_agg=lambda df: {"sec": "-".join(df["sec"])},
    )
    assert rs_split.count == 3
    assert list(rs_split.meta["sec"]) == ["a-b", "b-c", "d-e"]

    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            intervals,
            length=2,
            meta=pd.DataFrame({"sec": ["only"]}),
            meta_agg=lambda df: {"sec": "-".join(df["sec"])},
        )


def test_step_metadata_per_block():
    intervals = [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 6.0]
    rs = RhythmicSegments.from_intervals(intervals, length=2)
    assert "step" in rs.meta
    assert rs.step.tolist() == [0, 1, 0, 1]
    np.testing.assert_array_equal(rs.meta["step"].to_numpy(), rs.step.to_numpy())


def test_rejects_step_override():
    intervals = [0.5, 1.0, 1.5]
    with pytest.raises(ValueError):
        RhythmicSegments.from_intervals(
            intervals,
            length=2,
            meta={"step": [0, 1, 2]},
        )


def test_intervals_length_two():
    intervals = [0.1, 0.2, np.nan, 0.3, 0.4, 0.5]
    meta = dict(label=["x", "y", "nan", "z", "w", "v"])
    rs = RhythmicSegments.from_intervals(
        intervals,
        length=2,
        meta=meta,
        meta_agg=lambda df: {"label": df.iloc[0]["label"]},
    )
    assert rs.count == 3
    assert list(rs.meta["label"]) == ["x", "z", "w"]
